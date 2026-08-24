import asyncio
import io
import json
import os
import re
import time

from google import genai
from google.genai import types as genai_types
from google.oauth2 import service_account
from PIL.Image import Image

from logger import log_info
from providers.vision_model import VisionModel

USE_VERTEX_AI = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "false").lower() == "true"

DEFAULT_MODEL = "gemini-robotics-er-2-streaming-preview"
DEFAULT_PROMPT = (
    "Detect the objects in the image relevant to a robot manipulating the scene. "
    "Output a JSON array where each entry is an object with keys "
    "'box_2d' (a list [ymin, xmin, ymax, xmax] normalized to 0-1000) and 'label' "
    "(a short name for the object). Return only the JSON array."
)
TURN_TIMEOUT = 10.0
MAX_RECONNECT_ATTEMPTS = 3


class GeminiRoboticsLiveProvider(VisionModel):
    @staticmethod
    def is_available() -> bool:
        if USE_VERTEX_AI:
            return bool(os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE"))
        return bool(os.getenv("GOOGLE_API_KEY"))

    def __init__(self, name=None, connection=None, **kwargs):
        super().__init__(name=name, connection=connection, **kwargs)
        self.model = kwargs.get("model") or name
        if self.model:
            # `name` may still carry the raw "provider[param=value,...]" suffix
            # when instantiated directly from the model string (see connection.py).
            self.model = re.sub(r"\[.*\]\s*$", "", self.model).strip()
        if not self.model or self.model in (
            "gemini_robotics_live",
            "gemini-robotics-live",
            "robotics_live",
            "robotics-live",
        ):
            self.model = DEFAULT_MODEL

        self.prompt = kwargs.get("prompt") or DEFAULT_PROMPT
        self.api_key = kwargs.get("api_key") or os.getenv("GOOGLE_API_KEY")

        self.client = None
        self.session = None
        self.session_context = None
        self.receive_task = None
        self.pending_turn = None
        self.is_closing = False
        self.last_detected_Boxes = []

        log_info(
            f"Gemini Robotics provider model: {self.model} vertexai: {USE_VERTEX_AI} "
            f"sampling_rate: {self.sampling_rate} prompt: {self.prompt}"
        )

    @property
    def is_ready(self) -> bool:
        return self.session is not None

    async def load(self):
        if USE_VERTEX_AI:
            scopes = [
                "https://www.googleapis.com/auth/generative-language",
                "https://www.googleapis.com/auth/cloud-platform",
            ]
            credentials = service_account.Credentials.from_service_account_file(
                os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE"), scopes=scopes
            )
            self.client = genai.Client(
                credentials=credentials,
                http_options=genai.types.HttpOptions(api_version="v1beta1"),
            )
        else:
            self.client = genai.Client(
                api_key=self.api_key,
                http_options=genai.types.HttpOptions(api_version="v1beta"),
            )
        await self._open_session()
        self.receive_task = asyncio.ensure_future(self._receive_loop())

    async def _open_session(self):
        config = genai_types.LiveConnectConfig(
            response_modalities=["TEXT"],
            media_resolution=genai_types.MediaResolution.MEDIA_RESOLUTION_MEDIUM,
            context_window_compression=genai_types.ContextWindowCompressionConfig(
                trigger_tokens=104857,
                sliding_window=genai_types.SlidingWindow(target_tokens=52428),
            ),
        )
        self.session_context = self.client.aio.live.connect(model=self.model, config=config)
        self.session = await self.session_context.__aenter__()

    async def _close_session(self):
        if self.session_context is not None:
            try:
                await self.session_context.__aexit__(None, None, None)
            except Exception as e:
                log_info(f"Error closing Gemini Robotics session: {e}")
            self.session_context = None
        self.session = None

    async def _receive_loop(self):
        reconnect_attempts = 0
        while not self.is_closing:
            text_parts = []
            try:
                async for message in self.session.receive():
                    if message.server_content:
                        server_content = message.server_content
                        if server_content.model_turn and server_content.model_turn.parts:
                            for part in server_content.model_turn.parts:
                                if part.text:
                                    text_parts.append(part.text)
                        if server_content.turn_complete:
                            if self.pending_turn and not self.pending_turn.done():
                                self.pending_turn.set_result("".join(text_parts))
                            text_parts = []
                reconnect_attempts = 0
            except asyncio.CancelledError:
                raise
            except Exception as e:
                if self.is_closing:
                    break

                log_info(f"Error receiving Gemini Robotics events: {e}")
                if self.pending_turn and not self.pending_turn.done():
                    self.pending_turn.set_exception(e)

                reconnect_attempts += 1
                if reconnect_attempts > MAX_RECONNECT_ATTEMPTS:
                    log_info("Failed to reconnect Gemini Robotics session, giving up.")
                    await self._close_session()
                    break

                log_info(f"Reconnecting Gemini Robotics session (attempt {reconnect_attempts})...")
                await self._close_session()
                try:
                    await self._open_session()
                except Exception as conn_err:
                    log_info(f"Reconnection attempt {reconnect_attempts} failed: {conn_err}")
                    await asyncio.sleep(2)

    def clear_overlay(self):
        self.last_detected_Boxes = []

    async def _run_turn(self, image_bytes: bytes) -> str:
        if not self.session:
            return ""

        self.pending_turn = asyncio.get_event_loop().create_future()
        try:
            await self.session.send_client_content(
                turns=genai_types.Content(
                    role="user",
                    parts=[
                        genai_types.Part(inline_data=genai_types.Blob(data=image_bytes, mime_type="image/jpeg")),
                        genai_types.Part(text=self.prompt),
                    ],
                ),
                turn_complete=True,
            )
            return await asyncio.wait_for(self.pending_turn, timeout=TURN_TIMEOUT)
        except Exception as e:
            log_info(f"Error running Gemini Robotics turn: {e}")
            return ""
        finally:
            self.pending_turn = None

    async def process_frame(self, image: Image):
        width, height = image.width, image.height
        if not self.session:
            return

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=80)
        image_bytes = buffer.getvalue()

        start = time.monotonic()
        try:
            text = await self._run_turn(image_bytes)
        finally:
            self.stats.record_processing_time(time.monotonic() - start)

        detections = self._parse_detections(text or "", width, height)

        detected_Boxes = []
        labels = set()
        for detection in detections:
            labels.add(detection["label"])
            detected_Boxes.append(
                {
                    "coords": detection["coords"],
                    "label": detection["label"],
                    "color": self.get_color(detection["label"]),
                }
            )

        self.last_detected_Boxes = detected_Boxes

        self.notify_detections(detections, labels)

        objects = []
        for detection in detections:
            ymin, xmin, ymax, xmax = detection["box_2d"]
            objects.append(
                {
                    "label": detection["label"],
                    "top": round(min(1.0, max(0.0, ymin / 1000.0)), 2),
                    "left": round(min(1.0, max(0.0, xmin / 1000.0)), 2),
                    "bottom": round(min(1.0, max(0.0, ymax / 1000.0)), 2),
                    "right": round(min(1.0, max(0.0, xmax / 1000.0)), 2),
                }
            )
        self.notify_objects(objects)

    def _parse_detections(self, text: str, width: int, height: int):
        cleaned = text.strip()
        fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
        if fenced:
            cleaned = fenced.group(1)

        try:
            items = json.loads(cleaned)
        except (json.JSONDecodeError, TypeError):
            return []

        if not isinstance(items, list):
            return []

        detections = []
        for item in items:
            if not isinstance(item, dict):
                continue

            box = item.get("box_2d")
            if not isinstance(box, list) or len(box) != 4:
                continue

            label = str(item.get("label", "object"))
            ymin, xmin, ymax, xmax = box
            x1 = xmin / 1000 * width
            y1 = ymin / 1000 * height
            x2 = xmax / 1000 * width
            y2 = ymax / 1000 * height
            detections.append({"coords": [x1, y1, x2, y2], "label": label, "box_2d": box})

        return detections

    async def close(self):
        log_info("Closing Gemini Robotics provider")
        self.is_closing = True

        if self.pending_turn and not self.pending_turn.done():
            self.pending_turn.cancel()

        if self.receive_task is not None:
            self.receive_task.cancel()
            try:
                await self.receive_task
            except (asyncio.CancelledError, Exception):
                pass
            self.receive_task = None

        await self._close_session()
        self.client = None

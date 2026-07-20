import asyncio
import base64
import io
import os
import re
import time
from collections import deque
from datetime import datetime

from openai import AsyncOpenAI
from PIL import ImageDraw
from PIL.Image import Image

from logger import log_info, log_warn
from providers.vision_model import VisionModel
from utils import parse_bool, parse_int

DEFAULT_MODEL = "nvidia/Cosmos-Reason2-2B"
DEFAULT_ENDPOINT = "http://localhost:8000/v1"
DEFAULT_PROMPT = (
    "You are watching frames sampled from the last few seconds of a live camera stream. "
    "Describe concisely what is happening, focusing on people, objects and actions. "
    "Answer in one or two short sentences, no preamble."
)

# Sliding-window defaults matching NVIDIA's RT-VLM reference design
# (chunked stateless requests against a vLLM OpenAI-compatible endpoint).
DEFAULT_WINDOW_SECONDS = 8
DEFAULT_OVERLAP_SECONDS = 2
DEFAULT_FPS = 4
DEFAULT_RESOLUTION = 448
DEFAULT_MAX_TOKENS = 1024

THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class CosmosProvider(VisionModel):
    """Provider for NVIDIA Cosmos Reason served through an OpenAI-compatible
    endpoint (vLLM). Cosmos has no streaming video input, so the live stream is
    processed as a sliding window of sampled frames: every `window` seconds the
    frames captured at `fps` are sent as one multi-image request, with the last
    `overlap` seconds shared between consecutive windows for temporal
    continuity."""

    # Frames are sampled by wall-clock time (`fps`), not by frame count.
    DEFAULT_SAMPLING_RATE = 1

    def __init__(self, name=None, connection=None, **kwargs):
        super().__init__(name=name, connection=connection, **kwargs)
        self.model = kwargs.get("model") or os.getenv("COSMOS_MODEL") or DEFAULT_MODEL
        self.endpoint = kwargs.get("endpoint") or os.getenv("COSMOS_ENDPOINT") or DEFAULT_ENDPOINT
        self.api_key = (
            kwargs.get("api_key")
            or (connection.api_key if connection else None)
            or os.getenv("COSMOS_API_KEY")
            or "EMPTY"
        )
        self.prompt = kwargs.get("prompt") or os.getenv("COSMOS_PROMPT") or DEFAULT_PROMPT

        self.window_seconds = parse_int(
            kwargs.get("window"), parse_int(os.getenv("COSMOS_WINDOW_SECONDS"), DEFAULT_WINDOW_SECONDS)
        )
        self.overlap_seconds = parse_int(
            kwargs.get("overlap"), parse_int(os.getenv("COSMOS_OVERLAP_SECONDS"), DEFAULT_OVERLAP_SECONDS)
        )
        self.fps = parse_int(kwargs.get("fps"), parse_int(os.getenv("COSMOS_FPS"), DEFAULT_FPS))
        self.resolution = parse_int(
            kwargs.get("resolution"), parse_int(os.getenv("COSMOS_RESOLUTION"), DEFAULT_RESOLUTION)
        )
        self.max_tokens = parse_int(
            kwargs.get("max_tokens"), parse_int(os.getenv("COSMOS_MAX_TOKENS"), DEFAULT_MAX_TOKENS)
        )
        # Cosmos is trained to read timestamps drawn at the bottom of each
        # frame for temporal localization.
        self.add_timestamps = parse_bool(kwargs.get("timestamps"), parse_bool(os.getenv("COSMOS_TIMESTAMPS"), True))

        if self.window_seconds < 1:
            self.window_seconds = DEFAULT_WINDOW_SECONDS
        if self.fps < 1:
            self.fps = DEFAULT_FPS
        if not 0 <= self.overlap_seconds < self.window_seconds:
            log_warn(
                f"Cosmos overlap ({self.overlap_seconds}s) must be shorter than the window "
                f"({self.window_seconds}s); using {DEFAULT_OVERLAP_SECONDS}s"
            )
            self.overlap_seconds = min(DEFAULT_OVERLAP_SECONDS, self.window_seconds - 1)

        self.client = None
        self.last_answer = ""

        # Rolling buffer of (monotonic_ts, data_uri) covering the last `window` seconds.
        self._frames = deque()
        self._last_capture = 0.0
        self._first_capture = None
        self._last_dispatch = None
        self._inference_inflight = False

        log_info(
            f"Cosmos provider model: {self.model} endpoint: {self.endpoint} window: {self.window_seconds}s "
            f"overlap: {self.overlap_seconds}s fps: {self.fps} resolution: {self.resolution} "
            f"timestamps: {self.add_timestamps} draw: {self.draw_detections}"
        )

    @property
    def is_ready(self) -> bool:
        return self.client is not None

    @property
    def stride_seconds(self) -> int:
        """Seconds between consecutive window dispatches."""
        return self.window_seconds - self.overlap_seconds

    async def connect(self):
        self.client = AsyncOpenAI(base_url=self.endpoint, api_key=self.api_key)

    def _now(self) -> float:
        """Monotonic clock, overridable in tests."""
        return time.monotonic()

    def clear_overlay(self):
        self.last_answer = ""
        self._frames.clear()
        self._first_capture = None
        self._last_dispatch = None

    async def process_frame(self, image: Image):
        now = self._now()
        if now - self._last_capture < 1.0 / self.fps:
            return
        self._last_capture = now
        if self._first_capture is None:
            self._first_capture = now

        self._frames.append((now, self._encode_frame(image)))
        while self._frames and self._frames[0][0] < now - self.window_seconds:
            self._frames.popleft()

        if self._inference_inflight:
            return
        window_filled = now - self._first_capture >= self.window_seconds
        stride_elapsed = self._last_dispatch is None or now - self._last_dispatch >= self.stride_seconds
        if window_filled and stride_elapsed:
            self._inference_inflight = True
            self._last_dispatch = now
            asyncio.ensure_future(self._run_window_inference(list(self._frames)))

    def _encode_frame(self, image: Image) -> str:
        """Downscale, optionally stamp a timestamp, and JPEG-encode to a data URI."""
        frame = image.convert("RGB")
        frame.thumbnail((self.resolution, self.resolution))

        if self.add_timestamps:
            draw = ImageDraw.Draw(frame)
            text = datetime.now().strftime("%H:%M:%S")
            try:
                bbox = draw.textbbox((0, 0), text)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            except AttributeError:
                tw, th = draw.textsize(text)
            x, y = 4, frame.height - th - 6
            draw.rectangle([x - 2, y - 2, x + tw + 4, y + th + 4], fill=(0, 0, 0))
            draw.text((x, y), text, fill=(255, 255, 255))

        buffer = io.BytesIO()
        frame.save(buffer, format="JPEG", quality=80)
        data = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{data}"

    async def _run_window_inference(self, frames):
        try:
            # Media is listed before text to match training inputs.
            content = [{"type": "image_url", "image_url": {"url": uri}} for _, uri in frames]
            content.append({"type": "text", "text": self.prompt})

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                max_tokens=self.max_tokens,
                temperature=0.6,
            )

            message = response.choices[0].message
            answer = THINK_TAG_RE.sub("", message.content or "").strip()
            reasoning = getattr(message, "reasoning_content", None)

            self.last_answer = answer

            raw = {
                "text": answer,
                "frames": len(frames),
                "window_start": frames[0][0],
                "window_end": frames[-1][0],
            }
            if reasoning:
                raw["reasoning"] = reasoning
            self.notify_detections(raw, answer)
        except Exception as e:
            log_warn(f"Cosmos error processing window of {len(frames)} frames: {e}")
        finally:
            self._inference_inflight = False

    def draw_overlay(self, image: Image) -> Image:
        if not self.last_answer:
            return image

        image = image.copy()
        draw = ImageDraw.Draw(image)
        text = self.last_answer if len(self.last_answer) <= 120 else self.last_answer[:117] + "..."
        try:
            bbox = draw.textbbox((0, 0), text)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except AttributeError:
            tw, th = draw.textsize(text)
        x, y = 8, image.height - th - 12
        draw.rectangle([x - 4, y - 4, x + tw + 4, y + th + 4], fill=(0, 0, 0))
        draw.text((x, y), text, fill=(255, 255, 255))
        return image

    async def close(self):
        log_info("Closing Cosmos provider")
        self.client = None
        self.clear_overlay()

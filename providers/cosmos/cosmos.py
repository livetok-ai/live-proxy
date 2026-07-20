import asyncio
import base64
import io
import os
import re
import time
from collections import deque
from datetime import datetime

import torch
from PIL import ImageDraw
from PIL.Image import Image
from transformers import AutoProcessor, Cosmos3OmniForConditionalGeneration

from logger import log_info, log_warn
from providers.vision_model import VisionModel
from utils import parse_bool, parse_int

DEFAULT_MODEL = "nvidia/Cosmos3-Nano"
DEFAULT_PROMPT = (
    "You are watching frames sampled from the last few seconds of a live camera stream. "
    "Describe concisely what is happening, focusing on people, objects and actions. "
    "Answer in one or two short sentences, no preamble."
)

# Sliding-window defaults matching NVIDIA's RT-VLM reference design
# (chunked stateless requests, now run through a local Cosmos3-Nano model).
DEFAULT_WINDOW_SECONDS = 8
DEFAULT_OVERLAP_SECONDS = 2
DEFAULT_FPS = 4
DEFAULT_RESOLUTION = 448
DEFAULT_MAX_TOKENS = 1024

# Reasoning models emit an optional <think>...</think> block before the answer.
THINK_TAG_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


class CosmosProvider(VisionModel):
    """Provider for NVIDIA Cosmos, run locally through transformers. Cosmos has
    no streaming video input, so the live stream is processed as a sliding
    window of sampled frames: every `window` seconds the frames captured at
    `fps` are sent as one multi-image generation call, with the last
    `overlap` seconds shared between consecutive windows for temporal
    continuity."""

    # Frames are sampled by wall-clock time (`fps`), not by frame count.
    DEFAULT_SAMPLING_RATE = 1

    # Model + processor are loaded once per process and shared across
    # connections/providers, like YoloProvider does for its ultralytics model.
    _shared_model = None
    _shared_processor = None
    _shared_model_id = None
    _shared_lock = None

    @classmethod
    async def setup(cls, model_id: str):
        if cls._shared_lock is None:
            cls._shared_lock = asyncio.Lock()

        async with cls._shared_lock:
            if cls._shared_model is not None and cls._shared_model_id == model_id:
                return

            log_info(f"Loading Cosmos model: {model_id} (this can take a while)...")
            loop = asyncio.get_event_loop()
            start = time.monotonic()

            def _load():
                processor = AutoProcessor.from_pretrained(model_id)
                model = Cosmos3OmniForConditionalGeneration.from_pretrained(
                    model_id,
                    dtype=torch.bfloat16,
                    device_map="auto",
                )
                return processor, model

            cls._shared_processor, cls._shared_model = await loop.run_in_executor(None, _load)
            cls._shared_model_id = model_id
            log_info(f"Loaded Cosmos model: {model_id} in {time.monotonic() - start:.1f}s")

    def __init__(self, name=None, connection=None, **kwargs):
        super().__init__(name=name, connection=connection, **kwargs)
        self.model = kwargs.get("model") or os.getenv("COSMOS_MODEL") or DEFAULT_MODEL
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

        self.processor = None
        self._model_instance = None
        self.last_answer = ""

        # Rolling buffer of (monotonic_ts, data_uri) covering the last `window` seconds.
        self._frames = deque()
        # -inf, not 0.0: a real (or mocked) clock reading can legitimately be
        # 0.0, which would make the very first frame look like it arrived
        # within 1/fps of an already-captured frame and get dropped.
        self._last_capture = float("-inf")
        self._first_capture = None
        self._last_dispatch = None
        self._inference_inflight = False

        log_info(
            f"Cosmos provider model: {self.model} window: {self.window_seconds}s "
            f"overlap: {self.overlap_seconds}s fps: {self.fps} resolution: {self.resolution} "
            f"timestamps: {self.add_timestamps} draw: {self.draw_detections}"
        )

    @property
    def is_ready(self) -> bool:
        return self._model_instance is not None and self.processor is not None

    @property
    def stride_seconds(self) -> int:
        """Seconds between consecutive window dispatches."""
        return self.window_seconds - self.overlap_seconds

    async def connect(self):
        await CosmosProvider.setup(self.model)
        self.processor = CosmosProvider._shared_processor
        self._model_instance = CosmosProvider._shared_model

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

    def _generate(self, messages):
        """Blocking generation call, run in a thread executor to keep the
        asyncio event loop responsive."""
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model_instance.device, torch.bfloat16)

        with torch.no_grad():
            generated_ids = self._model_instance.generate(
                **inputs,
                do_sample=True,
                temperature=0.6,
                max_new_tokens=self.max_tokens,
            )
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output[0] if output else ""

    async def _run_window_inference(self, frames):
        try:
            # Media is listed before text to match training inputs.
            content = [{"type": "image_url", "image_url": {"url": uri}} for _, uri in frames]
            content.append({"type": "text", "text": self.prompt})
            messages = [{"role": "user", "content": content}]

            loop = asyncio.get_event_loop()
            text = await loop.run_in_executor(None, self._generate, messages)

            think_match = THINK_TAG_RE.search(text)
            reasoning = think_match.group(1).strip() if think_match else None
            answer = THINK_TAG_RE.sub("", text).strip()

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
        self.processor = None
        self._model_instance = None
        self.clear_overlay()

import asyncio
import os
import time

from av import AudioFrame, AudioResampler
from google import genai
from google.genai import types as genai_types
from google.oauth2 import service_account
from PIL.Image import Image

from logger import log_debug, log_info
from providers.vision_model import VisionModel

SAMPLE_RATE = 16000
# Cap the rolling audio buffer so a long-running session never accumulates
# more than a few seconds of audio in a single non-streaming request.
MAX_AUDIO_SECONDS = 5

USE_VERTEX_AI = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "false").lower() == "true"

DEFAULT_MODEL = "gemini-flash-latest"
DEFAULT_PROMPT = "Describe the scene in one short sentence."
# Without a timeout, a stalled call to the Gemini API blocks the single
# in-flight processing slot forever, silently dropping every subsequent
# sampled frame with no error ever logged (see VisionModel._processing gate).
REQUEST_TIMEOUT_SECONDS = 15


class GeminiVisionProvider(VisionModel):
    """Non-streaming counterpart to GeminiProvider (see llm_live.py).

    Instead of holding a bidirectional Live API session open, this periodically
    sends the most recent video frame plus any audio captured since the last
    frame to a single, one-shot generate_content call using a non-live Gemini
    3 model, and surfaces the text response as a scene-description overlay
    (see VisionModel/notify_detections)."""

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
            self.model = self.model.split("[")[0].strip()
        if not self.model or self.model in ("gemini_vision", "gemini-vision"):
            self.model = DEFAULT_MODEL

        self.prompt = kwargs.get("prompt") or (connection.system_instructions if connection else None) or DEFAULT_PROMPT
        self.api_key = (
            kwargs.get("api_key") or (connection.api_key if connection else None) or os.getenv("GOOGLE_API_KEY")
        )

        self.client = None
        self.last_description = None

        self.resampler = AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
        self._audio_chunks = []
        self._audio_seconds = 0.0

        log_info(
            f"Gemini Vision provider model: {self.model} vertexai: {USE_VERTEX_AI} "
            f"sampling_rate: {self.sampling_rate} prompt: {self.prompt}"
        )

    @property
    def is_ready(self) -> bool:
        return self.client is not None

    async def connect(self):
        if USE_VERTEX_AI:
            scopes = [
                "https://www.googleapis.com/auth/generative-language",
                "https://www.googleapis.com/auth/cloud-platform",
            ]
            credentials = service_account.Credentials.from_service_account_file(
                os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE"), scopes=scopes
            )
            self.client = genai.Client(credentials=credentials)
        else:
            # Force the Gemini Developer API: GOOGLE_GENAI_USE_VERTEXAI would
            # otherwise hijack routing even when an api_key is passed explicitly.
            self.client = genai.Client(api_key=self.api_key, vertexai=False)

    async def send(self, input):
        if isinstance(input, AudioFrame):
            if not self.input_enabled:
                return
            for frame in self.resampler.resample(input):
                self._audio_chunks.append(frame.to_ndarray().tobytes())
                self._audio_seconds += frame.samples / SAMPLE_RATE
            while self._audio_seconds > MAX_AUDIO_SECONDS and len(self._audio_chunks) > 1:
                dropped = self._audio_chunks.pop(0)
                self._audio_seconds -= len(dropped) / 2 / SAMPLE_RATE
            return

        await super().send(input)

    def _take_audio(self):
        if not self._audio_chunks:
            return None
        audio_bytes = b"".join(self._audio_chunks)
        self._audio_chunks = []
        self._audio_seconds = 0.0
        return audio_bytes

    def clear_overlay(self):
        self.last_description = None

    async def process_frame(self, image: Image):
        contents = [self.prompt, image]
        audio_bytes = self._take_audio()
        if audio_bytes:
            contents.append(genai_types.Part.from_bytes(data=audio_bytes, mime_type=f"audio/pcm;rate={SAMPLE_RATE}"))

        log_debug(
            f"GeminiVisionProvider frame #{self.frame_count} request: model={self.model} "
            f"image_size={image.width}x{image.height} audio_bytes={len(audio_bytes) if audio_bytes else 0}",
            context=self._log_context,
        )

        start = time.monotonic()
        try:
            response = await asyncio.wait_for(
                self.client.aio.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=genai_types.GenerateContentConfig(temperature=0.4),
                ),
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except Exception as e:
            log_debug(
                f"GeminiVisionProvider frame #{self.frame_count} request failed after "
                f"{(time.monotonic() - start) * 1000:.1f}ms: {type(e).__name__}: {e}",
                context=self._log_context,
            )
            raise
        finally:
            self.stats.record_processing_time(time.monotonic() - start)

        description = (response.text or "").strip()
        log_debug(
            f"GeminiVisionProvider frame #{self.frame_count} response in "
            f"{(time.monotonic() - start) * 1000:.1f}ms: {description!r}",
            context=self._log_context,
        )
        if not description or description == self.last_description:
            return

        self.last_description = description
        self.notify_detections(description, description)

    async def close(self):
        log_info("Closing Gemini Vision provider")
        self.client = None

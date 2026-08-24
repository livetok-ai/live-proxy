import asyncio
import os
import time

from av import AudioFrame, AudioResampler
from google import genai
from google.genai import types as genai_types
from PIL.Image import Image

from logger import log_info
from providers.vision_model import VisionModel

SAMPLE_RATE = 16000
# Cap the rolling audio buffer so a long-running session never accumulates
# more than a few seconds of audio in a single non-streaming request.
MAX_AUDIO_SECONDS = 5

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
            f"Gemini Vision provider model: {self.model} sampling_rate: {self.sampling_rate} prompt: {self.prompt}"
        )

    @property
    def is_ready(self) -> bool:
        return self.client is not None

    async def connect(self):
        # Force the Gemini Developer API: non-live vision models used here aren't
        # necessarily published on Vertex AI, and GOOGLE_GENAI_USE_VERTEXAI would
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
        finally:
            self.stats.record_processing_time(time.monotonic() - start)

        description = (response.text or "").strip()
        if not description or description == self.last_description:
            return

        self.last_description = description
        self.notify_detections(description, description)

    async def close(self):
        log_info("Closing Gemini Vision provider")
        self.client = None

import base64
import contextlib
import io
from typing import AsyncGenerator, AsyncIterator

from av import AudioFrame, AudioResampler
from openai import AsyncOpenAI
from PIL.Image import Image

from logger import log_info
from model import Input, Model, Output
from utils import parse_bool, parse_int

SAMPLE_RATE = 24000
AUDIO_PTIME = 0.02


class OpenAI(Model):
    @property
    def is_llm(self) -> bool:
        return True

    @property
    def supports_audio(self) -> bool:
        return True

    @property
    def supports_text(self) -> bool:
        return True

    @property
    def supports_video(self) -> bool:
        return True

    def __init__(self, name=None, connection=None, **kwargs):
        super().__init__(name=name, connection=connection, **kwargs)
        self.session = None
        self.tool_callback = connection.call_tool if connection else None
        self.model = kwargs.get("model") or name
        self.api_key = connection.api_key if connection else None
        self.system_instructions = connection.system_instructions if connection else None
        self.resampler = AudioResampler(
            format="s16",
            layout="mono",
            rate=SAMPLE_RATE,
            frame_size=int(SAMPLE_RATE * AUDIO_PTIME),
        )

        # Support sampling and use_video_buffer parameters only from kwargs
        self.sampling_rate = parse_int(kwargs.get("sampling"), 30)
        self.use_video_buffer = parse_bool(kwargs.get("use_video_buffer"), False)

        self.frame_count = 0
        from utils import VideoBuffer

        self.video_buffer = VideoBuffer(max_size=10)

    async def connect(self):
        self.client = AsyncOpenAI(api_key=self.api_key)

        # Build the session config with optional system instructions
        session_config = {}
        if self.system_instructions:
            session_config["instructions"] = self.system_instructions

        model_name = "gpt-4o-realtime-preview-2025-06-03" if self.model == "openai" else self.model
        self.session_context = self.client.beta.realtime.connect(model=model_name)
        self.session = await self.session_context.__aenter__()

    async def send(self, input: Input):
        if not self.session:
            return
        if isinstance(input, str):
            await self.session.conversation.item.create(
                item={
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": input}],
                }
            )
            await self.session.response.create()
        elif isinstance(input, AudioFrame):
            for frame in self.resampler.resample(input):
                data = frame.to_ndarray().tobytes()
                audio = base64.b64encode(data).decode("utf-8")
                await self.session.input_audio_buffer.append(audio=audio)
        elif isinstance(input, Image):
            self.frame_count += 1
            should_process = (self.frame_count % self.sampling_rate == 1) or (self.sampling_rate <= 1)
            if not should_process:
                return

            if self.use_video_buffer:
                input = self.video_buffer.add_and_composite(input)

            array = io.BytesIO()
            input.save(array, format="JPEG")
            video = base64.b64encode(array.getvalue()).decode("utf-8")

            await self.session.conversation.item.create(
                item={
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{video}"},
                        }
                    ],
                }
            )

    async def recv(self) -> AsyncIterator[Output]:
        if not self.session:
            return
        async for event in self.session:
            if event.type == "response.audio.delta":
                data = base64.b64decode(event.delta)

                frame = AudioFrame(format="s16", layout="mono", samples=len(data) / 2)
                frame.sample_rate = SAMPLE_RATE
                frame.planes[0].update(data)

                yield frame

    async def close(self):
        log_info("Closing OpenAI session")
        if self.session_context:
            await self.session_context.__aexit__(None, None, None)
        self.session = None


@contextlib.asynccontextmanager
async def connect_openai(
    model: str, system_instructions=None, tools=None, tool_callback=None, voice=None, language=None, api_key=None
) -> AsyncGenerator[OpenAI, None]:
    openai = OpenAI(
        name=model,
        system_instructions=system_instructions,
        tools=tools,
        tool_callback=tool_callback,
        voice=voice,
        language=language,
        api_key=api_key,
    )
    await openai.connect()
    try:
        yield openai
    finally:
        await openai.close()

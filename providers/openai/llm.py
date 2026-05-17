import base64
import contextlib
import io
from typing import AsyncGenerator, AsyncIterator

from av import AudioFrame, AudioResampler
from openai import AsyncOpenAI
from PIL.Image import Image

from logger import log_info
from model import Input, Model, Output

SAMPLE_RATE = 24000
AUDIO_PTIME = 0.02


class OpenAI(Model):
    def __init__(self):
        super().__init__()
        self.session = None
        self.tool_callback = None
        self.resampler = AudioResampler(
            format="s16",
            layout="mono",
            rate=SAMPLE_RATE,
            frame_size=int(SAMPLE_RATE * AUDIO_PTIME),
        )

    async def connect(
        self,
        model: str,
        system_instructions=None,
        tools=None,
        tool_callback=None,
        voice=None,
        language=None,
        api_key=None,
    ):
        self.client = AsyncOpenAI(api_key=api_key)
        self.tool_callback = tool_callback

        # Build the session config with optional system instructions
        session_config = {}
        if system_instructions:
            session_config["instructions"] = system_instructions

        self.session_context = self.client.beta.realtime.connect(
            model="gpt-4o-realtime-preview-2025-06-03" if model == "openai" else model
        )
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
    openai = OpenAI()
    await openai.connect(model, system_instructions, tools, tool_callback, voice, language, api_key)
    try:
        yield openai
    finally:
        await openai.close()

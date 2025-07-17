import base64
import contextlib
import io
from typing import AsyncGenerator, AsyncIterator

from av import AudioFrame, AudioResampler
from openai import AsyncOpenAI
from PIL.Image import Image

from model import Input, Model, Output

SAMPLE_RATE = 24000
AUDIO_PTIME = 0.02


class OpenAI(Model):
    def __init__(self, session):
        super().__init__()
        self.session = session
        self.resampler = AudioResampler(
            format="s16",
            layout="mono",
            rate=SAMPLE_RATE,
            frame_size=int(SAMPLE_RATE * AUDIO_PTIME),
        )

    async def send(self, input: Input):
        if isinstance(input, str):
            await self.session.conversation.item.create(
                item={
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Say hello!"}],
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
        async for event in self.session:
            if event.type == "response.audio.delta":
                data = base64.b64decode(event.delta)

                frame = AudioFrame(format="s16", layout="mono", samples=len(data) / 2)
                frame.sample_rate = SAMPLE_RATE
                frame.planes[0].update(data)

                yield frame

    async def close(self):
        await self.session.close()


@contextlib.asynccontextmanager
async def connect_openai(system_instructions=None) -> AsyncGenerator[OpenAI, None]:
    client = AsyncOpenAI()

    # Build the session config with optional system instructions
    session_config = {}
    if system_instructions:
        session_config["instructions"] = system_instructions

    async with client.beta.realtime.connect(model="gpt-4o-realtime-preview-2025-06-03") as conn:
        # TODO: Add system instructions and greeting
        # await conn.session.conversation.item.create(
        #     item={
        #         "type": "message",
        #         "role": "user",
        #         "content": [{"type": "input_text", "text": "Greet the user"}],
        #     }
        # )
        # await conn.session.response.create()
        yield OpenAI(conn)

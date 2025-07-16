import contextlib
import io
from typing import AsyncGenerator, AsyncIterator

from av import AudioFrame, AudioResampler
from google import genai
from PIL.Image import Image

# from logger import log_info
from logger import log_info
from model import Input, Model, Output, ModelEvents

SAMPLE_RATE = 16000
AUDIO_PTIME = 0.02


class Gemini(Model):
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
            await self.session.send(input=input, end_of_turn=True)
        elif isinstance(input, AudioFrame):
            for frame in self.resampler.resample(input):
                blob = genai.types.BlobDict(
                    data=frame.to_ndarray().tobytes(),
                    mime_type=f"audio/pcm;rate={SAMPLE_RATE}",
                )
                await self.session.send(input=blob)
        elif isinstance(input, Image):
            array = io.BytesIO()
            input.save(array, format="JPEG")

            blob = genai.types.BlobDict(
                data=array.getvalue(),
                mime_type="image/jpeg",
            )
            await self.session.send(input=blob)

    async def recv(self) -> AsyncIterator[Output]:
        received = self.session.receive()
        async for event in received:
            if event.data:
                mime_type = event.server_content.model_turn.parts[
                    0
                ].inline_data.mime_type
                sample_rate = int(mime_type.split("rate=")[1])

                frame = AudioFrame(
                    format="s16", layout="mono", samples=len(event.data) / 2
                )
                frame.sample_rate = sample_rate
                frame.planes[0].update(event.data)

                yield frame
            else:
                if event.server_content.interrupted:
                    # log_info(f"Interrupted: {event.server_content.interrupted}")
                    self._emit(
                        ModelEvents.INTERRUPTED, event.server_content.interrupted
                    )

                if event.server_content:
                    if event.server_content.input_transcription:
                        # log_info(f"Input audio transcription: {event.server_content.input_transcription}")
                        self._emit(
                            ModelEvents.INPUT_TRANSCRIPTION,
                            event.server_content.input_transcription.text,
                        )
                    if event.server_content.output_transcription:
                        # log_info(f"Output audio transcription: {event.server_content.output_transcription}")
                        self._emit(
                            ModelEvents.OUTPUT_TRANSCRIPTION,
                            event.server_content.output_transcription.text,
                        )

    async def close(self):
        if self.session is None:
            return
        await self.session.close()
        self.session = None


# gemini-2.5-flash-preview-native-audio-dialog
# gemini-live-2.5-flash-preview
@contextlib.asynccontextmanager
async def connect_gemini(system_instructions=None) -> AsyncGenerator[Gemini, None]:
    client = genai.Client()

    config = genai.types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        # enable_affective_dialog=True,
        realtime_input_config=genai.types.RealtimeInputConfig(
            activity_handling=genai.types.ActivityHandling.START_OF_ACTIVITY_INTERRUPTS,
        ),
        system_instruction=system_instructions,
        context_window_compression=(
            # Configures compression with default parameters.
            genai.types.ContextWindowCompressionConfig(
                sliding_window=genai.types.SlidingWindow(),
            )
        ),
        input_audio_transcription=genai.types.AudioTranscriptionConfig(),
        output_audio_transcription=genai.types.AudioTranscriptionConfig(),
    )

    async with client.aio.live.connect(
        model="gemini-2.5-flash-preview-native-audio-dialog",
        config=config,
    ) as session:
        await session.send(input="Greet the user", end_of_turn=True)
        yield Gemini(session)

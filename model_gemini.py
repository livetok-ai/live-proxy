import contextlib
import io
from typing import AsyncGenerator, AsyncIterator

from av import AudioFrame, AudioResampler
from google import genai
from PIL.Image import Image

# from logger import log_info
from logger import log_info
from model import Input, Model, ModelEvents, Output

SAMPLE_RATE = 16000
AUDIO_PTIME = 0.02


class Gemini(Model):
    def __init__(self, session, tools, tool_callback=None):
        super().__init__()
        self.session = session
        self.tools = tools
        self.tool_callback = tool_callback
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

    async def _handle_tool_call(self, event):
        function_responses = []
        for fc in event.tool_call.function_calls:
            if self.tool_callback:
                result = await self.tool_callback(fc.name, fc.id, fc.args)
            else:
                result = {"error": "No tool callback available"}

            function_response = genai.types.FunctionResponse(
                id=fc.id,
                name=fc.name,
                response={
                    "result": result,
                },
            )
            function_responses.append(function_response)

        await self.session.send_tool_response(function_responses=function_responses)

    async def recv(self) -> AsyncIterator[Output]:
        received = self.session.receive()
        async for event in received:
            if event.server_content:
                if event.data:
                    mime_type = event.server_content.model_turn.parts[0].inline_data.mime_type
                    sample_rate = int(mime_type.split("rate=")[1])

                    frame = AudioFrame(format="s16", layout="mono", samples=len(event.data) / 2)
                    frame.sample_rate = sample_rate
                    frame.planes[0].update(event.data)

                    yield frame
                if event.server_content.interrupted:
                    # log_info(f"Interrupted: {event.server_content.interrupted}")
                    self._emit(ModelEvents.INTERRUPTED)

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

            if event.tool_call:
                try:
                    await self._handle_tool_call(event)
                except Exception as e:
                    log_info(f"Error handling tool call: {e}")
                    await self.session.send_tool_response(
                        function_responses=[
                            genai.types.FunctionResponse(
                                id=fc.id,
                                name=fc.name,
                                response={"result": {"error": str(e)}},
                            )
                            for fc in event.tool_call.function_calls
                        ]
                    )

    async def close(self):
        log_info("Closing Gemini session")

        if self.session is None:
            return
        await self.session.close()
        self.session = None


@contextlib.asynccontextmanager
async def connect_gemini(
    model: str, system_instructions=None, tools=None, tool_callback=None, voice=None, language=None
) -> AsyncGenerator[Gemini, None]:
    client = genai.Client(
        http_options=genai.types.HttpOptions(api_version="v1alpha"),
    )

    all_tools = (
        [
            {
                "function_declarations": [
                    {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["parameters"],
                    }
                    for tool in tools
                ]
            }
        ]
        if tools
        else None
    )

    # Build speech config if voice or language is specified
    speech_config = None
    if voice or language:
        speech_config_dict = {}
        if language:
            speech_config_dict["language_code"] = language
        if voice:
            speech_config_dict["voice_config"] = {"prebuilt_voice_config": {"voice_name": voice}}
        speech_config = genai.types.SpeechConfig(**speech_config_dict)
    print(voice, language, speech_config)

    config = genai.types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        enable_affective_dialog=True if "native" in model else None,
        # proactivity=genai.types.ProactivityConfig(proactive_audio=True),
        system_instruction=system_instructions,
        tools=all_tools,
        speech_config=speech_config,
        context_window_compression=(
            genai.types.ContextWindowCompressionConfig(
                sliding_window=genai.types.SlidingWindow(),
            )
        ),
        input_audio_transcription=genai.types.AudioTranscriptionConfig(),
        output_audio_transcription=genai.types.AudioTranscriptionConfig(),
    )

    async with client.aio.live.connect(
        model="gemini-2.5-flash-preview-native-audio-dialog" if model == "gemini" else model,
        config=config,
    ) as session:
        await session.send(input="Greet the user", end_of_turn=True)
        yield Gemini(session, tools, tool_callback)

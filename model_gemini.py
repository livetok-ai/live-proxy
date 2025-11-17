import asyncio
import contextlib
import os
import io
from typing import AsyncGenerator, AsyncIterator

from av import AudioFrame, AudioResampler
from google import genai, auth, oauth2
from PIL.Image import Image

# from logger import log_info
from logger import log_info
from model import Input, Model, ModelEvents, Output
from services.cartesia.tts import CartesiaTTS

SAMPLE_RATE = 16000
AUDIO_PTIME = 0.02
USE_VERTEX_AI = os.getenv("USE_VERTEX_AI", "false").lower() == "true"


class Gemini(Model):
    def __init__(self):
        super().__init__()
        self.model = None
        self.system_instructions = None
        self.tools = None
        self.tool_callback = None
        self.voice = None
        self.language = None
        self.api_key = None

        self.tts = None
        self.client = None
        self.session = None
        self.session_context = None
        self.previous_session_handle = None

        self.resampler = AudioResampler(
            format="s16",
            layout="mono",
            rate=SAMPLE_RATE,
            frame_size=int(SAMPLE_RATE * AUDIO_PTIME),
        )

    async def connect(self, model, system_instructions, tools, tool_callback, voice, language, api_key):
        log_info(f"Connecting to Gemini model: {model}")

        self.model = model
        self.system_instructions = system_instructions
        self.tools = tools
        self.tool_callback = tool_callback
        self.voice = voice
        self.language = language
        self.api_key = api_key

        gemini_model = model
        if model.endswith("/cartesia"):
            gemini_model = model.replace("/cartesia", "")
            if not self.tts and os.getenv("CARTESIA_API_KEY"):
                self.tts = CartesiaTTS()

        if USE_VERTEX_AI:
            scopes = [
                "https://www.googleapis.com/auth/generative-language",
                "https://www.googleapis.com/auth/cloud-platform",
            ]
            credentials = oauth2.service_account.Credentials.from_service_account_file(
                os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE"), scopes=scopes
            )
            self.client = genai.Client(
                vertexai=True,
                credentials=credentials,
                http_options=genai.types.HttpOptions(api_version="v1beta1"),
            )
        else:
            self.client = genai.Client(
                api_key=api_key,
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

        config = genai.types.LiveConnectConfig(
            response_modalities=["TEXT"] if self.tts else ["AUDIO"],
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
            session_resumption=(genai.types.SessionResumptionConfig(handle=self.previous_session_handle)),
        )

        self.session_context = self.client.aio.live.connect(
            # Vertex AI model="gemini-live-2.5-flash-preview-native-audio-09-2025",
            model="gemini-live-2.5-flash-preview" if gemini_model == "gemini" else gemini_model,
            config=config,
        )
        self.session = await self.session_context.__aenter__()

        if self.tts and not self.tts.connected:
            await self.tts.connect()

        log_info(f"Connected to Gemini model: {model}")

    async def interrupt(self):
        self._emit(ModelEvents.INTERRUPTED)
        if self.tts:
            await self.tts.reset()

    async def send(self, input: Input):
        try:
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
        except Exception as e:
            log_info(f"Error sending input: {e}")

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

    async def _process_session_events(self, output_queue):
        """Process events from self.session and put outputs in the queue"""

        while self.session:
            try:
                received = self.session.receive()
                async for event in received:
                    if event.server_content:
                        if event.server_content.model_turn:
                            # log_info(f"Received model turn: {event.server_content.model_turn}")
                            text = event.server_content.model_turn.parts[0].text
                            if text:
                                self._emit(
                                    ModelEvents.OUTPUT_TRANSCRIPTION,
                                    text,
                                )

                                # log_info(f"Received model turn: {text")
                                if self.tts:
                                    await self.tts.send(text)

                        if event.data:
                            # log_info(f"Received data: {event.data}")
                            mime_type = event.server_content.model_turn.parts[0].inline_data.mime_type
                            parsed_mime_type = mime_type.split("rate=")
                            sample_rate = int(parsed_mime_type[1]) if len(parsed_mime_type) > 1 else 24000

                            frame = AudioFrame(format="s16", layout="mono", samples=len(event.data) / 2)
                            frame.sample_rate = sample_rate
                            frame.planes[0].update(event.data)

                            await output_queue.put(frame)

                        if event.server_content.interrupted:
                            # log_info(f"Interrupted: {event.server_content.interrupted}")
                            await self.interrupt()

                        if event.server_content.input_transcription and event.server_content.input_transcription.text:
                            # log_info(f"Input audio transcription: {event.server_content.input_transcription}")
                            await self.interrupt()
                            self._emit(
                                ModelEvents.INPUT_TRANSCRIPTION,
                                event.server_content.input_transcription.text,
                            )
                        if event.server_content.output_transcription and event.server_content.output_transcription.text:
                            # log_info(f"Output audio transcription: {event.server_content.output_transcription}")
                            self._emit(
                                ModelEvents.OUTPUT_TRANSCRIPTION,
                                event.server_content.output_transcription.text,
                            )

                    if event.usage_metadata:
                        log_info(
                            f"Usage metadata: "
                            f"prompt_token_count={event.usage_metadata.prompt_token_count} "
                            f"cached_content_token_count={event.usage_metadata.cached_content_token_count} "
                            f"response_token_count={event.usage_metadata.response_token_count} "
                            f"tool_use_prompt_token_count={event.usage_metadata.tool_use_prompt_token_count} "
                            f"thoughts_token_count={event.usage_metadata.thoughts_token_count} "
                            f"total_token_count={event.usage_metadata.total_token_count}"
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

                    if event.go_away:
                        log_info(f"Received go away: {event.go_away}")

                    if event.session_resumption_update:
                        # log_info(f"Received session resumption update: {event.session_resumption_update}")
                        update = event.session_resumption_update
                        if update.resumable and update.new_handle:
                            self.previous_session_handle = update.new_handle

            except Exception as e:
                log_info(f"Error processing session events: {e}.")

                if self.session:
                    log_info(f"Reconnecting with handle {self.previous_session_handle}...")
                    await self.connect(
                        self.model,
                        self.system_instructions,
                        self.tools,
                        self.tool_callback,
                        self.voice,
                        self.language,
                        self.api_key,
                    )
        # Signal that session processing is done
        await output_queue.put(None)

    async def _process_tts_events(self, output_queue):
        """Process audio frames from self.tts and put them in the queue"""
        if not self.tts:
            return

        try:
            async for frame in self.tts.recv():
                await output_queue.put(frame)
        except Exception as e:
            log_info(f"Error processing TTS events: {e}")
        finally:
            # Signal that TTS processing is done
            await output_queue.put(None)

    async def recv(self) -> AsyncIterator[Output]:
        # Create a queue to collect outputs from both tasks
        output_queue = asyncio.Queue()

        # Start both processing tasks in parallel
        session_task = asyncio.create_task(self._process_session_events(output_queue))
        tts_task = asyncio.create_task(self._process_tts_events(output_queue))

        # Track how many tasks have completed
        completed_tasks = 0
        total_tasks = 2 if self.tts else 1

        # Yield outputs as they arrive
        while completed_tasks < total_tasks:
            output = await output_queue.get()

            if output is None:
                # One of the tasks has completed
                completed_tasks += 1
            else:
                # log_info(f"Yielding output: {output}")
                yield output

        log_info("Waiting for both tasks to finish")
        # Wait for both tasks to finish
        await asyncio.gather(session_task, tts_task, return_exceptions=True)
        log_info("All tasks finished")

    async def close(self):
        log_info("Closing Gemini session")

        # Close both session and tts in parallel
        close_tasks = []

        if self.session_context is not None:
            close_tasks.append(self.session_context.__aexit__(None, None, None))

        if self.tts is not None:
            close_tasks.append(self.tts.close())

        # Mark as closing
        self.session = None
        self.session_context = None
        self.tts = None

        # Wait for all close operations to complete
        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)


@contextlib.asynccontextmanager
async def connect_gemini(
    model: str, system_instructions=None, tools=None, tool_callback=None, voice=None, language=None, api_key=None
) -> AsyncGenerator[Gemini, None]:
    gemini = Gemini()
    await gemini.connect(model, system_instructions, tools, tool_callback, voice, language, api_key)
    try:
        yield gemini
    finally:
        await gemini.close()


if __name__ == "__main__":
    import asyncio

    async def main():
        async with connect_gemini(model="gemini") as gemini:
            print("Connected to Gemini")
            await gemini.send("Hello, how are you?")
            async for output in gemini.recv():
                print(output)

    asyncio.run(main())

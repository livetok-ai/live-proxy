import asyncio
import contextlib
import os
import time
from typing import AsyncGenerator, AsyncIterator

from av import AudioFrame, AudioResampler
from google import genai
from google.oauth2 import service_account
from PIL.Image import Image

# from logger import log_info
from logger import log_debug, log_info
from model import Input, Model, ModelEvents, Output
from providers.cartesia import CartesiaProvider
from utils import parse_bool, parse_int

SAMPLE_RATE = 16000
AUDIO_PTIME = 0.02
USE_VERTEX_AI = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "false").lower() == "true"


class GeminiProvider(Model):
    @staticmethod
    def is_available() -> bool:
        if USE_VERTEX_AI:
            return bool(os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE"))
        return bool(os.getenv("GOOGLE_API_KEY"))

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
        self.model = kwargs.get("model") or name
        if self.model and self.model.endswith("/cartesia"):
            self.model = self.model.replace("/cartesia", "")

        self.system_instructions = connection.system_instructions if connection else None
        self.tools = {}
        if connection and connection.tools:
            for tool in connection.tools:
                self.tools[tool["name"]] = (tool, None)
        self.tool_callback = connection.call_tool if connection else None
        self.voice = connection.voice if connection else None
        self.language = connection.language if connection else None
        self.api_key = connection.api_key if connection else None

        self.tts = None
        self.client = None
        self.session = None
        self.session_context = None
        self.previous_session_handle = None
        self.is_closing = False
        self.reconnect_attempts = 0

        # Support sampling and use_video_buffer parameters only from kwargs
        self.sampling_rate = parse_int(kwargs.get("sampling"), 30)
        self.use_video_buffer = parse_bool(kwargs.get("use_video_buffer"), False)

        self.frame_count = 0
        from utils import VideoBuffer

        self.video_buffer = VideoBuffer(max_size=10)

        self.resampler = AudioResampler(
            format="s16",
            layout="mono",
            rate=SAMPLE_RATE,
            frame_size=int(SAMPLE_RATE * AUDIO_PTIME),
        )
        log_info(
            f"Gemini provider model: {self.model} vertexai: {USE_VERTEX_AI} tools: {len(self.tools) if self.tools else 0} sampling: {self.sampling_rate} use_video_buffer: {self.use_video_buffer}"
        )

    async def disconnect_session(self):
        if self.session_context is not None:
            try:
                await self.session_context.__aexit__(None, None, None)
            except Exception as e:
                log_info(f"Error exiting session context: {e}")
            self.session_context = None
        self.session = None

    async def connect(self):
        await self.disconnect_session()
        self.is_closing = False
        system_instructions = self.system_instructions
        tools = [tool_dict for tool_dict, _ in self.tools.values()]
        voice = self.voice
        language = self.language
        api_key = self.api_key
        model = self.model

        gemini_model = self.model
        if self.model and self.model.endswith("/cartesia"):
            gemini_model = self.model.replace("/cartesia", "")
            if not self.tts and os.getenv("CARTESIA_API_KEY"):
                self.tts = CartesiaProvider()

        if USE_VERTEX_AI:
            scopes = [
                "https://www.googleapis.com/auth/generative-language",
                "https://www.googleapis.com/auth/cloud-platform",
            ]
            credentials = service_account.Credentials.from_service_account_file(
                os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE"), scopes=scopes
            )
            self.client = genai.Client(
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

        if USE_VERTEX_AI:
            model = (
                "gemini-live-2.5-flash-native-audio" if "native" in gemini_model or not gemini_model else gemini_model
            )
        else:
            model = "gemini-3.1-flash-live-preview" if gemini_model == "gemini" or not gemini_model else gemini_model

        config = genai.types.LiveConnectConfig(
            response_modalities=["TEXT"] if self.tts else ["AUDIO"],
            enable_affective_dialog=True if "native" in model else None,
            proactivity=genai.types.ProactivityConfig(proactive_audio=True) if "native" in model else None,
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
            thinking_config=genai.types.ThinkingConfig(thinking_level="low") if "native" not in model else None,
        )

        self.session_context = self.client.aio.live.connect(
            model=model,
            config=config,
        )
        self.session = await self.session_context.__aenter__()

        if self.tts and not self.tts.connected:
            await self.tts.connect()

    async def interrupt(self):
        self._emit(ModelEvents.INTERRUPTED)
        if self.tts:
            await self.tts.reset()

    async def send(self, input: Input):
        if not self.session:
            return
        try:
            if isinstance(input, str):
                await self.session.send_realtime_input(text=input)
            elif isinstance(input, AudioFrame):
                for frame in self.resampler.resample(input):
                    await self.session.send_realtime_input(
                        audio=genai.types.Blob(
                            data=frame.to_ndarray().tobytes(),
                            mime_type=f"audio/pcm;rate={SAMPLE_RATE}",
                        )
                    )
            elif isinstance(input, Image):
                self.frame_count += 1
                should_process = (self.frame_count % self.sampling_rate == 1) or (self.sampling_rate <= 1)
                if not should_process:
                    return

                if self.use_video_buffer:
                    input = self.video_buffer.add_and_composite(input)

                log_debug(f"Gemini frame #{self.frame_count} processing started", context=self._log_context)
                start = time.monotonic()

                await self.session.send_realtime_input(video=input)

                elapsed_ms = (time.monotonic() - start) * 1000
                log_debug(
                    f"Gemini frame #{self.frame_count} processing finished in {elapsed_ms:.1f}ms",
                    context=self._log_context,
                )
        except Exception as e:
            if not self.is_closing:
                log_info(f"Error sending input: {e}")
                self.session = None
                if self.session_context:
                    try:
                        await self.session_context.__aexit__(None, None, None)
                    except Exception:
                        pass

    async def send_info(self, info: str):
        if not self.session:
            return
        try:
            log_info(f"Sending info to Gemini: {info}")
            await self.session.send_client_content(
                turns=genai.types.Content(
                    role="user",
                    parts=[genai.types.Part(text=info)],
                ),
                turn_complete=False,
            )
        except Exception as e:
            if not self.is_closing:
                log_info(f"Error sending info: {e}")
                self.session = None
                if self.session_context:
                    try:
                        await self.session_context.__aexit__(None, None, None)
                    except Exception:
                        pass

    def _convert_params_list_to_schema(self, params_list):
        properties = {}
        required = []
        for param in params_list:
            name = param.get("name")
            p_type = param.get("type", "string").upper()
            if p_type == "STRING":
                p_type = "STRING"
            elif p_type == "NUMBER":
                p_type = "NUMBER"
            elif p_type == "BOOLEAN":
                p_type = "BOOLEAN"
            elif p_type == "INTEGER":
                p_type = "INTEGER"
            else:
                p_type = "STRING"

            p_desc = param.get("description", "")

            properties[name] = {
                "type": p_type,
            }
            if p_desc:
                properties[name]["description"] = p_desc

            required.append(name)

        return {"type": "OBJECT", "properties": properties, "required": required}

    def add_tool(self, tool_dict, callback):
        name = tool_dict["name"]

        parameters = tool_dict.get("parameters")
        if isinstance(parameters, list):
            parameters = self._convert_params_list_to_schema(parameters)

        genai_tool = {"name": name, "description": tool_dict.get("description", ""), "parameters": parameters}
        self.tools[name] = (genai_tool, callback)

        if self.session and getattr(self, "connection", None):
            log_info(f"Reconnecting Gemini to apply new tool: {name}")

            async def do_reconnect():
                await self.close()
                await self.connect()

            asyncio.create_task(do_reconnect())

    async def _handle_tool_call(self, event):
        function_responses = []
        for fc in event.tool_call.function_calls:
            if fc.name in self.tools:
                _, callback = self.tools[fc.name]
                try:
                    if callback:
                        if asyncio.iscoroutinefunction(callback):
                            result = await callback(fc.args)
                        else:
                            result = callback(fc.args)
                    elif self.tool_callback:
                        result = await self.tool_callback(fc.name, fc.id, fc.args)
                    else:
                        result = {"error": "No tool callback available"}
                except Exception as e:
                    result = {"error": str(e)}
            elif self.tool_callback:
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
                                    {"text": text, "final": False},
                                )

                                # log_info(f"Received model turn: {text")
                                if self.tts:
                                    await self.tts.send(text)

                        if event.data:
                            mime_type = event.server_content.model_turn.parts[0].inline_data.mime_type
                            parsed_mime_type = mime_type.split("rate=")
                            sample_rate = int(parsed_mime_type[1]) if len(parsed_mime_type) > 1 else 24000

                            frame = AudioFrame(format="s16", layout="mono", samples=len(event.data) / 2)
                            frame.sample_rate = sample_rate
                            frame.planes[0].update(event.data)

                            await output_queue.put(frame)

                        if event.server_content.interrupted:
                            log_info(f"Interrupted: {event.server_content.interrupted}")
                            await self.interrupt()

                        if event.server_content.input_transcription and event.server_content.input_transcription.text:
                            log_info(f"Input audio transcription: {event.server_content.input_transcription}")
                            await self.interrupt()
                            self._emit(
                                ModelEvents.INPUT_TRANSCRIPTION,
                                {"text": event.server_content.input_transcription.text, "final": False},
                            )
                        if event.server_content.output_transcription and event.server_content.output_transcription.text:
                            log_info(f"Output audio transcription: {event.server_content.output_transcription}")
                            self._emit(
                                ModelEvents.OUTPUT_TRANSCRIPTION,
                                {"text": event.server_content.output_transcription.text, "final": False},
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
                        log_info(f"Received tool call: {event.tool_call}")
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
                if self.is_closing:
                    break

                log_info(f"Error processing session events: {e}.")

                # Try to reconnect up to 3 times
                reconnected = False
                while self.reconnect_attempts < 3:
                    self.reconnect_attempts += 1
                    log_info(
                        f"Reconnecting with handle {self.previous_session_handle} (attempt {self.reconnect_attempts})..."
                    )
                    try:
                        await self.connect()
                        reconnected = True
                        break
                    except Exception as conn_err:
                        log_info(f"Reconnection attempt {self.reconnect_attempts} failed: {conn_err}")
                        await asyncio.sleep(2)

                if not reconnected:
                    log_info("Failed to reconnect Gemini session after 3 attempts. Closing connection.")
                    if self.connection:
                        asyncio.create_task(self.connection.close())
                    break
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

        # log_info("Waiting for both tasks to finish")
        # Wait for both tasks to finish
        await asyncio.gather(session_task, tts_task, return_exceptions=True)
        # log_info("All tasks finished")

    async def close(self):
        log_info("Closing Gemini session")
        self.is_closing = True

        # Close both session and tts in parallel
        close_tasks = [self.disconnect_session()]

        if self.tts is not None:
            close_tasks.append(self.tts.close())
            self.tts = None

        # Wait for all close operations to complete
        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)


@contextlib.asynccontextmanager
async def connect_gemini(
    model: str, system_instructions=None, tools=None, tool_callback=None, voice=None, language=None, api_key=None
) -> AsyncGenerator[GeminiProvider, None]:
    class DummyConnection:
        def __init__(self):
            self.system_instructions = system_instructions
            self.tools = tools
            self.call_tool = tool_callback
            self.voice = voice
            self.language = language
            self.api_key = api_key

    gemini = GeminiProvider(name=model, connection=DummyConnection())
    await gemini.connect()
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

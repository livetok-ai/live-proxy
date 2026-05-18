import asyncio
import fractions
import json

# Suppress macOS Objective-C duplicate class warnings from av/cv2 during imports
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass
from typing import Optional

import numpy as np

suppress_objc_warnings = sys.platform == "darwin"
if suppress_objc_warnings:
    try:
        stderr_fd = sys.stderr.fileno()
        dup_stderr = os.dup(stderr_fd)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, stderr_fd)
        os.close(devnull)
    except Exception:
        suppress_objc_warnings = False

try:
    from aiortc import (
        MediaStreamTrack,
        RTCConfiguration,
        RTCIceServer,
        RTCPeerConnection,
        RTCSessionDescription,
    )
    from av import AudioFrame, VideoFrame
    from PIL import Image

    import metrics
    from logger import log_info
    from model import ModelEvents
    from providers.gemini.llm import Gemini
    from providers.openai.llm import OpenAI
    from providers.simli.visual import Simli
    from providers.yolo.yolo import YoloProvider
finally:
    if suppress_objc_warnings:
        try:
            os.dup2(dup_stderr, stderr_fd)
            os.close(dup_stderr)
        except Exception:
            pass

MODEL_MAP = {
    "gemini": Gemini,
    "openai": OpenAI,
    "yolo": YoloProvider,
    "simli": Simli,
}

AUDIO_PTIME = 0.02
AUDIO_BITRATE = 32000
USE_VIDEO_BUFFER = False

# Error codes:
# 1XXX - Connection errors
# 2XXX - Audio errors
# 3XXX - Video errors
ERROR_VIDEO_GENERATION = 3001


class SendingTrack(MediaStreamTrack):
    def __init__(self, kind="audio", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kind = kind
        self.queue = asyncio.Queue()

    async def recv(self):
        return await self.queue.get()


class Connection:
    def __init__(self, closed=None, tool_call=None, public_ip=None):
        self.id = str(uuid.uuid4())
        self.recv_audio_track = None
        self.recv_video_track = None
        self.send_audio_track = None
        self.send_video_track = None
        self.pc = None
        self.models = []
        self.system_instructions = None
        self.timeout_task = None
        self.last_message_time = None
        self.start_time = None
        self.transcript = []
        self.output_audio_queue = asyncio.Queue()
        self.connected = False
        self.closed = closed
        self.tool_call = tool_call
        self.data_channel = None
        self.public_ip = public_ip

    async def start(
        self, sdp, model, system_instructions=None, tools=None, voice=None, language=None, api_key=None, avatar=None
    ):
        """Start the RTC connection with the given parameters"""
        self.info(
            f"Starting with {model} {system_instructions} {tools} {voice} {language} {'***' if api_key else 'None'} {avatar}"
        )
        self.system_instructions = system_instructions
        self.tools = tools
        self.voice = voice
        self.language = language
        self.api_key = api_key
        self.video = "m=video" in sdp
        has_simli = any(name.strip().startswith("simli") for name in model.split(";"))
        self.avatar = has_simli and self.video

        is_webrtc = "fingerprint" in sdp
        from sip.peerconnection import SimplePeerConnection

        self.pc = (
            RTCPeerConnection(
                RTCConfiguration(
                    iceServers=[
                        RTCIceServer(
                            urls="stun:stun.l.google.com:19302",
                        )
                    ]
                )
            )
            if is_webrtc
            else SimplePeerConnection(public_ip=self.public_ip)
        )

        self.last_message_time = time.time()
        self.start_time = time.time()
        asyncio.ensure_future(self._run(model))

        offer = RTCSessionDescription(sdp=sdp, type="offer")
        await self.pc.setRemoteDescription(offer)

        has_yolo = any(name.strip().startswith("yolo") for name in model.split(";"))
        if self.avatar or (has_yolo and avatar and self.video):
            self.info(f"Track video added for {'Simli' if self.avatar else 'YOLO'}")
            self.send_video_track = SendingTrack("video")
            self.pc.addTrack(self.send_video_track)
        else:
            # Check if the client has a video track with recv direction (meaning client can receive video from us)
            client_has_video_recv = False
            for transceiver in self.pc.getTransceivers():
                if transceiver.kind == "video":
                    if transceiver.direction in ("sendonly", "sendrecv") or transceiver.currentDirection in (
                        "sendonly",
                        "sendrecv",
                    ):
                        client_has_video_recv = True
                        break

            if has_yolo and client_has_video_recv:
                self.info("Track video added for YOLO overlay")
                self.send_video_track = SendingTrack("video")
                self.pc.addTrack(self.send_video_track)

        answer = await self.pc.createAnswer()
        await self.pc.setLocalDescription(answer)

        sdp_response = self.pc.localDescription.sdp
        found = re.findall(r"a=rtpmap:(\d+) opus/48000/2", sdp_response)
        if found:
            # usedtx=1;  Makes the LLM response much slower
            sdp_response = sdp_response.replace(
                "opus/48000/2\r\n",
                "opus/48000/2\r\n" + f"a=fmtp:{found[0]} useinbandfec=1;maxaveragebitrate={AUDIO_BITRATE}\r\n",
            )
        # Remove lines with a=fingerprint:sha-384 or a=fingerprint:sha-512
        sdp_response = re.sub(r"^a=fingerprint:sha-(384|512) .*\r\n", "", sdp_response, flags=re.MULTILINE)

        return sdp_response

    def info(self, msg, *args):
        log_info(msg, *args, context=self.id)

    async def on_established(self):
        self.info("Connection established")
        assert self.connected

        for m in self.models:
            if not isinstance(m, Simli) and not isinstance(m, YoloProvider):
                await m.send("Greet the user using language " + self.language)

    async def call_tool(self, tool_name, tool_id, parameters):
        """Wrapper for tool calls that uses the provided tool_call func"""
        if self.tool_call:
            return await self.tool_call(self, tool_name, tool_id, parameters, self.tools)
        else:
            return {"error": "Tool calling not configured"}

    async def _run(self, model):
        self.info("Connection started")

        # Start timeout timer
        self.timeout_task = asyncio.create_task(self._timeout_monitor(self.info))

        @self.pc.on("datachannel")
        def on_datachannel(channel):
            self.data_channel = channel

            @channel.on("message")
            async def on_message(message):
                self.last_message_time = time.time()
                for m in self.models:
                    if not isinstance(m, Simli):
                        await m.send(message)

        @self.pc.on("connectionstatechange")
        async def on_connectionstatechange():
            if not self.pc:
                return

            self.info("Connection state is %s", self.pc.connectionState)
            if self.pc.connectionState == "failed" or self.pc.connectionState == "closed":
                await self.close()

            if not self.connected and self.pc and self.pc.connectionState == "connected":
                self.connected = True

                has_genai = any(not isinstance(m, Simli) and not isinstance(m, YoloProvider) for m in self.models)
                if has_genai:
                    await self.on_established()

        @self.pc.on("track")
        def on_track(track):
            self.info("Track %s received", track.kind)

            if track.kind == "audio":
                # Only accept the first track received for now
                if self.recv_audio_track:
                    return

                self.recv_audio_track = track
                self.send_audio_track = SendingTrack("audio")
                self.info("Track audio added")
                self.pc.addTrack(self.send_audio_track)
                asyncio.ensure_future(run_recv_audio_track())

            elif track.kind == "video":
                # Only accept the first track received for now
                if self.recv_video_track:
                    return

                self.recv_video_track = track
                asyncio.ensure_future(run_recv_video_track())

            @track.on("ended")
            async def on_ended():
                self.info("Track %s ended", track.kind)

        async def run_recv_audio_track():
            start_time = time.time()
            while True:
                try:
                    frame = await self.recv_audio_track.recv()
                    self.last_message_time = time.time()
                    # Ignore first 3 seconds of audio because in some platforms (iOS) looks like there is a bug and sends some noise
                    if time.time() - start_time < 3:
                        continue
                    for m in self.models:
                        if not isinstance(m, Simli) and not isinstance(m, YoloProvider):
                            await m.send(frame)

                except Exception as e:
                    self.info("Error receiving frame: %s", e)
                    break

        async def run_recv_video_track():
            buffer = []
            last_frame_time = 0
            while self.pc and self.pc.connectionState != "closed":
                try:
                    frame = await self.recv_video_track.recv()
                    self.last_message_time = time.time()
                    has_active_video_models = any(not isinstance(m, Simli) for m in self.models)
                    if not has_active_video_models:
                        continue

                    # Limit the frame rate processed to 1 fps
                    if time.time() - last_frame_time < 1:
                        continue
                    last_frame_time = time.time()

                    image = frame.to_image()
                    image.pts = frame.pts
                    image.time_base = frame.time_base

                    if USE_VIDEO_BUFFER:
                        buffer.append(image)
                        if len(buffer) > 10:
                            buffer.pop(0)

                        # Compose horizontally all the images in buffer
                        composite = Image.new("RGB", (image.width * len(buffer), image.height))
                        for i in range(len(buffer)):
                            composite.paste(buffer[i], (image.width * i, 0))

                        for m in self.models:
                            if not isinstance(m, Simli):
                                await m.send(composite)
                    else:
                        for m in self.models:
                            if not isinstance(m, Simli):
                                await m.send(image)

                except Exception as e:
                    self.info("Error receiving frame: %s", e)
                    break

        first_video_frame = True

        async def run_recv_genai(session):
            nonlocal first_video_frame
            try:
                video_session = next((m for m in self.models if isinstance(m, Simli)), None)
                async for frame in session.recv():
                    if not self.pc or self.pc.connectionState == "closed":
                        break
                    if isinstance(frame, VideoFrame):
                        if self.send_video_track:
                            if first_video_frame:
                                first_video_frame = False
                                if self.data_channel and self.data_channel.readyState == "open":
                                    try:
                                        message = json.dumps({"type": "video_started"})
                                        self.data_channel.send(message)
                                    except Exception:
                                        pass
                            self.send_video_track.queue.put_nowait(frame)
                    else:
                        if video_session:
                            asyncio.create_task(video_session.send_audio(frame))
                        else:
                            self.output_audio_queue.put_nowait(frame)
            except Exception as e:
                self.info("Error receiving from genai: %s", e)

        async def run_generate_video():
            try:
                first_frame = True
                video_session = next((m for m in self.models if isinstance(m, Simli)), None)

                async def run_receive_synced_audio():
                    async for frame in video_session.recv_audio():
                        if not self.pc or self.pc.connectionState == "closed":
                            break
                        self.send_audio_track.queue.put_nowait(frame)

                asyncio.create_task(run_receive_synced_audio())

                async for frame in video_session.recv_video():
                    if not self.pc or self.pc.connectionState == "closed":
                        break
                    if first_frame:
                        first_frame = False
                        if self.data_channel and self.data_channel.readyState == "open":
                            message = json.dumps({"type": "video_started"})
                            self.data_channel.send(message)
                    self.send_video_track.queue.put_nowait(frame)
            except Exception as e:
                self.info("Error receiving video: %s", e)
                if self.data_channel and self.data_channel.readyState == "open":
                    message = json.dumps({"type": "error", "code": ERROR_VIDEO_GENERATION, "message": str(e)})
                    self.data_channel.send(message)
                video_session = next((m for m in self.models if isinstance(m, Simli)), None)
                if video_session:
                    await video_session.close()
                    if video_session in self.models:
                        self.models.remove(video_session)

        async def run_send_audio_track():
            timestamp = 0
            buffer = b""
            sample_rate = 0
            samples = 0
            next_send_time = time.time()

            while self.pc and self.pc.connectionState != "closed":
                # Check if there's a frame in the output queue
                if (not buffer or len(buffer) < samples * 2) and not self.output_audio_queue.empty():
                    frame = self.output_audio_queue.get_nowait()
                    sample_rate = frame.sample_rate
                    samples = int(sample_rate * AUDIO_PTIME)
                    buffer += frame.to_ndarray().tobytes()

                # Don't send audio until we have at least one genai frame to
                # learn the sample rate
                if sample_rate:
                    audio_frame = AudioFrame(format="s16", layout="mono", samples=samples)
                    audio_frame.sample_rate = sample_rate

                    if len(buffer) >= samples * 2:
                        # We have enough data to send
                        audio_frame.planes[0].update(buffer[: samples * 2])
                        buffer = buffer[samples * 2 :]
                    else:
                        # Not enough data, create silence frame
                        silence_data = np.zeros(samples, dtype=np.int16).tobytes()
                        audio_frame.planes[0].update(silence_data)
                    timestamp += sample_rate * AUDIO_PTIME
                    audio_frame.pts = timestamp
                    audio_frame.time_base = fractions.Fraction(1, sample_rate)

                    video_session = next((m for m in self.models if isinstance(m, Simli)), None)
                    if not video_session:
                        self.send_audio_track.queue.put_nowait(audio_frame)

                # Calculate how much time to sleep to maintain AUDIO_PTIME interval
                sleep_time = next_send_time - time.time()
                if sleep_time > 0:
                    await asyncio.sleep(min(sleep_time, AUDIO_PTIME))
                next_send_time += AUDIO_PTIME

        def add_transcript(role, content):
            if self.transcript and self.transcript[-1]["role"] == role:
                prev = self.transcript[-1]
                prev["content"] += content
            else:
                self.transcript.append(
                    {
                        "timestamp": int(time.time() * 1000),
                        "role": role,
                        "content": content,
                    }
                )
                # Log full transcript when new item is added
                if len(self.transcript) > 2:
                    self.info(f"Transcript updated. {self.transcript[-2]['role']} -> {self.transcript[-2]['content']}")

        def on_input_transcription(input_transcription):
            add_transcript("user", input_transcription)
            if self.data_channel and self.data_channel.readyState == "open":
                message = json.dumps({"type": "transcription", "role": "user", "content": input_transcription})
                self.data_channel.send(message)

        def on_output_transcription(output_transcription):
            add_transcript("model", output_transcription)
            if self.data_channel and self.data_channel.readyState == "open":
                message = json.dumps({"type": "transcription", "role": "model", "content": output_transcription})
                self.data_channel.send(message)

        def on_interrupted(event=None):
            # self.info(f"Received INTERRUPTED event, clearing output queue {self.output_audio_queue.qsize()}")
            while not self.output_audio_queue.empty():
                self.output_audio_queue.get_nowait()
            if self.send_audio_track:
                while not self.send_audio_track.queue.empty():
                    self.send_audio_track.queue.get_nowait()
            if self.send_video_track:
                while not self.send_video_track.queue.empty():
                    self.send_video_track.queue.get_nowait()
            video_session = next((m for m in self.models if isinstance(m, Simli)), None)
            if video_session:
                asyncio.create_task(video_session.clear())

        try:
            # 1. Instantiate and connect models
            model_names = [s.strip() for s in model.split(";")]
            for model_name in model_names:
                model_class = None
                for prefix, cls in MODEL_MAP.items():
                    if model_name.startswith(prefix):
                        model_class = cls
                        break

                if model_class:
                    if model_class == YoloProvider:
                        draw_detections = (
                            "draw" in model_name or "overlay" in model_name or "draw_detections" in model_name
                        )
                        sampling_rate = 5
                        for part in model.split(";"):
                            part_clean = part.strip()
                            if part_clean.startswith("sampling="):
                                try:
                                    sampling_rate = int(part_clean.split("=")[1])
                                except ValueError:
                                    pass
                        m = YoloProvider(draw_detections=draw_detections, sampling_rate=sampling_rate)
                        await m.connect(
                            model_name,
                            self.system_instructions,
                            self.tools,
                            self.call_tool,
                            self.voice,
                            self.language,
                            self.api_key,
                            connection=self,
                        )
                    else:
                        m = model_class()
                        await m.connect(
                            model_name,
                            self.system_instructions,
                            self.tools,
                            self.call_tool,
                            self.voice,
                            self.language,
                            self.api_key,
                        )
                    self.models.append(m)

            # 2. Register event listeners for all models
            for m in self.models:
                m.on(ModelEvents.INPUT_TRANSCRIPTION, on_input_transcription)
                m.on(ModelEvents.OUTPUT_TRANSCRIPTION, on_output_transcription)
                m.on(ModelEvents.INTERRUPTED, on_interrupted)

            self.info("Connected to models: %s", [type(m).__name__ for m in self.models])

            # 3. Start model-specific tasks
            for m in self.models:
                if not isinstance(m, Simli):
                    asyncio.ensure_future(run_recv_genai(m))

            video_session = next((m for m in self.models if isinstance(m, Simli)), None)
            if video_session:
                asyncio.ensure_future(run_generate_video())

            if self.connected:
                await self.on_established()

            await run_send_audio_track()
            self.info("Connection finished")

        except Exception as e:
            self.info("Error sending frame: %s", e)

        try:
            await self.close()
        except Exception as e:
            self.info("Error closing connection: %s", e)

        # Notify parent about connection closure
        if self.closed:
            self.closed(self)

        self.info(f"Connection stopped. Transcript: {len(self.transcript)}.")

    async def _timeout_monitor(self, info):
        """Monitor for timeout - close connection if no messages received for 1 minute"""
        while self.pc and self.pc.connectionState != "closed":
            await asyncio.sleep(5)  # Check every 5 seconds
            if self.last_message_time and time.time() - self.last_message_time > 60:
                self.info("Connection timed out - no messages received for 1 minute")
                asyncio.ensure_future(self.close())
                break

    async def close(self):
        self.info("Closing connection")

        if self.timeout_task:
            self.timeout_task.cancel()
            self.timeout_task = None
        if self.pc:
            await self.pc.close()
            self.pc = None
        for m in self.models:
            await m.close()
        self.models = []


@dataclass(frozen=True)
class ConnectionInfo:
    connection: Connection
    callback: Optional[str] = None
    metadata: Optional[dict] = None

    def __hash__(self):
        return self.connection.__hash__()


class ConnectionManager:
    """Manages all active connections. Singleton pattern."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.connections = set()  # Set of ConnectionInfo objects
            cls._instance._closed_callback = None
            cls._instance._tool_call_callback = None
        return cls._instance

    def configure_callbacks(self, closed_callback=None, tool_call_callback=None):
        """Configure the callbacks that will be used for all connections."""
        self._closed_callback = closed_callback
        self._tool_call_callback = tool_call_callback

    def create_connection(self, callback=None, metadata=None, public_ip=None):
        """Create a new connection and add it to the manager."""

        def on_connection_closed(connection):
            # Find and remove the connection info
            conn_info = None
            for info in self.connections:
                if info.connection == connection:
                    conn_info = info
                    break

            if conn_info and self._closed_callback:
                self._closed_callback(conn_info)

        async def tool_call_wrapper(connection, tool_name, tool_id, parameters, tools):
            # Find connection info
            conn_info = None
            for info in self.connections:
                if info.connection == connection:
                    conn_info = info
                    break

            if conn_info and self._tool_call_callback:
                return await self._tool_call_callback(conn_info, tool_name, tool_id, parameters, tools)
            return {"error": "Tool calling not configured"}

        connection = Connection(closed=on_connection_closed, tool_call=tool_call_wrapper, public_ip=public_ip)
        conn_info = ConnectionInfo(connection=connection, callback=callback, metadata=metadata)
        self.connections.add(conn_info)

        # Update metrics
        metrics.increment_connection()
        metrics.set_open_connections(len(self.connections))

        return conn_info

    def find_connection_by_id(self, connection_id):
        """Find connection info by connection ID."""
        if not connection_id:
            return None

        for info in self.connections:
            if info.connection.id == connection_id:
                return info

        return None

    def remove_connection(self, conn_info):
        """Remove a connection from the manager."""
        self.connections.discard(conn_info)
        # Update metrics
        metrics.set_open_connections(len(self.connections))

    async def close_all(self):
        """Close all connections."""
        coros = [conn_info.connection.close() for conn_info in self.connections]
        await asyncio.gather(*coros)
        self.connections.clear()

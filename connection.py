import asyncio
import fractions
import json
import logging

# Suppress macOS Objective-C duplicate class warnings from av/cv2 during imports
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass
from typing import Optional

# logging.getLogger("aiortc.rtcrtpsender").setLevel(logging.DEBUG)
# logging.getLogger("aiortc.rtcrtpreceiver").setLevel(logging.DEBUG)

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

PREFERRED_VIDEO_CODEC = "H264"  # codec name as it appears in SDP a=rtpmap lines
VIDEO_MIN_BITRATE = 400_000  # 400 kbps — floor enforced by encoder clamping
VIDEO_MAX_BITRATE = 4_000_000  # 4 Mbps

try:
    import aiortc.codecs.vpx as _vpx

    _vpx.MIN_BITRATE = VIDEO_MIN_BITRATE
    _vpx.MAX_BITRATE = VIDEO_MAX_BITRATE
    _vpx.DEFAULT_BITRATE = VIDEO_MAX_BITRATE

    import aiortc.codecs.h264 as _h264

    _h264.MIN_BITRATE = VIDEO_MIN_BITRATE
    _h264.MAX_BITRATE = VIDEO_MAX_BITRATE
    _h264.DEFAULT_BITRATE = VIDEO_MAX_BITRATE

    from aiortc import (
        MediaStreamTrack,
        RTCConfiguration,
        RTCIceServer,
        RTCPeerConnection,
        RTCSessionDescription,
    )
    from av import AudioFrame, VideoFrame

    import metrics
    from logger import log_debug, log_info, log_warn
    from model import Model, ModelEvents
    from providers.cartesia.tts import CartesiaTTS
    from providers.face_landmarker.face_landmarker import FaceLandmarkerProvider
    from providers.gemini.llm import Gemini
    from providers.inception.inception import InceptionProvider
    from providers.openai.llm import OpenAI
    from providers.sam3.sam3 import SamProvider
    from providers.simli.visual import Simli
    from providers.text_sentiment.text_sentiment import TextSentimentProvider
    from providers.yolo.yolo import YoloProvider
    from providers.local_llm.local_llm import LocalLLM
    from providers.ocr.ocr import OCRProvider
    from providers.insivision.visual import Insivision
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
    "sam": SamProvider,
    "sam3": SamProvider,
    "sam2": SamProvider,
    "simli": Simli,
    "face_landmarker": FaceLandmarkerProvider,
    "text_sentiment": TextSentimentProvider,
    "inception": InceptionProvider,
    "cartesia": CartesiaTTS,
    "local_llm": LocalLLM,
    "ocr": OCRProvider,
    "insivision": Insivision,
}


def parse_model(model_str: str):
    """
    Parse a model string like 'yolo[sampling=5, draw=1]' into ('yolo', {'sampling': 5, 'draw': 1})
    """
    model_str = model_str.strip()
    match = re.match(r"^([^\[]+)\[(.*)\]$", model_str)
    if match:
        name = match.group(1).strip()
        params_str = match.group(2).strip()
        params = {}
        if params_str:
            for part in params_str.split(","):
                if "=" in part:
                    k, v = part.split("=", 1)
                    k = k.strip().strip("'\"")
                    v = v.strip().strip("'\"")
                    if v.lower() == "true":
                        v = True
                    elif v.lower() == "false":
                        v = False
                    else:
                        try:
                            if "." in v:
                                v = float(v)
                            else:
                                v = int(v)
                        except ValueError:
                            pass
                    params[k] = v
        return name, params
    return model_str, {}


def split_models(model_str: str):
    """
    Split model string by comma or semicolon, but only when they are outside of brackets '[...]'.
    For example: 'sam[version=sam2.1_t.pt,draw=true],yolo[draw=true]' -> ['sam[version=sam2.1_t.pt,draw=true]', 'yolo[draw=true]']
    """
    parts = []
    current = []
    bracket_depth = 0
    for char in model_str:
        if char == "[":
            bracket_depth += 1
            current.append(char)
        elif char == "]":
            bracket_depth -= 1
            current.append(char)
        elif char in (";", ",") and bracket_depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    if current:
        parts.append("".join(current).strip())
    return [p for p in parts if p]


AUDIO_PTIME = 0.02
AUDIO_BITRATE = 32000
USE_VIDEO_BUFFER = False


def _reorder_video_codecs(sdp: str, preferred: str = PREFERRED_VIDEO_CODEC) -> str:
    """Move preferred video codec payload types to the front of every m=video line.

    Operates on both the offer (to steer aiortc's codec selection) and the
    answer (to advertise server preference to the client).
    """
    lines = sdp.split("\r\n")
    for i, line in enumerate(lines):
        if not line.startswith("m=video "):
            continue
        parts = line.split()
        pts = parts[3:]
        preferred_pts = [pt for pt in pts if any(l.startswith(f"a=rtpmap:{pt} {preferred}/") for l in lines)]
        other_pts = [pt for pt in pts if pt not in preferred_pts]
        lines[i] = " ".join(parts[:3] + preferred_pts + other_pts)
    return "\r\n".join(lines)


# Error codes:
# 1XXX - Connection errors
# 2XXX - Audio errors
# 3XXX - Video errors
ERROR_VIDEO_GENERATION = 3001


VIDEO_CLOCK_RATE = 90000
VIDEO_TIME_BASE = fractions.Fraction(1, VIDEO_CLOCK_RATE)


class SendingTrack(MediaStreamTrack):
    def __init__(self, kind="audio", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kind = kind
        self.queue = asyncio.Queue()
        self._video_pts = 0
        self._video_last_time: Optional[float] = None

    async def recv(self):
        frame = await self.queue.get()
        if self.kind == "video" and (frame.pts is None or frame.time_base is None):
            now = time.time()
            if self._video_last_time is not None:
                self._video_pts += int((now - self._video_last_time) * VIDEO_CLOCK_RATE)
            self._video_last_time = now
            frame.pts = self._video_pts
            frame.time_base = VIDEO_TIME_BASE
        return frame


class Connection:
    def __init__(self, closed=None, tool_call=None, public_ip=None):
        self.id = str(uuid.uuid4())
        self.recv_audio_track = None
        self.recv_video_track = None
        self.send_audio_track = None
        self.send_video_track = None
        self.pc = None
        self.models = []
        self.first_video_frame = True
        self.system_instructions = None
        self.script = None
        self.timeout_task = None
        self.last_message_time = None
        self.start_time = None
        self.transcript = []
        self.output_audio_queue = asyncio.Queue()
        self.connected = False
        self.closed = closed
        self.tool_call = tool_call
        self._data_channel = None  # primary channel (first registered)
        self.data_channels = []  # all channels (up to MAX_DATA_CHANNELS)
        self.public_ip = public_ip
        self._connecting = False
        self._closing = False

    async def start(
        self,
        sdp,
        model,
        system_instructions=None,
        tools=None,
        voice=None,
        language=None,
        api_key=None,
        script=None,
    ):
        """Start the RTC connection with the given parameters"""
        self.debug(
            f"Starting with {model} {system_instructions} {tools} {voice} {language} {'***' if api_key else 'None'} {'with dynamic script' if script else 'no script'}"
        )
        self.system_instructions = system_instructions
        self.script = script
        self.tools = tools
        self.voice = voice
        self.language = language
        self.api_key = api_key

        # 1) Create default models
        if isinstance(model, list):
            for item in model:
                name = item.get("name", "")
                params = item.get("parameters") or {}
                self._add_model_internal(name, params)
        else:
            for model_name_str in split_models(model or ""):
                self._add_model_internal(model_name_str)

        # 2) Call setup in scripts (this could create more models)
        try:
            from script_manager import run_setup

            await run_setup(self)
        except Exception as e:
            self.info(f"Error running setup script: {e}")

        # 3) Register events in all models
        for m in self.models:
            m.on(ModelEvents.INPUT_TRANSCRIPTION, self._on_input_transcription)
            m.on(ModelEvents.OUTPUT_TRANSCRIPTION, self._on_output_transcription)
            m.on(ModelEvents.INTERRUPTED, self._on_interrupted)

        self._connecting = True

        # 4) Call connect in all the models
        self.info("Connecting to models: %s", [type(m).__name__ for m in self.models])
        for m in self.models:
            await m.connect()

        self.info("Connected to models: %s", [type(m).__name__ for m in self.models])

        # 5) Start model-specific tasks (starting receiver tasks)
        for m in self.models:
            asyncio.ensure_future(self._run_recv_genai(m))

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
        asyncio.ensure_future(self._run())

        offer = RTCSessionDescription(sdp=_reorder_video_codecs(sdp), type="offer")
        await self.pc.setRemoteDescription(offer)

        self._add_video_track_if_needed()

        answer = await self.pc.createAnswer()
        await self.pc.setLocalDescription(answer)

        sdp_response = _reorder_video_codecs(self.pc.localDescription.sdp)
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

    def debug(self, msg, *args):
        log_debug(msg, *args, context=self.id)

    def info(self, msg, *args):
        log_info(msg, *args, context=self.id)

    def warn(self, msg, *args):
        log_warn(msg, *args, context=self.id)

    def _add_video_track_if_needed(self):
        if not self.send_video_track and self.pc and hasattr(self.pc, "getTransceivers"):
            for transceiver in self.pc.getTransceivers():
                if transceiver.kind == "video":
                    if transceiver.direction in ("recvonly", "sendrecv") or transceiver.currentDirection in (
                        "recvonly",
                        "sendrecv",
                    ):
                        self.info("Track video added because client wants to receive video")
                        self.send_video_track = SendingTrack("video")
                        self.pc.addTrack(self.send_video_track)
                        asyncio.ensure_future(self._apply_video_bitrate())
                        break

    async def _apply_video_bitrate(self, bitrate: int = VIDEO_MAX_BITRATE):
        for _ in range(50):
            await asyncio.sleep(0.1)
            for sender in self.pc.getSenders():
                if sender.track and sender.track.kind == "video":
                    enc = getattr(sender, "_RTCRtpSender__encoder", None)
                    if enc and hasattr(enc, "target_bitrate"):
                        enc.target_bitrate = bitrate
                        self.info(
                            f"Video encoder bitrate set to {bitrate // 1000} kbps (min {VIDEO_MIN_BITRATE // 1000} kbps)"
                        )
                        return

    def get_model(self, name: str):
        model_name, _ = parse_model(name)
        model_class = None
        for prefix, cls in MODEL_MAP.items():
            if model_name.startswith(prefix):
                model_class = cls
                break
        if not model_class:
            return None
        for m in self.models:
            if isinstance(m, model_class):
                return m
        return None

    def _add_transcript(self, role, content):
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

    def _on_input_transcription(self, input_transcription):
        if isinstance(input_transcription, dict):
            text = input_transcription.get("text", "")
        else:
            text = input_transcription

        self._add_transcript("user", text)
        message = json.dumps({"type": "transcription", "role": "user", "content": text})
        self._broadcast_to_channels(message)
        for m in self.models:
            if hasattr(m, "handle_transcription"):
                asyncio.create_task(m.handle_transcription(text))

    def _on_output_transcription(self, output_transcription):
        if isinstance(output_transcription, dict):
            text = output_transcription.get("text", "")
        else:
            text = output_transcription

        self._add_transcript("model", text)
        message = json.dumps({"type": "transcription", "role": "model", "content": text})
        self._broadcast_to_channels(message)
        for m in self.models:
            if isinstance(m, (Gemini, OpenAI)):
                m._emit("response", {"text": text})

    MAX_DATA_CHANNELS = 10

    def _broadcast_to_channels(self, message: str):
        """Broadcast a pre-serialized JSON string to all open data channels."""
        sent = False
        for ch in self.data_channels:
            if ch.readyState == "open":
                try:
                    ch.send(message)
                    sent = True
                except Exception as e:
                    self.warn(f"Failed to send to channel {ch.label}: {e}")
        if not sent:
            self.warn("Could not broadcast: no open data channels")

    def send_data(self, data):
        """Send arbitrary data to all clients over data channels, serialized as JSON"""
        self._broadcast_to_channels(json.dumps(data))

    def _send_raw_json(self, json_str):
        """Send pre-serialized JSON string to all clients over data channels"""
        self._broadcast_to_channels(json_str)

    def _on_interrupted(self, event=None):
        # self.info(f"Received INTERRUPTED event, clearing output queue {self.output_audio_queue.qsize()}")
        while not self.output_audio_queue.empty():
            self.output_audio_queue.get_nowait()
        if self.send_audio_track:
            while not self.send_audio_track.queue.empty():
                self.send_audio_track.queue.get_nowait()
        if self.send_video_track:
            while not self.send_video_track.queue.empty():
                self.send_video_track.queue.get_nowait()
        for m in self.models:
            if hasattr(m, "clear"):
                asyncio.create_task(m.clear())

    async def _run_recv_genai(self, session):
        try:
            video_session = next((m for m in self.models if hasattr(m, "send_audio")), None)
            async for frame in session.recv():
                if not self.pc or self.pc.connectionState == "closed":
                    break
                if isinstance(frame, VideoFrame):
                    if self.send_video_track:
                        if self.first_video_frame:
                            self.first_video_frame = False
                            self._broadcast_to_channels(json.dumps({"type": "video_started"}))
                        self.send_video_track.queue.put_nowait(frame)
                elif isinstance(frame, AudioFrame):
                    if session == video_session:
                        if self.send_audio_track:
                            self.send_audio_track.queue.put_nowait(frame)
                    else:
                        if video_session:
                            asyncio.create_task(video_session.send_audio(frame))
                        else:
                            self.output_audio_queue.put_nowait(frame)
        except Exception as e:
            self.info("Error receiving from genai: %s", e)

    def _add_model_internal(self, name: str, kwargs: Optional[dict] = None) -> Optional[Model]:
        if self._connecting:
            raise RuntimeError("Cannot add models after connection has started")

        model_name, params = parse_model(name)
        if kwargs:
            params.update(kwargs)
        model_class = None
        for prefix, cls in MODEL_MAP.items():
            if model_name.startswith(prefix):
                model_class = cls
                break

        if not model_class:
            self.info(f"Model map not found for {name}")
            return None

        # Check if already added
        existing = self.get_model(name)
        if existing:
            return existing

        # Instantiate
        m = model_class(name=name, connection=self, **params)
        self.info(f"Model instantiated: {name}")
        self.models.append(m)

        self._add_video_track_if_needed()

        return m

    def add_model_sync(self, name: str, kwargs: Optional[dict] = None) -> Optional[Model]:
        return self._add_model_internal(name, kwargs)

    async def add_model(self, name: str, kwargs: Optional[dict] = None) -> Optional[Model]:
        return self._add_model_internal(name, kwargs)

    async def on_established(self):
        self.info("Connection established")
        assert self.connected

        for m in self.models:
            if m.is_llm:
                await m.send("Greet the user using language " + self.language)

    async def call_tool(self, tool_name, tool_id, parameters):
        """Wrapper for tool calls that uses the provided tool_call func"""
        if self.tool_call:
            return await self.tool_call(self, tool_name, tool_id, parameters, self.tools)
        else:
            return {"error": "Tool calling not configured"}

    async def _run(self):
        self.info("Connection started")

        # Start timeout timer
        self.timeout_task = asyncio.create_task(self._timeout_monitor(self.info))

        @self.pc.on("datachannel")
        def on_datachannel(channel):
            if len(self.data_channels) >= self.MAX_DATA_CHANNELS:
                self.warn(f"Rejecting datachannel '{channel.label}': max {self.MAX_DATA_CHANNELS} channels reached")
                return
            self.data_channels.append(channel)
            if self.data_channel is None:
                self.data_channel = channel  # keep primary reference for legacy compat

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

                has_llm = any(m.is_llm for m in self.models)
                if has_llm:
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
                self.info("Track video received")
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
                        if not m.video_support:
                            await m.send(frame)

                except Exception as e:
                    self.info("Error receiving frame: %s", e)
                    break

        async def run_recv_video_track():
            while self.pc and self.pc.connectionState != "closed":
                try:
                    frame = await self.recv_video_track.recv()
                    self.last_message_time = time.time()
                    has_active_video_models = any(m.supports_video for m in self.models)
                    if not has_active_video_models:
                        continue

                    image = frame.to_image()
                    image.pts = frame.pts
                    image.time_base = frame.time_base

                    for m in self.models:
                        if m.supports_video:
                            await m.send(image)

                except Exception as e:
                    self.info("Error receiving frame: %s", e)
                    break

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

                    video_session = next((m for m in self.models if hasattr(m, "send_audio")), None)
                    if not video_session and self.send_audio_track:
                        self.send_audio_track.queue.put_nowait(audio_frame)

                # Calculate how much time to sleep to maintain AUDIO_PTIME interval
                sleep_time = next_send_time - time.time()
                if sleep_time > 0:
                    await asyncio.sleep(min(sleep_time, AUDIO_PTIME))
                next_send_time += AUDIO_PTIME

        try:
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
        if self._closing:
            return
        self._closing = True
        self.info("Closing connection")

        if not getattr(self, "_teardown_called", False):
            self._teardown_called = True
            try:
                from script_manager import run_teardown

                await run_teardown(self)
            except Exception as e:
                self.info(f"Error running teardown script: {e}")

        if self.timeout_task:
            self.timeout_task.cancel()
            self.timeout_task = None
        if self.pc:
            await self.pc.close()
            self.pc = None
        models_to_close = self.models
        self.models = []
        for m in models_to_close:
            await m.close()

    @property
    def data_channel(self):
        return self._data_channel

    @data_channel.setter
    def data_channel(self, channel):
        self._data_channel = channel
        if channel is not None and channel not in self.data_channels:
            self.data_channels.append(channel)


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

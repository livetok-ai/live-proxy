import asyncio
import fractions
import json
import re
import time
import uuid
from dataclasses import dataclass
from typing import Optional

import numpy as np
from aiortc import (
    MediaStreamTrack,
    RTCConfiguration,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
)
from av import AudioFrame
from PIL import Image

import metrics
from logger import log_info
from model import ModelEvents
from model_gemini import connect_gemini
from model_openai import connect_openai
from simplepeerconnection import SimplePeerConnection

AUDIO_PTIME = 0.02
AUDIO_BITRATE = 32000
USE_VIDEO_BUFFER = False


class SendingTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.queue = asyncio.Queue()

    async def recv(self):
        return await self.queue.get()


class Connection:
    def __init__(self, closed=None, tool_call=None, public_ip=None):
        self.id = str(uuid.uuid4())
        self.recv_audio_track = None
        self.recv_video_track = None
        self.send_track = None
        self.pc = None
        self.genai_session = None
        self.system_instructions = None
        self.timeout_task = None
        self.last_message_time = None
        self.start_time = None
        self.transcript = []
        self.output_queue = asyncio.Queue()
        self.connected = False
        self.closed = closed
        self.tool_call = tool_call
        self.data_channel = None
        self.public_ip = public_ip

    async def start(self, sdp, model, system_instructions=None, tools=None, voice=None, language=None, api_key=None):
        """Start the RTC connection with the given parameters"""
        self.system_instructions = system_instructions
        self.tools = tools
        self.voice = voice
        self.language = language
        self.api_key = api_key
        is_webrtc = "fingerprint" in sdp
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

        assert self.genai_session
        assert self.connected

        await self.genai_session.send("Greet the user")

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
                if self.genai_session:
                    await self.genai_session.send(message)

        @self.pc.on("connectionstatechange")
        async def on_connectionstatechange():
            if not self.pc:
                return

            self.info("Connection state is %s", self.pc.connectionState)
            if self.pc.connectionState == "failed" or self.pc.connectionState == "closed":
                await self.close()

            if not self.connected and self.pc and self.pc.connectionState == "connected":
                self.connected = True

                if self.genai_session:
                    await self.on_established()

        @self.pc.on("track")
        def on_track(track):
            self.info("Track %s received", track.kind)

            if track.kind == "audio":
                # Only accept the first track received for now
                if self.recv_audio_track:
                    return

                self.recv_audio_track = track
                self.send_track = SendingTrack()
                self.pc.addTrack(self.send_track)
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
                    # Ignore first 5 seconds of audio because in some platforms (iOS) looks like there is a bug and sends some noise
                    if time.time() - start_time < 5:
                        continue
                    if not self.genai_session:
                        continue
                    await self.genai_session.send(frame)

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
                    if not self.genai_session:
                        continue

                    # Limit the frame rate processed to 1 fps
                    if time.time() - last_frame_time < 1:
                        continue
                    last_frame_time = time.time()

                    image = frame.to_image()

                    if USE_VIDEO_BUFFER:
                        buffer.append(image)
                        if len(buffer) > 10:
                            buffer.pop(0)

                        # Compose horizontally all the images in buffer
                        composite = Image.new("RGB", (image.width * len(buffer), image.height))
                        for i in range(len(buffer)):
                            composite.paste(buffer[i], (image.width * i, 0))

                        await self.genai_session.send(composite)
                    else:
                        await self.genai_session.send(image)

                except Exception as e:
                    self.info("Error receiving frame: %s", e)
                    break

        async def run_recv_genai():
            try:
                while self.pc and self.pc.connectionState != "closed":
                    async for frame in self.genai_session.recv():
                        self.output_queue.put_nowait(frame)
            except Exception as e:
                self.info("Error receiving from genai: %s", e)

        async def run_send_track():
            timestamp = 0
            buffer = b""
            sample_rate = 0
            samples = 0
            next_send_time = time.time()

            while self.pc and self.pc.connectionState != "closed":
                # Check if there's a frame in the output queue
                if (not buffer or len(buffer) < samples * 2) and not self.output_queue.empty():
                    frame = self.output_queue.get_nowait()
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
                    self.send_track.queue.put_nowait(audio_frame)

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
            # self.info(f"Received INTERRUPTED event, clearing output queue {self.output_queue.qsize()}")
            while not self.output_queue.empty():
                self.output_queue.get_nowait()
            while not self.send_track.queue.empty():
                self.send_track.queue.get_nowait()

        try:
            connect_genai = connect_openai if model == "openai" else connect_gemini

            async with connect_genai(
                model, self.system_instructions, self.tools, self.call_tool, self.voice, self.language, self.api_key
            ) as session:
                self.info("Connected to GenAI session")
                self.genai_session = session

                self.genai_session.on(ModelEvents.INPUT_TRANSCRIPTION, on_input_transcription)
                self.genai_session.on(ModelEvents.OUTPUT_TRANSCRIPTION, on_output_transcription)
                self.genai_session.on(ModelEvents.INTERRUPTED, on_interrupted)

                if self.connected:
                    await self.on_established()

                # Start the genai receiver task
                asyncio.ensure_future(run_recv_genai())

                await run_send_track()
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
        if self.genai_session:
            await self.genai_session.close()
            self.genai_session = None


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
            cls._instance = super(ConnectionManager, cls).__new__(cls)
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

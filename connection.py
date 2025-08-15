import asyncio
import fractions
import re
import time
import uuid

import numpy as np
from aiortc import (
    MediaStreamTrack,
    RTCConfiguration,
    RTCPeerConnection,
    RTCSessionDescription,
    RTCIceServer,
)
from av import AudioFrame
from PIL import Image

from logger import log_info
from model import ModelEvents
from model_gemini import connect_gemini
from model_openai import connect_openai

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
    def __init__(self, closed=None, tool_call=None):
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

    async def start(self, sdp, model, system_instructions=None, tools=None, voice=None, language=None):
        """Start the RTC connection with the given parameters"""
        self.system_instructions = system_instructions
        self.tools = tools
        self.voice = voice
        self.language = language
        offer = RTCSessionDescription(sdp=sdp, type="offer")
        self.pc = RTCPeerConnection(
            RTCConfiguration(
                iceServers=[
                    RTCIceServer(
                        urls="stun:stun.l.google.com:19302",
                    )
                ]
            )
        )

        self.last_message_time = time.time()
        self.start_time = time.time()
        asyncio.ensure_future(self._run(model))

        await self.pc.setRemoteDescription(offer)

        answer = await self.pc.createAnswer()
        await self.pc.setLocalDescription(answer)

        sdp_response = self.pc.localDescription.sdp
        found = re.findall(r"a=rtpmap:(\d+) opus/48000/2", sdp_response)
        if found:
            sdp_response = sdp_response.replace(
                "opus/48000/2\r\n",
                "opus/48000/2\r\n" + f"a=fmtp:{found[0]} useinbandfec=1;usedtx=1;maxaveragebitrate={AUDIO_BITRATE}\r\n",
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

            if not self.connected and self.pc.connectionState == "connected":
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
            while True:
                try:
                    frame = await self.recv_audio_track.recv()
                    self.last_message_time = time.time()
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

        def on_input_transcription(input_transcription):
            self.info(f"Input transcription: {input_transcription}")
            add_transcript("user", input_transcription)

        def on_output_transcription(output_transcription):
            self.info(f"Output transcription: {output_transcription}")
            add_transcript("model", output_transcription)

        def on_interrupted(event=None):
            self.info(f"Received INTERRUPTED event, clearing output queue {self.output_queue.qsize()}")
            while not self.output_queue.empty():
                self.output_queue.get_nowait()
            while not self.send_track.queue.empty():
                self.send_track.queue.get_nowait()

        try:
            connect_genai = connect_openai if model == "openai" else connect_gemini

            async with connect_genai(
                model, self.system_instructions, self.tools, self.call_tool, self.voice, self.language
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

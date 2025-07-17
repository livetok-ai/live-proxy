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
    recv_audio_track = None
    recv_video_track = None
    send_track = None
    pc = None
    genai_session = None
    system_instructions = None
    timeout_task = None
    last_message_time = None
    start_time = None
    transcript = []
    output_queue = asyncio.Queue()

    def __init__(self, on_closed=None):
        self.on_closed = on_closed

    async def start(self, sdp, model, system_instructions=None):
        """Start the RTC connection with the given parameters"""
        self.system_instructions = system_instructions

        offer = RTCSessionDescription(sdp=sdp, type="offer")
        self.pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))

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

    async def _run(self, model):
        self.pc_id = str(uuid.uuid4())

        # Use the shared log_info from logger.py
        def info(msg, *args):
            log_info(msg, *args, context=self.pc_id)

        info("Connection started")

        # Start timeout timer
        self.timeout_task = asyncio.create_task(self._timeout_monitor(info))

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

            info("Connection state is %s", self.pc.connectionState)
            if self.pc.connectionState == "failed" or self.pc.connectionState == "closed":
                await self.close()

        @self.pc.on("track")
        def on_track(track):
            info("Track %s received", track.kind)

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
                info("Track %s ended", track.kind)

        async def run_recv_audio_track():
            while True:
                try:
                    frame = await self.recv_audio_track.recv()
                    self.last_message_time = time.time()
                    if not self.genai_session:
                        continue
                    await self.genai_session.send(frame)

                except Exception as e:
                    info("Error receiving frame: %s", e)
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
                    info("Error receiving frame: %s", e)
                    break

        async def run_recv_genai():
            while self.pc and self.pc.connectionState != "closed":
                async for frame in self.genai_session.recv():
                    self.output_queue.put_nowait(frame)

        async def run_send_track():
            timestamp = 0
            buffer = b""
            active = False
            sample_rate = 0
            samples = 0
            next_send_time = time.time()

            while self.pc and self.pc.connectionState != "closed":
                # Check if there's a frame in the output queue
                if not self.output_queue.empty():
                    frame = self.output_queue.get_nowait()
                    sample_rate = frame.sample_rate
                    samples = int(sample_rate * AUDIO_PTIME)
                    active = True
                    buffer += frame.to_ndarray().tobytes()

                # Don't send audio until we have at least one genai frame to
                # learn the sample rate
                if active:
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
                        "timestamp": time.time(),
                        "role": role,
                        "content": content,
                    }
                )

        def on_input_transcription(input_transcription):
            info(f"Input transcription: {input_transcription}")
            add_transcript("user", input_transcription)

        def on_output_transcription(output_transcription):
            info(f"Output transcription: {output_transcription}")
            add_transcript("model", output_transcription)

        def on_interrupted(event=None):
            info("Received INTERRUPTED event, clearing output queue")
            while not self.output_queue.empty():
                self.output_queue.get_nowait()
            while not self.send_track.queue.empty():
                self.send_track.queue.get_nowait()

        try:
            connect_genai = connect_openai if model == "openai" else connect_gemini
            async with connect_genai(self.system_instructions) as session:
                info("Connected to GenAI session")
                self.genai_session = session

                self.genai_session.on(ModelEvents.INPUT_TRANSCRIPTION, on_input_transcription)
                self.genai_session.on(ModelEvents.OUTPUT_TRANSCRIPTION, on_output_transcription)
                self.genai_session.on(ModelEvents.INTERRUPTED, on_interrupted)

                # Start the genai receiver task
                asyncio.ensure_future(run_recv_genai())

                await run_send_track()
                info("Connection finished")

        except Exception as e:
            info("Error sending frame: %s", e)

        try:
            await self.close()
        except Exception as e:
            info("Error closing connection: %s", e)

        # Notify parent about connection closure
        if self.on_closed:
            self.on_closed(self)

        info(f"Connection stopped. Transcript: {len(self.transcript)}.")

    async def _timeout_monitor(self, info):
        """Monitor for timeout - close connection if no messages received for 1 minute"""
        while self.pc and self.pc.connectionState != "closed":
            await asyncio.sleep(5)  # Check every 5 seconds
            if self.last_message_time and time.time() - self.last_message_time > 60:
                info("Connection timed out - no messages received for 1 minute")
                await self.close()
                break

    async def close(self):
        if self.timeout_task:
            self.timeout_task.cancel()
            self.timeout_task = None
        if self.pc:
            await self.pc.close()
            self.pc = None
        if self.genai_session:
            await self.genai_session.close()
            self.genai_session = None

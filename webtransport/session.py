"""Duck-typed 'peer connection' for WebTransport/raw-QUIC, mirroring sip/peerconnection.py.

connection.py drives any transport through a small aiortc-like interface
(connectionState / addTrack / on("track"|"datachannel"|"connectionstatechange") / close()).
This module implements that interface on top of framed binary messages
(see protocol.py) instead of RTP, so Connection._run() works unmodified.
"""

import asyncio
import io
import time

import numpy as np
from av import AudioFrame, VideoFrame
from PIL import Image
from pyee.asyncio import AsyncIOEventEmitter

from webtransport.protocol import (
    FRAME_TYPE_AUDIO,
    FRAME_TYPE_CONTROL,
    FRAME_TYPE_VIDEO,
    pack_audio,
    pack_control,
    pack_video,
)

DEFAULT_SAMPLE_RATE = 48000
JPEG_QUALITY = 80


class WebTransportTrack:
    """Receive-side track fed by incoming framed messages."""

    def __init__(self, kind: str):
        self.kind = kind
        self._queue = asyncio.Queue()

    async def recv(self):
        return await self._queue.get()


class WebTransportChannel(AsyncIOEventEmitter):
    """Datachannel-like object carrying JSON control messages."""

    label = "webtransport-control"

    def __init__(self, send_control):
        super().__init__()
        self._send_control = send_control
        self.readyState = "open"

    def send(self, message):
        if isinstance(message, str):
            message = message.encode("utf-8")
        self._send_control(message)

    def close(self):
        self.readyState = "closed"


class WebTransportPeerConnection(AsyncIOEventEmitter):
    """Adapts a single duplex byte stream (WebTransport or raw QUIC) to the pc-like interface."""

    def __init__(self, send_bytes, public_ip=None):
        super().__init__()
        self._send_bytes = send_bytes
        self._public_ip = public_ip
        self._recv_tracks = {}
        self._local_tracks = {}
        self._channel = None
        self._closed = False
        self._send_tasks = []

    @property
    def connectionState(self) -> str:
        return "closed" if self._closed else "connected"

    def addTrack(self, track):
        if track.kind in self._local_tracks:
            return
        self._local_tracks[track.kind] = track
        self._send_tasks.append(asyncio.ensure_future(self._run_send(track)))

    async def close(self):
        if self._closed:
            return
        self._closed = True
        for task in self._send_tasks:
            task.cancel()
        if self._channel:
            self._channel.close()

    def notify_connected(self):
        self.emit("connectionstatechange")

    def handle_frame(self, frame):
        """Called by the server transport layer with an already-decoded frame."""
        if frame.type == FRAME_TYPE_CONTROL:
            if self._channel is None:
                self._channel = WebTransportChannel(self._send_control)
                self.emit("datachannel", self._channel)
            self._channel.emit("message", frame.payload.decode("utf-8", errors="replace"))
        elif frame.type == FRAME_TYPE_AUDIO:
            self._handle_audio_frame(frame)
        elif frame.type == FRAME_TYPE_VIDEO:
            self._handle_video_frame(frame)

    def _handle_audio_frame(self, frame):
        track = self._recv_tracks.get("audio")
        if track is None:
            track = WebTransportTrack("audio")
            self._recv_tracks["audio"] = track
            self.emit("track", track)

        sample_rate = frame.extra.get("sample_rate", DEFAULT_SAMPLE_RATE)
        channels = frame.extra.get("channels", 1)
        samples = np.frombuffer(frame.payload, dtype=np.int16)
        if channels > 1:
            samples = samples.reshape(-1, channels)[:, 0].copy()

        av_frame = AudioFrame(format="s16", layout="mono", samples=len(samples))
        av_frame.sample_rate = sample_rate
        av_frame.planes[0].update(samples.tobytes())
        av_frame.pts = int(frame.timestamp_us * sample_rate / 1_000_000)
        track._queue.put_nowait(av_frame)

    def _handle_video_frame(self, frame):
        track = self._recv_tracks.get("video")
        if track is None:
            track = WebTransportTrack("video")
            self._recv_tracks["video"] = track
            self.emit("track", track)

        image = Image.open(io.BytesIO(frame.payload)).convert("RGB")
        av_frame = VideoFrame.from_image(image)
        av_frame.pts = frame.timestamp_us
        track._queue.put_nowait(av_frame)

    def _send_control(self, payload: bytes):
        self._send_bytes(pack_control(payload))

    async def _run_send(self, track):
        while not self._closed:
            frame = await track.recv()
            try:
                if track.kind == "audio" and isinstance(frame, AudioFrame):
                    pcm = frame.to_ndarray().astype(np.int16).tobytes()
                    self._send_bytes(
                        pack_audio(
                            pcm,
                            sample_rate=frame.sample_rate,
                            channels=1,
                            timestamp_us=int(time.time() * 1_000_000),
                        )
                    )
                elif track.kind == "video" and isinstance(frame, VideoFrame):
                    image = frame.to_image()
                    buf = io.BytesIO()
                    image.save(buf, format="JPEG", quality=JPEG_QUALITY)
                    self._send_bytes(pack_video(buf.getvalue(), timestamp_us=int(time.time() * 1_000_000)))
            except Exception:
                pass

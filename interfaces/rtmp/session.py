"""Duck-typed 'peer connection' for RTMP ingest, mirroring interfaces/webtransport/session.py.

connection.py drives any transport through a small aiortc-like interface
(connectionState / addTrack / on("track"|"connectionstatechange") / close()).
This module implements that interface on top of FLV audio/video tags received
over an RTMP publish session: AAC and H.264 payloads are decoded with PyAV so
Connection._run() consumes plain AudioFrame/VideoFrame objects unmodified.

RTMP publishing is one-way (client -> server), so outgoing tracks are dropped.
"""

import asyncio
import fractions

import av
from pyee.asyncio import AsyncIOEventEmitter

from logger import log_trace, log_warn

FLV_CODEC_AAC = 10
FLV_CODEC_AVC = 7

VIDEO_CLOCK_RATE = 90000
VIDEO_TIME_BASE = fractions.Fraction(1, VIDEO_CLOCK_RATE)


class RTMPTrack(AsyncIOEventEmitter):
    """Receive-side track fed by decoded RTMP audio/video messages.

    Extends AsyncIOEventEmitter because Connection registers an on("ended") handler."""

    def __init__(self, kind: str):
        super().__init__()
        self.kind = kind
        self._queue = asyncio.Queue()

    async def recv(self):
        return await self._queue.get()


class RTMPPeerConnection(AsyncIOEventEmitter):
    """Adapts an RTMP publish session to the pc-like interface used by Connection."""

    def __init__(self, context=None):
        super().__init__()
        self._context = context
        self._recv_tracks = {}
        self._closed = False
        self._audio_codec = None
        self._video_codec = None
        self._frame_counts = {"audio": 0, "video": 0}
        self._unsupported_warned = set()

    @property
    def connectionState(self) -> str:
        return "closed" if self._closed else "connected"

    def addTrack(self, track):
        # RTMP publish sessions are receive-only; there is no return media path.
        pass

    async def close(self):
        if self._closed:
            return
        self._closed = True
        self.emit("connectionstatechange")

    def notify_connected(self):
        self.emit("connectionstatechange")

    def handle_audio(self, timestamp_ms: int, payload: bytes):
        """Handle an RTMP audio message (FLV audio tag body)."""
        if self._closed or len(payload) < 2:
            return
        sound_format = payload[0] >> 4
        if sound_format != FLV_CODEC_AAC:
            self._warn_unsupported(f"audio format {sound_format} (only AAC is supported)")
            return

        packet_type = payload[1]
        if packet_type == 0:  # AAC sequence header (AudioSpecificConfig)
            self._audio_codec = av.CodecContext.create("aac", "r")
            self._audio_codec.extradata = bytes(payload[2:])
            return
        if self._audio_codec is None:
            return

        try:
            for frame in self._audio_codec.decode(av.Packet(payload[2:])):
                frame.pts = int(timestamp_ms * frame.sample_rate / 1000)
                frame.time_base = fractions.Fraction(1, frame.sample_rate)
                self._push_frame("audio", frame, timestamp_ms)
        except Exception as e:
            log_trace(f"RTMP audio decode error: {e}", context=self._context)

    def handle_video(self, timestamp_ms: int, payload: bytes):
        """Handle an RTMP video message (FLV video tag body)."""
        if self._closed or len(payload) < 5:
            return
        codec_id = payload[0] & 0x0F
        if payload[0] & 0x80 or codec_id != FLV_CODEC_AVC:
            self._warn_unsupported(f"video codec {codec_id} (only H.264 is supported)")
            return

        avc_packet_type = payload[1]
        if avc_packet_type == 0:  # AVC sequence header (AVCDecoderConfigurationRecord)
            self._video_codec = av.CodecContext.create("h264", "r")
            self._video_codec.extradata = bytes(payload[5:])
            return
        if avc_packet_type != 1 or self._video_codec is None:
            return

        try:
            for frame in self._video_codec.decode(av.Packet(payload[5:])):
                frame.pts = timestamp_ms * (VIDEO_CLOCK_RATE // 1000)
                frame.time_base = VIDEO_TIME_BASE
                self._push_frame("video", frame, timestamp_ms)
        except Exception as e:
            log_trace(f"RTMP video decode error: {e}", context=self._context)

    def _push_frame(self, kind: str, frame, timestamp_ms: int):
        track = self._recv_tracks.get(kind)
        if track is None:
            track = RTMPTrack(kind)
            self._recv_tracks[kind] = track
            self.emit("track", track)

        self._frame_counts[kind] += 1
        if kind == "audio":
            log_trace(
                f"RTMP audio frame received #{self._frame_counts['audio']} ts={timestamp_ms}ms "
                f"samples={frame.samples} rate={frame.sample_rate}",
                context=self._context,
            )
        else:
            log_trace(
                f"RTMP video frame received #{self._frame_counts['video']} ts={timestamp_ms}ms "
                f"size={frame.width}x{frame.height}",
                context=self._context,
            )
        track._queue.put_nowait(frame)

    def _warn_unsupported(self, what: str):
        if what not in self._unsupported_warned:
            self._unsupported_warned.add(what)
            log_warn(f"RTMP: unsupported {what}", context=self._context)

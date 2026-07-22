"""Connection recorders.

Modeled on aiortc's own MediaRecorder (aiortc/contrib/media.py): PyAV opens
an output container, one encoded stream is added per media kind on first
frame, and every decoded frame received from the connection's tracks is fed
through `stream.encode()` and muxed as it arrives — no separate encode pass,
no temp files. FileRecorder follows that approach directly (mp4 container).
MCAPRecorder swaps the PyAV container for an MCAP writer, since MCAP has no
built-in encoded A/V container concept — each frame becomes one message on
a per-kind topic instead of a muxed stream.
"""

import base64
import json
import os
import time
from typing import Optional, cast

import av
import av.container
import cv2
from av import AudioFrame, VideoFrame
from av.audio.stream import AudioStream
from av.video.stream import VideoStream
from mcap.writer import Writer as McapWriter

from logger import log_info, log_warn


class Recorder:
    """Base class for connection recorders. Subclasses receive every decoded
    audio/video frame the connection's tracks produce and are responsible
    for writing them to `path` in their own format."""

    def __init__(self, path: str):
        self.path = path

    def add_audio_frame(self, frame: AudioFrame) -> None:
        raise NotImplementedError

    def add_video_frame(self, frame: VideoFrame) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class FileRecorder(Recorder):
    """Records the connection's audio+video tracks into a single .mp4 file,
    the same way aiortc.contrib.media.MediaRecorder does: PyAV (Python's
    wrapper around ffmpeg's libav* libraries) opens an output container,
    adds one encoded stream per track (aac / libx264), and muxes each
    decoded frame as it arrives."""

    def __init__(self, path: str):
        super().__init__(path if path.endswith(".mp4") else f"{path}.mp4")
        self._container: Optional[av.container.OutputContainer] = av.open(self.path, mode="w")
        self._audio_stream: Optional[AudioStream] = None
        self._video_stream: Optional[VideoStream] = None

    def add_audio_frame(self, frame: AudioFrame) -> None:
        if self._container is None:
            return
        if self._audio_stream is None:
            self._audio_stream = cast(AudioStream, self._container.add_stream("aac"))
        for packet in self._audio_stream.encode(frame):
            self._container.mux(packet)

    def add_video_frame(self, frame: VideoFrame) -> None:
        if self._container is None:
            return
        if self._video_stream is None:
            self._video_stream = cast(VideoStream, self._container.add_stream("libx264", rate=30))
            self._video_stream.pix_fmt = "yuv420p"
            self._video_stream.width = frame.width
            self._video_stream.height = frame.height
        for packet in self._video_stream.encode(frame):
            self._container.mux(packet)

    def close(self) -> None:
        if self._container is None:
            return
        for stream in (self._audio_stream, self._video_stream):
            if stream is None:
                continue
            try:
                for packet in stream.encode(None):
                    self._container.mux(packet)
            except Exception as e:
                log_warn(f"FileRecorder: error flushing stream: {e}")
        self._container.close()
        self._container = None
        log_info(f"FileRecorder wrote {self.path}")


# Minimal jsonschema definitions for two well-known Foxglove message types
# (https://docs.foxglove.dev/docs/visualization/message-schemas/introduction)
# so files produced by MCAPRecorder open directly in Foxglove Studio without
# any custom extension/plugin. Byte fields are base64-encoded, matching the
# JSON mapping convention MCAP's "json" message encoding follows.
_COMPRESSED_IMAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "timestamp": {
            "type": "object",
            "properties": {"sec": {"type": "integer"}, "nsec": {"type": "integer"}},
        },
        "frame_id": {"type": "string"},
        "data": {"type": "string", "contentEncoding": "base64"},
        "format": {"type": "string"},
    },
}

_RAW_AUDIO_SCHEMA = {
    "type": "object",
    "properties": {
        "timestamp": {
            "type": "object",
            "properties": {"sec": {"type": "integer"}, "nsec": {"type": "integer"}},
        },
        "data": {"type": "string", "contentEncoding": "base64"},
        "format": {"type": "string"},
        "sample_rate": {"type": "integer"},
        "number_of_channels": {"type": "integer"},
    },
}


def _ros_time(now_ns: int) -> dict:
    return {"sec": now_ns // 1_000_000_000, "nsec": now_ns % 1_000_000_000}


class MCAPRecorder(Recorder):
    """Records the connection's audio+video tracks into a single .mcap file
    (https://mcap.dev/), the standard log container used for robotics
    recordings. Video frames are JPEG-compressed and written as
    `foxglove.CompressedImage` messages on topic "/video"; audio frames are
    written as raw PCM `foxglove.RawAudio` messages on topic "/audio" —
    both well-known Foxglove schemas."""

    def __init__(self, path: str):
        super().__init__(path if path.endswith(".mcap") else f"{path}.mcap")
        self._file = open(self.path, "wb")
        self._writer: Optional[McapWriter] = McapWriter(self._file)
        self._writer.start()
        self._video_channel_id: Optional[int] = None
        self._audio_channel_id: Optional[int] = None

    def _register_video_channel(self) -> int:
        schema_id = self._writer.register_schema(
            name="foxglove.CompressedImage",
            encoding="jsonschema",
            data=json.dumps(_COMPRESSED_IMAGE_SCHEMA).encode(),
        )
        return self._writer.register_channel(topic="/video", message_encoding="json", schema_id=schema_id)

    def _register_audio_channel(self) -> int:
        schema_id = self._writer.register_schema(
            name="foxglove.RawAudio",
            encoding="jsonschema",
            data=json.dumps(_RAW_AUDIO_SCHEMA).encode(),
        )
        return self._writer.register_channel(topic="/audio", message_encoding="json", schema_id=schema_id)

    def add_video_frame(self, frame: VideoFrame) -> None:
        if self._writer is None:
            return
        if self._video_channel_id is None:
            self._video_channel_id = self._register_video_channel()

        ok, jpeg = cv2.imencode(".jpg", frame.to_ndarray(format="bgr24"))
        if not ok:
            log_warn("MCAPRecorder: failed to JPEG-encode video frame")
            return

        now_ns = time.time_ns()
        message = {
            "timestamp": _ros_time(now_ns),
            "frame_id": "",
            "data": base64.b64encode(jpeg.tobytes()).decode("ascii"),
            "format": "jpeg",
        }
        self._writer.add_message(
            channel_id=self._video_channel_id,
            log_time=now_ns,
            publish_time=now_ns,
            data=json.dumps(message).encode(),
        )

    def add_audio_frame(self, frame: AudioFrame) -> None:
        if self._writer is None:
            return
        if self._audio_channel_id is None:
            self._audio_channel_id = self._register_audio_channel()

        pcm = frame.to_ndarray().astype("int16").tobytes()
        now_ns = time.time_ns()
        message = {
            "timestamp": _ros_time(now_ns),
            "data": base64.b64encode(pcm).decode("ascii"),
            "format": "pcm-s16",
            "sample_rate": frame.sample_rate,
            "number_of_channels": len(frame.layout.channels) if frame.layout else 1,
        }
        self._writer.add_message(
            channel_id=self._audio_channel_id,
            log_time=now_ns,
            publish_time=now_ns,
            data=json.dumps(message).encode(),
        )

    def close(self) -> None:
        if self._writer is None:
            return
        try:
            self._writer.finish()
        finally:
            self._file.close()
            self._writer = None
        log_info(f"MCAPRecorder wrote {self.path}")


RECORDINGS_DIR = os.environ.get("RECORDINGS_DIR", "recordings")


def create_recorder(recording, connection_id: str) -> Optional[Recorder]:
    """Build the recorder requested for a connection, or None.

    `recording` mirrors the `recording` field accepted by the create
    connection API: falsy disables recording, `True` (or "file") selects
    FileRecorder (.mp4), "mcap" selects MCAPRecorder (.mcap).
    """
    if not recording:
        return None

    os.makedirs(RECORDINGS_DIR, exist_ok=True)
    path = os.path.join(RECORDINGS_DIR, connection_id)

    if recording == "mcap":
        return MCAPRecorder(path)
    return FileRecorder(path)

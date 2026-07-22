"""Binary framing for audio/video/control messages sent over a WebTransport, raw QUIC or
WebSocket connection.

Each message is length-prefixed so it can be parsed out of a byte stream:

    [4 bytes: length][1 byte: type][8 bytes: timestamp_us][type-specific extra][payload]

Types:
    AUDIO:   extra = [4 bytes sample_rate][1 byte channels], payload = raw PCM s16le
    VIDEO:   extra = [1 byte flags (bit0=keyframe)], payload = JPEG-encoded frame
    CONTROL: extra = (none), payload = UTF-8 JSON
"""

import struct

FRAME_TYPE_AUDIO = 1
FRAME_TYPE_VIDEO = 2
FRAME_TYPE_CONTROL = 3

_LENGTH = struct.Struct(">I")
_HEADER = struct.Struct(">BQ")  # type, timestamp_us
_AUDIO_EXTRA = struct.Struct(">IB")  # sample_rate, channels
_VIDEO_EXTRA = struct.Struct(">B")  # flags


def pack_audio(payload: bytes, sample_rate: int, channels: int = 1, timestamp_us: int = 0) -> bytes:
    body = _HEADER.pack(FRAME_TYPE_AUDIO, timestamp_us) + _AUDIO_EXTRA.pack(sample_rate, channels) + payload
    return _LENGTH.pack(len(body)) + body


def pack_video(payload: bytes, timestamp_us: int = 0, keyframe: bool = True) -> bytes:
    flags = 1 if keyframe else 0
    body = _HEADER.pack(FRAME_TYPE_VIDEO, timestamp_us) + _VIDEO_EXTRA.pack(flags) + payload
    return _LENGTH.pack(len(body)) + body


def pack_control(payload: bytes, timestamp_us: int = 0) -> bytes:
    body = _HEADER.pack(FRAME_TYPE_CONTROL, timestamp_us) + payload
    return _LENGTH.pack(len(body)) + body


class Frame:
    __slots__ = ("type", "timestamp_us", "extra", "payload")

    def __init__(self, type_, timestamp_us, extra, payload):
        self.type = type_
        self.timestamp_us = timestamp_us
        self.extra = extra
        self.payload = payload


class FrameDecoder:
    """Incrementally parses length-prefixed frames out of a stream of bytes."""

    def __init__(self):
        self._buffer = b""

    def feed(self, data: bytes) -> list[Frame]:
        self._buffer += data
        frames = []
        while True:
            if len(self._buffer) < _LENGTH.size:
                break
            (length,) = _LENGTH.unpack_from(self._buffer, 0)
            end = _LENGTH.size + length
            if len(self._buffer) < end:
                break
            body = self._buffer[_LENGTH.size : end]
            self._buffer = self._buffer[end:]
            frames.append(self._parse(body))
        return frames

    def _parse(self, body: bytes) -> Frame:
        frame_type, timestamp_us = _HEADER.unpack_from(body, 0)
        offset = _HEADER.size
        extra = {}
        if frame_type == FRAME_TYPE_AUDIO:
            sample_rate, channels = _AUDIO_EXTRA.unpack_from(body, offset)
            offset += _AUDIO_EXTRA.size
            extra = {"sample_rate": sample_rate, "channels": channels}
        elif frame_type == FRAME_TYPE_VIDEO:
            (flags,) = _VIDEO_EXTRA.unpack_from(body, offset)
            offset += _VIDEO_EXTRA.size
            extra = {"keyframe": bool(flags & 1)}
        return Frame(frame_type, timestamp_us, extra, body[offset:])

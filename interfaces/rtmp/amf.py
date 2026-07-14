"""Minimal AMF0 encoder/decoder, just enough for RTMP command messages."""

import struct

AMF0_NUMBER = 0x00
AMF0_BOOLEAN = 0x01
AMF0_STRING = 0x02
AMF0_OBJECT = 0x03
AMF0_NULL = 0x05
AMF0_UNDEFINED = 0x06
AMF0_ECMA_ARRAY = 0x08
AMF0_OBJECT_END = 0x09
AMF0_STRICT_ARRAY = 0x0A
AMF0_LONG_STRING = 0x0C


def _read_utf8(data: bytes, pos: int) -> tuple[str, int]:
    (length,) = struct.unpack_from(">H", data, pos)
    pos += 2
    return data[pos : pos + length].decode("utf-8", errors="replace"), pos + length


def _decode_object(data: bytes, pos: int) -> tuple[dict, int]:
    obj = {}
    while True:
        key, pos = _read_utf8(data, pos)
        if key == "" and data[pos] == AMF0_OBJECT_END:
            return obj, pos + 1
        value, pos = decode(data, pos)
        obj[key] = value


def decode(data: bytes, pos: int = 0):
    """Decode a single AMF0 value, returning (value, next_pos)."""
    marker = data[pos]
    pos += 1
    if marker == AMF0_NUMBER:
        return struct.unpack_from(">d", data, pos)[0], pos + 8
    if marker == AMF0_BOOLEAN:
        return bool(data[pos]), pos + 1
    if marker == AMF0_STRING:
        return _read_utf8(data, pos)
    if marker == AMF0_OBJECT:
        return _decode_object(data, pos)
    if marker in (AMF0_NULL, AMF0_UNDEFINED):
        return None, pos
    if marker == AMF0_ECMA_ARRAY:
        return _decode_object(data, pos + 4)  # skip the (approximate) entry count
    if marker == AMF0_STRICT_ARRAY:
        (count,) = struct.unpack_from(">I", data, pos)
        pos += 4
        items = []
        for _ in range(count):
            value, pos = decode(data, pos)
            items.append(value)
        return items, pos
    if marker == AMF0_LONG_STRING:
        (length,) = struct.unpack_from(">I", data, pos)
        pos += 4
        return data[pos : pos + length].decode("utf-8", errors="replace"), pos + length
    raise ValueError(f"Unsupported AMF0 marker: {marker:#x}")


def decode_all(data: bytes) -> list:
    """Decode all consecutive AMF0 values in the buffer."""
    values = []
    pos = 0
    while pos < len(data):
        value, pos = decode(data, pos)
        values.append(value)
    return values


def encode(value) -> bytes:
    if value is None:
        return bytes([AMF0_NULL])
    if isinstance(value, bool):
        return bytes([AMF0_BOOLEAN, 1 if value else 0])
    if isinstance(value, (int, float)):
        return bytes([AMF0_NUMBER]) + struct.pack(">d", float(value))
    if isinstance(value, str):
        raw = value.encode("utf-8")
        return bytes([AMF0_STRING]) + struct.pack(">H", len(raw)) + raw
    if isinstance(value, dict):
        out = bytearray([AMF0_OBJECT])
        for key, item in value.items():
            raw = key.encode("utf-8")
            out += struct.pack(">H", len(raw)) + raw + encode(item)
        out += b"\x00\x00" + bytes([AMF0_OBJECT_END])
        return bytes(out)
    raise ValueError(f"Cannot encode {type(value)} as AMF0")


def encode_all(*values) -> bytes:
    return b"".join(encode(value) for value in values)

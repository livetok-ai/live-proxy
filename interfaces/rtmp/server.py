"""RTMP ingest server exposing live-proxy connections to RTMP publishers (e.g. ffmpeg/OBS).

Implements the minimum of the RTMP spec needed to accept a publish session:
simple handshake, chunk stream demuxing, and the AMF0 command exchange
(connect / createStream / publish). Once a client publishes, an
RTMPPeerConnection (see session.py) is created and handed to on_session with
the connection parameters parsed from the stream key query string
(e.g. rtmp://host/live/mystream?model=yolo).
"""

import asyncio
import os
import ssl
import struct
import urllib.parse

import aiohttp

from interfaces.rtmp import amf
from interfaces.rtmp.session import RTMPPeerConnection
from logger import log_debug, log_info, log_trace, log_warn
from utils import default_connection_params

RTMP_VERSION = 3
HANDSHAKE_SIZE = 1536
DEFAULT_CHUNK_SIZE = 128
OUT_CHUNK_SIZE = 4096
WINDOW_ACK_SIZE = 2_500_000

MSG_SET_CHUNK_SIZE = 1
MSG_ABORT = 2
MSG_ACK = 3
MSG_USER_CONTROL = 4
MSG_WINDOW_ACK_SIZE = 5
MSG_SET_PEER_BANDWIDTH = 6
MSG_AUDIO = 8
MSG_VIDEO = 9
MSG_DATA_AMF0 = 18
MSG_COMMAND_AMF0 = 20

CONTROL_CSID = 2
COMMAND_CSID = 3


class _ChunkStreamState:
    """Per-chunk-stream-id demuxing state (last message header + partial payload)."""

    def __init__(self):
        self.timestamp = 0
        self.timestamp_delta = 0
        self.length = 0
        self.type_id = 0
        self.stream_id = 0
        self.extended = False
        self.buffer = b""


class RTMPClientHandler:
    """Handles a single RTMP client connection."""

    def __init__(self, reader, writer, on_session, callback_url=None):
        self._reader = reader
        self._writer = writer
        self._on_session = on_session
        self._callback_url = callback_url
        self._app = None
        self._chunk_streams = {}
        self._in_chunk_size = DEFAULT_CHUNK_SIZE
        self._out_chunk_size = DEFAULT_CHUNK_SIZE
        self._ack_window = WINDOW_ACK_SIZE
        self._bytes_in = 0
        self._last_ack = 0
        self.pc = None

    async def run(self):
        await self._handshake()
        while True:
            type_id, stream_id, timestamp, payload = await self._read_message()
            await self._dispatch(type_id, stream_id, timestamp, payload)
            if self._bytes_in - self._last_ack >= self._ack_window:
                self._last_ack = self._bytes_in
                self._send_message(CONTROL_CSID, MSG_ACK, 0, struct.pack(">I", self._bytes_in & 0xFFFFFFFF))

    async def close(self):
        if self.pc:
            await self.pc.close()
            self.pc = None
        self._writer.close()

    async def _readexactly(self, n: int) -> bytes:
        data = await self._reader.readexactly(n)
        self._bytes_in += n
        return data

    async def _handshake(self):
        c0 = await self._readexactly(1)
        if c0[0] != RTMP_VERSION:
            raise ValueError(f"Unsupported RTMP version {c0[0]}")
        c1 = await self._readexactly(HANDSHAKE_SIZE)
        s1 = b"\x00" * 8 + os.urandom(HANDSHAKE_SIZE - 8)
        self._writer.write(bytes([RTMP_VERSION]) + s1 + c1)  # S0 + S1 + S2 (echo of C1)
        await self._writer.drain()
        await self._readexactly(HANDSHAKE_SIZE)  # C2

    async def _read_message(self):
        """Read chunks until a complete message is assembled."""
        while True:
            message = await self._read_chunk()
            if message is not None:
                return message

    async def _read_chunk(self):
        b0 = (await self._readexactly(1))[0]
        fmt = b0 >> 6
        csid = b0 & 0x3F
        if csid == 0:
            csid = 64 + (await self._readexactly(1))[0]
        elif csid == 1:
            ext = await self._readexactly(2)
            csid = 64 + ext[0] + ext[1] * 256

        cs = self._chunk_streams.setdefault(csid, _ChunkStreamState())

        if fmt == 0:
            header = await self._readexactly(11)
            timestamp = int.from_bytes(header[0:3], "big")
            cs.length = int.from_bytes(header[3:6], "big")
            cs.type_id = header[6]
            cs.stream_id = int.from_bytes(header[7:11], "little")
            cs.extended = timestamp == 0xFFFFFF
            if cs.extended:
                timestamp = int.from_bytes(await self._readexactly(4), "big")
            cs.timestamp = timestamp
            cs.timestamp_delta = 0
        elif fmt == 1:
            header = await self._readexactly(7)
            delta = int.from_bytes(header[0:3], "big")
            cs.length = int.from_bytes(header[3:6], "big")
            cs.type_id = header[6]
            cs.extended = delta == 0xFFFFFF
            if cs.extended:
                delta = int.from_bytes(await self._readexactly(4), "big")
            cs.timestamp_delta = delta
            cs.timestamp += delta
        elif fmt == 2:
            delta = int.from_bytes(await self._readexactly(3), "big")
            cs.extended = delta == 0xFFFFFF
            if cs.extended:
                delta = int.from_bytes(await self._readexactly(4), "big")
            cs.timestamp_delta = delta
            cs.timestamp += delta
        else:  # fmt == 3: no header; timestamp advances only when starting a new message
            if cs.extended:
                await self._readexactly(4)
            if not cs.buffer:
                cs.timestamp += cs.timestamp_delta

        remaining = cs.length - len(cs.buffer)
        cs.buffer += await self._readexactly(min(remaining, self._in_chunk_size))

        if len(cs.buffer) == cs.length:
            payload, cs.buffer = cs.buffer, b""
            return cs.type_id, cs.stream_id, cs.timestamp, payload
        return None

    async def _dispatch(self, type_id, stream_id, timestamp, payload):
        if type_id == MSG_AUDIO:
            if self.pc:
                self.pc.handle_audio(timestamp, payload)
        elif type_id == MSG_VIDEO:
            if self.pc:
                self.pc.handle_video(timestamp, payload)
        elif type_id == MSG_SET_CHUNK_SIZE:
            self._in_chunk_size = struct.unpack(">I", payload[:4])[0] & 0x7FFFFFFF
            log_debug(f"RTMP: client set chunk size to {self._in_chunk_size}")
        elif type_id == MSG_WINDOW_ACK_SIZE:
            self._ack_window = struct.unpack(">I", payload[:4])[0]
        elif type_id == MSG_COMMAND_AMF0:
            await self._handle_command(payload, stream_id)
        elif type_id == MSG_DATA_AMF0:
            try:
                log_debug(f"RTMP: data message: {amf.decode_all(payload)}")
            except Exception as e:
                log_warn(f"RTMP: could not decode data message: {e}")
        elif type_id in (MSG_ACK, MSG_USER_CONTROL, MSG_ABORT, MSG_SET_PEER_BANDWIDTH):
            pass
        else:
            log_trace(f"RTMP: ignoring message type {type_id} ({len(payload)} bytes)")

    async def _handle_command(self, payload, stream_id):
        try:
            values = amf.decode_all(payload)
        except Exception as e:
            log_warn(f"RTMP: could not decode command: {e}")
            return
        if not values:
            return

        name = values[0]
        transaction_id = values[1] if len(values) > 1 else 0
        log_debug(f"RTMP: command '{name}' received")

        if name == "connect":
            if len(values) > 2 and isinstance(values[2], dict):
                self._app = values[2].get("app")
            self._send_message(CONTROL_CSID, MSG_WINDOW_ACK_SIZE, 0, struct.pack(">I", WINDOW_ACK_SIZE))
            self._send_message(CONTROL_CSID, MSG_SET_PEER_BANDWIDTH, 0, struct.pack(">IB", WINDOW_ACK_SIZE, 2))
            self._send_message(CONTROL_CSID, MSG_SET_CHUNK_SIZE, 0, struct.pack(">I", OUT_CHUNK_SIZE))
            self._out_chunk_size = OUT_CHUNK_SIZE
            self._send_command(
                "_result",
                transaction_id,
                {"fmsVer": "FMS/3,0,1,123", "capabilities": 31},
                {
                    "level": "status",
                    "code": "NetConnection.Connect.Success",
                    "description": "Connection succeeded.",
                    "objectEncoding": 0,
                },
            )
        elif name == "createStream":
            self._send_command("_result", transaction_id, None, 1)
        elif name == "publish":
            stream_key = values[3] if len(values) > 3 and isinstance(values[3], str) else ""
            try:
                await self._start_session(stream_key)
            except Exception:
                self._send_command(
                    "onStatus",
                    0,
                    None,
                    {"level": "error", "code": "NetStream.Publish.BadName", "description": "Failed to start session."},
                    stream_id=stream_id,
                )
                raise
            self._send_command(
                "onStatus",
                0,
                None,
                {"level": "status", "code": "NetStream.Publish.Start", "description": "Publishing."},
                stream_id=stream_id,
            )
        elif name in ("deleteStream", "FCUnpublish", "closeStream"):
            if self.pc:
                await self.pc.close()
                self.pc = None
        # releaseStream / FCPublish need no response

    async def _start_session(self, stream_key: str):
        stream_name, _, query = stream_key.partition("?")
        path = f"{self._app}/{stream_name}" if self._app else stream_name

        if self._callback_url:
            success, params = await self._make_callback(path)
            if not success:
                raise RuntimeError(f"Callback to {self._callback_url} failed")
            params = params or {}
        else:
            params = default_connection_params()
            if params:
                log_info(f"RTMP: no callback URL configured, using default connection params: {list(params.keys())}")

        # Stream key query parameters override the callback/default configuration
        params.update({key: value[0] for key, value in urllib.parse.parse_qs(query).items()})
        params.setdefault("metadata", {"source": "rtmp", "path": path})

        log_info(f"RTMP: publish start path='{path}' params={list(params.keys())}")

        self.pc = RTMPPeerConnection(context=f"rtmp:{stream_name}")
        self.pc.notify_connected()
        await self._on_session(self.pc, params)

    async def _make_callback(self, path: str):
        """Make HTTP callback to retrieve the connection configuration for a publish path.

        Returns (success, response_data), mirroring SIPServer.make_callback."""
        try:
            payload = {"path": path}
            log_info(f"RTMP: making callback to {self._callback_url} with payload: {payload}")

            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            connector = aiohttp.TCPConnector(ssl=ssl_context)

            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.post(
                    self._callback_url, json=payload, timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    log_info(f"RTMP: callback response: {response.status}")
                    if 200 <= response.status < 300:
                        try:
                            return True, await response.json()
                        except Exception as e:
                            log_warn(f"RTMP: error parsing callback response: {e}")
                            return True, {}
                    return False, None
        except Exception as e:
            log_warn(f"RTMP: error making callback to {self._callback_url}: {e}")
            return False, None

    def _send_command(self, name, transaction_id, *args, stream_id=0):
        self._send_message(COMMAND_CSID, MSG_COMMAND_AMF0, stream_id, amf.encode_all(name, transaction_id, *args))

    def _send_message(self, csid, type_id, stream_id, payload, timestamp=0):
        out = bytearray()
        out += bytes([csid])  # fmt 0 basic header
        out += timestamp.to_bytes(3, "big")
        out += len(payload).to_bytes(3, "big")
        out += bytes([type_id])
        out += stream_id.to_bytes(4, "little")
        for pos in range(0, len(payload), self._out_chunk_size):
            if pos > 0:
                out += bytes([0xC0 | csid])  # fmt 3 continuation
            out += payload[pos : pos + self._out_chunk_size]
        self._writer.write(bytes(out))


class RTMPServer:
    """TCP server accepting RTMP publish sessions."""

    def __init__(self, host: str, port: int, on_session, callback_url=None):
        self.host = host
        self.port = port
        self._on_session = on_session
        self.callback_url = callback_url
        self._server = None

    async def _handle_client(self, reader, writer):
        addr = writer.get_extra_info("peername")
        log_info(f"RTMP: new connection from {addr}")
        handler = RTMPClientHandler(reader, writer, self._on_session, callback_url=self.callback_url)
        try:
            await handler.run()
        except (asyncio.IncompleteReadError, ConnectionResetError, BrokenPipeError):
            log_info(f"RTMP: connection from {addr} closed by peer")
        except Exception as e:
            log_warn(f"RTMP: error handling client {addr}: {e}")
        finally:
            try:
                await handler.close()
            except Exception:
                pass

    async def bind(self):
        """Bind the TCP socket and start accepting connections. Returns once ready."""
        self._server = await asyncio.start_server(self._handle_client, self.host, self.port)
        log_info(f"RTMP server listening on {self.host}:{self.port} (TCP)")
        log_info(f"RTMP server callback URL: {self.callback_url or '<empty>'}")

    async def start(self):
        """Bind and run until closed. Suitable for asyncio.gather() alongside other servers."""
        if self._server is None:
            await self.bind()
        async with self._server:
            await self._server.serve_forever()

    def close(self):
        if self._server:
            self._server.close()

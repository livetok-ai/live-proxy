"""QUIC server exposing live-proxy connections over WebTransport (HTTP/3) and raw QUIC.

Transport/HTTP-3/WebTransport handshaking is all handled by aioquic; the only
thing this module owns is: (1) picking the right aioquic API for each ALPN,
and (2) bootstrapping a Connection once the first control frame (containing
the connection parameters as JSON, see protocol.py) arrives on a session's
media stream.
"""

import asyncio
import json
from typing import Callable, Optional

from aioquic.asyncio import QuicConnectionProtocol, serve
from aioquic.h3.connection import H3_ALPN, H3Connection
from aioquic.h3.events import H3Event, HeadersReceived, WebTransportStreamDataReceived
from aioquic.quic.configuration import QuicConfiguration
from aioquic.quic.events import ProtocolNegotiated, QuicEvent

from interfaces.webtransport.certs import certificate_hash_base64, generate_self_signed_cert
from interfaces.webtransport.protocol import FRAME_TYPE_CONTROL, FrameDecoder
from interfaces.webtransport.session import WebTransportPeerConnection
from logger import log_info, log_warn

RAW_QUIC_ALPN = "live-proxy-quic"


async def run_media_session(send_bytes: Callable[[bytes], None], recv_chunk, on_session):
    """Shared bootstrap for any framed transport (WebTransport, raw QUIC, WebSocket):
    wait for the init control frame, build the Connection from its JSON payload,
    then keep feeding frames."""
    decoder = FrameDecoder()
    pc: Optional[WebTransportPeerConnection] = None

    while True:
        data = await recv_chunk()
        if not data:
            break

        frames = decoder.feed(data)
        if pc is None:
            init_index = next((i for i, f in enumerate(frames) if f.type == FRAME_TYPE_CONTROL), None)
            if init_index is None:
                continue
            try:
                params = json.loads(frames[init_index].payload.decode("utf-8"))
            except Exception as e:
                log_warn(f"Invalid WebTransport init message: {e}")
                break

            pc = WebTransportPeerConnection(send_bytes=send_bytes)
            pc.notify_connected()
            await on_session(pc, params)

            for frame in frames[init_index + 1 :]:
                pc.handle_frame(frame)
        else:
            for frame in frames:
                pc.handle_frame(frame)

    if pc is not None:
        await pc.close()


class QuicMediaProtocol(QuicConnectionProtocol):
    def __init__(self, *args, on_session, **kwargs):
        super().__init__(*args, **kwargs)
        self._on_session = on_session
        self._http: Optional[H3Connection] = None
        self._session_tasks = {}  # webtransport session_id -> asyncio.Task

    def quic_event_received(self, event: QuicEvent) -> None:
        if isinstance(event, ProtocolNegotiated):
            if event.alpn_protocol in H3_ALPN:
                self._http = H3Connection(self._quic, enable_webtransport=True)
            return

        if self._http is not None:
            for h3_event in self._http.handle_event(event):
                self._h3_event_received(h3_event)
        else:
            # Raw QUIC: fall back to the default stream_handler-based dispatch.
            super().quic_event_received(event)

    def _h3_event_received(self, event: H3Event) -> None:
        if isinstance(event, HeadersReceived):
            headers = dict(event.headers)
            is_connect = headers.get(b":method") == b"CONNECT" and headers.get(b":protocol") == b"webtransport"
            if is_connect:
                self._http.send_headers(event.stream_id, [(b":status", b"200")])
            else:
                self._http.send_headers(event.stream_id, [(b":status", b"404")], end_stream=True)
        elif isinstance(event, WebTransportStreamDataReceived):
            self._webtransport_stream_data_received(event.session_id, event.stream_id, event.data)

    def _webtransport_stream_data_received(self, session_id: int, stream_id: int, data: bytes) -> None:
        if session_id not in self._session_tasks:
            queue: asyncio.Queue = asyncio.Queue()

            def send_bytes(payload: bytes, sid=stream_id):
                self._quic.send_stream_data(sid, payload)
                self.transmit()

            async def recv_chunk():
                return await queue.get()

            self._session_tasks[session_id] = {
                "queue": queue,
                "task": asyncio.ensure_future(run_media_session(send_bytes, recv_chunk, self._on_session)),
            }

        self._session_tasks[session_id]["queue"].put_nowait(data)


async def _raw_quic_stream_handler(on_session, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    async def recv_chunk():
        return await reader.read(65536)

    try:
        await run_media_session(writer.write, recv_chunk, on_session)
    finally:
        writer.close()


class WebTransportServer:
    """Runs a single QUIC listener that accepts both WebTransport (h3) and raw QUIC sessions."""

    def __init__(self, host: str, port: int, on_session, certificate=None, private_key=None):
        self.host = host
        self.port = port
        self._on_session = on_session
        self.certificate_hash_b64 = None

        self.configuration = QuicConfiguration(
            is_client=False,
            alpn_protocols=[*H3_ALPN, RAW_QUIC_ALPN],
            # Required for H3Connection(enable_webtransport=True): it advertises H3_DATAGRAM support,
            # which needs this transport parameter negotiated even though we only use streams.
            max_datagram_frame_size=65536,
        )
        if certificate is not None and private_key is not None:
            self.configuration.certificate = certificate
            self.configuration.private_key = private_key
        else:
            cert, key = generate_self_signed_cert(hostname=host if host not in ("0.0.0.0", "") else "localhost")
            self.configuration.certificate = cert
            self.configuration.private_key = key
            self.certificate_hash_b64 = certificate_hash_base64(cert)
            log_info(
                "WebTransport server using an ephemeral self-signed certificate "
                "(pass --wt-cert-file/--wt-key-file to use a real one)"
            )

        self._server = None

    async def bind(self):
        """Bind the UDP socket and start accepting QUIC connections. Returns once ready."""

        def create_protocol(*args, **kwargs):
            return QuicMediaProtocol(*args, on_session=self._on_session, **kwargs)

        self._server = await serve(
            self.host,
            self.port,
            configuration=self.configuration,
            create_protocol=create_protocol,
            stream_handler=lambda reader, writer: asyncio.ensure_future(
                _raw_quic_stream_handler(self._on_session, reader, writer)
            ),
        )
        log_info(f"WebTransport/QUIC server listening on {self.host}:{self.port} (UDP)")

    async def start(self):
        """Bind and run until closed. Suitable for asyncio.gather() alongside other servers."""
        if self._server is None:
            await self.bind()
        await asyncio.Event().wait()

    def close(self):
        if self._server:
            self._server.close()

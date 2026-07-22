"""WebSocket server exposing live-proxy connections using the same framing as WebTransport/raw QUIC.

Reuses the binary frame protocol and session bootstrap from interfaces/webtransport/
(see protocol.py and server.py:run_media_session) so a client can drive a connection over
a plain WebSocket instead of WebTransport, raw QUIC, RTMP or WebRTC.
"""

import asyncio
import ssl
from typing import Optional

from websockets.asyncio.server import Server, ServerConnection, serve
from websockets.exceptions import ConnectionClosed

from interfaces.webtransport.server import run_media_session
from logger import log_info


async def _handler(websocket: ServerConnection, on_session) -> None:
    async def recv_chunk() -> Optional[bytes]:
        try:
            return await websocket.recv(decode=False)
        except ConnectionClosed:
            return None

    def send_bytes(payload: bytes) -> None:
        asyncio.ensure_future(websocket.send(payload))

    await run_media_session(send_bytes, recv_chunk, on_session)


class WebSocketServer:
    """Runs a WebSocket listener that accepts live-proxy sessions framed like WebTransport/raw QUIC."""

    def __init__(self, host: str, port: int, on_session, ssl_context: Optional[ssl.SSLContext] = None):
        self.host = host
        self.port = port
        self._on_session = on_session
        self._ssl_context = ssl_context
        self._server: Optional[Server] = None

    @property
    def secure(self) -> bool:
        return self._ssl_context is not None

    async def bind(self) -> None:
        """Bind the TCP socket and start accepting WebSocket connections. Returns once ready."""
        self._server = await serve(
            lambda websocket: _handler(websocket, self._on_session),
            self.host,
            self.port,
            ssl=self._ssl_context,
            max_size=None,
        )
        log_info(f"WebSocket server listening on {self.host}:{self.port} (TCP)")

    async def start(self) -> None:
        """Bind and run until closed. Suitable for asyncio.gather() alongside other servers."""
        if self._server is None:
            await self.bind()
        await self._server.wait_closed()

    def close(self) -> None:
        if self._server:
            self._server.close()

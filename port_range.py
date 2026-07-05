"""UDP port range restriction helpers for the WebRTC (aiortc/ICE) and SIP (RTP) transports.

Each transport reads its own min/max env vars. If a pair is not set (or invalid),
the transport falls back to its current behavior of letting the OS pick an
ephemeral port.
"""

import asyncio
import contextlib
import os
import random
import socket
from typing import Optional

from logger import log_warn


def _read_port_range(min_var: str, max_var: str) -> Optional[tuple[int, int]]:
    min_raw, max_raw = os.environ.get(min_var), os.environ.get(max_var)
    if not min_raw or not max_raw:
        return None
    try:
        min_port, max_port = int(min_raw), int(max_raw)
    except ValueError:
        log_warn(f"Ignoring {min_var}/{max_var}: not valid integers")
        return None
    if not (0 < min_port <= max_port <= 65535):
        log_warn(f"Ignoring {min_var}/{max_var}: invalid range {min_port}-{max_port}")
        return None
    return min_port, max_port


def get_webrtc_port_range() -> Optional[tuple[int, int]]:
    return _read_port_range("WEBRTC_MIN_PORT", "WEBRTC_MAX_PORT")


def get_sip_port_range() -> Optional[tuple[int, int]]:
    return _read_port_range("SIP_MIN_PORT", "SIP_MAX_PORT")


def bind_udp_socket_in_range(host: str, port_range: tuple[int, int]) -> socket.socket:
    """Bind a UDP socket to a random free port within the given inclusive range."""
    min_port, max_port = port_range
    ports = list(range(min_port, max_port + 1))
    random.shuffle(ports)
    last_error: Optional[OSError] = None
    for port in ports:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.bind((host, port))
            return sock
        except OSError as e:
            sock.close()
            last_error = e
    raise OSError(f"No available UDP port in range {min_port}-{max_port} on {host}") from last_error


@contextlib.asynccontextmanager
async def restrict_ice_gathering_port_range(port_range: Optional[tuple[int, int]]):
    """Temporarily constrain aioice's UDP candidate sockets to a port range.

    aioice binds host candidates via `loop.create_datagram_endpoint(..., local_addr=(host, 0))`.
    We monkeypatch the event loop for the duration of ICE gathering to instead bind a
    pre-selected socket within the configured range. No-op if `port_range` is None.
    """
    if port_range is None:
        yield
        return

    loop = asyncio.get_event_loop()
    original_create_datagram_endpoint = loop.create_datagram_endpoint

    async def patched_create_datagram_endpoint(protocol_factory, local_addr=None, **kwargs):
        if local_addr is not None and "sock" not in kwargs:
            sock = bind_udp_socket_in_range(local_addr[0], port_range)
            sock.setblocking(False)
            return await original_create_datagram_endpoint(protocol_factory, sock=sock, **kwargs)
        return await original_create_datagram_endpoint(protocol_factory, local_addr=local_addr, **kwargs)

    loop.create_datagram_endpoint = patched_create_datagram_endpoint
    try:
        yield
    finally:
        loop.create_datagram_endpoint = original_create_datagram_endpoint

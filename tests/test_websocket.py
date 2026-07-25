"""Tests for the WebSocket transport."""

import asyncio
import json
import socket

import pytest
import websockets

from interfaces.websocket.server import WebSocketServer
from interfaces.webtransport.protocol import pack_audio, pack_control


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class TestWebSocketIntegration:
    @pytest.mark.asyncio
    async def test_session_bootstrap_and_audio_frame(self):
        port = _free_tcp_port()
        sessions = []

        async def on_session(pc, params):
            sessions.append((pc, params))

        server = WebSocketServer(host="127.0.0.1", port=port, on_session=on_session)
        await server.bind()
        try:
            async with websockets.connect(f"ws://127.0.0.1:{port}/") as client:
                await client.send(pack_control(json.dumps({"model": "", "metadata": {"test": True}}).encode("utf-8")))
                await client.send(pack_audio(b"\x00\x01" * 160, sample_rate=16000, channels=1))
                await asyncio.sleep(0.2)

                assert len(sessions) == 1
                pc, params = sessions[0]
                assert params == {"model": "", "metadata": {"test": True}}
                assert pc.connectionState == "connected"

                await pc.close()
        finally:
            server.close()

    @pytest.mark.asyncio
    async def test_server_sends_frames_back_to_client(self):
        port = _free_tcp_port()
        pcs = []

        async def on_session(pc, params):
            pcs.append(pc)

        server = WebSocketServer(host="127.0.0.1", port=port, on_session=on_session)
        await server.bind()
        try:
            async with websockets.connect(f"ws://127.0.0.1:{port}/") as client:
                await client.send(pack_control(json.dumps({"model": ""}).encode("utf-8")))
                await asyncio.sleep(0.2)

                assert len(pcs) == 1
                pcs[0]._send_control(b'{"hello": true}')

                message = await asyncio.wait_for(client.recv(), timeout=1)
                assert b'{"hello": true}' in message
        finally:
            server.close()

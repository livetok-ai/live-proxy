import asyncio
import json
import os
import struct
import sys
from typing import AsyncIterator

import aiohttp
import numpy as np
from av import VideoFrame

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from logger import log_info
from model import Input, Model, Output


class InsivisionProvider(Model):
    @property
    def supports_audio(self) -> bool:
        return False

    @property
    def supports_video(self) -> bool:
        return True

    def __init__(self, name=None, connection=None, **kwargs):
        super().__init__(name=name, connection=connection, **kwargs)
        self.url = kwargs.get("url") or os.getenv("INSIVISION_URL", "ws://localhost:8766")
        self._ws = None
        self._session = None
        self._frame_queue: asyncio.Queue = asyncio.Queue()
        self._connected = False
        log_info(f"Insivision provider url={self.url}")

    async def connect(self):
        if not self.url:
            raise ValueError("INSIVISION_URL is required")

        log_info(f"Insivision connecting to WebSocket {self.url}")
        try:
            self._session = aiohttp.ClientSession()
            self._ws = await self._session.ws_connect(self.url)
            self._connected = True
            log_info(f"Insivision WebSocket connection established {self.url}")
        except Exception as e:
            log_info(f"Insivision WebSocket connection failed {self.url}: {e}")
            raise

        asyncio.ensure_future(self._recv_loop())

    async def _recv_loop(self):
        try:
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.BINARY:
                    await self._frame_queue.put(msg.data)
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    log_info(f"Insivision WebSocket error: {self._ws.exception()}")
                    break
                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    log_info(f"Insivision WebSocket closed by server: code={self._ws.close_code}")
                    break
        except Exception as e:
            log_info(f"Insivision WebSocket recv error: {e}")
        finally:
            await self._frame_queue.put(None)

    async def send(self, input: Input):
        if not self._connected or not self._ws or self._ws.closed:
            return

        try:
            if isinstance(input, str):
                try:
                    json.loads(input)
                    # if isinstance(msg, dict) and msg.get("type") in ("keydown", "keyup"):
                    #     log_info(f"Insivision key event: type={msg['type']} key={msg.get('key')} code={msg.get('code')}")
                except (json.JSONDecodeError, TypeError):
                    pass
                await self._ws.send_str(input)
            elif isinstance(input, bytes):
                await self._ws.send_str(input.decode("utf-8", errors="replace"))
        except Exception as e:
            log_info(f"Insivision send error: {e}")

    async def recv_video(self) -> AsyncIterator[VideoFrame]:
        while True:
            data = await self._frame_queue.get()
            if data is None:
                break

            try:
                w, h = struct.unpack(">HH", data[:4])
                arr = np.frombuffer(data[4:], dtype=np.uint8).reshape(h, w, 3)
                frame = VideoFrame.from_ndarray(arr, format="rgb24")
                yield frame
            except Exception as e:
                log_info(f"Insivision frame decode error: {e}")

    async def recv(self) -> AsyncIterator[Output]:
        async for frame in self.recv_video():
            yield frame

    async def close(self):
        log_info("Insivision closing WebSocket connection")
        self._connected = False
        if self._ws and not self._ws.closed:
            await self._ws.close()
            log_info("Insivision WebSocket closed")
        if self._session and not self._session.closed:
            await self._session.close()
        self._ws = None
        self._session = None

import asyncio
import os
import time

import aiohttp

from logger import log_info


class Registration:
    """Periodically registers this process with an external API via HTTP POST,
    controlled by the REGISTRATION_URL and REGISTRATION_TIME env vars."""

    DEFAULT_REGISTRATION_TIME = 60

    def __init__(self, host=None, port=None):
        self.host = host
        self.port = port
        self.url = os.getenv("REGISTRATION_URL")
        self.interval = float(os.getenv("REGISTRATION_TIME", self.DEFAULT_REGISTRATION_TIME))

    async def run(self):
        if not self.url:
            log_info("Registration skipped: REGISTRATION_URL not set")
            return

        log_info("Registration started")
        while True:
            await self._register()
            await asyncio.sleep(self.interval)

    @staticmethod
    def _available_models():
        """Model names this instance can currently serve, i.e. every MODEL_MAP
        alias whose provider class reports is_available() (see model.py)."""
        from connection import MODEL_MAP

        return sorted(name for name, model_cls in MODEL_MAP.items() if model_cls.is_available())

    async def _register(self):
        data = {
            "host": self.host,
            "port": self.port,
            "models": self._available_models(),
            "timestamp": int(time.time() * 1000),
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.url,
                    json=data,
                    headers={"Content-Type": "application/json"},
                ) as response:
                    log_info("Registration request sent to %s, status: %d", self.url, response.status)
        except Exception as e:
            log_info("Failed to send registration request to %s: %s", self.url, e)

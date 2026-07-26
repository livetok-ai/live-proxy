import asyncio
import os
import ssl
from typing import Optional

import aiohttp
from PIL import Image

from logger import log_info, log_warn


def default_connection_params() -> dict:
    """Connection parameters from the DEFAULT_MODEL / DEFAULT_PROMPT / DEFAULT_SYSTEM_PROMPT /
    DEFAULT_RECORDING env vars, used to configure SIP/RTMP connections when no callback URL is
    configured."""
    params = {}
    default_model = os.getenv("DEFAULT_MODEL")
    default_prompt = os.getenv("DEFAULT_PROMPT")
    if default_model and default_prompt:
        # Pass the prompt as a model parameter (rather than system_instructions) so
        # providers like cosmos, which read their prompt from their own kwarg, pick it up.
        params["model"] = [{"name": default_model, "parameters": {"prompt": default_prompt}}]
    elif default_model:
        params["model"] = default_model
    if os.getenv("DEFAULT_SYSTEM_PROMPT"):
        params["system_instructions"] = os.getenv("DEFAULT_SYSTEM_PROMPT")
    if os.getenv("DEFAULT_RECORDING") is not None:
        params["recording"] = parse_bool(os.getenv("DEFAULT_RECORDING"))
    return params


async def post_callback(url: str, payload: dict, context: str = "") -> tuple[bool, Optional[dict]]:
    """POST a JSON payload to a callback URL and return (success, response_data).

    Shared by the RTMP and SIP servers to fetch connection parameters from their
    configured callback URL when a client connects."""
    try:
        log_info(f"making callback to {url} with payload: {payload}", context=context)
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        connector = aiohttp.TCPConnector(ssl=ssl_context)

        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=5)) as response:
                log_info(f"callback response: {response.status}", context=context)
                if 200 <= response.status < 300:
                    try:
                        return True, await response.json()
                    except Exception as e:
                        log_warn(f"error parsing callback response: {e}", context=context)
                        return True, {}
                return False, None
    except Exception as e:
        log_warn(f"error making callback to {url}: {e}", context=context)
        return False, None


class VideoBuffer:
    """Utility class to maintain a rolling buffer of PIL images and compose them horizontally."""

    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self.buffer = []

    def add_and_composite(self, image: Image.Image) -> Image.Image:
        """Add an image to the rolling buffer and return a horizontally composited image of the buffer's contents."""
        self.buffer.append(image)
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)

        # Compose horizontally all the images in buffer
        composite = Image.new("RGB", (image.width * len(self.buffer), image.height))
        for i in range(len(self.buffer)):
            composite.paste(self.buffer[i], (image.width * i, 0))

        # Copy any custom properties from original image if present (like pts/time_base)
        if hasattr(image, "pts"):
            composite.pts = image.pts
        if hasattr(image, "time_base"):
            composite.time_base = image.time_base

        return composite


def limit_queue_size(queue: asyncio.Queue, max_size: int = 10) -> None:
    """Discards the oldest items in the queue until the queue size is below max_size."""
    while queue.qsize() >= max_size:
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            break


def parse_bool(value, default: bool = False) -> bool:
    """Parse a boolean value from any type (bool, int, str), falling back to default if None."""
    if value is None:
        return default
    if isinstance(value, (int, str)):
        try:
            return bool(int(value)) if str(value).isdigit() else (str(value).lower() == "true")
        except (ValueError, TypeError):
            return default
    return bool(value)


def parse_int(value, default: int) -> int:
    """Parse an integer value, falling back to a default value if invalid or None."""
    if value is None:
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

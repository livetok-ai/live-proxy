import asyncio
import os

from PIL import Image


def default_connection_params() -> dict:
    """Connection parameters from the DEFAULT_MODEL / DEFAULT_SYSTEM_PROMPT / DEFAULT_RECORDING
    env vars, used to configure SIP/RTMP connections when no callback URL is configured."""
    params = {}
    if os.getenv("DEFAULT_MODEL"):
        params["model"] = os.getenv("DEFAULT_MODEL")
    if os.getenv("DEFAULT_SYSTEM_PROMPT"):
        params["system_instructions"] = os.getenv("DEFAULT_SYSTEM_PROMPT")
    if os.getenv("DEFAULT_RECORDING") is not None:
        params["recording"] = parse_bool(os.getenv("DEFAULT_RECORDING"))
    return params


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

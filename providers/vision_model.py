import asyncio
from abc import abstractmethod
from typing import AsyncIterator, Tuple

from av import VideoFrame
from PIL import ImageDraw
from PIL.Image import Image

from logger import log_info
from model import Input, Model, Output
from utils import limit_queue_size, parse_bool, parse_int

# Stable, curated color palette shared by every box/mask overlay so the same
# label always gets the same color across providers.
DETECTION_COLORS = [
    (255, 75, 75),  # Red
    (75, 123, 255),  # Blue
    (75, 255, 123),  # Green
    (180, 75, 255),  # Purple
    (255, 140, 0),  # Orange
    (0, 206, 209),  # Cyan
    (255, 215, 0),  # Yellow
    (255, 105, 180),  # Pink
    (255, 20, 147),  # Deep Pink
    (0, 250, 154),  # Medium Spring Green
]


def color_for_label(label: str) -> Tuple[int, int, int]:
    """Deterministic, stable color for a given label string."""
    h = 0
    for char in label:
        h = (h * 31 + ord(char)) & 0xFFFFFFFF
    return DETECTION_COLORS[h % len(DETECTION_COLORS)]


def draw_box_with_label(draw: ImageDraw.ImageDraw, coords, text: str, color, width: int = 3):
    """Draw a bounding box outline with a filled label tag above its top-left corner."""
    x1, y1, x2, y2 = coords
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)

    try:
        bbox = draw.textbbox((x1, y1), text)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        tw, th = draw.textsize(text)

    text_bg = [x1, max(0, y1 - th - 6), x1 + tw + 6, y1]
    draw.rectangle(text_bg, fill=color)
    draw.text((x1 + 3, max(0, y1 - th - 4)), text, fill=(255, 255, 255))


def draw_centered_label(draw: ImageDraw.ImageDraw, center, text: str, color, bounds):
    """Draw a filled label tag centered at `center`, clamped within `bounds` (width, height)."""
    center_x, center_y = center
    width, height = bounds

    try:
        bbox = draw.textbbox((center_x, center_y), text)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        tw, th = draw.textsize(text)

    text_x = max(0, min(width - tw - 6, center_x - tw / 2.0))
    text_y = max(0, min(height - th - 6, center_y - th / 2.0))

    text_bg = [text_x, text_y, text_x + tw + 6, text_y + th + 4]
    draw.rectangle(text_bg, fill=color)
    draw.text((text_x + 3, text_y + 1), text, fill=(255, 255, 255))


class VisionModel(Model):
    """Base class for providers that run inference on incoming video frames.

    Every frame received via `send()` is forwarded downstream immediately —
    inference never delays or drops a frame, so the outgoing framerate always
    matches the incoming one, even while inference is only sampling a subset
    of frames or is failing outright. At most one frame is being run through
    inference at a time; a frame arriving while one is already in flight
    replaces any previously queued frame, so the backlog never grows past a
    single pending frame. Overlays always draw the latest available
    inference result without waiting for a new one to arrive.
    """

    DEFAULT_SAMPLING_RATE = 10

    @property
    def supports_video(self) -> bool:
        return True

    @property
    def video_support(self) -> bool:
        return True

    def __init__(self, name=None, connection=None, **kwargs):
        super().__init__(name=name, connection=connection, **kwargs)
        self.draw_detections = parse_bool(kwargs.get("draw"), False)
        self.sampling_rate = parse_int(kwargs.get("sampling"), self.DEFAULT_SAMPLING_RATE)

        self.frame_count = 0
        self.output_queue = asyncio.Queue()
        self._processing = False
        self._pending_frame = None

    @property
    def overlay_enabled(self) -> bool:
        return self.draw_detections

    @property
    def is_ready(self) -> bool:
        """Whether the provider has finished connecting and can process frames."""
        return True

    def get_color(self, label: str):
        return color_for_label(label)

    @abstractmethod
    async def process_frame(self, image: Image):
        """Run inference on `image`. Implementations should update whatever
        state `draw_overlay` reads and emit any detection events."""
        raise NotImplementedError

    def draw_overlay(self, image: Image) -> Image:
        """Return an image with the latest inference result drawn on top.
        Default implementation draws nothing."""
        return image

    def clear_overlay(self):
        """Called when input is disabled; subclasses should reset whatever
        detection state `draw_overlay` reads."""
        pass

    async def send(self, input: Input):
        if not self.is_ready or not isinstance(input, Image):
            return

        self.frame_count += 1
        should_process = (self.frame_count % self.sampling_rate == 1) or (self.sampling_rate <= 1)

        if self.input_enabled and should_process:
            if self._processing:
                # Inference is already in flight — keep only the most recent
                # frame to process next so the backlog never grows.
                self._pending_frame = input
            else:
                self._processing = True
                asyncio.ensure_future(self._run_process_frame(input))
        elif not self.input_enabled:
            self._pending_frame = None
            self.clear_overlay()
            self.reset_detections()

        # Always forward the frame, whether or not it was picked for inference.
        if self.output_enabled:
            drawn_image = self.draw_overlay(input) if self.overlay_enabled else input

            new_frame = VideoFrame.from_image(drawn_image)
            if hasattr(input, "pts"):
                new_frame.pts = input.pts
            if hasattr(input, "time_base"):
                new_frame.time_base = input.time_base

            limit_queue_size(self.output_queue, 10)
            self.output_queue.put_nowait(new_frame)

    async def _run_process_frame(self, image: Image):
        try:
            await self.process_frame(image)
        except Exception as e:
            log_info(f"{type(self).__name__} error processing frame: {e}")
        finally:
            self._processing = False
            if self._pending_frame is not None:
                next_frame = self._pending_frame
                self._pending_frame = None
                self._processing = True
                asyncio.ensure_future(self._run_process_frame(next_frame))

    async def recv(self) -> AsyncIterator[Output]:
        while True:
            try:
                frame = await self.output_queue.get()
                yield frame
            except asyncio.CancelledError:
                break

import asyncio
import time
from abc import abstractmethod
from typing import AsyncIterator, Tuple

from av import VideoFrame
from PIL.Image import Image

from logger import log_info, log_trace
from model import Input, Model, Output
from utils import limit_queue_size, parse_int

# Stable, curated color palette shared by every provider so the same
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


class VisionModel(Model):
    """Base class for providers that run inference on incoming video frames.

    Every frame received via `send()` is forwarded downstream immediately —
    inference never delays or drops a frame, so the outgoing framerate always
    matches the incoming one, even while inference is only sampling a subset
    of frames or is failing outright. At most one frame is being run through
    inference at a time; a frame arriving while one is already in flight
    replaces any previously queued frame, so the backlog never grows past a
    single pending frame.
    """

    DEFAULT_SAMPLING_RATE = 10

    @property
    def supports_video(self) -> bool:
        return True

    def __init__(self, name=None, connection=None, **kwargs):
        super().__init__(name=name, connection=connection, **kwargs)
        self.sampling_rate = parse_int(kwargs.get("sampling"), self.DEFAULT_SAMPLING_RATE)

        self.frame_count = 0
        self.output_queue = asyncio.Queue()
        self._processing = False
        self._pending_frame = None
        self._load_task = None

    @property
    def is_ready(self) -> bool:
        """Whether the provider has finished connecting and can process frames."""
        return True

    async def connect(self):
        """Kick off `load()` in the background instead of awaiting it here.

        Many vision providers load a local model (onto disk and/or GPU),
        which can take seconds. Connection.start()/_prepare() awaits
        connect() for every model before it starts reading input for this
        connection at all — for an RTMP publish, that means the publisher's
        socket isn't being drained while the model loads, so its buffer fills
        up and the stream drops before a single frame is processed. Running
        `load()` in the background lets input start flowing immediately;
        frames are simply dropped by `send()` (via `is_ready`) until loading
        finishes.
        """
        self._load_task = asyncio.ensure_future(self._run_load())

    async def _run_load(self):
        try:
            await self.load()
        except Exception as e:
            log_info(f"{type(self).__name__} failed to load: {e}", context=self._log_context)

    async def load(self):
        """Override to perform the (possibly slow) model/connection setup
        that used to live in `connect()`. Runs in the background — check
        `is_ready` before depending on whatever state this sets up."""
        pass

    async def wait_until_loaded(self):
        """Wait for the background load kicked off by connect() to finish.
        Callers (mainly tests) that need `is_ready` state right after
        connecting should await this instead of assuming connect() blocks."""
        if self._load_task is not None:
            await self._load_task

    def get_color(self, label: str):
        return color_for_label(label)

    @abstractmethod
    async def process_frame(self, image: Image):
        """Run inference on `image`. Implementations should update whatever
        detection state and emit any detection events."""
        raise NotImplementedError

    def clear_overlay(self):
        """Called when input is disabled; subclasses should reset whatever
        detection state they hold."""
        pass

    def notify_objects(self, objects):
        """Emit the standard 'objects' event with normalized bounding box coordinates."""
        self._emit("objects", objects)

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
            new_frame = VideoFrame.from_image(input)
            if hasattr(input, "pts"):
                new_frame.pts = input.pts
            if hasattr(input, "time_base"):
                new_frame.time_base = input.time_base

            limit_queue_size(self.output_queue, 10)
            self.output_queue.put_nowait(new_frame)

    async def _run_process_frame(self, image: Image):
        start = time.monotonic()
        log_trace(
            f"{type(self).__name__} processing frame #{self.frame_count} begin",
            context=self._log_context,
        )
        try:
            await self.process_frame(image)
        except Exception as e:
            log_info(
                f"{type(self).__name__} error processing frame #{self.frame_count}: {e}",
                context=self._log_context,
            )
        else:
            elapsed_ms = (time.monotonic() - start) * 1000
            log_trace(
                f"{type(self).__name__} processed frame #{self.frame_count} in {elapsed_ms:.1f}ms",
                context=self._log_context,
            )
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

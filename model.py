import asyncio
import functools
import time
from abc import abstractmethod
from collections import defaultdict
from typing import Any, AsyncIterator, Callable, Optional, Union

from av import AudioFrame, VideoFrame
from PIL.Image import Image

Input = Union[str, AudioFrame, Image]
Output = Union[AudioFrame, VideoFrame]


class ModelEvents:
    """Constants for model event types."""

    INPUT_TRANSCRIPTION = "input_transcription"
    OUTPUT_TRANSCRIPTION = "output_transcription"
    INTERRUPTED = "interrupted"
    INFERENCE = "inference"


def _frame_kind(value: Any) -> Optional[str]:
    """Classify a send()/recv() payload as one of the three frame kinds tracked
    in ProviderStats, or None if it's not a countable frame (e.g. None)."""
    if isinstance(value, AudioFrame):
        return "audio"
    if isinstance(value, (VideoFrame, Image)):
        return "video"
    if isinstance(value, str):
        return "data"
    return None


class ProviderStats:
    """Per-model frame counters plus an average processing-time metric.

    Frame counts are updated automatically around every send()/recv() call
    (see Model.__init_subclass__), from the provider's own point of view:
    "received" is input handed to the provider via send(), "sent" is output
    yielded from recv(). processing_time is populated by providers that run
    actual inference (see Model.run_shared_inference)."""

    def __init__(self):
        self.audio_frames_received = 0
        self.video_frames_received = 0
        self.data_frames_received = 0
        self.audio_frames_sent = 0
        self.video_frames_sent = 0
        self.data_frames_sent = 0
        self.video_processed_count = 0
        self.audio_processed_count = 0
        self._processing_seconds_total = 0.0
        self._processing_count = 0

    def record_received(self, kind: str):
        setattr(self, f"{kind}_frames_received", getattr(self, f"{kind}_frames_received") + 1)

    def record_sent(self, kind: str):
        setattr(self, f"{kind}_frames_sent", getattr(self, f"{kind}_frames_sent") + 1)

    def record_processing_time(self, seconds: float, kind: str = "video"):
        self._processing_seconds_total += seconds
        self._processing_count += 1
        setattr(self, f"{kind}_processed_count", getattr(self, f"{kind}_processed_count") + 1)

    @property
    def video_processed_avg_time(self) -> Optional[float]:
        if not self._processing_count:
            return None
        return (self._processing_seconds_total / self._processing_count) * 1000

    def as_dict(self) -> dict:
        data = {
            "audio_frames_received": self.audio_frames_received,
            "video_frames_received": self.video_frames_received,
            "data_frames_received": self.data_frames_received,
            "audio_frames_sent": self.audio_frames_sent,
            "video_frames_sent": self.video_frames_sent,
            "data_frames_sent": self.data_frames_sent,
            "video_processed_count": self.video_processed_count,
            "audio_processed_count": self.audio_processed_count,
        }
        avg = self.video_processed_avg_time
        if avg is not None:
            data["video_processed_avg_time"] = round(avg, 2)
        return data

    def __str__(self) -> str:
        return ", ".join(f"{k}={v}" for k, v in self.as_dict().items())


class Model:
    DETECTION_EVENT = "objects_detected"

    # Attributes every Model has that aren't provider-specific configuration.
    _BASE_ATTRS = {"name", "connection", "input_enabled", "output_enabled"}
    # Substrings that mark an attribute as sensitive, so it's never exposed via get_properties().
    _SENSITIVE_ATTR_MARKERS = ("key", "token", "secret", "password")

    def __init_subclass__(cls, **kwargs):
        """Auto-instrument every subclass's own send()/recv() to update
        ProviderStats counters, without requiring each provider to do it.
        Only wraps methods defined directly on `cls` (not inherited ones),
        so a class that doesn't override send/recv keeps its parent's
        already-instrumented version instead of double-counting."""
        super().__init_subclass__(**kwargs)
        if "send" in cls.__dict__:
            cls.send = Model._instrument_send(cls.__dict__["send"])
        if "recv" in cls.__dict__:
            cls.recv = Model._instrument_recv(cls.__dict__["recv"])

    @staticmethod
    def _instrument_send(func):
        @functools.wraps(func)
        async def wrapper(self, input, *args, **kwargs):
            kind = _frame_kind(input)
            if kind:
                self.stats.record_received(kind)
            return await func(self, input, *args, **kwargs)

        return wrapper

    @staticmethod
    def _instrument_recv(func):
        @functools.wraps(func)
        async def wrapper(self, *args, **kwargs):
            async for output in func(self, *args, **kwargs):
                kind = _frame_kind(output)
                if kind:
                    self.stats.record_sent(kind)
                yield output

        return wrapper

    def __init__(self, name=None, connection=None, **kwargs):
        self.name = name
        self.connection = connection
        self._event_handlers = defaultdict(list)
        self.input_enabled = True
        self.output_enabled = True
        self._last_detections = None
        self.stats = ProviderStats()

    async def connect(self):
        pass

    def get_properties(self) -> dict:
        """Return provider-specific configuration for display (e.g. prompt, model version).

        Generically picks up simple scalar attributes set by subclasses, skipping internal
        state and anything that looks like a credential.
        """
        properties = {}
        for key, value in vars(self).items():
            if key.startswith("_") or key in self._BASE_ATTRS:
                continue
            if any(marker in key.lower() for marker in self._SENSITIVE_ATTR_MARKERS):
                continue
            if isinstance(value, (str, int, float, bool)) or value is None:
                properties[key] = value
        return properties

    @property
    def supports_audio(self) -> bool:
        return False

    @property
    def supports_text(self) -> bool:
        return False

    @property
    def supports_video(self) -> bool:
        return False

    @property
    def is_llm(self) -> bool:
        return False

    def enable_input(self):
        self.input_enabled = True

    def disable_input(self):
        self.input_enabled = False

    def enable_output(self):
        self.output_enabled = True

    def disable_output(self):
        self.output_enabled = False

    @property
    def _log_context(self) -> Optional[str]:
        return self.connection.id if self.connection else self.name

    def on(self, event_type: str, handler: Callable[[Any], None]):
        """Register an event handler for the specified event type.

        Args:
            event_type: The type of event (e.g., 'input_transcription', 'output_transcription')
            handler: Callable that will be invoked when the event occurs
        """
        self._event_handlers[event_type].append(handler)

    def off(self, event_type: str, handler: Callable[[Any], None]):
        """Unregister an event handler for the specified event type.

        Args:
            event_type: The type of event (e.g., 'input_transcription', 'output_transcription')
            handler: The handler to remove
        """
        if handler in self._event_handlers[event_type]:
            self._event_handlers[event_type].remove(handler)

    def notify_detections(self, raw, labels):
        """Emit the per-frame "inference" event with the raw result and, when the
        detected labels changed since the last notification (including the first
        one), the provider-specific detection event (see DETECTION_EVENT).

        `labels` can be a set/iterable of labels (emitted as a sorted list) or a
        single label string (emitted as-is)."""
        self._emit(ModelEvents.INFERENCE, raw)

        if labels is None:
            normalized = set()
        elif isinstance(labels, str):
            normalized = {labels}
        else:
            normalized = set(labels)

        if self._last_detections is None or normalized != self._last_detections:
            self._last_detections = normalized
            self._emit(self.DETECTION_EVENT, labels if isinstance(labels, str) else sorted(normalized))

    def reset_detections(self):
        """Forget the last notified labels so the next notification re-emits."""
        self._last_detections = None

    def _emit(self, event_type: str, data: Optional[Any] = None):
        """Emit an event to all registered handlers.

        Args:
            event_type: The type of event being emitted
            data: The event data to pass to handlers
        """
        from logger import log_info, log_trace

        log_trace(f"Event generated: {event_type}", context=self._log_context)

        for handler in self._event_handlers[event_type]:
            try:
                handler(data)
            except Exception as e:
                log_info(f"Error in event handler for {event_type}: {e}")

    @classmethod
    async def setup(cls):
        """Perform any heavy initialization or model downloading needed before use."""
        pass

    @classmethod
    def _get_shared_inference_semaphore(cls) -> asyncio.Semaphore:
        """Per-class semaphore serializing calls to a model instance that's shared
        (via a `_shared_*` class attribute) across every connection's provider
        instance. Without it, two connections could run inference on the same
        model at once, which risks GPU OOM from duplicated activations/KV-cache
        and, for models offloaded across devices, races in the device-placement
        hooks. Stored on `cls.__dict__` directly so each subclass sharing its own
        model gets its own semaphore rather than inheriting one from a sibling."""
        if "_shared_inference_semaphore" not in cls.__dict__:
            cls._shared_inference_semaphore = asyncio.Semaphore(1)
        return cls._shared_inference_semaphore

    async def run_shared_inference(self, func: Callable[[], Any]) -> Any:
        """Run a blocking, zero-arg `func` in the default executor, serialized
        against other connections whose provider instance shares the same
        class-level model. Use for any inference call against a `_shared_*`
        model/processor/reader set up in `setup()`. Times the call into
        `self.stats.video_processed_avg_time` since this is the common choke
        point for actual per-frame/per-window inference work."""
        loop = asyncio.get_event_loop()
        async with type(self)._get_shared_inference_semaphore():
            start = time.monotonic()
            try:
                return await loop.run_in_executor(None, func)
            finally:
                self.stats.record_processing_time(time.monotonic() - start)

    @abstractmethod
    async def send(self, _input: Input):
        pass

    @abstractmethod
    async def recv(self) -> AsyncIterator[Output]:
        pass

    async def send_info(self, info: str):
        """Send arbitrary information to the session."""
        pass

    async def close(self):
        pass

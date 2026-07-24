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


class Model:
    DETECTION_EVENT = "objects_detected"

    def __init__(self, name=None, connection=None, **kwargs):
        self.name = name
        self.connection = connection
        self._event_handlers = defaultdict(list)
        self.input_enabled = True
        self.output_enabled = True
        self._last_detections = None

    async def connect(self):
        pass

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

from abc import abstractmethod
from typing import AsyncIterator, Union, Callable, Any
from collections import defaultdict

from av import AudioFrame
from PIL.Image import Image

Input = Union[str, AudioFrame, Image]
Output = Union[AudioFrame]


class ModelEvents:
    """Constants for model event types."""
    INPUT_TRANSCRIPTION = "input_transcription"
    OUTPUT_TRANSCRIPTION = "output_transcription"


class Model:
    def __init__(self):
        self._event_handlers = defaultdict(list)
    
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
    
    def _emit(self, event_type: str, data: Any):
        """Emit an event to all registered handlers.
        
        Args:
            event_type: The type of event being emitted
            data: The event data to pass to handlers
        """
        for handler in self._event_handlers[event_type]:
            try:
                handler(data)
            except Exception as e:
                # Import here to avoid circular imports
                try:
                    from logger import log_info
                    log_info(f"Error in event handler for {event_type}: {e}")
                except ImportError:
                    print(f"Error in event handler for {event_type}: {e}")

    @abstractmethod
    async def send(self, _input: Input):
        pass

    @abstractmethod
    async def recv(self) -> AsyncIterator[Output]:
        pass

    async def close(self):
        pass

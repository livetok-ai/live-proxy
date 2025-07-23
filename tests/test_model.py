import unittest
from typing import AsyncIterator
from unittest.mock import patch

from model import Input, Model, ModelEvents, Output


class MockModel(Model):
    """Test implementation of Model for testing event functionality."""

    def __init__(self):
        super().__init__()
        self.sent_inputs = []
        self.recv_outputs = []

    async def send(self, input: Input):
        self.sent_inputs.append(input)

    async def recv(self) -> AsyncIterator[Output]:
        for output in self.recv_outputs:
            yield output

    def trigger_event(self, event_type: str, data):
        """Helper method to trigger events for testing."""
        self._emit(event_type, data)


class TestModelEvents(unittest.TestCase):
    """Test cases for the Model event system."""

    def setUp(self):
        self.model = MockModel()
        self.handler_calls = []

    def create_handler(self, name):
        """Create a test handler that records calls."""

        def handler(data):
            self.handler_calls.append((name, data))

        return handler

    def test_event_constants(self):
        """Test that event constants are defined correctly."""
        self.assertEqual(ModelEvents.INPUT_TRANSCRIPTION, "input_transcription")
        self.assertEqual(ModelEvents.OUTPUT_TRANSCRIPTION, "output_transcription")

    def test_on_method_registers_handler(self):
        """Test that on() method registers event handlers."""
        handler = self.create_handler("test_handler")

        self.model.on(ModelEvents.INPUT_TRANSCRIPTION, handler)
        self.model.trigger_event(ModelEvents.INPUT_TRANSCRIPTION, "test_data")

        self.assertEqual(len(self.handler_calls), 1)
        self.assertEqual(self.handler_calls[0], ("test_handler", "test_data"))

    def test_multiple_handlers_same_event(self):
        """Test that multiple handlers can be registered for the same event."""
        handler1 = self.create_handler("handler1")
        handler2 = self.create_handler("handler2")

        self.model.on(ModelEvents.INPUT_TRANSCRIPTION, handler1)
        self.model.on(ModelEvents.INPUT_TRANSCRIPTION, handler2)
        self.model.trigger_event(ModelEvents.INPUT_TRANSCRIPTION, "test_data")

        self.assertEqual(len(self.handler_calls), 2)
        self.assertIn(("handler1", "test_data"), self.handler_calls)
        self.assertIn(("handler2", "test_data"), self.handler_calls)

    def test_different_events_separate_handlers(self):
        """Test that different events trigger separate handlers."""
        handler1 = self.create_handler("input_handler")
        handler2 = self.create_handler("output_handler")

        self.model.on(ModelEvents.INPUT_TRANSCRIPTION, handler1)
        self.model.on(ModelEvents.OUTPUT_TRANSCRIPTION, handler2)

        self.model.trigger_event(ModelEvents.INPUT_TRANSCRIPTION, "input_data")
        self.assertEqual(len(self.handler_calls), 1)
        self.assertEqual(self.handler_calls[0], ("input_handler", "input_data"))

        self.model.trigger_event(ModelEvents.OUTPUT_TRANSCRIPTION, "output_data")
        self.assertEqual(len(self.handler_calls), 2)
        self.assertEqual(self.handler_calls[1], ("output_handler", "output_data"))

    def test_off_method_removes_handler(self):
        """Test that off() method removes event handlers."""
        handler = self.create_handler("test_handler")

        self.model.on(ModelEvents.INPUT_TRANSCRIPTION, handler)
        self.model.trigger_event(ModelEvents.INPUT_TRANSCRIPTION, "test_data")
        self.assertEqual(len(self.handler_calls), 1)

        self.model.off(ModelEvents.INPUT_TRANSCRIPTION, handler)
        self.model.trigger_event(ModelEvents.INPUT_TRANSCRIPTION, "test_data2")
        self.assertEqual(len(self.handler_calls), 1)  # No new calls

    def test_off_method_only_removes_specific_handler(self):
        """Test that off() only removes the specific handler."""
        handler1 = self.create_handler("handler1")
        handler2 = self.create_handler("handler2")

        self.model.on(ModelEvents.INPUT_TRANSCRIPTION, handler1)
        self.model.on(ModelEvents.INPUT_TRANSCRIPTION, handler2)

        self.model.off(ModelEvents.INPUT_TRANSCRIPTION, handler1)
        self.model.trigger_event(ModelEvents.INPUT_TRANSCRIPTION, "test_data")

        self.assertEqual(len(self.handler_calls), 1)
        self.assertEqual(self.handler_calls[0], ("handler2", "test_data"))

    def test_off_method_nonexistent_handler(self):
        """Test that off() handles nonexistent handlers gracefully."""
        handler = self.create_handler("test_handler")

        # Should not raise an exception
        self.model.off(ModelEvents.INPUT_TRANSCRIPTION, handler)
        self.model.trigger_event(ModelEvents.INPUT_TRANSCRIPTION, "test_data")
        self.assertEqual(len(self.handler_calls), 0)

    def test_handler_exception_handling(self):
        """Test that exceptions in handlers are caught and logged."""

        def failing_handler(data):
            raise ValueError("Test exception")

        working_handler = self.create_handler("working_handler")

        self.model.on(ModelEvents.INPUT_TRANSCRIPTION, failing_handler)
        self.model.on(ModelEvents.INPUT_TRANSCRIPTION, working_handler)

        # Should not raise exception and working handler should still be called
        with patch("logger.log_info") as mock_log_info:
            self.model.trigger_event(ModelEvents.INPUT_TRANSCRIPTION, "test_data")

            # Check that error was logged
            mock_log_info.assert_called_once()
            args = mock_log_info.call_args[0][0]
            self.assertIn("Error in event handler", args)
            self.assertIn("input_transcription", args)
            self.assertIn("Test exception", args)

        # Working handler should still have been called
        self.assertEqual(len(self.handler_calls), 1)
        self.assertEqual(self.handler_calls[0], ("working_handler", "test_data"))

    def test_emit_with_no_handlers(self):
        """Test that _emit() works when no handlers are registered."""
        # Should not raise an exception
        self.model.trigger_event(ModelEvents.INPUT_TRANSCRIPTION, "test_data")
        self.assertEqual(len(self.handler_calls), 0)

    def test_custom_event_types(self):
        """Test that custom event types work with the system."""
        handler = self.create_handler("custom_handler")

        self.model.on("custom_event", handler)
        self.model.trigger_event("custom_event", "custom_data")

        self.assertEqual(len(self.handler_calls), 1)
        self.assertEqual(self.handler_calls[0], ("custom_handler", "custom_data"))

    def test_model_inheritance_preserves_events(self):
        """Test that model inheritance preserves event functionality."""

        class DerivedModel(MockModel):
            def __init__(self):
                super().__init__()
                self.custom_data = "derived"

        derived = DerivedModel()
        handler = self.create_handler("derived_handler")

        derived.on(ModelEvents.INPUT_TRANSCRIPTION, handler)
        derived.trigger_event(ModelEvents.INPUT_TRANSCRIPTION, "derived_data")

        self.assertEqual(len(self.handler_calls), 1)
        self.assertEqual(self.handler_calls[0], ("derived_handler", "derived_data"))
        self.assertEqual(derived.custom_data, "derived")


if __name__ == "__main__":
    unittest.main()

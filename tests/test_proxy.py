"""Tests for the proxy module."""

import pytest
import asyncio
from unittest.mock import AsyncMock


class TestSendingTrack:
    """Test cases for SendingTrack class."""

    def test_sending_track_initialization(self):
        """Test SendingTrack initializes correctly."""
        # Import here to avoid dependency issues
        from connection import SendingTrack

        track = SendingTrack()
        assert track.kind == "audio"
        assert track.queue is not None

    @pytest.mark.asyncio
    async def test_sending_track_recv(self):
        """Test SendingTrack recv method."""
        from connection import SendingTrack

        track = SendingTrack()
        test_data = "test_frame"

        # Put data in queue
        await track.queue.put(test_data)

        # Receive data
        result = await track.recv()
        assert result == test_data


class TestConnection:
    """Test cases for Connection class."""

    def test_connection_initialization(self):
        """Test Connection initializes with None values."""
        from connection import Connection

        connection = Connection()
        assert connection.recv_audio_track is None
        assert connection.recv_video_track is None
        assert connection.send_track is None
        assert connection.pc is None
        assert connection.genai_session is None

    @pytest.mark.asyncio
    async def test_connection_close(self):
        """Test Connection close method."""
        from connection import Connection

        connection = Connection()

        # Mock peer connection and session
        mock_pc = AsyncMock()
        mock_session = AsyncMock()

        connection.pc = mock_pc
        connection.genai_session = mock_session

        await connection.close()

        # Verify close was called
        mock_pc.close.assert_called_once()
        mock_session.close.assert_called_once()

        # Verify attributes are reset
        assert connection.pc is None
        assert connection.genai_session is None

    @pytest.mark.asyncio
    async def test_connection_close_with_none_values(self):
        """Test Connection close method when pc and session are None."""
        from connection import Connection

        connection = Connection()

        # Should not raise exception when closing None values
        await connection.close()

        assert connection.pc is None
        assert connection.genai_session is None


def test_basic_import():
    """Test that we can import the logger module without errors."""
    import logger

    # Basic smoke test - logger module should be importable
    assert logger is not None
    assert hasattr(logger, "log_info")


def test_audio_constants():
    """Test that audio constants are defined."""
    import connection

    assert hasattr(connection, "AUDIO_PTIME")
    assert hasattr(connection, "AUDIO_BITRATE")
    assert hasattr(connection, "USE_VIDEO_BUFFER")

    assert connection.AUDIO_PTIME == 0.02
    assert connection.AUDIO_BITRATE == 32000
    assert connection.USE_VIDEO_BUFFER is False


@pytest.mark.asyncio
async def test_queue_operations():
    """Test basic asyncio queue operations."""
    queue = asyncio.Queue()

    # Test put and get
    test_value = "test"
    await queue.put(test_value)
    result = await queue.get()

    assert result == test_value


if __name__ == "__main__":
    pytest.main([__file__])

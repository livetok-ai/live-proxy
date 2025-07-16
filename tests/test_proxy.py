"""Tests for the proxy module."""

import pytest
import asyncio
from unittest.mock import AsyncMock


class TestSendingTrack:
    """Test cases for SendingTrack class."""

    def test_sending_track_initialization(self):
        """Test SendingTrack initializes correctly."""
        # Import here to avoid dependency issues
        from proxy import SendingTrack

        track = SendingTrack()
        assert track.kind == "audio"
        assert track.queue is not None

    @pytest.mark.asyncio
    async def test_sending_track_recv(self):
        """Test SendingTrack recv method."""
        from proxy import SendingTrack

        track = SendingTrack()
        test_data = "test_frame"

        # Put data in queue
        await track.queue.put(test_data)

        # Receive data
        result = await track.recv()
        assert result == test_data


class TestRTCConnection:
    """Test cases for RTCConnection class."""

    def test_rtc_connection_initialization(self):
        """Test RTCConnection initializes with None values."""
        from proxy import RTCConnection

        connection = RTCConnection()
        assert connection.recv_audio_track is None
        assert connection.recv_video_track is None
        assert connection.send_track is None
        assert connection.pc is None
        assert connection.genai_session is None

    @pytest.mark.asyncio
    async def test_rtc_connection_close(self):
        """Test RTCConnection close method."""
        from proxy import RTCConnection

        connection = RTCConnection()

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
    async def test_rtc_connection_close_with_none_values(self):
        """Test RTCConnection close method when pc and session are None."""
        from proxy import RTCConnection

        connection = RTCConnection()

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
    import proxy

    assert hasattr(proxy, "AUDIO_PTIME")
    assert hasattr(proxy, "AUDIO_BITRATE")
    assert hasattr(proxy, "USE_VIDEO_BUFFER")

    assert proxy.AUDIO_PTIME == 0.02
    assert proxy.AUDIO_BITRATE == 32000
    assert proxy.USE_VIDEO_BUFFER is False


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

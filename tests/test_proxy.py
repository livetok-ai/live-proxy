"""Tests for the proxy module."""

import asyncio
from unittest.mock import AsyncMock

import pytest


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
        assert connection.send_audio_track is None
        assert connection.send_video_track is None
        assert connection.pc is None
        assert connection.models == []

    @pytest.mark.asyncio
    async def test_connection_close(self):
        """Test Connection close method."""
        from connection import Connection

        connection = Connection()

        # Mock peer connection and model
        mock_pc = AsyncMock()
        mock_model = AsyncMock()

        connection.pc = mock_pc
        connection.models = [mock_model]

        await connection.close()

        # Verify close was called
        mock_pc.close.assert_called_once()
        mock_model.close.assert_called_once()

        # Verify attributes are reset
        assert connection.pc is None
        assert connection.models == []

    @pytest.mark.asyncio
    async def test_connection_close_with_none_values(self):
        """Test Connection close method when pc and models are empty."""
        from connection import Connection

        connection = Connection()

        # Should not raise exception when closing None values
        await connection.close()

        assert connection.pc is None
        assert connection.models == []

    @pytest.mark.asyncio
    async def test_add_duplicate_model_warn(self):
        """Test that adding a model that already exists logs a warning."""
        from unittest.mock import MagicMock, patch

        from connection import Connection

        conn = Connection()
        mock_model = MagicMock()
        conn.models = [mock_model]

        # Mock the get_model to return the existing model
        with patch.object(conn, "get_model", return_value=mock_model):
            with patch.object(conn, "warn") as mock_warn:
                # Mock MODEL_MAP check so it doesn't fail
                with patch("connection.MODEL_MAP", {"yolo": MagicMock()}):
                    res = await conn.add_model("yolo")
                    assert res == mock_model
                    mock_warn.assert_called_once_with("Model already exists: yolo")

    @pytest.mark.asyncio
    async def test_multiple_video_models_single_sending_track(self):
        """Test that having multiple video models does not create multiple SendingTracks."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from connection import Connection

        conn = Connection()
        mock_pc = MagicMock()
        mock_transceiver = MagicMock()
        mock_transceiver.kind = "video"
        mock_transceiver.direction = "sendrecv"
        mock_pc.getTransceivers.return_value = [mock_transceiver]
        conn.pc = mock_pc

        from providers.sam3.sam3 import Sam3Provider
        from providers.yolo.yolo import YoloProvider

        # Mock the connect methods to do nothing
        with patch.object(YoloProvider, "connect", new_callable=AsyncMock):
            with patch.object(Sam3Provider, "connect", new_callable=AsyncMock):
                # Add first video model
                await conn.add_model("yolo")
                track1 = conn.send_video_track
                assert track1 is not None

                # Add second video model
                await conn.add_model("sam3")
                assert conn.send_video_track is track1

    @pytest.mark.asyncio
    async def test_video_sending_track_created_independent_of_models(self):
        """Test that the video sending track is created if the client has video recv, independently of models."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from connection import Connection

        conn = Connection()
        mock_pc = MagicMock()
        mock_pc.setRemoteDescription = AsyncMock()
        mock_pc.createAnswer = AsyncMock()
        mock_pc.setLocalDescription = AsyncMock()
        mock_pc.localDescription = MagicMock()
        mock_pc.localDescription.sdp = "v=0\r\na=rtpmap:111 opus/48000/2\r\n"

        with patch("connection.RTCPeerConnection", return_value=mock_pc):
            with patch("connection.RTCSessionDescription"):
                # Mock _run so it doesn't start model tasks
                with patch.object(conn, "_run", new_callable=AsyncMock):
                    await conn.start(
                        sdp="v=0\r\no=-\r\ns=-\r\nt=0 0\r\na=fingerprint:sha-256 XX\r\nm=video 9 UDP/TLS/RTP/SAVPF",
                        model="text_sentiment"
                    )

        assert conn.send_video_track is not None
        mock_pc.addTrack.assert_called_once_with(conn.send_video_track)


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

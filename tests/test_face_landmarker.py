import fractions
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from providers.face_landmarker import FaceLandmarkerProvider


@pytest.mark.asyncio
async def test_face_landmarker_provider_init():
    """Test FaceLandmarkerProvider initialization and default attributes."""
    provider = FaceLandmarkerProvider()
    assert provider.detector is None
    assert provider.last_emotion is None


@pytest.mark.asyncio
async def test_face_landmarker_provider_connect():
    """Test FaceLandmarkerProvider connect and model loading (mocked)."""
    provider = FaceLandmarkerProvider(name="face_landmarker")

    with patch("os.path.exists", return_value=True), patch(
        "providers.face_landmarker.face_landmarker.FaceLandmarkerProvider._load_detector"
    ) as mock_load:
        mock_load.return_value = MagicMock()
        await provider.connect()
        assert provider.detector is not None
        mock_load.assert_called_once()

    await provider.close()


@pytest.mark.asyncio
async def test_face_landmarker_provider_send_frame():
    """Test FaceLandmarkerProvider processing frames and detecting emotion."""
    provider = FaceLandmarkerProvider()

    # Setup mock detector
    provider.detector = MagicMock()

    # Mock _process_frame to return a mock result
    with patch("providers.face_landmarker.face_landmarker.FaceLandmarkerProvider._process_frame") as mock_process:
        mock_process.return_value = ("happy", 0.95, [10, 10, 90, 90])

        # Create a 100x100 dummy black image
        img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))

        await provider.send(img)
        assert provider.last_emotion == "happy"

    await provider.close()


class DummyTransceiver:
    def __init__(self, kind, direction, currentDirection):
        self.kind = kind
        self.direction = direction
        self.currentDirection = currentDirection


class DummyPeerConnection:
    def __init__(self, transceivers):
        self.transceivers = transceivers

    def getTransceivers(self):
        return self.transceivers


class DummyConnection:
    def __init__(self, transceivers):
        self.pc = DummyPeerConnection(transceivers)


@pytest.mark.asyncio
async def test_face_landmarker_provider_draw_detections():
    """Test FaceLandmarkerProvider draws detections and yields VideoFrames when enabled."""
    transceiver = DummyTransceiver("video", "sendrecv", "sendrecv")
    conn = DummyConnection([transceiver])

    provider = FaceLandmarkerProvider(name="face_landmarker", connection=conn, draw=True)

    with patch("os.path.exists", return_value=True), patch(
        "providers.face_landmarker.face_landmarker.FaceLandmarkerProvider._load_detector"
    ) as mock_load:
        mock_load.return_value = MagicMock()
        await provider.connect()
        assert provider.overlay_enabled is True

    # Send a dummy frame with a mocked result
    img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
    img.pts = 12345
    img.time_base = fractions.Fraction(1, 30)

    with patch("providers.face_landmarker.face_landmarker.FaceLandmarkerProvider._process_frame") as mock_process:
        mock_process.return_value = ("happy", 0.95, [10, 10, 90, 90])
        await provider.send(img)

    # Receive the frame and check it is returned
    received_frames = []
    async for frame in provider.recv():
        received_frames.append(frame)
        break

    assert len(received_frames) == 1
    assert received_frames[0].pts == 12345
    await provider.close()


@pytest.mark.asyncio
async def test_face_landmarker_provider_sampling():
    """Test FaceLandmarkerProvider frame sampling rate."""
    transceiver = DummyTransceiver("video", "sendrecv", "sendrecv")
    conn = DummyConnection([transceiver])

    provider = FaceLandmarkerProvider(name="face_landmarker", connection=conn, draw=True, sampling=5)

    with patch("os.path.exists", return_value=True), patch(
        "providers.face_landmarker.face_landmarker.FaceLandmarkerProvider._load_detector"
    ) as mock_load:
        mock_load.return_value = MagicMock()
        await provider.connect()
        assert provider.overlay_enabled is True
        assert provider.sampling_rate == 5

    # Mock the actual _process_frame to count calls
    call_count = 0

    def dummy_process(input_img, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        return ("neutral", 0.8, [10, 10, 90, 90])

    provider._process_frame = dummy_process

    # Send 5 dummy frames
    img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
    for _ in range(5):
        await provider.send(img)

    # Process frame should only be called once (the first frame)
    assert call_count == 1
    assert provider.frame_count == 5

    # Send 6th frame
    await provider.send(img)
    # Process frame should be called again (the 6th frame)
    assert call_count == 2
    assert provider.frame_count == 6

    await provider.close()


@pytest.mark.asyncio
async def test_face_landmarker_provider_queue_limit():
    """Test FaceLandmarkerProvider output queue size is capped at 10 frames."""
    provider = FaceLandmarkerProvider(name="face_landmarker")

    # Mock detector
    provider.detector = MagicMock()

    # Enable output to queue frames
    provider.output_enabled = True

    # Send 12 dummy frames with a mocked _process_frame to avoid mediapipe dependency
    with patch("providers.face_landmarker.face_landmarker.FaceLandmarkerProvider._process_frame") as mock_process:
        mock_process.return_value = ("happy", 0.95, [10, 10, 90, 90])
        img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
        for i in range(12):
            img.pts = i
            await provider.send(img)

    # Queue size should be exactly 10
    assert provider.output_queue.qsize() == 10

    # The first two frames should have been discarded.
    first_frame = await provider.output_queue.get()
    assert first_frame.pts == 2

    await provider.close()

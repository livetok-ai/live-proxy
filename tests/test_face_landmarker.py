import asyncio
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
    assert provider.last_detected_Boxes == []


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

    # Mock _detect_emotion to return a mock result
    with patch("providers.face_landmarker.face_landmarker.FaceLandmarkerProvider._detect_emotion") as mock_detect:
        mock_detect.return_value = ("happy", 0.95, [10, 10, 90, 90])

        emitted = []
        provider.on("emotions_detected", lambda data: emitted.append(data))

        # Create a 100x100 dummy black image
        img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))

        await provider.send(img)
        await asyncio.sleep(0.05)  # inference runs in the background

        assert emitted == ["happy"]

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
async def test_face_landmarker_provider_forwards_frames():
    """Test FaceLandmarkerProvider yields VideoFrames for every frame sent."""
    transceiver = DummyTransceiver("video", "sendrecv", "sendrecv")
    conn = DummyConnection([transceiver])

    provider = FaceLandmarkerProvider(name="face_landmarker", connection=conn)

    with patch("os.path.exists", return_value=True), patch(
        "providers.face_landmarker.face_landmarker.FaceLandmarkerProvider._load_detector"
    ) as mock_load:
        mock_load.return_value = MagicMock()
        await provider.connect()

    # Send a dummy frame with a mocked result
    img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
    img.pts = 12345
    img.time_base = fractions.Fraction(1, 30)

    with patch("providers.face_landmarker.face_landmarker.FaceLandmarkerProvider._detect_emotion") as mock_detect:
        mock_detect.return_value = ("happy", 0.95, [10, 10, 90, 90])
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
async def test_face_landmarker_provider_queue_limit():
    """Test FaceLandmarkerProvider output queue size is capped at 10 frames."""
    provider = FaceLandmarkerProvider(name="face_landmarker")

    # Mock detector
    provider.detector = MagicMock()

    # Enable output to queue frames
    provider.output_enabled = True

    # Send 12 dummy frames with a mocked _detect_emotion to avoid mediapipe dependency
    with patch("providers.face_landmarker.face_landmarker.FaceLandmarkerProvider._detect_emotion") as mock_detect:
        mock_detect.return_value = ("happy", 0.95, [10, 10, 90, 90])
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

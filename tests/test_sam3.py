import asyncio
import fractions

import numpy as np
import pytest
from PIL import Image

from providers.sam3 import SamProvider


@pytest.mark.asyncio
async def test_sam3_provider_init():
    """Test SamProvider initialization and default attributes."""
    provider = SamProvider()
    assert provider.model is None
    assert provider.model_version == "sam2.1_t.pt"


@pytest.mark.asyncio
async def test_sam3_provider_connect():
    """Test SamProvider connect and model loading."""
    provider = SamProvider(name="sam", connection=None, version="sam2.1_t.pt")
    # Mocking the actual SAM model load to keep unit tests fast and offline-friendly
    called = False

    async def mock_setup(model_version):
        nonlocal called
        called = True
        SamProvider._shared_model = "mock_sam_model"

    original_setup = SamProvider.setup
    SamProvider.setup = mock_setup

    try:
        await provider.connect()
        assert called is True
        assert provider.model == "mock_sam_model"
        assert provider.model_version == "sam2.1_t.pt"
    finally:
        SamProvider.setup = original_setup
        SamProvider._shared_model = None
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
async def test_sam3_provider_draw_detections():
    """Test Sam3Provider draws detections and yields VideoFrames when enabled."""
    transceiver = DummyTransceiver("video", "sendrecv", "sendrecv")
    conn = DummyConnection([transceiver])

    provider = SamProvider(name="sam", connection=conn, draw=True)

    # Mock setup
    async def mock_setup(model_version):
        SamProvider._shared_model = "mock_sam_model"

    original_setup = SamProvider.setup
    SamProvider.setup = mock_setup

    try:
        await provider.connect()
        assert provider.overlay_enabled is True

        # Send a dummy frame
        img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
        img.pts = 12345
        img.time_base = fractions.Fraction(1, 30)

        # Mock the predict call on the model
        class MockMask:
            def __init__(self):
                # Contour points for an object
                self.xy = [np.array([[10, 10], [10, 20], [20, 20], [20, 10]])]

        class MockResult:
            def __init__(self):
                self.masks = MockMask()

        def dummy_model(input_img, *args, **kwargs):
            return [MockResult()]

        provider.model = dummy_model

        await provider.send(img)

        # Receive the frame and check it is returned
        received_frames = []
        async for frame in provider.recv():
            received_frames.append(frame)
            break

        assert len(received_frames) == 1
        assert received_frames[0].pts == 12345
    finally:
        SamProvider.setup = original_setup
        SamProvider._shared_model = None
        await provider.close()


@pytest.mark.asyncio
async def test_sam3_provider_sampling():
    """Test Sam3Provider frame sampling rate."""
    transceiver = DummyTransceiver("video", "sendrecv", "sendrecv")
    conn = DummyConnection([transceiver])

    provider = SamProvider(name="sam", connection=conn, draw=True, sampling=5)

    async def mock_setup(model_version):
        SamProvider._shared_model = "mock_sam_model"

    original_setup = SamProvider.setup
    SamProvider.setup = mock_setup

    try:
        await provider.connect()
        assert provider.overlay_enabled is True
        assert provider.sampling_rate == 5

        # Mock the actual SAM model inference to count calls
        call_count = 0

        def dummy_model(input_img, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            return []

        provider.model = dummy_model

        # Send 5 dummy frames
        img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
        for _ in range(5):
            await provider.send(img)

        # Inference now runs in the background; give the scheduled task a chance to finish.
        await asyncio.sleep(0.05)

        # SAM model should only be called once (the first frame)
        assert call_count == 1
        assert provider.frame_count == 5

        # Send 6th frame
        await provider.send(img)
        await asyncio.sleep(0.05)
        # SAM model should be called again (the 6th frame)
        assert call_count == 2
        assert provider.frame_count == 6
    finally:
        SamProvider.setup = original_setup
        SamProvider._shared_model = None
        await provider.close()


@pytest.mark.asyncio
async def test_sam3_provider_queue_limit():
    """Test SamProvider output queue size is capped at 10 frames."""
    provider = SamProvider(name="sam")

    async def mock_setup(model_version):
        SamProvider._shared_model = "mock_sam_model"

    original_setup = SamProvider.setup
    SamProvider.setup = mock_setup

    try:
        await provider.connect()

        # Enable output to queue frames
        provider.output_enabled = True

        # Mock model predict
        def dummy_model(input_img, *args, **kwargs):
            return []

        provider.model = dummy_model

        # Send 12 dummy frames
        img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
        for i in range(12):
            img.pts = i
            await provider.send(img)

        # Queue size should be exactly 10
        assert provider.output_queue.qsize() == 10

        # The first two frames should have been discarded.
        first_frame = await provider.output_queue.get()
        assert first_frame.pts == 2
    finally:
        SamProvider.setup = original_setup
        SamProvider._shared_model = None
        await provider.close()

import fractions

import numpy as np
import pytest
from PIL import Image

from providers.sam3.sam3 import Sam3Provider


@pytest.mark.asyncio
async def test_sam3_provider_init():
    """Test Sam3Provider initialization and default attributes."""
    provider = Sam3Provider()
    assert provider.model is None
    assert provider.model_version == "sam2.1_t.pt"


@pytest.mark.asyncio
async def test_sam3_provider_connect():
    """Test Sam3Provider connect and model loading."""
    provider = Sam3Provider()
    # Mocking the actual SAM model load to keep unit tests fast and offline-friendly
    called = False

    async def mock_setup(model_version):
        nonlocal called
        called = True
        Sam3Provider._shared_model = "mock_sam_model"

    original_setup = Sam3Provider.setup
    Sam3Provider.setup = mock_setup

    try:
        await provider.connect(
            model="sam3;version=sam2.1_t.pt", system_instructions=None, tools=None, tool_callback=None, voice=None, language=None, api_key=None
        )
        assert called is True
        assert provider.model == "mock_sam_model"
        assert provider.model_version == "sam2.1_t.pt"
    finally:
        Sam3Provider.setup = original_setup
        Sam3Provider._shared_model = None
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

    provider = Sam3Provider(draw_detections=True)

    # Mock setup
    async def mock_setup(model_version):
        Sam3Provider._shared_model = "mock_sam_model"
    original_setup = Sam3Provider.setup
    Sam3Provider.setup = mock_setup

    try:
        await provider.connect(model="sam3", connection=conn)
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
        Sam3Provider.setup = original_setup
        Sam3Provider._shared_model = None
        await provider.close()


@pytest.mark.asyncio
async def test_sam3_provider_sampling():
    """Test Sam3Provider frame sampling rate."""
    transceiver = DummyTransceiver("video", "sendrecv", "sendrecv")
    conn = DummyConnection([transceiver])

    provider = Sam3Provider(draw_detections=True, sampling_rate=5)

    async def mock_setup(model_version):
        Sam3Provider._shared_model = "mock_sam_model"
    original_setup = Sam3Provider.setup
    Sam3Provider.setup = mock_setup

    try:
        await provider.connect(model="sam3", connection=conn)
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

        # SAM model should only be called once (the first frame)
        assert call_count == 1
        assert provider.frame_count == 5

        # Send 6th frame
        await provider.send(img)
        # SAM model should be called again (the 6th frame)
        assert call_count == 2
        assert provider.frame_count == 6
    finally:
        Sam3Provider.setup = original_setup
        Sam3Provider._shared_model = None
        await provider.close()

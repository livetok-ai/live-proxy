import pytest
from PIL import Image
import numpy as np
import fractions
from providers.yolo.yolo import YoloProvider

@pytest.mark.asyncio
async def test_yolo_provider_init():
    """Test YoloProvider initialization and default attributes."""
    provider = YoloProvider()
    assert provider.model is None

@pytest.mark.asyncio
async def test_yolo_provider_connect():
    """Test YoloProvider connect and model loading."""
    provider = YoloProvider()
    await provider.connect(
        model="yolo",
        system_instructions=None,
        tools=None,
        tool_callback=None,
        voice=None,
        language=None,
        api_key=None
    )
    assert provider.model is not None
    await provider.close()

@pytest.mark.asyncio
async def test_yolo_provider_send_frame():
    """Test YoloProvider processing and logging of dummy frames."""
    provider = YoloProvider()
    await provider.connect(
        model="yolo",
        system_instructions=None,
        tools=None,
        tool_callback=None,
        voice=None,
        language=None,
        api_key=None
    )
    
    # Create a 100x100 dummy black image
    img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
    
    # Sending frame should process successfully without throwing exceptions
    await provider.send(img)
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
async def test_yolo_provider_draw_detections():
    """Test YoloProvider draws detections and yields VideoFrames when enabled."""
    transceiver = DummyTransceiver("video", "sendrecv", "sendrecv")
    conn = DummyConnection([transceiver])

    provider = YoloProvider(draw_detections=True)
    await provider.connect(
        model="yolo",
        connection=conn
    )
    assert provider.overlay_enabled is True

    # Send a dummy frame
    img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
    img.pts = 12345
    img.time_base = fractions.Fraction(1, 30)

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
async def test_yolo_provider_draw_detections_disabled_by_direction():
    """Test YoloProvider overlay is disabled if client does not support receiving."""
    # When client only sends (recvonly on server, so server has no send capability)
    transceiver = DummyTransceiver("video", "recvonly", "recvonly")
    conn = DummyConnection([transceiver])

    provider = YoloProvider(draw_detections=True)
    await provider.connect(
        model="yolo",
        connection=conn
    )
    assert provider.overlay_enabled is False
    await provider.close()


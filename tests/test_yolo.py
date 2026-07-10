import asyncio
import fractions

import numpy as np
import pytest
from PIL import Image

from providers.yolo import YoloProvider


@pytest.mark.asyncio
async def test_yolo_provider_init():
    """Test YoloProvider initialization and default attributes."""
    provider = YoloProvider()
    assert provider.model is None


@pytest.mark.asyncio
async def test_yolo_provider_connect():
    """Test YoloProvider connect and model loading."""
    provider = YoloProvider(name="yolo")
    await provider.connect()
    assert provider.model is not None
    await provider.close()


@pytest.mark.asyncio
async def test_yolo_provider_send_frame():
    """Test YoloProvider processing and logging of dummy frames."""
    provider = YoloProvider(name="yolo")
    await provider.connect()

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

    provider = YoloProvider(name="yolo", connection=conn, draw=True)
    await provider.connect()
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
async def test_yolo_provider_sampling():
    """Test YoloProvider frame sampling rate."""
    transceiver = DummyTransceiver("video", "sendrecv", "sendrecv")
    conn = DummyConnection([transceiver])

    provider = YoloProvider(name="yolo", connection=conn, draw=True, sampling=5)
    await provider.connect()
    assert provider.overlay_enabled is True
    assert provider.sampling_rate == 5

    # Mock the actual YOLO model inference to count calls
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

    # YOLO model should only be called once (the first frame)
    assert call_count == 1
    assert provider.frame_count == 5

    # Send 6th frame
    await provider.send(img)
    await asyncio.sleep(0.05)
    # YOLO model should be called again (the 6th frame)
    assert call_count == 2
    assert provider.frame_count == 6

    await provider.close()


@pytest.mark.asyncio
async def test_yolo_provider_queue_limit():
    """Test YoloProvider output queue size is capped at 10 frames."""
    provider = YoloProvider(name="yolo")
    await provider.connect()

    # Enable output to queue frames
    provider.output_enabled = True

    # Send 12 dummy frames
    img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
    for i in range(12):
        img.pts = i
        await provider.send(img)

    # Queue size should be exactly 10
    assert provider.output_queue.qsize() == 10

    # The first two frames (pts 0 and 1) should have been discarded.
    # The remaining frames in the queue should be pts 2 through 11.
    first_frame = await provider.output_queue.get()
    assert first_frame.pts == 2

    await provider.close()

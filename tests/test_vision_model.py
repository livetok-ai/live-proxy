import asyncio
import fractions

import numpy as np
import pytest
from PIL import Image

from providers.vision_model import VisionModel


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
        self.id = "dummy-connection"


class DummyVisionModel(VisionModel):
    """Minimal concrete VisionModel used to exercise the sampling logic that
    every provider inherits, without depending on any real provider's model
    loading or inference code."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.process_frame_calls = 0

    async def process_frame(self, image):
        self.process_frame_calls += 1


@pytest.mark.asyncio
async def test_vision_model_sampling_rate_from_kwargs():
    provider = DummyVisionModel(name="dummy", sampling=5)
    assert provider.sampling_rate == 5


@pytest.mark.asyncio
async def test_vision_model_sampling_rate_default():
    provider = DummyVisionModel(name="dummy")
    assert provider.sampling_rate == VisionModel.DEFAULT_SAMPLING_RATE


@pytest.mark.asyncio
async def test_vision_model_samples_every_nth_frame():
    """Only every Nth frame should be run through inference in the background."""
    transceiver = DummyTransceiver("video", "sendrecv", "sendrecv")
    conn = DummyConnection([transceiver])

    provider = DummyVisionModel(name="dummy", connection=conn, sampling=5)
    assert provider.sampling_rate == 5

    img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
    for _ in range(5):
        await provider.send(img)

    # Inference runs in the background; give the scheduled task a chance to finish.
    await asyncio.sleep(0.05)

    assert provider.process_frame_calls == 1
    assert provider.frame_count == 5

    # Send 6th frame
    await provider.send(img)
    await asyncio.sleep(0.05)
    assert provider.process_frame_calls == 2
    assert provider.frame_count == 6

    await provider.close()


@pytest.mark.asyncio
async def test_vision_model_forwards_every_frame():
    """Every frame is forwarded downstream, whether or not it was sampled."""
    transceiver = DummyTransceiver("video", "sendrecv", "sendrecv")
    conn = DummyConnection([transceiver])

    provider = DummyVisionModel(name="dummy", connection=conn)
    img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
    img.pts = 12345
    img.time_base = fractions.Fraction(1, 30)

    await provider.send(img)

    received_frames = []
    async for frame in provider.recv():
        received_frames.append(frame)
        break

    assert len(received_frames) == 1
    assert received_frames[0].pts == 12345
    await provider.close()

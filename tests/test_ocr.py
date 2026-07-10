import asyncio
import fractions
import numpy as np
import pytest
from PIL import Image

from providers.ocr import OCRProvider


@pytest.mark.asyncio
async def test_ocr_provider_init():
    """Test OCRProvider initialization and default attributes."""
    provider = OCRProvider()
    assert provider.reader is None
    assert provider.languages == ["en"]
    assert provider.overlay_enabled is False
    assert provider.sampling_rate == 5


@pytest.mark.asyncio
async def test_ocr_provider_custom_languages():
    """Test OCRProvider custom languages parsing."""
    # Test + separator
    provider = OCRProvider(languages="en+es")
    assert provider.languages == ["en", "es"]

    # Test , separator
    provider = OCRProvider(languages="en,fr")
    assert provider.languages == ["en", "fr"]

    # Test | separator
    provider = OCRProvider(languages="en|de")
    assert provider.languages == ["en", "de"]


class MockEasyOCRReader:
    def __init__(self, languages=None, gpu=False):
        self.languages = languages
        self.gpu = gpu

    def readtext(self, image_np, *args, **kwargs):
        # Returns [ (bbox, text, conf) ]
        # bbox points: top-left, top-right, bottom-right, bottom-left
        return [
            ([[10, 30], [60, 30], [60, 40], [10, 40]], "World", 0.90),
            ([[10, 10], [50, 10], [50, 20], [10, 20]], "Hello", 0.95),
        ]


@pytest.mark.asyncio
async def test_ocr_provider_send_frame_and_events():
    """Test OCRProvider processes frames and emits correct sorted text events."""
    provider = OCRProvider(name="ocr", languages="en")

    # Mock the reader directly to avoid heavy setup/downloads in test
    mock_reader = MockEasyOCRReader(["en"], False)
    provider.reader = mock_reader

    emitted_events = {}

    def on_event(event_name):
        def handler(data):
            emitted_events[event_name] = data

        return handler

    provider.on("texts_detected", on_event("texts_detected"))
    provider.on("inference", on_event("inference"))

    # Create dummy black image
    img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))

    await provider.send(img)
    await asyncio.sleep(0.05)  # inference runs in the background

    # Validate that we emitted events with correctly sorted results
    # EasyOCR results should be sorted top-to-bottom: "Hello" (y=10) should come before "World" (y=30)
    assert "texts_detected" in emitted_events
    assert emitted_events["texts_detected"] == ["Hello", "World"]  # unique sorted set

    assert "inference" in emitted_events
    # raw per-detection content, in layout order: top-to-bottom
    assert [item["text"] for item in emitted_events["inference"]] == ["Hello", "World"]

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
async def test_ocr_provider_draw_detections():
    """Test OCRProvider draws boxes and outputs VideoFrames when draw is True."""
    transceiver = DummyTransceiver("video", "sendrecv", "sendrecv")
    conn = DummyConnection([transceiver])

    provider = OCRProvider(name="ocr", connection=conn, draw=True, sampling=1)
    provider.reader = MockEasyOCRReader()

    # Send a dummy frame with pts/time_base
    img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
    img.pts = 9999
    img.time_base = fractions.Fraction(1, 30)

    await provider.send(img)

    # Receive the drawn frame from the output queue
    received_frames = []
    async for frame in provider.recv():
        received_frames.append(frame)
        break

    assert len(received_frames) == 1
    assert received_frames[0].pts == 9999
    await provider.close()


@pytest.mark.asyncio
async def test_ocr_provider_sampling():
    """Test OCRProvider frame sampling rate."""
    provider = OCRProvider(name="ocr", sampling=3)

    call_count = 0

    def dummy_readtext(img_np):
        nonlocal call_count
        call_count += 1
        return []

    mock_reader = MockEasyOCRReader()
    mock_reader.readtext = dummy_readtext
    provider.reader = mock_reader

    img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))

    # Send 3 frames
    await provider.send(img)  # Frame 1: Processes
    await provider.send(img)  # Frame 2: Skips
    await provider.send(img)  # Frame 3: Skips
    await asyncio.sleep(0.05)  # inference runs in the background

    assert call_count == 1
    assert provider.frame_count == 3

    # Send 4th frame
    await provider.send(img)  # Frame 4: Processes (4 % 3 = 1)
    await asyncio.sleep(0.05)
    assert call_count == 2
    assert provider.frame_count == 4

    await provider.close()


@pytest.mark.asyncio
async def test_ocr_provider_queue_limit():
    """Test OCRProvider output queue limits queue size to 10."""
    provider = OCRProvider(name="ocr")
    provider.reader = MockEasyOCRReader()
    provider.output_enabled = True

    # Send 12 dummy frames
    img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
    for i in range(12):
        img.pts = i
        await provider.send(img)

    # Output queue should be capped at 10
    assert provider.output_queue.qsize() == 10

    # The first frame in queue should be pts 2 (pts 0 and 1 discarded)
    first_frame = await provider.output_queue.get()
    assert first_frame.pts == 2

    await provider.close()

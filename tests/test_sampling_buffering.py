from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image

from model import Model
from providers.face_landmarker.face_landmarker import FaceLandmarkerProvider
from providers.gemini.llm import Gemini
from providers.inception.inception import InceptionProvider
from providers.openai.llm import OpenAI
from providers.sam3.sam3 import SamProvider
from providers.simli.visual import Simli
from providers.text_sentiment.text_sentiment import TextSentimentProvider
from providers.yolo.yolo import YoloProvider
from utils import VideoBuffer


def test_base_model_capabilities():
    """Test that the base Model class defaults to False for all capabilities."""
    model = Model()
    assert model.supports_audio is False
    assert model.supports_text is False
    assert model.supports_video is False


def test_provider_capabilities():
    """Test that all specific model providers override capabilities correctly."""
    # Yolo
    yolo = YoloProvider()
    assert yolo.supports_audio is False
    assert yolo.supports_text is False
    assert yolo.supports_video is True

    # SAM
    sam = SamProvider()
    assert sam.supports_audio is False
    assert sam.supports_text is False
    assert sam.supports_video is True

    # FaceLandmarker
    face = FaceLandmarkerProvider()
    assert face.supports_audio is False
    assert face.supports_text is False
    assert face.supports_video is True

    # Inception
    inception = InceptionProvider()
    assert inception.supports_audio is False
    assert inception.supports_text is False
    assert inception.supports_video is True

    # Simli
    simli = Simli()
    assert simli.supports_audio is True
    assert simli.supports_text is False
    assert simli.supports_video is False

    # TextSentiment
    sentiment = TextSentimentProvider()
    assert sentiment.supports_audio is False
    assert sentiment.supports_text is True
    assert sentiment.supports_video is False

    # Gemini
    gemini = Gemini()
    assert gemini.supports_audio is True
    assert gemini.supports_text is True
    assert gemini.supports_video is False

    # OpenAI
    openai = OpenAI()
    assert openai.supports_audio is True
    assert openai.supports_text is True
    assert openai.supports_video is True


def test_video_buffer():
    """Test that VideoBuffer works correctly."""
    buffer = VideoBuffer(max_size=3)
    img1 = Image.new("RGB", (10, 10))
    img2 = Image.new("RGB", (10, 10))
    img3 = Image.new("RGB", (10, 10))
    img4 = Image.new("RGB", (10, 10))

    # Add first image
    res = buffer.add_and_composite(img1)
    assert res.width == 10
    assert len(buffer.buffer) == 1

    # Add second image
    res = buffer.add_and_composite(img2)
    assert res.width == 20
    assert len(buffer.buffer) == 2

    # Add third image
    res = buffer.add_and_composite(img3)
    assert res.width == 30
    assert len(buffer.buffer) == 3

    # Add fourth image (should pop first one and size should remain 3 images, so width = 30)
    res = buffer.add_and_composite(img4)
    assert res.width == 30
    assert len(buffer.buffer) == 3
    assert buffer.buffer == [img2, img3, img4]


@pytest.mark.asyncio
async def test_gemini_sampling_and_buffering():
    """Test that Gemini model correctly performs sampling and uses the video buffer."""
    # Sampling = 3, use_video_buffer = True
    gemini = Gemini(sampling=3, use_video_buffer=True)
    gemini.session = AsyncMock()

    img = Image.new("RGB", (10, 10))

    # Frame 1: count=1, should process (1 % 3 == 1)
    await gemini.send(img)
    assert gemini.frame_count == 1
    assert gemini.session.send_realtime_input.call_count == 1
    # Check that it sent a 1-image composite (width 10)
    sent_img = gemini.session.send_realtime_input.call_args[1]["video"]
    assert sent_img.width == 10

    # Reset mock
    gemini.session.send_realtime_input.reset_mock()

    # Frame 2: count=2, should skip (2 % 3 != 1)
    await gemini.send(img)
    assert gemini.frame_count == 2
    assert gemini.session.send_realtime_input.call_count == 0

    # Frame 3: count=3, should skip (3 % 3 != 1)
    await gemini.send(img)
    assert gemini.frame_count == 3
    assert gemini.session.send_realtime_input.call_count == 0

    # Frame 4: count=4, should process (4 % 3 == 1)
    await gemini.send(img)
    assert gemini.frame_count == 4
    assert gemini.session.send_realtime_input.call_count == 1
    # Since use_video_buffer is True and we've processed 2 frames, the composite width should be 20
    sent_img = gemini.session.send_realtime_input.call_args[1]["video"]
    assert sent_img.width == 20


@pytest.mark.asyncio
async def test_openai_sampling_and_buffering():
    """Test that OpenAI model correctly performs sampling and uses the video buffer."""
    # Sampling = 2, use_video_buffer = False
    openai = OpenAI(sampling=2, use_video_buffer=False)
    openai.session = MagicMock()
    openai.session.conversation = MagicMock()
    openai.session.conversation.item = MagicMock()
    openai.session.conversation.item.create = AsyncMock()

    img = Image.new("RGB", (10, 10))

    # Frame 1: count=1, should process (1 % 2 == 1)
    await openai.send(img)
    assert openai.frame_count == 1
    assert openai.session.conversation.item.create.call_count == 1

    # Reset mock
    openai.session.conversation.item.create.reset_mock()

    # Frame 2: count=2, should skip (2 % 2 != 1)
    await openai.send(img)
    assert openai.frame_count == 2
    assert openai.session.conversation.item.create.call_count == 0

    # Frame 3: count=3, should process (3 % 2 == 1)
    await openai.send(img)
    assert openai.frame_count == 3
    assert openai.session.conversation.item.create.call_count == 1

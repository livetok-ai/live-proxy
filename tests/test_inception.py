import numpy as np
import pytest
import torch
from PIL import Image

from providers.inception import InceptionProvider


@pytest.mark.asyncio
async def test_inception_provider_init():
    """Test InceptionProvider initialization and default attributes."""
    provider = InceptionProvider()
    assert provider.mtcnn is None
    assert provider.resnet is None
    assert provider.sampling_rate == 150
    assert provider.input_enabled is True


@pytest.mark.asyncio
async def test_inception_provider_connect():
    """Test InceptionProvider connect and model loading."""
    provider = InceptionProvider(name="inception")
    await provider.connect()

    assert provider.mtcnn is not None
    assert provider.resnet is not None
    assert provider.device == torch.device("cpu")
    await provider.close()


@pytest.mark.asyncio
async def test_inception_provider_send_frame_no_face():
    """Test InceptionProvider with dummy frame containing no face."""
    provider = InceptionProvider(name="inception")
    await provider.connect()

    # Create a 160x160 dummy black image (contains no face)
    img = Image.fromarray(np.zeros((160, 160, 3), dtype=np.uint8))

    # We track if faces event is fired (it shouldn't be since there's no face)
    emitted = []

    def on_faces(data):
        emitted.append(data)

    provider.on("faces", on_faces)

    # Process frame
    await provider.send(img)

    assert len(emitted) == 0
    await provider.close()


@pytest.mark.asyncio
async def test_inception_provider_send_frame_with_face_mocked():
    """Test InceptionProvider emits face_embeddings event when face is detected (mocked)."""
    provider = InceptionProvider(name="inception")
    await provider.connect()

    # Mock MTCNN and ResNet models so they return deterministic values instantly
    dummy_face_tensor = torch.zeros(3, 160, 160)
    dummy_embedding_tensor = torch.ones(1, 512)

    provider.mtcnn = lambda img: dummy_face_tensor
    provider.resnet = lambda x: dummy_embedding_tensor

    img = Image.fromarray(np.zeros((160, 160, 3), dtype=np.uint8))

    # Track emitted events
    emitted = []

    def on_faces(data):
        emitted.append(data)

    provider.on("faces", on_faces)

    # Process frame
    await provider.send(img)

    assert len(emitted) == 1
    assert "embedding" in emitted[0]
    embedding_list = emitted[0]["embedding"]
    assert len(embedding_list) == 512
    # Verify values are correct (mocked as ones)
    assert all(val == 1.0 for val in embedding_list)

    await provider.close()


@pytest.mark.asyncio
async def test_inception_provider_sampling():
    """Test InceptionProvider frame sampling rate."""
    provider = InceptionProvider(name="inception", sampling=3)
    await provider.connect()

    assert provider.sampling_rate == 3

    # Count how many times _process_frame is executed
    process_calls = 0

    def dummy_process_frame(image):
        nonlocal process_calls
        process_calls += 1
        return np.ones(512, dtype=np.float32)

    provider._process_frame = dummy_process_frame

    img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))

    # Send 4 dummy frames
    for _ in range(4):
        await provider.send(img)

    # With sampling_rate=3, the processed frames should be:
    # Frame 1: processed (1 % 3 == 1)
    # Frame 2: skipped (2 % 3 == 2)
    # Frame 3: skipped (3 % 3 == 0)
    # Frame 4: processed (4 % 3 == 1)
    assert process_calls == 2
    assert provider.frame_count == 4

    await provider.close()

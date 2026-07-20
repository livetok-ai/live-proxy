import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from model import ModelEvents
from providers.cosmos import CosmosProvider
from providers.cosmos.cosmos import (
    DEFAULT_FPS,
    DEFAULT_MODEL,
    DEFAULT_OVERLAP_SECONDS,
    DEFAULT_RESOLUTION,
    DEFAULT_WINDOW_SECONDS,
)


class FakeCompletions:
    def __init__(self, content="<think>reasoning...</think>A person walks by."):
        self.calls = []
        self.content = content

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(content=self.content, reasoning_content=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeClient:
    def __init__(self, completions):
        self.chat = SimpleNamespace(completions=completions)


def make_image(width=64, height=48):
    return Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8))


async def drain_tasks():
    """Let scheduled inference tasks run to completion."""
    for _ in range(5):
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_cosmos_provider_defaults():
    provider = CosmosProvider(name="cosmos")
    assert provider.client is None
    assert not provider.is_ready
    assert provider.model == DEFAULT_MODEL
    assert provider.window_seconds == DEFAULT_WINDOW_SECONDS
    assert provider.overlap_seconds == DEFAULT_OVERLAP_SECONDS
    assert provider.fps == DEFAULT_FPS
    assert provider.resolution == DEFAULT_RESOLUTION
    assert provider.stride_seconds == DEFAULT_WINDOW_SECONDS - DEFAULT_OVERLAP_SECONDS


@pytest.mark.asyncio
async def test_cosmos_provider_settings_overrides():
    provider = CosmosProvider(
        name="cosmos",
        model="nvidia/Cosmos-Reason2-8B",
        endpoint="http://gpu:8000/v1",
        window=10,
        overlap=3,
        fps=2,
        resolution=336,
        max_tokens=256,
        timestamps=False,
    )
    assert provider.model == "nvidia/Cosmos-Reason2-8B"
    assert provider.endpoint == "http://gpu:8000/v1"
    assert provider.window_seconds == 10
    assert provider.overlap_seconds == 3
    assert provider.fps == 2
    assert provider.resolution == 336
    assert provider.max_tokens == 256
    assert provider.add_timestamps is False


@pytest.mark.asyncio
async def test_cosmos_provider_invalid_overlap_clamped():
    provider = CosmosProvider(name="cosmos", window=4, overlap=9)
    assert provider.overlap_seconds < provider.window_seconds


@pytest.mark.asyncio
async def test_cosmos_provider_ignores_frames_when_not_connected():
    provider = CosmosProvider(name="cosmos")
    await provider.send(make_image())
    assert len(provider._frames) == 0


@pytest.mark.asyncio
async def test_cosmos_provider_sliding_window_inference():
    provider = CosmosProvider(name="cosmos", window=2, overlap=1, fps=1, timestamps=False)
    completions = FakeCompletions()
    provider.client = FakeClient(completions)

    inference_events = []
    provider.on(ModelEvents.INFERENCE, lambda raw: inference_events.append(raw))

    clock = {"now": 0.0}
    provider._now = lambda: clock["now"]

    # t=0 and t=1: window (2s) not filled yet, no request.
    await provider.process_frame(make_image())
    clock["now"] = 1.0
    await provider.process_frame(make_image())
    await drain_tasks()
    assert len(completions.calls) == 0

    # A frame arriving faster than 1/fps is not sampled.
    clock["now"] = 1.4
    await provider.process_frame(make_image())
    assert len(provider._frames) == 2

    # t=2: window filled, one request dispatched with all buffered frames.
    clock["now"] = 2.0
    await provider.process_frame(make_image())
    await drain_tasks()
    assert len(completions.calls) == 1

    call = completions.calls[0]
    assert call["model"] == DEFAULT_MODEL
    content = call["messages"][0]["content"]
    # Media before text: three sampled frames followed by the text prompt.
    assert [part["type"] for part in content] == ["image_url", "image_url", "image_url", "text"]
    assert content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")

    # The <think> block is stripped from the emitted answer and overlay text.
    assert len(inference_events) == 1
    assert inference_events[0]["text"] == "A person walks by."
    assert inference_events[0]["frames"] == 3
    assert provider.last_answer == "A person walks by."

    # t=3: stride (window - overlap = 1s) elapsed, next request only keeps
    # frames inside the 2s window (t=0 dropped).
    clock["now"] = 3.0
    await provider.process_frame(make_image())
    await drain_tasks()
    assert len(completions.calls) == 2
    second_content = completions.calls[1]["messages"][0]["content"]
    assert [part["type"] for part in second_content] == ["image_url", "image_url", "image_url", "text"]

    await provider.close()
    assert provider.client is None
    assert provider.last_answer == ""


@pytest.mark.asyncio
async def test_cosmos_provider_single_inflight_request():
    provider = CosmosProvider(name="cosmos", window=1, overlap=0, fps=1, timestamps=False)

    started = asyncio.Event()
    release = asyncio.Event()

    class BlockingCompletions(FakeCompletions):
        async def create(self, **kwargs):
            started.set()
            await release.wait()
            return await super().create(**kwargs)

    completions = BlockingCompletions()
    provider.client = FakeClient(completions)

    clock = {"now": 0.0}
    provider._now = lambda: clock["now"]

    await provider.process_frame(make_image())
    clock["now"] = 1.0
    await provider.process_frame(make_image())
    await started.wait()

    # While a request is in flight, later windows are not dispatched.
    clock["now"] = 2.0
    await provider.process_frame(make_image())
    clock["now"] = 3.0
    await provider.process_frame(make_image())
    assert len(completions.calls) == 0

    release.set()
    await drain_tasks()
    assert len(completions.calls) == 1

    # Once free, the next frame dispatches the freshest window.
    clock["now"] = 4.0
    await provider.process_frame(make_image())
    await drain_tasks()
    assert len(completions.calls) == 2


@pytest.mark.asyncio
async def test_cosmos_provider_overlay():
    provider = CosmosProvider(name="cosmos", draw=True)
    image = make_image(320, 240)

    # Without an answer the frame is passed through untouched.
    assert provider.draw_overlay(image) is image

    provider.last_answer = "A person walks by."
    drawn = provider.draw_overlay(image)
    assert drawn is not image
    assert drawn.size == image.size

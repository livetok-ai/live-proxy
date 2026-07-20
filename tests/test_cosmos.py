import asyncio
import threading

import numpy as np
import pytest
import torch
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


class FakeBatchFeature(dict):
    def __init__(self, input_ids):
        super().__init__(input_ids=input_ids)

    @property
    def input_ids(self):
        return self["input_ids"]

    def to(self, *args, **kwargs):
        return self


class FakeProcessor:
    """Stands in for AutoProcessor: records every chat-template call and
    returns a fixed-length fake token batch."""

    def __init__(self, decoded="<think>reasoning...</think>A person walks by."):
        self.calls = []
        self.decoded = decoded

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        return FakeBatchFeature(input_ids=torch.zeros((1, 4), dtype=torch.long))

    def batch_decode(self, generated_ids_trimmed, **kwargs):
        return [self.decoded]


class FakeModel:
    """Stands in for Cosmos3OmniForConditionalGeneration.generate: returns
    the prompt tokens plus a handful of fake generated tokens."""

    device = "cpu"

    def __init__(self):
        self.generate_calls = []

    def generate(self, input_ids=None, **kwargs):
        self.generate_calls.append(kwargs)
        extra = torch.ones((input_ids.shape[0], 3), dtype=torch.long)
        return torch.cat([input_ids, extra], dim=1)


def make_image(width=64, height=48):
    return Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8))


async def drain_tasks(provider=None, timeout=5.0):
    """Let scheduled inference tasks run to completion.

    Generation now runs in a real thread pool executor, so a fixed number of
    `asyncio.sleep(0)` yields isn't enough to guarantee it has finished; poll
    the inflight flag (with a timeout as a safety net) when a provider is
    given, and fall back to a few plain yields otherwise."""
    if provider is None:
        for _ in range(5):
            await asyncio.sleep(0)
        return
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while provider._inference_inflight and loop.time() < deadline:
        await asyncio.sleep(0.01)


def attach_fake_backend(provider, decoded="<think>reasoning...</think>A person walks by."):
    provider.processor = FakeProcessor(decoded=decoded)
    provider._model_instance = FakeModel()
    return provider.processor, provider._model_instance


@pytest.mark.asyncio
async def test_cosmos_provider_defaults():
    provider = CosmosProvider(name="cosmos")
    assert provider._model_instance is None
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
        window=10,
        overlap=3,
        fps=2,
        resolution=336,
        max_tokens=256,
        timestamps=False,
    )
    assert provider.model == "nvidia/Cosmos-Reason2-8B"
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
    processor, model = attach_fake_backend(provider)

    inference_events = []
    provider.on(ModelEvents.INFERENCE, lambda raw: inference_events.append(raw))

    clock = {"now": 0.0}
    provider._now = lambda: clock["now"]

    # t=0 and t=1: window (2s) not filled yet, no request.
    await provider.process_frame(make_image())
    clock["now"] = 1.0
    await provider.process_frame(make_image())
    await drain_tasks(provider)
    assert len(processor.calls) == 0

    # A frame arriving faster than 1/fps is not sampled.
    clock["now"] = 1.4
    await provider.process_frame(make_image())
    assert len(provider._frames) == 2

    # t=2: window filled, one generation call dispatched with all buffered frames.
    clock["now"] = 2.0
    await provider.process_frame(make_image())
    await drain_tasks(provider)
    assert len(processor.calls) == 1

    call = processor.calls[0]
    content = call["messages"][0]["content"]
    # Media before text: three sampled frames followed by the text prompt.
    assert [part["type"] for part in content] == ["image_url", "image_url", "image_url", "text"]
    assert content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert len(model.generate_calls) == 1

    # The <think> block is stripped from the emitted answer and overlay text,
    # and surfaced separately as "reasoning".
    assert len(inference_events) == 1
    assert inference_events[0]["text"] == "A person walks by."
    assert inference_events[0]["reasoning"] == "reasoning..."
    assert inference_events[0]["frames"] == 3
    assert provider.last_answer == "A person walks by."

    # t=3: stride (window - overlap = 1s) elapsed, next request only keeps
    # frames inside the 2s window (t=0 dropped).
    clock["now"] = 3.0
    await provider.process_frame(make_image())
    await drain_tasks(provider)
    assert len(processor.calls) == 2
    second_content = processor.calls[1]["messages"][0]["content"]
    assert [part["type"] for part in second_content] == ["image_url", "image_url", "image_url", "text"]

    await provider.close()
    assert provider._model_instance is None
    assert provider.last_answer == ""


@pytest.mark.asyncio
async def test_cosmos_provider_single_inflight_request():
    provider = CosmosProvider(name="cosmos", window=1, overlap=0, fps=1, timestamps=False)

    # Real threading primitives: generate() now runs in a worker thread via
    # run_in_executor, not as an awaitable coroutine.
    started = threading.Event()
    release = threading.Event()

    class BlockingModel(FakeModel):
        def generate(self, input_ids=None, **kwargs):
            started.set()
            release.wait()
            return super().generate(input_ids=input_ids, **kwargs)

    processor, _ = attach_fake_backend(provider)
    provider._model_instance = BlockingModel()

    clock = {"now": 0.0}
    provider._now = lambda: clock["now"]

    await provider.process_frame(make_image())
    clock["now"] = 1.0
    await provider.process_frame(make_image())
    await asyncio.get_event_loop().run_in_executor(None, started.wait)

    # While a request is in flight, later windows are not dispatched.
    clock["now"] = 2.0
    await provider.process_frame(make_image())
    clock["now"] = 3.0
    await provider.process_frame(make_image())
    assert len(processor.calls) == 1

    release.set()
    await drain_tasks(provider)
    assert len(provider._model_instance.generate_calls) == 1

    # Once free, the next frame dispatches the freshest window.
    clock["now"] = 4.0
    await provider.process_frame(make_image())
    await drain_tasks(provider)
    assert len(processor.calls) == 2


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

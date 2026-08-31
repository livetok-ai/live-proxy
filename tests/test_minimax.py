import asyncio

import numpy as np
import pytest

from model import ModelEvents
from providers.minimax import MinimaxProvider
from providers.minimax.minimax import DEFAULT_FPS, DEFAULT_MODEL, DEFAULT_NUM_FRAMES, WORKFLOW, snap_num_frames


class FakePipe:
    """Stands in for the real Modular Diffusers pipeline: records every call
    and returns a fixed number of solid-color frames instead of running
    actual generation."""

    def __init__(self, frame_size=(32, 24)):
        self.calls = []
        self.frame_size = frame_size

    def __call__(self, prompt, num_frames, output, **kwargs):
        self.calls.append({"prompt": prompt, "num_frames": num_frames, "output": output, **kwargs})
        w, h = self.frame_size
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        return {"videos": [[frame] * num_frames]}


def attach_fake_backend(provider, **kwargs):
    pipe = FakePipe(**kwargs)
    provider.pipe = pipe
    return pipe


async def drain_output(provider, expected, timeout=5.0):
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    frames = []
    while len(frames) < expected and loop.time() < deadline:
        try:
            frames.append(await asyncio.wait_for(provider.output_queue.get(), timeout=deadline - loop.time()))
        except asyncio.TimeoutError:
            break
    return frames


def test_snap_num_frames():
    assert snap_num_frames(1) == 5
    assert snap_num_frames(5) == 5
    assert snap_num_frames(6) == 22
    assert snap_num_frames(22) == 22
    assert snap_num_frames(23) == 39


@pytest.mark.asyncio
async def test_minimax_provider_defaults():
    provider = MinimaxProvider(name="minimax")
    assert provider.pipe is None
    assert not provider.is_ready
    assert provider.model == DEFAULT_MODEL
    assert provider.num_frames == snap_num_frames(DEFAULT_NUM_FRAMES)
    assert provider.fps == DEFAULT_FPS


@pytest.mark.asyncio
async def test_minimax_provider_settings_overrides():
    provider = MinimaxProvider(name="minimax", model="MiniMaxAI/MiniMax-H3-Max", prompt="a duck", num_frames=6, fps=8)
    assert provider.model == "MiniMaxAI/MiniMax-H3-Max"
    assert provider.prompt == "a duck"
    assert provider.num_frames == 22  # snapped up from 6
    assert provider.fps == 8


@pytest.mark.asyncio
async def test_minimax_provider_prompt_falls_back_to_system_instructions():
    class FakeConnection:
        system_instructions = "a fox in the snow"

    provider = MinimaxProvider(name="minimax", connection=FakeConnection())
    assert provider.prompt == "a fox in the snow"


@pytest.mark.asyncio
async def test_minimax_provider_does_not_consume_input():
    """Text-to-video: send() is a no-op, no video/audio is ever forwarded to it."""
    provider = MinimaxProvider(name="minimax")
    assert not provider.supports_video
    assert not provider.supports_audio
    await provider.send("anything")
    assert provider._generation_task is None


@pytest.mark.asyncio
async def test_minimax_provider_generates_at_least_one_frame():
    provider = MinimaxProvider(name="minimax", num_frames=5, fps=1000)  # fast pacing for the test
    pipe = attach_fake_backend(provider, frame_size=(16, 12))

    inference_events = []
    provider.on(ModelEvents.INFERENCE, lambda raw: inference_events.append(raw))

    # Generation normally kicks off from _run_load() once the model finishes
    # loading; trigger it directly here since the pipe is faked.
    provider._generation_task = asyncio.ensure_future(provider._run_generation())
    await provider._generation_task

    assert len(pipe.calls) == 1
    call = pipe.calls[0]
    assert call["prompt"] == provider.prompt
    assert call["num_frames"] == 5
    assert call["output"] == ["videos"]
    assert "image" not in call

    frames = await drain_output(provider, expected=5)
    assert len(frames) >= 1
    assert frames[0].width == 16
    assert frames[0].height == 12

    assert len(inference_events) == 1
    assert inference_events[0]["frames"] == 5
    assert provider.last_clip_frames == 5

    await provider.close()


@pytest.mark.asyncio
async def test_minimax_provider_recv_yields_generated_frames():
    provider = MinimaxProvider(name="minimax", num_frames=5, fps=1000)
    attach_fake_backend(provider, frame_size=(8, 8))

    provider._generation_task = asyncio.ensure_future(provider._run_generation())
    await provider._generation_task

    received = []

    async def collect():
        async for frame in provider.recv():
            received.append(frame)
            if len(received) >= 5:
                break

    await asyncio.wait_for(collect(), timeout=5.0)
    assert len(received) == 5

    await provider.close()


@pytest.mark.asyncio
async def test_minimax_provider_load_triggers_generation():
    provider = MinimaxProvider(name="minimax", num_frames=5, fps=1000)

    async def fake_load():
        attach_fake_backend(provider, frame_size=(8, 8))

    provider.load = fake_load
    await provider._run_load()

    assert provider._generation_task is not None
    await provider._generation_task
    assert provider.last_clip_frames == 5

    await provider.close()


def test_minimax_registered_in_model_map():
    from connection import MODEL_MAP

    assert MODEL_MAP["minimax"] is MinimaxProvider


def test_minimax_workflow_supports_text_to_video():
    assert WORKFLOW == "fl2va"

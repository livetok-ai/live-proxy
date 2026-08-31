import asyncio
import os
import time
from typing import AsyncIterator, Optional

import numpy as np
import torch
from av import VideoFrame

from logger import log_info, log_trace, log_warn
from model import Input, Model, Output
from utils import parse_int

DEFAULT_MODEL = os.getenv("MINIMAX_MODEL") or "MiniMaxAI/MiniMax-H3"
# First/last-frame-to-video-and-audio workflow: also covers pure text-to-video
# when no conditioning image is passed. See https://huggingface.co/MiniMaxAI/MiniMax-H3.
WORKFLOW = "fl2va"
DEFAULT_PROMPT = (
    os.getenv("MINIMAX_PROMPT")
    or "A short, natural continuation of the scene with gentle camera motion and realistic lighting."
)
# MiniMax-H3 requires num_frames == 17*k + 5; 22 (k=1) is the smallest clip
# beyond the bare-minimum 5-frame stub.
DEFAULT_NUM_FRAMES = parse_int(os.getenv("MINIMAX_NUM_FRAMES"), 22)
DEFAULT_FPS = parse_int(os.getenv("MINIMAX_FPS"), 24)
DEFAULT_SEED = os.getenv("MINIMAX_SEED")
# Without a conditioning image, the pipeline can't derive a canvas size from
# one, so height/width must always be supplied for text-to-video — this is
# its own default 16:9 canvas (both multiples of 32).
DEFAULT_HEIGHT = parse_int(os.getenv("MINIMAX_HEIGHT"), 768)
DEFAULT_WIDTH = parse_int(os.getenv("MINIMAX_WIDTH"), 1344)


def snap_num_frames(n: int) -> int:
    """Round `n` up to the nearest value MiniMax-H3 accepts (17*k + 5, k >= 0)."""
    if n <= 5:
        return 5
    k = -(-(n - 5) // 17)  # ceil division
    return 17 * k + 5


class MinimaxProvider(Model):
    """Provider for MiniMax-H3, run locally through diffusers' Modular
    Diffusers pipeline (huggingface/diffusers#14355).

    This is text-to-video, not vision: it takes no video/audio input at all
    — as soon as the model finishes loading it generates one short clip from
    `prompt` (or the connection's system instructions), and streams the
    resulting frames out through recv() once they're ready.
    """

    DETECTION_EVENT = "video_generated"

    # Model + pipeline are loaded once per process and shared across
    # connections/providers, like CosmosProvider does for its model.
    _shared_pipe = None
    _shared_model_id = None
    _shared_lock = None

    @classmethod
    async def setup(cls, model_id: str = DEFAULT_MODEL):
        if cls._shared_lock is None:
            cls._shared_lock = asyncio.Lock()

        async with cls._shared_lock:
            if cls._shared_pipe is not None and cls._shared_model_id == model_id:
                return

            log_info(f"Loading MiniMax-H3 model: {model_id} (this can take a while)...")
            loop = asyncio.get_event_loop()
            start = time.monotonic()

            def _load():
                from diffusers import ComponentsManager, ModularPipeline

                device = "cuda" if torch.cuda.is_available() else "cpu"
                manager = ComponentsManager()
                manager.enable_auto_cpu_offload(device=device)
                pipe = ModularPipeline.from_pretrained(model_id, workflow=WORKFLOW, components_manager=manager)
                pipe.load_components(dtype=torch.bfloat16)
                return pipe

            cls._shared_pipe = await loop.run_in_executor(None, _load)
            cls._shared_model_id = model_id
            log_info(f"Loaded MiniMax-H3 model: {model_id} in {time.monotonic() - start:.1f}s")

    def __init__(self, name=None, connection=None, **kwargs):
        super().__init__(name=name, connection=connection, **kwargs)
        self.model = kwargs.get("model") or DEFAULT_MODEL
        self.prompt = kwargs.get("prompt") or (connection.system_instructions if connection else None) or DEFAULT_PROMPT
        self.num_frames = snap_num_frames(parse_int(kwargs.get("num_frames"), DEFAULT_NUM_FRAMES))
        self.fps = parse_int(kwargs.get("fps"), DEFAULT_FPS)
        self.height = parse_int(kwargs.get("height"), DEFAULT_HEIGHT)
        self.width = parse_int(kwargs.get("width"), DEFAULT_WIDTH)
        seed = kwargs.get("seed", DEFAULT_SEED)
        self.seed: Optional[int] = int(seed) if seed is not None else None

        self.pipe = None
        self.output_queue: asyncio.Queue = asyncio.Queue()
        self.frame_count = 0
        self.last_clip_frames = 0

        self._generation_task = None
        self._load_task = None

        log_info(
            f"Minimax provider model: {self.model} workflow: {WORKFLOW} "
            f"num_frames: {self.num_frames} fps: {self.fps} prompt: {self.prompt}"
        )

    @property
    def is_ready(self) -> bool:
        return self.pipe is not None

    async def connect(self):
        # Loading a 33B-parameter model can take minutes; run it in the
        # background so connection setup isn't blocked on it.
        self._load_task = asyncio.ensure_future(self._run_load())

    async def _run_load(self):
        try:
            await self.load()
            # Text-to-video needs no input trigger: generate as soon as the
            # model is ready.
            self._generation_task = asyncio.ensure_future(self._run_generation())
        except Exception as e:
            log_warn(f"Minimax provider failed to load: {e}", context=self._log_context)

    async def load(self):
        await MinimaxProvider.setup(self.model)
        self.pipe = MinimaxProvider._shared_pipe

    async def wait_until_loaded(self):
        if self._load_task is not None:
            await self._load_task

    async def send(self, input: Input):
        # Text-to-video only: no video/audio input is consumed.
        return

    def _generate(self):
        """Blocking generation call, run in a thread executor to keep the
        asyncio event loop responsive."""
        log_trace("Minimax _generate begin", context=self._log_context)
        start = time.monotonic()
        try:
            call_kwargs = dict(
                prompt=self.prompt,
                num_frames=self.num_frames,
                # No conditioning image, so height/width can't be derived from
                # one (see MiniMaxH3ResizeStep) — always pass them explicitly.
                height=self.height,
                width=self.width,
                output=["videos"],
            )
            if self.seed is not None:
                call_kwargs["generator"] = torch.Generator().manual_seed(self.seed)

            results = self.pipe(**call_kwargs)
            frames = results["videos"][0]
        except Exception as e:
            elapsed_ms = (time.monotonic() - start) * 1000
            log_trace(f"Minimax _generate end: error {e!r} in {elapsed_ms:.1f}ms", context=self._log_context)
            raise
        elapsed_ms = (time.monotonic() - start) * 1000
        log_trace(f"Minimax _generate end: {len(frames)} frames in {elapsed_ms:.1f}ms", context=self._log_context)
        return frames

    async def _run_generation(self):
        try:
            # Serialized against other connections sharing the same model
            # instance (see Model.run_shared_inference).
            frames = await self.run_shared_inference(self._generate)

            self.last_clip_frames = len(frames)
            self.notify_detections({"frames": len(frames)}, "video_generated")

            for frame in frames:
                self.frame_count += 1
                await self.output_queue.put(self._to_video_frame(frame))
                await asyncio.sleep(1.0 / self.fps)
        except Exception as e:
            log_warn(f"Minimax error generating clip: {e}", context=self._log_context)

    @staticmethod
    def _to_video_frame(frame) -> VideoFrame:
        array = frame if isinstance(frame, np.ndarray) else np.array(frame)
        if array.dtype != np.uint8:
            if array.max() <= 1.0:
                array = (np.clip(array, 0, 1) * 255).astype(np.uint8)
            else:
                array = np.clip(array, 0, 255).astype(np.uint8)
        return VideoFrame.from_ndarray(array, format="rgb24")

    async def recv(self) -> AsyncIterator[Output]:
        while True:
            try:
                frame = await self.output_queue.get()
                yield frame
            except asyncio.CancelledError:
                break

    async def close(self):
        log_info("Closing Minimax provider", context=self._log_context)
        if self._generation_task is not None and not self._generation_task.done():
            self._generation_task.cancel()
        self.pipe = None


async def _main():
    """Standalone smoke test: run with `python -m providers.minimax.minimax`.

    On real GPU hardware (this is meant to run on an H100 server) this loads
    the actual MiniMax-H3 weights and performs genuine text-to-video
    generation. This dev machine has no CUDA GPU, so a real load is
    attempted first and, when it fails (as expected here), falls back to a
    stub pipe that mimics the real pipeline's call signature and output
    shape. That still exercises every bit of the provider's own logic — the
    background generation task, decoding results into VideoFrame objects,
    and the output queue — end to end; only the actual model weights are
    faked.
    """
    # On a machine with no CUDA GPU we can't run the real weights anyway, so
    # avoid hanging on a network call / huge download by forcing offline mode
    # (from_pretrained() then fails fast if nothing is cached). A real GPU
    # box needs actual Hub access, so leave it alone there.
    if not torch.cuda.is_available():
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    provider = MinimaxProvider(num_frames=5)  # smallest valid clip, fastest to verify

    try:
        await provider.load()
        print(f"Loaded real MiniMax-H3 weights for {provider.model}")
    except Exception as e:
        print(f"Real MiniMax-H3 load failed ({e!r}); this machine can't host a 33B-parameter model.")
        print("Falling back to a stub pipe to smoke-test the provider's own orchestration logic.")

        class _StubPipe:
            def __call__(self, prompt, num_frames, output, **kwargs):
                frame = np.zeros((64, 64, 3), dtype=np.uint8)
                return {"videos": [[frame] * num_frames]}

        MinimaxProvider._shared_pipe = _StubPipe()
        MinimaxProvider._shared_model_id = provider.model
        provider.pipe = MinimaxProvider._shared_pipe

    provider._generation_task = asyncio.ensure_future(provider._run_generation())
    await provider._generation_task

    received = []
    while not provider.output_queue.empty():
        received.append(provider.output_queue.get_nowait())

    assert len(received) >= 1, "expected at least one generated frame"
    print(f"Minimax generated {len(received)} frame(s), first frame size: {received[0].width}x{received[0].height}")

    await provider.close()


if __name__ == "__main__":
    asyncio.run(_main())

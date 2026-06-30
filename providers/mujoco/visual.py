import asyncio
import concurrent.futures
import json
import os
import platform
import threading
import time
from typing import AsyncIterator

import mujoco
import numpy as np
from av import VideoFrame

# Set headless GL backend before the renderer is created on Linux.
if "MUJOCO_GL" not in os.environ and platform.system() == "Linux" and not os.environ.get("DISPLAY"):
    os.environ["MUJOCO_GL"] = "egl"

from logger import log_info
from model import Input, Model, Output

# ---------------------------------------------------------------------------
# Simulation constants
# ---------------------------------------------------------------------------

_KEY_DELTA = 0.012  # radians per held-key step
_ZOOM_STEP = 0.2
_ZOOM_MIN = 0.5
_ZOOM_MAX = 10.0
_ZOOM_INIT = 3.0

_KEY_JOINTS = {
    "arrowup":    [(1,  1)],
    "arrowdown":  [(1, -1)],
    "arrowleft":  [(0,  1)],
    "arrowright": [(0, -1)],
    "a":          [(3, -1)],
    "q":          [(3,  1)],
    "w":          [(5, -1)],
    "s":          [(5,  1)],
    "e":          [(7,  1), (8,  1)],
    "d":          [(7, -1), (8, -1)],
}


# ---------------------------------------------------------------------------
# Inline simulation engine (no dependency on mujoco/provider/mujoco.py)
# ---------------------------------------------------------------------------

class _Sim:
    """Thin wrapper around MuJoCo model/data/renderer.  All methods are
    blocking and intended to be called from a thread executor."""

    def __init__(self, scene_xml: str, fps: int, width: int, height: int):
        self.scene_xml = scene_xml
        self.fps = fps
        self.width = width
        self.height = height
        self._model = None
        self._data = None
        self._renderer = None
        self._cam = None
        self._held_keys: set = set()
        self._keys_lock = threading.Lock()
        self._target_qpos = None

    # -- lifecycle --

    def start(self):
        self._model = mujoco.MjModel.from_xml_path(self.scene_xml)
        self._data = mujoco.MjData(self._model)
        self._renderer = mujoco.Renderer(self._model, height=self.height, width=self.width)
        self._cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self._model, self._cam)
        self._cam.distance = _ZOOM_INIT
        self._target_qpos = self._data.qpos.copy()

    def stop(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # -- simulation --

    def step(self):
        self._apply_held_keys()
        controlled = {idx for pairs in _KEY_JOINTS.values() for idx, _ in pairs}
        for i in controlled:
            self._data.qpos[i] = self._target_qpos[i]
            self._data.qvel[i] = 0.0
        mujoco.mj_step(self._model, self._data)

    def render(self) -> np.ndarray:
        self._renderer.update_scene(self._data, self._cam)
        return self._renderer.render()

    # -- events --

    def process_event(self, cmd: dict):
        event = cmd.get("type") or cmd.get("event")
        key = cmd.get("key", "").lower()
        with self._keys_lock:
            if event in ("keypressed", "keydown"):
                self._held_keys.add(key)
                self._apply_one_shot(key)
            elif event == "keyup":
                self._held_keys.discard(key)

    # -- properties --

    @property
    def timestep(self) -> float:
        return self._model.opt.timestep

    @property
    def frame_interval(self) -> float:
        return 1.0 / self.fps

    @property
    def sim_time(self) -> float:
        return self._data.time

    # -- internal --

    def _apply_one_shot(self, key: str):
        if key == "r":
            mujoco.mj_resetData(self._model, self._data)
            self._target_qpos = self._data.qpos.copy()
        elif key in ("+", "="):
            self._cam.distance = max(_ZOOM_MIN, self._cam.distance - _ZOOM_STEP)
        elif key == "-":
            self._cam.distance = min(_ZOOM_MAX, self._cam.distance + _ZOOM_STEP)

    def _apply_held_keys(self):
        with self._keys_lock:
            keys = set(self._held_keys)
        for key in keys:
            for joint_idx, direction in _KEY_JOINTS.get(key, []):
                lo = self._model.jnt_range[joint_idx, 0]
                hi = self._model.jnt_range[joint_idx, 1]
                self._target_qpos[joint_idx] = float(np.clip(
                    self._target_qpos[joint_idx] + direction * _KEY_DELTA, lo, hi,
                ))


# ---------------------------------------------------------------------------
# live-proxy Model
# ---------------------------------------------------------------------------

class MujocoModel(Model):
    """In-process MuJoCo simulation provider for live-proxy.

    Runs physics and rendering in a thread executor so the asyncio event loop
    stays responsive.  Streams VideoFrame objects via recv() and accepts
    key-event JSON strings via send().
    """

    @property
    def supports_audio(self) -> bool:
        return False

    @property
    def video_support(self) -> bool:
        return True

    def __init__(self, name=None, connection=None, **kwargs):
        super().__init__(name=name, connection=connection, **kwargs)
        _default = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "scenes", "franka_emika_panda", "scene_factory.xml"
        ))
        scene_xml = kwargs.get("scene_xml") or os.getenv("MUJOCO_SCENE_XML") or _default
        fps = int(kwargs.get("fps", 30))
        width = int(kwargs.get("width", 1280))
        height = int(kwargs.get("height", 720))

        self._sim = _Sim(scene_xml=scene_xml, fps=fps, width=width, height=height)
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._frame_queue: asyncio.Queue = asyncio.Queue(maxsize=5)
        self._running = False
        self._loop_task = None
        log_info(f"MujocoModel scene={scene_xml} {width}x{height}@{fps}fps")

    async def connect(self):
        self._running = True
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self._executor, self._sim.start)
        self._loop_task = asyncio.ensure_future(self._sim_loop())

    async def _sim_loop(self):
        loop = asyncio.get_event_loop()
        next_frame_time = 0.0
        try:
            while self._running:
                loop_start = time.perf_counter()

                await loop.run_in_executor(self._executor, self._sim.step)

                if self._sim.sim_time >= next_frame_time:
                    frame_arr = await loop.run_in_executor(self._executor, self._sim.render)  # same executor thread as step()
                    if frame_arr is not None:
                        vf = VideoFrame.from_ndarray(frame_arr, format="rgb24")
                        if self._frame_queue.full():
                            try:
                                self._frame_queue.get_nowait()
                            except asyncio.QueueEmpty:
                                pass
                        self._frame_queue.put_nowait(vf)
                    next_frame_time += self._sim.frame_interval

                elapsed = time.perf_counter() - loop_start
                sleep_time = self._sim.timestep - elapsed
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            log_info(f"MujocoModel sim loop error: {e}")
        finally:
            await loop.run_in_executor(self._executor, self._sim.stop)
            log_info("MujocoModel simulation stopped")

    async def send(self, input: Input):
        if not self._running:
            return
        try:
            if isinstance(input, str):
                cmd = json.loads(input)
                if isinstance(cmd, dict):
                    self._sim.process_event(cmd)
            elif isinstance(input, bytes):
                cmd = json.loads(input.decode("utf-8", errors="replace"))
                if isinstance(cmd, dict):
                    self._sim.process_event(cmd)
        except Exception as e:
            log_info(f"MujocoModel send error: {e}")

    async def recv(self) -> AsyncIterator[Output]:
        while self._running:
            try:
                frame = await asyncio.wait_for(self._frame_queue.get(), timeout=1.0)
                yield frame
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                log_info(f"MujocoModel recv error: {e}")
                break

    async def close(self):
        log_info("MujocoModel closing")
        self._running = False
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
            self._loop_task = None
        self._executor.shutdown(wait=False)

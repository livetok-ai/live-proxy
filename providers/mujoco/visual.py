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

_KEY_JOINT_NAMES = {
    "arm": {
        "arrowup": [("joint2", 1)],
        "arrowdown": [("joint2", -1)],
        "arrowleft": [("joint1", 1)],
        "arrowright": [("joint1", -1)],
        "a": [("joint4", -1)],
        "q": [("joint4", 1)],
        "w": [("joint6", -1)],
        "s": [("joint6", 1)],
        "e": [("finger_joint1", 1), ("finger_joint2", 1)],
        "d": [("finger_joint1", -1), ("finger_joint2", -1)],
    },
}

# ---------------------------------------------------------------------------
# Free-flyer / legged locomotion config — used by robots whose base is a
# free joint (no fixed mount) and that move by translating/turning the whole
# body rather than nudging individual joints in place.
# ---------------------------------------------------------------------------

_QUAD_LEG_OFFSETS = {"fl": 0.0, "hr": 0.0, "fr": np.pi, "hl": np.pi}  # diagonal trot pairing
_QUAD_GAIT_AMP_HY = 0.3  # thigh swing amplitude (rad)
_QUAD_GAIT_AMP_KN = -0.35  # knee bend delta during swing lift (rad); negative bends further
_QUAD_GAIT_PHASE_SPEED = 5.0  # gait cycles per second of full-speed movement

_LOCOMOTION_CONFIG = {
    "quadruped": {
        "forward_keys": {"arrowup": 1.0, "arrowdown": -1.0},
        "turn_keys": {"arrowright": 1.0, "arrowleft": -1.0},
        "vertical_keys": {},
        "move_speed": 0.5,  # m/s
        "turn_speed": 1.4,  # rad/s
        "vertical_speed": 0.0,
        "gait": True,
    },
    "drone": {
        "forward_keys": {"w": 1.0, "s": -1.0},
        "turn_keys": {"arrowright": 1.0, "arrowleft": -1.0},
        "vertical_keys": {"arrowup": 1.0, "arrowdown": -1.0},
        "move_speed": 0.8,  # m/s
        "turn_speed": 1.4,  # rad/s
        "vertical_speed": 0.5,  # m/s
        "gait": False,
    },
}


# ---------------------------------------------------------------------------
# Inline simulation engine (no dependency on mujoco/provider/mujoco.py)
# ---------------------------------------------------------------------------


class _Sim:
    """Thin wrapper around MuJoCo model/data/renderer.  All methods are
    blocking and intended to be called from a thread executor."""

    def __init__(self, scene_xml: str, fps: int, width: int, height: int, robot: str = None):
        self.scene_xml = scene_xml
        self.fps = fps
        self.width = width
        self.height = height
        self.robot = robot
        self._model = None
        self._data = None
        self._renderer = None
        self._cam = None
        self._held_keys: set = set()
        self._keys_lock = threading.Lock()
        self._target_qpos = None
        self._key_joints: dict = {}
        self._loco_cfg = _LOCOMOTION_CONFIG.get(robot)
        self._loco: dict = {}
        self._quad_legs: dict = {}

    # -- lifecycle --

    def start(self):
        self._model = mujoco.MjModel.from_xml_path(self.scene_xml)
        self._data = mujoco.MjData(self._model)
        if self._model.nkey > 0:
            try:
                key_id = self._model.key("home").id
                mujoco.mj_resetDataKeyframe(self._model, self._data, key_id)
            except KeyError:
                pass
        self._renderer = mujoco.Renderer(self._model, height=self.height, width=self.width)
        self._cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self._model, self._cam)
        self._cam.distance = _ZOOM_INIT
        self._target_qpos = self._data.qpos.copy()
        self._key_joints = self._resolve_key_joints()
        if self._loco_cfg is not None:
            self._resolve_locomotion()

    def _resolve_key_joints(self) -> dict:
        names = _KEY_JOINT_NAMES.get(self.robot, {})
        resolved = {}
        for key, mappings in names.items():
            entries = []
            for joint_name, direction in mappings:
                try:
                    joint = self._model.joint(joint_name)
                except KeyError:
                    continue
                entries.append((int(joint.qposadr[0]), int(joint.dofadr[0]), int(joint.id), direction))
            if entries:
                resolved[key] = entries
        return resolved

    def _resolve_locomotion(self):
        # The base is the model's only free joint; teleport it directly each
        # frame instead of relying on contact-driven dynamics, which is far
        # more stable for a keyboard-driven demo than a from-scratch balance
        # controller.
        free_joint_id = next(
            i for i in range(self._model.njnt) if self._model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE
        )
        qpos_addr = int(self._model.jnt_qposadr[free_joint_id])
        dof_addr = int(self._model.jnt_dofadr[free_joint_id])
        qpos = self._data.qpos
        self._loco = {
            "qpos_addr": qpos_addr,
            "dof_addr": dof_addr,
            "x": float(qpos[qpos_addr]),
            "y": float(qpos[qpos_addr + 1]),
            "z": float(qpos[qpos_addr + 2]),
            "yaw": 0.0,
            "phase": 0.0,
        }
        if self._loco_cfg.get("gait"):
            self._quad_legs = {}
            for leg in _QUAD_LEG_OFFSETS:
                names = {axis: f"{leg}_{axis}" for axis in ("hx", "hy", "kn")}
                try:
                    joints = {axis: self._model.joint(name) for axis, name in names.items()}
                except KeyError:
                    continue
                self._quad_legs[leg] = {
                    axis: {
                        "qpos_addr": int(joints[axis].qposadr[0]),
                        "dof_addr": int(joints[axis].dofadr[0]),
                        "neutral": float(qpos[int(joints[axis].qposadr[0])]),
                    }
                    for axis in ("hx", "hy", "kn")
                }

    def stop(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # -- simulation --

    def step(self):
        if self._loco_cfg is not None:
            self._step_locomotion()
        else:
            self._apply_held_keys()
            controlled = {
                (qpos_addr, dof_addr)
                for entries in self._key_joints.values()
                for qpos_addr, dof_addr, _, _ in entries
            }
            nq = self._model.nq
            nv = self._model.nv
            for qpos_addr, dof_addr in controlled:
                if qpos_addr >= nq or dof_addr >= nv:
                    continue
                self._data.qpos[qpos_addr] = self._target_qpos[qpos_addr]
                self._data.qvel[dof_addr] = 0.0
        mujoco.mj_step(self._model, self._data)

    def _step_locomotion(self):
        with self._keys_lock:
            keys = set(self._held_keys)
        cfg = self._loco_cfg
        dt = self._model.opt.timestep

        forward = sum(v for k, v in cfg["forward_keys"].items() if k in keys)
        turn = sum(v for k, v in cfg["turn_keys"].items() if k in keys)
        vertical = sum(v for k, v in cfg["vertical_keys"].items() if k in keys)

        loco = self._loco
        loco["yaw"] += turn * cfg["turn_speed"] * dt
        loco["x"] += forward * cfg["move_speed"] * dt * np.cos(loco["yaw"])
        loco["y"] += forward * cfg["move_speed"] * dt * np.sin(loco["yaw"])
        loco["z"] += vertical * cfg["vertical_speed"] * dt
        if cfg["gait"]:
            loco["phase"] += forward * _QUAD_GAIT_PHASE_SPEED * dt

        qpos_addr = loco["qpos_addr"]
        dof_addr = loco["dof_addr"]
        half_yaw = loco["yaw"] / 2.0
        self._data.qpos[qpos_addr + 0] = loco["x"]
        self._data.qpos[qpos_addr + 1] = loco["y"]
        self._data.qpos[qpos_addr + 2] = loco["z"]
        self._data.qpos[qpos_addr + 3] = np.cos(half_yaw)
        self._data.qpos[qpos_addr + 4] = 0.0
        self._data.qpos[qpos_addr + 5] = 0.0
        self._data.qpos[qpos_addr + 6] = np.sin(half_yaw)
        self._data.qvel[dof_addr : dof_addr + 6] = 0.0

        if cfg["gait"]:
            phase = loco["phase"]
            for leg, offset in _QUAD_LEG_OFFSETS.items():
                axes = self._quad_legs.get(leg)
                if not axes:
                    continue
                s = np.sin(phase + offset)
                hy = axes["hy"]["neutral"] + _QUAD_GAIT_AMP_HY * s
                kn = axes["kn"]["neutral"] + _QUAD_GAIT_AMP_KN * max(0.0, s)
                hx = axes["hx"]["neutral"]
                for axis, value in (("hy", hy), ("kn", kn), ("hx", hx)):
                    self._data.qpos[axes[axis]["qpos_addr"]] = value
                    self._data.qvel[axes[axis]["dof_addr"]] = 0.0

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
            if self._model.nkey > 0:
                try:
                    key_id = self._model.key("home").id
                    mujoco.mj_resetDataKeyframe(self._model, self._data, key_id)
                except KeyError:
                    mujoco.mj_resetData(self._model, self._data)
            else:
                mujoco.mj_resetData(self._model, self._data)
            self._target_qpos = self._data.qpos.copy()
            if self._loco_cfg is not None:
                self._resolve_locomotion()
        elif key in ("+", "="):
            self._cam.distance = max(_ZOOM_MIN, self._cam.distance - _ZOOM_STEP)
        elif key == "-":
            self._cam.distance = min(_ZOOM_MAX, self._cam.distance + _ZOOM_STEP)

    def _apply_held_keys(self):
        with self._keys_lock:
            keys = set(self._held_keys)
        njnt = self._model.njnt
        nq = self._model.nq
        for key in keys:
            for qpos_addr, _dof_addr, joint_id, direction in self._key_joints.get(key, []):
                if joint_id >= njnt or qpos_addr >= nq:
                    continue
                lo = self._model.jnt_range[joint_id, 0]
                hi = self._model.jnt_range[joint_id, 1]
                self._target_qpos[qpos_addr] = float(
                    np.clip(
                        self._target_qpos[qpos_addr] + direction * _KEY_DELTA,
                        lo,
                        hi,
                    )
                )


# ---------------------------------------------------------------------------
# Scene registry — maps a `robot` param to a scene XML under ./scenes
# ---------------------------------------------------------------------------

_SCENES_DIR = os.path.join(os.path.dirname(__file__), "scenes")
_ROBOT_SCENES = {
    "arm": os.path.join(_SCENES_DIR, "franka_emika_panda", "scene_factory.xml"),
    "quadruped": os.path.join(_SCENES_DIR, "boston_dynamics_spot", "scene.xml"),
    "drone": os.path.join(_SCENES_DIR, "bitcraze_crazyflie_2", "scene.xml"),
}


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
        _default = _ROBOT_SCENES["arm"]
        robot = kwargs.get("robot")
        scene_xml = kwargs.get("scene_xml") or _ROBOT_SCENES.get(robot) or os.getenv("MUJOCO_SCENE_XML") or _default
        fps = int(kwargs.get("fps", 30))
        width = int(kwargs.get("width", 1280))
        height = int(kwargs.get("height", 720))

        self._sim = _Sim(scene_xml=scene_xml, fps=fps, width=width, height=height, robot=robot)
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
        frame_interval = self._sim.frame_interval
        steps_per_frame = max(1, round(frame_interval / self._sim.timestep))
        start_wall = time.perf_counter()
        frame_count = 0
        try:
            while self._running:
                try:
                    for _ in range(steps_per_frame):
                        await loop.run_in_executor(self._executor, self._sim.step)

                    frame_arr = await loop.run_in_executor(
                        self._executor, self._sim.render
                    )  # same executor thread as step()
                    if frame_arr is not None:
                        vf = VideoFrame.from_ndarray(frame_arr, format="rgb24")
                        if self._frame_queue.full():
                            try:
                                self._frame_queue.get_nowait()
                            except asyncio.QueueEmpty:
                                pass
                        self._frame_queue.put_nowait(vf)
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    log_info(f"MujocoModel sim step error: {e}")

                frame_count += 1
                sleep_time = (start_wall + frame_count * frame_interval) - time.perf_counter()
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
        except asyncio.CancelledError:
            pass
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

# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for real-world tasks, teleoperation, configuration, and layout."""

from __future__ import annotations

import asyncio
import dataclasses
import importlib
import importlib.util
import io
import os
import pickle
import re
import subprocess
import sys
import textwrap
import threading
import time
import types
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import Mock

import gymnasium as gym
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

import rlinf.scheduler.hardware.accelerators.nvidia_gpu as nvidia_gpu
import rlinf.utils.robosuite_compat as robosuite_compat
from rlinf.envs.real import load_tasks
from rlinf.envs.real.dosw1.base import DOSW1Env, DOSW1EnvConfig
from rlinf.envs.real.franka.base import FrankaEnv
from rlinf.envs.real.franka.dual_franka_joint import (
    DualFrankaJointEnv,
)
from rlinf.envs.real.gim_arm.base import GimArmEnv, GimArmEnvConfig
from rlinf.envs.real.wrappers.teleop.config import (  # noqa: E402
    NO_DEVICE,
    resolve_teleop_device,
    resolve_teleop_devices,
)
from rlinf.envs.real.wrappers.teleop.intervention import (  # noqa: E402
    TeleopDevice,
    TeleopIntervention,
    TeleopSample,
)
from rlinf.envs.real.xsquare.base import Turtle2Env, Turtle2EnvConfig
from rlinf.envs.sim.robotwin.seed_utils import partition_success_seeds
from rlinf.robotics import (
    DualFrankaConfig,
    FrankaConfig,
    PiperConfig,
    SO101Config,
)
from rlinf.robotics.discovery import RobotDiscovery
from rlinf.scheduler.hardware.accelerators.nvidia_gpu import (
    EGL_DEVICE_ID_ENV_VARS,
    NvidiaGPUManager,
)
from rlinf.scheduler.manager.net_emulation import (
    CrossDCPair,
    NetEmulationConfig,
    NetEmulationManager,
)

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _robot_info(config):
    """Describe test hardware without probing a physical device."""
    if config is None:
        return None
    from rlinf.robotics.discovery import RobotDiscovery, RobotInfo

    robot_type = next(
        name
        for name, registration in RobotDiscovery.registry.items()
        if isinstance(config, registration.config_cls)
    )
    return RobotInfo(
        type=robot_type, model=config.hardware_model(robot_type), config=config
    )


def _assert_legacy_transition(env) -> None:
    observation, _ = env.reset()
    transition = env.step(env.action_space.sample())

    assert set(observation) == {"state", "frames"}
    assert set(transition[0]) == {"state", "frames"}
    assert len(transition) == 5
    env.close()


@pytest.mark.parametrize(
    ("module_name", "class_name", "override"),
    [
        (
            "rlinf.envs.real.franka",
            "FrankaEnv",
            {"step_frequency": 10000.0},
        ),
        ("rlinf.envs.real.so101", "SO101ReachEnv", {}),
    ],
)
def test_a_hardware_free_env_repeats_with_a_seed(module_name, class_name, override):
    """reset(seed=...) must pin what a dummy env reports.

    With no hardware the env samples its declared space, and a Gymnasium
    space seeds itself from entropy. Two runs of one config then disagree on
    every value, so a dummy end-to-end run can only catch a change of shape.
    That is how an observation built from two different moments went
    unnoticed: it was correctly shaped every time.
    """
    import importlib

    from robot_mocks import mocked_sdks

    def observe(seed):
        with mocked_sdks():
            env_cls = getattr(importlib.import_module(module_name), class_name)
            cfg = {"is_dummy": True, "enable_camera_player": False, **override}
            try:
                env = env_cls(
                    override_cfg=cfg,
                    worker_info=None,
                    env_idx=0,
                    robot_info=_robot_info(
                        FrankaConfig(node_rank=0, camera_serials=["dummy"])
                        if class_name == "FrankaEnv"
                        else None
                    ),
                )
            except TypeError:
                env = env_cls(cfg)
            observation, _ = env.reset(seed=seed)
            return np.concatenate(
                [
                    np.asarray(v).reshape(-1)
                    for _, v in sorted(observation["state"].items())
                ]
            )

    assert np.array_equal(observe(7), observe(7)), "the same seed must repeat"
    assert not np.array_equal(observe(7), observe(8)), "a different seed must differ"


def test_a_franka_observation_comes_from_one_snapshot():
    """Every field a policy sees must describe the same instant.

    _read_robot takes one snapshot per step and the observation is built from
    it. Reading the gripper live instead would mix two moments in one
    recorded transition, by up to a control period.
    """
    import ast
    import inspect
    import textwrap

    source = textwrap.dedent(inspect.getsource(FrankaEnv._get_observation))
    tree = ast.parse(source)

    reads = {
        ast.unparse(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and ast.unparse(node).startswith(("self._franka_state", "self._end_effector"))
    }
    live = sorted(r for r in reads if r.startswith("self._end_effector"))

    assert live == [], (
        f"the observation reads hardware directly: {live}. "
        "Take the value from self._franka_state, which _read_robot fills once."
    )


@pytest.mark.placement
def test_franka_depth_reaches_the_observation_only_when_asked_for():
    """A rig without a depth camera keeps the schema a policy already reads.

    Depth arrives as its own key rather than a fourth channel, so an existing
    policy sees the frames it always saw and one that wants depth reads
    metres without knowing which camera produced them.
    """
    from robot_mocks import mocked_sdks
    from robot_mocks.cameras import DEPTH_FAR, DEPTH_NEAR, DEPTH_SCALE, SERIAL

    from rlinf.scheduler import AcceleratorType, AcceleratorUtil

    # get_accelerator_type returns the type as a plain string, so compare by
    # value; `is` against the enum member is never true.
    if AcceleratorUtil.get_accelerator_type() == AcceleratorType.MUSA_GPU:
        # This test builds a second rig after closing the first, and MUSA
        # reserves so much address space per process that the second placement
        # cannot allocate its worker. The unit-test workflow already runs
        # placement tests one per process for the same reason.
        pytest.skip("MUSA cannot place a second rig in one process")

    with mocked_sdks():
        from rlinf.envs.real.franka.base import FrankaEnv

        def build(enable_camera_depth):
            return FrankaEnv(
                override_cfg={
                    "enable_camera_depth": enable_camera_depth,
                    "enable_camera_player": False,
                },
                worker_info=None,
                env_idx=0,
                robot_info=_robot_info(
                    FrankaConfig(
                        node_rank=0,
                        robot_ip="0.0.0.0",
                        camera_serials=[SERIAL],
                        disable_validate=True,
                    )
                ),
            )

        env = build(False)
        try:
            observation, _ = env.reset()
            assert set(env.observation_space.spaces) == {"state", "frames"}
            assert set(observation) == {"state", "frames"}
        finally:
            env.close()

        env = build(True)
        try:
            observation, _ = env.reset()
            assert set(observation) == {"state", "frames", "depths"}
            depth = observation["depths"]["wrist_1"]
            frame = observation["frames"]["wrist_1"]
            # Cropped and resized to the same view as the frame beside it.
            assert depth.shape == frame.shape[:2]
            assert depth.dtype == np.float32
            # Resampling by nearest keeps every pixel at a distance something
            # was actually measured at; averaging would invent readings between
            # the near and far halves, where nothing is.
            distances = np.unique(depth)
            assert len(distances) == 2
            assert np.allclose(
                distances, [DEPTH_NEAR * DEPTH_SCALE, DEPTH_FAR * DEPTH_SCALE]
            )
        finally:
            env.close()


def test_depth_reaches_the_policy_split_like_the_frames_beside_it():
    """A policy reads depth the way it reads images: main view, then the rest.

    The runner never sees the per-camera dict, so depth has to follow the same
    main/extra split as the frames, keyed by the same camera.
    """
    from rlinf.envs.real.env import RealWorldEnv

    def wrap(raw_observation):
        env = RealWorldEnv.__new__(RealWorldEnv)
        env.main_image_key = "wrist_1"
        env.task_descriptions = ["pick up the cube"]
        return env._wrap_obs(raw_observation)

    frames = {
        "wrist_1": np.zeros((1, 4, 4, 3), dtype=np.uint8),
        "wrist_2": np.ones((1, 4, 4, 3), dtype=np.uint8),
    }
    state = {"tcp_pose": np.zeros((1, 7), dtype=np.float32)}

    observation = wrap({"state": state, "frames": frames})
    assert "main_depths" not in observation
    assert "extra_view_depths" not in observation

    observation = wrap(
        {
            "state": state,
            "frames": frames,
            "depths": {
                "wrist_1": np.full((1, 4, 4), 0.5, dtype=np.float32),
                "wrist_2": np.full((1, 4, 4), 1.5, dtype=np.float32),
            },
        }
    )
    assert observation["main_depths"].shape == (1, 4, 4)
    assert torch.allclose(observation["main_depths"], torch.tensor(0.5))
    # The extra views stack on axis 1, as the extra images do.
    assert observation["extra_view_depths"].shape == (1, 1, 4, 4)
    assert torch.allclose(observation["extra_view_depths"], torch.tensor(1.5))


def test_franka_dummy_preserves_legacy_policy_schema():
    env = FrankaEnv(
        override_cfg={
            "is_dummy": True,
            "enable_camera_player": False,
            "step_frequency": 10000.0,
        },
        worker_info=None,
        env_idx=0,
        robot_info=_robot_info(FrankaConfig(node_rank=0, camera_serials=["dummy"])),
    )

    assert env.action_space.shape == (7,)
    assert env.robot is None
    _assert_legacy_transition(env)


def test_dual_franka_dummy_preserves_legacy_policy_schema():
    env = DualFrankaJointEnv(
        override_cfg={
            "is_dummy": True,
            "enable_camera_player": False,
            "step_frequency": 10000.0,
        },
        worker_info=None,
        env_idx=0,
        robot_info=_robot_info(
            DualFrankaConfig(node_rank=0, base_camera_serials=["dummy"])
        ),
    )

    assert env.action_space.shape == (16,)
    assert env.robot is None
    _assert_legacy_transition(env)


def test_gim_arm_dummy_preserves_legacy_policy_schema():
    env = GimArmEnv(
        config=GimArmEnvConfig(
            is_dummy=True,
            enable_camera_player=False,
            step_frequency=10000.0,
        ),
        worker_info=None,
        robot_info=None,
        env_idx=0,
    )

    assert env.action_space.shape == (7,)
    assert env.robot is None
    _assert_legacy_transition(env)


def test_dosw1_dummy_preserves_legacy_policy_schema():
    env = DOSW1Env(
        config=DOSW1EnvConfig(
            is_dummy=True,
            camera_names=[],
            enable_camera_player=False,
            step_frequency=10000.0,
        ),
        worker_info=None,
        robot_info=None,
        env_idx=0,
    )

    assert env.action_space.shape == (14,)
    assert env.robot is None
    _assert_legacy_transition(env)


def test_turtle2_dummy_preserves_legacy_policy_schema():
    env = Turtle2Env(
        config=Turtle2EnvConfig(
            is_dummy=True,
            step_frequency=10000.0,
        ),
        worker_info=None,
        robot_info=None,
        env_idx=0,
    )

    assert env.action_space.shape == (7,)
    assert env.robot is None
    _assert_legacy_transition(env)


class _TerminatingEnv(gym.Env):
    """Terminates on its second step and counts how often it is reset."""

    observation_space = gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32)
    action_space = gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32)

    def __init__(self):
        self.step_count = 0
        self.resets = 0

    def reset(self, *, seed=None, options=None):
        self.resets += 1
        self.step_count = 0
        return np.array([0.0], dtype=np.float32), {}

    def step(self, action):
        self.step_count += 1
        return (
            np.array([self.step_count], dtype=np.float32),
            float(self.step_count),
            self.step_count >= 2,
            False,
            {},
        )


def test_the_vector_env_keeps_stepping_a_terminated_env():
    """The runner resets on its own schedule, so the batch must not.

    gymnasium renamed the batch arrays in 1.x and added an autoreset mode
    whose DISABLED setting refuses to step a terminated env rather than
    carrying on, so both halves are worth pinning: nothing is reset, and the
    observation is the one the env produced after it terminated.
    """
    from rlinf.envs.real.venv import NoAutoResetSyncVectorEnv

    env = NoAutoResetSyncVectorEnv([_TerminatingEnv, _TerminatingEnv])
    try:
        env.reset(seed=0)
        after_reset = [inner.resets for inner in env.envs]
        for _ in range(4):
            observation, reward, terminated, truncated, _ = env.step(
                np.zeros((2, 1), dtype=np.float32)
            )

        assert [inner.resets for inner in env.envs] == after_reset
        assert terminated.tolist() == [True, True]
        assert observation.ravel().tolist() == [4.0, 4.0]
        assert reward.tolist() == [4.0, 4.0]
    finally:
        env.close()


def _turtle2_camera_check(camera_ids, ready):
    """Run _check_cameras against a rig with the given cameras."""
    env = Turtle2Env.__new__(Turtle2Env)
    env.config = SimpleNamespace(is_dummy=False)
    env.hardware = SimpleNamespace(camera_ids=list(camera_ids))
    env._camera_parts = lambda: [
        SimpleNamespace(is_ready=lambda state=state: state) for state in ready
    ]
    env._check_cameras()


def test_turtle2_accepts_a_camera_selected_by_a_nonzero_id():
    """``camera_ids`` selects hardware; the parts are named by position.

    The shipped default is ``[2]``, one camera, so reading the id as a slot
    rejected a healthy rig.
    """
    _turtle2_camera_check([2], [True])
    _turtle2_camera_check([1], [True])
    _turtle2_camera_check([0, 1, 2], [True, True, True])


def test_turtle2_refuses_a_camera_that_is_not_delivering():
    """A stalled camera is still named by the id that selected it."""
    with pytest.raises(ValueError, match="Camera 3 not available"):
        _turtle2_camera_check([2], [False])
    # Built short: the robot has fewer cameras than the config asked for.
    with pytest.raises(ValueError, match="Camera 3 not available"):
        _turtle2_camera_check([2], [])
    with pytest.raises(ValueError, match="Camera 2 not available"):
        _turtle2_camera_check([0, 1, 2], [True, False, True])


def test_franka_builds_cameras_after_applying_hardware_info(monkeypatch):
    from rlinf.envs.real.franka.base import FrankaEnvConfig
    from rlinf.robotics import FrankaConfig, RobotInfo
    from rlinf.robotics.robots.franka import FrankaRobot

    captured = {}

    class BuiltRobot:
        def connect(self):
            pass

        def child(self, name, part_type=None):
            # The env reaches for the arm and, beside it, the end effector,
            # naming the class it expects each to be.
            assert name in ("arm", "end_effector")
            assert part_type is not None, "the env should say what it expects"
            # is_hand is part of the end-effector contract: the env asks the
            # part which kind it is rather than trusting the config alone.
            return SimpleNamespace(owner=object(), is_hand=False, is_gripper=True)

    def build(**kwargs):
        captured.update(kwargs)
        return BuiltRobot()

    monkeypatch.setattr(FrankaRobot, "build", build)
    env = FrankaEnv.__new__(FrankaEnv)
    env.config = FrankaEnvConfig()
    env.robot_info = RobotInfo(
        type="Robot",
        model="Franka",
        config=FrankaConfig(
            node_rank=0,
            robot_ip="10.0.0.1",
            camera_serials=["hardware-camera"],
        ),
    )
    env.env_idx = 0
    env.node_rank = 0
    env.env_worker_rank = 3
    env.hardware = env.robot_info.config

    env._setup_hardware()

    assert [info.serial_number for info in env._camera_infos] == ["hardware-camera"]
    assert list(captured["cameras"]) == ["wrist_1"]


def test_gim_arm_reopens_the_existing_camera_after_a_stall(monkeypatch):
    from rlinf.robotics.parts.cameras import CameraInfo

    class Camera:
        def __init__(self):
            self._camera_info = CameraInfo("wrist_1", "camera")
            self.reads = 0
            self.reopens = 0

        @property
        def name(self) -> str:
            """As BaseCamera exposes it, so the env need not reach inside."""
            return self._camera_info.name

        def get_frame(self):
            self.reads += 1
            if self.reads == 1:
                raise __import__("queue").Empty
            return np.zeros((8, 8, 3), dtype=np.uint8)

        def reopen(self):
            self.reopens += 1

    env = GimArmEnv.__new__(GimArmEnv)
    camera = Camera()
    env._cameras = [camera]
    env._logger = SimpleNamespace(warning=lambda *args, **kwargs: None)
    env.camera_player = SimpleNamespace(put_frame=lambda frames: None)
    env.observation_space = gym.spaces.Dict(
        {
            "frames": gym.spaces.Dict(
                {"wrist_1": gym.spaces.Box(0, 255, shape=(4, 4, 3), dtype=np.uint8)}
            )
        }
    )
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)

    frames = env._get_camera_frames()

    assert camera.reopens == 1
    assert frames["wrist_1"].shape == (4, 4, 3)


def test_dual_franka_reads_depth_beside_each_frame():
    """Both arms' cameras report depth through the same reading as the frame.

    The dual env reads each camera on its own so a stalled one cannot stall
    the control loop, which is why depth has to arrive on that path too.
    """
    near, far = 0.5, 1.5

    class _Camera:
        def __init__(self, with_depth):
            self.timeouts = []
            self._with_depth = with_depth

        def get_observation(self, timeout=5, attempts=1, wait=0.0):
            self.timeouts.append(timeout)
            frame = np.zeros((8, 8, 3), dtype=np.uint8)
            if not self._with_depth:
                return {"frame": frame}
            # The near/far step sits off the resize grid so an averaging
            # resample would show up as a distance nothing measured.
            depth = np.full((8, 8), far, dtype=np.float32)
            depth[:, :3] = near
            return {"frame": frame, "depth": depth}

    def read(with_depth):
        env = DualFrankaJointEnv.__new__(DualFrankaJointEnv)
        camera = _Camera(with_depth)
        env._cameras = {"left_wrist_0_rgb": camera}
        env._last_camera_frame = {}
        env._logger = SimpleNamespace(error=lambda *args, **kwargs: None)
        env.camera_player = SimpleNamespace(put_frame=lambda frames: None)
        env.observation_space = gym.spaces.Dict(
            {
                "frames": gym.spaces.Dict(
                    {
                        "left_wrist_0_rgb": gym.spaces.Box(
                            0, 255, shape=(4, 4, 3), dtype=np.uint8
                        )
                    }
                )
            }
        )
        return camera, env._get_camera_observation()

    camera, (frames, depths) = read(with_depth=False)
    assert frames["left_wrist_0_rgb"].shape == (4, 4, 3)
    assert depths == {}
    # Read on the control period, not the default timeout, so a stalled
    # camera falls back to its last reading instead of holding the loop.
    assert camera.timeouts == [0.5]

    _, (frames, depths) = read(with_depth=True)
    depth = depths["left_wrist_0_rgb"]
    assert depth.shape == (4, 4)
    # Nearest resampling keeps every pixel at a measured distance.
    distances = np.unique(depth)
    assert len(distances) == 2
    assert np.allclose(distances, [near, far])


def test_dual_franka_declares_depth_only_where_a_camera_captures_it():
    """The depth space follows the cameras, so a rig without one is unchanged."""

    def camera_spaces(enable_camera_depth):
        env = DualFrankaJointEnv.__new__(DualFrankaJointEnv)
        env.hardware = DualFrankaConfig(
            node_rank=0, base_camera_serials=["dummy"], camera_type="realsense"
        )
        env.config = SimpleNamespace(enable_camera_depth=enable_camera_depth)
        return env._build_camera_spaces()

    assert set(camera_spaces(False)) == {"frames"}

    spaces = camera_spaces(True)
    assert set(spaces) == {"frames", "depths"}
    depth_space = spaces["depths"]["base_0_rgb"]
    assert depth_space.shape == spaces["frames"]["base_0_rgb"].shape[:2]
    assert depth_space.dtype == np.float32


def test_dual_franka_runs_independent_arm_calls_concurrently():
    env = DualFrankaJointEnv.__new__(DualFrankaJointEnv)
    env._arm_executors = (
        ThreadPoolExecutor(max_workers=1),
        ThreadPoolExecutor(max_workers=1),
    )
    rendezvous = threading.Barrier(2)

    def call(side):
        rendezvous.wait(timeout=1.0)
        return side

    try:
        assert env._run_arm_calls(lambda: call("left"), lambda: call("right")) == (
            "left",
            "right",
        )
    finally:
        for executor in env._arm_executors:
            executor.shutdown(wait=True)


def test_dual_franka_applies_a_tasks_compliance_to_both_arms():
    env = DualFrankaJointEnv.__new__(DualFrankaJointEnv)
    env.config = SimpleNamespace(compliance_param={"translational_stiffness": 800})
    env._arm_executors = (
        ThreadPoolExecutor(max_workers=1),
        ThreadPoolExecutor(max_workers=1),
    )

    class Arm:
        def __init__(self):
            self.applied = None

        def reconfigure_compliance_params(self, params):
            self.applied = params

    env._left_arm, env._right_arm = Arm(), Arm()
    try:
        env._reconfigure_compliance()
    finally:
        for executor in env._arm_executors:
            executor.shutdown(wait=True)

    assert env._left_arm.applied == {"translational_stiffness": 800}
    assert env._right_arm.applied == {"translational_stiffness": 800}


def test_dual_franka_does_not_wait_for_gripper_motion():
    env = DualFrankaJointEnv.__new__(DualFrankaJointEnv)
    env.config = SimpleNamespace(binary_gripper_threshold=0.5)
    env._logger = SimpleNamespace(warning=lambda *args, **kwargs: None)
    env._arm_executors = (
        ThreadPoolExecutor(max_workers=1),
        ThreadPoolExecutor(max_workers=1),
    )
    entered = threading.Event()
    release = threading.Event()

    class Hand:
        is_open = True

        def close(self):
            entered.set()
            assert release.wait(timeout=1.0)

    try:
        changed = env._gripper_action(0, Hand(), -1.0)
        assert changed
        assert entered.wait(timeout=1.0)
        assert not release.is_set(), "the control loop must not wait for the gripper"
    finally:
        release.set()
        for executor in env._arm_executors:
            executor.shutdown(wait=True)


def test_franka_reward_model_waits_for_the_worker_result():
    class Work:
        def wait(self):
            return [np.array([0.75], dtype=np.float32)]

    env = FrankaEnv.__new__(FrankaEnv)
    env.config = SimpleNamespace(reward_image_key=None)
    env._reward_worker = SimpleNamespace(compute_reward=lambda _batch: Work())

    reward = env._compute_reward_model(
        {"frames": {"wrist_1": np.zeros((4, 4, 3), dtype=np.uint8)}}
    )

    assert reward == pytest.approx(0.75)


def test_direct_gello_stream_keeps_both_arm_commands_concurrent():
    from rlinf.envs.real.wrappers.teleop.adapters import DualGelloJointStream

    rendezvous = threading.Barrier(2)

    class Controller:
        def move_joints(self, _target):
            rendezvous.wait(timeout=1.0)

    class Leader:
        ready = True

        def get_observation(self):
            return {"joint_position": np.zeros(7), "grip": np.zeros(1)}

    env = DualFrankaJointEnv.__new__(DualFrankaJointEnv)
    env._left_ctrl = Controller()
    env._right_ctrl = Controller()
    env._arm_executors = (
        ThreadPoolExecutor(max_workers=1),
        ThreadPoolExecutor(max_workers=1),
    )
    streamer = DualGelloJointStream(
        Leader(), Leader(), gripper_enabled=False, direct_stream=False
    )

    try:
        streamer.stream_once(env)
    finally:
        for executor in env._arm_executors:
            executor.shutdown(wait=True)


class FakeEnv(gym.Env):
    """Record actions passed to ``step``."""

    def __init__(self) -> None:
        self.stepped: list[np.ndarray] = []
        self.reset_calls = 0
        self.closed = False

    def step(self, action):
        self.stepped.append(np.asarray(action))
        return {"obs": 1}, 0.0, False, False, {}

    def reset(self, **kwargs):
        self.reset_calls += 1
        return {"obs": 1}, {}

    def close(self):
        self.closed = True


class ScriptedDevice(TeleopDevice):
    """Return one scripted sample per read."""

    def __init__(self, samples: list[TeleopSample]) -> None:
        self.samples = list(samples)
        self.reads = 0
        self.resets = 0
        self.closed = False
        self.before_steps = 0
        self.fallback_action: Optional[np.ndarray] = None

    def read(self, env: Any, policy_action: np.ndarray) -> TeleopSample:
        sample = self.samples[min(self.reads, len(self.samples) - 1)]
        self.reads += 1
        return sample

    def reset(self, env: Any) -> None:
        self.resets += 1

    def before_step(self, env: Any) -> None:
        self.before_steps += 1

    def fallback(self, env: Any, policy_action: np.ndarray) -> np.ndarray:
        if self.fallback_action is not None:
            return self.fallback_action
        return policy_action

    def close(self) -> None:
        self.closed = True


POLICY = np.array([0.0, 0.0, 0.0])


EXPERT = np.array([1.0, 1.0, 1.0])


def test_active_sample_replaces_the_policy_action():
    env = FakeEnv()
    wrapper = TeleopIntervention(
        env, ScriptedDevice([TeleopSample(action=EXPERT, active=True)])
    )

    _, _, _, _, info = wrapper.step(POLICY)

    assert np.array_equal(env.stepped[0], EXPERT)
    assert np.array_equal(info["intervene_action"], EXPERT)


def test_inactive_device_leaves_the_policy_action_alone():
    env = FakeEnv()
    wrapper = TeleopIntervention(
        env, ScriptedDevice([TeleopSample(action=None, active=False)])
    )

    _, _, _, _, info = wrapper.step(POLICY)

    assert np.array_equal(env.stepped[0], POLICY)
    assert "intervene_action" not in info


def test_control_is_held_between_samples_then_released():
    env = FakeEnv()
    device = ScriptedDevice(
        [
            TeleopSample(action=EXPERT, active=True),
            TeleopSample(action=EXPERT, active=False),
        ]
    )
    wrapper = TeleopIntervention(env, device)

    wrapper.step(POLICY)  # Operator moves.
    wrapper.step(POLICY)  # Quiet sample within the hold window.
    assert np.array_equal(env.stepped[1], EXPERT)

    device.timeout = 0.0  # Hold window expires.
    wrapper.step(POLICY)
    assert np.array_equal(env.stepped[2], POLICY)


class _StubDevice:
    """A teleop device that reads nothing, for arbitration tests.

    Stands in for the parts of the device contract the group touches, so a
    test can supply only the mapping it wants to exercise.
    """

    NEEDS = ()
    CLIPS_TO_ACTION_SPACE = False
    APPLIES_WHILE_IDLE = False
    HOLD_WINDOW = None
    is_connected = True

    def get_observation(self):
        return {}

    def publish(self, reading):
        return {}

    def hold(self, context):
        return {}

    def on_action_chunk_begin(self):
        pass

    def on_reset(self, context=None):
        pass

    def connect(self):
        pass

    def disconnect(self):
        pass

    def drive(self, context, reading=None):
        if reading is None:
            reading = self.get_observation()
        return self.action(reading, context)


def test_an_unfilled_part_keeps_the_policy_action():
    import numpy as np

    from rlinf.envs.real.wrappers.teleop.composed import ComposedTeleop
    from rlinf.robotics.actions import ActionKind
    from rlinf.robotics.parts.teleop import TeleopEntry, TeleopGroup

    class Fixed(_StubDevice):
        PRODUCES = {"hand": ActionKind.HAND}

        def action(self, reading, context):
            from rlinf.robotics.parts.teleop import TeleopAction

            return TeleopAction(parts={"hand": np.full(6, 0.5)}, driving=True)

    layout = {"arm": slice(0, 6), "hand": slice(6, 12)}
    group = TeleopGroup(
        [TeleopEntry(Fixed())],
        available={"arm": ActionKind.CARTESIAN_DELTA, "hand": ActionKind.HAND},
    )
    device = ComposedTeleop(group, layout)

    policy = np.arange(12, dtype=np.float64)
    sample = device.read(_FakeLayoutEnv(), policy)

    assert np.allclose(sample.action[:6], policy[:6])  # Arm remains policy-driven.
    assert np.allclose(sample.action[6:], 0.5)  # Glove controls the hand.


def test_an_idle_glove_keeps_its_hand_pose_without_claiming_the_arm():
    from rlinf.envs.real.wrappers.teleop.composed import ComposedTeleop
    from rlinf.robotics.actions import ActionKind
    from rlinf.robotics.parts.teleop import TeleopAction, TeleopEntry, TeleopGroup

    class HeldHand(_StubDevice):
        PRODUCES = {"hand": ActionKind.HAND}
        APPLIES_WHILE_IDLE = True

        def action(self, reading, context):
            return TeleopAction(parts={"hand": np.full(6, 0.5)}, driving=False)

    layout = {"arm": slice(0, 6), "hand": slice(6, 12)}
    group = TeleopGroup(
        [TeleopEntry(HeldHand())],
        available={"arm": ActionKind.CARTESIAN_DELTA, "hand": ActionKind.HAND},
    )
    policy = np.arange(12, dtype=np.float64)
    sample = ComposedTeleop(group, layout).read(_FakeLayoutEnv(), policy)
    env = FakeEnv()

    _, _, _, _, info = TeleopIntervention(env, ScriptedDevice([sample])).step(policy)

    assert np.allclose(env.stepped[0][:6], policy[:6])
    assert np.allclose(env.stepped[0][6:], 0.5)
    assert "intervene_action" not in info


class _FakeLayoutEnv:
    """Provide the attributes required by teleoperation layout tests."""

    unwrapped = None

    def get_wrapper_attr(self, name):
        raise AttributeError(name)


def test_mark_flag_is_opt_in():
    sample = TeleopSample(action=EXPERT, active=True)

    plain = TeleopIntervention(FakeEnv(), ScriptedDevice([sample]))
    flagged = TeleopIntervention(FakeEnv(), ScriptedDevice([sample]), mark_flag=True)

    assert "intervene_flag" not in plain.step(POLICY)[4]
    assert flagged.step(POLICY)[4]["intervene_flag"] == np.ones(1)


def test_device_info_reaches_the_step_info():
    env = FakeEnv()
    wrapper = TeleopIntervention(
        env,
        ScriptedDevice([TeleopSample(action=EXPERT, active=True, info={"left": True})]),
    )

    assert wrapper.step(POLICY)[4]["left"] is True


def test_reset_resyncs_the_device_and_drops_the_hold():
    env = FakeEnv()
    device = ScriptedDevice(
        [
            TeleopSample(action=EXPERT, active=True),
            TeleopSample(action=EXPERT, active=False),
        ]
    )
    wrapper = TeleopIntervention(env, device)

    wrapper.step(POLICY)
    assert wrapper.intervening

    wrapper.reset()

    assert device.resets == 1
    assert env.reset_calls == 1
    assert not wrapper.intervening


def test_close_releases_the_device_before_the_env():
    env = FakeEnv()
    device = ScriptedDevice([TeleopSample(action=None, active=False)])
    wrapper = TeleopIntervention(env, device)

    wrapper.close()

    assert device.closed
    assert env.closed


def test_before_step_runs_ahead_of_the_env():
    env = FakeEnv()
    device = ScriptedDevice([TeleopSample(action=None, active=False)])
    wrapper = TeleopIntervention(env, device)

    wrapper.step(POLICY)

    assert device.before_steps == 1


def test_read_is_abstract():
    with pytest.raises(TypeError):
        TeleopDevice()


SINGLE_ARM = ("spacemouse", "gello", "pico")


def test_retired_single_key_still_selects_its_device_and_warns():
    with pytest.warns(DeprecationWarning, match="'teleop_device' is retired"):
        device = resolve_teleop_device({"teleop_device": "gello"}, supported=SINGLE_ARM)

    assert device == "gello"


def test_missing_config_falls_back_to_the_default():
    assert resolve_teleop_device({}, supported=SINGLE_ARM) == NO_DEVICE
    assert (
        resolve_teleop_device({}, supported=SINGLE_ARM, default="spacemouse")
        == "spacemouse"
    )


def test_retired_boolean_still_selects_its_device_and_warns():
    with pytest.warns(DeprecationWarning, match="use_pico"):
        device = resolve_teleop_device({"use_pico": True}, supported=SINGLE_ARM)

    assert device == "pico"


def test_all_retired_booleans_off_means_no_device():
    with pytest.warns(DeprecationWarning):
        device = resolve_teleop_device(
            {"use_spacemouse": False, "use_gello": False},
            supported=SINGLE_ARM,
            default="spacemouse",
        )

    assert device == NO_DEVICE


def test_two_retired_booleans_on_is_an_error():
    with pytest.raises(ValueError, match="Only one teleop device"):
        resolve_teleop_device(
            {"use_spacemouse": True, "use_pico": True}, supported=SINGLE_ARM
        )


def test_disagreeing_old_and_new_keys_are_refused():
    with pytest.raises(ValueError, match="cannot be reconciled"):
        resolve_teleop_device(
            {"teleop_device": "pico", "use_spacemouse": True},
            supported=SINGLE_ARM,
        )


def test_agreeing_old_and_new_keys_only_warn():
    with pytest.warns(DeprecationWarning, match="redundant"):
        device = resolve_teleop_device(
            {"teleop_device": "pico", "use_pico": True}, supported=SINGLE_ARM
        )

    assert device == "pico"


def test_device_the_env_cannot_drive_is_refused():
    with pytest.raises(ValueError, match="Unsupported teleop device"):
        resolve_teleop_device(
            {"teleop_device": "spacemouse"}, supported=("gello_joint", "pico")
        )


def test_none_is_always_allowed():
    with pytest.warns(DeprecationWarning):
        device = resolve_teleop_device({"teleop_device": "none"}, supported=("pico",))

    assert device == NO_DEVICE


def test_shipped_configs_use_the_new_key():
    roots = [_ROOT / "examples", _ROOT / "evaluations", _ROOT / "tests"]
    offenders = []
    for root in roots:
        for path in root.rglob("*.yaml"):
            for number, line in enumerate(path.read_text().splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if any(
                    stripped.startswith(f"{flag}:")
                    for flag in (
                        "teleop_device",
                        "use_spacemouse",
                        "use_gello",
                        "use_gello_joint",
                        "use_pico",
                    )
                ):
                    offenders.append(f"{path.relative_to(_ROOT)}:{number}")

    assert offenders == []


_REAL = _ROOT / "rlinf" / "envs" / "real"


_ROBOTS = ("franka", "dosw1", "gim_arm", "xsquare", "ur5e")


EXPECTED_IDS = {
    "UR5eEnv-v1",
    "FrankaEnv-v1",
    "PegInsertionEnv-v1",
    "FrankaBinRelocationEnv-v1",
    "BottleEnv-v1",
    "DexpnpEnv-v1",
    "DualFrankaJointEnv-v1",
    "DualFrankaTCPEnv-v1",
    "DOSW1PickEnv-v1",
    "ButtonEnv-v1",
    "GimArmPegInsertionEnv-v1",
}


def test_no_robot_keeps_a_tasks_subpackage():
    leftovers = [name for name in _ROBOTS if (_REAL / name / "tasks").exists()]

    assert leftovers == []


def test_every_robot_folder_has_a_base():
    missing = [name for name in _ROBOTS if not (_REAL / name / "base.py").exists()]

    assert missing == []


def test_all_task_ids_are_registered():
    from gymnasium.envs.registration import registry

    from rlinf.envs.real import RealWorldEnv

    assert RealWorldEnv is not None
    assert EXPECTED_IDS <= set(registry)


def test_every_entry_point_resolves():
    from gymnasium.envs.registration import registry

    from rlinf.envs.real import RealWorldEnv

    assert RealWorldEnv is not None
    unresolved = []
    for env_id in sorted(EXPECTED_IDS):
        entry_point = registry[env_id].entry_point
        module_name, _, attribute = str(entry_point).partition(":")
        module = importlib.import_module(module_name)
        if getattr(module, attribute, None) is None:
            unresolved.append(f"{env_id} -> {entry_point}")

    assert unresolved == []


def test_task_tables_cover_the_wrapped_robots():
    from rlinf.envs.real import dosw1, franka, ur5e, xsquare

    declared = (
        set(franka.TASKS) | set(dosw1.TASKS) | set(xsquare.TASKS) | set(ur5e.TASKS)
    )

    # GimArm registers its environment class without a wrapper factory.
    assert declared == EXPECTED_IDS - {"GimArmPegInsertionEnv-v1"}


def test_pose_math_is_not_filed_under_a_robot():
    from rlinf.envs.real.utils import pose

    assert hasattr(pose, "construct_adjoint_matrix")
    assert not (_REAL / "franka" / "utils.py").exists()


def test_task_configs_state_only_their_compliance_deltas():
    from rlinf.envs.real.franka.base import COMPLIANCE_DEFAULTS
    from rlinf.envs.real.franka.bin_relocation import BinEnvConfig
    from rlinf.envs.real.franka.bottle import BottleConfig
    from rlinf.envs.real.franka.dex_pnp import DexpnpConfig
    from rlinf.envs.real.franka.peg_insertion import PegInsertionConfig

    deltas = {
        cls.__name__: {
            key
            for key, value in cls().compliance_param.items()
            if COMPLIANCE_DEFAULTS[key] != value
        }
        for cls in (PegInsertionConfig, BottleConfig, BinEnvConfig, DexpnpConfig)
    }

    # Every task receives the complete gain set after defaults are applied.
    for cls in (PegInsertionConfig, BottleConfig, BinEnvConfig, DexpnpConfig):
        assert set(cls().compliance_param) == set(COMPLIANCE_DEFAULTS)
    assert {name: len(keys) for name, keys in deltas.items()} == {
        "PegInsertionConfig": 1,
        "BottleConfig": 8,
        "BinEnvConfig": 11,
        "DexpnpConfig": 6,
    }


def test_unknown_compliance_gain_is_refused():
    import pytest

    from rlinf.envs.real.franka.base import compliance

    with pytest.raises(KeyError, match="Unknown compliance gains"):
        compliance(translational_stifness=1000)


# Wrapper families


def test_wrappers_are_split_by_what_they_change():
    real = _ROOT / "rlinf" / "envs" / "real"
    wrappers = real / "wrappers"

    assert wrappers.is_dir(), "the three families live under one parent"
    for family in ("teleop", "transforms", "episode"):
        assert (wrappers / family / "__init__.py").exists(), family

    # Top-level modules contain robot packages and shared environment machinery.
    loose = sorted(
        path.stem for path in real.glob("*.py") if path.name != "__init__.py"
    )
    assert loose == ["cartesian", "env", "registry", "venv"], loose


def test_no_teleop_wrapper_is_left_outside_teleop():
    real = _ROOT / "rlinf" / "envs" / "real"
    strays = sorted(
        path.name
        for family in ("transforms", "episode")
        for path in (real / "wrappers" / family).glob("*.py")
        if "intervention" in path.name and "leader_follower" not in path.name
    )

    assert strays == []


def test_a_held_button_device_does_not_keep_control_after_release():
    from rlinf.robotics.parts.teleop import Pico, PicoTcp

    assert Pico.HOLD_WINDOW == 0.0
    assert PicoTcp.HOLD_WINDOW == 0.0


def test_streaming_device_lifecycle_without_hardware():
    from rlinf.envs.real.wrappers.teleop.intervention import TeleopSample
    from rlinf.envs.real.wrappers.teleop.streaming import TeleopStreamer

    ticks = []

    class Fake(TeleopStreamer):
        def read(self, env, policy_action):
            return TeleopSample(action=None, active=False)

        def stream_once(self, env):
            ticks.append(1)

    device = Fake(period=0.001, enabled=True)
    device.before_reset(None, {})
    device.reset(None)
    device.after_reset(None)
    deadline = time.monotonic() + 1.0
    while not ticks and time.monotonic() < deadline:
        time.sleep(0.01)
    assert ticks, "stream thread never ran"
    assert device.streaming

    device.close()

    assert not device.streaming


# Keyboard sessions


def _keyboard_session(monkeypatch, queued):
    """Build a keyboard session that replays queued key batches."""
    from rlinf.envs.real.wrappers.episode import session as session_module

    class FakeListener:
        def __init__(self):
            self.batches = list(queued)

        def pop_pressed_keys(self):
            return self.batches.pop(0) if self.batches else []

        def get_key(self):
            batch = self.pop_pressed_keys()
            return batch[0] if batch else None

    monkeypatch.setattr(session_module, "KeyboardListener", FakeListener)

    class Env(gym.Env):
        def __init__(self):
            self.resets = 0

        def reset(self, seed=None, options=None):
            self.resets += 1
            return {}, {}

        def step(self, action):
            return {}, 0.0, False, False, {}

    return session_module.KeyboardSession(Env())


def test_repeat_presses_within_the_debounce_window_are_dropped(monkeypatch):
    session = _keyboard_session(monkeypatch, [["a"], ["a"], ["b"]])

    assert list(session.presses()) == ["a"]
    assert list(session.presses()) == []  # Same key within the debounce window.
    assert list(session.presses()) == ["b"]  # A different key is accepted.


def test_presses_queued_between_episodes_do_not_leak(monkeypatch):
    session = _keyboard_session(monkeypatch, [["c"], ["a"]])

    session.reset()

    assert session.env.resets == 1
    assert list(session.presses()) == ["a"]  # The queued key was drained.


def test_every_keyboard_wrapper_shares_the_session(monkeypatch):
    from rlinf.envs.real.wrappers.episode import (
        KeyboardEvalControlWrapper,
        KeyboardRewardDoneMultiStageWrapper,
        KeyboardRewardDoneWrapper,
        KeyboardRLTPolicySwitchWrapper,
        KeyboardStartEndWrapper,
    )
    from rlinf.envs.real.wrappers.episode.session import KeyboardSession

    for wrapper in (
        KeyboardEvalControlWrapper,
        KeyboardRLTPolicySwitchWrapper,
        KeyboardStartEndWrapper,
        KeyboardRewardDoneWrapper,
        KeyboardRewardDoneMultiStageWrapper,
    ):
        assert issubclass(wrapper, KeyboardSession), wrapper.__name__


def test_episode_wrappers_report_through_the_logger():
    episode_dir = _ROOT / "rlinf" / "envs" / "real" / "episode"
    offenders = sorted(
        path.name
        for path in episode_dir.glob("*.py")
        if re.search(r"^\s*print\(", path.read_text(), re.M)
    )

    assert offenders == []


def test_euler_conversion_is_one_wrapper_for_any_arm_count():
    import numpy as np
    from gymnasium import spaces

    from rlinf.envs.real.wrappers.transforms import (
        DualQuat2EulerWrapper,
        Quat2EulerWrapper,
    )

    class Env(gym.Env):
        def __init__(self, dim):
            self.observation_space = spaces.Dict(
                {
                    "state": spaces.Dict(
                        {"tcp_pose": spaces.Box(-np.inf, np.inf, (dim,))}
                    )
                }
            )

    identity_quat = np.array([0.0, 0.0, 0.0, 1.0])
    one = np.concatenate([np.array([1.0, 2.0, 3.0]), identity_quat])

    single = Quat2EulerWrapper(Env(7))
    dual = DualQuat2EulerWrapper(Env(14))

    assert single.observation_space["state"]["tcp_pose"].shape == (6,)
    assert dual.observation_space["state"]["tcp_pose"].shape == (12,)

    got = single.observation({"state": {"tcp_pose": one.copy()}})["state"]["tcp_pose"]
    assert np.allclose(got, [1.0, 2.0, 3.0, 0.0, 0.0, 0.0])

    both = dual.observation({"state": {"tcp_pose": np.concatenate([one, one])}})
    assert np.allclose(both["state"]["tcp_pose"], [1.0, 2.0, 3.0, 0, 0, 0] * 2)


# Full wrapper stacks built through the production builders


def _dummy_franka(env_cls=None, **overrides):
    from rlinf.envs.real.franka.base import FrankaEnv

    cfg = {
        "is_dummy": True,
        "enable_camera_player": False,
        "step_frequency": 10000.0,
    }
    robot_info = overrides.pop(
        "robot_info", _robot_info(FrankaConfig(node_rank=0, camera_serials=["dummy"]))
    )
    cfg.update(overrides)
    return (env_cls or FrankaEnv)(
        override_cfg=cfg,
        worker_info=None,
        env_idx=0,
        robot_info=robot_info,
    )


def _chain(env):
    """Return wrapper class names from outermost to innermost."""
    names = []
    while hasattr(env, "env"):
        names.append(type(env).__name__)
        env = env.env
    return names


def test_wrapper_stack_converts_the_pose_it_hands_the_policy():
    from rlinf.envs.real.wrappers import build_stack

    env = _dummy_franka()
    raw, _ = env.reset()
    assert raw["state"]["tcp_pose"].shape == (7,)

    wrapped = build_stack(
        env, {"teleop": "none", "no_gripper": False, "use_relative_frame": True}
    )
    observation, _ = wrapped.reset()

    assert _chain(wrapped) == ["Quat2EulerWrapper", "RelativeFrame"]
    assert observation["state"]["tcp_pose"].shape == (6,)
    wrapped.close()


def test_no_teleop_device_leaves_no_intervention_in_the_stack():
    from rlinf.envs.real.wrappers import build_stack

    wrapped = build_stack(
        _dummy_franka(),
        {"teleop": "none", "no_gripper": False, "use_relative_frame": False},
    )

    assert not any("Intervention" in name for name in _chain(wrapped))
    wrapped.close()


def test_no_gripper_narrows_the_action_the_policy_must_produce():
    from rlinf.envs.real.wrappers import build_stack

    env = _dummy_franka()
    assert env.action_space.shape == (7,)

    wrapped = build_stack(
        env,
        {"teleop": "none", "no_gripper": True, "use_relative_frame": False},
    )

    assert "GripperCloseEnv" in _chain(wrapped)
    assert wrapped.action_space.shape == (6,)
    wrapped.close()


def test_the_no_gripper_default_does_not_wrap_a_dexterous_hand():
    from rlinf.envs.real.wrappers import build_stack

    wrapped = build_stack(
        _dummy_franka(
            robot_info=_robot_info(
                FrankaConfig(
                    node_rank=0,
                    camera_serials=["dummy"],
                    end_effector_type="ruiyan_hand",
                )
            ),
            hand_target_state=np.zeros(6),
            hand_reset_state=np.zeros(6),
        ),
        {"teleop": "none", "use_relative_frame": False},
    )

    assert "GripperCloseEnv" not in _chain(wrapped)
    assert wrapped.action_space.shape == (12,)
    wrapped.close()


def test_gim_arm_keeps_the_unwrapped_legacy_action_and_observation_schema():
    from rlinf.envs.real.gim_arm.base import GimArmEnv, GimArmEnvConfig
    from rlinf.envs.real.wrappers import build_stack

    env = GimArmEnv(
        config=GimArmEnvConfig(is_dummy=True),
        worker_info=None,
        robot_info=None,
        env_idx=0,
    )
    wrapped = build_stack(env, {})

    assert wrapped is env
    assert wrapped.action_space.shape == (7,)
    assert wrapped.observation_space["state"]["tcp_pose"].shape == (7,)
    wrapped.close()


def test_dual_franka_keeps_the_legacy_no_gripper_default():
    from rlinf.envs.real.franka.dual_franka_joint import DualFrankaJointEnv
    from rlinf.envs.real.wrappers import build_stack

    env = DualFrankaJointEnv(
        override_cfg={
            "is_dummy": True,
            "enable_camera_player": False,
            "step_frequency": 10000.0,
        },
        worker_info=None,
        env_idx=0,
        robot_info=_robot_info(
            DualFrankaConfig(
                node_rank=0,
                base_camera_serials=["dummy"],
                left_camera_serials=[],
                right_camera_serials=[],
            )
        ),
    )

    with pytest.raises(NotImplementedError, match="no_gripper"):
        build_stack(env, {"teleop": "none"})
    env.close()


def test_a_task_env_runs_with_its_own_config():
    from rlinf.envs.real.franka.base import COMPLIANCE_DEFAULTS
    from rlinf.envs.real.franka.peg_insertion import PegInsertionEnv

    env = _dummy_franka(
        PegInsertionEnv, target_ee_pose=[0.5, 0.0, 0.1, -3.14, 0.0, 0.0]
    )

    assert env.config.task_description == "peg and insertion"
    # Task-specific gains override shared defaults.
    assert set(env.config.compliance_param) == set(COMPLIANCE_DEFAULTS)
    assert env.config.compliance_param["translational_stiffness"] == 2000

    env.reset()
    observation, reward, terminated, truncated, info = env.step(
        env.action_space.sample()
    )

    assert set(observation) == {"state", "frames"}
    assert isinstance(bool(terminated), bool)
    env.close()


def test_every_registered_task_builds_through_its_entry_point():
    import gymnasium as gym

    from rlinf.envs.real import RealWorldEnv  # noqa: F401  (registers the tasks)

    cfg = {
        "teleop": "none",
        "no_gripper": False,
        "use_relative_frame": False,
    }
    built = []
    for env_id in ("FrankaEnv-v1", "PegInsertionEnv-v1", "BottleEnv-v1"):
        env = gym.make(
            env_id,
            override_cfg={
                "is_dummy": True,
                "enable_camera_player": False,
                "step_frequency": 10000.0,
            },
            worker_info=None,
            env_idx=0,
            env_cfg=cfg,
            robot_info=_robot_info(FrankaConfig(node_rank=0, camera_serials=["dummy"])),
        )
        env.reset()
        env.close()
        built.append(env_id)

    assert built == ["FrankaEnv-v1", "PegInsertionEnv-v1", "BottleEnv-v1"]


def test_converted_pose_stays_inside_the_observation_space():
    from rlinf.envs.real.wrappers import build_stack

    wrapped = build_stack(
        _dummy_franka(),
        {"teleop": "none", "no_gripper": False, "use_relative_frame": False},
    )
    observation, _ = wrapped.reset()

    assert (
        observation["state"]["tcp_pose"].dtype
        == wrapped.observation_space["state"]["tcp_pose"].dtype
    )
    assert wrapped.observation_space.contains(observation)
    wrapped.close()


# Multiple teleoperation devices


class _FakeInner:
    """Provide the environment attributes read by teleoperation builders."""

    def __init__(self, **config: Any) -> None:
        self.config = SimpleNamespace(**config)


def test_one_named_device_resolves_to_one_entry():
    assert resolve_teleop_devices({"teleop": "gello"}, supported=SINGLE_ARM) == [
        "gello"
    ]


def test_saying_nothing_resolves_to_no_entries():
    assert resolve_teleop_devices({}, supported=SINGLE_ARM) == []
    assert resolve_teleop_devices({"teleop": "none"}, supported=SINGLE_ARM) == []


def test_a_list_keeps_every_device_it_names():
    assert resolve_teleop_devices(
        {"teleop": ["spacemouse", "glove"]}, supported=("spacemouse", "glove")
    ) == ["spacemouse", "glove"]


def test_an_entry_carries_its_own_options():
    entries = resolve_teleop_devices(
        {"teleop": [{"gello_joint": {"port": "/dev/left", "drives": "left"}}]},
        supported=("gello_joint",),
    )

    assert entries == [{"gello_joint": {"port": "/dev/left", "drives": "left"}}]


def test_one_device_may_appear_twice_on_different_branches():
    entries = resolve_teleop_devices(
        {
            "teleop": [
                {"gello_joint": {"drives": "left"}},
                {"gello_joint": {"drives": "right"}},
            ]
        },
        supported=("gello_joint",),
    )

    assert [entry["gello_joint"]["drives"] for entry in entries] == ["left", "right"]


def test_a_listed_device_the_env_cannot_drive_is_refused():
    with pytest.raises(ValueError, match="Unsupported teleop device"):
        resolve_teleop_devices(
            {"teleop": ["spacemouse", "glove"]}, supported=("gello_joint", "pico")
        )


def test_a_list_supersedes_a_retired_key_underneath_it():
    with pytest.warns(DeprecationWarning, match="supersedes"):
        entries = resolve_teleop_devices(
            {"teleop": ["spacemouse"], "teleop_device": "none"},
            supported=SINGLE_ARM,
        )

    assert entries == ["spacemouse"]


def test_a_list_supersedes_a_retired_boolean():
    with pytest.warns(DeprecationWarning, match="supersedes"):
        entries = resolve_teleop_devices(
            {"teleop": ["spacemouse"], "use_pico": True}, supported=SINGLE_ARM
        )

    assert entries == ["spacemouse"]


def test_an_empty_list_is_refused():
    with pytest.raises(ValueError, match="'teleop' is empty"):
        resolve_teleop_devices({"teleop": []}, supported=SINGLE_ARM)


def test_none_cannot_share_the_list():
    with pytest.raises(ValueError, match="cannot share the list"):
        resolve_teleop_devices({"teleop": ["none", "spacemouse"]}, supported=SINGLE_ARM)


def test_none_alone_in_a_list_means_nobody_takes_over():
    assert resolve_teleop_devices({"teleop": ["none"]}, supported=SINGLE_ARM) == []


def test_a_two_key_entry_is_refused():
    with pytest.raises(ValueError, match="mapping of one name"):
        resolve_teleop_devices(
            {"teleop": [{"spacemouse": {}, "glove": {}}]},
            supported=("spacemouse", "glove"),
        )


def test_no_device_is_named_in_the_wrapper_stack():
    import inspect

    from rlinf.envs.real import wrappers

    source = inspect.getsource(wrappers)
    for device in ("spacemouse", "glove", "gello_joint", "pico"):
        assert device not in source, f"the wrapper stack still names {device!r}"


def test_a_leader_arm_reads_the_joint_convention_from_the_env():
    from rlinf.envs.real.wrappers.teleop.builder import EnvFacts
    from rlinf.robotics.actions import ActionKind
    from rlinf.robotics.parts.teleop import TeleopDevice

    facts = EnvFacts(
        layout={"left.arm": slice(0, 7), "left.end_effector": slice(7, 8)},
        kinds={
            "left.arm": ActionKind.JOINT_DELTA,
            "left.end_effector": ActionKind.GRIPPER,
        },
        joint_action_scale=0.25,
    )
    entry = TeleopDevice.named("gello_joint").from_config(
        {"left_gello_port": "/dev/left"}, {"drives": "left"}, facts
    )

    assert entry.device.use_delta is True
    assert entry.device.action_scale == 0.25


def test_an_entry_option_wins_over_the_env_default():
    from rlinf.envs.real.wrappers.teleop.builder import EnvFacts
    from rlinf.robotics.actions import ActionKind
    from rlinf.robotics.parts.teleop import TeleopDevice

    facts = EnvFacts(
        layout={"left.arm": slice(0, 7), "left.end_effector": slice(7, 8)},
        kinds={
            "left.arm": ActionKind.JOINT_DELTA,
            "left.end_effector": ActionKind.GRIPPER,
        },
        joint_action_scale=0.25,
    )
    entry = TeleopDevice.named("gello_joint").from_config(
        {"left_gello_port": "/dev/left"},
        {"drives": "left", "action_scale": 0.5},
        facts,
    )

    assert entry.device.action_scale == 0.5
    assert entry.device.use_delta is True


def test_a_failed_teleop_connect_leaves_no_device_open():
    from rlinf.robotics.actions import ActionKind
    from rlinf.robotics.parts.teleop import TeleopEntry, TeleopGroup, TeleopPart

    log: list[str] = []

    class Device(TeleopPart):
        def __init__(self, tag, fail=False):
            self.tag, self.fail = tag, fail

        def _open(self):
            if self.fail:
                raise RuntimeError("cable unplugged")
            log.append(f"open:{self.tag}")
            return self.tag

        def _release(self, device):
            log.append(f"close:{self.tag}")

        PRODUCES = {
            "arm": ActionKind.CARTESIAN_DELTA,
            "end_effector": ActionKind.GRIPPER,
        }

        @property
        def observation_features(self):
            return {}

        def get_observation(self):
            return {}

        def action(self, reading, context):
            from rlinf.robotics.parts.teleop import TeleopAction

            return TeleopAction()

    kinds = {
        f"{side}.{part}": kind
        for side in ("left", "right")
        for part, kind in (
            ("arm", ActionKind.CARTESIAN_DELTA),
            ("end_effector", ActionKind.GRIPPER),
        )
    }
    group = TeleopGroup(
        [
            TeleopEntry(Device("first"), drives="left"),
            TeleopEntry(Device("second", fail=True), drives="right"),
        ],
        available=kinds,
    )

    with pytest.raises(RuntimeError, match="cable unplugged"):
        group.connect()

    assert log == ["open:first", "close:first"], (
        f"the device opened before the failure was left open: {log}"
    )


def test_the_glove_reads_the_key_the_shipped_configs_set():
    from rlinf.robotics.parts.teleop import TeleopDevice

    glove = TeleopDevice.named("glove")
    cfg = {
        "glove_config": {
            "left_port": "/dev/ttyACM7",
            "right_port": "/dev/ttyACM8",
            "frequency": 90,
            "config_file": "/etc/glove.json",
        }
    }

    device = glove.from_config(cfg, {}, None).device
    assert device._left_port == "/dev/ttyACM7"
    assert device._right_port == "/dev/ttyACM8"
    assert device._frequency == 90
    assert device._config_file == "/etc/glove.json"

    # Per-entry options override shared device configuration.
    overridden = glove.from_config(cfg, {"left_port": "/dev/override"}, None).device
    assert overridden._left_port == "/dev/override"

    # The documented default applies when the option is omitted.
    assert glove.from_config({}, {}, None).device._left_port == "/dev/ttyACM0"


def test_every_shipped_teleop_config_key_is_one_a_device_reads():
    import pathlib
    import re

    from rlinf.robotics.parts.teleop import TeleopDevice

    # Every device reads its own config, so scan the whole devices package.
    source = "".join(
        path.read_text()
        for path in pathlib.Path("rlinf/robotics/parts/teleop").glob("*.py")
    )
    read = set(re.findall(r'cfg\.get\(\s*f?"([a-z_{}]+)"', source))
    # A key built per side, like "{drives}_gello_port", covers both sides.
    read |= {
        f"{side}_{key.split('}_', 1)[1]}"
        for key in read
        if key.startswith("{")
        for side in ("left", "right")
    }

    configured: set[str] = set()
    wrapper_keys = {
        "keyboard_reward_wrapper",
        "keyboard_wrapper_kwargs",
        "keyboard_intervention_wrapper",
    }
    for path in pathlib.Path("examples/embodiment/config").glob("*.yaml"):
        text = path.read_text()
        if "teleop:" not in text:
            continue
        configured |= {
            key
            for key in re.findall(r"^\s{4}([a-z_]+):", text, re.MULTILINE)
            if key.split("_")[0] in TeleopDevice.names() and key not in wrapper_keys
        }

    unread = sorted(configured - read)
    assert not unread, f"configs set {unread}, which no teleop device reads"


def test_a_teleop_device_is_one_class_that_registers_itself():
    from rlinf.robotics.parts.teleop import TeleopDevice

    # The shipped devices are all registered. Asserting a superset rather than
    # an exact list keeps adding one from editing this test.
    assert set(TeleopDevice.names()) >= {
        "gello",
        "gello_joint",
        "glove",
        "pico",
        "so101_leader",
        "spacemouse",
    }

    # One class answers all three questions: hardware, mapping, and config.
    for name in TeleopDevice.names():
        device_cls = TeleopDevice.named(name)
        assert issubclass(device_cls, TeleopDevice)
        assert callable(device_cls._open)
        assert callable(device_cls.get_observation)
        assert callable(device_cls.action)
        assert callable(device_cls.from_config)
        assert callable(device_cls.streamer)
        assert device_cls.PRODUCES, f"{name} fills no action part"

    # Only a device that bypasses step overrides the streamer hook.
    quiet = [
        name
        for name in TeleopDevice.names()
        if "streamer" not in vars(TeleopDevice.named(name))
    ]
    assert "gello_joint" not in quiet, "the one device that streams must say so"
    assert set(quiet) >= {"gello", "glove", "pico", "so101_leader", "spacemouse"}

    with pytest.raises(ValueError, match="Unknown teleop device"):
        TeleopDevice.named("no_such_device")

    # Duplicate names fail deterministically instead of depending on import order.
    with pytest.raises(ValueError, match="already registered"):

        @TeleopDevice.register("pico")
        class Second(TeleopDevice):
            @classmethod
            def entry(cls, cfg, options, facts):
                raise AssertionError("never built")


def test_every_env_only_offers_teleop_devices_that_exist():
    from rlinf.envs.real.dosw1.base import DOSW1Env
    from rlinf.envs.real.franka.base import FrankaEnv
    from rlinf.envs.real.franka.dual_base import DualFrankaEnv
    from rlinf.envs.real.xsquare.base import Turtle2Env
    from rlinf.robotics.parts.teleop import TeleopDevice

    known = set(TeleopDevice.names())
    for env_cls in (FrankaEnv, DualFrankaEnv, Turtle2Env, DOSW1Env):
        offered = set(getattr(env_cls, "TELEOP", ()))
        unknown = sorted(offered - known)
        assert not unknown, f"{env_cls.__name__} offers {unknown}, which do not exist"


def test_the_streamer_comes_from_the_registry_not_the_stack():
    from rlinf.envs.real.wrappers.teleop.builder import EnvFacts
    from rlinf.robotics.parts.teleop import TeleopDevice

    quiet = EnvFacts(layout={}, kinds={}, direct_stream=False)
    assert TeleopDevice.named("gello_joint").streamer({}, quiet, []) is None


# Environment action declarations


def _declared(cls, **attrs):
    """Return the action parts declared by an environment class."""
    return cls.action_parts(SimpleNamespace(**attrs))


def test_every_env_declares_parts_that_tile_its_action():
    from rlinf.envs.real.dosw1.base import DOSW1Env
    from rlinf.envs.real.franka.base import FrankaEnv
    from rlinf.envs.real.gim_arm.base import GimArmEnv
    from rlinf.envs.real.xsquare.base import Turtle2Env

    cases = [
        (7, _declared(FrankaEnv, _is_hand=False)),
        (
            12,
            _declared(
                FrankaEnv, _is_hand=True, _ee_interface=SimpleNamespace(action_dim=6)
            ),
        ),
        (7, _declared(GimArmEnv)),
        (7, _declared(Turtle2Env, config=SimpleNamespace(use_arm_ids=[1]))),
        (14, _declared(Turtle2Env, config=SimpleNamespace(use_arm_ids=[0, 1]))),
        (14, _declared(DOSW1Env)),
    ]
    for width, parts in cases:
        assert sum(part.width for part in parts) == width


def test_a_two_armed_robot_names_both_arms():
    from rlinf.envs.real.xsquare.base import Turtle2Env

    parts = _declared(Turtle2Env, config=SimpleNamespace(use_arm_ids=[0, 1]))

    assert [part.name for part in parts] == [
        "left.arm",
        "left.end_effector",
        "right.arm",
        "right.end_effector",
    ]


def test_two_arms_of_the_same_width_can_mean_different_things():
    from rlinf.envs.real.franka.base import FrankaEnv
    from rlinf.envs.real.gim_arm.base import GimArmEnv
    from rlinf.robotics.actions import ActionKind

    franka_arm = _declared(FrankaEnv, _is_hand=False)[0]
    gim_arm = _declared(GimArmEnv)[0]

    assert franka_arm.width == gim_arm.width == 6
    assert franka_arm.kind is ActionKind.CARTESIAN_DELTA
    assert gim_arm.kind is ActionKind.JOINT_POSITION


def test_an_env_that_declares_nothing_cannot_be_teleoperated():
    from rlinf.envs.real.wrappers.teleop.layout import action_spec

    class Bare:
        unwrapped = None

        def get_wrapper_attr(self, name):
            raise AttributeError(name)

    Bare.unwrapped = Bare()
    with pytest.raises(AttributeError, match="does not declare action_parts"):
        action_spec(Bare())


def test_a_declaration_that_does_not_tile_the_action_is_refused():
    import gymnasium as gym

    from rlinf.envs.real.wrappers.teleop.layout import action_spec
    from rlinf.robotics.actions import ActionKind, ActionPart

    class Wrong:
        action_space = gym.spaces.Box(-1, 1, (7,), np.float32)

        def action_parts(self):
            return (ActionPart("arm", 6, ActionKind.CARTESIAN_DELTA),)

        def get_wrapper_attr(self, name):
            return getattr(self, name)

    env = Wrong()
    env.unwrapped = env
    with pytest.raises(ValueError, match="declares parts covering 6"):
        action_spec(env)


def test_a_device_that_means_something_else_is_refused():
    from rlinf.robotics.actions import ActionKind
    from rlinf.robotics.parts.teleop import SpaceMouse, TeleopEntry, TeleopGroup

    joint_arm = {
        "arm": ActionKind.JOINT_POSITION,
        "end_effector": ActionKind.GRIPPER,
    }
    # A SpaceMouse drives Cartesian deltas, so it cannot fill a joint arm.
    with pytest.raises(ValueError, match="mean different things"):
        TeleopGroup([TeleopEntry(SpaceMouse())], available=joint_arm)


# Teleoperation compatibility of shipped configurations


def _task_classes():
    """Return registered real-world task classes keyed by Gym ID.

    The robot packages are read from the list the env package itself loads,
    so a newly added robot is covered here without editing this helper.
    """
    import importlib

    import rlinf.envs.real as real

    classes = {}
    for name in real._ROBOT_PACKAGES:
        module = importlib.import_module(name, real.__name__)
        classes.update(getattr(module, "TASKS", {}))
    return classes


def _env_configs():
    """Return each shipped environment config and its Gym ID.

    A run config layers an ``env/<name>`` file into ``env.train`` or
    ``env.eval`` and may override the device there, so both halves are read.
    """
    import yaml

    roots = [_ROOT / "examples", _ROOT / "evaluations"]
    env_files = {}
    for root in roots:
        for path in root.rglob("env/*.yaml"):
            try:
                doc = yaml.safe_load(path.read_text()) or {}
            except yaml.YAMLError:
                continue
            env_id = (doc.get("init_params") or {}).get("id")
            # Simulated environments have no real-world teleoperation contract.
            if env_id and str(doc.get("env_type", "")) in ("real", "realworld"):
                env_files[path.stem] = (path, doc, str(env_id))

    for path, doc, env_id in env_files.values():
        yield path, doc, env_id

    pattern = re.compile(r"env/([\w-]+)@env\.(train|eval)")
    for root in roots:
        for path in root.rglob("*.yaml"):
            if path.parent.name == "env":
                continue
            try:
                doc = yaml.safe_load(path.read_text()) or {}
            except yaml.YAMLError:
                continue
            for entry in doc.get("defaults") or []:
                match = pattern.search(str(entry))
                if not match:
                    continue
                base = env_files.get(match.group(1))
                if base is None:
                    continue
                section = ((doc.get("env") or {}).get(match.group(2))) or {}
                if any(key in section for key in ("teleop", "teleop_device")):
                    yield path, section, base[2]


def test_shipped_configs_name_devices_their_env_can_drive():
    classes = _task_classes()
    offenders = []
    for path, section, env_id in _env_configs():
        env_cls = classes.get(env_id)
        if env_cls is None:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                resolve_teleop_devices(
                    section,
                    supported=getattr(env_cls, "TELEOP", ()),
                    default=getattr(env_cls, "TELEOP_DEFAULT", NO_DEVICE),
                )
        except ValueError as error:
            offenders.append(f"{path.relative_to(_ROOT)} ({env_id}): {error}")

    assert offenders == []


def test_every_shipped_config_names_a_registered_task():
    classes = _task_classes()
    unknown = sorted(
        {
            env_id
            for _, _, env_id in _env_configs()
            if env_id not in classes and env_id.endswith("-v1")
        }
    )

    assert unknown == []


def test_the_retired_env_type_still_resolves_and_warns():
    from rlinf.envs import SupportedEnvType

    assert SupportedEnvType("real") is SupportedEnvType.REAL
    with pytest.warns(DeprecationWarning, match="'realworld' is retired"):
        assert SupportedEnvType("realworld") is SupportedEnvType.REAL


def test_no_worker_compares_the_env_type_to_a_bare_string():
    offenders = []
    for path in (_ROOT / "rlinf" / "workers").rglob("*.py"):
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if re.search(r'env_type\s*[!=]=\s*["\']', line):
                offenders.append(f"{path.relative_to(_ROOT)}:{number}")

    assert offenders == []


def _resolved(doc, value):
    """Resolve a ``${a.b.c}`` interpolation within one document."""
    if not isinstance(value, str) or not value.startswith("${"):
        return value
    node = doc
    for key in value[2:-1].split("."):
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return _resolved(doc, node)


def _merged_sections(path, doc):
    """Yield each environment section after Hydra-style composition.

    A section layers its own keys over the ``env/<name>`` file its ``defaults``
    names, and ``override_cfg`` merges key by key rather than wholesale. What
    the section leaves out it inherits, which is how a task ends up on a
    different robot than the one it trained on.
    """
    import yaml

    pattern = re.compile(r"env/([\w-]+)@env\.(train|eval)")
    for entry in doc.get("defaults") or []:
        match = pattern.search(str(entry))
        if not match:
            continue
        base_path = path.parent / "env" / f"{match.group(1)}.yaml"
        if not base_path.exists():
            continue
        try:
            base = yaml.safe_load(base_path.read_text()) or {}
        except yaml.YAMLError:
            continue
        section = ((doc.get("env") or {}).get(match.group(2))) or {}
        merged = {**base, **section}
        merged["override_cfg"] = {
            **(base.get("override_cfg") or {}),
            **(section.get("override_cfg") or {}),
        }
        yield match.group(2), merged, str((base.get("init_params") or {}).get("id"))


def test_shipped_configs_give_the_policy_the_action_width_it_expects():
    import yaml

    from rlinf.envs.real.franka.base import FrankaEnv

    classes = _task_classes()
    offenders = []
    for path in (_ROOT / "examples").rglob("*.yaml"):
        if path.parent.name == "env":
            continue
        try:
            doc = yaml.safe_load(path.read_text()) or {}
        except yaml.YAMLError:
            continue
        if not isinstance(doc, dict):
            continue
        # A model section may itself be an interpolation, e.g. ${actor.model}.
        model = _resolved(doc, (doc.get("rollout") or {}).get("model"))
        if not isinstance(model, dict):
            continue
        action_dim = _resolved(doc, model.get("action_dim"))
        if not isinstance(action_dim, int):
            continue
        for name, section, env_id in _merged_sections(path, doc):
            # This legacy action layout applies only to single-arm Franka.
            env_cls = classes.get(env_id)
            if env_cls is None or not issubclass(env_cls, FrankaEnv):
                continue
            hardware_configs = [
                config
                for group in doc.get("cluster", {}).get("node_groups", [])
                if group.get("hardware", {}).get("type") == "Franka"
                for config in group["hardware"].get("configs", [])
            ]
            end_effector = str(
                hardware_configs[0].get("end_effector_type", "franka_gripper")
                if hardware_configs
                else "franka_gripper"
            )
            from rlinf.robotics.robots.franka import FrankaRobot

            hardware = hardware_configs[0] if hardware_configs else {}
            driver = FrankaRobot.end_effector_class(
                backend=hardware.get("backend"),
                gripper_type=hardware.get("gripper_type"),
                end_effector_type=hardware.get("end_effector_type"),
            )
            parts = FrankaEnv.action_parts(
                SimpleNamespace(_is_hand=driver.is_hand, _ee_interface=driver)
            )
            width = sum(part.width for part in parts)
            if width != action_dim:
                offenders.append(
                    f"{path.relative_to(_ROOT)} env.{name}: {end_effector} gives "
                    f"{width}, model wants {action_dim}"
                )

    assert offenders == []


def test_direct_stream_gello_opens_one_reader_per_port(monkeypatch):
    """The streamer reuses the group's devices instead of reopening ports."""
    import numpy as np

    from rlinf.envs.real.wrappers.teleop import builder as builder_module
    from rlinf.robotics.parts.teleop import (
        GelloJoint,
        TeleopDevice,
        TeleopEntry,
        gello_joint,
    )

    opened = []

    class FakeExpert:
        def __init__(self, port):
            opened.append(port)
            self.ready = True

        def get_action(self):
            return np.zeros(7), np.zeros(1)

        def close(self):
            pass

    # The reader lives in the device module now, so the class is what a test
    # replaces rather than a separate module.
    monkeypatch.setattr(gello_joint, "GelloJointExpert", FakeExpert)

    arms = {side: GelloJoint(port=f"/dev/{side}") for side in ("left", "right")}
    entries = [TeleopEntry(arm, drives=side) for side, arm in arms.items()]
    try:
        for arm in arms.values():
            arm.connect()
        assert opened == ["/dev/left", "/dev/right"]

        facts = builder_module.EnvFacts(layout={}, kinds={}, direct_stream=True)
        streamer = TeleopDevice.named("gello_joint").streamer({}, facts, entries)

        assert opened == ["/dev/left", "/dev/right"], (
            f"the streamer opened more readers: {opened}"
        )
        assert streamer.left_arm is arms["left"]
        assert streamer.right_arm is arms["right"]
    finally:
        for arm in arms.values():
            arm.disconnect()


def _so101_env(robot_info=None, **overrides):
    """Build an SO-101 reach env against the faked lerobot SDK."""
    from rlinf.envs.real.so101 import SO101ReachEnv

    settings = {
        "target_joint_qpos": [0.0] * 5,
        # Do not pace the test at the real 10 Hz control rate.
        "step_frequency": 1000.0,
        # Headless: there is no display to show frames on.
        "enable_camera_player": False,
        "reset_duration": 0.02,
        "reset_joint_speed": 10.0,
    }
    settings.update(overrides)
    return SO101ReachEnv(
        settings,
        env_idx=0,
        robot_info=robot_info
        or _robot_info(
            SO101Config(
                node_rank=0, serial_port="/dev/mock-so101", calibration_id="bench"
            )
        ),
    )


@pytest.mark.placement
def test_so101_env_runs_a_whole_episode_against_a_faked_arm():
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        env = _so101_env(max_num_steps=3)
        try:
            observation, _ = env.reset()
            assert set(observation["state"]) == {
                "arm_joint_position",
                "gripper_position",
            }
            assert observation["state"]["arm_joint_position"].shape == (5,)

            # Five joints plus one gripper opening, which is 0..1 not -1..1.
            assert env.action_space.shape == (6,)
            assert env.action_space.low[5] == pytest.approx(0.0)
            assert env.action_space.high[5] == pytest.approx(1.0)
            assert [(part.name, part.width) for part in env.action_parts()] == [
                ("arm", 5),
                ("end_effector", 1),
            ]

            for _ in range(3):
                _, _, _, truncated, _ = env.step(np.zeros(6, dtype=np.float32))
            assert truncated, "the episode should truncate at max_num_steps"
        finally:
            env.close()


@pytest.mark.placement
def test_so101_env_keeps_its_action_in_radians_across_a_degree_driver():
    """The env speaks radians; only the driver may speak lerobot's units.

    The arm runs in a scheduler worker, so the fake servo bus is not reachable
    from here. Read the action back instead: the driver converts on the way
    out and again on the way in, so a stray conversion anywhere on the env's
    side of the boundary would come back scaled.
    ``test_so101_commands_are_converted_back_to_degrees`` pins the degrees
    themselves, against an arm declared in this process.
    """
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        env = _so101_env()
        try:
            env.reset()
            observation, *_ = env.step(
                np.array([np.pi / 2, 0, 0, 0, 0, 0.25], dtype=np.float32)
            )

            state = observation["state"]
            assert state["arm_joint_position"][0] == pytest.approx(np.pi / 2)
            assert state["gripper_position"][0] == pytest.approx(0.25)
        finally:
            env.close()


@pytest.mark.placement
def test_so101_env_scores_the_distance_to_the_target_configuration():
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        env = _so101_env(target_joint_qpos=[0.0] * 5, reward_threshold=0.05)
        try:
            env.reset()
            _, reward, terminated, _, _ = env.step(np.zeros(6, dtype=np.float32))
            assert reward == pytest.approx(1.0)
            assert terminated

            # The fake arm holds whatever was last written, so a command away
            # from the target is measured as a miss on the next read.
            _, reward, terminated, _, _ = env.step(
                np.array([1.0, 0, 0, 0, 0, 0.0], dtype=np.float32)
            )
            assert reward == pytest.approx(0.0)
            assert not terminated
        finally:
            env.close()


@pytest.mark.placement
def test_so101_env_clips_an_action_to_the_joint_limits():
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        env = _so101_env(joint_limit_low=[-0.5] * 5, joint_limit_high=[0.5] * 5)
        try:
            env.reset()
            observation, *_ = env.step(
                np.array([10.0, -10.0, 0, 0, 0, 2.0], dtype=np.float32)
            )

            state = observation["state"]
            assert state["arm_joint_position"][0] == pytest.approx(0.5)
            assert state["arm_joint_position"][1] == pytest.approx(-0.5)
            # The gripper is clipped into 0..1 before it is scaled.
            assert state["gripper_position"][0] == pytest.approx(1.0)
        finally:
            env.close()


def test_so101_env_runs_without_hardware_when_dummy():
    """A dummy env samples its own space, so it needs no lerobot at all."""
    from rlinf.envs.real.so101 import SO101ReachEnv

    env = SO101ReachEnv({"is_dummy": True, "target_joint_qpos": [0.0] * 5})
    try:
        observation, _ = env.reset()
        assert observation in env.observation_space
        assert env.robot is None
        env.step(np.zeros(6, dtype=np.float32))
    finally:
        env.close()


def test_so101_reach_refuses_a_target_that_is_not_five_joints():
    from rlinf.envs.real.so101 import SO101ReachEnv

    with pytest.raises(ValueError, match="5 arm joints"):
        SO101ReachEnv({"is_dummy": True, "target_joint_qpos": [0.0] * 6})


def test_so101_task_is_registered_with_gymnasium():
    import gymnasium as gym

    import rlinf.envs.real as real

    real.load_tasks()
    assert "SO101ReachEnv-v1" in gym.registry


@pytest.mark.placement
def test_so101_env_resizes_camera_frames_to_the_declared_shape():
    """A camera delivers its native resolution; the space fixes one size."""
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        env = _so101_env(
            robot_info=_robot_info(
                SO101Config(
                    node_rank=0,
                    serial_port="/dev/mock-so101",
                    camera_serials=["MOCK0001"],
                    camera_type="realsense",
                )
            )
        )
        try:
            observation, _ = env.reset()
            frame = observation["frames"]["wrist_1"]
            assert frame.shape == (128, 128, 3)
            assert frame.dtype == np.uint8
            # An observation outside its own space fails Gymnasium's checker.
            assert observation in env.observation_space
        finally:
            env.close()


@pytest.mark.placement
def test_so101_env_omits_frames_entirely_when_no_camera_is_configured():
    """Gymnasium rejects an empty Dict space, so the key is dropped instead."""
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        env = _so101_env()
        try:
            observation, _ = env.reset()
            assert "frames" not in env.observation_space.spaces
            assert "frames" not in observation
            assert observation in env.observation_space
        finally:
            env.close()


@pytest.mark.placement
def test_so101_env_control_can_run_faster_than_camera_capture():
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        env = _so101_env(
            step_frequency=60.0,
            robot_info=_robot_info(
                SO101Config(
                    node_rank=0,
                    serial_port="/dev/mock-so101",
                    camera_serials=["MOCK0001"],
                )
            ),
        )
        try:
            env.reset()
            start = time.monotonic()
            for _ in range(12):
                observation, *_ = env.step(np.zeros(6, dtype=np.float32))
                assert observation in env.observation_space
            # Twelve fresh captures alone would take 0.8 s at the default 15 FPS.
            assert time.monotonic() - start < 0.6
        finally:
            env.close()


@pytest.mark.placement
def test_so101_env_can_connect_without_resetting_then_reset_explicitly():
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        env = _so101_env(reset_joint_qpos=[0.2] * 5, reset_on_init=False)
        try:
            state = env.robot.get_observation()["arm"]
            np.testing.assert_allclose(state["arm_joint_position"], np.zeros(5))
            observation, _ = env.reset()
            np.testing.assert_allclose(
                observation["state"]["arm_joint_position"], [0.2] * 5
            )
        finally:
            env.close()


@pytest.mark.placement
def test_so101_tool_home_uses_the_environment_reset(monkeypatch):
    from robot_mocks import mocked_sdks

    from toolkits.realworld_check.test_so101_env import drive

    commands = iter(["home", "quit"])
    monkeypatch.setattr("builtins.input", lambda _: next(commands))
    with mocked_sdks():
        env = _so101_env(reset_joint_qpos=[0.2] * 5, reset_on_init=False)
        try:
            drive(env, step=0.01, invert=False, reset=False)
            assert env.num_steps == 0
            state = env.robot.get_observation()["arm"]
            np.testing.assert_allclose(state["arm_joint_position"], [0.2] * 5)
        finally:
            env.close()


@pytest.mark.placement
def test_so101_tool_teleop_forwards_gripper_only_and_stationary_commands(monkeypatch):
    from robot_mocks import mocked_sdks

    from toolkits.realworld_check.test_so101_env import teleop

    with mocked_sdks() as made:
        leader = made["lerobot.teleoperators.so_leader"].SO101Leader
        openings = iter([25.0, 50.0, 50.0])

        def get_action(device):
            try:
                opening = next(openings)
            except StopIteration:
                raise KeyboardInterrupt from None
            return {**device.positions, "gripper.pos": opening}

        monkeypatch.setattr(leader, "get_action", get_action)
        env = _so101_env()
        try:
            teleop(env, "/dev/mock-leader", "leader")
            assert env.num_steps == 3
            state = env.robot.get_observation()["arm"]
            assert state["end_effector"]["state"][0] == pytest.approx(0.5)
        finally:
            env.close()


def _piper_env(robot_info=None, **overrides):
    """Build a Piper reach env against the faked pyAgxArm SDK."""
    from rlinf.envs.real.piper import PiperReachEnv

    settings = {
        "target_joint_qpos": [0.0] * 6,
        # Do not pace the test at the real 10 Hz control rate.
        "step_frequency": 1000.0,
        # Headless: there is no display to show frames on.
        "enable_camera_player": False,
    }
    settings.update(overrides)
    return PiperReachEnv(
        settings,
        env_idx=0,
        robot_info=robot_info or _robot_info(PiperConfig(node_rank=0)),
    )


def test_a_hardware_type_resolves_without_importing_the_robot_package():
    """The lookup loads the modules that register hardware configs.

    In a subprocess, because this one has already imported them.
    """
    import pathlib
    import subprocess
    import sys

    program = (
        "import sys\n"
        "from rlinf.scheduler.cluster.config import ClusterConfig\n"
        "from rlinf.scheduler.hardware import NodeHardwareConfig\n"
        "assert not [m for m in sys.modules if m.startswith('rlinf.robotics')]\n"
        "assert not NodeHardwareConfig._hardware_config_registry\n"
        "from omegaconf import OmegaConf\n"
        "cfg = OmegaConf.create({'num_nodes': 1, 'component_placement':"
        " {'env': {'node_group': 'f', 'placement': 0}}, 'node_groups':"
        " [{'label': 'f', 'node_ranks': 0, 'hardware': {'type': 'Franka',"
        " 'configs': [{'robot_ip': '0.0.0.0', 'node_rank': 0}]}}]})\n"
        "hw = ClusterConfig.from_dict_cfg(cfg).node_groups[0].hardware\n"
        "print(type(hw.configs[0]).__name__)\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        cwd=str(pathlib.Path(__file__).resolve().parents[2]),
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "FrankaConfig", done.stdout


def test_entry_points_reach_realworldenv_through_its_package():
    """``rlinf.envs.real`` registers its Gymnasium tasks from ``__getattr__``.

    Reaching ``RealWorldEnv`` at ``rlinf.envs.real.env`` bypasses that.
    """
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parents[2]
    offenders = []
    for path in list((root / "examples").rglob("*.py")) + list(
        (root / "toolkits").rglob("*.py")
    ):
        if re.search(r"^from rlinf\.envs\.real\.env import", path.read_text(), re.M):
            offenders.append(str(path.relative_to(root)))

    assert offenders == [], (
        "import RealWorldEnv from rlinf.envs.real, not rlinf.envs.real.env: "
        f"{offenders}"
    )


@pytest.mark.placement
def test_piper_env_runs_a_whole_episode_against_a_faked_arm():
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        env = _piper_env(max_num_steps=3)
        try:
            observation, _ = env.reset()
            assert set(observation["state"]) == {
                "arm_joint_position",
                "tcp_pose",
                "gripper_position",
            }
            assert observation["state"]["arm_joint_position"].shape == (6,)
            assert observation["state"]["tcp_pose"].shape == (7,)

            # The gripper axis is 0..1, not -1..1.
            assert env.action_space.shape == (7,)
            assert env.action_space.low[6] == pytest.approx(0.0)
            assert env.action_space.high[6] == pytest.approx(1.0)
            assert [(part.name, part.width) for part in env.action_parts()] == [
                ("arm", 6),
                ("end_effector", 1),
            ]

            for _ in range(3):
                _, _, _, truncated, _ = env.step(np.zeros(7, dtype=np.float32))
            assert truncated, "the episode should truncate at max_num_steps"
        finally:
            env.close()


@pytest.mark.placement
def test_piper_env_commands_reach_the_arm_in_radians():
    """pyAgxArm takes radians, so nothing on this path rescales them.

    The arm runs in a scheduler worker, so the fake CAN session is not
    reachable from here; the action is read back instead.
    ``test_piper_commands_go_out_in_radians_and_are_held_to_the_travel`` and
    ``test_piper_gripper_rides_the_arm_connection`` pin what reaches the SDK,
    against an arm declared in this process.
    """
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        env = _piper_env()
        try:
            env.reset()
            observation, *_ = env.step(
                np.array([0.5, 1.0, -1.0, 0, 0, 0, 0.25], dtype=np.float32)
            )

            state = observation["state"]
            assert state["arm_joint_position"] == pytest.approx(
                [0.5, 1.0, -1.0, 0.0, 0.0, 0.0]
            )
            assert state["gripper_position"][0] == pytest.approx(0.25)
        finally:
            env.close()


@pytest.mark.placement
def test_piper_env_clips_an_action_to_the_joint_limits():
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        env = _piper_env(joint_limit_low=[-0.5] * 6, joint_limit_high=[0.5] * 6)
        try:
            env.reset()
            observation, *_ = env.step(
                np.array([10.0, -10.0, 0, 0, 0, 0, 2.0], dtype=np.float32)
            )

            joints = observation["state"]["arm_joint_position"]
            assert joints[0] == pytest.approx(0.5)
            # The env bound is looser than the arm's, so the arm holds joint 2
            # at the travel its firmware accepts.
            assert joints[1] == pytest.approx(0.0)
            assert observation["state"]["gripper_position"][0] == pytest.approx(1.0)
        finally:
            env.close()


@pytest.mark.placement
def test_piper_env_scores_the_distance_to_the_target_configuration():
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        env = _piper_env(target_joint_qpos=[0.0] * 6, reward_threshold=0.05)
        try:
            env.reset()
            _, reward, terminated, _, _ = env.step(np.zeros(7, dtype=np.float32))
            assert reward == pytest.approx(1.0)
            assert terminated

            # The fake arm holds whatever was last written, so a command away
            # from the target is measured as a miss on the next read.
            _, reward, terminated, _, _ = env.step(
                np.array([1.0, 0, 0, 0, 0, 0, 0.0], dtype=np.float32)
            )
            assert reward == pytest.approx(0.0)
            assert not terminated
        finally:
            env.close()


def test_piper_env_runs_without_hardware_when_dummy():
    """A dummy env samples its own space, so it needs no pyAgxArm at all."""
    from rlinf.envs.real.piper import PiperReachEnv

    env = PiperReachEnv({"is_dummy": True, "target_joint_qpos": [0.0] * 6})
    try:
        observation, _ = env.reset()
        assert observation in env.observation_space
        assert env.robot is None
        env.step(np.zeros(7, dtype=np.float32))
    finally:
        env.close()


@pytest.mark.placement
def test_piper_env_without_a_gripper_has_a_six_wide_action():
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        env = _piper_env(
            robot_info=_robot_info(PiperConfig(node_rank=0, with_gripper=False))
        )
        try:
            observation, _ = env.reset()
            assert "gripper_position" not in observation["state"]
            assert [part.name for part in env.action_parts()] == ["arm"]
            assert env.action_space.shape == (6,)
            observation, *_ = env.step(np.zeros(6, dtype=np.float32))
            assert observation in env.observation_space
        finally:
            env.close()


def test_piper_reach_refuses_a_target_that_is_not_six_joints():
    from rlinf.envs.real.piper import PiperReachEnv

    with pytest.raises(ValueError, match="6 arm joints"):
        PiperReachEnv({"is_dummy": True, "target_joint_qpos": [0.0] * 5})


def test_piper_task_is_registered_with_gymnasium():
    import gymnasium as gym

    import rlinf.envs.real as real

    real.load_tasks()
    assert "PiperReachEnv-v1" in gym.registry


@pytest.mark.placement
def test_piper_env_resizes_camera_frames_to_the_declared_shape():
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        env = _piper_env(
            robot_info=_robot_info(
                PiperConfig(
                    node_rank=0, camera_serials=["MOCK0001"], camera_type="realsense"
                )
            )
        )
        try:
            observation, _ = env.reset()
            frame = observation["frames"]["wrist_1"]
            assert frame.shape == (128, 128, 3)
            assert frame.dtype == np.uint8
            assert observation in env.observation_space
        finally:
            env.close()


def _yam_env(**overrides):
    """Build a dual YAM env against the faked i2rt and RealSense SDKs."""
    from rlinf.envs.real.yam import DualYamJointEnv
    from rlinf.robotics import DualYamConfig

    settings = {
        "joint_limit_min": [[-2.61799, 0.0, 0.0, -1.69297, -1.5708, -2.0944]] * 2,
        "joint_limit_max": [[3.14159, 3.66519, 3.14159, 1.5708, 1.5708, 2.0944]] * 2,
        "max_joint_delta": 0.05,
        # Do not pace the test at the real 30 Hz control rate.
        "step_frequency": 1000.0,
        # The fake cameras capture 64x48.
        "image_height": 48,
        "image_width": 64,
        "manual_episode_control_only": True,
    }
    settings.update(overrides)
    hardware = DualYamConfig(
        node_rank=0,
        left_can_channel="can0",
        right_can_channel="can1",
        camera_serials=["MOCK0001", "MOCK0002", "MOCK0003"],
        disable_validate=True,
    )
    return DualYamJointEnv(settings, robot_info=_robot_info(hardware))


@pytest.mark.placement
def test_yam_env_holds_each_command_to_a_step_per_tick():
    """A far target moves one step a tick; a non-finite one holds both arms."""
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        env = _yam_env()
        try:
            observation, _ = env.reset()
            assert observation["state"]["joint_position"].shape == (14,)
            assert set(observation["frames"]) == {"top_rgb", "left_rgb", "right_rgb"}

            target = np.zeros(14, dtype=np.float32)
            target[[0, 7]] = 0.5
            target[[6, 13]] = 1.5
            observation, _, _, _, info = env.step(target)
            state = observation["state"]["joint_position"]
            assert info["action_clipped"]
            assert state[[0, 7]] == pytest.approx([0.05, 0.05])
            assert state[[6, 13]] == pytest.approx([1.0, 1.0])

            target[2] = np.nan
            observation, _, _, _, info = env.step(target)
            assert info["action_rejected"] == "non_finite_action"
            assert observation["state"]["joint_position"] == pytest.approx(state)
        finally:
            env.close()


@pytest.mark.placement
def test_yam_env_lets_a_joint_past_its_limit_move_back_but_not_further():
    """A joint read past a task limit is not locked out of every command.

    The fake arm rests at zero, so a J2 lower limit of 0.05 starts it outside.
    """
    from robot_mocks import mocked_sdks

    lower = [[-2.61799, 0.05, 0.0, -1.69297, -1.5708, -2.0944]] * 2
    with mocked_sdks():
        env = _yam_env(joint_limit_min=lower)
        try:
            env.reset()
            target = np.zeros(14, dtype=np.float32)
            target[[1, 8]] = -0.5
            observation, _, _, _, info = env.step(target)
            assert info["action_rejected"] is None
            assert observation["state"]["joint_position"][[1, 8]] == pytest.approx(
                [0.0, 0.0]
            )

            target[[1, 8]] = 0.5
            observation, *_ = env.step(target)
            assert observation["state"]["joint_position"][[1, 8]] == pytest.approx(
                [0.05, 0.05]
            )
        finally:
            env.close()


@pytest.mark.placement
@pytest.mark.parametrize("broken_side", ["left", "right"])
def test_a_dropped_pico_holds_both_yam_arms_until_both_grips_release(broken_side):
    """Either controller dropping out holds both arms, even the one still valid.

    Neither arm drives again until both grips have been let go together.
    """
    import time

    from robot_mocks import mocked_sdks
    from robot_mocks.arms import FreeToolKinematics
    from robot_mocks.teleop import pico_packet

    with mocked_sdks():
        from rlinf.envs.real.wrappers import build_stack
        from rlinf.robotics.parts.transports import pico as transport

        zmq = transport.zmq
        base = _yam_env()
        base.arm_kinematics = lambda side: FreeToolKinematics()
        env = build_stack(
            base,
            {
                "teleop": [{"pico": {"drives": "left"}}, {"pico": {"drives": "right"}}],
                "no_gripper": False,
                "pico": {"zmq_addr": "tcp://x:1"},
                "yam_episode": {"wait_for_record_button": False},
            },
        )

        def tick(frame):
            zmq.frame = frame
            time.sleep(0.05)
            return env.step(np.zeros(14, dtype=np.float32))[-1]

        held = {"grip": 1.0}
        broken = {"orientation": [0.0, 0.0, 0.0, 0.0]}
        healthy_side = "right" if broken_side == "left" else "left"
        moved = {**held, "position": [0.0, 0.0, -0.1]}
        try:
            zmq.frame = pico_packet()
            env.reset()
            assert tick(pico_packet())["pico_fault"] is None
            assert tick(pico_packet(**{healthy_side: held}))[healthy_side]

            measured = base.get_joint_positions().copy()
            info = tick(
                pico_packet(
                    {"X": True, "A": True},
                    **{broken_side: broken, healthy_side: moved},
                )
            )
            assert info["pico_fault"]
            assert not info[healthy_side]
            assert np.allclose(info["accepted_action"], measured)

            info = tick(pico_packet(**{broken_side: broken, healthy_side: moved}))
            assert not info[healthy_side]
            assert np.allclose(info["accepted_action"], measured)

            info = tick(pico_packet(**{healthy_side: moved}))
            assert info["pico_fault"] is not None and not info[healthy_side]

            assert tick(pico_packet())["pico_fault"] is None
            assert tick(pico_packet(**{healthy_side: held}))[healthy_side]
        finally:
            env.close()


@pytest.mark.placement
@pytest.mark.parametrize("left_slow", [False, True])
def test_yam_pico_checks_successful_targets_after_both_ik_solves(left_slow):
    import time

    from robot_mocks import mocked_sdks
    from robot_mocks.arms import FreeToolKinematics
    from robot_mocks.teleop import pico_packet

    class SlowKinematics(FreeToolKinematics):
        max_solve_s = 0.03
        slow = False

        def solve(self, target, seed, gripper):
            if self.slow:
                time.sleep(0.15)
            return super().solve(target, seed, gripper)

    with mocked_sdks():
        from rlinf.envs.real.wrappers import build_stack
        from rlinf.robotics.parts.transports import pico as transport

        solvers = {side: SlowKinematics() for side in ("left", "right")}
        base = _yam_env()
        base.arm_kinematics = lambda side: solvers[side]
        env = build_stack(
            base,
            {
                "teleop": [{"pico": {"drives": "left"}}, {"pico": {"drives": "right"}}],
                "no_gripper": False,
                "pico": {"zmq_addr": "tcp://x:1", "max_tick_s": 0.1},
                "yam_episode": {"wait_for_record_button": False},
            },
        )

        def tick(frame):
            transport.zmq.frame = frame
            time.sleep(0.05)
            return env.step(np.zeros(14, dtype=np.float32))[-1]

        held = {"grip": 1.0}
        moved = {**held, "position": [0.0, 0.0, -0.1]}
        try:
            transport.zmq.frame = pico_packet()
            env.reset()
            assert tick(pico_packet())["pico_fault"] is None
            assert tick(pico_packet(left=held, right=held))["left"]
            measured = base.get_joint_positions().copy()
            solvers["left"].slow = left_slow
            solvers["right"].slow = True
            info = tick(pico_packet(left=moved, right=moved))
            assert info["right_ik_fault"] == "solve_timeout"
            if left_slow:
                assert info["left_ik_fault"] == "solve_timeout"
                assert info["pico_fault"] is None
            else:
                assert info["pico_fault"] == "left:stale_after_ik"
                assert not info["left"] and not info["right"]
            assert np.allclose(info["accepted_action"], measured)
        finally:
            env.close()


def test_so101_leader_reports_radians_and_a_zero_to_one_grip():
    """The leader is the same servos as the follower, so it converts alike."""
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        from rlinf.robotics.parts.teleop import SO101Leader

        leader = SO101Leader(port="/dev/mock-leader", calibration_id="lead")
        leader.connect()
        try:
            leader._device.positions.update(
                {"shoulder_pan.pos": 90.0, "gripper.pos": 40.0}
            )
            reading = leader.get_observation()
            assert reading["joint_position"][0] == pytest.approx(np.pi / 2)
            assert reading["grip"][0] == pytest.approx(0.4)
            # It never reaches lerobot's interactive calibration.
            assert leader._device.calibrate_calls == 0
        finally:
            leader.disconnect()


def test_so101_leader_releases_the_servo_bus():
    """lerobot spells release ``disconnect``, which the base class does not try.

    The device overrides ``_release`` for exactly that reason. Without this the
    serial port would leak on every disconnect and nothing would say so.
    """
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        from rlinf.robotics.parts.teleop import SO101Leader

        leader = SO101Leader(port="/dev/mock-leader")
        leader.connect()
        handle = leader._device
        assert handle.is_connected
        leader.disconnect()
        assert not handle.is_connected


def test_so101_leader_refuses_an_uncalibrated_arm():
    from robot_mocks import mocked_sdks

    with mocked_sdks() as made:
        from rlinf.robotics.parts.teleop import SO101Leader

        fake = made["lerobot.teleoperators.so_leader"].SO101Leader
        fake.calibrated = False
        try:
            leader = SO101Leader(port="/dev/mock-leader")
            with pytest.raises(RuntimeError, match="no calibration") as raised:
                leader.connect()
            # The refusal has to be actionable: the operator should be able to
            # copy a command out of it rather than go looking for one.
            assert "--calibrate" in str(raised.value)
        finally:
            fake.calibrated = True


def test_so101_leader_calibrates_when_the_caller_asks():
    """The standalone entry point has a terminal, so it may opt in."""
    from robot_mocks import mocked_sdks

    with mocked_sdks() as made:
        from rlinf.robotics.parts.teleop import SO101Leader

        fake = made["lerobot.teleoperators.so_leader"].SO101Leader
        fake.calibrated = False
        try:
            leader = SO101Leader(port="/dev/mock-leader", calibrate=True)
            # The same arm that refuses above now opens, because calibration
            # is allowed to run.
            leader.connect()
            assert leader.is_connected
            leader.disconnect()
        finally:
            fake.calibrated = True


def test_so101_leader_only_drives_once_the_operator_moves_it():
    """A leader resting in its holder must not take control from the policy."""
    from rlinf.robotics.parts.teleop import SO101Leader

    binding = SO101Leader(port="/dev/unused", movement_epsilon=0.01)
    at_rest = {"joint_position": np.zeros(5), "grip": np.array([0.0])}
    context = {"joint_positions": np.zeros((1, 5))}

    assert not binding.action(at_rest, context).driving

    moved = {"joint_position": np.array([0.5, 0, 0, 0, 0]), "grip": np.array([0.7])}
    sample = binding.action(moved, context)
    assert sample.driving
    # The leader's pose is the follower's target, joint for joint.
    assert sample.parts["arm"] == pytest.approx(moved["joint_position"])
    # And the grip stays on the 0..1 axis the SO-101 env opens over.
    assert sample.parts["end_effector"][0] == pytest.approx(0.7)


@pytest.mark.placement
def test_so101_env_is_driven_by_its_leader():
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        import gymnasium as gym

        import rlinf.envs.real as real

        real.load_tasks()
        env = gym.make(
            "SO101ReachEnv-v1",
            override_cfg={
                "step_frequency": 1000.0,
                "enable_camera_player": False,
            },
            worker_info=None,
            env_idx=0,
            env_cfg={
                "teleop": [{"so101_leader": {"port": "/dev/mock-leader"}}],
                "no_gripper": False,
                "use_relative_frame": False,
            },
            robot_info=_robot_info(
                SO101Config(
                    node_rank=0, serial_port="/dev/mock-so101", calibration_id="bench"
                )
            ),
        )
        try:
            env.reset()
            policy_action = np.zeros(6, dtype=np.float32)

            observation, *_ = env.step(policy_action)
            assert observation["state"]["arm_joint_position"] == pytest.approx(
                np.zeros(5)
            )

            leader = list(env.get_wrapper_attr("device").group.devices)[0]
            leader._device.positions.update(
                {"shoulder_pan.pos": 45.0, "gripper.pos": 80.0}
            )
            observation, *_ = env.step(policy_action)
            assert observation["state"]["arm_joint_position"][0] == pytest.approx(
                np.deg2rad(45.0), abs=1e-4
            )
            assert observation["state"]["gripper_position"][0] == pytest.approx(0.8)
        finally:
            env.close()


@pytest.mark.parametrize(
    "path",
    [
        "evaluations/realworld/realworld_pnp_eval.yaml",
        "evaluations/realworld/realworld_pnp_eval_dreamzero.yaml",
        "evaluations/realworld/realworld_pnp_eval_pi05_sft_RTC.yaml",
        "examples/embodiment/config/realworld_pnp_dagger_openpi.yaml",
        "examples/embodiment/config/realworld_pnp_rlpd_cnn_async.yaml",
        "examples/reward/config/realworld_teleop.yaml",
    ],
)
def test_pnp_examples_discover_cameras_without_serial_placeholders(path):
    import yaml
    from robot_mocks import mocked_sdks

    from rlinf.robotics import FrankaConfig
    from rlinf.robotics.parts.cameras import BaseCamera

    root = Path(__file__).resolve().parents[2]
    doc = yaml.safe_load((root / path).read_text())
    entries = [
        config
        for group in doc["cluster"]["node_groups"]
        if group.get("hardware", {}).get("type") == "Franka"
        for config in group["hardware"]["configs"]
    ]
    assert entries
    with mocked_sdks():
        discovered = sorted(BaseCamera.backend("realsense").discover())
        assert discovered
        for entry in entries:
            assert "camera_serials" not in entry
            config = FrankaConfig(**entry)
            resources = RobotDiscovery.registry["Franka"].discovery_cls.enumerate(
                config.node_rank, [config]
            )
            assert resources.infos[0].config.camera_serials == discovered


def test_shipped_realworld_task_overrides_contain_no_hardware_fields():
    import yaml

    load_tasks()
    hardware_fields = {
        field.name
        for registration in RobotDiscovery.registry.values()
        for field in dataclasses.fields(registration.config_cls)
    }
    root = Path(__file__).resolve().parents[2]
    offenders = []

    def check(node, path):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "override_cfg" and isinstance(value, dict):
                    overlap = hardware_fields.intersection(value)
                    if overlap:
                        offenders.append((str(path.relative_to(root)), sorted(overlap)))
                check(value, path)
        elif isinstance(node, list):
            for value in node:
                check(value, path)

    for directory in (
        "examples",
        "evaluations",
        "tests",
    ):
        for path in (root / directory).rglob("*.yaml"):
            check(yaml.safe_load(path.read_text()), path)
    assert offenders == []


@pytest.mark.parametrize(
    "robot_type,env_id,hardware,frames,action_width",
    [
        (
            "Franka",
            "FrankaEnv-v1",
            {"camera_serials": ["MOCK0001"], "end_effector_type": "ruiyan_hand"},
            ["wrist_1"],
            12,
        ),
        (
            "DualFranka",
            "DualFrankaJointEnv-v1",
            {
                "base_camera_serials": ["MOCK0001"],
                "left_camera_serials": ["MOCK0002"],
                "right_camera_serials": [],
            },
            ["base_0_rgb", "left_wrist_0_rgb"],
            16,
        ),
        (
            "SO101",
            "SO101ReachEnv-v1",
            {
                "serial_port": "/dev/bench",
                "calibration_id": "bench",
                "camera_serials": ["MOCK0001", "MOCK0002"],
            },
            ["wrist_1", "wrist_2"],
            6,
        ),
        (
            "Piper",
            "PiperReachEnv-v1",
            {"model": "piper_h", "with_gripper": False, "camera_serials": []},
            [],
            6,
        ),
        (
            "GimArm",
            "GimArmPegInsertionEnv-v1",
            {
                "arm_variant": "gim_arm",
                "enable_gripper": False,
                "camera_serials": ["MOCK0001"],
            },
            ["wrist_1"],
            7,
        ),
        (
            "DOSW1",
            "DOSW1PickEnv-v1",
            {"robot_url": "bench", "camera_serials": ["MOCK0001"]},
            ["cam_front"],
            14,
        ),
        ("Turtle2", "ButtonEnv-v1", {"camera_ids": [0, 2]}, ["wrist_1", "wrist_2"], 7),
    ],
)
def test_task_schema_uses_enumerated_hardware_without_changing_it(
    robot_type, env_id, hardware, frames, action_width
):
    from robot_mocks import mocked_sdks

    load_tasks()
    registration = RobotDiscovery.registry[robot_type]
    config = registration.config_cls(node_rank=0, **hardware)
    with mocked_sdks():
        resources = registration.discovery_cls.enumerate(0, [config])
        info = resources.infos[0]
        before = pickle.dumps(info)
        env = gym.make(
            env_id,
            override_cfg={"is_dummy": True, "enable_camera_player": False}
            if robot_type != "Turtle2"
            else {"is_dummy": True},
            worker_info=None,
            robot_info=info,
            env_idx=0,
            env_cfg={
                "teleop": "none",
                "no_gripper": False,
                "use_relative_frame": False,
            },
        )
        try:
            observation, _ = env.reset(seed=3)
            assert sorted(observation.get("frames", {})) == sorted(frames)
            assert env.action_space.shape == (action_width,)
            assert pickle.dumps(info) == before
            assert not set(hardware) & {
                field.name for field in dataclasses.fields(env.unwrapped.config)
            }
            if robot_type == "Piper":
                assert info.model == "Piper"
                assert info.config.model == "piper_h"
        finally:
            env.close()


@pytest.mark.parametrize(
    "field,value",
    [
        ("port", "/dev/wrong-arm"),
        ("serial_port", "/dev/wrong-arm"),
        ("camera_serials", ["wrong-camera"]),
        ("calibration_id", "wrong-id"),
    ],
)
def test_task_overrides_cannot_redirect_the_allocated_robot(field, value):
    from rlinf.envs.real.so101 import SO101ReachEnv

    with pytest.raises(TypeError, match=field):
        SO101ReachEnv({"is_dummy": True, field: value})


def test_env_rejects_wrong_hardware_before_opening_an_arm(monkeypatch):
    from rlinf.envs.real.so101 import SO101ReachEnv
    from rlinf.robotics import PiperConfig, RobotInfo, SO101Robot

    build = Mock()
    monkeypatch.setattr(SO101Robot, "build", build)
    info = RobotInfo(type="Piper", model="Piper", config=PiperConfig(node_rank=0))
    with pytest.raises(TypeError, match="Expected SO101Config"):
        SO101ReachEnv({}, robot_info=info)
    build.assert_not_called()


def test_real_env_requires_a_robot_descriptor(monkeypatch):
    from rlinf.envs.real.so101 import SO101ReachEnv
    from rlinf.robotics import SO101Robot

    build = Mock()
    monkeypatch.setattr(SO101Robot, "build", build)
    with pytest.raises(ValueError, match="Supply robot_info"):
        SO101ReachEnv({})
    build.assert_not_called()


@pytest.mark.parametrize("controller_node_rank", [None, 7])
@pytest.mark.parametrize("compliance", [None, {}, {"max_step": 0.02}])
def test_franka_preserves_hardware_and_placement_at_construction(
    monkeypatch, controller_node_rank, compliance
):
    from rlinf.envs.real.franka import FrankaEnv
    from rlinf.robotics import FrankaConfig, FrankaRobot, RobotInfo

    config = FrankaConfig(
        node_rank=3,
        robot_ip="bench",
        controller_node_rank=controller_node_rank,
        camera_node_rank=5,
        camera_type="zed",
        camera_serials=["camera"],
        end_effector_type="ruiyan_hand",
        end_effector_config={"port": "/dev/hand"},
        compliance=compliance,
    )
    info = RobotInfo(type="Franka", model="Franka", config=config)
    before = pickle.dumps(info)
    build = Mock(side_effect=RuntimeError("stop before opening hardware"))
    monkeypatch.setattr(FrankaRobot, "build", build)
    with pytest.raises(RuntimeError, match="stop before opening hardware"):
        FrankaEnv(
            {},
            robot_info=info,
            worker_info=SimpleNamespace(cluster_node_rank=3, rank=2),
            env_idx=4,
        )
    kwargs = build.call_args.kwargs
    assert kwargs["robot_ip"] == "bench"
    assert kwargs["node_rank"] == (3 if controller_node_rank is None else 7)
    assert kwargs["camera_node_rank"] == 5
    assert kwargs["worker_rank"] == 2
    assert kwargs["env_idx"] == 4
    assert kwargs["end_effector_type"] == "ruiyan_hand"
    assert kwargs["end_effector_config"] == {"port": "/dev/hand"}
    assert kwargs["compliance"] == compliance
    camera = kwargs["cameras"]["wrist_1"]
    assert camera.serial_number == "camera"
    assert camera.camera_type == "zed"
    assert pickle.dumps(info) == before


@pytest.mark.parametrize(
    "shared,left,right,expected_left,expected_right",
    [
        (None, None, None, None, None),
        ({"max_step": 0.02}, None, None, {"max_step": 0.02}, {"max_step": 0.02}),
        (
            {"translational_stiffness": 900},
            {"max_step": 0.02},
            {},
            {"max_step": 0.02},
            {},
        ),
        ({"max_step": 0.02}, {}, None, {}, {"max_step": 0.02}),
    ],
)
def test_dual_franka_preserves_shared_and_per_arm_compliance(
    monkeypatch, shared, left, right, expected_left, expected_right
):
    from rlinf.envs.real.franka.dual_franka_tcp import DualFrankaTCPEnv
    from rlinf.robotics import DualFrankaRobot
    from rlinf.scheduler.hardware import NodeHardwareConfig

    hardware = NodeHardwareConfig(
        type="DualFranka",
        configs=[
            {
                "node_rank": 0,
                "left_robot_ip": "10.0.0.1",
                "right_robot_ip": "10.0.0.2",
                "compliance": shared,
                "left_compliance": left,
                "right_compliance": right,
            }
        ],
    )
    info = _robot_info(pickle.loads(pickle.dumps(hardware.configs[0])))
    before = pickle.dumps(info)
    build = Mock(side_effect=RuntimeError("stop before opening hardware"))
    monkeypatch.setattr(DualFrankaRobot, "build", build)
    with pytest.raises(RuntimeError, match="stop before opening hardware"):
        DualFrankaTCPEnv({}, robot_info=info)
    assert build.call_args.kwargs["left_compliance"] == expected_left
    assert build.call_args.kwargs["right_compliance"] == expected_right
    assert pickle.dumps(info) == before


@pytest.mark.parametrize("has_robot", [False, True])
def test_worker_passes_robot_descriptors_and_allows_dummy_cpu_placement(has_robot):
    from omegaconf import OmegaConf

    from rlinf.envs.real import RealWorldEnv
    from rlinf.robotics import RobotInfo, SO101Config
    from rlinf.scheduler.hardware import HardwareInfo

    allocated = (
        RobotInfo(
            type="SO101",
            model="SO101",
            config=SO101Config(node_rank=0, camera_serials=["camera"]),
        )
        if has_robot
        else HardwareInfo(type="CPU", model="CPU")
    )
    wrapper = RealWorldEnv.__new__(RealWorldEnv)
    wrapper.worker_info = SimpleNamespace(
        hardware_infos=[allocated], cluster_node_rank=0, rank=0
    )
    wrapper.override_cfg = {"is_dummy": True, "enable_camera_player": False}
    wrapper.cfg = OmegaConf.create(
        {"init_params": {"id": "SO101ReachEnv-v1"}, "teleop": "none"}
    )
    env = wrapper._create_env(0)
    try:
        observation, _ = env.reset()
        assert bool(observation.get("frames")) is has_robot
        assert env.unwrapped.robot_info is (allocated if has_robot else None)
    finally:
        env.close()


@pytest.fixture
def so101_tool(monkeypatch):
    from robot_mocks import mocked_sdks

    from rlinf.envs.real import so101
    from rlinf.robotics import SO101Config
    from toolkits.realworld_check import test_so101_env as tool

    for field in dataclasses.fields(SO101Config):
        monkeypatch.delenv(field.name.upper(), raising=False)
    constructor = Mock(return_value=Mock())
    monkeypatch.setattr(so101, "SO101ReachEnv", constructor)
    monkeypatch.setattr(tool, "drive", Mock())

    def run(*flags):
        monkeypatch.setattr(sys, "argv", ["test_so101_env", *flags])
        with mocked_sdks():
            tool.main()

    return run, constructor


@pytest.mark.parametrize(
    "flags,port,calibration,player",
    [
        ([], "/dev/from-env", "from-env", False),
        (
            ["--port", "/dev/from-env", "--id", "from-cli", "--enable-camera-player"],
            "/dev/from-env",
            "from-cli",
            True,
        ),
    ],
)
def test_so101_tool_enumerates_environment_and_cli_settings(
    monkeypatch, so101_tool, flags, port, calibration, player
):
    monkeypatch.setenv("SERIAL_PORT", "/dev/from-env")
    monkeypatch.setenv("CALIBRATION_ID", "from-env")
    monkeypatch.setenv("CAMERA_SERIALS", "MOCK0001,MOCK0002")
    monkeypatch.setenv("MAX_RELATIVE_TARGET", "12")
    run, constructor = so101_tool
    run(*flags)
    config = constructor.call_args.kwargs["robot_info"].config
    assert config.serial_port == port
    assert config.calibration_id == calibration
    assert config.camera_serials == ["MOCK0001", "MOCK0002"]
    assert config.max_relative_target == 12
    assert constructor.call_args.args[0]["enable_camera_player"] is player
    assert constructor.call_args.args[0]["step_frequency"] == 30.0
    assert constructor.call_args.args[0]["reset_on_init"] is False
    assert constructor.call_args.args[0]["reset_joint_speed"] == pytest.approx(
        np.deg2rad(30)
    )
    constructor.return_value.close.assert_called_once_with()


@pytest.mark.parametrize("fps", [15, 60])
def test_so101_tool_passes_the_requested_control_rate(so101_tool, fps):
    run, constructor = so101_tool
    run("--fps", str(fps))
    assert constructor.call_args.args[0]["step_frequency"] == fps


def test_so101_tool_passes_reset_speed_and_skips_startup_reset(so101_tool):
    from toolkits.realworld_check import test_so101_env as tool

    run, constructor = so101_tool
    run("--reset-speed", "10", "--no-reset")
    config = constructor.call_args.args[0]
    assert config["reset_joint_speed"] == pytest.approx(np.deg2rad(10))
    assert config["reset_on_init"] is False
    assert tool.drive.call_args.kwargs["reset"] is False


@pytest.mark.parametrize("option", ["--fps", "--reset-speed"])
@pytest.mark.parametrize("fps", ["0", "-1", "nan", "inf"])
def test_so101_tool_rejects_invalid_control_rates(so101_tool, fps, option):
    run, constructor = so101_tool
    with pytest.raises(SystemExit, match="2"):
        run(option, fps)
    constructor.assert_not_called()


@pytest.mark.parametrize("explicit_port", [False, True])
def test_so101_tool_ignores_unrelated_process_environment(
    monkeypatch, so101_tool, explicit_port
):
    from rlinf.robotics import SO101Config

    monkeypatch.setenv("PORT", "8080")
    monkeypatch.setenv("ID", "unrelated-service-id")
    run, constructor = so101_tool
    run(*(["--port", "/dev/ttyACM1", "--id", "follower"] if explicit_port else []))
    config = constructor.call_args.kwargs["robot_info"].config
    assert config.serial_port == (
        "/dev/ttyACM1" if explicit_port else SO101Config(node_rank=0).serial_port
    )
    assert config.calibration_id == ("follower" if explicit_port else None)


@pytest.mark.parametrize("ports", [None, "/dev/ttyACM0", "/dev/ttyACM0,/dev/ttyACM1"])
def test_so101_tool_rejects_a_port_not_in_the_configured_rig(
    monkeypatch, so101_tool, ports, capsys
):
    monkeypatch.setenv(
        "CALIBRATION_ID",
        "leader,follower" if ports and "," in ports else "leader",
    )
    if ports is not None:
        monkeypatch.setenv("SERIAL_PORT", ports)
    run, constructor = so101_tool
    with pytest.raises(SystemExit) as error:
        run("--port", "/dev/typo")
    assert error.value.code == 2
    assert "No configured SO-101 matches --port" in capsys.readouterr().err
    constructor.assert_not_called()


def test_so101_tool_selects_a_port_with_its_own_calibration(monkeypatch, so101_tool):
    monkeypatch.setenv("SERIAL_PORT", "/dev/ttyACM0,/dev/ttyACM1")
    monkeypatch.setenv("CALIBRATION_ID", "leader,follower")
    monkeypatch.setenv("MAX_RELATIVE_TARGET", "5,12")
    run, constructor = so101_tool
    run("--port", "/dev/ttyACM1")
    config = constructor.call_args.kwargs["robot_info"].config
    assert (config.serial_port, config.calibration_id, config.max_relative_target) == (
        "/dev/ttyACM1",
        "follower",
        12,
    )


def test_so101_tool_requires_selection_with_multiple_arms(monkeypatch, so101_tool):
    monkeypatch.setenv("SERIAL_PORT", "/dev/ttyACM0,/dev/ttyACM1")
    monkeypatch.setenv("CALIBRATION_ID", "leader,follower")
    run, constructor = so101_tool
    with pytest.raises(SystemExit):
        run()
    constructor.assert_not_called()


def test_dummy_franka_requires_an_explicit_camera_layout():
    from rlinf.envs.real.franka import FrankaEnv
    from rlinf.robotics import FrankaConfig, RobotInfo

    with pytest.raises(ValueError, match="including in dummy mode"):
        FrankaEnv({"is_dummy": True})
    info = RobotInfo(
        type="Franka",
        model="Franka",
        config=FrankaConfig(node_rank=0, camera_serials=["dummy"]),
    )
    env = FrankaEnv({"is_dummy": True}, robot_info=info)
    try:
        observation, _ = env.reset()
        assert observation in env.observation_space
        assert list(observation["frames"]) == ["wrist_1"]
        assert env.robot is None
    finally:
        env.close()


def test_dual_franka_hardware_defaults_build_supported_franky_grippers():
    from rlinf.robotics import DualFrankaConfig, DualFrankaRobot
    from rlinf.robotics.parts.end_effectors import EndEffector
    from rlinf.robotics.parts.end_effectors.grippers.franky import FrankyGripper

    config = DualFrankaConfig(node_rank=0)
    assert config.left_gripper_type == config.right_gripper_type == "franka"
    robot = DualFrankaRobot.build(
        left_robot_ip="192.0.2.1",
        right_robot_ip="192.0.2.2",
        left_gripper_type=config.left_gripper_type,
        right_gripper_type=config.right_gripper_type,
    )
    grippers = robot.parts_of_type(EndEffector)
    assert len(grippers) == 2
    assert all(isinstance(gripper, FrankyGripper) for gripper in grippers.values())


def test_scheduler_hardware_passes_plain_nested_settings_to_the_hand(monkeypatch):
    from omegaconf import OmegaConf
    from robot_mocks import mocked_sdks

    from rlinf.robotics import FrankaRobot
    from rlinf.scheduler.hardware import NodeHardwareConfig

    source = OmegaConf.create(
        {
            "type": "Franka",
            "configs": [
                {
                    "node_rank": 0,
                    "end_effector_type": "ruiyan_hand",
                    "end_effector_config": {
                        "port": "/dev/hand",
                        "motor_ids": [1, 2, 3, 4, 5, 6],
                        "default_state": [0.0] * 6,
                    },
                }
            ],
        }
    )
    hardware = NodeHardwareConfig(**source).configs[0]
    settings = hardware.end_effector_config
    assert type(settings) is dict
    assert type(settings["motor_ids"]) is list
    assert type(settings["default_state"]) is list
    with mocked_sdks():
        import rlinf_dexhand.ruiyan as sdk

        driver = Mock()
        monkeypatch.setattr(sdk, "RuiyanHandDriver", driver)
        hand = FrankaRobot.declare_end_effector(
            robot_ip="192.0.2.1",
            node_rank=None,
            name="test-hand",
            end_effector_type=hardware.end_effector_type,
            end_effector_config=settings,
        )
        hand.connect()
        try:
            for name, value in settings.items():
                assert driver.call_args.kwargs[name] == value
            assert type(driver.call_args.kwargs["motor_ids"]) is list
            assert type(driver.call_args.kwargs["default_state"]) is list
        finally:
            hand.disconnect()


MUJOCO_EGL_DEVICE_ID, EGL_DEVICE_ID = EGL_DEVICE_ID_ENV_VARS

_MANAGED_ENV_VARS = ("CUDA_VISIBLE_DEVICES", "MUJOCO_GL", *EGL_DEVICE_ID_ENV_VARS)

# The node from the bug report: nine EGL devices, of which four are GPUs, and an
# EGL enumeration order that does not follow the CUDA one.
_EGL_INDEX_BY_CUDA_ORDINAL = {0: 2, 1: 3, 2: 0, 3: 1}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in _MANAGED_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    nvidia_gpu._egl_index_by_cuda_device.cache_clear()
    yield
    nvidia_gpu._egl_index_by_cuda_device.cache_clear()


@pytest.fixture(autouse=True)
def _restore_meta_path():
    saved = list(sys.meta_path)
    yield
    sys.meta_path[:] = saved


@pytest.fixture
def egl_devices(monkeypatch):
    """Enumerate the bug report's EGL topology instead of the real driver."""
    monkeypatch.setattr(
        nvidia_gpu,
        "_query_egl_index_by_cuda_ordinal",
        lambda: dict(_EGL_INDEX_BY_CUDA_ORDINAL),
    )


@pytest.fixture
def no_egl(monkeypatch):
    """A node whose driver cannot be asked about EGL devices."""

    def unavailable():
        raise OSError("libEGL.so.1: cannot open shared object file")

    monkeypatch.setattr(nvidia_gpu, "_query_egl_index_by_cuda_ordinal", unavailable)


def _become_a_rendering_worker(monkeypatch, cuda_device_id="3", egl_device_id="1"):
    """Reproduce a worker whose EGL index and CUDA device id disagree."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", cuda_device_id)
    monkeypatch.setenv("MUJOCO_GL", "egl")
    monkeypatch.setenv(MUJOCO_EGL_DEVICE_ID, egl_device_id)


def _accelerator_env_var(monkeypatch, visible_accelerators: list[str]) -> dict:
    monkeypatch.setattr(nvidia_gpu, "_torch_needs_avoid_record_streams", lambda: False)
    return NvidiaGPUManager.get_accelerator_env_var(visible_accelerators)


# ---------------------------------------------------------------------------
# CUDA -> EGL mapping
# ---------------------------------------------------------------------------


def test_every_cuda_device_maps_to_its_own_egl_index(egl_devices):
    resolved = {
        cuda_id: NvidiaGPUManager.get_egl_device_id(cuda_id) for cuda_id in range(4)
    }

    assert resolved == _EGL_INDEX_BY_CUDA_ORDINAL


def test_the_mapping_accepts_the_string_ids_placement_speaks_in(egl_devices):
    assert NvidiaGPUManager.get_egl_device_id("2") == 0


def test_ordinals_are_translated_through_cuda_visible_devices(monkeypatch, egl_devices):
    # EGL_CUDA_DEVICE_NV reports a device's position in CUDA_VISIBLE_DEVICES, so
    # in a process that only sees some GPUs those ordinals are not device ids.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6,7")

    assert NvidiaGPUManager.get_egl_device_id(4) == 2
    assert NvidiaGPUManager.get_egl_device_id(7) == 1
    # Devices this process cannot see have no readable mapping.
    assert NvidiaGPUManager.get_egl_device_id(0) is None


def test_an_unmapped_device_has_no_egl_index(egl_devices):
    assert NvidiaGPUManager.get_egl_device_id(8) is None


def test_a_uuid_device_has_no_egl_index(egl_devices):
    assert NvidiaGPUManager.get_egl_device_id("GPU-05d35c06-da01") is None


def test_the_mapping_is_empty_when_the_driver_cannot_be_queried(no_egl):
    assert NvidiaGPUManager.get_egl_device_id(0) is None


def test_the_driver_is_queried_once(monkeypatch):
    calls = []

    def counting_query():
        calls.append(None)
        return dict(_EGL_INDEX_BY_CUDA_ORDINAL)

    monkeypatch.setattr(nvidia_gpu, "_query_egl_index_by_cuda_ordinal", counting_query)

    NvidiaGPUManager.get_egl_device_id(0)
    NvidiaGPUManager.get_egl_device_id(1)

    assert len(calls) == 1


# ---------------------------------------------------------------------------
# Worker env vars
# ---------------------------------------------------------------------------


def test_the_worker_gets_the_egl_index_of_its_own_gpu(monkeypatch, egl_devices):
    monkeypatch.setenv("MUJOCO_GL", "egl")

    env_vars = _accelerator_env_var(monkeypatch, ["3"])

    assert env_vars["CUDA_VISIBLE_DEVICES"] == "3"
    # A CUDA device id is not an EGL index, so it must not be passed through.
    assert env_vars[MUJOCO_EGL_DEVICE_ID] == "1"
    assert env_vars[EGL_DEVICE_ID] == "1"


def test_a_multi_gpu_worker_renders_on_its_first_gpu(monkeypatch, egl_devices):
    monkeypatch.setenv("MUJOCO_GL", "egl")

    env_vars = _accelerator_env_var(monkeypatch, ["2", "3"])

    assert env_vars[MUJOCO_EGL_DEVICE_ID] == "0"


@pytest.mark.parametrize("backend", ["osmesa", "glx"])
def test_cpu_rendering_sets_no_egl_device(monkeypatch, egl_devices, backend):
    monkeypatch.setenv("MUJOCO_GL", backend)

    env_vars = _accelerator_env_var(monkeypatch, ["3"])

    assert not [name for name in EGL_DEVICE_ID_ENV_VARS if name in env_vars]


@pytest.mark.parametrize("backend", ["", "glfw", "OSMesa"])
def test_backends_robosuite_rewrites_to_egl_get_a_device(
    monkeypatch, egl_devices, backend
):
    # robosuite 1.4.1 forces GPU rendering to EGL for every value that is not
    # literally "osmesa" or "glx", down to the casing, so anything else has to
    # be treated as EGL here too.
    monkeypatch.setenv("MUJOCO_GL", backend)

    env_vars = _accelerator_env_var(monkeypatch, ["3"])

    assert env_vars[MUJOCO_EGL_DEVICE_ID] == "1"


def test_a_worker_without_gpus_gets_no_egl_device(monkeypatch, egl_devices):
    monkeypatch.setenv("MUJOCO_GL", "egl")

    env_vars = _accelerator_env_var(monkeypatch, [])

    assert not [name for name in EGL_DEVICE_ID_ENV_VARS if name in env_vars]


def test_an_unmappable_device_falls_back_to_the_cuda_id(monkeypatch, no_egl):
    # The pre-existing behaviour: right whenever the two namespaces happen to
    # agree, which is the common single-node case.
    monkeypatch.setenv("MUJOCO_GL", "egl")

    env_vars = _accelerator_env_var(monkeypatch, ["3"])

    assert env_vars[MUJOCO_EGL_DEVICE_ID] == "3"
    assert env_vars[EGL_DEVICE_ID] == "3"


# ---------------------------------------------------------------------------
# robosuite import shim
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_robosuite(tmp_path, monkeypatch):
    """A stand-in for robosuite that records the environment it was imported with."""
    bindings = tmp_path / "robosuite" / "utils"
    bindings.mkdir(parents=True)
    (tmp_path / "robosuite" / "__init__.py").write_text("")
    (bindings / "__init__.py").write_text("")
    (bindings / "binding_utils.py").write_text(
        textwrap.dedent(
            """
            import os

            SAW_DEVICE_ID = os.environ.get("MUJOCO_EGL_DEVICE_ID", None)
            """
        )
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    yield tmp_path
    for name in [n for n in sys.modules if n.split(".")[0] == "robosuite"]:
        del sys.modules[name]


def test_the_device_is_hidden_from_the_robosuite_import_check(
    monkeypatch, fake_robosuite
):
    # robosuite 1.4.1 asserts MUJOCO_EGL_DEVICE_ID occurs in
    # CUDA_VISIBLE_DEVICES, which a correct EGL index generally does not.
    _become_a_rendering_worker(monkeypatch)
    robosuite_compat.install_robosuite_egl_device_shim()

    bindings = importlib.import_module("robosuite.utils.binding_utils")

    assert bindings.SAW_DEVICE_ID is None
    assert os.environ[MUJOCO_EGL_DEVICE_ID] == "1"


def test_other_modules_import_with_the_device_visible(monkeypatch, fake_robosuite):
    _become_a_rendering_worker(monkeypatch)
    robosuite_compat.install_robosuite_egl_device_shim()

    assert importlib.import_module("robosuite") is not None
    assert os.environ[MUJOCO_EGL_DEVICE_ID] == "1"


def test_a_failed_robosuite_import_still_restores_the_device(
    monkeypatch, fake_robosuite
):
    _become_a_rendering_worker(monkeypatch)
    (fake_robosuite / "robosuite" / "utils" / "binding_utils.py").write_text(
        "raise RuntimeError('import failed')"
    )
    robosuite_compat.install_robosuite_egl_device_shim()

    with pytest.raises(RuntimeError, match="import failed"):
        importlib.import_module("robosuite.utils.binding_utils")

    assert os.environ[MUJOCO_EGL_DEVICE_ID] == "1"


def test_installing_the_shim_repeatedly_leaves_one_finder():
    # Importing rlinf.envs has already installed it once.
    robosuite_compat.install_robosuite_egl_device_shim()
    robosuite_compat.install_robosuite_egl_device_shim()

    finders = [
        finder
        for finder in sys.meta_path
        if isinstance(finder, robosuite_compat._RobosuiteBindingsFinder)
    ]
    assert len(finders) == 1


def test_importing_the_env_package_installs_the_shim():
    # Simulator subprocesses re-import rlinf.envs in a fresh interpreter and rely
    # on it to install the shim before any simulator is imported.
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import sys

                import rlinf.envs  # noqa: F401
                from rlinf.utils.robosuite_compat import _RobosuiteBindingsFinder

                assert any(
                    isinstance(f, _RobosuiteBindingsFinder) for f in sys.meta_path
                )
                assert "rlinf.scheduler" not in sys.modules, (
                    "the env package must not drag the scheduler into every "
                    "simulator subprocess"
                )
                """
            ),
        ],
        capture_output=True,
        text=True,
    )

    assert child.returncode == 0, child.stderr


@pytest.mark.skipif(
    importlib.util.find_spec("robosuite") is None, reason="robosuite is not installed"
)
def test_real_robosuite_imports_with_a_mismatched_egl_index():
    # The EGL index and the CUDA ordinal disagree here, which is what robosuite
    # 1.4.1 refuses to import with.
    env = os.environ.copy()
    env.update(
        {"CUDA_VISIBLE_DEVICES": "3", "MUJOCO_GL": "osmesa", MUJOCO_EGL_DEVICE_ID: "1"}
    )
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import os

                import rlinf.envs  # installs the shim
                import robosuite.utils.binding_utils  # noqa: F401

                assert os.environ["MUJOCO_EGL_DEVICE_ID"] == "1"
                """
            ),
        ],
        capture_output=True,
        text=True,
        env=env,
    )

    assert child.returncode == 0, child.stderr


class _FakeCudaTensor:
    def torch(self):
        return torch.tensor([1.0])


class _FakePx:
    def __init__(self):
        self.cuda_articulation_link_data = _FakeCudaTensor()
        self.cuda_articulation_qacc = _FakeCudaTensor()
        self.cuda_articulation_qf = _FakeCudaTensor()
        self.cuda_articulation_qpos = _FakeCudaTensor()
        self.cuda_articulation_qvel = _FakeCudaTensor()
        self.cuda_articulation_target_qpos = _FakeCudaTensor()
        self.cuda_articulation_target_qvel = _FakeCudaTensor()
        self.cuda_rigid_body_data = _FakeCudaTensor()
        self.cuda_rigid_dynamic_data = _FakeCudaTensor()

    def gpu_update_articulation_kinematics(self):
        return None


class _FakeScene:
    def __init__(self):
        self.px = _FakePx()
        self.timestep = None

    def get_timestep(self):
        return 0.02

    def set_timestep(self, timestep):
        self.timestep = timestep

    def _gpu_apply_all(self):
        return None

    def _gpu_fetch_all(self):
        return None


class _FakeBatchedRng:
    rngs = ["rng"]


class _FakeController:
    def __init__(self):
        self.loaded_state = None

    def get_state(self):
        return {"controller": torch.tensor([1.0])}

    def set_state(self, state):
        self.loaded_state = state


class _FakeAgent:
    def __init__(self):
        self.controller = _FakeController()


class _FakeEnv:
    def __init__(self):
        self.unwrapped = self
        self.device = "cpu"
        self.scene = _FakeScene()
        self._main_rng = "main_rng"
        self._batched_main_rng = _FakeBatchedRng()
        self._main_seed = 123
        self._episode_rng = "episode_rng"
        self._batched_episode_rng = _FakeBatchedRng()
        self._episode_seed = 456
        self.action_space = "action_space"
        self.single_action_space = "single_action_space"
        self._orig_single_action_space = "orig_single_action_space"
        self._elapsed_steps = torch.tensor([3])
        self._init_raw_obs = {"obs": torch.tensor([1.0])}
        self.agent = _FakeAgent()
        self.task_reset_states = {}
        self.task_metric_states = {}
        self.reset_seed = None
        self.reset_options = None
        self.loaded_state = None

    def get_state(self):
        return {"sim": torch.tensor([2.0])}

    def reset(self, seed=None, options=None):
        self.reset_seed = seed
        self.reset_options = options

    def set_state(self, state):
        self.loaded_state = state


class _FakeManiskillEnv:
    pass


def _load_maniskill_offload_module(monkeypatch):
    repo_root = Path(__file__).resolve().parents[2]
    module_path = (
        repo_root / "rlinf" / "envs" / "sim" / "maniskill" / "maniskill_offload_env.py"
    )

    fake_package = types.ModuleType("rlinf.envs.sim.maniskill")
    fake_package.__path__ = [str(module_path.parent)]
    fake_env_module = types.ModuleType("rlinf.envs.sim.maniskill.maniskill_env")
    fake_env_module.ManiskillEnv = _FakeManiskillEnv

    monkeypatch.setitem(sys.modules, "rlinf.envs.sim.maniskill", fake_package)
    monkeypatch.setitem(
        sys.modules, "rlinf.envs.sim.maniskill.maniskill_env", fake_env_module
    )
    monkeypatch.delitem(
        sys.modules, "rlinf.envs.sim.maniskill.maniskill_offload_env", raising=False
    )

    spec = importlib.util.spec_from_file_location(
        "rlinf.envs.sim.maniskill.maniskill_offload_env", module_path
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def _make_core(module):
    core = object.__new__(module._ManiskillEnvCore)
    core.env = _FakeEnv()
    core.seed = 10
    core.device = "cpu"
    core.prev_step_reward = torch.tensor([0.5])
    core.reset_state_ids = torch.tensor([1])
    core._generator = torch.Generator()
    core._generator.manual_seed(0)
    core.is_start = True
    core.record_metrics = False
    return core


def test_maniskill_offload_state_does_not_require_record_video_counter(monkeypatch):
    module = _load_maniskill_offload_module(monkeypatch)
    core = _make_core(module)

    state_buffer = core.get_state()
    state = torch.load(io.BytesIO(state_buffer), map_location="cpu", weights_only=False)

    assert "video_cnt" not in state


def test_maniskill_offload_load_state_accepts_state_without_video_counter(
    monkeypatch,
):
    module = _load_maniskill_offload_module(monkeypatch)
    monkeypatch.setattr(
        module, "set_batch_rng_state", lambda rng_state: _FakeBatchedRng()
    )
    core = _make_core(module)

    source_state = torch.load(
        io.BytesIO(core.get_state()),
        map_location="cpu",
        weights_only=False,
    )
    assert "video_cnt" not in source_state

    core.load_state(_serialize_state(source_state))

    assert core.env.reset_seed == core.seed
    assert core.env.reset_options == {"reconfigure": False}
    torch.testing.assert_close(core.env.loaded_state["sim"], torch.tensor([2.0]))
    assert core.env.scene.timestep == 0.02
    assert not hasattr(core, "video_cnt")


def test_maniskill_offload_load_state_ignores_legacy_video_counter(monkeypatch):
    module = _load_maniskill_offload_module(monkeypatch)
    monkeypatch.setattr(
        module, "set_batch_rng_state", lambda rng_state: _FakeBatchedRng()
    )
    core = _make_core(module)

    source_state = torch.load(
        io.BytesIO(core.get_state()),
        map_location="cpu",
        weights_only=False,
    )
    source_state["video_cnt"] = 7

    core.load_state(_serialize_state(source_state))

    assert not hasattr(core, "video_cnt")


def _serialize_state(state):
    buffer = io.BytesIO()
    torch.save(state, buffer)
    return buffer.getvalue()


def test_net_emulation_config_parses_legacy_crossdc_pairs():
    cfg = OmegaConf.create(
        {
            "enabled": True,
            "symmetric": True,
            "crossdc_pairs": [
                {"src": "Env:0", "dst": "Actor:0", "delay_ms": 10},
            ],
            "bandwidth_groups": [],
        }
    )

    net_cfg = NetEmulationConfig.from_cfg(cfg)

    assert net_cfg is not None
    assert net_cfg.crossdc_pairs == (
        CrossDCPair(src="Env:0", dst="Actor:0", delay_ms=10.0),
    )


def test_net_emulation_config_endpoint_ranges_equal_explicit_lists():
    def build_cfg(src, dst, members):
        return OmegaConf.create(
            {
                "enabled": True,
                "symmetric": True,
                "crossdc_pairs": [
                    {"src": src, "dst": dst, "delay_ms": 10},
                ],
                "bandwidth_groups": [
                    {"members": members, "bandwidth_mbps": 1000},
                ],
            }
        )

    explicit_cfg = build_cfg(
        ["Env:0", "Env:1"],
        ["Actor:0", "Actor:1"],
        ["Env:0", "Env:1", "Actor:0", "Actor:1"],
    )
    range_cfg = build_cfg(["Env:0-1"], ["Actor:0-1"], ["Env:0-1", "Actor:0-1"])

    assert NetEmulationConfig.from_cfg(range_cfg) == NetEmulationConfig.from_cfg(
        explicit_cfg
    )


@pytest.mark.parametrize("field_name", ["src", "dst"])
def test_net_emulation_config_rejects_empty_crossdc_pair_endpoint_lists(field_name):
    cfg = OmegaConf.create(
        {
            "enabled": True,
            "symmetric": True,
            "crossdc_pairs": [
                {
                    "src": ["Env:0"],
                    "dst": ["Actor:0"],
                    "delay_ms": 10,
                },
            ],
            "bandwidth_groups": [],
        }
    )
    cfg.crossdc_pairs[0][field_name] = []

    with pytest.raises(ValueError, match=field_name):
        NetEmulationConfig.from_cfg(cfg)


def test_net_emulation_config_disabled_returns_none():
    cfg = OmegaConf.create({"enabled": False, "crossdc_pairs": []})

    assert NetEmulationConfig.from_cfg(cfg) is None
    assert NetEmulationConfig.from_cfg(None) is None


def _manager(**overrides):
    """Build a NetEmulationManager directly, bypassing the Ray actor launch."""
    cfg = {
        "enabled": True,
        "symmetric": True,
        "crossdc_pairs": [
            {"src": "EnvGroup:0", "dst": "ActorGroup:0", "delay_ms": 100}
        ],
        "bandwidth_groups": [],
    }
    cfg.update(overrides)
    return NetEmulationManager(cfg)


def test_reserve_returns_zero_for_unemulated_links():
    manager = _manager()

    assert manager.reserve("EnvGroup:0", "RolloutGroup:0", 1024) == 0.0


def test_reserve_applies_link_delay_in_both_directions():
    manager = _manager()

    assert manager.reserve("EnvGroup:0", "ActorGroup:0", 0) == pytest.approx(
        0.1, abs=0.02
    )
    # symmetric: true mirrors every configured pair.
    assert manager.reserve("ActorGroup:0", "EnvGroup:0", 0) == pytest.approx(
        0.1, abs=0.02
    )


def test_reserve_ignores_group_suffix_in_endpoint_names():
    """Endpoints may be written with or without the trailing ``Group``."""
    manager = _manager(
        crossdc_pairs=[{"src": "Env:0", "dst": "Actor:0", "delay_ms": 100}]
    )

    assert manager.reserve("EnvGroup:0", "ActorGroup:0", 0) == pytest.approx(
        0.1, abs=0.02
    )


def test_reserve_charges_transfer_time_against_the_bandwidth_budget():
    # 8 Mbps == 1 MB/s, so a 1 MB payload occupies the link for one second.
    manager = _manager(
        bandwidth_groups=[
            {"members": ["EnvGroup:0"], "bandwidth_mbps": 8},
            {"members": ["ActorGroup:0"], "bandwidth_mbps": 8},
        ]
    )

    one_mb = 1_000_000
    assert manager.reserve("EnvGroup:0", "ActorGroup:0", one_mb) == pytest.approx(
        1.1, abs=0.02
    )
    # The uplink is still busy, so a second send queues behind the first.
    assert manager.reserve("EnvGroup:0", "ActorGroup:0", one_mb) == pytest.approx(
        2.1, abs=0.02
    )


def test_estimate_payload_size_counts_tensor_storage():
    tensor = torch.zeros(256, dtype=torch.float32)  # 1024 bytes of data

    assert NetEmulationManager.estimate_payload_size_bytes(tensor) == 1024
    assert NetEmulationManager.estimate_payload_size_bytes(None) == 0


def test_estimate_payload_size_walks_nested_containers():
    payload = {"a": torch.zeros(256, dtype=torch.float32), "b": [torch.zeros(256)]}

    size = NetEmulationManager.estimate_payload_size_bytes(payload)

    # Two tensors of 1024 bytes each, plus per-tensor overhead and pickled keys.
    assert size > 2048
    assert size < 2048 + 4 * 256


def test_estimate_payload_size_falls_back_to_pickle_for_plain_objects():
    payload = {"task": "pick up the cube", "step": 7}

    size = NetEmulationManager.estimate_payload_size_bytes(payload)

    assert size == len(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))


def test_reserve_broadcast_charges_the_uplink_once():
    # 8 Mbps == 1 MB/s on each side.
    manager = _manager(
        crossdc_pairs=[
            {"src": "Env:0", "dst": ["Actor:0", "Actor:1"], "delay_ms": 100}
        ],
        bandwidth_groups=[
            {"members": ["Env:0"], "bandwidth_mbps": 8},
            {"members": ["Actor:0", "Actor:1"], "bandwidth_mbps": 8},
        ],
    )

    one_mb = 1_000_000
    # Both receivers share one bandwidth group, so the payload crosses the link
    # once: 1s uplink + 0.1s delay, not 2s.
    assert manager.reserve_broadcast(
        "EnvGroup:0", ["ActorGroup:0", "ActorGroup:1"], one_mb
    ) == pytest.approx(1.1, abs=0.02)


def test_reserve_broadcast_waits_for_the_slowest_destination():
    manager = _manager(
        crossdc_pairs=[
            {"src": "Env:0", "dst": "Actor:0", "delay_ms": 50},
            {"src": "Env:0", "dst": "Actor:1", "delay_ms": 200},
        ],
    )

    assert manager.reserve_broadcast(
        "EnvGroup:0", ["ActorGroup:0", "ActorGroup:1"], 0
    ) == pytest.approx(0.2, abs=0.02)


def test_reserve_broadcast_ignores_unemulated_destinations():
    manager = _manager()

    assert (
        manager.reserve_broadcast("EnvGroup:0", ["RolloutGroup:0", "EnvGroup:1"], 4096)
        == 0.0
    )


def _first_eval_seeds(
    *,
    seed_count: int,
    total_num_envs: int,
    total_num_processes: int,
    group_size: int = 1,
    base_seed: int = 0,
) -> list[int]:
    success_seeds = torch.arange(seed_count, dtype=torch.long)
    selected_seeds = []
    for seed_offset in range(total_num_processes):
        num_envs = total_num_envs // total_num_processes
        num_group = num_envs // group_size
        worker_seeds = partition_success_seeds(
            success_seeds,
            base_seed=base_seed,
            seed_offset=seed_offset,
            total_num_processes=total_num_processes,
            num_group=num_group,
        )
        selected_seeds.extend(worker_seeds[:num_group].tolist())
    return selected_seeds


@pytest.mark.parametrize(
    ("seed_count", "total_num_envs", "total_num_processes"),
    [
        (320, 128, 4),
        (320, 128, 8),
        (320, 128, 16),
        (260, 128, 4),
        (200, 128, 8),
        (150, 128, 4),
    ],
)
def test_robotwin_eval_success_seeds_do_not_overlap_across_workers(
    seed_count: int,
    total_num_envs: int,
    total_num_processes: int,
):
    """Regression test for duplicate RoboTwin eval seeds across EnvWorkers."""
    selected_seeds = _first_eval_seeds(
        seed_count=seed_count,
        total_num_envs=total_num_envs,
        total_num_processes=total_num_processes,
    )

    assert len(selected_seeds) == total_num_envs
    assert len(set(selected_seeds)) == total_num_envs


def test_robotwin_eval_success_seed_order_is_controlled_by_base_seed():
    selected_seed_0 = _first_eval_seeds(
        seed_count=320,
        total_num_envs=128,
        total_num_processes=4,
        base_seed=0,
    )
    selected_seed_0_again = _first_eval_seeds(
        seed_count=320,
        total_num_envs=128,
        total_num_processes=4,
        base_seed=0,
    )
    selected_seed_1 = _first_eval_seeds(
        seed_count=320,
        total_num_envs=128,
        total_num_processes=4,
        base_seed=1,
    )

    assert selected_seed_0 == selected_seed_0_again
    assert selected_seed_0 != selected_seed_1


@pytest.mark.parametrize("continuous_tracking", [False, True])
def test_ur5e_local_delta_binary_gripper_and_moveL_reset(continuous_tracking):
    from robot_mocks import mocked_sdks
    from scipy.spatial.transform import Rotation as R

    from rlinf.envs.real.ur5e import UR5eEnv
    from rlinf.envs.real.utils.video import VideoPlayer
    from rlinf.robotics import RobotInfo, UR5eConfig, UR5eRobot
    from rlinf.robotics.parts.arms.ur5e import UR5eArm
    from rlinf.robotics.parts.end_effectors.grippers.pgi import PGIGripper

    info = RobotInfo(
        type="UR5e",
        model="UR5e",
        config=UR5eConfig(node_rank=0, camera_serials=["dummy"]),
    )
    with mocked_sdks() as sdks:
        arm = UR5eArm(
            "192.0.2.6", impedance={"continuous_tracking": continuous_tracking}
        )
        gripper = PGIGripper("/tmp/ur5e-env-gripper")
        robot = UR5eRobot(arm=arm, end_effector=gripper)
        robot.connect()
        env = UR5eEnv(
            {
                "is_dummy": True,
                "enable_camera_player": False,
                "target_ee_pose": [0] * 6,
                "ee_pose_limit_min": [-1] * 6,
                "ee_pose_limit_max": [1] * 6,
                "reset_ee_pose": [0.4, 0.0, 0.3, 0.0, 0.0, 0.0],
                "enable_random_reset": True,
                "joint_reset_cycle": 1,
                "payload_settle_seconds": 0.0,
                "ft_zero_settle_seconds": 0.0,
            },
            robot_info=info,
        )
        env.config.is_dummy = False
        env.robot = robot
        env._arm = arm
        env._end_effector = gripper
        env._cameras = {}
        env.camera_player = VideoPlayer(False)
        env._robot_state = arm.get_state()
        current = env._robot_state.tcp_pose.copy()
        action = np.array([0.01, 0.02, -0.01, 0.03, 0.02, 0.01, 0.5])
        try:
            obs, _, _, _, _ = env.step(action)
            force = sdks["rtde_control"].states["192.0.2.6"]["forces"][-1][2]
            np.testing.assert_allclose(force[:3], 1200 * action[:3], atol=1e-5)
            expected = R.from_euler("xyz", action[3:6]) * R.from_quat(current[3:])
            rotation_error = (expected * R.from_quat(current[3:]).inv()).as_rotvec()
            np.testing.assert_allclose(force[3:], 20 * rotation_error, atol=1e-5)
            np.testing.assert_allclose(obs["state"]["gripper_position"], [0.08])
            assert not isinstance(env, FrankaEnv)
            assert UR5eEnv.step is not FrankaEnv.step
            device = sdks["pyDHgripper"].PGE.instances[-1]
            reads = []
            original_read = device.read_pos

            def read_position():
                reads.append(1)
                return original_read()

            device.read_pos = read_position
            pose_reads = []
            receive = arm._device[1]
            original_pose = receive.getActualTCPPose

            def read_pose():
                pose_reads.append(1)
                return original_pose()

            receive.getActualTCPPose = read_pose
            env.begin_action_chunk(3, infer_index=7)
            for index in range(3):
                _, _, _, _, step_info = env.step(action)
                trace = step_info["ur_action_trace"]
                assert trace["infer_index"] == 7
                assert trace["action_index"] == index + 1
                assert trace["target_hz"] == env.config.step_frequency
                assert trace["control_sec"] == pytest.approx(
                    trace["command_sec"] + trace["pace_wait_sec"]
                )
                assert trace["control_hz"] == pytest.approx(1 / trace["control_sec"])
                assert trace["control_overrun_sec"] == pytest.approx(
                    max(0, (trace["target_interval_sec"] or 0) - 1 / trace["target_hz"])
                )
                np.testing.assert_array_equal(trace["action"], action)
                assert trace["fresh_observation"] == (index == 2)
                assert ("tcp_after" in trace) == (index == 2)
                if index > 0:
                    assert trace["target_interval_sec"] >= 1 / env.config.step_frequency
                assert len(reads) == (2 if index == 2 else 0)
                if not continuous_tracking:
                    assert len(pose_reads) == index + 1 + (index == 2)
            env.reset(seed=123)
            state = sdks["rtde_control"].states["192.0.2.6"]
            offset = np.random.default_rng(123).uniform(-0.03, 0.03, 3)
            np.testing.assert_allclose(
                np.asarray(state["moves"][-1][0])[:3],
                np.array([0.4, 0.0, 0.3]) + offset,
                atol=1e-4,
            )
            assert "zeroFtSensor" in state["events"]
            assert "moveJ" not in state["events"]
        finally:
            env.close()


def test_real_observation_config_preserves_ur_training_order():
    from omegaconf import OmegaConf

    from rlinf.envs.real.env import RealWorldEnv

    env = RealWorldEnv.__new__(RealWorldEnv)
    keys = ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_position"]
    env.cfg = OmegaConf.create({"state_keys": keys, "wrist_image_key": "wrist_image"})
    env.main_image_key = "exterior_image"
    env.task_descriptions = ["insert"]
    state = {
        key: np.full((1, dim), i, dtype=np.float32)
        for i, (key, dim) in enumerate(zip(keys, [6, 6, 3, 3, 1]))
    }
    raw = {
        "state": state,
        "frames": {
            "exterior_image": np.zeros((1, 128, 128, 3), dtype=np.uint8),
            "wrist_image": np.ones((1, 128, 128, 3), dtype=np.uint8),
        },
    }
    obs = env._wrap_obs(raw)
    np.testing.assert_array_equal(
        obs["states"].numpy(), np.concatenate(list(state.values()), axis=-1)
    )
    assert obs["states"].shape == (1, 19)
    assert obs["wrist_images"].shape == (1, 128, 128, 3)
    env.cfg.state_keys = ["tcp_pose"]
    with pytest.raises(ValueError, match="state_keys"):
        env._wrap_obs(raw)


@pytest.mark.parametrize("reference_frame", ["tcp", "base"])
@pytest.mark.parametrize("is_open", [False, True])
@pytest.mark.parametrize("manual_reward", [False, True])
def test_ur5e_rlt_keyboard_uses_policy_units_and_shared_controls(
    monkeypatch, reference_frame, is_open, manual_reward
):
    from robot_mocks import mocked_sdks
    from scipy.spatial.transform import Rotation

    from rlinf.envs.real.ur5e import UR5eEnv
    from rlinf.envs.real.ur5e.motion import ee_to_tcp, tcp_to_ee
    from rlinf.envs.real.utils.video import VideoPlayer
    from rlinf.envs.real.wrappers import build_stack
    from rlinf.robotics import RobotInfo, UR5eConfig, UR5eRobot
    from rlinf.robotics.parts.arms.ur5e import UR5eArm
    from rlinf.robotics.parts.end_effectors.grippers.pgi import PGIGripper

    monkeypatch.setenv("RLINF_KEYBOARD_DEVICE", "/dev/input/event-fake")
    monkeypatch.delenv("RLINF_FAKE_KEYS", raising=False)
    hardware = RobotInfo(
        type="UR5e",
        model="UR5e",
        config=UR5eConfig(
            node_rank=0,
            camera_serials=["dummy"],
            impedance={
                "control_frame_euler_deg": [10, -5, 30],
                "selection_vector": [1, 1, 1, 0, 0, 1],
            },
        ),
    )
    with mocked_sdks() as sdks:
        arm = UR5eArm("192.0.2.16", impedance=hardware.config.impedance)
        gripper = PGIGripper("/tmp/ur5e-rlt-gripper")
        robot = UR5eRobot(arm=arm, end_effector=gripper)
        robot.connect()
        env = UR5eEnv(
            {
                "is_dummy": True,
                "enable_camera_player": False,
                "reference_frame": reference_frame,
                "reset_ee_pose": [0.4, 0.0, 0.3, 0.3, -0.2, 0.8],
                "target_ee_pose": [0.9, 0.9, 0.9, 0, 0, 0],
                "ee_pose_limit_min": [-10] * 6,
                "ee_pose_limit_max": [10] * 6,
                "enable_random_reset": False,
                "enable_gripper_penalty": False,
                "action_scale": [1, 1, 1],
                "action_idle_hold_enabled": False,
                "payload_settle_seconds": 0,
                "ft_zero_settle_seconds": 0,
            },
            robot_info=hardware,
        )
        env.config.is_dummy = False
        env.robot, env._arm, env._end_effector = robot, arm, gripper
        env._cameras = {}
        env.camera_player = VideoPlayer(False)
        sdks["rtde_control"].states["192.0.2.16"]["tcp_offset"] = [
            0.01,
            -0.02,
            0.05,
            0.1,
            0.0,
            -0.2,
        ]
        wrapped = build_stack(
            env,
            {
                "teleop": [
                    {
                        "keyboard": {
                            "step_size_pos": 0.03,
                            "step_size_rot": 0.05,
                            "step_size_pos_alt": 0.006,
                            "step_size_rot_alt": 0.03,
                        }
                    }
                ],
                "keyboard_reward_wrapper": "rlt_policy_switch",
                "online_timing_enabled": True,
                "keyboard_wrapper_kwargs": {
                    "switch_key": "Key.f8",
                    **({"success_key": "Key.space"} if manual_reward else {}),
                },
            },
        )
        try:
            wrapped.reset()
            assert not wrapped.get_wrapper_attr("rlt_switch_flags")

            serial_reads = []
            gripper_device = sdks["pyDHgripper"].PGE.instances[-1]
            read_position = gripper_device.read_pos

            def record_position_read():
                serial_reads.append(1)
                return read_position()

            monkeypatch.setattr(gripper_device, "read_pos", record_position_read)
            env.begin_action_chunk(3)
            wrapped.get_wrapper_attr("on_action_chunk_begin")()
            for index in range(3):
                wrapped.step(np.r_[np.zeros(6), 1.0])
                # Teleop reads gripper status; only the last step reads the whole robot.
                assert len(serial_reads) == index + 1 + (2 if index == 2 else 0)
            env.begin_action_chunk(1)
            wrapped.get_wrapper_attr("on_action_chunk_begin")()

            def press(*keys):
                for keyboard in sdks["evdev"].devices:
                    _press_keys(keyboard, *keys)

            for keyboard in sdks["evdev"].devices:
                assert keyboard.emit(
                    *[
                        (key, 1)
                        for key in (
                            "s",
                            "d",
                            "q",
                            "r",
                            "g",
                            "b",
                            "o" if is_open else "l",
                        )
                    ]
                ).wait(2)
            current = arm.get_state().tcp_pose
            offset = arm.get_tcp_offset()
            ee = tcp_to_ee(current, offset)
            control = Rotation.from_euler("xyz", [10, -5, 30], degrees=True)
            base_delta = control.apply([0.03, 0.03, 0.03])
            rotation = (
                control * Rotation.from_euler("xyz", [0, 0, 0.05]) * control.inv()
            )
            expected = ee_to_tcp(
                np.r_[
                    ee[:3] + base_delta,
                    (rotation * Rotation.from_quat(ee[3:])).as_quat(),
                ],
                offset,
            )
            _, reward, terminated, truncated, info = wrapped.step(np.zeros(7))
            np.testing.assert_allclose(
                info["ur_action_trace"]["target_tcp"], expected, atol=1e-6
            )
            assert info["intervene_action"][6] == float(is_open)
            assert not info["rlt_switch_flags"]
            press("f8")
            _, reward, terminated, truncated, info = wrapped.step(np.zeros(7))
            assert (
                info["rlt_switch_flags"]
                and info["rlt_policy_switch_event"] == "enter_actor"
            )
            assert reward == 0 and not terminated and not truncated
            env.config.target_ee_pose[:3] = current[:3]
            _, reward, terminated, _, _ = wrapped.step(np.zeros(7))
            if manual_reward:
                assert reward == 0 and not terminated
                press("space")
                _, reward, terminated, _, _ = wrapped.step(np.zeros(7))
                assert reward == 0 and not terminated
                press("space")
                _, reward, terminated, _, _ = wrapped.step(np.zeros(7))
            assert reward == 1 and terminated
            if manual_reward:
                before = env.num_steps
                _, reward, terminated, _, _ = wrapped.step(np.ones(7))
                assert env.num_steps == before
                assert reward == 0 and terminated
            wrapped.reset()
            assert not wrapped.get_wrapper_attr("rlt_switch_flags")
        finally:
            wrapped.close()


@pytest.mark.parametrize("rtc_enabled", [False, True])
def test_ur5e_eval_config_compose(monkeypatch, rtc_enabled):
    from hydra import compose, initialize_config_dir

    root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("EMBODIED_PATH", str(root / "examples/embodiment"))
    with initialize_config_dir(
        config_dir=str(root / "evaluations/realworld"), version_base="1.3"
    ):
        cfg = compose(
            config_name="realworld_ur5e_eval_pi05_sft",
            overrides=[f"runner.rtc.enabled={str(rtc_enabled).lower()}"],
        )
    assert cfg.env.eval.init_params.id == "UR5eEnv-v1"
    impedance = cfg.cluster.node_groups[0].hardware.configs[0].impedance
    assert "rtde_frequency" not in impedance
    assert impedance.frequency == 100.0
    assert impedance.start_paused
    assert cfg.rollout.model.model_type == "openpi"
    assert cfg.rollout.model.pi05
    assert cfg.rollout.model.openpi.task == "eval"
    assert cfg.rollout.model.openpi.rtc_enabled == rtc_enabled
    assert cfg.actor.model.model_type == "openpi"
    assert list(cfg.env.eval.state_keys) == [
        "tcp_pose",
        "tcp_vel",
        "tcp_force",
        "tcp_torque",
        "gripper_position",
    ]
    assert not cfg.env.eval.use_relative_frame


@pytest.mark.parametrize("chunk_size", [3, 5])
def test_ur5e_inference_chunk_length_updates_framework_count(chunk_size, monkeypatch):
    from hydra import compose, initialize_config_dir

    from rlinf.config import get_rollout_chunk_count

    monkeypatch.setenv("EMBODIED_PATH", str(_ROOT / "examples/embodiment"))
    with initialize_config_dir(
        config_dir=str(_ROOT / "evaluations/realworld"), version_base="1.3"
    ):
        cfg = compose(
            config_name="realworld_ur5e_eval_pi05_sft",
            overrides=[f"rollout.model.openpi.action_chunk={chunk_size}"],
        )
    assert cfg.rollout.model.num_action_chunks == chunk_size
    assert cfg.rollout.model.openpi.action_horizon == 20
    assert get_rollout_chunk_count(
        cfg.env.eval, cfg.rollout.model.num_action_chunks
    ) == (cfg.env.eval.max_steps_per_rollout_epoch // chunk_size)


@pytest.mark.parametrize(
    "prediction_steps,chunk_size,denoising_steps", [(20, 3, 4), (24, 5, 6)]
)
def test_ur5e_rlt_lengths_follow_prediction_and_policy_settings(
    prediction_steps, chunk_size, denoising_steps, monkeypatch
):
    from hydra import compose, initialize_config_dir

    from rlinf.config import resolve_chunk_limits
    from rlinf.models.embodiment.mlp_policy.rlt_mlp_policy import RLTMLPPolicy

    monkeypatch.setenv("EMBODIED_PATH", str(_ROOT / "examples/embodiment"))
    with initialize_config_dir(
        config_dir=str(_ROOT / "examples/embodiment/config"), version_base="1.3"
    ):
        cfg = compose(
            config_name="ur_rlt_stage2_ac_mlp",
            overrides=[
                f"rollout.rlt_feature_model.openpi.action_horizon={prediction_steps}",
                f"actor.model.num_action_chunks={chunk_size}",
                f"rollout.rlt_feature_model.openpi.num_steps={denoising_steps}",
            ],
        )
    feature = cfg.rollout.rlt_feature_model
    assert feature.num_action_chunks == feature.openpi.action_chunk == prediction_steps
    assert cfg.actor.model.ref_num_action_chunks == prediction_steps
    assert cfg.rollout.model.ref_num_action_chunks == prediction_steps
    assert cfg.rollout.model.num_action_chunks == chunk_size
    assert feature.num_steps == feature.openpi.num_steps == denoising_steps
    resolve_chunk_limits(cfg, cfg.actor.model.num_action_chunks)
    assert (
        cfg.env.train.episode_action_limit
        == cfg.env.train.max_episode_chunks * chunk_size
    )

    model = RLTMLPPolicy(
        z_dim=8,
        proprio_dim=2,
        action_dim=7,
        num_action_chunks=cfg.actor.model.num_action_chunks,
        ref_num_action_chunks=cfg.actor.model.ref_num_action_chunks,
    )
    obs = {
        "z_rl": torch.zeros(1, 8),
        "proprio": torch.zeros(1, 2),
        "ref_chunk": torch.zeros(1, prediction_steps, 7),
    }
    actions, _ = model.predict_action_batch(obs, mode="eval")
    assert actions.shape == (1, chunk_size, 7)


def test_ur5e_rlt_config_shares_franka_training_contract(monkeypatch):
    from hydra import compose, initialize_config_dir

    root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("EMBODIED_PATH", str(root / "examples/embodiment"))
    with initialize_config_dir(
        config_dir=str(root / "examples/embodiment/config"), version_base="1.3"
    ):
        cfg = compose(config_name="ur_rlt_stage2_ac_mlp")
        franka = compose(config_name="realworld_rlt_stage2_ac_mlp")
    with initialize_config_dir(
        config_dir=str(root / "evaluations/realworld"), version_base="1.3"
    ):
        inference = compose(config_name="realworld_ur5e_eval_pi05_sft")
    OmegaConf.resolve(cfg)
    feature = cfg.rollout.rlt_feature_model
    assert cfg.cluster.num_nodes == inference.cluster.num_nodes
    training_nodes = OmegaConf.to_container(cfg.cluster.node_groups, resolve=True)
    inference_nodes = OmegaConf.to_container(
        inference.cluster.node_groups, resolve=True
    )
    # Each workflow selects its force-update mode and episode gate independently.
    for nodes in (training_nodes, inference_nodes):
        impedance = nodes[0]["hardware"]["configs"][0]["impedance"]
        for key in ("continuous_tracking", "start_paused"):
            impedance.pop(key, None)
    assert training_nodes == inference_nodes
    assert (
        cfg.cluster.component_placement.env == inference.cluster.component_placement.env
    )
    assert (
        cfg.cluster.component_placement.actor == cfg.cluster.component_placement.rollout
    )
    assert cfg.actor.model.model_type == "rlt_mlp_policy"
    assert cfg.cluster.node_groups[0].hardware.configs[0].impedance.start_paused
    assert cfg.actor.model.proprio_dim == len(feature.openpi_data.obs_indices) == 19
    assert feature.openpi_data == inference.rollout.model.openpi_data
    assert feature.openpi.config_name == "pi05_ur_state"
    assert feature.openpi.num_images_in_input == 2
    assert feature.openpi.use_rlt
    assert not feature.openpi.state_indices
    assert feature.openpi.action_horizon == feature.openpi.action_chunk == 20
    assert (
        cfg.actor.model.num_action_chunks == cfg.rollout.model.num_action_chunks == 10
    )
    assert cfg.actor.model.ref_num_action_chunks == feature.openpi.action_chunk
    assert cfg.actor.model.action_dim == feature.action_dim == 7
    assert cfg.rollout.pipeline_stage_num == 1
    assert cfg.rollout.collect_transitions
    ur_algorithm = OmegaConf.to_container(cfg.algorithm, resolve=True)
    warmup_cache_path = Path(ur_algorithm["rlt_schedule"].pop("warmup_cache_path"))
    assert warmup_cache_path == (
        Path(
            "/home/suchenyu/RLinf-rltoken/logs/online_rltoken/warmup/insert_black_rack_into_blue_slot.pt"
        )
    )
    franka_algorithm = OmegaConf.to_container(franka.algorithm, resolve=True)
    # UR uses data-budgeted updates and a larger replay window.
    for key, expected in (("update_epoch", 5), ("critic_actor_ratio", 2)):
        assert ur_algorithm.pop(key) == expected
        franka_algorithm.pop(key)
    assert not ur_algorithm["rlt_schedule"].pop("continuous_updates")
    franka_algorithm["rlt_schedule"].pop("continuous_updates")
    assert ur_algorithm["rlt_schedule"].pop("train_every_transitions") == 1
    for key in ("cache_size", "sample_window_size"):
        assert ur_algorithm["replay_buffer"].pop(key) == 5000
        franka_algorithm["replay_buffer"].pop(key)
    assert ur_algorithm == franka_algorithm
    assert cfg.algorithm.rlt_transition_replay
    assert cfg.algorithm.rlt_schedule.enable
    assert not cfg.algorithm.rlt_schedule.continuous_updates
    assert cfg.algorithm.rlt_schedule.warmup_min_size == 150
    assert cfg.algorithm.rlt_schedule.warmup_post_collect_updates == 150
    from rlinf.config import validate_episode_replay_cfg

    validate_episode_replay_cfg(cfg)
    ur_actor = OmegaConf.to_container(cfg.actor, resolve=True)
    franka_actor = OmegaConf.to_container(franka.actor, resolve=True)
    ur_actor["model"].pop("num_action_chunks")
    franka_actor["model"].pop("num_action_chunks")
    assert ur_actor == franka_actor
    assert cfg.runner.save_interval == 250
    assert cfg.runner.resume_dir is None
    assert feature.num_steps == franka.rollout.rlt_feature_model.num_steps == 4
    for mode in ("train", "eval"):
        env = cfg.env[mode]
        baseline = franka.env[mode]
        assert "ur_evaluation" not in env
        for key in (
            "keyboard_reward_wrapper",
            "ignore_terminations",
            "auto_reset",
        ):
            assert env[key] == baseline[key]
        assert env.max_episode_chunks == (60 if mode == "train" else 200)
        assert env.max_chunks_per_rollout_epoch == (50 if mode == "train" else 20)
        assert env.get("flush_replay_on_episode_end", False) == (mode == "train")
        assert (
            "max_episode_steps" not in env and "max_steps_per_rollout_epoch" not in env
        )
        assert env.init_params.id == "UR5eEnv-v1"
        assert env.override_cfg.gripper_close_threshold == 0.5
        assert env.keyboard_reward_wrapper == "rlt_policy_switch"
        assert not env.override_cfg.get("manual_episode_control_only", False)
        assert env.teleop[0].keyboard.step_size_pos == 0.03
        assert env.teleop[0].keyboard.step_size_pos_alt == 0.006
        assert env.keyboard_wrapper_kwargs.success_key == "Key.space"
        assert env.keyboard_wrapper_kwargs.success_double_press_interval == 0.5
        assert env.keyboard_wrapper_kwargs.wait_for_start
        assert env.keyboard_wrapper_kwargs.switch_key == "Key.dot"
        assert env.keyboard_wrapper_kwargs.switch_double_press_interval == 0.5
        assert env.override_cfg.target_ee_pose == env.override_cfg.reset_ee_pose
        assert env.state_keys == inference.env.eval.state_keys
        for key in (
            "reference_frame",
            "action_scale",
            "reset_ee_pose",
            "ee_pose_limit_min",
            "ee_pose_limit_max",
        ):
            assert env.override_cfg[key] == inference.env.eval.override_cfg[key]

        assert env.override_cfg.step_frequency > 0
        assert inference.env.eval.override_cfg.step_frequency > 0


@pytest.fixture
def keyboard_reset_env(monkeypatch, tmp_path):
    from hydra import compose, initialize_config_dir
    from robot_mocks.teleop import keyboard

    from rlinf.envs.real.env import RealWorldEnv
    from rlinf.robotics import RobotInfo, UR5eConfig
    from rlinf.scheduler import AcceleratorType, WorkerInfo
    from rlinf.scheduler.manager.worker_manager import WorkerAddress

    sdk = keyboard()
    monkeypatch.setitem(sys.modules, "evdev", sdk)
    monkeypatch.setenv("RLINF_KEYBOARD_DEVICE", "/dev/input/event-fake")
    monkeypatch.delenv("RLINF_FAKE_KEYS", raising=False)
    root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("EMBODIED_PATH", str(root / "examples/embodiment"))
    envs = []

    def build(
        robot_type="ur5e",
        interval=0.5,
        wait_for_start=False,
        reset=True,
        ur_eval=False,
        stage2_enabled=False,
        action_timing_enabled=True,
        save_logs=True,
        rlt_mode=False,
        episode_replay=False,
        episode_chunks=None,
    ):
        with initialize_config_dir(
            config_dir=str(root / "evaluations/realworld"), version_base="1.3"
        ):
            cfg = compose(config_name="realworld_ur5e_eval_pi05_sft")
        cfg.env.eval.override_cfg.is_dummy = True
        cfg.env.eval.override_cfg.ee_pose_limit_min = [-1] * 6
        cfg.env.eval.override_cfg.ee_pose_limit_max = [1] * 6
        cfg.env.eval.override_cfg.max_num_steps = 1
        cfg.env.eval.video_cfg.save_video = False
        cfg.env.eval.max_episode_steps = 1
        cfg.env.eval.ur_evaluation.enabled = ur_eval
        cfg.env.eval.ur_evaluation.stage2_enabled = stage2_enabled
        cfg.env.eval.ur_evaluation.action_timing_enabled = action_timing_enabled
        cfg.env.eval.ur_evaluation.save_logs = save_logs
        cfg.env.eval.ur_evaluation.log_dir = str(tmp_path / "logs")
        cfg.env.eval.ur_evaluation.num_episodes = 2
        cfg.env.eval.ur_evaluation.wait_for_start = wait_for_start
        cfg.env.eval.ur_evaluation.output_path = str(
            tmp_path / f"results-{len(envs)}.json"
        )
        cfg.env.eval.keyboard_reward_wrapper = None if ur_eval else "double_enter_reset"
        from omegaconf import open_dict

        with open_dict(cfg.env.eval):
            cfg.env.eval.keyboard_wrapper_kwargs = {
                "double_press_interval": interval,
                "wait_for_start": wait_for_start,
            }
        if rlt_mode:
            cfg.env.eval.keyboard_reward_wrapper = "rlt_policy_switch"
            cfg.env.eval.keyboard_wrapper_kwargs = {
                "switch_key": "Key.f8",
                "success_key": "Key.space",
                "start_with_actor": True,
            }
            cfg.env.eval.override_cfg.max_num_steps = 300
            cfg.env.eval.max_episode_steps = 300
            cfg.env.eval.auto_reset = True
        if episode_chunks is not None:
            with open_dict(cfg.env.eval):
                cfg.env.eval.flush_replay_on_episode_end = episode_replay
                cfg.env.eval.max_episode_chunks = episode_chunks
                cfg.env.eval.episode_action_limit = episode_chunks * 3
        serials = ["348522072391", "944622073455"]
        if robot_type == "franka":
            from dataclasses import fields

            from rlinf.envs.real.franka.base import FrankaEnvConfig

            allowed = {field.name for field in fields(FrankaEnvConfig)}
            cfg.env.eval.override_cfg = {
                key: value
                for key, value in cfg.env.eval.override_cfg.items()
                if key in allowed
            }
            cfg.env.eval.override_cfg.enable_random_reset = False
            cfg.env.eval.use_relative_frame = True
            cfg.env.eval.init_params.id = "FrankaEnv-v1"
            hardware = RobotInfo(
                type="Franka",
                model="Franka",
                config=FrankaConfig(node_rank=0, camera_serials=serials),
            )
        else:
            hardware = RobotInfo(
                type="UR5e",
                model="UR5e",
                config=UR5eConfig(node_rank=0, camera_serials=serials),
            )
        info = WorkerInfo(
            address=WorkerAddress("KeyboardTest", 0),
            rank=0,
            group_world_size=1,
            cluster_node_rank=0,
            accelerator_type=AcceleratorType.NO_ACCEL,
            accelerator_model="",
            accelerator_rank=0,
            node_ip="127.0.0.1",
            node_port=0,
            available_accelerators=[],
            hardware_infos=[hardware],
        )
        env = RealWorldEnv(cfg.env.eval, 1, 0, 1, info)
        envs.append(env)
        if rlt_mode:
            from rlinf.envs.real.wrappers.episode import KeyboardRLTPolicySwitchWrapper

            outer = env.env.envs[0]
            while isinstance(outer.env, gym.Wrapper):
                outer = outer.env
            outer.env = KeyboardRLTPolicySwitchWrapper(
                outer.env, **cfg.env.eval.keyboard_wrapper_kwargs
            )
        if reset:
            env.reset()
        return env, sdk.devices[-1]

    yield build
    for env in envs:
        env.close()
    assert all(device.closed for device in sdk.devices)


def _press_keys(device, *keys):
    edges = [edge for key in keys for edge in [(key, 1), (key, 0)]]
    assert device.emit(*edges).wait(2), "keyboard listener did not consume the edges"


@pytest.fixture
def rlt_keyboard_control(monkeypatch):
    from robot_mocks.teleop import keyboard

    from rlinf.envs.real.ur5e import UR5eEnv
    from rlinf.envs.real.wrappers.episode import KeyboardRLTPolicySwitchWrapper
    from rlinf.robotics import RobotInfo, UR5eConfig

    sdk = keyboard()
    monkeypatch.setitem(sys.modules, "evdev", sdk)
    monkeypatch.setenv("RLINF_KEYBOARD_DEVICE", "/dev/input/event-fake")
    monkeypatch.delenv("RLINF_FAKE_KEYS", raising=False)
    envs = []

    def build(start_with_actor=False, manual_reward=True, double_dot=False, reset=True):
        raw = UR5eEnv(
            {
                "is_dummy": True,
                "enable_camera_player": False,
                "step_frequency": 10000.0,
            },
            robot_info=RobotInfo(
                type="UR5e",
                model="UR5e",
                config=UR5eConfig(node_rank=0, camera_serials=["dummy"]),
            ),
        )
        env = KeyboardRLTPolicySwitchWrapper(
            raw,
            success_key="Key.space" if manual_reward else None,
            switch_key="Key.dot" if double_dot else "Key.f8",
            switch_double_press_interval=0.5 if double_dot else None,
            start_with_actor=start_with_actor,
        )
        envs.append(env)
        if reset:
            env.reset()
        return env, sdk.devices[-1]

    yield build
    for env in envs:
        env.close()


@pytest.mark.parametrize("manual_reward", [False, True])
def test_rlt_double_dot_switch_uses_press_edges_and_resets(
    rlt_keyboard_control, manual_reward
):
    env, reader = rlt_keyboard_control(double_dot=True, manual_reward=manual_reward)
    action = np.zeros(7)
    _press_keys(reader, "f8", "dot")
    assert not env.step(action)[-1]["rlt_switch_flags"]
    assert reader.emit(("dot", 2), ("dot", 0)).wait(2)
    assert not env.step(action)[-1]["rlt_switch_flags"]
    time.sleep(0.55)
    _press_keys(reader, "dot")
    assert not env.step(action)[-1]["rlt_switch_flags"]
    env.reset()
    _press_keys(reader, "dot")
    assert not env.step(action)[-1]["rlt_switch_flags"]
    env.reset()
    _press_keys(reader, "dot", "dot")
    time.sleep(0.55)
    assert env.step(action)[-1]["rlt_switch_flags"]
    _press_keys(reader, "dot", "dot")
    assert env.step(action)[-1]["rlt_switch_flags"]
    if manual_reward:
        _press_keys(reader, "space", "space")
        assert env.step(action)[1:4] == (1, True, False)
    env.reset()
    assert not env.step(action)[-1]["rlt_switch_flags"]


def test_rlt_start_requires_fresh_enter_after_each_reset(rlt_keyboard_control):
    from concurrent.futures import ThreadPoolExecutor

    env, reader = rlt_keyboard_control(double_dot=True)
    env.wait_for_start = True
    with ThreadPoolExecutor(max_workers=1) as executor:
        try:
            for key in ("enter", "kpenter"):
                _press_keys(reader, "enter")
                previous = env._accept_after
                pending = executor.submit(env.reset)
                deadline = time.monotonic() + 2
                while env._accept_after <= previous and time.monotonic() < deadline:
                    time.sleep(0.01)
                time.sleep(0.05)
                assert not pending.done()
                _press_keys(reader, "dot", "dot", "space", "space")
                time.sleep(0.05)
                assert not pending.done()
                _press_keys(reader, key)
                _, info = pending.result(timeout=2)
                assert not info["rlt_switch_flags"]
                assert not env.rlt_switch_flags
                assert env.step(np.zeros(7))[1:4] == (0, False, False)
            pending = executor.submit(env.reset)
            time.sleep(0.1)
            assert not pending.done()
            env.close()
            with pytest.raises(RuntimeError, match="closed while waiting for Enter"):
                pending.result(timeout=2)
        finally:
            env.close()


def test_rlt_reset_and_start_need_separate_enter_edges(rlt_keyboard_control):
    from concurrent.futures import ThreadPoolExecutor

    env, reader = rlt_keyboard_control(double_dot=True)
    env.wait_for_reset = env.wait_for_start = True
    with ThreadPoolExecutor(max_workers=1) as executor:
        try:
            for _ in range(2):
                _press_keys(reader, "dot", "dot")
                assert env.step(np.zeros(7))[-1]["rlt_switch_flags"]
                _press_keys(reader, "enter")
                pending = executor.submit(env.reset)
                time.sleep(0.1)
                assert not pending.done()
                assert env.rlt_switch_flags  # Episode reset has not run yet.
                _press_keys(reader, "enter", "enter")
                time.sleep(0.1)
                assert not env.rlt_switch_flags
                assert not pending.done()  # The queued second press cannot start.
                _press_keys(reader, "enter")
                pending.result(timeout=2)
            pending = executor.submit(env.reset)
            time.sleep(0.1)
            env.close()
            with pytest.raises(RuntimeError, match="closed while waiting for Enter"):
                pending.result(timeout=2)
        finally:
            env.close()


def test_rlt_first_reset_confirmation_is_not_repeated(rlt_keyboard_control):
    from concurrent.futures import ThreadPoolExecutor

    env, reader = rlt_keyboard_control(double_dot=True, reset=False)
    env.wait_for_first_reset = env.wait_for_start = True
    with ThreadPoolExecutor(max_workers=1) as executor:
        try:
            pending = executor.submit(env.reset)
            time.sleep(0.1)
            assert not env._initial_reset_complete
            assert not pending.done()
            _press_keys(reader, "enter", "enter")
            time.sleep(0.1)
            assert env._initial_reset_complete
            assert not pending.done()
            _press_keys(reader, "enter")
            pending.result(timeout=2)
            for _ in range(2):
                _press_keys(reader, "dot", "dot")
                assert env.step(np.zeros(7))[-1]["rlt_switch_flags"]
                pending = executor.submit(env.reset)
                time.sleep(0.1)
                assert not env.rlt_switch_flags  # Reset ran without confirmation.
                assert not pending.done()
                _press_keys(reader, "enter")
                pending.result(timeout=2)
        finally:
            env.close()


def test_rlt_success_requires_two_space_press_edges(rlt_keyboard_control):
    env, reader = rlt_keyboard_control()
    action = np.zeros(7)
    _press_keys(reader, "space")
    assert env.step(action)[1:4] == (0, False, False)
    assert reader.emit(("space", 2), ("space", 0)).wait(2)
    assert env.step(action)[1:4] == (0, False, False)
    _press_keys(reader, "space")
    assert env.step(action)[1:4] == (1, True, False)
    before = env.unwrapped.num_steps
    assert env.step(action)[1:4] == (0, True, False)
    assert env.unwrapped.num_steps == before


def test_rlt_success_uses_press_times_and_clears_on_reset(rlt_keyboard_control):
    env, reader = rlt_keyboard_control()
    action = np.zeros(7)
    _press_keys(reader, "space")
    time.sleep(0.55)
    _press_keys(reader, "space")
    assert env.step(action)[1:4] == (0, False, False)
    env.reset()
    _press_keys(reader, "space")
    assert env.step(action)[1:4] == (0, False, False)
    env.reset()
    _press_keys(reader, "space", "space")
    time.sleep(0.55)
    assert env.step(action)[1:4] == (1, True, False)


@pytest.mark.parametrize("manual_reward", [False, True])
def test_rlt_can_start_with_actor_without_switch_key(
    rlt_keyboard_control, manual_reward
):
    env, reader = rlt_keyboard_control(
        start_with_actor=True, manual_reward=manual_reward
    )
    assert env.rlt_switch_flags
    assert env.reset()[1]["rlt_switch_flags"]
    assert env.step(np.zeros(7))[-1]["rlt_switch_flags"]
    env.reset()
    assert env.rlt_switch_flags
    default, reader = rlt_keyboard_control()
    _press_keys(reader, "b")
    assert not default.step(np.zeros(7))[-1]["rlt_switch_flags"]
    _press_keys(reader, "f8")
    assert default.step(np.zeros(7))[-1]["rlt_switch_flags"]


@pytest.mark.parametrize("robot_type", ["ur5e", "franka"])
def test_double_enter_resets_each_robot_without_marking_success(
    keyboard_reset_env, robot_type
):
    env, device = keyboard_reset_env(robot_type)
    actions = np.zeros((1, 7), dtype=np.float32)
    _, reward, terminated, truncated, _ = env.step(actions)
    assert not terminated.any() and not truncated.any()
    assert not reward.any()
    assert env.elapsed_steps[0] == 1
    before = env.env.envs[0].unwrapped.num_steps
    _press_keys(device, "enter", "enter")
    _, reward, terminated, truncated, info = env.step(actions)
    assert truncated.all() and not terminated.any()
    assert env.elapsed_steps[0] == 0
    assert env.env.envs[0].unwrapped.num_steps == before
    assert not reward.any() and not info["final_info"]["episode"]["success_once"].any()
    # The next action belongs to the new episode, with no stale reset request.
    _, _, terminated, truncated, _ = env.step(actions)
    assert not terminated.any() and not truncated.any()
    assert env.env.envs[0].unwrapped.num_steps == before + 1


def test_keyboard_reset_discards_the_rest_of_an_action_chunk(keyboard_reset_env):
    env, device = keyboard_reset_env()
    _press_keys(device, "enter", "enter")
    before = env.env.envs[0].unwrapped.num_steps
    _, rewards, terminations, truncations, infos = env.chunk_step(
        np.zeros((1, 3, 7), dtype=np.float32)
    )
    assert env.env.envs[0].unwrapped.num_steps == before
    assert not rewards.any() and not terminations.any()
    assert truncations[0, -1]
    assert "final_observation" in infos[-1]
    assert env.elapsed_steps[0] == 0


def test_slow_enters_and_key_repeat_do_not_reset(keyboard_reset_env):
    env, device = keyboard_reset_env(interval=0.05)
    actions = np.zeros((1, 7), dtype=np.float32)
    _press_keys(device, "enter")
    time.sleep(0.08)
    _press_keys(device, "enter")
    _, _, _, truncated, _ = env.step(actions)
    assert not truncated.any()
    time.sleep(0.08)
    assert device.emit(("enter", 1), ("enter", 2), ("enter", 2), ("enter", 0)).wait(2)
    _, _, _, truncated, _ = env.step(actions)
    assert not truncated.any()


def test_reset_uses_capture_times_even_when_inference_delays_polling(
    keyboard_reset_env,
):
    env, device = keyboard_reset_env(interval=0.05)
    _press_keys(device, "kpenter", "kpenter")
    time.sleep(0.08)
    _, _, _, truncated, _ = env.step(np.zeros((1, 7), dtype=np.float32))
    assert truncated.all()


def test_keys_from_before_reset_do_not_reset_the_next_episode(keyboard_reset_env):
    env, device = keyboard_reset_env()
    _press_keys(device, "enter", "enter")
    env.reset()
    _, _, terminated, truncated, _ = env.step(np.zeros((1, 7), dtype=np.float32))
    assert not terminated.any() and not truncated.any()


def _wait_for_keyboard_start(env):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        if env.env.envs[0].get_wrapper_attr("waiting_for_start"):
            return
        time.sleep(0.005)
    pytest.fail("reset did not reach the operator start wait")


@pytest.mark.parametrize("robot_type", ["ur5e", "franka"])
def test_single_enter_starts_after_initial_and_manual_reset(
    keyboard_reset_env, robot_type
):
    env, device = keyboard_reset_env(robot_type, wait_for_start=True, reset=False)
    actions = np.zeros((1, 7), dtype=np.float32)
    with ThreadPoolExecutor(max_workers=1) as executor:
        initial = executor.submit(env.reset)
        try:
            _wait_for_keyboard_start(env)
            assert not initial.done()
            assert env.env.envs[0].unwrapped.num_steps == 0
            _press_keys(device, "enter")
            initial.result(timeout=2)
            # Starting with Enter must not become the first half of a reset double press.
            _press_keys(device, "enter")
            _, _, _, truncated, _ = env.step(actions)
            assert not truncated.any()
            before = env.env.envs[0].unwrapped.num_steps
            time.sleep(0.55)
            _press_keys(device, "enter", "enter")
            reset = executor.submit(env.step, actions)
            _wait_for_keyboard_start(env)
            assert not reset.done()
            assert env.env.envs[0].unwrapped.num_steps == before
            _press_keys(device, "kpenter")
            _, _, terminated, truncated, _ = reset.result(timeout=2)
            assert truncated.all() and not terminated.any()
            assert env.elapsed_steps[0] == 0
            _, _, _, truncated, _ = env.step(actions)
            assert not truncated.any()
            assert env.env.envs[0].unwrapped.num_steps == before + 1
        finally:
            env.close()


def test_closing_keyboard_control_interrupts_the_start_wait(keyboard_reset_env):
    env, _ = keyboard_reset_env(wait_for_start=True, reset=False)
    with ThreadPoolExecutor(max_workers=1) as executor:
        waiting = executor.submit(env.reset)
        try:
            _wait_for_keyboard_start(env)
            env.close()
            with pytest.raises(RuntimeError, match="closed while waiting for Enter"):
                waiting.result(timeout=2)
        finally:
            env.close()


def test_ur_tcp_reset_range_and_seed_match_the_config(keyboard_reset_env):
    from scipy.spatial.transform import Rotation

    env, _ = keyboard_reset_env()
    robot = env.env.envs[0].unwrapped
    nominal = np.asarray(robot.config.reset_ee_pose)
    robot.config.enable_random_reset = True
    robot.config.reset_pose_range = np.array([0.03, 0.03, 0.03, 0.0, 0.0, 0.0])
    bounds = np.asarray(robot.config.reset_pose_range)
    env.reset(seed=123)
    sampled = robot.sample_reset_pose()
    expected_xyz = nominal[:3] + np.random.default_rng(123).uniform(
        -bounds[:3], bounds[:3]
    )
    np.testing.assert_allclose(sampled[:3], expected_xyz)
    np.testing.assert_allclose(
        sampled[3:], Rotation.from_euler("xyz", nominal[3:]).as_quat()
    )
    env.reset(seed=123)
    np.testing.assert_array_equal(robot.sample_reset_pose(), sampled)
    env.reset(seed=124)
    assert not np.array_equal(robot.sample_reset_pose()[:3], sampled[:3])
    robot.config.enable_random_reset = False
    fixed = robot.sample_reset_pose()
    np.testing.assert_array_equal(fixed[:3], nominal[:3])
    np.testing.assert_allclose(fixed[3:], sampled[3:])


@pytest.mark.parametrize("reference_frame", ["tcp", "base"])
def test_ur5e_target_composition_preserves_local_rotation_and_tool_offset(
    reference_frame,
):
    from scipy.spatial.transform import Rotation as R

    from rlinf.envs.real.ur5e.motion import URTargetComposer, ee_to_tcp, tcp_to_ee

    ee = np.r_[[0.4, 0.1, 0.3], R.from_euler("xyz", [0.2, -0.3, 1.0]).as_quat()]
    offset = np.r_[[0.02, 0.01, 0.1], R.from_euler("x", 0.15).as_quat()]
    tcp = ee_to_tcp(ee, offset)
    np.testing.assert_allclose(tcp_to_ee(tcp, offset), ee, atol=1e-12)
    action = np.array([0.001, -0.002, 0.003, 0.0, 0.0, 0.0])
    composer = URTargetComposer()
    result = composer.target(
        tcp,
        offset,
        action,
        reference_frame=reference_frame,
        hold_enabled=False,
        position_threshold=0.0001,
        rotation_threshold=0.001,
    )
    result_ee = tcp_to_ee(result, offset)
    delta = (
        R.from_quat(ee[3:]).apply(action[:3])
        if reference_frame == "tcp"
        else action[:3]
    )
    np.testing.assert_allclose(result_ee[:3], ee[:3] + delta, atol=1e-12)
    np.testing.assert_allclose(
        R.from_quat(result_ee[3:]).as_matrix(),
        R.from_quat(ee[3:]).as_matrix(),
        atol=1e-12,
    )
    action[3:6] = [0.1, -0.08, 0.04]
    result = composer.target(
        tcp,
        offset,
        action,
        reference_frame=reference_frame,
        hold_enabled=False,
        position_threshold=0.0001,
        rotation_threshold=0.001,
    )
    local = R.from_euler("xyz", action[3:6])
    expected = (
        R.from_quat(ee[3:]) * local
        if reference_frame == "tcp"
        else local * R.from_quat(ee[3:])
    )
    np.testing.assert_allclose(
        R.from_quat(tcp_to_ee(result, offset)[3:]).as_matrix(),
        expected.as_matrix(),
        atol=1e-12,
    )


def test_ur5e_idle_target_holds_axes_until_motion_resumes():
    from rlinf.envs.real.ur5e.motion import URTargetComposer

    composer = URTargetComposer()
    tcp = np.array([0.4, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0])
    offset = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    settings = {
        "reference_frame": "tcp",
        "hold_enabled": True,
        "position_threshold": 0.0001,
        "rotation_threshold": 0.001,
    }
    first = composer.target(tcp, offset, np.zeros(6), **settings)
    disturbed = tcp.copy()
    disturbed[0] += 0.002
    held = composer.target(disturbed, offset, np.zeros(6), **settings)
    np.testing.assert_allclose(held, first)
    action = np.array([0.003, 0.0, 0.0, 0.0, 0.0, 0.0])
    moving = composer.target(disturbed, offset, action, **settings)
    assert moving[0] == pytest.approx(disturbed[0] + 0.003)
    stopped = composer.target(disturbed, offset, np.zeros(6), **settings)
    assert stopped[0] == pytest.approx(disturbed[0])
    composer.clear()
    assert composer.anchor is None


def test_ur5e_observation_keeps_base_velocity_and_uses_ee_reset_reference():
    from scipy.spatial.transform import Rotation as R

    from rlinf.envs.real.ur5e import UR5eEnv
    from rlinf.envs.real.ur5e.motion import ee_to_tcp
    from rlinf.robotics import RobotInfo, UR5eConfig

    env = UR5eEnv(
        {"is_dummy": True},
        robot_info=RobotInfo(
            type="UR5e",
            model="UR5e",
            config=UR5eConfig(node_rank=0, camera_serials=["dummy"]),
        ),
    )
    env.config.is_dummy = False
    env._last_frames = {}
    env._reference_ee_pose = np.r_[
        [0.4, 0.0, 0.3], R.from_euler("z", np.pi / 2).as_quat()
    ]
    env._tcp_offset = np.r_[[0.0, 0.0, 0.1], [0.0, 0.0, 0.0, 1.0]]
    ee = env._reference_ee_pose.copy()
    ee[1] += 0.01
    env._robot_state.tcp_pose = ee_to_tcp(ee, env._tcp_offset)
    env._robot_state.tcp_vel = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    env._robot_state.gripper_position = 0.8
    obs = env._get_observation()["state"]
    np.testing.assert_allclose(obs["tcp_pose"][:3], [0.01, 0.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(obs["tcp_vel"], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])


@pytest.mark.parametrize("timing_enabled", [False, True])
@pytest.mark.parametrize("continuous", [False, True])
@pytest.mark.parametrize("outcome", ["success", "timeout", "reset", "error"])
def test_ur5e_online_rl_force_control_lifecycle(
    monkeypatch, tmp_path, continuous, outcome, timing_enabled, caplog
):
    import json

    caplog.set_level("INFO")
    from robot_mocks import mocked_sdks

    from rlinf.envs.real.ur5e import UR5eEnv
    from rlinf.envs.real.utils.video import VideoPlayer
    from rlinf.envs.real.wrappers import build_stack
    from rlinf.robotics import RobotInfo, UR5eConfig, UR5eRobot
    from rlinf.robotics.parts.arms.ur5e import UR5eArm
    from rlinf.robotics.parts.end_effectors.grippers.pgi import PGIGripper

    monkeypatch.setenv("RLINF_KEYBOARD_DEVICE", "/dev/input/event-fake")
    monkeypatch.delenv("RLINF_FAKE_KEYS", raising=False)
    with mocked_sdks() as sdks:
        arm = UR5eArm(
            "192.0.2.98",
            impedance={"continuous_tracking": continuous, "start_paused": True},
        )
        gripper = PGIGripper(str(tmp_path / "online-gripper"))
        robot = UR5eRobot(arm=arm, end_effector=gripper)
        robot.connect()
        raw = UR5eEnv(
            {
                "is_dummy": True,
                "enable_camera_player": False,
                "ee_pose_limit_min": [-1] * 6,
                "ee_pose_limit_max": [1] * 6,
                "reset_ee_pose": [0.4, 0, 0.3, 0, 0, 0],
                "payload_settle_seconds": 0,
                "ft_zero_settle_seconds": 0,
                "max_num_steps": 300,
            },
            robot_info=RobotInfo(
                type="UR5e",
                model="UR5e",
                config=UR5eConfig(node_rank=0, camera_serials=["dummy"]),
            ),
        )
        raw.config.is_dummy = False
        raw.robot, raw._arm, raw._end_effector = robot, arm, gripper
        raw._cameras = {}
        raw.camera_player = VideoPlayer(False)
        raw._robot_state = arm.get_state()
        env = build_stack(
            raw,
            {
                "keyboard_reward_wrapper": "rlt_policy_switch",
                "keyboard_wrapper_kwargs": {"success_key": "Key.space"},
                "max_episode_chunks": 1 if outcome == "timeout" else 150,
                "episode_action_limit": 2 if outcome == "timeout" else 300,
                "max_episode_steps": 1,  # Legacy value must not end this episode early.
                "online_timing_enabled": timing_enabled,
            },
        )
        assert arm._control_timing_enabled == timing_enabled
        reader = sdks["evdev"].devices[-1]
        state = sdks["rtde_control"].states["192.0.2.98"]
        action = np.zeros(7)
        try:
            time.sleep(0.025)
            assert not state["forces"]
            for _ in range(2):
                before = len(state["forces"])
                env.reset()
                time.sleep(0.025)
                assert len(state["forces"]) == before
                env.get_wrapper_attr("begin_action_chunk")(2)
                env.step(action)
                assert len(state["forces"]) > before
                if outcome == "reset":
                    env.reset()
                elif outcome == "error":
                    with pytest.raises(ValueError):
                        env.step(np.zeros(3))
                else:
                    if outcome == "success":
                        _press_keys(reader, "space", "space")
                    _, reward, terminated, truncated, _ = env.step(action)
                    assert (terminated, truncated) == (
                        outcome == "success",
                        outcome == "timeout",
                    )
                    assert reward == (1.0 if outcome == "success" else 0.0)
                after = len(state["forces"])
                time.sleep(0.025)
                assert len(state["forces"]) == after
                if outcome in ("success", "timeout"):
                    assert env.step(action)[1] == 0.0
                    time.sleep(0.025)
                    assert len(state["forces"]) == after
            env.reset()
            env.get_wrapper_attr("begin_action_chunk")(1)
            env.step(action)
            env.close()
            after = len(state["forces"])
            time.sleep(0.025)
            assert len(state["forces"]) == after
        finally:
            env.close()
            robot.disconnect()

    records = [
        json.loads(record.message.split("[TIMING_RECORD] ")[1])
        for record in caplog.records
        if "[TIMING_RECORD] " in record.message
    ]
    assert bool(records) == timing_enabled
    if timing_enabled:
        actions = [row for row in records if row["source"] == "UR action"]
        assert actions and all(row["chunk"] == 1 for row in actions)
        assert actions[0]["episode"] == 1 and actions[0]["action"] == 1
        assert actions[-1]["episode"] > 1 and actions[-1]["action"] == 1
        pd = [row for row in records if row["source"] == "UR background"]
        assert bool(pd) == continuous
        assert all(row["samples"] > 0 for row in pd)


@pytest.mark.parametrize("action_timing_enabled", [False, True])
@pytest.mark.parametrize("save_logs", [False, True])
def test_ur5e_evaluation_records_success_failure_and_discards_old_chunk(
    keyboard_reset_env,
    action_timing_enabled,
    save_logs,
    caplog,
):
    import json

    caplog.set_level("INFO")
    env, device = keyboard_reset_env(
        ur_eval=True, action_timing_enabled=action_timing_enabled, save_logs=save_logs
    )
    action = np.zeros((1, 7), dtype=np.float32)
    raw = env.env.envs[0]
    env.step(action)
    before = raw.unwrapped.num_steps
    _press_keys(device, "space", "space")
    _, reward, terminated, truncated, info = env.chunk_step(
        np.zeros((1, 3, 7), dtype=np.float32)
    )
    assert raw.unwrapped.num_steps == before
    assert terminated.any() and reward.sum() == 1
    assert info[-1]["final_info"]["episode"]["success_once"].all()
    _press_keys(device, "enter", "enter")
    _, reward, terminated, truncated, _ = env.step(action)
    assert truncated.all() and not terminated.any() and not reward.any()
    assert raw.get_wrapper_attr("evaluation_complete")
    output = raw.get_wrapper_attr("output_path")
    assert output.exists() == save_logs
    assert not any(
        r.message.startswith(("[ACTION]", "[PD]", "[TIME]")) for r in caplog.records
    )
    assert any("[EPISODE]" in r.message for r in caplog.records)
    if not save_logs:
        assert not output.with_name("ur5e-actions.jsonl").exists()
        return
    payload = json.loads(output.read_text())
    trace_path = output.with_name("ur5e-actions.jsonl")
    assert trace_path.exists()
    traces = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert ("control_hz" in traces[0]) == action_timing_enabled
    assert traces[0]["action"] == [0.0] * 7
    if action_timing_enabled:
        assert len(traces) == 1
        assert traces[0]["episode_index"] == 1
        assert traces[0]["step"] == 1
        assert traces[0]["action"] == [0.0] * 7
        assert traces[0]["action_interval_sec"] is None
        assert traces[0]["pace_wait_sec"] > 0
    assert payload["summary"] == {
        "episode_count": 2,
        "success_count": 1,
        "success_rate": 0.5,
    }
    assert [r["status"] for r in payload["episodes"]] == [
        "manual_success",
        "keyboard_reset",
    ]
    assert not raw.get_wrapper_attr("waiting_for_start")


@pytest.mark.parametrize("save_logs", [False, True])
def test_ur5e_evaluation_saves_background_frequency_without_action_timing(
    keyboard_reset_env, save_logs, caplog
):
    import json

    from robot_mocks import mocked_sdks

    from rlinf.robotics.parts.arms.ur5e import UR5eArm

    env, _ = keyboard_reset_env(
        ur_eval=True, action_timing_enabled=False, save_logs=save_logs
    )
    raw = env.env.envs[0]
    with mocked_sdks():
        arm = UR5eArm("192.0.2.94", impedance={"continuous_tracking": True})
        arm.connect()
        raw.unwrapped._arm = arm
        raw.unwrapped.config.is_dummy = False
        try:
            with caplog.at_level("INFO"):
                env.close()
            assert not any("[PD]" in r.message for r in caplog.records)
            paths = raw.get_wrapper_attr("_run_log_paths")
            if save_logs:
                timings = [
                    json.loads(line) for line in paths[1].read_text().splitlines()
                ]
                [timing] = [r for r in timings if r.get("component") == "controller"]
                assert timing["event"] == "force_mode"
                assert timing["target_hz"] == 100.0
                assert timing["samples"] >= 1
                assert "measured_hz" in timing
                assert "send_interval_sec_max" in timing
            else:
                assert paths is None
        finally:
            arm.disconnect()


def test_ur5e_evaluation_timeout_prevents_action_execution(keyboard_reset_env):
    env, _ = keyboard_reset_env(ur_eval=True)
    raw = env.env.envs[0]
    controller = raw
    while type(controller).__name__ != "UREvaluationWrapper":
        controller = controller.env
    controller._episode_started -= 61
    before = raw.unwrapped.num_steps
    _, _, _, truncated, _ = env.step(np.zeros((1, 7), dtype=np.float32))
    assert truncated.all() and raw.unwrapped.num_steps == before
    assert controller.results[0]["status"] == "timeout"


def test_ur5e_warmup_prepares_observations_before_episode_clock(keyboard_reset_env):
    env, _ = keyboard_reset_env(ur_eval=True, reset=False)
    wrapped = env.env.envs[0]
    obs, _ = env.reset(options={"ur_eval_prepare": True})
    assert obs["states"].shape == (1, 19)
    assert not wrapped.get_wrapper_attr("waiting_for_start")
    assert wrapped.get_wrapper_attr("_episode_started") == 0.0
    with pytest.raises(RuntimeError, match="after UR warmup"):
        env.step(np.ones((1, 7), dtype=np.float32))
    started = time.monotonic()
    env.reset()
    assert wrapped.get_wrapper_attr("_episode_started") >= started
    assert wrapped.get_wrapper_attr("results") == []
    assert wrapped.unwrapped.num_steps == 0


@pytest.mark.parametrize("continuous", [False, True])
@pytest.mark.parametrize("outcome", ["manual_success", "keyboard_reset", "timeout"])
def test_ur5e_force_control_runs_only_between_enter_and_episode_end(
    monkeypatch, tmp_path, continuous, outcome
):
    from robot_mocks import mocked_sdks

    from rlinf.envs.real.ur5e import UR5eEnv
    from rlinf.envs.real.ur5e.evaluation import UREvaluationWrapper
    from rlinf.envs.real.utils.video import VideoPlayer
    from rlinf.robotics import RobotInfo, UR5eConfig, UR5eRobot
    from rlinf.robotics.parts.arms.ur5e import UR5eArm
    from rlinf.robotics.parts.end_effectors.grippers.pgi import PGIGripper

    monkeypatch.setenv("RLINF_KEYBOARD_DEVICE", "/dev/input/event-fake")
    monkeypatch.delenv("RLINF_FAKE_KEYS", raising=False)
    with mocked_sdks() as sdks:
        sdk = sdks["evdev"]
        arm = UR5eArm(
            "192.0.2.97",
            impedance={"continuous_tracking": continuous, "start_paused": True},
        )
        gripper = PGIGripper(str(tmp_path / "gripper"))
        robot = UR5eRobot(arm=arm, end_effector=gripper)
        robot.connect()
        raw = UR5eEnv(
            {
                "is_dummy": True,
                "enable_camera_player": False,
                "ee_pose_limit_min": [-1] * 6,
                "ee_pose_limit_max": [1] * 6,
                "reset_ee_pose": [0.4, 0, 0.3, 0, 0, 0],
                "payload_settle_seconds": 0,
                "ft_zero_settle_seconds": 0,
            },
            robot_info=RobotInfo(
                type="UR5e",
                model="UR5e",
                config=UR5eConfig(node_rank=0, camera_serials=["dummy"]),
            ),
        )
        raw.config.is_dummy = False
        raw.robot, raw._arm, raw._end_effector = robot, arm, gripper
        raw._cameras = {}
        raw.camera_player = VideoPlayer(False)
        raw._robot_state = arm.get_state()
        env = UREvaluationWrapper(
            raw, num_episodes=2, wait_for_start=True, save_logs=False
        )
        reader = sdk.devices[-1]
        state = sdks["rtde_control"].states["192.0.2.97"]
        action = np.zeros(7)
        try:
            env.reset(options={"ur_eval_prepare": True})
            time.sleep(0.025)
            assert not state["forces"]
            with ThreadPoolExecutor(max_workers=1) as executor:
                try:
                    for episode in range(2):
                        before = len(state["forces"])
                        waiting = executor.submit(env.reset)
                        deadline = time.monotonic() + 2
                        while not env.waiting_for_start and time.monotonic() < deadline:
                            time.sleep(0.005)
                        assert env.waiting_for_start and not waiting.done()
                        time.sleep(0.025)
                        assert len(state["forces"]) == before
                        _press_keys(reader, "enter")
                        waiting.result(timeout=2)
                        env.step(action)
                        assert len(state["forces"]) > before
                        if outcome == "timeout":
                            env._episode_started -= 61
                        else:
                            key = "space" if outcome == "manual_success" else "enter"
                            _press_keys(reader, key, key)
                        _, _, terminated, truncated, _ = env.step(action)
                        assert terminated or truncated
                        assert env.results[-1]["status"] == outcome
                        after = len(state["forces"])
                        time.sleep(0.025)
                        env.step(action)
                        assert len(state["forces"]) == after
                        assert len(env.results) == episode + 1
                    env.reset()
                    after = len(state["forces"])
                    time.sleep(0.025)
                    assert len(state["forces"]) == after
                    assert env.evaluation_complete
                finally:
                    env.close()
        finally:
            env.close()
            robot.disconnect()


@pytest.mark.parametrize("save_logs", [False, True])
def test_ur5e_worker_keeps_step_timings_in_file_without_console(tmp_path, save_logs):
    import json
    from types import SimpleNamespace

    from omegaconf import OmegaConf

    from rlinf.workers.env.ur5e_eval_worker import record_timing

    messages = []
    worker = SimpleNamespace(
        cfg=OmegaConf.create(
            {
                "env": {"eval": {"ur_evaluation": {"save_logs": save_logs}}},
                "runner": {"logger": {"log_path": str(tmp_path)}},
            }
        ),
        log_info=messages.append,
    )
    for component in ("env", "rollout"):
        record_timing(worker, component, "step", infer_index=7, duration_sec=0.08)
    assert messages == []
    path = tmp_path / "logs" / "latest.jsonl"
    assert path.exists() == save_logs
    if save_logs:
        records = [json.loads(line) for line in path.read_text().splitlines()]
        assert [r["component"] for r in records] == ["env", "rollout"]
        assert all(r["duration_sec"] == 0.08 and r["infer_index"] == 7 for r in records)
    record_timing(worker, "rollout", "load", duration_sec=10.0)
    assert len(messages) == 1 and "load" in messages[0]


@pytest.mark.parametrize("action_timing_enabled", [False, True])
@pytest.mark.parametrize("save_logs", [False, True])
def test_ur5e_sync_worker_stops_after_episode_budget(
    keyboard_reset_env, tmp_path, action_timing_enabled, save_logs
):
    import json

    from omegaconf import OmegaConf

    from rlinf.workers.env.ur5e_eval_worker import URSyncEnvWorker

    env, device = keyboard_reset_env(
        ur_eval=True, action_timing_enabled=action_timing_enabled, save_logs=save_logs
    )
    worker = object.__new__(URSyncEnvWorker)
    worker.cfg = OmegaConf.create(
        {
            "env": {"eval": OmegaConf.to_container(env.cfg)},
            "rollout": {"group_name": "Rollout"},
            "runner": {"logger": {"log_path": str(tmp_path)}},
        }
    )
    worker.model_cfg = OmegaConf.create(
        {"model_type": "openpi", "num_action_chunks": 3, "action_dim": 7}
    )
    worker.stage_num = worker.eval_num_envs_per_stage = worker.eval_batch_size = 1
    worker.n_eval_chunk_steps = 100
    worker.eval_env_list = [env]
    worker.eval_prev_done = [None]
    worker.env_decoupled_mode = False
    worker.use_external_reward_model = worker.enable_rlt = False
    worker._obs_split_fn = None
    log_messages = []
    worker.log_info = log_messages.append
    sent = []
    worker.send_to = lambda **kwargs: sent.append(kwargs["data"])
    requests = []

    def recv(**kwargs):
        requests.append(kwargs)
        if len(requests) == 1:
            return torch.ones(1, 3, 7)
        _press_keys(device, "space", "space")
        return torch.zeros(1, 3, 7)

    worker.recv_from = recv
    result = asyncio.run(worker.evaluate(None, None))
    assert len(requests) == 3
    assert len(sent) == 4
    assert [bool(item["ur_eval_stop"].any()) for item in sent] == [
        False,
        False,
        False,
        True,
    ]
    assert [bool(item["ur_eval_warmup"].any()) for item in sent] == [
        True,
        False,
        False,
        False,
    ]
    assert result["success_once"].tolist() == [True, True]
    assert all(
        r["action_steps"] == 0 for r in env.env.envs[0].get_wrapper_attr("results")
    )
    assert (tmp_path / "logs" / "latest.jsonl").exists() == save_logs
    assert not any("[TIMING]" in message for message in log_messages)
    if not save_logs:
        assert env._closed
        return
    timings = [
        json.loads(line)
        for line in (tmp_path / "logs" / "latest.jsonl").read_text().splitlines()
    ]
    assert [row["chunk_id"] for row in timings] == [0, 1]
    assert all(row["action_hz"] > 0 and row["episode_boundary"] for row in timings)
    assert not any("[TIMING]" in message for message in log_messages)
    assert env._closed


def test_ur5e_rtc_worker_ends_only_after_final_episode(keyboard_reset_env):
    from rlinf.workers.env.ur5e_eval_worker import URRTCEnvWorker

    env, device = keyboard_reset_env(ur_eval=True)
    worker = object.__new__(URRTCEnvWorker)
    worker.eval_env_list = [env]
    messages = []
    worker.log_info = messages.append
    for complete in (False, True):
        _press_keys(device, "space", "space")
        output, _ = worker._evaluate_rtc_action(np.zeros((1, 7), dtype=np.float32), 0)
        assert output.dones.all()
        assert env.manual_episode_control_only is (not complete)


def test_ur5e_camera_input_keeps_full_view_and_matches_openpi_letterbox():
    from openpi_client.image_tools import resize_with_pad

    from rlinf.envs.real.ur5e import UR5eEnv
    from rlinf.robotics import RobotInfo, UR5eConfig

    env = UR5eEnv(
        {"is_dummy": True},
        robot_info=RobotInfo(
            type="UR5e",
            model="UR5e",
            config=UR5eConfig(node_rank=0, camera_serials=["dummy"]),
        ),
    )
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    rgb[:, :80] = [255, 10, 20]
    rgb[:, -80:] = [20, 30, 255]
    rgb[150:330, 250:390] = [0, 200, 0]
    env._cameras = {"cam": None}
    displayed = []
    env.camera_player = type(
        "Player", (), {"put_frame": lambda self, frames: displayed.append(frames)}
    )()
    frames, _ = env._frames_from({"cam": {"frame": rgb[..., ::-1]}})
    np.testing.assert_array_equal(frames["cam"], resize_with_pad(rgb, 224, 224))
    np.testing.assert_array_equal(displayed[0]["cam"], frames["cam"][..., ::-1])
    np.testing.assert_array_equal(displayed[0]["cam_full"], rgb[..., ::-1])
    assert not np.shares_memory(frames["cam"], displayed[0]["cam"])
    assert next(iter(env.observation_space["frames"].values())).shape == (224, 224, 3)


def test_ur5e_rollout_preserves_stop_flag_through_channel_batch_merge(tmp_path):
    import asyncio

    from omegaconf import OmegaConf

    from rlinf.workers.env.ur5e_eval_worker import URSyncRolloutWorker

    worker = object.__new__(URSyncRolloutWorker)
    worker.cfg = OmegaConf.create(
        {
            "env": {
                "group_name": "Env",
                "eval": {
                    "ur_evaluation": {
                        "save_logs": True,
                        "log_dir": str(tmp_path / "logs"),
                    }
                },
            },
            "runner": {"logger": {"log_path": str(tmp_path)}},
        }
    )
    worker.enable_offload = False
    worker.eval_batch_size = 1
    progress = []
    worker.log_info = progress.append
    messages = iter(
        [
            {
                "obs": {"states": torch.zeros(1, 19)},
                "final_obs": None,
                "ur_eval_stop": torch.tensor([stop]),
                "ur_infer_index": torch.tensor([0 if warmup else 7]),
                "ur_eval_warmup": torch.tensor([warmup]),
            }
            for stop, warmup in ((False, True), (False, False), (True, False))
        ]
    )

    class Received:
        async def async_wait(self):
            return worker._merge_obs_batches([next(messages)])

    def recv_from(**kwargs):
        # Match Worker.recv_from: only async_op=True returns an async work handle.
        if kwargs.get("async_op", False):
            return Received()
        return worker._merge_obs_batches([next(messages)])

    worker.recv_from = recv_from
    predictions = []

    def predict(obs, **kwargs):
        predictions.append(obs)
        return torch.zeros(1, 3, 7), {}

    worker._predict_rollout_actions = predict
    sent = []
    worker.send_to = lambda **kwargs: sent.append(kwargs["data"])
    asyncio.run(worker.evaluate(None, None))
    assert len(predictions) == len(sent) == 2
    assert any("[WARMUP]" in message and "Completed" in message for message in progress)

    import json

    import yaml

    logs = tmp_path / "logs"
    records = [
        yaml.safe_load(part)
        for part in (logs / "latest.yaml").read_text().split(" " + "-" * 60 + "\n")
        if part.strip()
    ]
    assert len(records) == 1
    assert records[0]["infer_time"] == records[0]["infer_index"] == 7
    assert list(records[0]) == [
        "episode_index",
        "episode_action_count",
        "action_horizon",
        "infer_time",
        "infer_index",
        "timestamp",
        "prompt",
        "state",
        "actions",
        "actions_delta",
    ]
    assert records[0]["state"] == [0.0] * 19
    assert records[0]["actions"] == [[0.0] * 7] * 3
    timings = [
        json.loads(line) for line in (logs / "latest.jsonl").read_text().splitlines()
    ]
    assert [row["infer_index"] for row in timings if row["event"] == "step"] == [7]


@pytest.fixture
def native_ur_force_engine(monkeypatch, tmp_path):
    for package in ("lerobot.cameras", "pyDHgripper", "av", "crcmod"):
        pytest.importorskip(package)
    from robot_mocks.ur5e import modules

    from evaluations.realworld.vla_precision_ur import ur
    from evaluations.realworld.vla_precision_ur.recording import Recorder
    from evaluations.realworld.vla_precision_ur.utils import FpsCounter

    sdk = modules()
    engine = object.__new__(ur.Inference)
    engine.__dict__.update(
        rtde_r=sdk["rtde_receive"].RTDEReceiveInterface("continuous-native"),
        rtde_c=sdk["rtde_control"].RTDEControlInterface("continuous-native"),
        continuous_tracking=True,
        _force_controller=None,
        _last_force_report=0.0,
        control_space="force",
        debug=False,
        fix_gripper=True,
        init_gripper=False,
        reference_frame="base",
        action_fps=50,
        kp=1200,
        kd=600,
        kp_rot=20,
        kd_rot=4,
        rtde_freq=100,
        pos_delta=0.048,
        vel_delta=4.0,
        use_control_frame=False,
        task_frame=[0.0] * 6,
        select_vector=[1, 1, 1, 0, 0, 1],
        force_type=2,
        force_limit=[2] * 6,
        filter_zero_output=False,
        _zero_force_lock_active=False,
        show_target_error=False,
        action_idle_hold_enabled=True,
        action_idle_position_threshold=0.0001,
        action_idle_rotation_threshold=0.001,
        action_timing_enabled=True,
        save_logs=True,
        _timing_path=tmp_path / "native-timing.jsonl",
        _velocity=0.2,
        _acceleration=0.2,
        payload_mass=1.601,
        payload_cog=[0.011, -0.002, 0.052],
        pre_reset_tcp_pose=None,
        init_tcp_pose=[0.4, 0.0, 0.3, 0.0, 0.0, 0.0],
        init_pose_range=[0.0] * 6,
        _episode_time_printer_stop=None,
        _episode_time_printer_thread=None,
        _ep_idx=1,
        num_episodes=1,
        ep_timeout=60,
        _success_episodes=0,
        _failed_episodes=0,
        _reset_requested=threading.Event(),
        _success_requested=threading.Event(),
        use_target_success=False,
        fps_action=FpsCounter("test"),
    )
    engine._clear_action_idle_hold()
    engine.recorder = object.__new__(Recorder)
    engine.recorder.save_logs = False
    engine.recorder.episode_results = []
    monkeypatch.setattr(ur, "PAYLOAD_SETTLE_TIME_S", 0)
    monkeypatch.setattr(ur, "FT_ZERO_SETTLE_TIME_S", 0)
    yield engine, sdk
    engine.close()


@pytest.mark.parametrize("chunk_size", [3, 5])
def test_vla_precision_ur_executes_and_records_entire_prediction_chunk(
    native_ur_force_engine, chunk_size
):
    import queue

    engine, _ = native_ur_force_engine
    engine.continuous_tracking = False
    engine._ep_steps = 0
    engine.recorder.save_logs = True
    engine.recorder.queue_log = queue.Queue()
    engine.recorder.queue_vis = queue.Queue()
    actions = np.full((chunk_size, 7), 0.001)
    obs = {"prompt": "test", "observation/state": np.zeros(19)}

    engine._submit_step({"actions": actions}, obs, infer_idx=7)

    assert len(engine.rtde_c.state["forces"]) == chunk_size
    record = engine.recorder.queue_log.get_nowait()
    assert record["action_horizon"] == chunk_size
    assert record["infer_index"] == 7
    np.testing.assert_allclose(record["actions"], actions[:, :6])
    assert engine.recorder.queue_vis.get_nowait() is obs


def test_native_ur_continuous_force_tracks_state_and_pauses_for_reset(
    native_ur_force_engine, monkeypatch
):
    engine, sdk = native_ur_force_engine
    state = engine.rtde_c.state
    zero_sensor = sdk["rtde_control"].RTDEControlInterface.zeroFtSensor

    def zero_while_paused(control):
        assert engine._force_controller is None or not engine._force_controller.running
        return zero_sensor(control)

    monkeypatch.setattr(
        sdk["rtde_control"].RTDEControlInterface, "zeroFtSensor", zero_while_paused
    )
    for _ in range(2):
        before = len(state["forces"])
        engine.reset_episode()
        time.sleep(0.025)
        assert len(state["forces"]) == before
        engine._start_episode()
        engine.execute_actions(np.array([[0.01, 0, 0, 0, 0, 0]]))
        np.testing.assert_allclose(state["forces"][-1][2][:3], [12, 0, 0], atol=1e-8)
        controller = engine._force_controller
        target_sequence = controller._sequence
        with controller.io_lock:
            state["pose"][0] = 0.405
        deadline = time.monotonic() + 1
        while (
            not np.isclose(state["forces"][-1][2][0], 6) and time.monotonic() < deadline
        ):
            time.sleep(0.005)
        np.testing.assert_allclose(state["forces"][-1][2][:3], [6, 0, 0], atol=1e-8)
        assert controller._sequence == target_sequence
        before = len(state["forces"])
        time.sleep(0.035)
        assert len(state["forces"]) > before
        engine.stop_force()
        assert not controller.running
        before = len(state["forces"])
        time.sleep(0.025)
        assert len(state["forces"]) == before
    engine._start_episode()
    engine.close()
    before = len(state["forces"])
    time.sleep(0.025)
    assert len(state["forces"]) == before
    assert not controller.running
    import json

    records = [json.loads(row) for row in engine._timing_path.read_text().splitlines()]
    assert any(row["event"] == "controller" and row["samples"] > 1 for row in records)


@pytest.mark.parametrize("outcome", ["manual_success", "timeout", "keyboard_reset"])
def test_native_ur_continuous_force_stops_at_episode_end(
    native_ur_force_engine, outcome
):
    engine, _ = native_ur_force_engine
    engine._start_episode()
    engine.execute_actions(np.array([[0.01, 0, 0, 0, 0, 0]]))
    if outcome == "manual_success":
        engine._success_requested.set()
    elif outcome == "keyboard_reset":
        engine._reset_requested.set()
    else:
        engine._ep_start -= 61
    assert engine._finish_episode_if_needed()
    assert engine.recorder.episode_results[-1]["status"] == outcome
    before = len(engine.rtde_c.state["forces"])
    time.sleep(0.025)
    assert len(engine.rtde_c.state["forces"]) == before
    assert not engine._force_controller.running


def test_native_ur_continuous_force_preserves_zero_filter(native_ur_force_engine):
    engine, _ = native_ur_force_engine
    engine._start_episode()
    engine.filter_zero_output = True
    engine.action_idle_rotation_threshold = -1
    engine.execute_actions(np.array([[0.01, 0, 0, 0, 0, 0]]))
    engine.execute_actions(np.zeros((1, 6)))
    zero_start = len(engine.rtde_c.state["forces"])
    time.sleep(0.025)
    assert len(engine.rtde_c.state["forces"]) > zero_start
    for command in engine.rtde_c.state["forces"][zero_start:]:
        assert command[1] == [0] * 6
        np.testing.assert_array_equal(command[2], np.zeros(6))
    engine.execute_actions(np.array([[0.02, 0, 0, 0, 0, 0]]))
    assert engine.rtde_c.state["forces"][-1][1] == engine.select_vector


@pytest.mark.parametrize("operation", ["forceMode", "forceModeStop"])
def test_native_ur_continuous_force_failure_blocks_reset(
    native_ur_force_engine, monkeypatch, operation
):
    engine, sdk = native_ur_force_engine
    engine._start_episode()
    state = engine.rtde_c.state
    state["events"].clear()
    with monkeypatch.context() as patch:
        patch.setattr(
            sdk["rtde_control"].RTDEControlInterface, operation, lambda *args: False
        )
        if operation == "forceMode":
            with pytest.raises(RuntimeError, match="force controller failed"):
                engine.execute_actions(np.array([[0.01, 0, 0, 0, 0, 0]]))
        with pytest.raises(RuntimeError, match="force controller failed"):
            engine.reset_episode()
    assert "moveL" not in state["events"]
    assert "zeroFtSensor" not in state["events"]


def test_vla_precision_ur_port_preserves_robot_execution_methods():
    """Keep the copied observation/control loop identical to its recorded source."""
    import ast
    import hashlib
    import json

    root = Path(__file__).resolve().parents[2]
    port = root / "evaluations/realworld/vla_precision_ur"
    provenance = json.loads((port / "provenance.json").read_text())

    class WithoutAnnotations(ast.NodeTransformer):
        def visit_arg(self, node):
            node.annotation = None
            return node

        def visit_FunctionDef(self, node):
            self.generic_visit(node)
            node.returns = None
            node._fields = tuple(
                field for field in node._fields if field != "type_params"
            )
            return node

    tree = WithoutAnnotations().visit(ast.parse((port / "ur.py").read_text()))
    backend = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Inference"
    )
    methods = {
        node.name: node for node in backend.body if isinstance(node, ast.FunctionDef)
    }
    expected = provenance["unchanged_ur_methods"]
    assert {
        "get_obs_state",
        "execute_actions",
        "_calculate_force",
        "_target_pose_from_action",
        "reset_episode",
        "run",
    } <= (
        expected.keys()
        | provenance["timed_ur_methods"].keys()
        | provenance["tracking_ur_methods"].keys()
        | provenance["control_ur_methods"].keys()
    )
    for name, digest in expected.items():
        assert hashlib.sha256(ast.dump(methods[name]).encode()).hexdigest() == digest, (
            name
        )


@pytest.mark.parametrize("fail", [False, True])
def test_vla_precision_ur_warmup_progress_stops_with_inference(fail, monkeypatch):
    from evaluations.realworld.vla_precision_ur.policy import warmup_policy

    tick = threading.Event()

    class ProgressStream(io.StringIO):
        def write(self, value):
            result = super().write(value)
            if threading.current_thread().name == "ur-warmup-timer":
                tick.set()
            return result

    stream = ProgressStream()
    monkeypatch.setattr(sys, "stderr", stream)
    observation = {"state": np.zeros(19)}
    noise = np.zeros((20, 32))
    expected = {"actions": np.zeros((20, 7))}

    class Policy:
        def infer(self, obs, **kwargs):
            assert obs is observation
            assert kwargs["noise"] is noise
            assert tick.wait(5), "Warmup timer did not update during inference"
            if fail:
                raise RuntimeError("inference failed")
            return expected

    if fail:
        with pytest.raises(RuntimeError, match="inference failed"):
            warmup_policy(Policy(), observation, noise=noise)
    else:
        assert warmup_policy(Policy(), observation, noise=noise) is expected
    output = stream.getvalue()
    assert output.count("elapsed") >= 2
    assert output.count("\n") == 1
    assert "\r[WARMUP]" in output
    assert ("Stopped before completion" if fail else "Completed") in output
    assert not any(t.name == "ur-warmup-timer" for t in threading.enumerate())


@pytest.mark.parametrize("chunk_size", [3, 5])
def test_vla_precision_ur_model_config_keeps_native_hardware_settings(chunk_size):
    import copy

    import yaml

    from evaluations.realworld.vla_precision_ur.policy import model_config_from_yaml

    path = (
        Path(__file__).resolve().parents[2]
        / "evaluations/realworld/vla_precision_ur.yaml"
    )
    raw = yaml.safe_load(path.read_text())
    raw["inference_stage"] = "stage1"
    raw["model"]["network"]["openpi"]["action_chunk"] = chunk_size
    before = copy.deepcopy(raw)
    model = model_config_from_yaml(raw)
    assert raw == before
    assert model.openpi.action_horizon == 20
    assert "action_horizon" not in raw["robot"]
    assert model.num_action_chunks == model.openpi.action_chunk == chunk_size
    assert model.openpi.rtc_enabled is False
    assert model.openpi_data.norm_stats_path == raw["model"]["norm_stats_path"]
    raw["model"]["network"]["openpi"]["rtc_enabled"] = True
    with pytest.raises(ValueError, match="without RTC"):
        model_config_from_yaml(raw)


@pytest.mark.parametrize("chunk_size", [0, 21])
def test_vla_precision_ur_rejects_chunk_outside_prediction_horizon(chunk_size):
    import yaml

    from evaluations.realworld.vla_precision_ur.policy import model_config_from_yaml

    path = _ROOT / "evaluations/realworld/vla_precision_ur.yaml"
    raw = yaml.safe_load(path.read_text())
    raw["inference_stage"] = "stage1"
    raw["model"]["network"]["openpi"]["action_chunk"] = chunk_size
    with pytest.raises(ValueError, match="openpi.action_chunk.*action_horizon 20"):
        model_config_from_yaml(raw)


@pytest.mark.parametrize(
    "setting",
    [
        False,
        True,
        "default",
        "reduce-overhead",
        "max-autotune",
        "max-autotune-no-cudagraphs",
    ],
)
def test_vla_precision_ur_sampler_compile_modes(setting, monkeypatch):
    import copy

    import yaml

    from evaluations.realworld.vla_precision_ur.policy import (
        _configure_sampler_compile,
        _torch_compile_setting,
        model_config_from_yaml,
    )

    path = _ROOT / "evaluations/realworld/vla_precision_ur.yaml"
    raw = yaml.safe_load(path.read_text())
    raw["inference_stage"] = "stage1"
    raw["model"]["network"]["openpi"]["torch_compile"] = setting
    before = copy.deepcopy(raw)
    cfg = model_config_from_yaml(raw)
    assert raw == before
    assert cfg.openpi.torch_compile is (setting is not False)
    assert _torch_compile_setting(raw) == setting

    class Sampler(torch.nn.Module):
        def sample_actions(self, observation):
            return observation + 1

    model = Sampler()
    original = model.sample_actions
    compiled = Mock(side_effect=original)
    compile_fn = Mock(return_value=compiled)
    monkeypatch.setattr(torch, "compile", compile_fn)
    _configure_sampler_compile(model, setting)
    assert model.sample_actions(torch.tensor(2)).item() == 3
    if isinstance(setting, str):
        compile_fn.assert_called_once_with(original, mode=setting)
        assert model.sample_actions is compiled
    else:
        compile_fn.assert_not_called()
        assert model.sample_actions == original


@pytest.mark.parametrize("setting", [None, 0, 1, "false", "true", "max-autotun"])
def test_vla_precision_ur_sampler_compile_rejects_invalid_setting(setting):
    import yaml

    from evaluations.realworld.vla_precision_ur.policy import model_config_from_yaml

    raw = yaml.safe_load(
        (_ROOT / "evaluations/realworld/vla_precision_ur.yaml").read_text()
    )
    raw["inference_stage"] = "stage1"
    raw["model"]["network"]["openpi"]["torch_compile"] = setting
    with pytest.raises(ValueError, match="torch_compile must be a YAML boolean"):
        model_config_from_yaml(raw)


@pytest.mark.parametrize("action_timing_enabled", [False, True])
@pytest.mark.parametrize("save_logs", [False, True])
@pytest.mark.parametrize("chunk_size", [3, 5])
def test_vla_precision_ur_native_observation_and_chunk_execution(
    action_timing_enabled, save_logs, chunk_size, caplog, tmp_path
):
    import logging

    caplog.set_level(logging.INFO)
    for package in ("lerobot.cameras", "pyDHgripper", "av", "crcmod"):
        pytest.importorskip(package)
    from robot_mocks.ur5e import modules

    from evaluations.realworld.vla_precision_ur.ur import Inference

    sdk = modules()
    engine = object.__new__(Inference)
    engine.rtde_r = sdk["rtde_receive"].RTDEReceiveInterface("native-port-test")
    engine.rtde_c = sdk["rtde_control"].RTDEControlInterface("native-port-test")
    engine.rtde_r.state["speed"] = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6]
    engine.rtde_c.state["tcp_offset"] = [0.03, -0.01, 0.08, 0.1, -0.2, 0.3]
    engine.fix_gripper = False
    engine.reference_frame = "tcp"
    engine.task_description = "Insert the black rack into the blue slot"
    engine.gripper = SimpleNamespace(position=0.35)
    engine._episode_reference_ee_pose = None
    engine.set_episode_reference_pose()

    class Camera:
        def read(self):
            rgb = np.zeros((480, 640, 3), dtype=np.uint8)
            rgb[..., 0] = 255
            return rgb

    engine.cameras = {"wrist_image": Camera(), "exterior_image": Camera()}
    obs = engine.get_obs_state()
    np.testing.assert_allclose(obs["observation/state"][:6], 0, atol=1e-6)
    np.testing.assert_allclose(
        obs["observation/state"][6:12], engine.rtde_r.state["speed"]
    )
    np.testing.assert_allclose(obs["observation/state"][12:18], [1, 2, 3, 4, 5, 6])
    assert obs["observation/state"][18] == np.float32(0.35)
    image = obs["observation/image"]
    assert image.shape == (224, 224, 3)
    np.testing.assert_array_equal(image[112, 112], [255, 0, 0])
    assert not image[:28].any()

    engine.action_fps = 100
    engine.debug = False
    engine.action_timing_enabled = action_timing_enabled
    engine._infer_index = 7
    engine.save_logs = save_logs
    engine._timing_path = tmp_path / "timing.jsonl"
    engine.control_space = "force"
    engine.filter_zero_output = False
    engine._zero_force_lock_active = False
    engine.action_idle_hold_enabled = True
    engine.action_idle_position_threshold = 0.0001
    engine.action_idle_rotation_threshold = 0.001
    engine._clear_action_idle_hold()
    engine.kp, engine.kd = 1200, 600
    engine.kp_rot, engine.kd_rot = 20, 4
    engine.rtde_freq = 100
    engine.pos_delta, engine.vel_delta = 0.048, 4.0
    engine.use_control_frame = False
    engine.task_frame = [0.0] * 6
    engine.select_vector = [1, 1, 1, 0, 0, 1]
    engine.force_type = 2
    engine.force_limit = [2] * 6
    from evaluations.realworld.vla_precision_ur.utils import FpsCounter

    engine.fps_action = FpsCounter("test")
    actions = np.zeros((chunk_size, 7))
    actions[:3, 6] = [0.1, 0.8, 0.9]
    actions[3:, 6] = 0.9
    engine.execute_actions(actions)
    assert len(engine.rtde_c.state["forces"]) == chunk_size
    assert engine._gripper_position == 0.9
    force = engine.rtde_c.state["forces"][0][2]
    np.testing.assert_allclose(force, [-60, 120, -180, 1.6, -2, 2.4], atol=1e-8)
    timing_logs = [
        record.message
        for record in caplog.records
        if record.message.startswith("[ACTION TIMING]")
    ]
    assert not timing_logs
    caplog.clear()
    engine.filter_zero_output = True
    engine.show_target_error = False
    engine.action_idle_rotation_threshold = -1
    engine.execute_actions(np.zeros((2, 7)))
    # The first zero action disables force; the second keeps the existing lock.
    assert len(engine.rtde_c.state["forces"]) == chunk_size + 1
    assert not any(
        r.message.startswith(("[ACTION TIMING]", "[DEBUG]", "[PD]"))
        for r in caplog.records
    )

    engine._record_timing("step", infer_sec=0.01, loop_sec=0.1)
    import json

    assert engine._timing_path.exists() == save_logs
    if not save_logs:
        return
    records = [
        json.loads(line) for line in engine._timing_path.read_text().splitlines()
    ]
    assert [row["event"] for row in records] == (
        ["action"] * (chunk_size + 2) + ["step"] if action_timing_enabled else ["step"]
    )
    if action_timing_enabled:
        assert records[1]["target_interval_sec"] >= 0.01
        assert records[2]["target_interval_sec"] >= 0.01
        for row in records[:chunk_size]:
            assert row["target_hz"] == 100
            assert row["control_sec"] > 0
            assert row["pace_wait_sec"] >= 0
            assert all(
                key in row
                for key in (
                    "state_read_sec",
                    "target_sec",
                    "impedance_sec",
                    "send_sec",
                    "gripper_target_sec",
                    "other_sec",
                    "control_overrun_sec",
                )
            )
        for row in records[chunk_size : chunk_size + 2]:
            assert row["state_read_sec"] == 0
            assert row["target_sec"] == 0
            assert row["impedance_sec"] == 0
        assert records[0]["impedance_sec"] >= 0
        assert records[0]["infer_index"] == 7
        assert [row["action_index"] for row in records[:3]] == [1, 2, 3]


@pytest.mark.parametrize("save_logs", [False, True])
def test_vla_precision_ur_recorder_save_logs_keeps_console_summary(
    save_logs, tmp_path, capsys
):
    import json

    pytest.importorskip("av")
    from evaluations.realworld.vla_precision_ur.recording import Recorder

    path = tmp_path / "actions.yaml"
    evaluation_path = tmp_path / "runs" / "test-ur5e-eval" / "ur5e-evaluation.json"
    recorder = Recorder(
        path,
        [tmp_path / "wrist.mp4", tmp_path / "exterior.mp4"],
        save_logs=save_logs,
        evaluation_path=evaluation_path,
    )
    if save_logs:
        initial = json.loads(evaluation_path.read_text())
        assert initial["episodes"] == []
        assert initial["summary"]["success_rate"] == 0
    recorder.submit_actions(np.zeros((3, 7)), 1, state=np.zeros(19))
    recorder.queue_log.join()
    recorder.submit_episode_result(1, 1.0, "manual_success", completed=True)
    if save_logs:
        first = json.loads(evaluation_path.read_text())
        assert len(first["episodes"]) == 1
        assert first["summary"]["success_rate"] == 1
        assert path.read_text().count("episode_result:") == 1
    recorder.submit_episode_result(2, 2.0, "keyboard_reset", completed=False)
    recorder.submit_episode_result(3, 3.0, "timeout", completed=False)
    if save_logs:
        latest = json.loads(evaluation_path.read_text())
        assert [item["status"] for item in latest["episodes"]] == [
            "manual_success",
            "keyboard_reset",
            "timeout",
        ]
        assert latest["summary"]["total_episodes"] == 3
        assert latest["summary"]["completed_episodes"] == 1
        assert latest["summary"]["incomplete_episodes"] == 2
        assert latest["summary"]["success_rate"] == pytest.approx(0.3333)
        assert path.read_text().count("episode_result:") == 3
    summary = recorder.submit_episode_summary()
    assert summary["completed_episodes"] == 1
    assert "Episode Timing Summary" in capsys.readouterr().out
    assert path.exists() == save_logs
    if save_logs:
        content = path.read_text()
        assert "state:" in content and "actions:" in content
        assert "episode_summary:" in content
        assert content.count("episode_result:") == 3
        final = json.loads(evaluation_path.read_text())
        assert final["episodes"] == latest["episodes"]
        assert final["summary"]["total_episodes"] == 3
        plot_path = path.with_name("actions_episode_time.png")
        assert Path(summary["episode_plot"]) == plot_path
        assert plot_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    else:
        assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("explicit_output", [False, True])
def test_vla_precision_episode_plot_cli_follows_resolved_log(
    tmp_path, monkeypatch, explicit_output
):
    from evaluations.realworld.vla_precision_ur import plots

    log_path = tmp_path / "results" / "logs" / "all_logs" / "log_test.yaml"
    log_path.parent.mkdir(parents=True)
    log_path.write_text(
        "episode_summary:\n"
        "  episodes:\n"
        "    - episode_index: 1\n"
        "      duration_sec: 1.5\n"
        "      completed: true\n"
    )
    latest = log_path.parent.parent / "latest.yaml"
    latest.symlink_to(log_path)
    args = ["plots", str(latest)]
    expected = log_path.with_name("log_test_episode_time.png")
    if explicit_output:
        expected = tmp_path / "custom" / "timing.png"
        args.extend(["--output", str(expected)])
    monkeypatch.setattr(sys, "argv", args)
    plots.main()
    assert expected.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


@pytest.mark.parametrize("reference_frame", ["tcp", "base"])
def test_ur5e_control_matches_native_vla_precision_with_tool_offset(reference_frame):
    for package in ("lerobot.cameras", "pyDHgripper", "av", "crcmod"):
        pytest.importorskip(package)
    from robot_mocks import mocked_sdks
    from scipy.spatial.transform import Rotation as R

    from evaluations.realworld.vla_precision_ur.ur import Inference
    from rlinf.envs.real.ur5e.motion import URTargetComposer
    from rlinf.robotics.parts.arms.ur5e import UR5eArm

    with mocked_sdks() as sdks:
        arm = UR5eArm(
            "192.0.2.31",
            impedance={
                "continuous_tracking": False,
                "control_frame_euler_deg": [10, 20, 30],
            },
        )
        arm.connect()
        native = object.__new__(Inference)
        native.rtde_r = sdks["rtde_receive"].RTDEReceiveInterface("192.0.2.31")
        native.rtde_c = sdks["rtde_control"].RTDEControlInterface("192.0.2.31")
        native.reference_frame = reference_frame
        native.action_idle_hold_enabled = True
        native.action_idle_position_threshold = 0.0001
        native.action_idle_rotation_threshold = 0.001
        native._clear_action_idle_hold()
        native.kp, native.kd = 1200, 600
        native.kp_rot, native.kd_rot = 20, 4
        native.rtde_freq = 100
        native.pos_delta, native.vel_delta = 0.048, 4.0
        native.use_control_frame = True
        native.control_to_base_rotation = R.from_euler(
            "xyz", [10, 20, 30], degrees=True
        ).as_matrix()
        composer = URTargetComposer()
        rng = np.random.default_rng(12)
        try:
            for index in range(40):
                state = native.rtde_r.state
                state["pose"] = np.r_[[0.4, 0.1, 0.3], rng.normal(0, 0.2, 3)].tolist()
                state["speed"] = rng.normal(0, 0.2, 6).tolist()
                state["tcp_offset"] = [0.02, -0.01, 0.08, 0.1, -0.2, 0.3]
                action = rng.normal(0, 0.01, 7)
                if index % 3 == 0:
                    action[:6] = 0
                measured = arm.get_state()
                target = composer.target(
                    measured.tcp_pose,
                    arm.get_tcp_offset(),
                    action,
                    reference_frame=reference_frame,
                    hold_enabled=True,
                    position_threshold=0.0001,
                    rotation_threshold=0.001,
                )
                expected = native._target_pose_from_action(action)
                np.testing.assert_allclose(target[:3], expected[:3], atol=1e-12)
                np.testing.assert_allclose(
                    R.from_quat(target[3:]).as_matrix(),
                    R.from_rotvec(expected[3:]).as_matrix(),
                    atol=1e-12,
                )
                expected_wrench = native._calculate_force(
                    expected, np.asarray(state["pose"]), np.asarray(state["speed"])
                )
                # The robot can move between target composition and command send.
                state["pose"] = np.r_[[0.5, 0.2, 0.4], rng.normal(0, 0.3, 3)].tolist()
                state["speed"] = rng.normal(0, 0.4, 6).tolist()
                arm.send_action_from_state({"tcp_pose": target}, measured)
                np.testing.assert_allclose(
                    state["forces"][-1][2],
                    expected_wrench,
                    atol=1e-9,
                )
        finally:
            arm.disconnect()


def test_rlt_queued_success_stops_before_action_and_records_hold(rlt_keyboard_control):
    env, reader = rlt_keyboard_control(start_with_actor=True)
    env.unwrapped._gripper_target = 1.0
    before = env.unwrapped.num_steps
    _press_keys(reader, "space", "space")
    _, reward, terminated, truncated, info = env.step(np.ones(7, dtype=np.float32))
    assert (reward, terminated, truncated) == (1, True, False)
    assert env.unwrapped.num_steps == before
    np.testing.assert_array_equal(info["intervene_action"], [0, 0, 0, 0, 0, 0, 1])
    assert env.step(np.ones(7))[1:4] == (0, True, False)
    assert env.unwrapped.num_steps == before


@pytest.mark.parametrize("operator_steps", [0, 2])
def test_rlt_success_chunk_reports_hold_actions_and_single_reward(
    keyboard_reset_env, operator_steps
):
    from rlinf.data.schema.embodied_types import EnvTransition

    env, reader = keyboard_reset_env(rlt_mode=True)
    _, reset_info = env.reset()
    assert (
        EnvTransition(
            rlt_switch_flags=reset_info["rlt_switch_flags"]
        ).rlt_switch_flags.dtype
        == torch.bool
    )
    raw = env.env.envs[0].unwrapped
    raw._gripper_target = 1.0
    env.intervened_steps[:] = operator_steps
    env.intervened_once[:] = operator_steps > 0
    _press_keys(reader, "space", "space")
    _, rewards, terminations, truncations, infos = env.chunk_step(
        np.ones((1, 10, 7), dtype=np.float32)
    )
    assert raw.num_steps == 0
    assert rewards.tolist() == [[1] + [0] * 9]
    assert terminations.tolist() == [[False] * 9 + [True]]
    assert not truncations.any()
    final = infos[-1]["final_info"]
    assert final["episode"]["success_no_intervened"].item() == (operator_steps == 0)
    assert final["episode"]["intervened_once"].item() == (operator_steps > 0)
    assert final["episode"]["intervened_steps"].item() == operator_steps
    assert final["episode"]["terminal_hold_steps"].item() == 10
    assert final["episode"]["recorded_override_steps"].item() == operator_steps + 10
    reset_transition = EnvTransition(rlt_switch_flags=infos[-1]["rlt_switch_flags"])
    assert reset_transition.rlt_switch_flags.dtype == torch.bool
    assert final["intervene_flag"].all()
    expected = torch.tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float32)
    assert torch.equal(
        final["intervene_action"].reshape(1, 10, 7), expected.expand(1, 10, 7)
    )
    assert env.terminal_hold_steps.item() == 0


def test_ur5e_evaluation_can_close_while_waiting_for_operator(keyboard_reset_env):
    from rlinf.workers.env.ur5e_eval_worker import URSyncEnvWorker

    env, device = keyboard_reset_env(
        ur_eval=True, wait_for_start=True, reset=False, save_logs=False
    )
    worker = object.__new__(URSyncEnvWorker)
    worker.cfg = OmegaConf.create(
        {
            "env": {"eval": OmegaConf.to_container(env.cfg)},
            "rollout": {"group_name": "Rollout"},
        }
    )
    worker.stage_num = worker.eval_num_envs_per_stage = worker.eval_batch_size = 1
    worker.env_decoupled_mode = False
    worker.env_list, worker.eval_env_list = [], [env]
    worker.eval_prev_done = [None]
    worker.enable_rlt = False
    worker.log_info = lambda *_: None
    worker.send_to = lambda **_: None
    worker.recv_from = lambda **_: torch.zeros(1, 3, 7)

    async def exercise():
        task = asyncio.create_task(worker.evaluate(None, None))
        try:
            for _ in range(100):
                if env.env.envs[0].get_wrapper_attr("waiting_for_start"):
                    break
                await asyncio.sleep(0.01)
            assert env.env.envs[0].get_wrapper_attr("waiting_for_start")
        finally:
            worker.close()
        with pytest.raises(RuntimeError, match="closed while waiting"):
            await asyncio.wait_for(task, 2)

    asyncio.run(exercise())
    assert device.closed and env._closed


@pytest.mark.parametrize("cancel", [False, True])
def test_async_env_stop_awaits_interaction_and_closes_keyboard(
    keyboard_reset_env, cancel
):
    import asyncio

    from rlinf.workers.env.async_env_worker import AsyncEnvWorker

    env, device = keyboard_reset_env()
    worker = object.__new__(AsyncEnvWorker)
    worker.env_list, worker.eval_env_list = [env], []
    worker._interact_task = None

    async def exercise():
        if cancel:
            worker._interact_task = asyncio.create_task(asyncio.Event().wait())
            await asyncio.sleep(0)
        await worker.stop()
        await worker.stop()
        assert not cancel or worker._interact_task.cancelled()

    asyncio.run(exercise())
    assert device.closed and env._closed
    with pytest.raises(RuntimeError, match="closed"):
        env.step(np.ones((1, 7)))


def test_async_env_stop_releases_arm_even_when_camera_close_fails():
    import asyncio

    from robot_mocks import mocked_sdks

    from rlinf.envs.real.ur5e import UR5eEnv
    from rlinf.robotics import RobotInfo, UR5eConfig, UR5eRobot
    from rlinf.robotics.parts.arms.ur5e import UR5eArm
    from rlinf.robotics.parts.end_effectors.grippers.pgi import PGIGripper
    from rlinf.workers.env.async_env_worker import AsyncEnvWorker

    with mocked_sdks() as sdks:
        arm = UR5eArm("192.0.2.55", impedance={"continuous_tracking": False})
        gripper = PGIGripper("/tmp/ur-cleanup-test-gripper")
        robot = UR5eRobot(arm=arm, end_effector=gripper)
        robot.connect()
        control, receive = arm._device
        env = UR5eEnv(
            {"is_dummy": True, "enable_camera_player": False},
            robot_info=RobotInfo(
                type="UR5e",
                model="UR5e",
                config=UR5eConfig(node_rank=0, camera_serials=["dummy"]),
            ),
        )
        env.robot, env._arm = robot, arm

        class Player:
            def stop(self):
                assert not arm.is_connected
                raise ValueError("camera player failed")

        env.camera_player = Player()
        worker = object.__new__(AsyncEnvWorker)
        worker.env_list, worker.eval_env_list = [env], []
        worker._interact_task = None
        with pytest.raises(ValueError, match="camera player failed"):
            asyncio.run(worker.stop())
        assert not arm.is_connected and not gripper.is_connected
        assert not control.connected and not receive.connected
        assert "forceModeStop" in control.state["events"]
        assert "stopScript" in control.state["events"]
        assert sdks["pyDHgripper"].PGE.instances[-1].closed
        asyncio.run(worker.stop())


@pytest.mark.parametrize("failure", ["exception", "interrupt", "terminate"])
def test_vla_precision_exit_always_releases_connections(failure):
    for package in ("lerobot.cameras", "pyDHgripper", "av", "crcmod"):
        pytest.importorskip(package)
    from robot_mocks.ur5e import modules

    from evaluations.realworld.vla_precision_ur.ur import Inference

    sdk = modules()
    engine = object.__new__(Inference)
    control = sdk["rtde_control"].RTDEControlInterface("exit-test")
    receive = sdk["rtde_receive"].RTDEReceiveInterface("exit-test")
    gripper = sdk["pyDHgripper"].PGE("/tmp/exit-test-gripper")
    engine.rtde_c, engine.rtde_r, engine.gripper = control, receive, gripper
    closed_cameras = []

    class Camera:
        def disconnect(self):
            closed_cameras.append(True)
            raise ValueError("camera close failed")

    engine.cameras = {"wrist": Camera()}
    engine._ep_steps = 0
    engine.action_chunk = 10
    engine.task_description = "test"
    engine.recorder = SimpleNamespace(submit_episode_summary=lambda **kwargs: None)
    engine._ask_save_video = lambda: None

    def prepare():
        raise {
            "exception": ValueError,
            "interrupt": KeyboardInterrupt,
            "terminate": SystemExit,
        }[failure]("inference failed")

    engine._prepare_inference = prepare
    if failure == "terminate":
        with pytest.raises(SystemExit):
            engine.run()
    else:
        engine.run()
    assert control.state["events"] == ["forceModeStop", "servoStop", "stopScript"]
    assert not control.connected and not receive.connected
    assert gripper.closed and closed_cameras == [True]
    engine.close()
    assert len(control.state["events"]) == 3


def test_rlt_chunk_progress_counts_execution_before_auto_reset(
    keyboard_reset_env, caplog
):
    from omegaconf import open_dict

    env, reader = keyboard_reset_env(rlt_mode=True)
    with open_dict(env.cfg):
        env.cfg.max_episode_chunks = 200
        env.cfg.episode_action_limit = 600
    env.env.envs[0].unwrapped.config.max_num_steps = 600
    actions = np.zeros((1, 3, 7), dtype=np.float32)
    with caplog.at_level("INFO"):
        env.reset()
        env.chunk_step(actions)
        env.chunk_step(actions)
        _press_keys(reader, "space", "space")
        env.chunk_step(actions)
    events = [
        record.getMessage()
        for record in caplog.records
        if "[EXEC]" in record.getMessage()
    ]
    assert events == [
        f"[EXEC] Episode chunks: {count}/200" for count in (0, 1, 2, 3, 0)
    ]


@pytest.mark.parametrize("episode_replay", [False, True])
@pytest.mark.parametrize("outcome", ["success", "timeout"])
def test_rlt_episode_replay_keeps_terminal_state_until_explicit_reset(
    keyboard_reset_env, episode_replay, outcome
):
    env, reader = keyboard_reset_env(
        rlt_mode=True,
        episode_replay=episode_replay,
        episode_chunks=1 if outcome == "timeout" else 150,
    )
    if outcome == "success":
        _press_keys(reader, "space", "space")
    obs, rewards, terminations, truncations, infos = env.chunk_step(
        np.zeros((1, 3, 7), dtype=np.float32)
    )
    assert bool(terminations.any()) == (outcome == "success")
    assert bool(truncations.any()) == (outcome == "timeout")
    assert float(rewards.sum()) == (1.0 if outcome == "success" else 0.0)
    assert "final_info" in infos[-1] and "final_observation" in infos[-1]
    assert env._episode_chunks == (1 if episode_replay else 0)
    if episode_replay:
        assert torch.equal(obs[-1]["states"], infos[-1]["final_observation"]["states"])
        env.reset()
        assert env._episode_chunks == 0


@pytest.mark.parametrize("receipt", ["correct", "wrong", "cancel", "stop"])
def test_rlt_episode_replay_waits_for_commit_before_reset(keyboard_reset_env, receipt):
    import asyncio

    from rlinf.data.schema.embodied_types import EnvOutput, EnvTransition, TrajectoryKey
    from rlinf.workers.env.env_worker import EnvWorker

    env, reader = keyboard_reset_env(
        rlt_mode=True, episode_replay=True, episode_chunks=150
    )
    from rlinf.envs.real.wrappers.episode import KeyboardRLTPolicySwitchWrapper

    wrapper = env.env.envs[0]
    while not isinstance(wrapper, KeyboardRLTPolicySwitchWrapper):
        wrapper = wrapper.env
    wrapper.start_with_actor = False
    _press_keys(reader, "space", "space")
    observations, _, terminations, truncations, _ = env.chunk_step(
        np.zeros((1, 3, 7), dtype=np.float32)
    )
    output = EnvOutput(
        obs=observations[-1],
        transition=EnvTransition(
            dones=terminations | truncations,
            terminations=terminations,
            truncations=truncations,
        ),
    )
    worker = object.__new__(EnvWorker)
    worker.env_list = [env]
    worker.should_stop = False
    worker.log_info = lambda *_: None
    end_key = TrajectoryKey(1, 0, 0, 0, 0)

    async def exercise():
        committed = asyncio.get_running_loop().create_future()

        class ReceiptChannel:
            def get(self, *, async_op):
                assert async_op
                return self

            async def async_wait(self):
                return await committed

        worker._replay_ack_channel = ReceiptChannel()
        pending = asyncio.create_task(
            worker._reset_after_episode_replay(end_key, output)
        )
        await asyncio.sleep(0)
        assert not pending.done()
        assert env._episode_chunks == 1
        if receipt == "cancel":
            pending.cancel()
        else:
            worker.should_stop = receipt == "stop"
            committed.set_result(end_key if receipt != "wrong" else None)
        if receipt in {"cancel", "stop"}:
            with pytest.raises(asyncio.CancelledError):
                await pending
        elif receipt == "wrong":
            with pytest.raises(RuntimeError, match="Expected replay receipt"):
                await pending
        else:
            reset_output = await pending
            assert not reset_output.dones.any()
            assert not reset_output.rlt_switch_flags.any()
        assert env._episode_chunks == (0 if receipt == "correct" else 1)

    asyncio.run(exercise())


@pytest.mark.parametrize("profile", ["online", "rlinf_eval", "vla_precision"])
@pytest.mark.parametrize("frequency", [50.0, 100.0, 200.0])
def test_ur5e_profiles_use_direct_rotational_gains(profile, frequency):
    from robot_mocks import mocked_sdks
    from scipy.spatial.transform import Rotation as R

    from rlinf.robotics.parts.arms.ur5e import UR5eArm

    root = Path(__file__).resolve().parents[2]
    paths = {
        "online": "examples/embodiment/config/ur_rlt_stage2_ac_mlp.yaml",
        "rlinf_eval": "evaluations/realworld/realworld_ur5e_eval_pi05_sft.yaml",
        "vla_precision": "evaluations/realworld/vla_precision_ur.yaml",
    }
    cfg = OmegaConf.load(root / paths[profile])
    expected = [-48, 96, -144, 1.6, -4.8, 4.8]
    current_pose = np.array([0.4, 0, 0.3, 0, 0, 0])
    speed = np.array([0.1, -0.2, 0.3, 0.1, 0.2, 0.3])
    target_pose = np.array([0.41, -0.02, 0.33, 0.1, -0.2, 0.3])
    if profile == "vla_precision":
        for package in ("lerobot.cameras", "pyDHgripper", "av", "crcmod"):
            pytest.importorskip(package)
        from evaluations.realworld.vla_precision_ur.ur import Inference

        settings = cfg.robot.force_mode
        engine = object.__new__(Inference)
        engine.kp, engine.kd = settings.kp, settings.kd
        engine.kp_rot, engine.kd_rot = settings.kp_rot, settings.kd_rot
        engine.pos_delta, engine.vel_delta = settings.pos_delta, settings.vel_delta
        engine.rtde_freq = frequency
        engine.use_control_frame = False
        np.testing.assert_allclose(
            engine._calculate_force(target_pose, current_pose, speed),
            expected,
            atol=1e-10,
        )
        return

    hardware = cfg.cluster.node_groups[0].hardware.configs
    settings = OmegaConf.to_container(hardware[0].impedance, resolve=True)
    settings["frequency"] = frequency
    with mocked_sdks() as sdks:
        arm = UR5eArm("192.0.2.42", impedance=settings)
        arm.connect()
        state = sdks["rtde_control"].states["192.0.2.42"]
        state["speed"] = speed.tolist()
        target = np.r_[target_pose[:3], R.from_rotvec(target_pose[3:]).as_quat()]
        try:
            arm.start_motion()
            arm.send_action({"tcp_pose": target})
            np.testing.assert_allclose(state["forces"][-1][2], expected, atol=1e-10)
        finally:
            arm.disconnect()


@pytest.mark.parametrize("frequency", [20.0, 50.0])
@pytest.mark.parametrize("detailed", [False, True])
def test_ur5e_chunk_target_cadence_includes_outer_processing(
    keyboard_reset_env, monkeypatch, frequency, detailed
):
    from rlinf.envs.real.ur5e import base

    env, _ = keyboard_reset_env(rlt_mode=True)
    raw = env.env.envs[0].unwrapped
    raw.config.step_frequency = frequency
    raw._action_trace_enabled = detailed
    clock = SimpleNamespace(value=1.0)

    def sleep(duration):
        clock.value += max(0.0, duration)

    def monotonic():
        clock.value += 1e-9
        return clock.value

    monkeypatch.setattr(
        base,
        "time",
        SimpleNamespace(monotonic=monotonic, sleep=sleep, time=time.time),
    )
    period = 1.0 / frequency
    raw.begin_action_chunk(4)
    sends = []
    for delay in [0.0, period * 0.3, period * 1.6, period * 0.2]:
        sleep(delay)
        _, _, _, _, info = raw.step(np.zeros(7))
        sends.append(raw._last_target_sent)
        if detailed:
            interval = info["ur_action_trace"]["target_interval_sec"]
            assert info["ur_action_trace"]["control_overrun_sec"] == pytest.approx(
                max(0, (interval or 0) - period)
            )
        else:
            assert raw._target_interval is None
            assert raw._target_pace_wait == 0
    np.testing.assert_allclose(
        np.diff(sends), [period, period * 1.6, period], atol=1e-6
    )
    # The final action retains one period before its observation is returned.
    assert clock.value == pytest.approx(sends[-1] + period)
    raw.begin_action_chunk(1)
    raw.step(np.zeros(7))
    assert raw._last_target_sent == pytest.approx(sends[-1] + period)


@pytest.mark.parametrize("frequency", [20.0, 50.0])
@pytest.mark.parametrize("continuous", [False, True])
def test_native_ur_target_cadence_includes_processing(
    native_ur_force_engine, monkeypatch, frequency, continuous
):
    import json

    engine, _ = native_ur_force_engine
    engine.action_fps = frequency
    engine.continuous_tracking = continuous
    engine._start_episode()
    # Simulate variable SDK state-read latency before target submission.
    original = engine.rtde_r.getActualTCPPose
    reads = []

    def read_pose():
        time.sleep(0.003 if len(reads) % 2 else 0.001)
        reads.append(1)
        return original()

    monkeypatch.setattr(engine.rtde_r, "getActualTCPPose", read_pose)
    engine.execute_actions(np.tile([0.001, 0, 0, 0, 0, 0], (3, 1)))
    engine.stop_force()
    records = [
        json.loads(line) for line in engine._timing_path.read_text().splitlines()
    ]
    intervals = [
        r["target_interval_sec"]
        for r in records
        if r.get("target_interval_sec") is not None
    ]
    assert len(intervals) == 2
    assert min(intervals) >= 1.0 / frequency
    assert max(intervals) < 1.0 / frequency + 0.015


def test_ur_stage2_double_dot_and_reset(keyboard_reset_env):
    env, device = keyboard_reset_env(ur_eval=True, stage2_enabled=True)
    wrapper = env.env.envs[0]
    action = np.zeros((1, 7), dtype=np.float32)
    assert not wrapper.get_wrapper_attr("critical_phase")
    _press_keys(device, "dot")
    env.step(action)
    assert not wrapper.get_wrapper_attr("critical_phase")
    _press_keys(device, "dot")
    env.step(action)
    assert wrapper.get_wrapper_attr("critical_phase")
    env.reset()
    assert not wrapper.get_wrapper_attr("critical_phase")


def test_ur_stage2_switch_timeout_and_reset():
    from rlinf.algorithms.rlt.inference import CriticalPhaseSwitch

    switch = CriticalPhaseSwitch()
    assert not switch.press(1.0)
    assert not switch.press(1.6)
    assert switch.press(1.8)
    assert switch.active
    assert not switch.press(1.9)
    switch.reset()
    assert not switch.active
    assert not switch.press(2.0)


@pytest.fixture
def stage2_eval_options(tmp_path):
    from omegaconf import OmegaConf

    training = OmegaConf.load(
        _ROOT / "examples/embodiment/config/ur_rlt_stage2_ac_mlp.yaml"
    )
    frozen = tmp_path / "stage1" / "model_state_dict"
    frozen.mkdir(parents=True)
    torch.save({}, frozen / "full_weights.pt")
    stats = tmp_path / "norm_stats.json"
    stats.write_text("{}")
    training.rollout.rlt_feature_model.model_path = str(frozen.parent)
    training.rollout.rlt_feature_model.openpi_data.norm_stats_path = str(stats)
    training_path = tmp_path / "training.yaml"
    OmegaConf.save(training, training_path)
    weights = tmp_path / "head.pt"
    torch.save({}, weights)
    return {
        "training_config": str(training_path),
        "checkpoint_path": str(weights),
        "num_steps": 4,
        "compile_mode": None,
        "mlp_compile_mode": None,
        "gripper_close_threshold": 0.5,
    }


def test_ur_stage2_profiles_preserve_stage1_and_select_training_pair(
    stage2_eval_options, monkeypatch
):
    import copy

    import yaml
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from evaluations.realworld.eval_ur5e import prepare_inference_config
    from evaluations.realworld.vla_precision_ur.policy import model_config_from_yaml
    from rlinf.config import validate_only_eval_rollout_model

    monkeypatch.setenv("EMBODIED_PATH", str(_ROOT / "examples/embodiment"))
    with initialize_config_dir(
        config_dir=str(_ROOT / "evaluations/realworld"), version_base="1.3"
    ):
        native = compose(config_name="realworld_ur5e_eval_pi05_sft")
    standalone = yaml.safe_load(
        (_ROOT / "evaluations/realworld/vla_precision_ur.yaml").read_text()
    )
    native.inference_stage = standalone["inference_stage"] = "stage1"
    assert prepare_inference_config(native) == native
    assert model_config_from_yaml(standalone).num_steps == 4
    native.inference_stage = standalone["inference_stage"] = "stage2"
    native.stage2 = OmegaConf.create(stage2_eval_options)
    standalone["stage2"] = stage2_eval_options
    before_native, before_standalone = copy.deepcopy(native), copy.deepcopy(standalone)
    selected = prepare_inference_config(native)
    validate_only_eval_rollout_model(selected.rollout.model)
    other = model_config_from_yaml(standalone)
    assert native == before_native and standalone == before_standalone
    assert selected.rollout.model == other
    assert other.num_steps == other.openpi.num_steps == 4
    assert other.num_action_chunks == 10 and other.openpi.action_horizon == 20
    assert selected.env.eval.ur_evaluation.stage2_enabled
    assert selected.env.eval.override_cfg.gripper_close_threshold == 0.5
    assert other.openpi_data.norm_stats_path.endswith("norm_stats.json")
    native.runner.rtc.enabled = True
    with pytest.raises(ValueError, match="synchronous"):
        prepare_inference_config(native)


def test_ur_stage2_rejects_missing_head(stage2_eval_options):
    from rlinf.algorithms.rlt.inference import stage2_configs

    Path(stage2_eval_options["checkpoint_path"]).unlink()
    with pytest.raises(FileNotFoundError, match="Stage2"):
        stage2_configs(stage2_eval_options)


@pytest.mark.parametrize("keys", [b"..", b"  ", b"\n\n"])
def test_vla_precision_ur_batched_terminal_keys(
    native_ur_force_engine, monkeypatch, keys
):
    import os
    import pty
    import termios

    from evaluations.realworld.vla_precision_ur import ur
    from rlinf.algorithms.rlt.inference import CriticalPhaseSwitch

    engine, _ = native_ur_force_engine
    engine.stage2_enabled = True
    engine._critical_switch = CriticalPhaseSwitch()
    engine._keyboard_listener_stop = threading.Event()
    master, slave = pty.openpty()
    original_settings = termios.tcgetattr(slave)
    try:
        with os.fdopen(os.dup(slave), "r") as stream:
            monkeypatch.setattr(ur.sys, "stdin", stream)
            thread = threading.Thread(target=engine._keyboard_episode_control_listener)
            engine._keyboard_listener_thread = thread
            thread.start()
            try:
                deadline = time.monotonic() + 2
                while termios.tcgetattr(slave)[3] & termios.ICANON:
                    assert time.monotonic() < deadline, (
                        "Listener did not enter cbreak mode"
                    )
                    time.sleep(0.01)
                os.write(master, keys)
                while time.monotonic() < deadline:
                    activated = (
                        engine._critical_switch.active
                        if keys == b".."
                        else engine._success_requested.is_set()
                        if keys == b"  "
                        else engine._reset_requested.is_set()
                    )
                    if activated:
                        break
                    time.sleep(0.01)
                assert activated, f"Listener did not consume both bytes: {keys!r}"
            finally:
                engine._keyboard_listener_stop.set()
                thread.join(timeout=2)
            assert not thread.is_alive()
            assert termios.tcgetattr(slave) == original_settings
    finally:
        os.close(master)
        os.close(slave)


@pytest.fixture
def terminal_keyboard_display(monkeypatch):
    from robot_mocks.teleop import x11_keyboard_display

    sdk, state = x11_keyboard_display()
    monkeypatch.setitem(sys.modules, "Xlib.display", sdk)
    monkeypatch.setenv("XDG_SESSION_TYPE", "x11")
    monkeypatch.delenv("WINDOWID", raising=False)
    monkeypatch.delenv("RLINF_KEYBOARD_TERMINAL_WINDOW", raising=False)
    return state


def test_terminal_keyboard_configuration_and_display_failure(terminal_keyboard_display):
    from rlinf.utils.keyboard_focus import (
        TerminalKeyboardFocus,
        configure_terminal_keyboard,
    )

    state = terminal_keyboard_display
    configure_terminal_keyboard(True)
    assert os.environ["RLINF_KEYBOARD_TERMINAL_WINDOW"] == "42"
    gate = TerminalKeyboardFocus()
    try:
        assert gate.accepts_input()
        state.active = 88
        assert not gate.accepts_input()
        state.active = 42
        assert gate.accepts_input()
        state.failed = True
        assert not gate.accepts_input()
        state.failed = False
        assert not gate.accepts_input()  # A disconnected display stays disabled.
    finally:
        gate.close()
    configure_terminal_keyboard(False)
    gate = TerminalKeyboardFocus()
    assert gate.accepts_input()
    gate.close()
    assert all(connection.closed for connection in state.connections)


def test_terminal_keyboard_rejects_unsupported_mode(monkeypatch):
    from rlinf.utils.keyboard_focus import configure_terminal_keyboard

    with pytest.raises(ValueError, match="YAML boolean"):
        configure_terminal_keyboard("false")
    monkeypatch.setenv("XDG_SESSION_TYPE", "wayland")
    with pytest.raises(RuntimeError, match="X11"):
        configure_terminal_keyboard(True)


def test_terminal_keyboard_discards_background_keys_and_held_state(
    monkeypatch, terminal_keyboard_display
):
    from robot_mocks.teleop import keyboard

    from rlinf.envs.real.wrappers.episode.keyboard import KeyboardListener
    from rlinf.utils.keyboard_focus import FOCUS_LOST, configure_terminal_keyboard

    state = terminal_keyboard_display
    configure_terminal_keyboard(True)
    sdk = keyboard()
    monkeypatch.setitem(sys.modules, "evdev", sdk)
    monkeypatch.setenv("RLINF_KEYBOARD_DEVICE", "/dev/input/event-fake")
    monkeypatch.delenv("RLINF_FAKE_KEYS", raising=False)
    listener = KeyboardListener()
    reader = sdk.devices[-1]

    def wait_until(predicate):
        deadline = time.monotonic() + 2
        while not predicate():
            assert time.monotonic() < deadline
            time.sleep(0.005)

    try:
        assert reader.emit(("s", 1), ("dot", 1)).wait(2)
        assert "s" in listener.get_pressed_keys()
        state.active = 0  # Minimized/no active window.
        wait_until(lambda: not listener.get_pressed_keys())
        assert listener.pop_pressed_keys() == [FOCUS_LOST]
        assert reader.emit(("enter", 1), ("space", 1), ("s", 2)).wait(2)
        assert listener.pop_pressed_keys() == []
        state.active = 42
        wait_until(lambda: bool(listener.pop_pressed_events()))
        assert reader.emit(("s", 2)).wait(2)
        assert not listener.get_pressed_keys()
        assert listener.get_key() is None
        assert reader.emit(("s", 0), ("s", 1)).wait(2)
        assert listener.get_pressed_keys() == frozenset({"s"})
        assert listener.pop_pressed_keys() == ["s"]
        assert reader.emit(("s", 0)).wait(2)
        assert not listener.get_pressed_keys()
    finally:
        listener.close()
    assert reader.closed
    assert all(connection.closed for connection in state.connections)


def test_terminal_keyboard_double_dot_cannot_span_focus_loss(
    terminal_keyboard_display, rlt_keyboard_control
):
    from rlinf.utils.keyboard_focus import configure_terminal_keyboard

    configure_terminal_keyboard(True)
    env, reader = rlt_keyboard_control(double_dot=True)
    action = np.zeros(7)
    _press_keys(reader, "dot")
    assert not env.step(action)[-1]["rlt_switch_flags"]
    terminal_keyboard_display.active = 88
    time.sleep(0.06)
    terminal_keyboard_display.active = 42
    time.sleep(0.06)
    _press_keys(reader, "dot")
    assert not env.step(action)[-1]["rlt_switch_flags"]
    _press_keys(reader, "dot")
    assert env.step(action)[-1]["rlt_switch_flags"]
    terminal_keyboard_display.active = 88
    time.sleep(0.06)
    terminal_keyboard_display.active = 42
    time.sleep(0.06)
    assert env.step(action)[-1]["rlt_switch_flags"]

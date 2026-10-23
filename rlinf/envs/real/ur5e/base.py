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

"""UR5e observations and local EE actions with RTDE impedance tracking."""

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation

from rlinf.envs.real.cartesian import CartesianEnv, CartesianEnvConfig
from rlinf.envs.real.utils.seeding import seed_sampled_spaces
from rlinf.robotics import Robot
from rlinf.robotics.parts.arms.ur5e import UR5eRobotState
from rlinf.robotics.parts.end_effectors import EndEffector
from rlinf.robotics.robots.ur5e import UR5eConfig, UR5eRobot

from .motion import URTargetComposer, matrix_pose, pose_matrix, tcp_to_ee


@dataclass
class UR5eEnvConfig(CartesianEnvConfig):
    """UR TCP reset settings; range uses metres and angular offsets in degrees."""

    target_ee_pose: np.ndarray = field(
        default_factory=lambda: np.array([0.5, 0.0, 0.1, -3.14, 0.0, 0.0])
    )
    reset_pose_range: np.ndarray = field(
        default_factory=lambda: np.array([0.03, 0.03, 0.03, 0.0, 0.0, 0.0])
    )

    reference_frame: str = "tcp"
    action_idle_hold_enabled: bool = True
    action_idle_position_threshold: float = 0.0001
    action_idle_rotation_threshold: float = 0.001
    gripper_close_threshold: float = 0.7
    payload_settle_seconds: float = 0.25
    ft_zero_settle_seconds: float = 0.50
    manual_episode_control_only: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        self.reset_pose_range = np.asarray(self.reset_pose_range, dtype=np.float64)
        if (
            self.reset_pose_range.shape != (6,)
            or not np.isfinite(self.reset_pose_range).all()
        ):
            raise ValueError("Expected reset_pose_range to contain six finite values.")
        self.reset_pose_range = np.abs(self.reset_pose_range)
        if self.reference_frame not in {"tcp", "base"}:
            raise ValueError("Expected UR reference_frame 'tcp' or 'base'.")
        if not 0 <= self.gripper_close_threshold <= 1:
            raise ValueError("Expected gripper_close_threshold in [0, 1].")
        settings = (
            self.action_idle_position_threshold,
            self.action_idle_rotation_threshold,
            self.payload_settle_seconds,
            self.ft_zero_settle_seconds,
        )
        if not np.isfinite(settings).all() or min(settings[2:]) < 0:
            raise ValueError(
                "Expected finite UR thresholds and nonnegative settle times."
            )


class UR5eEnv(CartesianEnv):
    """Compose local EE deltas without changing other Cartesian robots."""

    CONFIG_CLS = UR5eEnvConfig
    HARDWARE_CONFIG_CLS = UR5eConfig
    STATE_CLS = UR5eRobotState
    TELEOP_DEFAULT = "none"
    ACTION_WRAPPERS = ()
    TRANSFORMS = ("Quat2EulerWrapper",)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._composer = URTargetComposer()
        self._reference_ee_pose: np.ndarray | None = None
        self._tcp_offset = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        self._gripper_target: float | None = None
        super().__init__(*args, **kwargs)

    @staticmethod
    def episode_wrappers(cfg: Mapping[str, Any]) -> tuple:
        """Attach UR evaluation controls without registering a global mode."""
        settings = cfg.get("ur_evaluation", {})
        if not settings.get("enabled", False):
            return ()
        from functools import partial

        from .evaluation import UREvaluationWrapper

        return (
            partial(
                UREvaluationWrapper,
                **{key: value for key, value in settings.items() if key != "enabled"},
            ),
        )

    def _init_action_obs_spaces(self) -> None:
        super()._init_action_obs_spaces()
        for name in self.observation_space["frames"]:
            self.observation_space["frames"][name] = gym.spaces.Box(
                0,
                255,
                shape=(224, 224, 3),
                dtype=np.uint8,
            )
            self._base_observation_space["frames"][name] = self.observation_space[
                "frames"
            ][name]
        if "depths" in self.observation_space:
            for name in self.observation_space["depths"]:
                space = gym.spaces.Box(0.0, np.inf, shape=(224, 224), dtype=np.float32)
                self.observation_space["depths"][name] = space
                self._base_observation_space["depths"][name] = space

    def _frames_from(self, reading: dict[str, Any]) -> tuple[dict, dict]:
        """Preserve the full RGB field of view with OpenPI's letterbox resize."""
        from PIL import Image

        frames, depths, display = {}, {}, {}
        for name in self._cameras:
            rgb = np.asarray(reading[name]["frame"])[..., ::-1].copy()
            height, width = rgb.shape[:2]
            ratio = max(width / 224, height / 224)
            size = (int(width / ratio), int(height / ratio))
            origin = ((224 - size[0]) // 2, (224 - size[1]) // 2)
            canvas = Image.new("RGB", (224, 224), 0)
            canvas.paste(
                Image.fromarray(rgb).resize(size, Image.Resampling.BILINEAR), origin
            )
            frames[name] = np.asarray(canvas).copy()
            display[name] = frames[name]
            display[f"{name}_full"] = rgb
            if "depth" in reading[name]:
                depth = np.asarray(reading[name]["depth"], dtype=np.float32).squeeze()
                canvas_depth = Image.new("F", (224, 224), 0.0)
                canvas_depth.paste(
                    Image.fromarray(depth).resize(size, Image.Resampling.NEAREST),
                    origin,
                )
                depths[name] = np.asarray(canvas_depth).copy()
        self.camera_player.put_frame(display)
        return frames, depths

    def _get_observation(self) -> dict[str, Any]:
        observation = super()._get_observation()
        if self.config.is_dummy:
            return observation
        ee = tcp_to_ee(observation["state"]["tcp_pose"], self._tcp_offset)
        if self.config.reference_frame == "tcp" and self._reference_ee_pose is not None:
            ee = matrix_pose(
                np.linalg.inv(pose_matrix(self._reference_ee_pose)) @ pose_matrix(ee)
            )
        observation["state"]["tcp_pose"] = ee.astype(np.float32)
        # RTDE velocity and wrench retain the training data's base-frame convention.
        return observation

    def step(self, action: np.ndarray) -> tuple:
        """Apply one local EE increment, then return a fresh measured observation."""
        started = time.monotonic()
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError("Expected seven finite UR action values.")
        if not self.config.is_dummy:
            self._robot_state = self._read_robot()
            self._tcp_offset = self._arm.get_tcp_offset()
            scaled = action.copy()
            scaled[:3] *= self.config.action_scale[0]
            scaled[3:6] *= self.config.action_scale[1]
            target = self._composer.target(
                self._robot_state.tcp_pose,
                self._tcp_offset,
                scaled,
                reference_frame=self.config.reference_frame,
                hold_enabled=self.config.action_idle_hold_enabled,
                position_threshold=self.config.action_idle_position_threshold,
                rotation_threshold=self.config.action_idle_rotation_threshold,
            )
            self.next_position = self._clip_position_to_safety_box(target)
            self._move_action(self.next_position)
            self._end_effector_action(action[6:])
        self._num_steps += 1
        time.sleep(
            max(0.0, 1.0 / self.config.step_frequency - (time.monotonic() - started))
        )
        if not self.config.is_dummy:
            self._robot_state = self._read_robot()
        observation = self._get_observation()
        reward = self._calc_step_reward(observation, True)
        terminated = (
            reward == 1.0
            and self._success_hold_counter >= self.config.success_hold_steps
        )
        return (
            observation,
            reward * self.config.reward_scale,
            terminated,
            self._num_steps >= self.config.max_num_steps,
            {},
        )

    def _interpolate_move(self, pose: np.ndarray, timeout: float = 1.5) -> None:
        """Use the UR moveL reset path, including construction-time homing."""
        del timeout
        self._end_effector.open()
        self._gripper_target = 1.0
        self._arm.reset_tcp(
            pose,
            self.config.payload_settle_seconds,
            self.config.ft_zero_settle_seconds,
        )
        self._composer.clear()
        self._tcp_offset = self._arm.get_tcp_offset()
        self._robot_state = self._read_robot()
        self._reference_ee_pose = tcp_to_ee(
            self._robot_state.tcp_pose, self._tcp_offset
        )

    def reset(
        self,
        joint_reset: bool = False,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return to a sampled TCP pose, without periodic joint resets."""
        if joint_reset:
            raise ValueError(
                "UR5e task reset uses TCP poses; joint_reset must be False."
            )
        gym.Env.reset(self, seed=seed)
        self._composer.clear()
        self._gripper_target = None
        seed_sampled_spaces(seed, self._base_observation_space)
        if self.config.is_dummy:
            return self._get_observation(), {}
        self._success_hold_counter = 0
        self._arm.reconfigure_compliance_params(self.config.compliance_param)
        self.go_to_rest()
        self._clear_error()
        self._num_steps = 0
        return self.observe(), {}

    def sample_reset_pose(self) -> np.ndarray:
        """Sample base-frame XYZ and XYZW, following VLA-Precision's reset ranges."""
        pose = self.config.reset_ee_pose.copy()
        if self.config.enable_random_reset:
            bounds = self.config.reset_pose_range
            pose[:3] += self.np_random.uniform(-bounds[:3], bounds[:3])
            pose[3:] += np.deg2rad(self.np_random.uniform(-bounds[3:], bounds[3:]))
        return np.concatenate(
            (pose[:3], Rotation.from_euler("xyz", pose[3:]).as_quat())
        )

    def go_to_rest(self, joint_reset: bool = False) -> None:
        """Open the gripper, moveL to the reset TCP and zero the force sensor."""
        if joint_reset:
            raise ValueError(
                "UR5e task reset uses TCP poses; joint_reset must be False."
            )
        self._interpolate_move(self.sample_reset_pose())

    def _build_robot(
        self, controller_node_rank: int, camera_node_rank: int | None
    ) -> Robot:
        return UR5eRobot.build(
            config=self.hardware,
            node_rank=controller_node_rank,
            worker_rank=self.env_worker_rank,
            env_idx=self.env_idx,
            cameras={info.name: info for info in self._camera_infos},
        )

    @property
    def _ee_interface(self) -> EndEffector | type[EndEffector]:
        """Resolve the UR gripper before connecting, or return its live part."""
        part = getattr(self, "_end_effector", None)
        return (
            part
            if part is not None
            else EndEffector.backend(self.hardware.end_effector_type)
        )

    def _read_robot(self) -> UR5eRobotState:
        """Expose measured gripper position in the dataset's raw/1000 units."""
        state = super()._read_robot()
        state.gripper_position /= self._end_effector.max_width
        return state

    def _end_effector_action(self, ee_action: np.ndarray) -> bool:
        """Threshold the predicted opening, preserving measured feedback separately."""
        value = float(ee_action[0] > self.config.gripper_close_threshold)
        if value == self._gripper_target:
            return False
        changed = self._end_effector.command(
            np.array([value * self._end_effector.max_width])
        )
        self._gripper_target = value
        return changed

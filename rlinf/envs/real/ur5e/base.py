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

"""UR5e hardware using RLinf's existing Cartesian delta environment."""

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


@dataclass
class UR5eEnvConfig(CartesianEnvConfig):
    """UR TCP reset settings; range uses metres and angular offsets in degrees."""

    target_ee_pose: np.ndarray = field(
        default_factory=lambda: np.array([0.5, 0.0, 0.1, -3.14, 0.0, 0.0])
    )
    reset_pose_range: np.ndarray = field(
        default_factory=lambda: np.array([0.03, 0.03, 0.03, 0.0, 0.0, 0.0])
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.reset_pose_range = np.asarray(self.reset_pose_range, dtype=np.float64)
        if (
            self.reset_pose_range.shape != (6,)
            or not np.isfinite(self.reset_pose_range).all()
        ):
            raise ValueError("Expected reset_pose_range to contain six finite values.")
        self.reset_pose_range = np.abs(self.reset_pose_range)


class UR5eEnv(CartesianEnv):
    """UR hardware with shared Cartesian motion, observations and reset.

    The installed gripper accepts normalized absolute positions (0 closed,
    1 open). TCP deltas use the shared Cartesian composition rules.
    """

    CONFIG_CLS = UR5eEnvConfig
    HARDWARE_CONFIG_CLS = UR5eConfig
    STATE_CLS = UR5eRobotState
    TELEOP_DEFAULT = "none"
    ACTION_WRAPPERS = ()

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
        """Interpolate to the TCP reset pose through the UR impedance controller."""
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
        """Map normalized absolute gripper position to the standard width API."""
        value = float(np.clip(ee_action[0], 0.0, 1.0))
        return self._end_effector.command(
            np.array([value * self._end_effector.max_width])
        )

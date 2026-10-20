# Copyright 2025 The RLinf Authors.
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

from dataclasses import dataclass, field

import numpy as np

from rlinf.envs.real.cartesian import CartesianEnv, CartesianEnvConfig
from rlinf.robotics import FrankaConfig, FrankaRobot, Robot
from rlinf.robotics.parts.arms.franka import FrankaRobotState
from rlinf.robotics.parts.end_effectors import EndEffector

#: Default Cartesian impedance gains shared by Franka tasks.
COMPLIANCE_DEFAULTS: dict[str, float] = {
    "translational_stiffness": 1000,
    "translational_damping": 89,
    "rotational_stiffness": 150,
    "rotational_damping": 7,
    "translational_Ki": 0,
    "rotational_Ki": 0,
    "translational_clip_x": 0.003,
    "translational_clip_y": 0.003,
    "translational_clip_z": 0.01,
    "translational_clip_neg_x": 0.003,
    "translational_clip_neg_y": 0.003,
    "translational_clip_neg_z": 0.01,
    "rotational_clip_x": 0.02,
    "rotational_clip_y": 0.02,
    "rotational_clip_z": 0.02,
    "rotational_clip_neg_x": 0.02,
    "rotational_clip_neg_y": 0.02,
    "rotational_clip_neg_z": 0.02,
}


def compliance(**overrides: float) -> dict[str, float]:
    """Return :data:`COMPLIANCE_DEFAULTS` with ``overrides`` applied.

    Raises:
        KeyError: If an override is not a supported controller gain.
    """
    unknown = set(overrides) - set(COMPLIANCE_DEFAULTS)
    if unknown:
        raise KeyError(
            f"Unknown compliance gains {sorted(unknown)}. "
            f"Known: {sorted(COMPLIANCE_DEFAULTS)}."
        )
    return {**COMPLIANCE_DEFAULTS, **overrides}


@dataclass
class FrankaEnvConfig(CartesianEnvConfig):
    """Cartesian task settings with Franka's default target and joint reset."""

    target_ee_pose: np.ndarray = field(
        default_factory=lambda: np.array([0.5, 0.0, 0.1, -3.14, 0.0, 0.0])
    )
    joint_reset_qpos: list[float] = field(
        default_factory=lambda: [0, 0, 0, -1.9, -0, 2, 0]
    )


class FrankaEnv(CartesianEnv):
    """Franka hardware with shared Cartesian task behavior."""

    CONFIG_CLS = FrankaEnvConfig
    HARDWARE_CONFIG_CLS = FrankaConfig
    STATE_CLS = FrankaRobotState
    TELEOP = ("spacemouse", "gello", "glove", "pico")
    TELEOP_DEFAULT = "spacemouse"
    ACTION_WRAPPERS = ("GripperCloseEnv",)

    @property
    def _franka_state(self) -> FrankaRobotState:
        """Expose the state name used by existing Franka task subclasses."""
        return self._robot_state

    @_franka_state.setter
    def _franka_state(self, state: FrankaRobotState) -> None:
        self._robot_state = state

    def _build_robot(
        self, controller_node_rank: int, camera_node_rank: int | None
    ) -> Robot:
        """Declare this environment's arm, end effector and cameras."""
        hardware = self.hardware
        return FrankaRobot.build(
            robot_ip=self.hardware.robot_ip,
            env_idx=self.env_idx,
            node_rank=controller_node_rank,
            worker_rank=self.env_worker_rank,
            backend=self.hardware.backend,
            gripper_type=self.hardware.gripper_type,
            compliance=hardware.compliance,
            realtime_config=hardware.realtime_config,
            end_effector_type=self.hardware.end_effector_type,
            end_effector_config=self.hardware.end_effector_config,
            gripper_connection=self.hardware.gripper_connection,
            cameras={info.name: info for info in self._camera_infos},
            camera_node_rank=camera_node_rank,
        )

    @property
    def _ee_interface(self) -> EndEffector | type[EndEffector]:
        """Return the connected part, or its driver contract for dummy runs."""
        end_effector = getattr(self, "_end_effector", None)
        if end_effector is not None:
            return end_effector
        return FrankaRobot.end_effector_class(
            backend=self.hardware.backend,
            gripper_type=self.hardware.gripper_type,
            end_effector_type=self.hardware.end_effector_type,
        )

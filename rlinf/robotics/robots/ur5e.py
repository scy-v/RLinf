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

"""UR5e composition with an RTDE arm, serial gripper and existing cameras."""

from dataclasses import dataclass, field
from typing import Any

from rlinf.robotics.discovery import RobotConfig
from rlinf.robotics.parts.arms.ur5e import UR5eArm
from rlinf.robotics.parts.cameras import Camera
from rlinf.robotics.parts.end_effectors import EndEffector
from rlinf.robotics.robot import Robot


class UR5eRobot(Robot):
    """Robot parts share RLinf's placement and connection lifecycle."""

    ROBOT_TYPE = "UR5e"

    @classmethod
    def build(
        cls,
        *,
        config: "UR5eConfig",
        node_rank: int,
        worker_rank: int = 0,
        env_idx: int = 0,
        cameras: dict[str, Any] | None = None,
    ) -> "UR5eRobot":
        """Declare the arm and gripper on their controller node, plus cameras."""
        return cls(
            arm=UR5eArm(
                config.robot_ip,
                config.speed,
                config.acceleration,
                impedance=config.impedance,
                node_rank=node_rank,
                worker_name=f"UR5eArm-{worker_rank}-{env_idx}",
            ),
            end_effector=EndEffector.backend(config.end_effector_type).declare(
                port=config.gripper_connection,
                node_rank=node_rank,
                worker_name=f"UR5eGripper-{worker_rank}-{env_idx}",
                **config.end_effector_config,
            ),
            **Camera.declare(cameras, node_rank=config.camera_node_rank),
        )


@dataclass
class UR5eConfig(RobotConfig):
    """Physical endpoints and RTDE motion settings; no task or policy fields."""

    robot_ip: str = ""
    gripper_connection: str = "/dev/ur5e_left_gripper"
    end_effector_type: str = "pgi"
    end_effector_config: dict[str, Any] = field(default_factory=dict)
    camera_serials: list[str] | None = None
    camera_type: str = "realsense"
    camera_node_rank: int | None = None
    controller_node_rank: int | None = None
    impedance: dict[str, Any] = field(default_factory=dict)
    speed: float = 0.1
    acceleration: float = 0.2

    def __post_init__(self) -> None:
        """Normalize scheduler declarations without connecting hardware."""
        if self.camera_serials is not None:
            self.camera_serials = list(self.camera_serials)
        if self.speed <= 0 or self.acceleration <= 0:
            raise ValueError("Expected positive UR5e speed and acceleration.")


UR5eRobot.register_type(UR5eConfig)

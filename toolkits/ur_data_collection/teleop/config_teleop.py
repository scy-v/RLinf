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

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("lerobot_teleoperator_ur5e")
@dataclass
class UR5eTeleopConfig(TeleoperatorConfig):
    port: str = "4443"
    host: str = "0.0.0.0"
    use_gripper: bool = True
    init_gripper: bool = True
    step_size: float = 0.048
    rot_step_size: float = 0.05
    alternate_step_size: float | None = None
    alternate_rot_step_size: float | None = None
    reference_frame: str = "base"
    control_frame_euler_deg: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    select_vector: list | None = None

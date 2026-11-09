# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Adapt the unchanged keyboard-collection observation to RLinf PI05."""

from typing import Any

import numpy as np


class URRolloutPolicy:
    """Select SFT state fields and prepare the two camera inputs."""

    def __init__(
        self, policy: Any, state_names: list[str], obs_indices: list[int], task: str
    ) -> None:
        self.policy = policy
        self.state_names = tuple(state_names)
        self.indices = np.asarray(obs_indices, dtype=int) - 1
        if (
            len(self.indices) != 19
            or np.any(self.indices < 0)
            or np.any(self.indices >= len(self.state_names))
        ):
            raise ValueError("Expected 19 valid one-based UR state indices.")
        self.task = task

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Return decoded single-arm actions in the collection coordinate frame."""
        from evaluations.realworld.vla_precision_ur.image_tools import resize_with_pad

        state = np.asarray(
            [observation[name] for name in self.state_names], dtype=np.float32
        )
        return self.policy.infer(
            {
                "observation/state": state[self.indices],
                "observation/image": resize_with_pad(
                    observation["exterior_image"], 224, 224
                ),
                "observation/wrist_image": resize_with_pad(
                    observation["wrist_image"], 224, 224
                ),
                "prompt": self.task,
            }
        )

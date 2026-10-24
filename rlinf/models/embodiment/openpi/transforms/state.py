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
"""One-based observation selection shared by training and environment inputs."""

import dataclasses
from collections.abc import Sequence

import numpy as np
import torch


def validate_obs_indices(indices: Sequence[int]) -> tuple[int, ...]:
    """Validate ordered, unique, one-based observation indices."""
    indices = tuple(indices)
    if not indices or any(type(i) is not int or i < 1 for i in indices):
        raise ValueError(
            f"Expected nonempty positive integer obs_indices, got {indices}"
        )
    if len(set(indices)) != len(indices):
        raise ValueError(f"Expected unique obs_indices, got {indices}")
    return indices


@dataclasses.dataclass(frozen=True)
class SelectState:
    """Select raw observation dimensions before robot-specific transforms.

    Indices are one-based and always apply to the raw input, including when
    the selection has the same length as that input.
    """

    indices: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "indices", validate_obs_indices(self.indices))

    def select(self, state: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """Select the last axis, preserving NumPy/Torch type and device."""
        if state.shape[-1] < max(self.indices):
            raise ValueError(
                f"Expected raw state dimension >= {max(self.indices)}, got {state.shape[-1]}"
            )
        return state[..., [i - 1 for i in self.indices]]

    def __call__(self, data: dict) -> dict:
        return {**data, "observation/state": self.select(data["observation/state"])}

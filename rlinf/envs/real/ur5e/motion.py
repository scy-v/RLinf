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

"""UR pose composition and stationary-axis target holds."""

import numpy as np
from scipy.spatial.transform import Rotation


def pose_matrix(pose: np.ndarray) -> np.ndarray:
    """Convert XYZ + XYZW to a homogeneous pose."""
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix()
    result[:3, 3] = pose[:3]
    return result


def matrix_pose(matrix: np.ndarray) -> np.ndarray:
    """Convert a homogeneous pose to XYZ + XYZW."""
    return np.r_[matrix[:3, 3], Rotation.from_matrix(matrix[:3, :3]).as_quat()]


def tcp_to_ee(tcp: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """Remove the configured tool offset from a measured TCP pose."""
    return matrix_pose(pose_matrix(tcp) @ np.linalg.inv(pose_matrix(offset)))


def ee_to_tcp(ee: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """Apply the configured tool offset to an EE target."""
    return matrix_pose(pose_matrix(ee) @ pose_matrix(offset))


class URTargetComposer:
    """Compose local EE increments and latch idle axes in the base frame."""

    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        """Discard the hold anchor when an episode is reset."""
        self.anchor: np.ndarray | None = None
        self.previous_motion = np.zeros(6, dtype=bool)

    def target(
        self,
        tcp: np.ndarray,
        offset: np.ndarray,
        action: np.ndarray,
        *,
        reference_frame: str,
        hold_enabled: bool,
        position_threshold: float,
        rotation_threshold: float,
    ) -> np.ndarray:
        """Return an absolute TCP target without changing measured state."""
        ee = tcp_to_ee(tcp, offset)
        current_rotation = Rotation.from_quat(ee[3:])
        delta_rotation = Rotation.from_euler("xyz", action[3:6])
        if reference_frame == "tcp":
            delta_position = current_rotation.apply(action[:3])
            rotation = current_rotation * delta_rotation
            base_delta_rotation = (
                current_rotation * delta_rotation * current_rotation.inv()
            )
        else:
            delta_position = action[:3]
            rotation = delta_rotation * current_rotation
            base_delta_rotation = delta_rotation
        target = ee_to_tcp(np.r_[ee[:3] + delta_position, rotation.as_quat()], offset)
        if not hold_enabled or (position_threshold < 0 and rotation_threshold < 0):
            self.clear()
            return target
        current = np.r_[tcp[:3], Rotation.from_quat(tcp[3:]).as_rotvec()]
        candidate = np.r_[target[:3], Rotation.from_quat(target[3:]).as_rotvec()]
        if self.anchor is None:
            self.anchor = current.copy()
        translation_motion = (
            np.ones(3, dtype=bool)
            if position_threshold < 0
            else np.abs(delta_position) > position_threshold
        )
        rotation_motion = (
            rotation_threshold < 0
            or np.linalg.norm(base_delta_rotation.as_rotvec()) > rotation_threshold
        )
        motion = np.r_[translation_motion, np.repeat(rotation_motion, 3)]
        stopped = self.previous_motion & ~motion
        self.anchor[stopped] = current[stopped]
        held = self.anchor.copy()
        held[motion] = candidate[motion]
        self.anchor[motion] = current[motion]
        self.previous_motion = motion
        return np.r_[held[:3], Rotation.from_rotvec(held[3:]).as_quat()]

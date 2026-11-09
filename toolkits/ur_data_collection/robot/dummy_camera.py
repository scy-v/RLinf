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

"""A hardware-free camera that emits fixed dummy RGB frames."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from lerobot.cameras import Camera, CameraConfig, ColorMode
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from numpy.typing import NDArray


@CameraConfig.register_subclass("ur5e_dummy")
@dataclass(kw_only=True)
class DummyCameraConfig(CameraConfig):
    value: int = 0

    def __post_init__(self) -> None:
        if self.fps is None or self.width is None or self.height is None:
            raise ValueError("DummyCamera requires fps, width, and height.")
        if self.fps <= 0 or self.width <= 0 or self.height <= 0:
            raise ValueError("DummyCamera fps, width, and height must be positive.")
        if not 0 <= self.value <= 255:
            raise ValueError("DummyCamera value must be in [0, 255].")


class DummyCamera(Camera):
    """LeRobot camera implementation that never accesses camera hardware."""

    def __init__(self, config: DummyCameraConfig):
        super().__init__(config)
        self.config = config
        self._is_connected = False
        self._frame = np.full(
            (int(config.height), int(config.width), 3),
            fill_value=int(config.value),
            dtype=np.uint8,
        )

    def __str__(self) -> str:
        return f"DummyCamera({self.width}x{self.height})"

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        return []

    def connect(self, warmup: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} is already connected.")
        self._is_connected = True

    def read(self, color_mode: ColorMode | None = None) -> NDArray[Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        return self._frame.copy()

    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        return self.read()

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        self._is_connected = False

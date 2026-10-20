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

"""DH Robotics PGI/PGE grippers over a dedicated serial connection."""

from typing import Any

import numpy as np

from rlinf.robotics.parts.claims import DeviceClaim
from rlinf.robotics.parts.end_effectors.base import EndEffector
from rlinf.robotics.parts.end_effectors.grippers.base import BaseGripper


@EndEffector.register("pgi", "pge")
class PGIGripper(BaseGripper):
    """Expose PGE serial positions as physical opening widths in metres."""

    @classmethod
    def declare(
        cls,
        *,
        ros: Any = None,
        port: str | None = None,
        robot_ip: str | None = None,
        **settings: Any,
    ) -> "PGIGripper":
        """Declare a gripper on its dedicated serial connection."""
        return cls(port=port, **settings)

    def __init__(
        self,
        port: str,
        max_width: float = 0.07,
        min_position: int = 80,
        max_position: int = 1000,
        force: int = 30,
        speed: int = 50,
    ) -> None:
        if not port or max_width <= 0 or not 0 <= min_position < max_position <= 1000:
            raise ValueError(
                "Expected a serial port, positive max_width and 0 <= min_position < max_position <= 1000."
            )
        if not 20 <= force <= 100 or not 1 <= speed <= 100:
            raise ValueError("Expected PGI force in [20, 100] and speed in [1, 100].")
        self.port = port
        self._max_width = max_width
        self.min_position = min_position
        self.max_position = max_position
        self.force = force
        self.speed = speed
        self._claim = DeviceClaim(f"pgi:{port}", type(self).__name__)

    def _open(self) -> Any:
        from pyDHgripper import PGE

        self._claim.acquire()
        device = None
        try:
            device = PGE(port=self.port)
            device.init_feedback()
            device.set_force(self.force)
            device.set_vel(self.speed)
            return device
        except BaseException:
            try:
                if device is not None:
                    device.ser.close()
            finally:
                self._claim.release()
            raise

    def _release(self, device: Any) -> None:
        try:
            device.ser.close()
        finally:
            self._claim.release()

    @property
    def max_width(self) -> float:
        """Fully open width configured for the installed gripper."""
        return self._max_width

    @property
    def position(self) -> float:
        """Measured opening in metres, using the SDK's raw 0..1000 reading."""
        raw = float(self._device.read_pos())
        if not np.isfinite(raw) or not 0 <= raw <= 1000:
            raise RuntimeError(f"Expected PGI position in [0, 1000], got {raw}.")
        return raw / 1000 * self.max_width

    @property
    def is_open(self) -> bool:
        """Whether measured opening is above the configured travel midpoint."""
        return (
            self.position / self.max_width
            > (self.min_position + self.max_position) / 2000
        )

    def is_ready(self) -> bool:
        """Report whether this gripper owns an initialized serial connection."""
        return self.is_connected

    def move(self, width: float, speed: float = 0.3) -> None:
        """Send a nonblocking target, keeping serial writes on this part's owner."""
        if not np.isfinite(width):
            raise ValueError(f"Expected finite gripper width, got {width}.")
        raw = int(
            np.clip(
                round(width / self.max_width * 1000),
                self.min_position,
                self.max_position,
            )
        )
        self._device.set_pos(val=raw, blocking=False)

    def open(self, speed: float = 0.3) -> None:
        """Command the configured open position."""
        self.move(self.max_position / 1000 * self.max_width, speed)

    def close(self, speed: float = 0.3) -> None:
        """Command the configured closed position."""
        self.move(self.min_position / 1000 * self.max_width, speed)

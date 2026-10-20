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

"""Operator-requested episode reset, independent of robot hardware."""

import copy
import math
import threading
import time
from typing import Any

import gymnasium as gym

from .session import KeyboardSession


class KeyboardResetWrapper(KeyboardSession):
    """End an episode on two Enter press edges within a configured interval.

    Only the outer environment performs reset. Actions remain blocked after a
    request until reset completes, including the rest of an action chunk.
    """

    MANUAL_EPISODE_CONTROL_ONLY = True
    ENABLE_IN_DUMMY = True

    def __init__(
        self,
        env: gym.Env,
        double_press_interval: float = 0.5,
        wait_for_start: bool = False,
    ) -> None:
        if not math.isfinite(double_press_interval) or double_press_interval <= 0:
            raise ValueError("Expected a finite positive double_press_interval.")
        self._observe = None
        if wait_for_start:
            try:
                self._observe = env.get_wrapper_attr("observe")
            except AttributeError as exc:
                raise ValueError(
                    "wait_for_start requires an environment with observe()."
                ) from exc
        super().__init__(env)
        self.double_press_interval = double_press_interval
        self.wait_for_start = wait_for_start
        self._waiting_for_start = False
        self._stop = threading.Event()
        self._last_obs: Any = None
        self._accept_after = 0.0
        self.begin_episode()

    def begin_episode(self) -> None:
        """Clear the pending reset and double-press sequence."""
        self._first_enter: float | None = None
        self._reset_requested = False

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        obs, info = super().reset(**kwargs)
        self._accept_after = time.time()
        # Input while the robot is returning home belongs to the old episode.
        self.drain()
        if self.wait_for_start:
            self._wait_for_enter()
            obs = self._observe()
            self._accept_after = time.time()
            self.drain()
        self._last_obs = copy.deepcopy(obs)
        self.log(
            "Double-press Enter within %.2f seconds to reset.",
            self.double_press_interval,
        )
        return obs, info

    @property
    def waiting_for_start(self) -> bool:
        """Whether the homed robot is waiting for operator confirmation."""
        return self._waiting_for_start

    def _wait_for_enter(self) -> None:
        self._waiting_for_start = True
        self.log("Robot reset complete. Press Enter once to start the policy.")
        try:
            while not self._stop.is_set():
                for key, timestamp in self.listener.pop_pressed_events():
                    if timestamp >= self._accept_after and key in {
                        "Key.enter",
                        "Key.kpenter",
                    }:
                        self.log("Enter pressed; starting the policy.")
                        return
                self._stop.wait(0.02)
            raise RuntimeError("Keyboard control closed while waiting for Enter.")
        finally:
            self._waiting_for_start = False

    def close(self) -> None:
        """Interrupt the start wait and release keyboard and robot resources."""
        self._stop.set()
        super().close()

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        if self._last_obs is None:
            raise RuntimeError("Expected reset() before keyboard-controlled stepping.")
        for key, timestamp in self.listener.pop_pressed_events():
            if timestamp < self._accept_after:
                continue
            if key not in {"Key.enter", "Key.kpenter"}:
                self._first_enter = None
                continue
            if (
                self._first_enter is not None
                and 0 <= timestamp - self._first_enter <= self.double_press_interval
            ):
                self._reset_requested = True
                self._first_enter = None
                break
            self._first_enter = timestamp
        if self._reset_requested:
            return (
                copy.deepcopy(self._last_obs),
                0.0,
                False,
                True,
                {"manual_reset": True},
            )
        obs, _, _, _, info = self.env.step(action)
        self._last_obs = copy.deepcopy(obs)
        return obs, 0.0, False, False, info

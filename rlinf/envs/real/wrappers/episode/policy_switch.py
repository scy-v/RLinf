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

import copy
import math
import threading
import time
from typing import Any, SupportsFloat

import numpy as np
from gymnasium.core import ActType, Env, ObsType

from .session import KeyboardSession


class KeyboardRLTPolicySwitchWrapper(KeyboardSession):
    """Use the configured switch key to enter the RLT critical phase.

    An optional double-press success key replaces task rewards with labels.
    ``start_with_actor`` marks the critical phase from the first chunk after reset.
    The rollout route also checks training readiness before actor control.
    """

    def __init__(
        self,
        env: Env,
        success_key: str | None = None,
        success_double_press_interval: float = 0.5,
        switch_key: str = "b",
        start_with_actor: bool = False,
        switch_double_press_interval: float | None = None,
        wait_for_start: bool = False,
        wait_for_reset: bool = False,
        wait_for_first_reset: bool = False,
    ) -> None:
        if success_key == switch_key:
            raise ValueError(
                f"Expected distinct success and switch keys, got {success_key!r}."
            )
        if (
            not math.isfinite(success_double_press_interval)
            or success_double_press_interval <= 0
        ):
            raise ValueError(
                "Expected a finite positive success_double_press_interval."
            )
        if switch_double_press_interval is not None and (
            not math.isfinite(switch_double_press_interval)
            or switch_double_press_interval <= 0
        ):
            raise ValueError("Expected a finite positive switch_double_press_interval.")
        super().__init__(env)
        self.success_key = success_key
        self.success_double_press_interval = success_double_press_interval
        self.wait_for_first_reset = wait_for_first_reset
        self._initial_reset_complete = False
        self.wait_for_reset = wait_for_reset
        self.wait_for_start = wait_for_start
        self._stop = threading.Event()
        self.switch_key = switch_key
        self.switch_double_press_interval = switch_double_press_interval
        self._first_switch_press: float | None = None
        self.start_with_actor = start_with_actor
        self._rlt_switch_flags = start_with_actor
        self._manual_success = False
        self._last_obs: Any = None
        self._first_success_press: float | None = None
        self._accept_after = 0.0
        self._success_hold_action: np.ndarray | None = None

    @property
    def rlt_switch_flags(self) -> bool:
        """Return whether the episode is in its RLT critical phase."""
        return self._rlt_switch_flags

    def begin_episode(self) -> None:
        """Restore the configured initial control phase."""
        self._rlt_switch_flags = self.start_with_actor
        self._manual_success = False
        self._first_success_press = None
        self._first_switch_press = None
        self._success_hold_action = None

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        """Reset the episode and restore the initial control phase."""
        if self.wait_for_reset or (
            self.wait_for_first_reset and not self._initial_reset_complete
        ):
            stop = getattr(self.base_env(), "stop_motion", None)
            if callable(stop):
                stop()
            self._wait_for_enter("[RLT] Arm stopped. Press Enter to allow reset.")
        obs, info = super().reset(**kwargs)
        self._initial_reset_complete = True
        self._last_obs = copy.deepcopy(obs)
        if self.success_key is not None or self.start_with_actor:
            info["rlt_switch_flags"] = self._rlt_switch_flags
        if (
            self.success_key is not None
            or self.switch_double_press_interval is not None
        ):
            self._accept_after = time.time()
            self.drain()
        self.log(
            "[RLT] Episode reset; critical_phase=%s. Switch: %s (%s).",
            self._rlt_switch_flags,
            self.switch_key,
            "single press"
            if self.switch_double_press_interval is None
            else f"double press within {self.switch_double_press_interval:.2f} s",
        )
        if self.success_key is not None:
            self.log(
                "[RLT] Success: %s (double press within %.2f s).",
                self.success_key,
                self.success_double_press_interval,
            )
        if self.wait_for_start:
            self._wait_for_enter(
                "[RLT] Reset complete. Press Enter to start this episode."
            )
            observe = getattr(self.base_env(), "observe", None)
            if callable(observe):
                obs = observe()
                self._last_obs = copy.deepcopy(obs)
            self._accept_after = time.time()
            self.drain()
            self.log("[RLT] Enter confirmed; ready for the first policy action.")
        return obs, info

    def _wait_for_enter(self, prompt: str) -> None:
        """Accept a fresh Enter edge only after displaying this gate."""
        self._accept_after = time.time()
        self.drain()
        self.log(prompt)
        while not self._stop.is_set():
            if any(
                timestamp >= self._accept_after and key in {"Key.enter", "Key.kpenter"}
                for key, timestamp in self.listener.pop_pressed_events()
            ):
                return
            self._stop.wait(0.02)
        raise RuntimeError("RLT environment closed while waiting for Enter.")

    def close(self) -> None:
        """Cancel an outstanding start confirmation before releasing input."""
        self._stop.set()
        super().close()

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        if self._manual_success:
            return (
                copy.deepcopy(self._last_obs),
                0.0,
                True,
                False,
                self._terminal_info(),
            )
        event = self._consume_events() if self.success_key is not None else None
        if self._manual_success:
            self._stop_for_success()
            return (
                copy.deepcopy(self._last_obs),
                1.0,
                True,
                False,
                self._terminal_info(),
            )
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.success_key is not None:
            reward, terminated = 0.0, False
            self._last_obs = copy.deepcopy(obs)

        event = self._consume_events() or event
        if self._manual_success:
            self._stop_for_success()
            obs = copy.deepcopy(self._last_obs)
            reward, terminated, truncated = 1.0, True, False

        info["rlt_switch_flags"] = self._rlt_switch_flags
        info["rlt_policy_switch_event"] = event
        return obs, reward, terminated, truncated, info

    def _terminal_info(self) -> dict[str, Any]:
        return {
            "rlt_switch_flags": self._rlt_switch_flags,
            "rlt_policy_switch_event": "manual_success",
            "intervene_action": self._success_hold_action.copy(),
        }

    def _stop_for_success(self) -> None:
        base = self.base_env()
        stop = getattr(base, "stop_motion", None)
        if callable(stop):
            self._success_hold_action = np.asarray(stop(), dtype=np.float32)
        else:
            self._success_hold_action = np.zeros(
                self.action_space.shape, dtype=np.float32
            )
        observe = getattr(base, "observe", None)
        if callable(observe):
            self._last_obs = copy.deepcopy(observe())

    def _consume_events(self) -> str | None:
        event: str | None = None
        events = (
            [(key, time.time()) for key in self.presses()]
            if self.success_key is None and self.switch_double_press_interval is None
            else self.listener.pop_pressed_events()
        )
        for key, timestamp in events:
            if timestamp < self._accept_after:
                continue
            if key == self.switch_key:
                if self.switch_double_press_interval is not None:
                    previous = self._first_switch_press
                    self._first_switch_press = timestamp
                    if previous is None or not (
                        0 <= timestamp - previous <= self.switch_double_press_interval
                    ):
                        continue
                    self._first_switch_press = None
                if not self._rlt_switch_flags:
                    event = "enter_actor"
                    self._rlt_switch_flags = True
                    self.log("[RLT] Critical phase selected.")
                else:
                    event = "actor_already_active"
            elif key == self.success_key:
                previous = self._first_success_press
                self._first_success_press = timestamp
                if previous is None or not (
                    0 <= timestamp - previous <= self.success_double_press_interval
                ):
                    continue
                self._manual_success = True
                event = "manual_success"
                self.log("[RLT] Success confirmed; ending episode with reward 1.")
                break
        return event

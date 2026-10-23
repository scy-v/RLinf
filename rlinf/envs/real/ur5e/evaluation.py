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

"""UR-only keyboard evaluation with bounded episodes and durable results."""

import copy
import json
import math
import threading
import time
from pathlib import Path
from typing import Any

import gymnasium as gym

from rlinf.envs.real.wrappers.episode.session import KeyboardSession


class UREvaluationWrapper(KeyboardSession):
    """Record double-Space success, double-Enter failure and wall-clock timeouts."""

    MANUAL_EPISODE_CONTROL_ONLY = True

    def __init__(
        self,
        env: gym.Env,
        num_episodes: int = 20,
        episode_timeout_sec: float = 60.0,
        double_press_interval: float = 0.5,
        wait_for_start: bool = True,
        output_path: str = "./results/ur5e-evaluation.json",
    ) -> None:
        if num_episodes < 1 or not all(
            math.isfinite(x) and x > 0
            for x in (episode_timeout_sec, double_press_interval)
        ):
            raise ValueError(
                "Expected positive UR episode count, timeout and key interval."
            )
        self.num_episodes = num_episodes
        self.episode_timeout_sec = episode_timeout_sec
        self.double_press_interval = double_press_interval
        self.wait_for_start = wait_for_start
        self.output_path = Path(output_path).expanduser()
        self.results: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._last_obs = None
        self._boundary: tuple[bool, bool, dict] | None = None
        self._first_press: dict[str, float] = {}
        self._accept_after = 0.0
        self._episode_started = 0.0
        self._steps = 0
        self._waiting_for_start = False
        super().__init__(env)

    @property
    def evaluation_complete(self) -> bool:
        """Whether the configured number of episodes has been recorded."""
        return len(self.results) >= self.num_episodes

    @property
    def waiting_for_start(self) -> bool:
        """Whether an initial or reset pose is awaiting operator confirmation."""
        return self._waiting_for_start

    def _write_results(self) -> None:
        successes = sum(result["completed"] for result in self.results)
        payload = {
            "evaluation": "ur5e-rlinf",
            "episodes": self.results,
            "summary": {
                "episode_count": len(self.results),
                "success_count": successes,
                "success_rate": successes / len(self.results) if self.results else 0.0,
            },
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output_path.with_suffix(self.output_path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.output_path)

    def reset(self, **kwargs: Any) -> tuple[Any, dict]:
        if self.evaluation_complete:
            # No new episode is started after the last outcome; home once before exit.
            if self._boundary is not None:
                self._last_obs, _ = self.env.reset(**kwargs)
                self._boundary = None
            return copy.deepcopy(self._last_obs), {"evaluation_complete": True}
        self.drain()
        self._last_obs, info = self.env.reset(**kwargs)
        self._boundary = None
        self._first_press.clear()
        self._steps = 0
        self._accept_after = time.time()
        self.drain()
        if self.wait_for_start:
            self._waiting_for_start = True
            try:
                self.log("UR reset complete. Press Enter once to start.")
                while not self._stop.is_set():
                    events = self.listener.pop_pressed_events()
                    if any(
                        t >= self._accept_after and k in {"Key.enter", "Key.kpenter"}
                        for k, t in events
                    ):
                        break
                    self._stop.wait(0.02)
                if self._stop.is_set():
                    raise RuntimeError("UR evaluation closed while waiting for Enter.")
                self._last_obs = self.env.get_wrapper_attr("observe")()
            finally:
                self._waiting_for_start = False
            self._accept_after = time.time()
            self.drain()
        self._episode_started = time.monotonic()
        self._write_results()
        return copy.deepcopy(self._last_obs), info

    def _finish(self, status: str) -> tuple:
        completed = status == "manual_success"
        record = {
            "episode_index": len(self.results) + 1,
            "duration_sec": time.monotonic() - self._episode_started,
            "status": status,
            "completed": completed,
            "action_steps": self._steps,
        }
        self.results.append(record)
        self._write_results()
        info = {
            "eval_result": status,
            "manual_reset": not completed,
            "evaluation_complete": self.evaluation_complete,
        }
        self._boundary = (completed, not completed, info)
        return (
            copy.deepcopy(self._last_obs),
            float(completed),
            completed,
            not completed,
            info,
        )

    def step(self, action: Any) -> tuple:
        if self._boundary is not None:
            terminated, truncated, info = self._boundary
            return copy.deepcopy(self._last_obs), 0.0, terminated, truncated, info
        if self.evaluation_complete:
            return (
                copy.deepcopy(self._last_obs),
                0.0,
                False,
                False,
                {"evaluation_complete": True},
            )
        if self._last_obs is None:
            raise RuntimeError("Expected reset() before UR evaluation stepping.")
        for key, timestamp in self.listener.pop_pressed_events():
            if timestamp < self._accept_after:
                continue
            key = "Key.enter" if key == "Key.kpenter" else key
            if key not in {"Key.enter", "Key.space"}:
                self._first_press.clear()
                continue
            previous = self._first_press.get(key)
            self._first_press = {key: timestamp}
            if (
                previous is not None
                and 0 <= timestamp - previous <= self.double_press_interval
            ):
                return self._finish(
                    "manual_success" if key == "Key.space" else "keyboard_reset"
                )
        if time.monotonic() - self._episode_started >= self.episode_timeout_sec:
            return self._finish("timeout")
        self._last_obs, _, _, _, info = self.env.step(action)
        self._steps += 1
        return copy.deepcopy(self._last_obs), 0.0, False, False, info

    def close(self) -> None:
        self._stop.set()
        try:
            self._write_results()
        finally:
            super().close()

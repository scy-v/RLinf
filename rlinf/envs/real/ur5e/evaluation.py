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
from rlinf.utils.progress import ElapsedTime
from rlinf.utils.ur_logging import append_record, prepare_run_logs


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
        action_timing_enabled: bool = False,
        save_logs: bool = True,
        log_dir: str | None = None,
        stage2_enabled: bool = False,
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
        self._evaluation_closed = False
        self._last_obs = None
        self._boundary: tuple[bool, bool, dict] | None = None
        self._first_press: dict[str, float] = {}
        self._accept_after = 0.0
        self._episode_started = 0.0
        self._steps = 0
        self._waiting_for_start = False
        self._prepared = False
        self.action_timing_enabled = action_timing_enabled
        self.save_logs = save_logs
        self._run_log_paths = (
            prepare_run_logs(
                Path(log_dir) if log_dir else self.output_path.parent / "logs",
                self.output_path.parent.name,
            )
            if save_logs
            else None
        )
        self._episode_timer: ElapsedTime | None = None
        self._action_trace = None
        self._last_control_report = 0.0
        self._summary_written = False
        from rlinf.algorithms.rlt.inference import CriticalPhaseSwitch

        self.stage2_enabled = stage2_enabled
        self._critical_switch = CriticalPhaseSwitch()
        super().__init__(env)

    @property
    def critical_phase(self) -> bool:
        """Whether the next inference chunk should use the trained MLP."""
        return self.stage2_enabled and self._critical_switch.active

    @property
    def evaluation_complete(self) -> bool:
        """Whether the configured number of episodes has been recorded."""
        return len(self.results) >= self.num_episodes

    @property
    def waiting_for_start(self) -> bool:
        """Whether an initial or reset pose is awaiting operator confirmation."""
        return self._waiting_for_start

    def _write_results(self) -> None:
        if not self.save_logs:
            return
        if self._action_trace is not None:
            self._action_trace.flush()
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
        self._critical_switch.reset()
        self.base_env().stop_motion()
        self._record_control_timing(force=True)
        if self._episode_timer is not None:
            self._episode_timer.__exit__(None, None, None)
            self._episode_timer = None
        if self.evaluation_complete:
            # No new episode is started after the last outcome; home once before exit.
            if self._boundary is not None:
                self._last_obs, _ = self.env.reset(**kwargs)
                self._boundary = None
            return copy.deepcopy(self._last_obs), {"evaluation_complete": True}
        self.drain()
        if self._prepared:
            self._last_obs = self.env.get_wrapper_attr("observe")()
            info = {}
        else:
            self._last_obs, info = self.env.reset(**kwargs)
        self._prepared = bool(
            (kwargs.get("options") or {}).get("ur_eval_prepare", False)
        )
        self._boundary = None
        self._first_press.clear()
        self._steps = 0
        self._accept_after = time.time()
        self.drain()
        if self._prepared:
            self.log("UR warmup prepared with force control paused.")
            return copy.deepcopy(self._last_obs), info
        if self.wait_for_start:
            self._waiting_for_start = True
            try:
                self.log(
                    "UR reset complete. Force control paused; press Enter to start."
                )
                while not self._stop.is_set():
                    self._record_control_timing()
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
        self.base_env().start_motion()
        self.log("UR episode started; force control enabled.")
        if self.stage2_enabled:
            self.log(
                "Stage2: executing VLA actions. Double '.' within 0.5s to use MLP from the next chunk."
            )
        self._episode_started = time.monotonic()
        self._episode_timer = ElapsedTime(
            "TIME",
            f"Episode {len(self.results) + 1}",
            log=self._record_episode_progress,
        )
        self._episode_timer.__enter__()
        self._write_results()
        return copy.deepcopy(self._last_obs), info

    def _finish(self, status: str) -> tuple:
        self.base_env().stop_motion()
        self._record_control_timing(force=True)
        if self._episode_timer is not None:
            self._episode_timer.__exit__(None, None, None)
            self._episode_timer = None
        completed = status == "manual_success"
        record = {
            "episode_index": len(self.results) + 1,
            "duration_sec": time.monotonic() - self._episode_started,
            "status": status,
            "completed": completed,
            "action_steps": self._steps,
        }
        self.results.append(record)
        self.log(
            f"[EPISODE] {record['episode_index']}: {status}, "
            f"{record['duration_sec']:.1f}s, {self._steps} actions; "
            f"success {sum(r['completed'] for r in self.results)}/{len(self.results)}."
        )
        if self._run_log_paths is not None:
            append_record(self._run_log_paths[0], {"episode_result": record})
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
        if self._prepared:
            raise RuntimeError(
                "Expected evaluation start after UR warmup before stepping."
            )
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
            if key == "Key.focus_lost":
                self._first_press.clear()
                self._critical_switch.clear_pending_press()
                continue
            if timestamp < self._accept_after:
                continue
            key = "Key.enter" if key == "Key.kpenter" else key
            if self.stage2_enabled and key == "Key.dot":
                if self._critical_switch.press(timestamp):
                    self.log("Stage2: MLP selected for the next inference chunk.")
                continue
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
        self._record_control_timing()
        self._last_obs, _, _, _, info = self.env.step(action)
        self._steps += 1
        trace = info.pop("ur_action_trace", None)
        if trace is not None and self.save_logs:
            if self._action_trace is None:
                self.output_path.parent.mkdir(parents=True, exist_ok=True)
                self._action_trace = self.output_path.with_name(
                    "ur5e-actions.jsonl"
                ).open("a", encoding="utf-8")
            self._action_trace.write(
                json.dumps(
                    dict(
                        {
                            key: value
                            for key, value in trace.items()
                            if self.action_timing_enabled
                            or not key.endswith(("_sec", "_hz"))
                        },
                        episode_index=len(self.results) + 1,
                        step=self._steps,
                    )
                )
                + "\n"
            )
        if trace is not None and self.action_timing_enabled:
            if self._run_log_paths is not None:
                append_record(
                    self._run_log_paths[1],
                    {
                        "event": "action",
                        "component": "env",
                        **trace,
                        "episode_index": len(self.results) + 1,
                        "step": self._steps,
                    },
                )
        return copy.deepcopy(self._last_obs), 0.0, False, False, info

    def _record_control_timing(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_control_report < 1.0:
            return
        self._last_control_report = now
        timing = self.base_env().drain_control_timing()
        if timing is None:
            return
        if self._run_log_paths is not None:
            append_record(
                self._run_log_paths[1],
                {
                    "component": "controller",
                    "event": "force_mode",
                    "timestamp": time.time(),
                    **timing,
                },
            )

    def _record_episode_progress(self, message: str) -> None:
        """Keep periodic episode progress in the timing file."""
        if self._run_log_paths is not None:
            append_record(
                self._run_log_paths[1],
                {
                    "component": "env",
                    "event": "episode_progress",
                    "timestamp": time.time(),
                    "episode_index": len(self.results) + 1,
                    "message": message,
                },
            )

    def close(self) -> None:
        if self._evaluation_closed:
            return
        self._evaluation_closed = True
        self._stop.set()
        try:
            self.base_env().stop_motion()
            if self._episode_timer is not None:
                self._episode_timer.__exit__(None, None, None)
                self._episode_timer = None
            self._record_control_timing(force=True)
            self._write_results()
            if self._run_log_paths is not None and not self._summary_written:
                successes = sum(record["completed"] for record in self.results)
                append_record(
                    self._run_log_paths[0],
                    {
                        "episode_summary": {
                            "total_episodes": len(self.results),
                            "completed_episodes": successes,
                            "incomplete_episodes": len(self.results) - successes,
                            "success_rate": successes / len(self.results)
                            if self.results
                            else 0.0,
                            "episodes": self.results,
                        }
                    },
                )
                self._summary_written = True
        finally:
            try:
                if self._action_trace is not None:
                    self._action_trace.close()
                    self._action_trace = None
            finally:
                super().close()

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

"""Keyboard controls scoped to reset, rollout and result review."""

import threading
import time
from dataclasses import replace

from rlinf.utils.logging import get_logger

from .rollout_loop import ControlState
from .terminal import color_status

logger = get_logger()


class RolloutControls:
    """Recognize double presses only in the phase that owns their meaning."""

    DOUBLE_PRESS_SECONDS = 0.5

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = ControlState()
        self._phase = "idle"
        self._pressed = set()
        self._first_press = None
        self._listener = None
        self._reset_finished = threading.Event()

    def _enter_phase(self, phase: str) -> None:
        with self._lock:
            self._phase = phase
            self._state = ControlState(stop=self._state.stop)
            self._first_press = None
            # Keep held keys suppressed until release across phase boundaries.

    def idle(self) -> None:
        """Disable episode keys during saving, automatic reset and start prompts."""
        self._enter_phase("idle")

    def begin_reset(self) -> None:
        """Enable only the manual-reset completion key."""
        self._reset_finished.clear()
        self._enter_phase("reset")
        logger.info(color_status("[复位] 使用原运动键调整位置；右方向键进入自动复位。"))

    def reset_finished(self) -> bool:
        """Whether the operator requested the configured automatic reset."""
        return self._reset_finished.is_set()

    def begin_episode(self) -> None:
        """Enable rollout stop keys."""
        self._enter_phase("rollout")
        logger.info(
            color_status(
                "[采集] 按住运动键或夹爪键直接接管，松开恢复模型；0.5 秒内连按两次句号键结束当前轨迹；Esc 退出。"
            )
        )

    def begin_review(self) -> None:
        """Enable result labels after motion has stopped and before resetting."""
        self._enter_phase("review")
        logger.info(
            color_status(
                "[结果判定] 轨迹已停止。请在 0.5 秒内双击：空格=成功，F=失败，D=丢弃。"
            )
        )

    def snapshot(self) -> ControlState:
        """Read an immutable operator state."""
        with self._lock:
            return self._state

    def _double_press(self, key: str, now: float) -> bool:
        previous = self._first_press
        self._first_press = (key, now)
        if previous is not None and previous[0] == key:
            if 0 <= now - previous[1] <= self.DOUBLE_PRESS_SECONDS:
                self._first_press = None
                return True
        return False

    def handle(self, key: str, pressed: bool) -> None:
        """Consume physical key edges, ignoring held-key repeats."""
        key = key.lower()
        with self._lock:
            if not pressed:
                self._pressed.discard(key)
                return
            if key in self._pressed:
                return
            self._pressed.add(key)
            old = self._state
            if key == "esc":
                self._state = replace(old, stop=True)
                logger.info(color_status("[退出] 将丢弃当前未保存轨迹并释放设备。"))
            elif self._phase == "reset" and key == "right":
                self._reset_finished.set()
                logger.info(color_status("[复位] 已收到右方向键，准备自动复位。"))
            elif self._phase == "rollout":
                if key == "." and not old.finish_requested:
                    if self._double_press(key, time.monotonic()):
                        self._state = replace(old, finish_requested=True)
                        logger.info(
                            color_status(
                                "[采集] 已收到结束请求；当前 chunk 完成后进入结果判定。"
                            )
                        )
                else:
                    self._first_press = None
            elif self._phase == "review" and old.outcome is None and not old.discard:
                if key in ("space", "f", "d"):
                    if self._double_press(key, time.monotonic()):
                        self._state = replace(
                            old,
                            outcome=None if key == "d" else key == "space",
                            discard=key == "d",
                        )
                        label = {"space": "成功", "f": "失败", "d": "丢弃"}[key]
                        logger.info(
                            color_status("[结果判定] 已选择：%s。处理完成后进入复位。"),
                            label,
                        )
                else:
                    self._first_press = None

    def start(self) -> None:
        """Start the local keyboard listener."""
        from pynput import keyboard

        def key_name(key):
            return getattr(key, "name", None) or getattr(key, "char", "") or ""

        self._listener = keyboard.Listener(
            on_press=lambda key: self.handle(key_name(key), True),
            on_release=lambda key: self.handle(key_name(key), False),
        )
        self._listener.start()

    def close(self) -> None:
        """Stop and join the listener if it was started."""
        if self._listener is not None:
            self._listener.stop()
            self._listener.join(timeout=2)

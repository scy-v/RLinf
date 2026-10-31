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

"""VLA-Precision key mapping with per-action takeover."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ...actions import ActionKind
from ..base import Features, Observation
from .base import TeleopAction, TeleopDevice


@TeleopDevice.register("keyboard")
class Keyboard(TeleopDevice):
    """Drive one arm with the VLA-Precision key map in physical control units.

    Each action reads the held controls. Releasing them returns control to
    the policy for the next action.
    """

    PRODUCES = {"arm": ActionKind.CARTESIAN_DELTA, "end_effector": ActionKind.GRIPPER}
    NEEDS = ("policy_action", "keyboard_action_transform")
    HOLD_WINDOW = 0.0
    # Ordering preserves VLA-Precision's precedence for opposing held keys.
    KEY_MAPPING = {
        "z": (-1, 0.0),
        "s": (0, 1.0),
        "w": (0, -1.0),
        "d": (1, 1.0),
        "a": (1, -1.0),
        "q": (2, 1.0),
        "e": (2, -1.0),
        "r": (3, 1.0),
        "t": (3, -1.0),
        "g": (4, 1.0),
        "f": (4, -1.0),
        "b": (5, 1.0),
        "v": (5, -1.0),
        "l": ("button", 0),
        "o": ("button", 1),
        "Key.leftbrace": ("button", 2),
    }

    def __init__(
        self,
        step_size_pos: float = 0.048,
        step_size_rot: float = 0.05,
        step_size_pos_alt: float | None = None,
        step_size_rot_alt: float | None = None,
    ) -> None:
        self._step_sizes = (
            (step_size_pos, step_size_rot),
            (
                step_size_pos if step_size_pos_alt is None else step_size_pos_alt,
                step_size_rot if step_size_rot_alt is None else step_size_rot_alt,
            ),
        )
        if not np.isfinite(self._step_sizes).all() or np.min(self._step_sizes) <= 0:
            raise ValueError(
                "Expected finite positive keyboard translation/rotation steps."
            )
        self._step_mode = 0
        self._gripper = 1.0

    def _open(self) -> Any:
        from rlinf.envs.real.wrappers.episode.keyboard import KeyboardListener

        return KeyboardListener()

    @property
    def observation_features(self) -> Features:
        """Declare the held-key snapshot consumed by this device."""
        return {"pressed": {"shape": (len(self.KEY_MAPPING),), "dtype": "bool"}}

    def get_observation(self) -> Observation:
        """Read held controls and apply one step-mode toggle per Slash press."""
        for key in self._device.pop_pressed_keys():
            if key == "Key.slash":
                self._step_mode = 1 - self._step_mode
        held = self._device.get_pressed_keys()
        return {"pressed": np.array([key in held for key in self.KEY_MAPPING])}

    def on_reset(self, context: Mapping[str, Any]) -> None:
        """Restore the primary step mode and initially open gripper."""
        self._step_mode = 0
        self._gripper = 1.0
        self._device.pop_pressed_keys()

    def action(
        self, reading: Mapping[str, Any], context: Mapping[str, Any]
    ) -> TeleopAction:
        """Override the current policy action only while controls are held."""
        pose = np.zeros(6, dtype=np.float32)
        buttons = np.zeros(3, dtype=bool)
        position_step, rotation_step = self._step_sizes[self._step_mode]
        for pressed, mapping in zip(reading["pressed"], self.KEY_MAPPING.values()):
            if not pressed:
                continue
            axis, value = mapping
            if axis == "button":
                buttons[int(value)] = True
            elif axis != -1:
                pose[axis] = value * (rotation_step if axis >= 3 else position_step)
        close, open_gripper, stop = buttons
        pose_pressed = np.linalg.norm(pose) > self.MOVEMENT_EPSILON
        active = bool(pose_pressed or buttons.any())
        if not active:
            self._gripper = float(context["policy_action"][-1] > 0.5)
            return TeleopAction()
        if stop:
            pose[:] = 0
        if close:
            self._gripper = 0.0
        elif open_gripper:
            self._gripper = 1.0
        action = context["keyboard_action_transform"](np.r_[pose, self._gripper])
        info = {
            "pose_intervention_pressed": bool(pose_pressed),
            "gripper_intervention_pressed": bool(close or open_gripper),
            "stop_intervention_pressed": bool(stop),
            "close_gripper_pressed": bool(close),
            "open_gripper_pressed": bool(open_gripper),
        }
        info["keyboard_step_mode"] = self._step_mode
        return TeleopAction(
            parts={"arm": action[:6], "end_effector": action[6:]},
            driving=True,
            info=info,
        )

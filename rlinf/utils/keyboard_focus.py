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

"""Optional X11 terminal-window gating for physical keyboard events."""

import os
from pathlib import Path

from rlinf.utils.logging import get_logger

_WINDOW_ENV = "RLINF_KEYBOARD_TERMINAL_WINDOW"
FOCUS_LOST = "Key.focus_lost"


def validate_terminal_keyboard(value: bool) -> bool:
    """Validate the shared YAML switch without opening hardware or a display."""
    if not isinstance(value, bool):
        raise ValueError("keyboard_terminal_only must be a YAML boolean (true/false).")
    return value


def _display():
    if os.environ.get("XDG_SESSION_TYPE") == "wayland":
        raise RuntimeError(
            "keyboard_terminal_only=true requires an X11 desktop, not Wayland."
        )
    try:
        from Xlib.display import Display

        return Display()
    except Exception as exc:
        raise RuntimeError(
            "keyboard_terminal_only=true requires python-xlib and access to the launching X11 display."
        ) from exc


def _values(display, window, name):
    prop = window.get_full_property(display.intern_atom(name), 0)
    return [] if prop is None else list(prop.value)


def configure_terminal_keyboard(enabled: bool = False) -> None:
    """Resolve the launching terminal before Cluster propagates the environment.

    The window must belong to an ancestor process or be identified by WINDOWID.
    Never capture an arbitrary foreground window after model initialization.
    """
    validate_terminal_keyboard(enabled)
    os.environ[_WINDOW_ENV] = "0"
    if not enabled:
        return
    display = _display()
    try:
        root = display.screen().root
        explicit = os.environ.get("WINDOWID")
        if explicit:
            window_id = int(explicit, 0)
            display.create_resource_object("window", window_id).get_attributes()
        else:
            ancestors = set()
            pid = os.getpid()
            while pid > 1 and pid not in ancestors:
                ancestors.add(pid)
                status = Path(f"/proc/{pid}/status").read_text()
                pid = int(
                    next(
                        line.split()[1]
                        for line in status.splitlines()
                        if line.startswith("PPid:")
                    )
                )
            candidates = []
            for window_id in _values(display, root, "_NET_CLIENT_LIST"):
                window = display.create_resource_object("window", int(window_id))
                pids = _values(display, window, "_NET_WM_PID")
                if pids and int(pids[0]) in ancestors:
                    candidates.append(int(window_id))
            active = _values(display, root, "_NET_ACTIVE_WINDOW")
            if len(candidates) == 1:
                window_id = candidates[0]
            elif active and int(active[0]) in candidates:
                window_id = int(active[0])
            else:
                raise RuntimeError(
                    "Cannot uniquely identify the launching terminal window. "
                    "Run in a local X11 terminal with WINDOWID, or use keyboard_terminal_only=false."
                )
        os.environ[_WINDOW_ENV] = str(window_id)
        get_logger().info(
            "Keyboard controls require focus on launching X11 window %s.", window_id
        )
    finally:
        display.close()


class TerminalKeyboardFocus:
    """Own one X11 connection; default mode needs no display connection."""

    def __init__(self) -> None:
        self.window_id = int(os.environ.get(_WINDOW_ENV, "0"))
        self.display = _display() if self.window_id else None
        self._failed = False

    def accepts_input(self) -> bool:
        """Fail closed if the terminal is inactive, minimized, or disconnected."""
        if self.display is None:
            return True
        if self._failed:
            return False
        try:
            active = _values(
                self.display, self.display.screen().root, "_NET_ACTIVE_WINDOW"
            )
            return bool(active and int(active[0]) == self.window_id)
        except Exception:
            self._failed = True
            get_logger().error(
                "Keyboard focus query failed; keyboard controls disabled until restart."
            )
            return False

    def close(self) -> None:
        if self.display is not None:
            self.display.close()

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

"""Elapsed-time reporting for model loading and first-inference compilation."""

import shutil
import sys
import threading
import time
from collections.abc import Callable
from types import TracebackType


class ElapsedTime:
    """Report a blocking operation once per second and stop on success or failure.

    Args:
        label: Short phase name displayed in brackets.
        description: Operation being timed.
        log: Logger callback; when omitted, update stderr in place.
    """

    def __init__(
        self,
        label: str,
        description: str,
        log: Callable[[str], None] | None = None,
        *,
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled
        self.label = label
        self.description = description
        self.log = log
        self.duration = 0.0
        self._stopped = threading.Event() if enabled else None

    def _report(self, message: str, *, final: bool = False) -> None:
        message = f"[{self.label}] {message}"
        if self.log is not None:
            self.log(message)
        else:
            # Leave one column unused to avoid terminal automatic line wrapping.
            width = max(1, shutil.get_terminal_size().columns - 1)
            if len(message) > width:
                prefix = f"[{self.label}] "
                suffix = message.rsplit(": ", 1)[-1]
                message = (prefix + suffix)[-width:]
            padding = max(0, min(self._last_width, width) - len(message))
            self._stream.write("\r" + message + " " * padding + ("\n" if final else ""))
            self._stream.flush()
            self._last_width = len(message)

    def _update(self) -> None:
        while not self._stopped.wait(1.0):
            self._report(
                f"{self.description}: {time.monotonic() - self._started:.1f}s elapsed"
            )

    def __enter__(self) -> "ElapsedTime":
        if not self.enabled:
            return self
        self._stream = sys.stderr
        self._last_width = 0
        self._started = time.monotonic()
        self._report(f"{self.description}: 0.0s elapsed")
        self._thread = threading.Thread(
            target=self._update, name=f"ur-{self.label.lower()}-timer", daemon=True
        )
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if not self.enabled:
            return
        self._stopped.set()
        self._thread.join()
        self.duration = time.monotonic() - self._started
        status = "Completed" if exc_type is None else "Stopped before completion"
        self._report(
            f"{self.description}: {status} after {self.duration:.2f}s", final=True
        )

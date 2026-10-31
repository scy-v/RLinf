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

"""Process-exit cleanup for physical robots and their entry scripts."""

import atexit
import signal
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import FrameType

from rlinf.utils.logging import get_logger

_callbacks: dict[Callable[[], None], None] = {}
_registered = False
_previous_handlers = {}
_signal_registered = False


def _close_robots() -> None:
    for callback in reversed(tuple(_callbacks)):
        _callbacks.pop(callback, None)
        try:
            callback()
        except BaseException:
            get_logger().exception("Failed to release robot resources during exit.")


def _terminate(signum: int, frame: FrameType | None) -> None:
    _close_robots()
    previous = _previous_handlers.get(signum)
    if callable(previous):
        previous(signum, frame)
    raise SystemExit(128 + signum)


def install_robot_cleanup_handlers() -> None:
    """Install on the process main thread before worker-thread robot setup."""
    global _signal_registered
    if _signal_registered or threading.current_thread() is not threading.main_thread():
        return
    for sig in (signal.SIGINT, signal.SIGTERM):
        _previous_handlers[sig] = signal.getsignal(sig)
        signal.signal(sig, _terminate)
    _signal_registered = True


def register_robot_cleanup(callback: Callable[[], None]) -> None:
    """Keep a robot cleanup callback until explicit close or process exit."""
    global _registered
    _callbacks[callback] = None
    if not _registered:
        atexit.register(_close_robots)
        _registered = True
    install_robot_cleanup_handlers()


def unregister_robot_cleanup(callback: Callable[[], None]) -> None:
    """Remove a robot that has already been explicitly closed."""
    _callbacks.pop(callback, None)


@contextmanager
def termination_signals() -> Iterator[None]:
    """Run robot cleanup before the driver's worker-failure teardown."""
    signals = (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1)
    previous = {sig: signal.getsignal(sig) for sig in signals}

    def terminate(signum: int, frame: FrameType | None) -> None:
        # Cluster's SIGUSR1 handler kills workers before they can stop hardware.
        signal.signal(signal.SIGUSR1, signal.SIG_IGN)
        if signum == signal.SIGINT:
            raise KeyboardInterrupt
        raise SystemExit(128 + signum)

    for sig in signals:
        signal.signal(sig, terminate)
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)

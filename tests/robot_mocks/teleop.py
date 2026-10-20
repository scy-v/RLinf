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

"""Fake SpaceMouse, leader-arm, glove, VR, and keyboard interfaces.

Devices report an idle state by default. Tests can supply motion directly::

    with mocked_sdks() as made:
        made["pyspacemouse"].twist = (0.5, 0, 0, 0, 0, 0)
"""

from __future__ import annotations

import types
from typing import Any

import numpy as np

from ._fakes import module

DOF = 7


def spacemouse() -> types.ModuleType:
    """Return an idle ``pyspacemouse`` module with mutable input state."""
    fake = module("pyspacemouse")
    fake.twist = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    fake.buttons = (0, 0)

    class Device:
        def __init__(self):
            self.closed = False

        def read(self):
            if self.closed:
                raise OSError("read from a closed SpaceMouse")
            x, y, z, roll, pitch, yaw = fake.twist
            return types.SimpleNamespace(
                x=x,
                y=y,
                z=z,
                roll=roll,
                pitch=pitch,
                yaw=yaw,
                buttons=list(fake.buttons),
            )

        def close(self):
            self.closed = True

    # A fresh handle per open, as the real driver does, recorded so a test
    # can see what became of one after the part that opened it disconnected.
    fake.devices = []

    def _open(**_kwargs):
        device = Device()
        fake.devices.append(device)
        return device

    fake.open = _open
    return fake


def gello() -> dict[str, types.ModuleType]:
    """Return a ``gello_teleop`` module with a fixed leader-arm pose."""
    agent_module = module("gello_teleop.gello_teleop_agent")
    fk_module = module("gello_teleop.franka_fk")

    class GelloTeleopAgent:
        def __init__(self, port=None, **_kwargs):
            self.port = port
            self.closed = False
            agent_module.last_agent = self

        def get_action(self):
            return np.zeros(DOF), 0.0

        def close(self):
            self.closed = True

    class FrankaFK:
        def get_fk(self, _joints):
            return np.array([0.4, 0.0, 0.3]), np.array([0.0, 1.0, 0.0, 0.0])

    agent_module.GelloTeleopAgent = GelloTeleopAgent
    fk_module.FrankaFK = FrankaFK
    package = module("gello_teleop")
    package.gello_teleop_agent = agent_module
    package.franka_fk = fk_module
    return {
        "gello_teleop": package,
        "gello_teleop.gello_teleop_agent": agent_module,
        "gello_teleop.franka_fk": fk_module,
    }


def glove() -> dict[str, types.ModuleType]:
    """Return an ``rlinf_dexhand`` glove that reports open fingers."""

    class GloveExpert:
        def __init__(self, *_args, **_kwargs):
            self.angles = np.zeros(6)

        def get_angles(self):
            return self.angles

        def close(self):
            return None

        def stop(self):
            return None

    inner = module("rlinf_dexhand.glove", GloveExpert=GloveExpert)
    package = module("rlinf_dexhand")
    package.glove = inner
    return {"rlinf_dexhand": package, "rlinf_dexhand.glove": inner}


def pico_packet(
    buttons: dict[str, bool] | None = None,
    *,
    left: dict[str, Any] | None = None,
    right: dict[str, Any] | None = None,
) -> str:
    """Return one PICO publisher packet, both controllers at rest by default.

    The headset faces forward, so the transport can calibrate from it, and no
    grip is held, so no arm follows a controller. ``left`` and ``right``
    override fields of that controller, such as ``grip`` or ``position``.
    """
    import json

    controller = {
        "position": [0.0, 0.0, 0.0],
        "orientation": [0.0, 0.0, 0.0, 1.0],
        "grip": 0.0,
        "trigger": 0.0,
    }
    return json.dumps(
        {
            "headset_pose": [0.0, 1.6, 0.0, 0.0, 0.0, 0.0, 1.0],
            "left_controller": {**controller, **(left or {})},
            "right_controller": {**controller, **(right or {})},
            "buttons": dict(buttons or {}),
        }
    )


#: Subscribing here replays :func:`operator_packet` instead of the queue.
SCRIPTED_OPERATOR = "mock://pico-operator"


def operator_packet(elapsed: float) -> str:
    """Return what an operator's eight-second take publishes at ``elapsed``.

    Press record at 1 s, hold both grips from 2 s while drifting both
    controllers outward, close both grippers at 3 s and open them at 5 s, let
    go at 6 s and press record again to end the take.
    """
    phase = elapsed % 8.0
    offset = 0.003 * float(np.clip(phase - 2.0, 0.0, 2.0))
    grip = float(2.0 <= phase < 6.0)
    closing, opening = 3.0 <= phase < 3.4, 5.0 <= phase < 5.4
    return pico_packet(
        {
            "right_menu_button": 1.0 <= phase < 1.4 or 6.0 <= phase < 6.4,
            "X": closing,
            "A": closing,
            "Y": opening,
            "B": opening,
        },
        left={"position": [-(0.25 + offset), 1.25, -0.4], "grip": grip},
        right={"position": [0.25 + offset, 1.25, -0.4], "grip": grip},
    )


def vr() -> types.ModuleType:
    """Return a ``zmq`` module backed by a test-controlled packet queue.

    Queued ``packets`` are delivered first; then ``frame``, when set, is
    published over and over like a live controller. A socket connected to
    :data:`SCRIPTED_OPERATOR` instead receives an operator's take on repeat,
    so a collection run reaches its target unattended.
    """
    import time

    fake = module("zmq")
    fake.packets: list[Any] = []
    fake.frame: str | None = None
    fake.operator_started: float | None = None

    class Socket:
        def __init__(self, *_a, **_k):
            self.subscribed = []
            self.address = None
            self.hwm = None
            self.closed = False

        def connect(self, address):
            self.address = address
            if address == SCRIPTED_OPERATOR and fake.operator_started is None:
                fake.operator_started = time.monotonic()

        def set_hwm(self, value):
            self.hwm = value

        def setsockopt_string(self, *_a, **_k):
            return None

        def setsockopt(self, *_a, **_k):
            return None

        def close(self, linger=None):
            self.closed = True

        def recv_string(self, flags=0):
            if self.address == SCRIPTED_OPERATOR:
                # Paced like a publisher, so a polling thread does not spin.
                time.sleep(0.01)
                return operator_packet(time.monotonic() - fake.operator_started)
            if not fake.packets:
                if fake.frame is not None:
                    time.sleep(0.002)
                    return fake.frame
                raise fake.Again()
            packet = fake.packets.pop(0)
            return packet if isinstance(packet, str) else packet.decode("utf-8")

        def recv(self, flags=0):
            """Bytes, as pyzmq gives them; the publisher sends JSON."""
            packet = self.recv_string(flags)
            return packet.encode("utf-8")

    class Context:
        def __init__(self):
            self.sockets = []
            self.terminated = False

        def socket(self, _kind):
            sock = Socket()
            self.sockets.append(sock)
            fake.sockets.append(sock)
            return sock

        def term(self):
            self.terminated = True

    fake.sockets: list[Any] = []
    fake.Context = lambda: Context()
    fake.SUB = "SUB"
    fake.SUBSCRIBE = "SUBSCRIBE"
    fake.CONFLATE = "CONFLATE"
    fake.NOBLOCK = 1
    fake.RCVTIMEO = "RCVTIMEO"
    fake.Again = type("Again", (Exception,), {})
    fake.ZMQError = type("ZMQError", (Exception,), {})
    # pyzmq exposes the timeout exception as ``zmq.error.Again``, which is
    # what the transport catches to keep polling.
    fake.error = module("zmq.error", Again=fake.Again, ZMQError=fake.ZMQError)
    return fake


def keyboard() -> types.ModuleType:
    """Return an ``evdev`` module with one keyboard that types labels.

    ``RLINF_FAKE_KEYS`` names the keys it presses, cycling forever, so a run
    that waits for an operator can reach its target unattended. Left unset the
    device reports no presses, which is the shape a test wants when the
    keyboard is only incidental.
    """
    import os
    import queue
    import threading
    import time

    codes = types.SimpleNamespace(
        EV_KEY=1,
        KEY_A=30,
        KEY_B=48,
        KEY_C=46,
        KEY_Q=16,
        KEY_ENTER=28,
        KEY_KPENTER=96,
    )
    by_name = {
        name: getattr(codes, f"KEY_{name.upper()}")
        for name in ("a", "b", "c", "q", "enter", "kpenter")
    }
    codes.bytype = {
        codes.EV_KEY: {code: f"KEY_{name.upper()}" for name, code in by_name.items()}
    }
    devices = []

    class Device:
        def __init__(self, path: str) -> None:
            self.name = "fake keyboard"
            self.path = path
            self.closed = False
            self._read_fd, self._write_fd = os.pipe()
            self._batches = queue.Queue()
            self._stop = threading.Event()
            self._producer = None
            devices.append(self)
            keys = [
                name.strip().lower()
                for name in os.environ.get("RLINF_FAKE_KEYS", "").split(",")
                if name.strip().lower() in by_name
            ]
            if keys:
                self._producer = threading.Thread(
                    target=self._script, args=(keys,), daemon=True
                )
                self._producer.start()

        def capabilities(self, verbose: bool = False) -> dict[int, list[int]]:
            return {codes.EV_KEY: list(by_name.values())}

        def fileno(self) -> int:
            return self._read_fd

        def emit(self, *presses: tuple[str, int]) -> threading.Event:
            """Queue kernel-style edges; the returned event signals delivery."""
            delivered = threading.Event()
            events = []
            for name, value in presses:
                timestamp = time.time()
                events.append(
                    types.SimpleNamespace(
                        type=codes.EV_KEY,
                        code=by_name[name],
                        value=value,
                        timestamp=lambda timestamp=timestamp: timestamp,
                    )
                )
            self._batches.put((events, delivered))
            os.write(self._write_fd, b"k")
            return delivered

        def read(self):
            os.read(self._read_fd, 1)
            events, delivered = self._batches.get_nowait()
            yield from events
            delivered.set()

        def _script(self, keys):
            dwell = float(os.environ.get("RLINF_FAKE_KEY_DWELL", "0.3"))
            while not self._stop.is_set():
                for name in keys:
                    if self._stop.is_set():
                        return
                    self.emit((name, 1))
                    if self._stop.wait(dwell):
                        return
                    self.emit((name, 0))
                    if self._stop.wait(0.02):
                        return

        def close(self) -> None:
            if self.closed:
                return
            self._stop.set()
            if self._producer is not None:
                self._producer.join()
            os.close(self._read_fd)
            os.close(self._write_fd)
            self.closed = True

    return module(
        "evdev",
        InputDevice=Device,
        devices=devices,
        list_devices=lambda: ["/dev/input/event-fake"],
        ecodes=codes,
    )


def modules(**_: Any) -> dict[str, types.ModuleType]:
    """Return fake teleoperation SDKs keyed by import name."""
    zmq = vr()
    made: dict[str, types.ModuleType] = {
        "pyspacemouse": spacemouse(),
        "zmq": zmq,
        # The transport imports the submodule name to catch its timeout.
        "zmq.error": zmq.error,
        "evdev": keyboard(),
    }
    made.update(gello())
    made.update(glove())
    return made

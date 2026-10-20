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

"""RTDE and PGE vendor boundaries used by UR5e contract tests."""

import types
from typing import Any

from ._fakes import module


def modules() -> dict[str, types.ModuleType]:
    """Return stateful RTDE and serial fakes sharing each robot's pose."""
    states: dict[str, dict[str, Any]] = {}

    class Receive:
        def __init__(self, ip: str, *args):
            self.state = states.setdefault(
                ip,
                {
                    "pose": [0.4, 0.0, 0.3, 0.0, 0.0, 0.0],
                    "q": [0.0] * 6,
                    "moves": [],
                    "forces": [],
                    "events": [],
                },
            )
            self.connected = True

        def isConnected(self):
            return self.connected

        def getActualTCPPose(self):
            return self.state["pose"]

        def getActualTCPSpeed(self):
            return [0.0] * 6

        def getActualQ(self):
            return self.state["q"]

        def getActualQd(self):
            return [0.0] * 6

        def getActualTCPForce(self):
            return [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

        def disconnect(self):
            self.connected = False

    class Control(Receive):
        fail_move = False
        fail_force = False
        safe = True

        def isPoseWithinSafetyLimits(self, pose):
            return self.safe

        def forceModeSetGainScaling(self, gain):
            self.state["gain"] = gain
            return True

        def setPayload(self, mass, cog):
            self.state["payload"] = (mass, list(cog))
            return True

        def forceMode(self, frame, selection, wrench, mode, limits):
            self.state["forces"].append(
                (list(frame), list(selection), list(wrench), mode, list(limits))
            )
            return not self.fail_force

        def forceModeStop(self):
            self.state["events"].append("forceModeStop")
            return True

        def moveL(self, pose, speed, acceleration):
            self.state["moves"].append((list(pose), speed, acceleration))
            if self.fail_move:
                return False
            self.state["pose"] = list(pose)
            return True

        def moveJ(self, joints, speed, acceleration):
            self.state["events"].append("moveJ")
            self.state["q"] = list(joints)
            return True

        def stopScript(self):
            self.state["events"].append("stopScript")
            return True

    class PGE:
        instances = []

        def __init__(self, port):
            self.port = port
            self.raw = 1000
            self.closed = False
            self.ser = types.SimpleNamespace(close=self.close_serial)
            self.commands = []
            type(self).instances.append(self)

        def close_serial(self):
            self.closed = True

        def init_feedback(self):
            return True

        def set_force(self, force):
            self.force = force

        def set_vel(self, speed):
            self.speed = speed

        def read_pos(self):
            return self.raw

        def set_pos(self, val, blocking=True):
            self.write_uart(modbus_high_addr=0x01, modbus_low_addr=0x03, val=val)

        def write_uart(self, **command):
            self.commands.append(command)
            self.raw = command["val"]

    return {
        "rtde_control": module(
            "rtde_control", RTDEControlInterface=Control, states=states
        ),
        "rtde_receive": module("rtde_receive", RTDEReceiveInterface=Receive),
        "pyDHgripper": module("pyDHgripper", PGE=PGE),
    }

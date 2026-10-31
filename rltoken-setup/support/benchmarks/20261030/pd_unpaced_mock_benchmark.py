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

"""Measure controller software throughput using fake RTDE only."""

import json
import threading
import time
from pathlib import Path

from omegaconf import OmegaConf
from robot_mocks import mocked_sdks

from rlinf.robotics.parts.arms.ur5e import UR5eArm


class UnpacedEvent:
    def __init__(self):
        self.event = threading.Event()

    def set(self):
        self.event.set()

    def clear(self):
        self.event.clear()

    def is_set(self):
        return self.event.is_set()

    def wait(self, timeout=None):
        return self.event.is_set()


cfg = OmegaConf.load(
    "logs/online_rltoken/20261030-181859-ur_rlt_stage2_ac_mlp/tensorboard/config.yaml"
)
settings = OmegaConf.to_container(
    cfg.cluster.node_groups[0].hardware.configs[0].impedance, resolve=True
)
settings["start_paused"] = True
out = {
    "hardware": "CPU; vendor RTDE interfaces replaced by fake SDK; no physical device",
    "frequency_parameter": settings["frequency"],
    "method": "Run the unchanged UR5eArm._run_controller; skip only its final Event.wait in unpaced case. Preserve frequency=100 because wrench computation also uses frequency. Fake RTDE calls do not model real network/device delay.",
    "cases": [],
}
for paced, duration, repeats in [(True, 1.5, 1), (False, 1.0, 3)]:
    for rep in range(repeats):
        with mocked_sdks():
            arm = UR5eArm("192.0.2.123", impedance=settings)
            arm.connect()
            if not paced:
                arm._stop_event = UnpacedEvent()
            try:
                arm.start_motion()
                time.sleep(duration)
                arm.stop_motion()
                s = arm.drain_control_timing()
                assert arm._loop_error is None
                out["cases"].append(
                    {
                        "paced": paced,
                        "repetition": rep + 1,
                        "cycles": s["samples"],
                        "measured_hz": s["measured_hz"],
                        "work_mean_ms": s["cycle_work_sec_mean"] * 1000,
                        "fake_forceMode_mean_ms": s["force_mode_sec_mean"] * 1000,
                    }
                )
            finally:
                arm.disconnect()
root = Path("rltoken-setup/support/benchmarks/20261030")
(root / "pd_unpaced_mock_benchmark.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))

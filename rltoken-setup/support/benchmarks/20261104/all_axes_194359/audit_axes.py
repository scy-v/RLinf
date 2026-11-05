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

import json
from pathlib import Path

import numpy as np
import torch

O = Path("rltoken-setup/support/benchmarks/20261104/all_axes_194359")
D = {}
for name, run, step, warm in [
    ("new", "20261104-203520-ur_rlt_stage2_ac_mlp", 2522, 151),
    ("old", "20261103-224754-ur_rlt_stage2_ac_mlp", 3276, 150),
]:
    root = (
        Path("logs/online_rltoken")
        / run
        / f"ur_rlt_stage2_ac_mlp/checkpoints/global_step_{step}/actor/sac_components/replay_buffer/rank_0"
    )
    files = sorted(
        root.glob("trajectory_*.pt"), key=lambda p: int(p.name.split("_")[1])
    )
    rows = [torch.load(p, map_location="cpu", weights_only=False) for p in files]
    props = np.stack([r["curr_obs"]["proprio"].reshape(-1).numpy() for r in rows])
    flags = np.array([bool(r["intervene_flags"].any()) for r in rows])
    done = np.array([bool(r["dones"].any()) for r in rows])
    n = len(rows)
    before = np.ones(n, dtype=bool)
    seen = False
    for i, r in enumerate(rows):
        if flags[i]:
            seen = True
        before[i] = not seen
        if done[i]:
            seen = False
    d = {}
    for label, mask in [
        ("warmup", np.arange(n) < warm),
        ("online_nonhuman_chunk", (np.arange(n) >= warm) & ~flags),
        ("online_pre_first_intervention", (np.arange(n) >= warm) & before),
        ("online_human_chunk", (np.arange(n) >= warm) & flags),
    ]:
        p = props[mask]
        d[label] = {
            "n": len(p),
            "rp_rms_deg": np.rad2deg(np.sqrt((p[:, 3:5] ** 2).mean(0))).tolist(),
            "rp_abs_p95_deg": np.rad2deg(
                np.quantile(abs(p[:, 3:5]), 0.95, axis=0)
            ).tolist(),
            "angular_velocity_rms_rad_sec": np.sqrt((p[:, 9:12] ** 2).mean(0)).tolist(),
        }
    D[name] = d
(O / "axis_audit.json").write_text(json.dumps(D, indent=2) + "\n")
print(json.dumps(D, indent=2))

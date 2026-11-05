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
from omegaconf import OmegaConf

from rlinf.models.embodiment.mlp_policy import get_model

torch.set_num_threads(2)
O = Path("rltoken-setup/support/benchmarks/20261104/all_axes_194359")
r = Path("logs/online_rltoken/20261104-203520-ur_rlt_stage2_ac_mlp")
cp = r / "ur_rlt_stage2_ac_mlp/checkpoints/global_step_2522/actor"
p = json.loads((O / "probe.json").read_text())
fs = sorted(
    (cp / "sac_components/replay_buffer/rank_0").glob("trajectory_*.pt"),
    key=lambda p: int(p.name.split("_")[1]),
)
rows = [torch.load(f, weights_only=False, map_location="cpu") for f in fs]
obs = {
    k: torch.cat([x["curr_obs"][k].reshape(1, -1) for x in rows])
    for k in ["z_rl", "proprio", "ref_chunk"]
}
labels = np.full(len(rows), "warmup", dtype="<U20")
cfg = OmegaConf.load(r / "tensorboard/config.yaml")
h = get_model(cfg.actor.model, torch_dtype=torch.float32).eval()
h.load_state_dict(
    torch.load(
        cp / "model_state_dict/full_weights.pt", map_location="cpu", weights_only=True
    )
)
for e in p["episodes"][7:]:
    j = e["last_id"]
    rp = int(rows[j]["rewards"].reshape(-1).gt(0).nonzero()[0]) if e["success"] else 10
    for i in range(e["first_id"], j + 1):
        labels[i] = (
            "failure"
            if not e["success"]
            else "success_near30"
            if (j - i) * 10 + rp < 30
            else "success_far"
        )
D = {}
with torch.no_grad():
    for n in [909, 2658]:
        q = np.zeros(n)
        for k in range(0, n, 256):
            ob = {key: v[k : min(k + 256, n)] for key, v in obs.items()}
            for seed in range(4):
                torch.manual_seed(seed * 10000 + k)
                a = h.sac_forward(
                    ob, apply_reference_dropout=True, reference_dropout_prob=0.5
                )[0]
                q[k : min(k + 256, n)] += h.sac_q_forward(ob, a)[:, 0].numpy() / 4
        D[n] = {
            "mean": float(q.mean()),
            "groups": {
                str(g): {
                    "n": int((labels[:n] == g).sum()),
                    "weight": float((labels[:n] == g).mean()),
                    "q_mean": float(q[labels[:n] == g].mean()),
                }
                for g in np.unique(labels)
            },
        }
D["decomposition"] = {}
for g in np.unique(labels):
    a = D[909]["groups"][g]
    b = D[2658]["groups"][g]
    D["decomposition"][str(g)] = {
        "composition": (b["weight"] - a["weight"]) * (b["q_mean"] + a["q_mean"]) / 2,
        "within_group": (b["q_mean"] - a["q_mean"]) * (b["weight"] + a["weight"]) / 2,
    }
(O / "q_data_mix.json").write_text(json.dumps(D, indent=2) + "\n")
print(json.dumps(D, indent=2))

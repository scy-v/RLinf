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
OLD = Path("rltoken-setup/support/benchmarks/20261103/budget_224754/final")
paths = {
    "old": Path("logs/online_rltoken/20261103-224754-ur_rlt_stage2_ac_mlp"),
    "new": Path("logs/online_rltoken/20261104-203520-ur_rlt_stage2_ac_mlp"),
}
steps = {"old": 3276, "new": 2522}
heads = {}
batches = {}
D = {}
for tag, r in paths.items():
    cp = r / f"ur_rlt_stage2_ac_mlp/checkpoints/global_step_{steps[tag]}/actor"
    cfg = OmegaConf.load(r / "tensorboard/config.yaml")
    h = get_model(cfg.actor.model, torch_dtype=torch.float32).eval()
    h.load_state_dict(
        torch.load(
            cp / "model_state_dict/full_weights.pt",
            map_location="cpu",
            weights_only=True,
        )
    )
    heads[tag] = h
    ids = json.loads(((OLD if tag == "old" else O) / "probe.json").read_text())[
        "fixed_sample_ids"
    ]
    fs = {
        int(p.name.split("_")[1]): p
        for p in (cp / "sac_components/replay_buffer/rank_0").glob("trajectory_*.pt")
    }
    rows = [torch.load(fs[i], map_location="cpu", weights_only=False) for i in ids]
    batches[tag] = {
        k: torch.cat([v["curr_obs"][k].reshape(1, -1) for v in rows])
        for k in ["z_rl", "proprio", "ref_chunk"]
    }
for data, b in batches.items():
    for actor, h in heads.items():
        pars = [p for n, p in h.named_parameters() if not n.startswith("q_head.")]
        ref = b["ref_chunk"].reshape(192, 20, 7)[:, :10].reshape(192, 70)
        for critic, c in heads.items():
            vals = []
            for seed in range(4):
                torch.manual_seed(seed)
                a = h.sac_forward(
                    b, apply_reference_dropout=True, reference_dropout_prob=0.5
                )[0]
                bcl = 5 * (a - ref).square().mean()
                ql = -0.1 * c.sac_q_forward(b, a)[:, 0].mean()
                gb = torch.autograd.grad(
                    bcl, pars, retain_graph=True, allow_unused=True
                )
                gq = torch.autograd.grad(ql, pars, allow_unused=True)
                nb = sum(x.square().sum() for x in gb if x is not None).sqrt()
                nq = sum(x.square().sum() for x in gq if x is not None).sqrt()
                vals.append([float(nb), float(nq), float(nb / nq)])
            D[f"data={data},actor={actor},critic={critic}"] = {
                "BC_grad": np.mean(vals, 0)[0],
                "Q_grad": np.mean(vals, 0)[1],
                "ratio": np.mean(vals, 0)[2],
            }
(O / "gradient_cross.json").write_text(json.dumps(D, indent=2) + "\n")
print(json.dumps(D, indent=2))

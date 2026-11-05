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

"""Render compact metrics and fixed-input probes without duplicating raw data."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

p = Path(__file__).resolve().parent
a = np.load(p / "metrics.npz")
r = json.loads((p / "probe.json").read_text())
x = a["steps"]
mask = x > 15
fig, axs = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
for key, label in [("train/actor/q_pi", "q_pi"), ("train/critic/q_data", "q_data")]:
    axs[0, 0].plot(x[mask], a[key][mask], label=label, lw=1.5)
axs[0, 0].set_title("Online Q metrics")
axs[0, 0].legend()
for key, label in [
    ("train/actor/bc_loss", "BC total"),
    ("train/actor/bc_ref_loss", "BC reference"),
    ("train/actor/bc_human_loss", "BC human"),
]:
    axs[0, 1].plot(x[mask], a[key][mask], label=label, lw=1.2)
axs[0, 1].set_yscale("log")
axs[0, 1].set_title("BC errors (batch averages)")
axs[0, 1].legend()
axs[1, 0].plot(x[mask], a["train/sac/critic_loss"][mask], color="tab:green")
axs[1, 0].set_yscale("log")
axs[1, 0].set_title("Critic TD loss (batch averages)")
s = np.array([int(k) for k in r["fixed_inputs"]])
g = np.array([v["weighted_grad_Q_BC_ratio_cos"] for v in r["fixed_inputs"].values()])
axs[1, 1].plot(s, g[:, 2], marker="o", color="tab:purple")
axs[1, 1].axhline(1, color="grey", ls=":")
axs[1, 1].set_title("Weighted BC / Q gradient norm, same 192 inputs")
axs[1, 1].set_ylabel("Ratio")
for ax in axs.flat:
    ax.axvline(773, color="grey", ls="--", lw=1)
    ax.set_xlabel("Completed network rounds")
    ax.grid(alpha=0.2)
fig.suptitle(
    "2026-11-04: UTD=10, critic:actor=4:1, all axes enabled\nDashed line: resume checkpoint 773; final saved checkpoint: 2522"
)
fig.savefig(p / "training_analysis.png", dpi=140)
plt.close(fig)

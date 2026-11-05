# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Export four presentation figures from the original scalar log, without editing it."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import ScalarFormatter

OUT = Path(__file__).resolve().parent / "ppt_losses"
ROOT = Path(__file__).resolve().parents[5]
SOURCE = ROOT / "logs/online_rltoken/20261104-230711-ur_rlt_stage2_ac_mlp/metrics.log"
OUT.mkdir(exist_ok=True)
rows = [json.loads(line) for line in SOURCE.read_text().splitlines() if line.strip()]
# Match the success figure's first warmup-update start, rather than process launch.
start = 1793804892.0  # 2026-11-04 23:08:12 Asia/Shanghai
x = np.array([(r["timestamp"] - start) / 60 for r in rows])
steps = np.array([r["step"] for r in rows])
groups = steps // 250
unique = np.unique(groups)
smooth_x = np.array([x[groups == g].mean() for g in unique])
font = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams.update(
    {
        "font.family": font.get_name(),
        "axes.unicode_minus": False,
        "font.size": 16,
        "axes.titlesize": 22,
        "axes.labelsize": 18,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "pdf.fonttype": 42,
    }
)
specs = [
    (
        "negative_q",
        "−Q（−q_pi）",
        "−q_pi（未乘Q权重）",
        "train/actor/q_pi",
        -1,
        "#2166ac",
        False,
    ),
    (
        "bc_loss",
        "BC loss",
        "BC loss（未乘BC权重）",
        "train/actor/bc_loss",
        1,
        "#b36500",
        True,
    ),
    (
        "actor_loss",
        "Actor loss",
        "Actor loss",
        "train/sac/actor_loss",
        1,
        "#7a4c9f",
        False,
    ),
    (
        "critic_loss",
        "Critic loss",
        "Critic TD loss",
        "train/sac/critic_loss",
        1,
        "#16816a",
        True,
    ),
]
for name, title, ylabel, key, sign, color, logscale in specs:
    y = np.array([r["metrics"][key] * sign for r in rows])
    assert np.isfinite(y).all()
    smooth_y = np.array([y[groups == g].mean() for g in unique])
    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.plot(smooth_x, smooth_y, color=color, linewidth=3, label="每250训练轮均值")
    ax.set(
        title=title,
        xlabel="训练开始后经过的时间（分钟）",
        ylabel=ylabel,
        xlim=(0, max(x) + 0.5),
    )
    ax.set_xticks(np.arange(0, max(x) + 1, 10))
    if logscale:
        assert (y > 0).all()
        ax.set_yscale("log")
        ax.set_ylabel(ylabel + "（对数轴）")
    else:
        formatter = ScalarFormatter(useOffset=False)
        formatter.set_scientific(False)
        ax.yaxis.set_major_formatter(formatter)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(fontsize=13, loc="best", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=220)
    fig.savefig(OUT / f"{name}.pdf", dpi=220)
    plt.close(fig)
manifest = {
    "source": str(SOURCE),
    "sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    "records": len(rows),
    "first_step": int(steps[0]),
    "last_step": int(steps[-1]),
    "time_origin": "2026-11-04 23:08:12 Asia/Shanghai, first warmup network update",
    "smoothing": "Arithmetic means in floor(runner_step/250) bins; not TensorBoard EMA. Figures show bin means only; all original scalar records remain in the unchanged source log.",
    "metrics": [
        {"file": n, "key": k, "multiplier": s, "log_y": l}
        for n, t, y, k, s, c, l in specs
    ],
    "weights": "Online actor_loss = 5*bc_loss - 0.1*q_pi; plotted BC and -Q are unweighted.",
}
manifest["artifact_bytes"] = sum(
    p.stat().st_size for p in OUT.iterdir() if p.suffix in {".png", ".pdf"}
)
(OUT / "manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
)
print(json.dumps(manifest, ensure_ascii=False, indent=2))

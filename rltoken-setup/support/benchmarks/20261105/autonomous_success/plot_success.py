# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Recompute plots from this run's episode summaries and checkpoint save events."""

import csv
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

ROOT = Path(__file__).resolve().parents[5]
SOURCE = ROOT / "logs/online_rltoken/20261104-230711-ur_rlt_stage2_ac_mlp"
OUT = Path(__file__).resolve().parent
TZ = ZoneInfo("Asia/Shanghai")
font = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams["font.family"] = font.get_name()
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update(
    {
        "font.size": 16,
        "axes.titlesize": 21,
        "axes.labelsize": 18,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "pdf.fonttype": 42,
    }
)
lines = (SOURCE / "run_embodiment.log").read_text().splitlines()
ends, checkpoints = [], []
training_start = None
day = datetime(2026, 11, 4, tzinfo=TZ)
previous = None
step = 0
for line in lines:
    match = re.search(r"INFO (\d\d:\d\d:\d\d)", line)
    if not match:
        continue
    h, m, s = map(int, match[1].split(":"))
    timestamp = day.replace(hour=h, minute=m, second=s)
    if previous and previous - timestamp > timedelta(hours=12):
        day += timedelta(days=1)
        timestamp += timedelta(days=1)
    previous = timestamp
    if training_start is None and "[TRAIN] Starting Warmup updates:" in line:
        training_start = timestamp
    progress = re.search(r"Online RL runner rounds: (\d+)/", line)
    if progress:
        step = int(progress[1])
    save = re.search(r"Saving checkpoint at step (\d+)", line)
    if save:
        checkpoints.append({"step": int(save[1]), "time": timestamp})
    end = re.search(r"\[EPISODE\] Stopped after (\d+) actions: reward=([\d.]+)", line)
    if end:
        ends.append(
            {
                "episode": len(ends) + 1,
                "time": timestamp,
                "actions": int(end[1]),
                "reward": float(end[2]),
                "approximate_runner_step": step,
            }
        )
rows = []
for line in (SOURCE / "metrics.log").open():
    try:
        metrics = json.loads(line)["metrics"]
    except (ValueError, KeyError):
        continue
    if "env/success_no_intervened" not in metrics or len(rows) >= len(ends):
        continue
    row = ends[len(rows)].copy()
    assert bool(row["reward"]) == bool(metrics["env/success_once"])
    assert metrics["env/episode_len"] == ((row["actions"] + 9) // 10) * 10
    tail = (-row["actions"]) % 10
    hold = metrics.get("env/terminal_hold_steps", 0)
    interventions = metrics["env/intervened_steps"] - max(0, tail - hold)
    assert interventions >= 0
    row.update(
        raw_intervened_steps=metrics["env/intervened_steps"],
        terminal_hold_steps=hold,
        corrected_intervened_steps=interventions,
        autonomous=int(row["reward"] > 0 and interventions == 0),
    )
    row["outcome"] = (
        "自主成功"
        if row["autonomous"]
        else ("干预后成功" if row["reward"] else "最终失败")
    )
    row["rolling20"] = None
    rows.append(row)
    if len(rows) >= 20:
        row["rolling20"] = sum(r["autonomous"] for r in rows[-20:]) / 20
windows = [r for r in rows if r["rolling20"] is not None]
peak = max(r["rolling20"] for r in windows)
peaks = [r for r in windows if r["rolling20"] == peak]

assert training_start is not None, "Expected first network-update start in the run log."


def elapsed(timestamp):
    return (timestamp - training_start).total_seconds() / 60


fig, ax = plt.subplots(figsize=(8.5, 6))
xs = [elapsed(r["time"]) for r in windows]
ax.axvspan(0, xs[0], color="#eeeeee")
ax.text(
    xs[0] / 2,
    47,
    "未满20轮\n不计算成功率",
    ha="center",
    va="center",
    color="#666666",
    fontsize=14,
)
ax.plot(
    xs,
    [r["rolling20"] * 100 for r in windows],
    color="#2166ac",
    marker="o",
    markersize=4.5,
    linewidth=3,
)
for r in peaks:
    ax.scatter(elapsed(r["time"]), peak * 100, color="#d95f02", s=85, zorder=5)
ax.annotate(
    f"峰值 {peak:.0%}（16/20）",
    (elapsed(peaks[0]["time"]), peak * 100),
    xytext=(27, 96),
    textcoords="data",
    ha="center",
    fontsize=18,
    color="#b94c00",
    fontweight="bold",
    arrowprops={"arrowstyle": "->", "color": "#b94c00", "lw": 1.8},
)
for c in checkpoints:
    if c["step"] in (15000, 17500, 20000):
        x = elapsed(c["time"])
        ax.axvline(x, color="#888888", alpha=0.5, linestyle="--", linewidth=1.5)
        ax.text(
            x,
            5,
            f"ckpt {c['step']}",
            rotation=90,
            va="bottom",
            ha="right",
            color="#555555",
            fontsize=13,
        )
ax.set(
    title="最近20轮自主成功率",
    xlabel="训练开始后经过的时间（分钟）",
    ylabel="自主成功率（%）",
    ylim=(-3, 105),
    xlim=(0, elapsed(rows[-1]["time"]) + 1),
)
ax.set_xticks(range(0, int(elapsed(rows[-1]["time"])) + 2, 10))
ax.set_yticks(range(0, 101, 20))
ax.grid(axis="y", alpha=0.2)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"rolling20_success.{ext}", dpi=220)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8.5, 5.6))
for label, color, marker in [
    ("自主成功", "#218c61", "o"),
    ("干预后成功", "#e69f00", "x"),
    ("最终失败", "#c0392b", "x"),
]:
    subset = [r for r in rows if r["outcome"] == label]
    ax.scatter(
        [r["episode"] for r in subset],
        [r["autonomous"] for r in subset],
        label=label,
        c=color,
        marker=marker,
        s=45,
        linewidths=2,
    )
ax.set(
    title="逐轮自主完成结果",
    xlabel="测试轮次（episode）",
    ylabel="自主完成结果",
    ylim=(-0.25, 1.45),
    xlim=(0, len(rows) + 1),
)
ax.set_yticks([0, 1], ["失败", "成功"])
ax.set_xticks(list(range(0, len(rows) + 1, 10)))
ax.grid(alpha=0.2)
ax.legend(loc="upper center", ncol=3, fontsize=13, columnspacing=0.8, handletextpad=0.3)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"episode_outcomes.{ext}", dpi=220)
plt.close(fig)
with (OUT / "episodes.csv").open("w") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
summary = {
    "training_start": str(training_start),
    "time_axis": "Minutes since first network update (including warmup), not wall-clock time",
    "source": str(SOURCE),
    "cutoff": str(rows[-1]["time"]),
    "episodes": len(rows),
    "unmatched_completed_episodes": len(ends) - len(rows),
    "peak_rate": peak,
    "peak_windows": [
        {
            "first_episode": r["episode"] - 19,
            "last_episode": r["episode"],
            "end_time": str(r["time"]),
            "approximate_runner_step": r["approximate_runner_step"],
            "latest_saved_checkpoint": max(
                c["step"] for c in checkpoints if c["time"] <= r["time"]
            ),
        }
        for r in peaks
    ],
    "checkpoints": [{"step": c["step"], "time": str(c["time"])} for c in checkpoints],
    "latest20": rows[-1]["rolling20"],
    "method": "Episode summaries matched in order with reward and rounded chunk length assertions. Corrected intervention = intervened_steps - max(0, (-actions)%10 - terminal_hold_steps). Window time is last episode end, not a frozen-policy evaluation.",
}
(OUT / "summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
)
print(json.dumps(summary, ensure_ascii=False, indent=2))
for r in rows:
    if r["episode"] in [55, 59, 62, 66, 70, 74, 78, 80, 87, 93]:
        print(r["episode"], r["time"], r["approximate_runner_step"], r["rolling20"])

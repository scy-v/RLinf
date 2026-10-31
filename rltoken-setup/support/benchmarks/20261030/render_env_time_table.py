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

"""Render false-mode Env timings with explicit module hierarchy."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle

root = Path(__file__).parent
source = json.loads((root / "true_false_20hz_classified_comparison.json").read_text())
data = source["false"]["workers"]["with_learner"]["Env step"]
regular = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
bold = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
fig = plt.figure(figsize=(24, 16.6), dpi=100, facecolor="white")
ax = fig.add_axes((0, 0, 1, 1))
ax.set_xlim(0, 2400)
ax.set_ylim(1660, 0)
ax.axis("off")
main, detail, border = "#714354", "#70696a", "#e2dadd"
accent = "#8b4d35"


def text(x, y, value, size=24, color=main, strong=False, ha="left"):
    ax.text(
        x,
        y,
        value,
        fontsize=size,
        color=color,
        fontproperties=bold if strong else regular,
        ha=ha,
        va="center",
    )


text(1200, 66, "Rlinf-RLT的Env时间", size=38, strong=True, ha="center")
xs = [60, 1140, 1540, 1940, 2340]
top, header_h, row_h = 146, 96, 88
for i, label in enumerate(["模块（ms）", "平均值", "P95", "最大值"]):
    ax.add_patch(
        Rectangle(
            (xs[i], top),
            xs[i + 1] - xs[i],
            header_h,
            facecolor=main,
            edgecolor=border,
            linewidth=1.2,
        )
    )
    text(
        (xs[i] + xs[i + 1]) / 2,
        top + header_h / 2,
        label,
        size=26,
        color="white",
        ha="center",
    )
rows = [
    ("1  输入动作处理", "input_ms", "main"),
    ("2  执行动作Chunk", "execute_ms", "main"),
    ("2.1  动作格式准备", "action_prepare_ms", "detail"),
    ("2.2  chunk_step执行（含封装层）", "action_execution_ms", "detail"),
    ("2.2.1  动作处理与下发（3动作合计）", "command_ms", "detail"),
    ("2.2.2  节拍等待（3动作合计）", "pace_wait_ms", "detail"),
    ("2.2.3  观测与反馈处理（3动作合计）", "observation_ms", "detail"),
    ("2.3  Chunk结果整理", "chunk_result_ms", "detail"),
    ("3  发布观测与反馈", "publish_ms", "main"),
    ("3.1  请求信息整理", "request_ms", "detail"),
    ("3.2  Transition整理", "transition_ms", "detail"),
    ("3.3  发送数据", "send_ms", "detail"),
    ("Env有效工作合计", "work_ms", "total"),
]
for i, (label, key, kind) in enumerate(rows):
    y = top + header_h + i * row_h
    fill = (
        "#faf8f8" if kind == "detail" else ("#f5e9e2" if kind == "total" else "#f2eaf0")
    )
    color = detail if kind == "detail" else (accent if kind == "total" else main)
    if key == "execute_ms":
        fill, color = "#e4f0ec", "#386b5c"
    for c in range(4):
        ax.add_patch(
            Rectangle(
                (xs[c], y),
                xs[c + 1] - xs[c],
                row_h,
                facecolor=fill,
                edgecolor=border,
                linewidth=1.0,
            )
        )
    indent = 140 if label.startswith("2.2.") else (84 if kind == "detail" else 34)
    text(
        xs[0] + indent,
        y + row_h / 2,
        label,
        size=26,
        color=color,
        strong=kind != "detail",
    )
    for c, stat in enumerate(["mean", "p95", "max"], start=1):
        text(
            (xs[c] + xs[c + 1]) / 2,
            y + row_h / 2,
            format(data[key][stat], ".3f" if data[key]["mean"] < 0.1 else ".2f"),
            size=28,
            color=color,
            strong=kind != "detail",
            ha="center",
        )
# Cadence is a separate diagnostic, not an additive duration component.
cadence = source["false"]["target_cadence"]["with_learner"]["target_interval_ms"]
cadence_top = top + header_h + len(rows) * row_h + 32
for c, label in enumerate(
    ["频率指标（ms，不计入耗时合计）", "平均值", "P95", "最大值"]
):
    ax.add_patch(
        Rectangle(
            (xs[c], cadence_top),
            xs[c + 1] - xs[c],
            78,
            facecolor="#386b5c",
            edgecolor=border,
            linewidth=1.0,
        )
    )
    text(
        (xs[c] + xs[c + 1]) / 2,
        cadence_top + 39,
        label,
        size=24,
        color="white",
        ha="center",
    )
for c in range(4):
    ax.add_patch(
        Rectangle(
            (xs[c], cadence_top + 78),
            xs[c + 1] - xs[c],
            106,
            facecolor="#e4f0ec",
            edgecolor=border,
            linewidth=1.0,
        )
    )
text(
    xs[0] + 34,
    cadence_top + 131,
    "相邻动作指令下发间隔（目标50 ms）",
    size=26,
    color="#386b5c",
    strong=True,
)
for c, stat in enumerate(["mean", "p95", "max"], start=1):
    text(
        (xs[c] + xs[c + 1]) / 2,
        cadence_top + 131,
        f"{cadence[stat]:.2f}",
        size=30,
        color="#386b5c",
        strong=True,
        ha="center",
    )
for suffix in ["png", "svg"]:
    fig.savefig(
        root / f"rlinf_rlt_env_time_false.{suffix}",
        dpi=150,
        facecolor="white",
        metadata={"Title": "Rlinf-RLT的Env时间"},
    )
plt.close(fig)

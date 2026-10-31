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

"""Render false-mode Rollout timings with explicit module hierarchy."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle

root = Path(__file__).parent
source = json.loads((root / "true_false_20hz_classified_comparison.json").read_text())
data = source["false"]["workers"]["with_learner"]["Rollout step"]
regular = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
bold = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
fig = plt.figure(figsize=(24, 11.6), dpi=100, facecolor="white")
ax = fig.add_axes((0, 0, 1, 1))
ax.set_xlim(0, 2400)
ax.set_ylim(1160, 0)
ax.axis("off")
main, detail, border = "#214a73", "#626b78", "#d6dfe8"
accent = "#b13d46"


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


text(1200, 66, "Rlinf-RLT的Rollout时间", size=34, strong=True, ha="center")
xs = [60, 1140, 1540, 1940, 2340]
top, header_h, row_h = 138, 88, 88
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
        size=23,
        color="white",
        ha="center",
    )
rows = [
    ("1  输入数据处理", "input_ms", "main"),
    ("2  模型推理", "inference_ms", "main"),
    ("2.1  观测预处理", "preprocess_ms", "detail"),
    ("2.2  VLA前缀与KV Cache", "prefix_ms", "detail"),
    ("2.3  RLT编码得到z", "z_encode_ms", "detail"),
    ("2.4  生成参考动作", "reference_ms", "detail"),
    ("2.5  MLP及最终动作输出处理", "decision_ms", "detail"),
    ("3  发送动作至Env", "action_send_ms", "main"),
    ("4  整理并提交Transition数据", "data_submit_ms", "main"),
    ("Rollout有效工作合计", "work_ms", "total"),
]
for i, (label, key, kind) in enumerate(rows):
    y = top + header_h + i * row_h
    fill = (
        "#f7f8fa" if kind == "detail" else ("#fbecef" if kind == "total" else "#eaf1f8")
    )
    color = detail if kind == "detail" else (accent if kind == "total" else main)
    if key == "inference_ms":
        fill, color = "#eee5f8", "#703a9c"
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
    text(
        xs[0] + (90 if kind == "detail" else 34),
        y + row_h / 2,
        label,
        size=23,
        color=color,
        strong=kind != "detail",
    )
    for c, stat in enumerate(["mean", "p95", "max"], start=1):
        text(
            (xs[c] + xs[c + 1]) / 2,
            y + row_h / 2,
            f"{data[key][stat]:.2f}",
            size=25,
            color=color,
            strong=kind != "detail",
            ha="center",
        )
for suffix in ["png", "svg"]:
    fig.savefig(
        root / f"rlinf_rlt_rollout_time_false.{suffix}",
        dpi=150,
        facecolor="white",
        metadata={"Title": "Rlinf-RLT的Rollout时间"},
    )
plt.close(fig)

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

"""Render the measured false-mode learner timing table."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea
from matplotlib.patches import Rectangle

root = Path(__file__).parent
source = json.loads((root / "network_training_table_data.json").read_text())
font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
bold_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
regular = FontProperties(fname=font_path)
bold = FontProperties(fname=bold_path)
fig = plt.figure(figsize=(24, 8.5), dpi=100, facecolor="white")
ax = fig.add_axes((0, 0, 1, 1))
ax.set_xlim(0, 2400)
ax.set_ylim(850, 0)
ax.axis("off")
navy, gray, red = "#193f63", "#656f7b", "#b23b3b"


def text(x, y, value, size=20, color=navy, weight=False, ha="left"):
    ax.text(
        x,
        y,
        value,
        fontsize=size,
        color=color,
        fontproperties=bold if weight else regular,
        ha=ha,
        va="center",
    )


text(1200, 68, "Rlinf-RLT的网络训练时间", size=38, weight=True, ha="center")
xs = [60, 540, 1140, 1740, 2340]
top, header_h, row_h = 148, 100, 112
headers = ["指标（ms）", "预热阶段", "在线RL\n模型未推理", "在线RL\n模型推理中"]
for i, title in enumerate(headers):
    ax.add_patch(
        Rectangle(
            (xs[i], top),
            xs[i + 1] - xs[i],
            header_h,
            facecolor=navy,
            edgecolor="#d3e1ed",
            linewidth=1.2,
        )
    )
    text(
        (xs[i] + xs[i + 1]) / 2,
        top + header_h / 2,
        title,
        size=25,
        color="white",
        ha="center",
    )
rows = [
    ("Batch处理\n（采样与准备）", "batch_total_ms", False),
    ("单次Critic更新", "critic_ms", True),
    ("单次Actor更新", "actor_ms", True),
    ("单次目标网络软更新", "target_ms", True),
    ("一个网络更新Step\n（8次Critic＋2次Actor）", "network_step", True),
]
groups = ["warmup_steady", "idle", "active"]
for r, (label, key, network) in enumerate(rows):
    y = top + header_h + r * row_h
    fill = "#ffffff" if r % 2 == 0 else "#f1f6fa"
    for c in range(4):
        ax.add_patch(
            Rectangle(
                (xs[c], y),
                xs[c + 1] - xs[c],
                row_h,
                facecolor=fill,
                edgecolor="#d3e1ed",
                linewidth=1.1,
            )
        )
    text(
        xs[0] + 32,
        y + row_h / 2,
        label,
        size=25,
        color=red if network else "#222222",
        weight=network,
    )
    for c, g in enumerate(groups, start=1):
        value = (
            source["network_step"][g]
            if key == "network_step"
            else source["learner"][g][key]
        )
        center = (xs[c] + xs[c + 1]) / 2
        parts = [
            TextArea(
                f"均值 {value['mean']:.2f}",
                textprops={
                    "fontsize": 24,
                    "fontproperties": regular,
                    "color": red if network else "#222222",
                },
            ),
            TextArea(
                f"P95 {value['p95']:.2f}",
                textprops={"fontsize": 24, "fontproperties": regular, "color": gray},
            ),
            TextArea(
                f"最大值 {value['max']:.2f}",
                textprops={"fontsize": 24, "fontproperties": regular, "color": gray},
            ),
        ]
        line = HPacker(children=parts, align="baseline", pad=0, sep=8)
        ax.add_artist(
            AnnotationBbox(
                line, (center, y + row_h / 2), frameon=False, box_alignment=(0.5, 0.5)
            )
        )

for suffix in ["png", "svg"]:
    fig.savefig(
        root / f"rlinf_rlt_network_training_time_false.{suffix}",
        dpi=150,
        facecolor="white",
        metadata={"Title": "Rlinf-RLT的网络训练时间"},
    )
plt.close(fig)

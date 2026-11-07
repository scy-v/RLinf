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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle

BASE = Path(__file__).resolve().parent
FONT = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
BOLD = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
results = {}
for group in ("pi05_rtc", "rlt"):
    results.update(
        json.loads((BASE / f"inference_compile_{group}.json").read_text())["results"]
    )
    recheck = json.loads(
        (BASE / f"inference_compile_{group}_none_recheck.json").read_text()
    )
    for prefix in ("pi05", "rtc") if group == "pi05_rtc" else ("rlt",):
        samples = [
            x
            for i in (1, 2, 3)
            for x in recheck["results"][f"{prefix}_none_round{i}"]["samples_ms"]
        ]
        results[f"{prefix}_none"] = {"mean_ms": sum(samples) / len(samples)}

fig, ax = plt.subplots(figsize=(20, 13.2), dpi=140)
fig.patch.set_facecolor("#FAFBFD")
ax.set(xlim=(0, 2000), ylim=(1320, 0))
ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)


def text(x, y, value, size=20, color="#25334A", bold=False, ha="left"):
    ax.text(
        x,
        y,
        value,
        fontproperties=BOLD if bold else FONT,
        fontsize=size,
        color=color,
        va="center",
        ha=ha,
    )


def rect(x, y, w, h, color):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=color, edgecolor="none"))


text(70, 83, "Torch 编译推理延迟", size=48, bold=True)
rect(1630, 52, 28, 28, "#334155")
text(1674, 66, "深色行 = 默认采用", size=17)
rect(70, 160, 1860, 67, "#E8EDF3")
for x, label, align in [
    (98, "推理路径", "left"),
    (435, "编译范围", "left"),
    (1630, "平均耗时 / ms", "right"),
    (1817, "测量次数", "center"),
]:
    text(x, 193, label, size=25, bold=True, ha=align)

whole = "整段采样编译\n（前缀/KV cache＋全部去噪）"
groups = [
    (
        "PI05推理",
        "pi05",
        "#265FC5",
        "#EDF3FD",
        [
            ("none", "完全不使用 torch.compile", False),
            ("local", "RLinf的局部算子编译", False),
            ("whole", whole, True),
        ],
    ),
    (
        "RLT推理",
        "rlt",
        "#087F78",
        "#EAF6F3",
        [
            ("none", "完全不使用 torch.compile", False),
            ("local", "RLinf的局部算子编译", False),
            ("suffix", "参考动作去噪编译", True),
            ("prefix_suffix", "前缀/KV cache＋参考动作去噪编译", False),
            ("prefix_z_suffix", "前缀/KV cache＋z_rl 编码\n＋参考动作去噪编译", False),
        ],
    ),
    (
        "RTC PI05推理",
        "rtc",
        "#7550BB",
        "#F3EFFA",
        [
            ("none", "完全不使用 torch.compile", False),
            ("local", "RLinf的局部算子编译", False),
            ("whole", whole, True),
        ],
    ),
]
y = 239
for label, prefix, dark, light, rows in groups:
    height = 92 * len(rows)
    rect(70, y, 334, height, light)
    rect(70, y, 7, height, dark)
    text(100, y + height / 2, label, size=27, color=dark, bold=True)
    for kind, scope, default in rows:
        rect(410, y, 1520, 90, dark if default else light)
        fg = "white" if default else "#25334A"
        text(435, y + 45, scope, size=25, color=fg, bold=default)
        text(
            1630,
            y + 45,
            f"{results[f'{prefix}_{kind}']['mean_ms']:.2f}",
            size=33,
            color=fg,
            bold=True,
            ha="right",
        )
        text(
            1817,
            y + 45,
            "90" if kind == "none" else "30",
            size=28,
            color=fg,
            ha="center",
        )
        y += 92
    y += 14

for ext in ("png", "svg"):
    fig.savefig(
        BASE / f"torch_compile_latency_table.{ext}", facecolor=fig.get_facecolor()
    )
plt.close(fig)

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
import re
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).with_suffix(".json")


def stats(values):
    a = np.asarray(values) * 1000
    return (
        {
            "n": len(a),
            "mean_ms": float(a.mean()),
            "p95_ms": float(np.percentile(a, 95)),
            "max_ms": float(a.max()),
        }
        if len(a)
        else None
    )


report = {
    "scope": "all action rows in listed completed runs; 2026-11-06 session includes runs past midnight",
    "runs": [],
}
for path in sorted(
    (ROOT / "results/rlinf-ur/runs").glob("2026110[67]*/ur5e-actions.jsonl")
):
    cfg_path = path.parent / "tensorboard/config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    rows = [json.loads(line) for line in path.open()]
    threshold = cfg["env"]["eval"]["override_cfg"].get("gripper_close_threshold", 0.7)
    run = path.parent.name
    prediction_path = ROOT / f"results/rlinf-ur/logs/all_logs/log_{run}.yaml"
    timing_path = ROOT / f"results/rlinf-ur/logs/all_logs/timing_{run}.jsonl"
    timing = [json.loads(line) for line in timing_path.open()]
    predictions = [
        yaml.safe_load(part)
        for part in re.split(r"^ -+\s*$", prediction_path.read_text(), flags=re.M)
        if part.strip()
    ]
    segments = []
    for row in rows:
        state = "open" if row["action"][-1] > threshold else "close"
        if not segments or (segments[-1]["episode"], segments[-1]["state"]) != (
            row["episode_index"],
            state,
        ):
            segments.append(
                {"episode": row["episode_index"], "state": state, "rows": []}
            )
        segments[-1]["rows"].append(row)
    reversals = []
    for i in range(1, len(segments) - 1):
        before, opened, after = segments[i - 1 : i + 2]
        if (
            opened["state"] != "open"
            or not before["episode"] == opened["episode"] == after["episode"]
        ):
            continue
        selected = [before["rows"][-1]] + opened["rows"] + [after["rows"][0]]
        reversals.append(
            {
                "episode": opened["episode"],
                "open_steps": [opened["rows"][0]["step"], opened["rows"][-1]["step"]],
                "open_actions": len(opened["rows"]),
                "open_interval_ms": (
                    after["rows"][0]["timestamp"] - opened["rows"][0]["timestamp"]
                )
                * 1000,
                "trace": [
                    {k: x[k] for k in ["step", "action_index", "timestamp"]}
                    | {"gripper_action": x["action"][-1]}
                    for x in selected
                ],
            }
        )
    async_cfg = cfg["runner"].get("async_inference", cfg["runner"].get("rtc", {}))
    item = {
        "run": run,
        "sources": [
            str(p.relative_to(ROOT))
            for p in [path, cfg_path, prediction_path, timing_path]
        ],
        "continuous_tracking": cfg["cluster"]["node_groups"][0]["hardware"]["configs"][
            0
        ]["impedance"]["continuous_tracking"],
        "async": async_cfg.get("enabled", False),
        "rtc_enabled": cfg["rollout"]["model"]["openpi"]["rtc_enabled"],
        "threshold": threshold,
        "actions": len(rows),
        "close_open_close": reversals,
        "replan_model_ms": stats(
            [
                x["infer_sec"]
                for x in timing
                if x["event"] == "rtc_prediction" and x["request_type"] == "replan"
            ]
        ),
        "model_ms": stats(
            [
                x["infer_sec"]
                for x in timing
                if x["event"] == "step" and "infer_sec" in x
            ]
        ),
        "target_interval": stats(
            [
                x["target_interval_sec"]
                for x in rows
                if x.get("target_interval_sec") is not None
            ]
        ),
        "control": stats([x["control_sec"] for x in rows if "control_sec" in x]),
    }
    if run == "20261107-001310-ur5e-eval":
        item["closure_chunks"] = [
            {
                "episode": d["episode_index"],
                "chunk": d["episode_action_count"],
                "infer_index": d["infer_index"],
                "input_gripper": d["state"][-1],
                "predicted_gripper": [a[-1] for a in d["actions"]],
            }
            for d in predictions
            if "actions" in d
            and (
                (
                    d["episode_index"] == 1
                    and d["episode_action_count"] in [13, 14, 15, 36, 37]
                )
                or (
                    d["episode_index"] == 2
                    and d["episode_action_count"] in [16, 17, 18]
                )
            )
        ]
    report["runs"].append(item)
OUT.write_text(json.dumps(report, indent=2) + "\n")

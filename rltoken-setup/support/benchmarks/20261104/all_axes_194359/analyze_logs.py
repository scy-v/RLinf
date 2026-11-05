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

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from omegaconf import OmegaConf

O = Path("rltoken-setup/support/benchmarks/20261104/all_axes_194359")
TZ = ZoneInfo("Asia/Shanghai")
RUNS = [
    Path("logs/online_rltoken") / x
    for x in [
        "20261104-194359-ur_rlt_stage2_ac_mlp",
        "20261104-203520-ur_rlt_stage2_ac_mlp",
    ]
]


def stat(x):
    a = np.array(x, dtype=float)
    return (
        {
            "n": len(a),
            "mean": float(a.mean()),
            "median": float(np.median(a)),
            "p95": float(np.quantile(a, 0.95)),
            "max": float(a.max()),
            "sum": float(a.sum()),
        }
        if len(a)
        else {}
    )


D = {"runs": [], "episodes": [], "segments": {}, "warnings": []}
allmetrics = []
epsall = []
timearrays = []
for ri, r in enumerate(RUNS):
    rows = [
        json.loads(l) for l in (r / "metrics.log").read_text().splitlines() if l.strip()
    ]
    allmetrics += rows
    eps = []
    cur = {}
    lastsec = 0
    buffers = []
    for l in (r / "run_embodiment.log").read_text().splitlines():
        mt = re.search(r"\[(?:INFO|WARNING|ERROR) (\d\d:\d\d:\d\d)", l)
        if not mt:
            continue
        sec = (
            datetime.fromisoformat("2026-11-04T" + mt[1]).replace(tzinfo=TZ).timestamp()
        )
        lastsec = max(lastsec, sec)
        if "First RL action received" in l:
            cur = {"start": sec}
        if "Critical phase selected" in l:
            cur["critical"] = sec
        m = re.search(
            r"Stopped after (\d+) actions: reward=([\d.]+), terminated=(True|False), truncated=(True|False)",
            l,
        )
        if m:
            cur.update(
                end=sec,
                actions=int(m[1]),
                success=float(m[2]) > 0,
                timeout=m[4] == "True",
                run=ri,
                local_ordinal=len(eps) + 1,
                ordinal=len(epsall) + len(eps) + 1,
            )
            cur["critical_duration"] = (
                sec - cur["critical"] if "critical" in cur else None
            )
            cur["duration"] = sec - cur["start"] if "start" in cur else None
            eps.append(cur)
            cur = {}
        m = re.search(r"Episode committed before reset: episode=(\d+), chunks=(\d+)", l)
        if m and eps:
            eps[-1].update(log_episode=int(m[1]), full_chunks=int(m[2]))
        m = re.search(r"Critical chunks: (\d+) available; (\d+) collected total", l)
        if m:
            buffers.append([sec, int(m[1]), int(m[2])])
        if "[WARNING " in l or "[ERROR " in l:
            D["warnings"].append(l[-500:])
    # Associate update batches with preceding episode, warmup combines first seven.
    for row in rows:
        candidates = [e for e in eps if e["end"] <= row["timestamp"]]
        if candidates:
            e = candidates[-1]
            e.update(
                step=row["step"],
                new_chunks=row["metrics"]["train/replay/transition_count"],
                critic_updates=row["metrics"]["train/rlt/critic_updates_run"],
            )
    ts = []
    for line in (r / "timing.log").read_text().splitlines():
        if "[TIMING_RECORD] " not in line:
            continue
        try:
            t = json.loads(line.split("[TIMING_RECORD] ", 1)[1])
            ts.append(t)
        except json.JSONDecodeError:
            pass
    sources = {}
    for source in sorted({x["source"] for x in ts}):
        xs = [x for x in ts if x["source"] == source]
        fields = {}
        for k in [
            "work_ms",
            "inference_ms",
            "execute_ms",
            "input_ms",
            "action_send_ms",
            "data_submit_ms",
            "actor_ms",
            "critic_ms",
            "sample_ms",
            "prepare_ms",
            "measured_hz",
            "max_send_interval_ms",
            "action_interval_ms",
        ]:
            vals = [x[k] for x in xs if isinstance(x.get(k), (int, float))]
            if vals:
                fields[k] = stat(vals)
        sources[source] = {"records": len(xs), "fields": fields}
    learners = [x for x in ts if x["source"] == "Learner update"]
    last = learners[-1]
    first = learners[0]
    cps = sorted(
        (r / "ur_rlt_stage2_ac_mlp/checkpoints").glob("global_step_*"),
        key=lambda p: int(p.name.split("_")[-1]),
    )
    configs = OmegaConf.load(r / "tensorboard/config.yaml")
    rd = {
        "path": str(r),
        "resume": configs.runner.resume_dir,
        "first_metric_step": rows[0]["step"],
        "last_metric_step": rows[-1]["step"],
        "start_step": rows[-1]["start_step"],
        "runner_elapsed_to_last_metric": rows[-1]["elapsed_seconds"],
        "first_update_start": first["timestamp"] - first["work_ms"] / 1000,
        "last_update_end": last["timestamp"],
        "last_logged_update": last["update"],
        "learner_records": len(learners),
        "actor_update_records": sum(bool(x.get("actor_updated")) for x in learners),
        "last_log_time": lastsec,
        "completed_episodes": len(eps),
        "successes": sum(e["success"] for e in eps),
        "active_rollout_sec": sum(e["duration"] or 0 for e in eps),
        "timing": sources,
        "buffers_last": buffers[-1],
        "checkpoints": [int(p.name.split("_")[-1]) for p in cps],
        "source_sha256": {
            fn: hashlib.sha256((r / fn).read_bytes()).hexdigest()
            for fn in [
                "metrics.log",
                "run_embodiment.log",
                "timing.log",
                "tensorboard/config.yaml",
            ]
        },
    }
    D["runs"].append(rd)
    epsall += eps
    timearrays += ts
D["episodes"] = epsall
keys = [
    "train/actor/q_pi",
    "train/critic/q_data",
    "train/actor/bc_loss",
    "train/actor/bc_ref_loss",
    "train/actor/bc_human_loss",
    "train/sac/critic_loss",
    "train/sac/actor_loss",
    "train/actor/human_mask_ratio",
    "train/actor/grad_norm",
    "train/critic/grad_norm",
]
for lo, hi in [
    (16, 270),
    (271, 773),
    (774, 1251),
    (1252, 1775),
    (1776, 2279),
    (2280, 2522),
]:
    rs = [r for r in allmetrics if lo <= r["step"] <= hi]
    d = {}
    for k in keys:
        w = (
            "train/rlt/actor_updates_run"
            if k.startswith("train/actor/") or k == "train/sac/actor_loss"
            else "train/rlt/critic_updates_run"
        )
        d[k] = sum(r["metrics"][k] * r["metrics"][w] for r in rs) / sum(
            r["metrics"][w] for r in rs
        )
    D["segments"][f"{lo}-{hi}"] = {"batches": len(rs), "means": d}
D["episode_blocks"] = {}
for lo, hi in [(1, 7), (8, 35), (36, 55), (56, 75), (76, 96)]:
    es = [e for e in epsall if lo <= e["ordinal"] <= hi]
    D["episode_blocks"][f"{lo}-{hi}"] = {
        "n": len(es),
        "successes": sum(e["success"] for e in es),
        "critical_seconds": stat(
            [e["critical_duration"] for e in es if e["critical_duration"] is not None]
        ),
        "action_count": stat([e["actions"] for e in es]),
    }
D["max_batch_grad"] = {
    k: max(r["metrics"].get(k, 0) for r in allmetrics)
    for k in ["train/actor/grad_norm", "train/critic/grad_norm"]
}
D["nonfinite_metrics"] = [
    [r["step"], k]
    for r in allmetrics
    for k, v in r["metrics"].items()
    if isinstance(v, (float, int)) and not np.isfinite(v)
]
D["timing_blocks"] = {}
for name, lo, hi in [
    ("initial", 0, 10000),
    ("middle", 10000, 20000),
    ("late", 20000, 99999),
]:
    # Assign timing to last reported learner count at timestamp; startup excluded by bounds for rollout.
    xs = []
    for t in timearrays:
        prior = [r for r in allmetrics if r["timestamp"] <= t["timestamp"]]
        u = prior[-1]["metrics"]["train/rlt/update_step"] if prior else 0
        if lo <= u < hi:
            xs.append(t)
    D["timing_blocks"][name] = {
        s: stat(
            [
                x[k]
                for x in xs
                if x["source"] == s and isinstance(x.get(k), (int, float))
            ]
        )
        for s, k in [
            ("Rollout step", "inference_ms"),
            ("Env step", "execute_ms"),
            ("PD loop", "measured_hz"),
        ]
    }
(O / "logs.json").write_text(json.dumps(D, indent=2, ensure_ascii=False) + "\n")
np.savez_compressed(
    O / "metrics.npz",
    steps=np.array([r["step"] for r in allmetrics]),
    timestamps=np.array([r["timestamp"] for r in allmetrics]),
    **{k: np.array([r["metrics"].get(k, np.nan) for r in allmetrics]) for k in keys},
)
print(
    json.dumps(
        {k: v for k, v in D.items() if k not in ["episodes", "warnings", "runs"]},
        indent=2,
    )
)
for r in D["runs"]:
    print({k: v for k, v in r.items() if k not in ["timing", "source_sha256"]})

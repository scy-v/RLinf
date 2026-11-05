# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Read-only CPU analysis of the continuous-update 23:07 run."""

import csv
import hashlib
import itertools
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch
from omegaconf import OmegaConf

from rlinf.models.embodiment.mlp_policy import get_model

O = Path(__file__).resolve().parent
R = Path("logs/online_rltoken/20261104-230711-ur_rlt_stage2_ac_mlp")
STEPS = [2500, 7500, 12500, 15000, 17500, 20000, 22500]


def cp(s):
    return R / f"ur_rlt_stage2_ac_mlp/checkpoints/global_step_{s}/actor"


def stat(x):
    a = np.asarray(x, dtype=float)
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


torch.set_num_threads(2)
cfg = OmegaConf.load(R / "tensorboard/config.yaml")
metrics = [
    json.loads(l) for l in (R / "metrics.log").read_text().splitlines() if l.strip()
]
eps = []
cur = {}
day = datetime(2026, 11, 4, tzinfo=ZoneInfo("Asia/Shanghai"))
prev = None
step = 0
for l in (R / "run_embodiment.log").read_text().splitlines():
    mt = re.search(r"INFO (\d\d:\d\d:\d\d)", l)
    if not mt:
        continue
    h, m, s = map(int, mt[1].split(":"))
    t = day.replace(hour=h, minute=m, second=s)
    if prev and prev - t > timedelta(hours=12):
        day += timedelta(days=1)
        t += timedelta(days=1)
    prev = t
    st = re.search(r"Online RL runner rounds: (\d+)/", l)
    if st:
        step = int(st[1])
    if "First RL action received" in l:
        cur = {"start": t.timestamp()}
    if "Critical phase selected" in l:
        cur["critical"] = t.timestamp()
    end = re.search(r"Stopped after (\d+) actions: reward=([\d.]+)", l)
    if end:
        cur.update(
            episode=len(eps) + 1,
            end=t.timestamp(),
            actions=int(end[1]),
            success=float(end[2]) > 0,
            step=step,
        )
        eps.append(cur)
        cur = {}
D = {
    "source": str(R),
    "metrics_rows": len(metrics),
    "first_metrics": metrics[0]["step"],
    "last_metrics": metrics[-1]["step"],
    "episodes": eps,
    "segments": {},
    "timing": {},
    "checkpoints": {},
    "probe": {},
    "q_cross": {},
    "action_drift": {},
}
heads = {}
counts = {}
for s in STEPS:
    state = json.loads(
        (cp(s) / "sac_components/rlt_training_state_rank_0.json").read_text()
    )
    D["checkpoints"][s] = state
    counts[s] = state["total_transitions_added"]
    h = get_model(cfg.actor.model, torch_dtype=torch.float32).eval()
    h.load_state_dict(
        torch.load(
            cp(s) / "model_state_dict/full_weights.pt",
            weights_only=True,
            map_location="cpu",
        )
    )
    heads[s] = h
files = sorted(
    (cp(22500) / "sac_components/replay_buffer/rank_0").glob("trajectory_*.pt"),
    key=lambda p: int(p.name.split("_")[1]),
)
assert len(files) == 2153
rows = [torch.load(p, weights_only=False, map_location="cpu") for p in files]
obs = {
    k: torch.cat([r["curr_obs"][k].reshape(1, -1) for r in rows])
    for k in ["z_rl", "proprio", "ref_chunk"]
}
acts = torch.cat([r["actions"].reshape(1, 10, 7) for r in rows])
refs = obs["ref_chunk"].reshape(-1, 20, 7)[:, :10]
rew = torch.cat([r["rewards"].reshape(1, 10) for r in rows])
flags = torch.cat([r["intervene_flags"].reshape(1, 10, 7).any(-1) for r in rows])
done = np.array([bool(r["dones"].any()) for r in rows])
start = 150
ei = 0
labels = np.full(len(rows), "warmup", dtype="<U20")
for j in range(150, len(rows)):
    if not done[j]:
        continue
    e = eps[ei]
    ei += 1
    succ = bool(rew[j].gt(0).any())
    assert succ == e["success"]
    rp = int(rew[j].gt(0).nonzero()[0]) if succ else 10
    before = flags[start : j + 1].clone()
    if succ:
        before[-1, rp:] = False
    e.update(
        first_id=start,
        last_id=j,
        chunks=j - start + 1,
        replay_pre_reward_interventions=int(before.sum()),
        autonomous=succ and not bool(before.any()),
    )
    for i in range(start, j + 1):
        labels[i] = (
            "failure"
            if not succ
            else ("success_near30" if (j - i) * 10 + rp < 30 else "success_far")
        )
    start = j + 1
assert ei == 93 and start == len(rows)
# Audit the previously published correction against replay flags before reward.
prevrows = list(csv.DictReader((O.parent / "autonomous_success/episodes.csv").open()))
D["autonomy_audit"] = {
    "mismatches": [
        e["episode"]
        for e, x in zip(eps, prevrows)
        if "autonomous" in e and e["autonomous"] != bool(int(x["autonomous"]))
    ]
}
for lo, hi in [(1, 38), (39, 54), (55, 78), (79, 93)]:
    es = eps[lo - 1 : hi]
    t0 = es[0]["start"]
    t1 = es[-1]["end"]
    ms = [x for x in metrics if t0 <= x["timestamp"] <= t1]
    keys = [
        "train/actor/q_pi",
        "train/critic/q_data",
        "train/actor/bc_loss",
        "train/actor/bc_human_loss",
        "train/actor/bc_ref_loss",
        "train/sac/critic_loss",
        "train/sac/actor_loss",
        "train/actor/human_mask_ratio",
        "train/actor/grad_norm",
        "train/critic/grad_norm",
        "train/actor/weighted_bc",
        "train/actor/weighted_q",
    ]
    D["segments"][f"{lo}-{hi}"] = {
        "start": t0,
        "end": t1,
        "steps": [ms[0]["step"], ms[-1]["step"]],
        "successes": sum(e["success"] for e in es),
        "autonomous": sum(e["autonomous"] for e in es),
        "n": len(es),
        "human_steps": stat([e["replay_pre_reward_interventions"] for e in es]),
        "critical_sec": stat([e["end"] - e["critical"] for e in es if "critical" in e]),
        "loss": {
            k: stat([x["metrics"][k] for x in ms if k in x["metrics"]]) for k in keys
        },
    }
D["nonfinite_metrics"] = [
    (x["step"], k)
    for x in metrics
    for k, v in x["metrics"].items()
    if isinstance(v, (float, int)) and not np.isfinite(v)
]
D["all_time"] = {
    "network_first_last_seconds": metrics[-1]["timestamp"] - metrics[0]["timestamp"],
    "rollout_seconds": sum(e["end"] - e["start"] for e in eps),
    "last_update": metrics[-1]["metrics"]["train/rlt/update_step"],
}
timings = list(csv.DictReader((R / "timing.tsv").open(), delimiter="\t"))
for source in sorted({r["source"] for r in timings}):
    ts = [r for r in timings if r["source"] == source]
    D["timing"][source] = {
        k: stat([float(r[k]) for r in ts if r.get(k)])
        for k in [
            "work_ms",
            "inference_ms",
            "execute_ms",
            "active_interval_ms",
            "action_interval_ms",
        ]
        if any(r.get(k) for r in ts)
    }
D["timing_by_phase"] = {}
for phase, segment in D["segments"].items():
    ts = [
        r
        for r in timings
        if segment["start"]
        <= datetime.fromisoformat(r["timestamp"]).timestamp()
        <= segment["end"]
    ]
    D["timing_by_phase"][phase] = {
        source: {
            k: stat([float(r[k]) for r in ts if r["source"] == source and r.get(k)])
            for k in ["work_ms", "inference_ms", "execute_ms"]
            if any(r["source"] == source and r.get(k) for r in ts)
        }
        for source in sorted({r["source"] for r in ts})
    }
center = np.median(
    np.stack(
        [
            r["next_obs"]["proprio"].reshape(-1).numpy()[:3]
            for r in rows[150:]
            if bool(r["rewards"].gt(0).any())
        ]
    ),
    axis=0,
)
dist = np.linalg.norm(obs["proprio"].numpy()[:, :3] - center, axis=1) * 1000
eligible = np.flatnonzero(
    (~flags.any(-1)).numpy() & ~done & (np.arange(len(rows)) >= 150)
)


def pick(a, n):
    return a[np.linspace(0, len(a) - 1, min(n, len(a))).astype(int)]


ids = np.sort(
    np.r_[
        pick(eligible[dist[eligible] <= 30], 160),
        pick(eligible[dist[eligible] > 30], 32),
    ]
)
b = {k: v[ids] for k, v in obs.items()}
rf = refs[ids].reshape(len(ids), -1)
D["fixed_sample_ids"] = ids.tolist()
D["fixed_near_count"] = int((dist[ids] <= 30).sum())
means = {}
for s, h in heads.items():
    with torch.no_grad():
        pi = h.sac_forward(b, deterministic=True)[0]
        means[s] = pi
        q = h.sac_q_forward(b, pi)[:, 0]
        qr = h.sac_q_forward(b, rf)[:, 0]
        diff = (pi - rf).reshape(-1, 10, 7)
        result = {
            "qpi": float(q.mean()),
            "qref": float(qr.mean()),
            "delta_q_abs": float((q - qr).abs().mean()),
            "xyz_vs_ref_mm": float(diff[:, :, :3].norm(dim=-1).mean() * 1000),
            "rp_rms_degrees": np.rad2deg(
                pi.reshape(-1, 10, 7)[:, :, 3:5].square().mean((0, 1)).sqrt().numpy()
            ).tolist(),
        }
        result["axis_replace_q_drop"] = {}
        for name, sl in [
            ("xyz", slice(0, 3)),
            ("rp", slice(3, 5)),
            ("yaw", slice(5, 6)),
            ("gripper", slice(6, 7)),
        ]:
            a = pi.reshape(-1, 10, 7).clone()
            a[:, :, sl] = refs[ids][:, :, sl]
            result["axis_replace_q_drop"][name] = float(
                (q - h.sac_q_forward(b, a.reshape(len(ids), -1))[:, 0]).mean()
            )
    pars = [p for name, p in h.named_parameters() if not name.startswith("q_head.")]
    grads = []
    for seed in range(4):
        torch.manual_seed(seed)
        pi = h.sac_forward(b, apply_reference_dropout=True, reference_dropout_prob=0.5)[
            0
        ]
        lq = -0.1 * h.sac_q_forward(b, pi)[:, 0].mean()
        lb = 5 * (pi - rf).square().mean()
        gq = torch.autograd.grad(lq, pars, retain_graph=True, allow_unused=True)
        gb = torch.autograd.grad(lb, pars, allow_unused=True)
        gq = [torch.zeros_like(p) if g is None else g for p, g in zip(pars, gq)]
        gb = [torch.zeros_like(p) if g is None else g for p, g in zip(pars, gb)]
        nq = sum(g.square().sum() for g in gq).sqrt()
        nb = sum(g.square().sum() for g in gb).sqrt()
        cos = sum((a * z).sum() for a, z in zip(gq, gb)) / (nq * nb)
        grads.append([float(nq), float(nb), float(nb / nq), float(cos)])
    result["gradient_Q_BC_ratio_cos"] = np.mean(grads, axis=0).tolist()
    D["probe"][s] = result
    print("PROBE", s, result, flush=True)
for a, z in [
    (7500, 12500),
    (12500, 15000),
    (15000, 17500),
    (17500, 20000),
    (20000, 22500),
    (17500, 22500),
]:
    d = (means[z] - means[a]).reshape(-1, 10, 7)
    D["action_drift"][f"{a}-{z}"] = {
        "xyz_mm": float(d[:, :, :3].norm(dim=-1).mean() * 1000),
        "xyz_bias_mm": (d[:, :, :3].mean((0, 1)) * 1000).tolist(),
    }
with torch.no_grad():
    for a, z in [(12500, 17500), (17500, 22500)]:
        vv = [a, z]
        qp = np.zeros((2, 2, 2))
        for di, ds in enumerate(vv):
            nn = counts[ds]
            for ci, cs in enumerate(vv):
                for ai, as_ in enumerate(vv):
                    for k in range(0, nn, 256):
                        ob = {key: v[k : min(k + 256, nn)] for key, v in obs.items()}
                        for seed in range(4):
                            torch.manual_seed(seed * 10000 + k)
                            ac = heads[as_].sac_forward(
                                ob,
                                apply_reference_dropout=True,
                                reference_dropout_prob=0.5,
                            )[0]
                            qp[di, ci, ai] += float(
                                heads[cs].sac_q_forward(ob, ac)[:, 0].sum()
                            ) / (nn * 4)
        contrib = np.zeros(3)
        for perm in itertools.permutations(range(3)):
            coord = [0, 0, 0]
            for dim in perm:
                before = qp[tuple(coord)]
                coord[dim] = 1
                contrib[dim] += qp[tuple(coord)] - before
        D["q_cross"][f"{a}-{z}"] = {
            "qpi_cube": qp.tolist(),
            "contributions": dict(
                zip(["data", "critic", "actor"], (contrib / 6).tolist())
            ),
        }
        print("CROSS", a, z, D["q_cross"][f"{a}-{z}"], flush=True)
D["replay_stages"] = {}
for s, nn in counts.items():
    es = [e for e in eps if e.get("last_id", 99999) < nn]
    D["replay_stages"][s] = {
        "records": nn,
        "episodes": len(es),
        "autonomous": sum(e["autonomous"] for e in es),
        "labels": {k: int((labels[:nn] == k).sum()) for k in np.unique(labels[:nn])},
        "human_slot_ratio": float(flags[:nn].float().mean()),
    }
D["rp_observed_rms_degrees"] = np.rad2deg(
    obs["proprio"][150:, 3:5].square().mean(0).sqrt().numpy()
).tolist()
D["source_sha256"] = {
    f: hashlib.sha256((R / f).read_bytes()).hexdigest()
    for f in ["metrics.log", "tensorboard/config.yaml", "run_embodiment.log"]
}
(O / "analysis.json").write_text(json.dumps(D, ensure_ascii=False, indent=2) + "\n")
print("COMPLETE", flush=True)

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

"""Offline checkpoint and replay comparisons for the all-axis November 4 run."""

import itertools
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from rlinf.models.embodiment.mlp_policy import get_model

O = Path(__file__).resolve().parent
RUNS = [
    Path("logs/online_rltoken") / n
    for n in [
        "20261104-194359-ur_rlt_stage2_ac_mlp",
        "20261104-203520-ur_rlt_stage2_ac_mlp",
    ]
]
STEPS = [270, 773, 1251, 1775, 2279, 2522]


def cp(s):
    return (
        RUNS[int(s > 773)] / f"ur_rlt_stage2_ac_mlp/checkpoints/global_step_{s}/actor"
    )


def main():
    torch.set_num_threads(2)
    cfg = OmegaConf.load(RUNS[1] / "tensorboard/config.yaml")
    heads = {}
    counts = {}
    for s in STEPS:
        h = get_model(cfg.actor.model, torch_dtype=torch.float32).eval()
        h.load_state_dict(
            torch.load(
                cp(s) / "model_state_dict/full_weights.pt",
                map_location="cpu",
                weights_only=True,
            )
        )
        heads[s] = h
        counts[s] = json.loads(
            (cp(s) / "sac_components/rlt_training_state_rank_0.json").read_text()
        )["total_transitions_added"]
    files = sorted(
        (cp(2522) / "sac_components/replay_buffer/rank_0").glob("trajectory_*.pt"),
        key=lambda p: int(p.name.split("_")[1]),
    )
    assert [int(p.name.split("_")[1]) for p in files] == list(range(counts[2522]))
    rows = [torch.load(p, weights_only=False, map_location="cpu") for p in files]
    n = len(rows)
    obs = {
        k: torch.cat([r["curr_obs"][k].reshape(1, -1) for r in rows])
        for k in ["z_rl", "proprio", "ref_chunk"]
    }
    acts = torch.cat([r["actions"].reshape(1, 10, 7) for r in rows])
    refs = obs["ref_chunk"].reshape(n, 20, 7)[:, :10]
    rew = torch.cat([r["rewards"].reshape(1, 10) for r in rows])
    flags = torch.cat([r["intervene_flags"].reshape(1, 10, 7).any(-1) for r in rows])
    done = torch.tensor([bool(r["dones"].any()) for r in rows])
    success = rew.gt(0).any(-1)
    eps = []
    labels = np.full(n, "warmup", dtype="<U20")
    rets = np.full(n, np.nan)
    start = 0
    le = json.loads((O / "logs.json").read_text())["episodes"]
    for j in range(n):
        if not done[j]:
            continue
        succ = bool(success[j])
        rp = int(rew[j].gt(0).nonzero()[0]) if succ else 10
        hs = hm = 0
        g = 0.0
        for i in range(start, j + 1):
            before = torch.ones(10, dtype=torch.bool)
            if i == j and succ:
                before[rp:] = False
            hs += int((flags[i] & before).sum())
            hm += int(
                (flags[i] & before & (acts[i, :, :3].abs().amax(-1) > 1e-7)).sum()
            )
        for i in range(j, start - 1, -1):
            g = (
                float((rew[i] * torch.tensor(0.96 ** np.arange(10))).sum())
                + 0.96**10 * g
            )
            rets[i] = g
            if i >= 151:
                labels[i] = (
                    "failure"
                    if not succ
                    else "success_near30"
                    if (j - i) * 10 + rp < 30
                    else "success_far"
                )
        e = {
            "first_id": start,
            "last_id": j,
            "chunks": j - start + 1,
            "success": succ,
            "human_pre_success_steps": hs,
            "human_motion_steps": hm,
            "no_pre_success_override": succ and hs == 0,
            "ordinal": len(eps) + 1,
        }
        loge = le[len(eps)]
        assert loge["success"] == succ, (e, loge)
        e.update(
            {
                k: loge.get(k)
                for k in [
                    "run",
                    "local_ordinal",
                    "log_episode",
                    "step",
                    "critical_duration",
                    "end",
                ]
            }
        )
        eps.append(e)
        start = j + 1
    assert len(eps) == 95 and start == n and eps[6]["last_id"] == 150
    center = np.median(
        np.stack(
            [
                r["next_obs"]["proprio"].reshape(-1).numpy()[:3]
                for r in rows[151:]
                if bool(r["rewards"].gt(0).any())
            ]
        ),
        axis=0,
    )
    dist = np.linalg.norm(obs["proprio"].numpy()[:, :3] - center, axis=1) * 1000
    eligible = np.flatnonzero((~flags.any(-1) & ~done).numpy() & (np.arange(n) >= 151))
    near = eligible[dist[eligible] <= 30]
    far = eligible[dist[eligible] > 30]

    # Equal temporal strata prevent long late episodes alone dominating fixed-input probes.
    def pick(a, num):
        return a[np.linspace(0, len(a) - 1, min(num, len(a))).astype(int)]

    ids = np.sort(np.concatenate([pick(near, 160), pick(far, 32)]))
    assert len(np.unique(ids)) == 192
    b = {k: v[ids] for k, v in obs.items()}
    rf = refs[ids].reshape(192, -1)
    D = {
        "run_paths": [str(p) for p in RUNS],
        "last_saved_step": 2522,
        "records": n,
        "warmup_records": 151,
        "episodes": eps,
        "fixed_sample_ids": ids.tolist(),
        "fixed_sample_near_count": int((dist[ids] <= 30).sum()),
        "goal_center_m": center.tolist(),
        "fixed_inputs": {},
        "replay_stages": {},
        "q_cross": {},
        "axis_replay": {},
        "action_drift": {},
    }

    def axis(mask):
        a = acts[mask]
        ref = refs[mask]
        p = obs["proprio"][mask]
        nm = ~flags[mask]
        return {
            "nonhuman_slots": int(nm.sum()),
            "policy_rms_rp_degrees": np.rad2deg(
                a[nm][:, 3:5].square().mean(0).sqrt().numpy()
            ).tolist(),
            "reference_rms_rp_degrees": np.rad2deg(
                ref[nm][:, 3:5].square().mean(0).sqrt().numpy()
            ).tolist(),
            "observed_relative_rp_rms_degrees": np.rad2deg(
                p[:, 3:5].square().mean(0).sqrt().numpy()
            ).tolist(),
            "observed_relative_rp_abs_p95_degrees": np.rad2deg(
                np.quantile(np.abs(p[:, 3:5].numpy()), 0.95, axis=0)
            ).tolist(),
            "policy_xyz_rms_mm": (a[nm][:, :3].square().mean(0).sqrt() * 1000).tolist(),
            "force_norm_mean": float(p[:, 12:15].norm(dim=-1).mean()),
        }

    for name, lo, hi in [
        ("online_all", 151, n),
        ("early_online", 151, counts[773]),
        ("middle", counts[773], counts[1775]),
        ("late", counts[1775], n),
    ]:
        mask = torch.arange(n).ge(lo) & torch.arange(n).lt(hi)
        D["axis_replay"][name] = axis(mask)
    for s, nn in counts.items():
        es = [e for e in eps[7:] if e["last_id"] < nn]
        ll = labels[:nn]
        D["replay_stages"][s] = {
            "records": nn,
            "online_episodes": len(es),
            "successes": sum(e["success"] for e in es),
            "no_pre_success_override_success": sum(
                e["no_pre_success_override"] for e in es
            ),
            "human_slot_ratio": float(flags[:nn].float().mean()),
            "labels": {k: int((ll == k).sum()) for k in np.unique(ll)},
            "observed_discounted_return_online": float(np.mean(rets[151:nn])),
        }
    means = {}
    for s, h in heads.items():
        with torch.no_grad():
            pi = h.sac_forward(b, deterministic=True)[0]
            means[s] = pi
            q = h.sac_q_forward(b, pi)[:, 0]
            qr = h.sac_q_forward(b, rf)[:, 0]
            diff = (pi - rf).reshape(192, 10, 7)
            m = {
                "q_pi_mean": float(q.mean()),
                "q_ref_mean": float(qr.mean()),
                "action_delta_q_abs_mean": float((q - qr).abs().mean()),
                "xyz_vs_ref_mean_mm": float(diff[:, :, :3].norm(dim=-1).mean() * 1000),
                "rp_output_rms_deg": np.rad2deg(
                    pi.reshape(192, 10, 7)[:, :, 3:5]
                    .square()
                    .mean((0, 1))
                    .sqrt()
                    .numpy()
                ).tolist(),
                "xyz_bias_vs_ref_mm": (diff[:, :, :3].mean((0, 1)) * 1000).tolist(),
                "det_ref_mse": float(diff.square().mean()),
            }
            axisq = {}
            for name, sl in [
                ("xyz", slice(0, 3)),
                ("rp", slice(3, 5)),
                ("yaw", slice(5, 6)),
                ("gripper", slice(6, 7)),
            ]:
                a = pi.reshape(192, 10, 7).clone()
                a[:, :, sl] = refs[ids][:, :, sl]
                axisq[name] = float(
                    (q - h.sac_q_forward(b, a.reshape(192, -1))[:, 0]).mean()
                )
            m["q_drop_replace_axis_with_reference"] = axisq
        pars = [p for name, p in h.named_parameters() if not name.startswith("q_head.")]
        grad = []
        for seed in range(4):
            torch.manual_seed(seed)
            pi = h.sac_forward(
                b, apply_reference_dropout=True, reference_dropout_prob=0.5
            )[0]
            lq = -0.1 * h.sac_q_forward(b, pi)[:, 0].mean()
            lb = 5 * (pi - rf).square().mean()
            gq = torch.autograd.grad(lq, pars, retain_graph=True, allow_unused=True)
            gb = torch.autograd.grad(lb, pars, allow_unused=True)
            gq = [torch.zeros_like(p) if g is None else g for p, g in zip(pars, gq)]
            gb = [torch.zeros_like(p) if g is None else g for p, g in zip(pars, gb)]
            nq = sum(g.square().sum() for g in gq).sqrt()
            nb = sum(g.square().sum() for g in gb).sqrt()
            cos = sum((a * b).sum() for a, b in zip(gq, gb)) / (nq * nb)
            grad.append([float(nq), float(nb), float(nb / nq), float(cos)])
        m["weighted_grad_Q_BC_ratio_cos"] = np.mean(grad, axis=0).tolist()
        m["grad_seeds"] = grad
        D["fixed_inputs"][s] = m
        print("FIXED", s, m, flush=True)
    for a, z in [(270, 773), (773, 1775), (1775, 2522), (2279, 2522), (773, 2522)]:
        d = (means[z] - means[a]).reshape(192, 10, 7)
        D["action_drift"][f"{a}-{z}"] = {
            "xyz_mean_l2_mm": float(d[:, :, :3].norm(dim=-1).mean() * 1000),
            "xyz_bias_mm": (d[:, :, :3].mean((0, 1)) * 1000).tolist(),
            "rp_rms_deg": np.rad2deg(
                d[:, :, 3:5].square().mean((0, 1)).sqrt().numpy()
            ).tolist(),
        }
    with torch.no_grad():
        for a, z in [(773, 2522), (1775, 2522)]:
            vv = [a, z]
            qp = np.zeros((2, 2, 2))
            qd = np.zeros((2, 2))
            for di, ds in enumerate(vv):
                nn = counts[ds]
                for ci, cs in enumerate(vv):
                    for k in range(0, nn, 256):
                        ob = {key: v[k : min(k + 256, nn)] for key, v in obs.items()}
                        aa = acts[k : min(k + 256, nn)].reshape(-1, 70)
                        qd[di, ci] += (
                            float(heads[cs].sac_q_forward(ob, aa).mean(-1).sum()) / nn
                        )
                        for ai, as_ in enumerate(vv):
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
                coords = [0, 0, 0]
                for dim in perm:
                    before = qp[tuple(coords)]
                    coords[dim] = 1
                    contrib[dim] += qp[tuple(coords)] - before
            contrib /= 6
            D["q_cross"][f"{a}-{z}"] = {
                "q_data": qd.tolist(),
                "q_pi": qp.tolist(),
                "contributions": dict(
                    zip(["data", "critic", "actor"], contrib.tolist())
                ),
            }
            print("CROSS", a, z, D["q_cross"][f"{a}-{z}"], flush=True)
    D["episode_blocks"] = {}
    for lo, hi in [(8, 35), (36, 55), (56, 75), (76, 95)]:
        es = [e for e in eps if lo <= e["ordinal"] <= hi]
        D["episode_blocks"][f"{lo}-{hi}"] = {
            "n": len(es),
            "successes": sum(e["success"] for e in es),
            "no_pre_success_override": [
                e["ordinal"] for e in es if e["no_pre_success_override"]
            ],
            "human_pre_success_steps_mean": float(
                np.mean([e["human_pre_success_steps"] for e in es])
            ),
            "human_motion_steps_mean": float(
                np.mean([e["human_motion_steps"] for e in es])
            ),
        }
    (O / "probe.json").write_text(json.dumps(D, ensure_ascii=False, indent=2) + "\n")
    np.savez_compressed(
        O / "fixed_actions.npz",
        sample_ids=ids,
        reference=rf.numpy(),
        **{str(s): a.numpy() for s, a in means.items()},
    )
    print("AXIS", D["axis_replay"], "EPISODES", D["episode_blocks"], flush=True)


if __name__ == "__main__":
    main()

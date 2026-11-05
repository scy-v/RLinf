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

"""Measure local Q response on adjacent replay states near successful endpoints."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

from rlinf.models.embodiment.mlp_policy import get_model

RUN = Path("logs/online_rltoken/20261103-224754-ur_rlt_stage2_ac_mlp")
OUTPUT = Path(__file__).resolve().parent
CHECKPOINTS = (2003, 1520, 3276)
# Units: mm, degrees, N, m/s, Nm, rad/s, gripper units, cosine similarity.
GATES = {
    "strict": (2, 0.5, 1, 0.02, 0.15, 0.05, 0.01, 0.99),
    "primary": (5, 2, 2, 0.03, 0.3, 0.1, 0.02, 0.98),
    "loose": (10, 3, 5, 0.05, 0.5, 0.2, 0.03, 0.95),
}


def checkpoint(step):
    return RUN / f"ur_rlt_stage2_ac_mlp/checkpoints/global_step_{step}/actor"


def describe(values):
    a = np.asarray(values)
    return {
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "p10": float(np.quantile(a, 0.1)),
        "p90": float(np.quantile(a, 0.9)),
    }


def summarize(q, mask, episode):
    # Columns: s1/ref1, s1/pi1, s2/ref1, s2/pi1,
    #          s1/ref2, s1/pi2, s2/ref2, s2/pi2.
    q = q[mask].astype(np.float64)
    if len(q) == 0:
        return {"pairs": 0}
    ds = np.abs(q[:, [0, 1, 4, 5]] - q[:, [2, 3, 6, 7]]).mean(1)
    da = np.abs(q[:, [1, 3, 5, 7]] - q[:, [0, 2, 4, 6]]).mean(1)
    native = (np.abs(q[:, 1] - q[:, 0]) + np.abs(q[:, 7] - q[:, 6])) / 2
    e = episode[mask]
    unique = np.unique(e)
    totals = np.array(
        [[ds[e == i].sum(), da[e == i].sum(), (e == i).sum()] for i in unique]
    )
    rng = np.random.default_rng(20261104)
    draws = rng.integers(0, len(unique), size=(1000, len(unique)))
    bootstrap = totals[draws].sum(1)
    ratios = bootstrap[:, 0] / np.maximum(bootstrap[:, 1], 1e-15)
    return {
        "pairs": len(q),
        "episodes": len(unique),
        "state_delta": describe(ds),
        "action_delta": describe(da),
        "ratio_of_means": float(ds.mean() / da.mean()),
        "ratio_episode_cluster_ci95": np.quantile(ratios, [0.025, 0.975]).tolist(),
        "episode_equal_weight_ratio": float(
            np.mean(totals[:, 0] / totals[:, 2]) / np.mean(totals[:, 1] / totals[:, 2])
        ),
        "fraction_state_delta_larger": float(np.mean(ds > da)),
        "native_action_delta_mean": float(native.mean()),
        "native_state_to_action_ratio": float(ds.mean() / native.mean()),
        "mean_signed_policy_advantage": float(
            (q[:, [1, 3, 5, 7]] - q[:, [0, 2, 4, 6]]).mean()
        ),
    }


def main():
    torch.set_num_threads(2)
    root = checkpoint(3276) / "sac_components/replay_buffer/rank_0"
    files = sorted(
        root.glob("trajectory_*.pt"), key=lambda p: int(p.name.split("_")[1])
    )
    assert [int(p.name.split("_")[1]) for p in files] == list(range(3396))
    rows = [torch.load(p, weights_only=False, map_location="cpu") for p in files]
    obs = {
        k: torch.cat([r["curr_obs"][k].reshape(1, -1) for r in rows])
        for k in ("z_rl", "proprio", "ref_chunk")
    }
    state = obs["proprio"].numpy()
    next_state = np.stack([r["next_obs"]["proprio"].reshape(-1).numpy() for r in rows])
    done = np.array([bool(r["dones"].any()) for r in rows])
    human = np.array([bool(r["intervene_flags"].any()) for r in rows])
    success = np.array([bool(r["rewards"].gt(0).any()) for r in rows])
    goals = next_state[150:][success[150:], :3]
    center = np.median(goals, axis=0)
    distance = np.linalg.norm(state[:, :3] - center, axis=1) * 1000
    episode = np.full(len(rows), -1)
    remaining = np.full(len(rows), -1)
    successful_episode = np.zeros(len(rows), dtype=bool)
    start = 150
    ep = 0
    for i in range(150, len(rows)):
        if done[i]:
            episode[start : i + 1] = ep
            successful_episode[start : i + 1] = success[i]
            if success[i]:
                remaining[start : i + 1] = np.arange(i - start, -1, -1)
            start, ep = i + 1, ep + 1
    assert start == len(rows)
    ids = np.arange(150, len(rows) - 1)
    valid = (
        ~done[ids]
        & ~done[ids + 1]
        & ~human[ids]
        & ~human[ids + 1]
        & (episode[ids] == episode[ids + 1])
    )
    ids = ids[valid]
    assert np.array_equal(next_state[ids], state[ids + 1])
    p1, p2 = state[ids], state[ids + 1]
    z1, z2 = obs["z_rl"].numpy()[ids], obs["z_rl"].numpy()[ids + 1]
    distances = np.column_stack(
        [
            np.linalg.norm(p2[:, :3] - p1[:, :3], axis=1) * 1000,
            np.rad2deg(
                (
                    Rotation.from_euler("xyz", p2[:, 3:6])
                    * Rotation.from_euler("xyz", p1[:, 3:6]).inv()
                ).magnitude()
            ),
            np.linalg.norm(p2[:, 12:15] - p1[:, 12:15], axis=1),
            np.linalg.norm(p2[:, 6:9] - p1[:, 6:9], axis=1),
            np.linalg.norm(p2[:, 15:18] - p1[:, 15:18], axis=1),
            np.linalg.norm(p2[:, 9:12] - p1[:, 9:12], axis=1),
            np.abs(p2[:, 18] - p1[:, 18]),
            (z1 * z2).sum(1)
            / (np.linalg.norm(z1, axis=1) * np.linalg.norm(z2, axis=1)),
        ]
    )
    gates = {
        name: (distances[:, :7] <= np.array(limits[:7])).all(1)
        & (distances[:, 7] >= limits[7])
        for name, limits in GATES.items()
    }
    near = (distance[ids] <= 30) & (distance[ids + 1] <= 30)
    approach = ~near & (distance[ids] <= 50) & (distance[ids + 1] <= 50)
    far_count = int((gates["loose"] & ~near & ~approach).sum())
    near_ids = np.flatnonzero(near & gates["loose"])
    candidates = np.flatnonzero(approach & gates["loose"])
    control_count = min(len(candidates), int(np.ceil(len(near_ids) / 9)))
    controls = candidates[
        np.linspace(0, len(candidates) - 1, control_count).astype(int)
    ]
    chosen = np.sort(np.concatenate([near_ids, controls]))
    pair_ids = ids[chosen]
    gates = {k: v[chosen] for k, v in gates.items()}
    near = near[chosen]
    geometry = distances[chosen]
    regions = {
        "near_30mm_loose": near,
        "near_30mm_primary": near & gates["primary"],
        "near_30mm_strict": near & gates["strict"],
        "near_20mm_primary": gates["primary"]
        & (distance[pair_ids] <= 20)
        & (distance[pair_ids + 1] <= 20),
        "near_30mm_contact_primary": near
        & gates["primary"]
        & (np.linalg.norm(state[pair_ids, 12:15], axis=1) >= 3)
        & (np.linalg.norm(state[pair_ids + 1, 12:15], axis=1) >= 3),
        "near_30mm_failed_episode_primary": near
        & gates["primary"]
        & ~successful_episode[pair_ids],
        "near_30mm_successful_episode_primary": near
        & gates["primary"]
        & successful_episode[pair_ids],
        "near_30mm_success_last_5_chunks_loose": near
        & (remaining[pair_ids] >= 2)
        & (remaining[pair_ids] <= 5),
        "approach_30_to_50mm_control": ~near,
    }
    arrays = {
        "pair_record_ids": np.column_stack([pair_ids, pair_ids + 1]),
        "episode_ids": episode[pair_ids],
        "geometry": geometry,
        "goal_distance_mm": np.column_stack(
            [distance[pair_ids], distance[pair_ids + 1]]
        ),
        **{"mask_" + k: v for k, v in regions.items()},
    }
    report = {
        "run": str(RUN),
        "last_saved_step": 3276,
        "method": {
            "q_head": "q1",
            "dtype": "float32",
            "device": "cpu",
            "actor": "deterministic; no reference dropout or action noise",
            "state": "stored z_rl visual feature plus raw 19D proprio",
            "pairing": "adjacent nonterminal chunks in same episode; neither chunk intervened",
            "success_proxy": "median successful next_obs position; geometry is not visual insertion annotation",
            "goal_center_m": center.tolist(),
            "goal_spread_mm": describe(np.linalg.norm(goals - center, axis=1) * 1000),
            "gates_units": [
                "mm",
                "degrees",
                "N",
                "m/s",
                "Nm",
                "rad/s",
                "gripper",
                "cosine",
            ],
            "gates": GATES,
            "initial_eligible_adjacent_pairs": len(ids),
            "selected_pairs": len(pair_ids),
            "near_pairs": int(near.sum()),
            "control_pairs": int((~near).sum()),
            "eligible_pairs_beyond_50mm": far_count,
            "bootstrap": "1000 resamples of episodes, seed=20261104; temporal pairs are clustered",
            "source_config_sha256": hashlib.sha256(
                (RUN / "tensorboard/config.yaml").read_bytes()
            ).hexdigest(),
        },
        "groups": {},
        "checkpoints": {},
    }
    for name, mask in regions.items():
        report["groups"][name] = {
            "pairs": int(mask.sum()),
            "episodes": len(np.unique(episode[pair_ids][mask])),
            "geometry": [describe(v) for v in geometry[mask].T] if mask.any() else [],
        }
    cfg = OmegaConf.load(RUN / "tensorboard/config.yaml")
    o1 = {k: v[pair_ids] for k, v in obs.items()}
    o2 = {k: v[pair_ids + 1] for k, v in obs.items()}
    ref1 = o1["ref_chunk"].reshape(-1, 20, 7)[:, :10]
    ref2 = o2["ref_chunk"].reshape(-1, 20, 7)[:, :10]

    for step in CHECKPOINTS:
        model = get_model(cfg.actor.model, torch_dtype=torch.float32).eval()
        weights = checkpoint(step) / "model_state_dict/full_weights.pt"
        model.load_state_dict(
            torch.load(weights, map_location="cpu", weights_only=True)
        )
        model_results = {
            "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest()
        }
        with torch.no_grad():
            pi1 = model.sac_forward(o1, deterministic=True)[0].reshape(-1, 10, 7)
            pi2 = model.sac_forward(o2, deterministic=True)[0].reshape(-1, 10, 7)
            change_ref = (ref1 - ref2)[:, :, :3].norm(dim=-1).mean(1).numpy() * 1000
            change_pi = (pi1 - pi2)[:, :, :3].norm(dim=-1).mean(1).numpy() * 1000
            action_difference = (pi1 - ref1)[:, :, :3].norm(dim=-1).mean(
                1
            ).numpy() * 1000
            action_local = (change_ref <= 5) & (change_pi <= 5)
            group_masks = {
                **regions,
                "near_primary_action_local": regions["near_30mm_primary"]
                & action_local,
            }
            arrays[f"action_local_{step}"] = action_local
            model_results["action_distance_near_primary_mm"] = {
                "reference_between_states": describe(
                    change_ref[regions["near_30mm_primary"]]
                ),
                "policy_between_states": describe(
                    change_pi[regions["near_30mm_primary"]]
                ),
                "policy_vs_reference": describe(
                    action_difference[regions["near_30mm_primary"]]
                ),
            }
            if step == 2003:
                fixed_pi1, fixed_pi2 = pi1.clone(), pi2.clone()
            for variant in ("full", "rp_matched", "xyz_only", "fixed_actor_2003"):
                a1, a2 = pi1.clone(), pi2.clone()
                if variant == "fixed_actor_2003":
                    a1, a2 = fixed_pi1, fixed_pi2
                elif variant == "rp_matched":
                    a1[:, :, 3:5], a2[:, :, 3:5] = ref1[:, :, 3:5], ref2[:, :, 3:5]
                elif variant == "xyz_only":
                    a1[:, :, 3:], a2[:, :, 3:] = ref1[:, :, 3:], ref2[:, :, 3:]
                values = []
                for ob, action in (
                    (o1, ref1),
                    (o1, a1),
                    (o2, ref1),
                    (o2, a1),
                    (o1, ref2),
                    (o1, a2),
                    (o2, ref2),
                    (o2, a2),
                ):
                    values.append(
                        model.sac_q_forward(ob, action.reshape(len(pair_ids), -1))[
                            :, 0
                        ].numpy()
                    )
                q = np.stack(values, axis=1)
                assert np.isfinite(q).all()
                arrays[f"q_{step}_{variant}"] = q
                model_results[variant] = {
                    name: summarize(q, mask, episode[pair_ids])
                    for name, mask in group_masks.items()
                }
        report["checkpoints"][step] = model_results
        print(step, json.dumps(model_results["full"]["near_30mm_primary"]), flush=True)
    np.savez_compressed(OUTPUT / "paired_q.npz", **arrays)
    (OUTPUT / "results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print("groups", json.dumps(report["groups"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

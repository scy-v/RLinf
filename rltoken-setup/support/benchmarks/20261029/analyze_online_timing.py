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

"""Read-only timing analysis using raw records and their producer timestamps."""

import argparse
import bisect
import collections
import csv
import datetime
import json
import re
import statistics
from pathlib import Path

import numpy as np


def stats(values):
    values = [float(x) for x in values if x is not None]
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": max(values),
    }


def fields(rows, keys):
    return {key: stats([r.get(key) for r in rows]) for key in keys}


def analyze(directory):
    path = Path(directory)
    records = collections.defaultdict(list)
    events = []
    malformed = []
    for line_no, line in enumerate(
        (path / "run_embodiment.log").read_text().splitlines(), 1
    ):
        if "[TIMING_RECORD] " in line:
            try:
                row = json.JSONDecoder().raw_decode(
                    line.split("[TIMING_RECORD] ", 1)[1]
                )[0]
            except (ValueError, KeyError):
                malformed.append(line_no)
                continue
            row["line"] = line_no
            row["start"] = row["timestamp"] - row["work_ms"] / 1000
            records[row["source"]].append(row)
        elif any(
            t in line
            for t in [
                "[LOAD]",
                "[INFER]",
                "[SAVE]",
                "[REPLAY]",
                "[BUFFER]",
                "[EPISODE]",
                "Warmup updates",
                "Critical phase selected",
                "Reset complete",
                "UserWarning:",
                "[ERROR]",
                "Traceback",
                "Saving checkpoint",
            ]
        ):
            match = re.search(r"(?:INFO|WARNING|ERROR) (\d\d:\d\d:\d\d)", line)
            stamp = None
            if match:
                date = path.name[:8]
                stamp = (
                    datetime.datetime.strptime(date + match[1], "%Y%m%d%H:%M:%S")
                    .replace(tzinfo=datetime.timezone(datetime.timedelta(hours=8)))
                    .timestamp()
                )
            events.append({"line": line_no, "timestamp_second": stamp, "text": line})
    for rows in records.values():
        rows.sort(key=lambda r: r["timestamp"])
    actions = records["UR action"]
    inferences = records["Rollout inference"]
    learners = records["Learner update"]
    runners = records["Runner step"]
    episodes = collections.defaultdict(list)
    for action in actions:
        episodes[int(action["episode"])].append(action)
    windows = [
        (rows[0]["start"], rows[-1]["timestamp"], ep)
        for ep, rows in sorted(episodes.items())
    ]

    def situation(row):
        for start, end, _ in windows:
            if row["start"] >= start and row["timestamp"] <= end:
                return "active_episode"
            if row["timestamp"] > start and row["start"] < end:
                return "straddles_episode_boundary"
        return "outside_action_execution"

    for rows in records.values():
        for row in rows:
            row["situation"] = situation(row)
    result = {
        "run": str(path),
        "method": "Raw producer timestamps. Active episode: entire measured call inside first-to-last UR action window. Outside execution includes reset and user waiting. Boundary-straddling calls excluded from those two groups. Times in ms unless marked seconds.",
        "records": {s: len(rows) for s, rows in records.items()},
        "malformed_raw_lines": malformed,
        "missing_indices": {
            s: sorted(
                set(range(1, max(int(r["sample"]) for r in rows) + 1))
                - {int(r["sample"]) for r in rows}
            )
            for s, rows in records.items()
        },
        "events": [
            e
            for e in events
            if not (
                (
                    "[LOAD]" in e["text"]
                    or "[INFER]" in e["text"]
                    or "[SAVE]" in e["text"]
                )
                and "Completed after" not in e["text"]
            )
            and "[PROGRESS]" not in e["text"]
        ],
    }
    learner_keys = [
        "work_ms",
        "sample_ms",
        "prepare_ms",
        "critic_ms",
        "actor_ms",
        "target_ms",
        "other_ms",
    ]
    groups = {
        "warmup_first": [r for r in learners if r["update"] == 1],
        "warmup_steady": [r for r in learners if 1 < r["update"] <= 150],
    }
    for label in [
        "outside_action_execution",
        "active_episode",
        "straddles_episode_boundary",
    ]:
        groups["online_" + label] = [
            r for r in learners if r["update"] > 150 and r["situation"] == label
        ]
    result["learners"] = {}
    for name, rows in groups.items():
        result["learners"][name] = fields(rows, learner_keys)
        result["learners"][name]["critic_only_call"] = stats(
            [r["work_ms"] for r in rows if not r["actor_updated"]]
        )
        joint = [r for r in rows if r["actor_updated"]]
        result["learners"][name]["joint_call"] = fields(joint, learner_keys)
    result["warmup_wall_seconds"] = learners[149]["timestamp"] - learners[0]["start"]
    result["actor_update_count"] = sum(bool(r["actor_updated"]) for r in learners)
    result["inference_by_policy_version"] = {
        name: fields(
            [r for r in inferences if (r["policy_version"] == 0) == untrained],
            ["work_ms"],
        )
        for name, untrained in [("before_learner", True), ("after_learner", False)]
    }
    result["episodes"] = []
    cursor = 0
    per_chunk = []
    for ep, rows in sorted(episodes.items()):
        chunks = collections.defaultdict(list)
        for row in rows:
            chunks[int(row["chunk"])].append(row)
        ack = next(
            e
            for e in events
            if "[REPLAY]" in e["text"] and f"episode={ep}," in e["text"]
        )
        count = int(re.search(r"chunks=(\d+)", ack["text"]).group(1))
        assert len(chunks) == max(chunks) and count - len(chunks) in (0, 1)
        inf = inferences[cursor : cursor + count + 1]
        cursor += count + 1
        assert len(inf) == count + 1
        firsts = []
        gaps = []
        overhead = []
        chunk_durations = []
        for k, chunk in sorted(chunks.items()):
            prediction = inf[k - 1]
            assert prediction["timestamp"] <= chunk[0]["start"] + 0.01, (ep, k)
            firsts.append(chunk[0]["start"])
            chunk_durations.append(1000 * (chunk[-1]["timestamp"] - chunk[0]["start"]))
            entry = {
                "episode": ep,
                "chunk": k,
                "inference_ms": prediction["work_ms"],
                "inference_end": prediction["timestamp"],
                "inference_line": prediction["line"],
                "first_action_start": chunk[0]["start"],
                "last_action_end": chunk[-1]["timestamp"],
                "action_line": chunk[0]["line"],
            }
            if k > 1:
                previous = chunks[k - 1][-1]
                gap = 1000 * (chunk[0]["start"] - previous["timestamp"])
                gaps.append(gap)
                overhead.append(gap - prediction["work_ms"])
                entry.update(
                    gap_ms=gap,
                    non_inference_gap_ms=gap - prediction["work_ms"],
                    start_to_start_ms=1000
                    * (chunk[0]["start"] - chunks[k - 1][0]["start"]),
                )
            per_chunk.append(entry)
        stop = next(
            e
            for e in events
            if "[EPISODE] Stopped after" in e["text"]
            and rows[-1]["timestamp"] - 1
            < e["timestamp_second"]
            < rows[-1]["timestamp"] + 3
        )
        ack = next(
            e
            for e in events
            if "[REPLAY]" in e["text"] and f"episode={ep}," in e["text"]
        )
        ready = next(
            (
                e
                for e in events
                if "Reset complete" in e["text"] and e["line"] > ack["line"]
            ),
            None,
        )
        terminal = inf[-1]
        item = {
            "episode": ep,
            "chunks": count,
            "chunks_with_action_records": len(chunks),
            "action_records": len(rows),
            "duration_seconds": rows[-1]["timestamp"] - rows[0]["start"],
            "first_action_time": datetime.datetime.fromtimestamp(
                rows[0]["start"], datetime.timezone(datetime.timedelta(hours=8))
            ).isoformat(),
            "last_action_time": datetime.datetime.fromtimestamp(
                rows[-1]["timestamp"], datetime.timezone(datetime.timedelta(hours=8))
            ).isoformat(),
            "inference": fields(inf[:-1], ["work_ms"]),
            "terminal_inference_ms": terminal["work_ms"],
            "chunk_gap_ms": stats(gaps),
            "non_inference_gap_ms": stats(overhead),
            "chunk_execution_ms": stats(chunk_durations),
            "chunk_start_period_ms": stats(
                [1000 * (b - a) for a, b in zip(firsts, firsts[1:])]
            ),
            "action": fields(
                rows,
                [
                    "work_ms",
                    "control_ms",
                    "command_ms",
                    "pace_wait_ms",
                    "observation_ms",
                ],
            ),
            "stop_event": stop,
            "ack_event": ack,
            "stop_to_ack_seconds_rounded": ack["timestamp_second"]
            - stop["timestamp_second"],
            "last_action_to_ack_seconds_bounds": [
                max(0, ack["timestamp_second"] - rows[-1]["timestamp"]),
                ack["timestamp_second"] + 1 - rows[-1]["timestamp"],
            ],
            "terminal_inference_end_to_ack_seconds_bounds": [
                max(0, ack["timestamp_second"] - terminal["timestamp"]),
                ack["timestamp_second"] + 1 - terminal["timestamp"],
            ],
            "ack_to_reset_ready_seconds_rounded": ready["timestamp_second"]
            - ack["timestamp_second"]
            if ready
            else None,
        }
        result["episodes"].append(item)
    assert cursor == len(inferences), (cursor, len(inferences))
    result["chunk_gaps"] = {
        name: fields(
            [r for r in per_chunk if "gap_ms" in r and (r["episode"] <= 2) == before],
            ["gap_ms", "non_inference_gap_ms", "start_to_start_ms"],
        )
        for name, before in [("before_learner", True), ("after_learner", False)]
    }
    result["action_position"] = {
        str(i): fields(
            [r for r in actions if (int(r["action"]) - 1) % 3 == i - 1],
            [
                "work_ms",
                "control_ms",
                "command_ms",
                "pace_wait_ms",
                "observation_ms",
                "action_interval_ms",
            ],
        )
        for i in [1, 2, 3]
    }
    result["top_chunk_gaps"] = sorted(
        [r for r in per_chunk if "gap_ms" in r], key=lambda r: r["gap_ms"], reverse=True
    )[:20]
    result["top_inferences"] = sorted(
        inferences, key=lambda r: r["work_ms"], reverse=True
    )[:12]
    result["top_learners"] = sorted(
        [r for r in learners if r["update"] > 150],
        key=lambda r: r["work_ms"],
        reverse=True,
    )[:12]
    metric_rows = {}
    current = None
    for line in (path / "metrics.log").read_text().splitlines():
        match = re.search(r"Global Step:\s*(\d+)/", line)
        if match:
            current = int(match[1])
            metric_rows[current] = {}
        if current is not None:
            for key, value in re.findall(
                r"([a-zA-Z_][a-zA-Z_0-9/]*)=(-?[0-9]+(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?)",
                line,
            ):
                metric_rows[current][key] = float(value)
    learner_ends = [r["timestamp"] for r in learners]
    for row in runners:
        a = bisect.bisect_left(learner_ends, row["start"] - 0.001)
        b = bisect.bisect_right(learner_ends, row["timestamp"])
        calls = learners[a:b]
        row["learner_calls_found"] = len(calls)
        row["learner_ms"] = sum(x["work_ms"] for x in calls)
        metric = metric_rows.get(int(row["step"]), {})
        # Worker-side synchronization timer; does not represent the complete two-ended barrier.
        row["actor_sync_ms"] = metric.get("actor/actor/sync_model_to_rollout", None)
        if row["actor_sync_ms"] is not None:
            row["actor_sync_ms"] *= 1000
        row["metric_keys"] = list(metric) if row["actor_sync_ms"] is None else []
        row["non_learner_ms"] = row["work_ms"] - row["learner_ms"]
        row["residual_ms"] = (
            row["non_learner_ms"] - row["actor_sync_ms"]
            if row["actor_sync_ms"] is not None
            else None
        )
    result["runner_groups"] = {}
    for name, rows in {
        "warmup_steady": [r for r in runners if 1 < r["step"] < 19],
        "online_outside_action_execution": [
            r
            for r in runners
            if r["step"] >= 20
            and r["step"] != 1000
            and r["situation"] == "outside_action_execution"
        ],
        "online_active_episode": [
            r
            for r in runners
            if r["step"] >= 20
            and r["step"] != 1000
            and r["situation"] == "active_episode"
        ],
        "online_straddles_episode_boundary": [
            r
            for r in runners
            if r["step"] >= 20
            and r["step"] != 1000
            and r["situation"] == "straddles_episode_boundary"
        ],
        "checkpoint": [r for r in runners if r["step"] == 1000],
        "first_wait_for_data": [r for r in runners if r["step"] == 1],
    }.items():
        result["runner_groups"][name] = fields(
            rows,
            ["work_ms", "learner_ms", "actor_sync_ms", "non_learner_ms", "residual_ms"],
        )
    result["runner_count_mismatches"] = [
        r for r in runners if r["learner_calls_found"] != r["critic_updates"]
    ]
    result["top_runners"] = sorted(
        [r for r in runners if r["step"] > 19], key=lambda r: r["work_ms"], reverse=True
    )[:12]
    result["runner_missing_sync_metric_example"] = next(
        (r for r in runners if r["actor_sync_ms"] is None), None
    )
    result["first_inference_to_action_ms"] = {
        r["episode"]: 1000 * (r["first_action_start"] - r["inference_end"])
        for r in per_chunk
        if r["chunk"] == 1
    }
    result["old_periodic_boundaries"] = fields(
        [r for r in per_chunk if r["chunk"] in (51, 101, 151) and "gap_ms" in r],
        ["gap_ms", "inference_ms", "non_inference_gap_ms"],
    )
    checkpoint = next((r for r in runners if r["step"] == 1000), None)
    if checkpoint is not None:
        result["chunks_during_checkpoint_step"] = fields(
            [
                r
                for r in per_chunk
                if checkpoint["start"]
                <= r["first_action_start"]
                <= checkpoint["timestamp"]
                and "gap_ms" in r
            ],
            ["gap_ms", "inference_ms", "non_inference_gap_ms"],
        )
    result["max_learner_component_sum_error_ms"] = max(
        abs(r["work_ms"] - sum(r.get(k, 0) for k in learner_keys if k != "work_ms"))
        for r in learners
    )
    return result, per_chunk


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result, chunks = analyze(args.run)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    with output.with_suffix(".chunks.csv").open("w") as f:
        columns = list(dict.fromkeys(k for row in chunks for k in row))
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(chunks)
    print(
        json.dumps(
            {
                k: result[k]
                for k in [
                    "records",
                    "missing_indices",
                    "warmup_wall_seconds",
                    "actor_update_count",
                    "runner_missing_sync_metric_example",
                ]
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

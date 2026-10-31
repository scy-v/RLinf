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

"""Process-local diagnostic wrappers used only by the hardware-free startup probe."""

import functools
import inspect
import json
import threading
import time
from collections import Counter
from pathlib import Path

EVENTS = []
COUNTS = Counter()
ROLE = None
OUTPUT = Path(__file__).parent / "startup_trace_original_wait"
OUTPUT.mkdir(exist_ok=True)


def capture(name, owner, started, wall):
    group = getattr(owner, "_group_name", "")
    key = (name, group)
    COUNTS[key] += 1
    duration = time.perf_counter() - started
    if COUNTS[key] <= 3 or duration > 0.01:
        EVENTS.append(
            {
                "role": ROLE,
                "name": name,
                "group": group,
                "call": COUNTS[key],
                "start": wall,
                "end": time.time(),
                "ms": duration * 1000,
                "thread": threading.current_thread().name,
            }
        )


def wrap(cls, name):
    original = getattr(cls, name)
    if getattr(original, "_startup_probe", False):
        return
    if inspect.iscoroutinefunction(original):

        @functools.wraps(original)
        async def measured(*args, **kwargs):
            started, wall = time.perf_counter(), time.time()
            try:
                return await original(*args, **kwargs)
            finally:
                capture(cls.__name__ + "." + name, args[0], started, wall)
    else:

        @functools.wraps(original)
        def measured(*args, **kwargs):
            started, wall = time.perf_counter(), time.time()
            try:
                return original(*args, **kwargs)
            finally:
                capture(cls.__name__ + "." + name, args[0], started, wall)

    measured._startup_probe = True
    setattr(cls, name, measured)


def install(role):
    global ROLE
    ROLE = role
    from rlinf.data.storage.replay.buffer import TrajectoryReplayBuffer
    from rlinf.envs.real.env import RealWorldEnv
    from rlinf.envs.real.ur5e.base import UR5eEnv
    from rlinf.envs.real.venv import NoAutoResetSyncVectorEnv
    from rlinf.scheduler import Worker
    from rlinf.scheduler.collective.collective_group import CollectiveGroup
    from rlinf.scheduler.collective.multi_channel_pg import MultiChannelProcessGroup
    from rlinf.workers.actor import fsdp_rlt_ac_policy_worker
    from rlinf.workers.actor.fsdp_rlt_ac_policy_worker import (
        AsyncRLTACFSDPPolicy,
        RLTACReplayMixin,
    )
    from rlinf.workers.env.env_worker import EnvWorker
    from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

    wrap(fsdp_rlt_ac_policy_worker, "all_reduce_dict")
    targets = [
        (Worker, ["_get_collective_group"]),
        (
            CollectiveGroup,
            [
                "_init_process_group",
                "_atomic_send",
                "_atomic_recv",
                "_recv_tensor_dataclass",
                "_recv_tensor_list",
                "_recv_tensor_list_via_ipc",
                "_get_object_info",
            ],
        ),
        (MultiChannelProcessGroup, ["init"]),
        (AsyncRLTACFSDPPolicy, ["_drain_received_trajectories"]),
        (
            RLTACReplayMixin,
            [
                "_ingest_rollout_trajectories",
                "_transition_replay_trajectories",
                "_transition_replay_metrics",
                "_cache_warmup_transitions",
            ],
        ),
        (TrajectoryReplayBuffer, ["add_trajectories"]),
        (EnvWorker, ["_recv_actions", "env_interact_step"]),
        (MultiStepRolloutWorker, ["_send_actions"]),
        (RealWorldEnv, ["step", "_wrap_obs", "chunk_step"]),
        (UR5eEnv, ["step"]),
        (NoAutoResetSyncVectorEnv, ["step"]),
    ]
    for cls, names in targets:
        for name in names:
            wrap(cls, name)


def dump():
    (OUTPUT / ("startup_trace_" + ROLE + ".json")).write_text(
        json.dumps(EVENTS, indent=2) + "\n"
    )

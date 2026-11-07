# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""UR evaluation termination without changing generic sync or RTC workers."""

import asyncio
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rlinf.data.schema.embodied_types import EnvOutput, RTCRequest
from rlinf.envs.utils import get_env_attr
from rlinf.scheduler import Channel
from rlinf.utils.progress import ElapsedTime
from rlinf.utils.startup_timing import (
    StartupTiming,
    startup_section,
    synchronize_startup_device,
)
from rlinf.utils.ur_logging import append_record, prepare_run_logs
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.env.rtc_env_worker import RTCEnvWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker
from rlinf.workers.rollout.hf.rtc_huggingface_worker import RTCMultiStepRolloutWorker


def run_log_paths(worker: Any) -> tuple[Path, Path] | None:
    """Resolve shared run files without creating output when saving is disabled."""
    settings = (
        getattr(worker, "cfg", {})
        .get("env", {})
        .get("eval", {})
        .get("ur_evaluation", {})
    )
    if not settings.get("save_logs", True):
        return None
    log_path = (
        getattr(worker, "cfg", {}).get("runner", {}).get("logger", {}).get("log_path")
    )
    if log_path is None:
        return None
    if not hasattr(worker, "_ur_run_log_paths"):
        directory = Path(settings.get("log_dir") or Path(log_path) / "logs")
        worker._ur_run_log_paths = prepare_run_logs(directory, Path(log_path).name)
    return worker._ur_run_log_paths


def record_rtc_detail(worker: Any, component: str, event: str, **values: Any) -> None:
    """Write RTC details only when action timing and file logging are enabled.

    Action indices and rollout steps are zero-based; delays are action steps.
    """
    settings = worker.cfg.get("env", {}).get("eval", {}).get("ur_evaluation", {})
    if not settings.get("action_timing_enabled", False):
        return
    paths = run_log_paths(worker)
    if paths is not None:
        append_record(
            paths[1],
            {
                "component": component,
                "event": f"rtc_{event}",
                "timestamp": time.time(),
                "monotonic_sec": time.perf_counter(),
                **values,
            },
        )


def record_prediction(
    worker: Any, obs: dict, actions: Any, infer_index: int, **extra: Any
) -> None:
    """Save the model input state and predicted chunk under one inference ID."""
    paths = run_log_paths(worker)
    if paths is None:
        return

    def array(value: Any) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=np.float64)

    chunk = array(actions).reshape(-1, 7)
    state = array(obs["states"]).reshape(-1)
    prompts = obs.get("task_descriptions", [""])
    append_record(
        paths[0],
        {
            **extra,
            "infer_time": infer_index,
            "infer_index": infer_index,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "prompt": prompts[0] if isinstance(prompts, list) else str(prompts),
            "state": np.round(state, 6).tolist(),
            "actions": np.round(chunk, 6).tolist(),
            "actions_delta": np.round(
                np.vstack((chunk[0, :6], np.diff(chunk[:, :6], axis=0))), 6
            ).tolist(),
        },
    )


def record_timing(worker: Any, component: str, event: str, **values: Any) -> None:
    """Persist UR timings outside the action-control path."""
    paths = run_log_paths(worker)
    if paths is not None:
        append_record(
            paths[1],
            {
                "component": component,
                "event": event,
                "timestamp": time.time(),
                **values,
            },
        )
    if event not in {"step", "rtc_step"}:
        worker.log_info(f"[TIMING] {component} {event}: {values}")


class URRolloutTiming:
    """Time UR model setup before any episode begins."""

    def init_worker(self) -> None:
        torch.set_num_threads(int(self.cfg.rollout.get("torch_num_threads", 4)))
        with (
            ElapsedTime(
                "LOAD",
                "Checkpoint / model / device setup",
                self.log_info,
            ) as timer,
            StartupTiming(
                "load",
                self.log_info,
                record=lambda event, **fields: record_timing(
                    self, "rollout", event, **fields
                ),
                synchronize=synchronize_startup_device,
                profile="rlinf_ur",
            ),
        ):
            super().init_worker()
            from rlinf.algorithms.rlt.inference import Stage2Inference, is_stage2

            if is_stage2(self.cfg):
                self._stage2_policy = Stage2Inference.load(
                    self.hf_model, self.cfg.stage2
                )
            mode = self.cfg.rollout.get("sampler_compile_mode")
            if mode is not None and not is_stage2(self.cfg):
                with startup_section("model.compile_setup"):
                    self.hf_model.sample_actions = torch.compile(
                        self.hf_model.sample_actions, mode=mode
                    )
                    if self.cfg.rollout.model.openpi.rtc_enabled:
                        self.hf_model.enable_rtc_torch_compile(mode=mode)
        record_timing(self, "rollout", "load", duration_sec=timer.duration)

    @torch.inference_mode()
    def _predict_rollout_actions(self, env_obs: dict, **kwargs: Any) -> tuple:
        if getattr(self, "_stage2_policy", None) is not None:
            return self._stage2_policy.predict(env_obs, noise=kwargs.get("noise"))
        return super()._predict_rollout_actions(env_obs, **kwargs)


def evaluation_complete(env: Any) -> bool:
    """Read the UR episode counter from the single real-world environment."""
    return bool(env.unwrapped.env.envs[0].get_wrapper_attr("evaluation_complete"))


class URSyncEnvWorker(EnvWorker):
    """Run the normal synchronous policy exchange until the UR episode budget ends."""

    async def evaluate(self, input_channel: Channel, rollout_channel: Channel) -> dict:
        # Keep close RPCs responsive while policy responses or operator input block.
        return await asyncio.to_thread(self._evaluate, input_channel, rollout_channel)

    def _evaluate(self, input_channel: Channel, rollout_channel: Channel) -> dict:
        if (
            self.stage_num != 1
            or self.cfg.env.eval.total_num_envs != 1
            or self.env_decoupled_mode
        ):
            raise ValueError(
                "UR evaluation requires one environment and one coupled pipeline stage."
            )
        metrics = defaultdict(list)
        env = self.eval_env_list[0]
        try:
            env.is_start = True
            self.eval_prev_done[0] = torch.zeros(
                self.eval_num_envs_per_stage, dtype=torch.bool
            )
            obs, infos = env.reset(options={"ur_eval_prepare": True})
            self._send_ur_observation(
                rollout_channel, EnvOutput(obs=obs, env_infos=infos), warmup=True
            )
            self.recv_from(
                group_name=self.cfg.rollout.group_name,
                channel=input_channel,
                tag="eval_rollout_results",
                route_key=0,
                batch_size=self.eval_batch_size,
            )
            obs, infos = env.reset()
            output = EnvOutput(obs=obs, env_infos=infos)
            previous_episode, episode_action_count = None, 0
            for chunk_id in range(self.n_eval_chunk_steps):
                started = time.perf_counter()
                episode_index = (
                    len(env.unwrapped.env.envs[0].get_wrapper_attr("results")) + 1
                )
                episode_action_count = (
                    episode_action_count + 1 if previous_episode == episode_index else 1
                )
                previous_episode = episode_index
                self._send_ur_observation(
                    rollout_channel,
                    output,
                    infer_index=chunk_id + 1,
                    episode_index=episode_index,
                    episode_action_count=episode_action_count,
                )
                policy_output = self.recv_from(
                    group_name=self.cfg.rollout.group_name,
                    channel=input_channel,
                    tag="eval_rollout_results",
                    route_key=0,
                    batch_size=self.eval_batch_size,
                )
                actions = (
                    policy_output.actions
                    if hasattr(policy_output, "actions")
                    else policy_output
                )
                if isinstance(actions, torch.Tensor):
                    actions = actions.detach().cpu().numpy()
                received = time.perf_counter()
                env.unwrapped.env.envs[0].get_wrapper_attr("begin_action_chunk")(
                    actions.shape[1],
                    infer_index=chunk_id + 1,
                )
                output, info = self.env_evaluate_step(actions, 0)
                finished = time.perf_counter()
                elapsed = finished - started
                execution = finished - received
                record_timing(
                    self,
                    "env",
                    "step",
                    chunk_id=chunk_id,
                    infer_index=chunk_id + 1,
                    loop_sec=elapsed,
                    loop_hz=1.0 / elapsed,
                    policy_wait_sec=received - started,
                    action_and_obs_sec=execution,
                    action_hz=actions.shape[1] / execution,
                    action_steps=int(actions.shape[1]),
                    episode_boundary=bool(info),
                )
                for key, value in info.items():
                    metrics[key].append(value)
                if evaluation_complete(env):
                    break
            self._send_ur_observation(rollout_channel, output, stop=True)
            self.finish_rollout(mode="eval")
        finally:
            self.close()
        return {
            key: torch.cat(value, dim=0).contiguous().cpu()
            for key, value in metrics.items()
        }

    def _send_ur_observation(
        self,
        channel: Channel,
        output: EnvOutput,
        stop: bool = False,
        warmup: bool = False,
        infer_index: int = 0,
        episode_index: int = 0,
        episode_action_count: int = 0,
    ) -> None:
        data = self._build_rollout_input_data(output.to_dict())
        from rlinf.algorithms.rlt.inference import is_stage2

        if is_stage2(self.cfg):
            active = (
                self.eval_env_list[0]
                .unwrapped.env.envs[0]
                .get_wrapper_attr("critical_phase")
            )
            data["obs"]["rlt_switch_flags"] = torch.tensor(
                [bool(active) and not warmup]
            )
        data["ur_eval_stop"] = torch.tensor([stop])
        data["ur_eval_warmup"] = torch.tensor([warmup])
        data["ur_infer_index"] = torch.tensor([infer_index])
        data["ur_episode_index"] = torch.tensor([episode_index])
        data["ur_episode_action_count"] = torch.tensor([episode_action_count])
        self.send_to(
            group_name=self.cfg.rollout.group_name,
            channel=channel,
            data=data,
            split_fn=self._obs_split_fn,
            mode="eval",
            tag="rollout_results",
            route_key=0,
        )


class URSyncRolloutWorker(URRolloutTiming, MultiStepRolloutWorker):
    """Use the existing model predictor with an explicit UR stop message."""

    def _merge_obs_batches(self, obs_batches: list[dict[str, Any]]) -> dict[str, Any]:
        merged = super()._merge_obs_batches(obs_batches)
        if obs_batches:
            for key in (
                "ur_infer_index",
                "ur_episode_index",
                "ur_episode_action_count",
            ):
                merged[key] = torch.cat(
                    [batch.get(key, torch.tensor([0])) for batch in obs_batches], dim=0
                )
            merged["ur_eval_stop"] = torch.cat(
                [batch["ur_eval_stop"] for batch in obs_batches], dim=0
            )
            merged["ur_eval_warmup"] = torch.cat(
                [
                    batch.get("ur_eval_warmup", torch.tensor([False]))
                    for batch in obs_batches
                ],
                dim=0,
            )
        return merged

    async def evaluate(self, input_channel: Channel, output_channel: Channel) -> None:
        infer_index = 0
        if self.enable_offload:
            self.reload_model()
        try:
            while True:
                output = await self.recv_from(
                    group_name=self.cfg.env.group_name,
                    channel=input_channel,
                    tag="eval_rollout_results",
                    route_key=0,
                    async_op=True,
                    batch_size=self.eval_batch_size,
                    merge_fn=self._merge_obs_batches,
                    infer_batch_size_fn=self._infer_env_batch_size,
                ).async_wait()
                if bool(output["ur_eval_stop"].any()):
                    break
                warmup = bool(output["ur_eval_warmup"].any())
                started = time.perf_counter()
                if warmup:
                    with (
                        ElapsedTime(
                            "WARMUP",
                            "First inference / compilation",
                            self.log_info,
                        ),
                        StartupTiming(
                            "warmup",
                            self.log_info,
                            record=lambda event, **fields: record_timing(
                                self, "rollout", event, **fields
                            ),
                            synchronize=synchronize_startup_device,
                            profile="rlinf_ur",
                        ),
                    ):
                        actions, _ = self._predict_rollout_actions(
                            output["obs"], mode="eval"
                        )
                        if isinstance(actions, torch.Tensor):
                            actions = actions.detach().cpu().contiguous()
                    record_timing(
                        self,
                        "rollout",
                        "warmup",
                        duration_sec=time.perf_counter() - started,
                    )
                else:
                    actions, _ = self._predict_rollout_actions(
                        output["obs"],
                        mode="eval",
                        final_obs=output.get("final_obs"),
                    )
                if isinstance(actions, torch.Tensor):
                    actions = actions.detach().cpu().contiguous()
                if not warmup:
                    elapsed = time.perf_counter() - started
                    infer_index = int(output["ur_infer_index"][0]) or infer_index + 1
                    record_prediction(
                        self,
                        output["obs"],
                        actions,
                        infer_index,
                        episode_index=int(output["ur_episode_index"][0]),
                        episode_action_count=int(output["ur_episode_action_count"][0]),
                        action_horizon=int(actions.shape[1]),
                        controller="mlp"
                        if bool(
                            output["obs"]
                            .get("rlt_switch_flags", torch.tensor([False]))
                            .any()
                        )
                        else "vla",
                    )
                    record_timing(
                        self,
                        "rollout",
                        "step",
                        infer_index=infer_index,
                        infer_sec=elapsed,
                        infer_hz=1.0 / elapsed,
                    )
                self.send_to(
                    group_name=self.cfg.env.group_name,
                    channel=output_channel,
                    data=actions,
                    tag="eval_rollout_results",
                    route_key=0,
                    async_op=True,
                    batch_size=self.eval_batch_size,
                )
        finally:
            if self.enable_offload:
                self.offload_model()


class URRTCEnvWorker(RTCEnvWorker):
    """Retain RTC replanning and stop after the final UR outcome."""

    def _record_rtc_event(self, event: str, **values: Any) -> None:
        record_rtc_detail(self, "env", event, **values)

    def init_worker(self) -> None:
        super().init_worker()
        for env in self.eval_env_list:
            env.manual_episode_control_only = env.unwrapped.manual_episode_control_only
            env.unwrapped.env.envs[0].get_wrapper_attr("defer_camera_refresh")()

    def send_rtc_request(
        self, rollout_channel: Channel, rtc_request: RTCRequest, mode: str = "eval"
    ) -> None:
        """Acquire new images only when dispatching a replan; reset supplies bootstrap."""
        if rtc_request.request_type == "replan":
            started = time.perf_counter()
            refresh = get_env_attr(self.eval_env_list[0], "refresh_camera_observation")
            rtc_request.obs = refresh(rtc_request.obs)
            self._record_rtc_event(
                "observation",
                chunk_id=rtc_request.chunk_id,
                observation_sec=time.perf_counter() - started,
            )
        super().send_rtc_request(rollout_channel, rtc_request, mode)

    def _evaluate_rtc_action(self, env_action: Any, stage_id: int) -> tuple:
        started = time.perf_counter()
        output, info = super()._evaluate_rtc_action(env_action, stage_id)
        env = self.eval_env_list[stage_id]
        now = time.perf_counter()
        since = getattr(self, "_ur_rtc_since", started)
        count = getattr(self, "_ur_rtc_count", 0) + 1
        duration = getattr(self, "_ur_rtc_duration", 0.0) + now - started
        if now - since >= 1.0 or evaluation_complete(env):
            record_timing(
                self,
                "env",
                "rtc_step",
                action_steps=count,
                action_sec=duration / count,
                action_hz=count / (now - since),
            )
            since, count, duration = now, 0, 0.0
        self._ur_rtc_since, self._ur_rtc_count, self._ur_rtc_duration = (
            since,
            count,
            duration,
        )
        if evaluation_complete(env):
            # The inherited RTC loop drains pending requests and sends its normal stop.
            env.manual_episode_control_only = False
        return output, info

    async def evaluate(self, input_channel: Channel, rollout_channel: Channel) -> dict:
        # Keep close RPCs responsive while policy responses or operator input block.
        return await asyncio.to_thread(self._evaluate, input_channel, rollout_channel)

    def _evaluate(self, input_channel: Channel, rollout_channel: Channel) -> dict:
        try:
            obs, _ = self.eval_env_list[0].reset(options={"ur_eval_prepare": True})
            self.send_rtc_request(
                rollout_channel,
                RTCRequest(obs=obs, request_type="warmup"),
                mode="eval",
            )
            self.recv_rtc_response(input_channel, mode="eval", async_op=False)
            return super().evaluate(input_channel, rollout_channel)
        finally:
            self.close()


class URRTCRolloutWorker(URRolloutTiming, RTCMultiStepRolloutWorker):
    """Warm up async inference before operator-controlled episodes."""

    show_eval_progress = False

    @torch.inference_mode()
    def _predict_rtc(self, request: RTCRequest) -> Any:
        started = time.perf_counter()
        if request.request_type == "warmup":
            sampling_mode = (
                "RTC guidance"
                if self.cfg.rollout.model.openpi.rtc_enabled
                else "ordinary sampling"
            )
            self.log_info(f"Async inference enabled; {sampling_mode}.")
            with (
                ElapsedTime(
                    "WARMUP",
                    "First inference / compilation",
                    self.log_info,
                ),
                StartupTiming(
                    "warmup",
                    self.log_info,
                    record=lambda event, **fields: record_timing(
                        self, "rollout", event, **fields
                    ),
                    synchronize=synchronize_startup_device,
                    profile="rlinf_ur_async",
                ),
            ):
                try:
                    response = super()._predict_rtc(request)
                    if self.cfg.rollout.model.openpi.rtc_enabled:
                        rtc_cfg = self.cfg.runner.get(
                            "async_inference", self.cfg.runner.get("rtc", {})
                        )
                        # Replanning uses a different sampler from bootstrap inference.
                        super()._predict_rtc(
                            RTCRequest(
                                obs=request.obs,
                                request_type="replan",
                                executed_horizon=int(rtc_cfg.min_exec_horizon),
                                predicted_delay_steps=int(rtc_cfg.initial_delay_steps),
                            )
                        )
                finally:
                    self._rtc_eval_model_actions = None
            record_timing(
                self, "rollout", "warmup", duration_sec=time.perf_counter() - started
            )
            return response
        response = super()._predict_rtc(request)
        elapsed = time.perf_counter() - started
        record_rtc_detail(
            self,
            "rollout",
            "prediction",
            request_type=request.request_type,
            sampling_mode="rtc" if response.guidance_applied else "ordinary",
            chunk_id=request.chunk_id,
            executed_horizon=request.executed_horizon,
            predicted_delay_steps=request.predicted_delay_steps,
            infer_sec=elapsed,
        )
        infer_index = getattr(self, "_ur_infer_index", 0) + 1
        self._ur_infer_index = infer_index
        record_prediction(
            self, request.obs, response.actions, infer_index, chunk_id=request.chunk_id
        )
        record_timing(
            self,
            "rollout",
            "step",
            infer_index=infer_index,
            chunk_id=request.chunk_id,
            infer_sec=elapsed,
            infer_hz=1.0 / elapsed,
        )
        return response

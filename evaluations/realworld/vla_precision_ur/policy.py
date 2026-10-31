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

"""Expose RLinf inference through the standalone UR backend's Policy interface."""

from __future__ import annotations

import copy
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rlinf.utils.logging import get_logger
from rlinf.utils.progress import ElapsedTime
from rlinf.utils.startup_timing import (
    StartupTiming,
    startup_section,
    synchronize_startup_device,
)


def warmup_policy(
    policy: Any,
    observation: dict,
    *,
    timing_recorder: Callable[..., None] | None = None,
    **kwargs: Any,
) -> dict:
    """Run first inference with a live elapsed-time display and stop it on exit."""
    with ElapsedTime("WARMUP", "First inference / compilation") as timer:
        with StartupTiming(
            "warmup",
            get_logger().info,
            record=timing_recorder,
            synchronize=synchronize_startup_device,
            component="vla_precision",
        ):
            result = policy.infer(observation, **kwargs)
    if timing_recorder is not None:
        timing_recorder("warmup", duration_sec=timer.duration)
    return result


def _torch_compile_setting(config: dict[str, Any]) -> bool | str:
    """Resolve the UR sampler's legacy boolean or OpenPI compile mode."""
    setting = config["model"]["network"]["openpi"].get("torch_compile", True)
    if isinstance(setting, bool):
        return setting
    modes = ("default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs")
    if isinstance(setting, str) and setting in modes:
        return setting
    raise ValueError(
        "UR openpi.torch_compile must be a YAML boolean or one of "
        f"{modes}; got {setting!r}"
    )


def _configure_sampler_compile(model: torch.nn.Module, setting: bool | str) -> None:
    """Compile the complete sampler only for an explicit mode string."""
    from rlinf.utils.logging import get_logger

    if isinstance(setting, str):
        model.sample_actions = torch.compile(model.sample_actions, mode=setting)
        get_logger().info(
            "UR sampler torch.compile mode=%s; first inference compiles and warms up the model",
            setting,
        )
    else:
        get_logger().info(
            "UR sampler uses the original Python loop; local kernel compilation=%s",
            setting,
        )


def load_norm_stats(path: str | Path) -> dict:
    """Load the explicit checkpoint statistics using OpenPI's serialization."""
    from openpi.shared import normalize

    path = Path(path).expanduser()
    return normalize.deserialize_json(path.read_text())


def model_config_from_yaml(config: dict[str, Any]) -> Any:
    """Build a model configuration without starting a Ray cluster or hardware."""
    from omegaconf import OmegaConf

    model = config["model"]
    network = copy.deepcopy(model["network"])
    # The shared loader accepts booleans; complete-sampler modes belong to UR.
    network["openpi"]["torch_compile"] = _torch_compile_setting(config) is not False
    if config["policy"].get("location", "local") != "local":
        raise ValueError("This comparison entry requires policy.location=local")
    if config["robot"].get("fix_gripper", False):
        raise ValueError(
            "This 19-state/7-action checkpoint requires robot.fix_gripper=false"
        )
    if network["openpi"].get("rtc_enabled", False):
        raise ValueError(
            "The VLA-Precision comparison uses synchronous inference without RTC"
        )
    if network["openpi"]["action_env_dim"] != 7 or network["action_dim"] != 7:
        raise ValueError(
            "The VLA-Precision single-UR backend requires seven action dimensions"
        )
    network["model_path"] = str(Path(model["checkpoint_dir"]).expanduser())
    network["num_steps"] = int(model["sample_steps"])
    network["openpi_data"]["norm_stats_path"] = str(
        Path(model["norm_stats_path"]).expanduser()
    )
    cfg = OmegaConf.create(network)
    chunk_size = int(cfg.openpi.get("action_chunk", cfg.num_action_chunks))
    prediction_steps = int(cfg.openpi.action_horizon)
    if not 1 <= chunk_size <= prediction_steps:
        raise ValueError(
            f"Expected openpi.action_chunk between 1 and action_horizon "
            f"{prediction_steps}, got {chunk_size}."
        )
    return cfg


class RLinfPolicy:
    """Return RLinf's decoded action chunk to the standalone UR backend."""

    def __init__(self, model: torch.nn.Module):
        self.model = model

    @torch.inference_mode()
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:
        env_obs = {
            "states": torch.from_numpy(
                np.asarray(obs["observation/state"])[None].copy()
            ),
            "main_images": torch.from_numpy(
                np.asarray(obs["observation/image"])[None].copy()
            ),
            "wrist_images": torch.from_numpy(
                np.asarray(obs["observation/wrist_image"])[None].copy()
            ),
            "task_descriptions": [obs["prompt"]],
        }
        noise_tensor = None
        if noise is not None:
            noise = np.asarray(noise)
            if noise.ndim == 2:
                noise = noise[None]
            noise_tensor = torch.as_tensor(noise, device=self.model.device)
        started = time.perf_counter()
        actions, _ = self.model.predict_action_batch(
            env_obs, mode="eval", noise=noise_tensor
        )
        with startup_section("inference.to_cpu"):
            actions = actions[0].detach().to(device="cpu", dtype=torch.float32).numpy()
        return {
            "actions": actions,
            "policy_timing": {"infer_ms": (time.perf_counter() - started) * 1000},
        }


def create_policy(
    config: dict[str, Any], *, timing_recorder: Callable[..., None] | None = None
) -> Any:
    """Load the complete policy with an elapsed-time display."""
    with ElapsedTime("LOAD", "Checkpoint / model / device setup") as timer:
        with StartupTiming(
            "load",
            get_logger().info,
            record=timing_recorder,
            synchronize=synchronize_startup_device,
            component="vla_precision",
        ):
            policy = _create_policy(config)
    if timing_recorder is not None:
        timing_recorder("load", duration_sec=timer.duration)
    return policy


def _create_policy(config: dict[str, Any]) -> Any:
    """Use RLinf's model factory and default evaluation input/output processing."""
    with startup_section("policy.imports"):
        from rlinf.models.embodiment.openpi import get_model

    with startup_section("policy.config"):
        cfg = model_config_from_yaml(config)
    torch.set_num_threads(int(config["model"].get("torch_num_threads", 4)))
    get_logger().info(
        "Loading RLinf checkpoint for standalone UR inference: %s",
        cfg.model_path,
    )
    model = get_model(cfg)
    with startup_section("model.to_device"):
        model = model.to(str(config["model"].get("device", "cuda:0"))).eval()
    with startup_section("model.compile_setup"):
        _configure_sampler_compile(model, _torch_compile_setting(config))
    return RLinfPolicy(model)


def check_model(config: dict[str, Any]) -> None:
    """Compare standalone and RLinf worker I/O with fixed noise, without hardware."""
    from rlinf.utils.logging import get_logger

    policy = create_policy(config)
    model = policy.model
    rng = np.random.default_rng(42)
    obs = {
        "observation/state": rng.normal(0, 0.01, 19).astype(np.float32),
        "observation/image": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
        "prompt": config["task"]["description"],
    }
    noise = rng.normal(size=(model.action_horizon, model.action_dim)).astype(np.float32)
    standalone_actions = warmup_policy(
        policy,
        obs,
        noise=noise,
    )["actions"]
    env_obs = {
        "states": torch.from_numpy(obs["observation/state"][None].copy()),
        "main_images": torch.from_numpy(obs["observation/image"][None].copy()),
        "wrist_images": torch.from_numpy(obs["observation/wrist_image"][None].copy()),
        "task_descriptions": [obs["prompt"]],
    }
    with torch.inference_mode():
        original_actions, _ = model.predict_action_batch(
            env_obs, mode="eval", noise=torch.from_numpy(noise[None]).to(model.device)
        )
    actual = original_actions[0].detach().cpu().numpy()
    if (
        standalone_actions.shape != actual.shape
        or not np.isfinite(standalone_actions).all()
    ):
        raise ValueError(
            f"Expected finite action chunk shape {actual.shape}, got {standalone_actions.shape}"
        )
    np.testing.assert_allclose(actual, standalone_actions, rtol=1e-5, atol=1e-6)
    get_logger().info(
        "RLinf Policy GPU check passed: action chunk shape=%s, prediction horizon=%s, max abs diff=%.8g; no hardware connected",
        standalone_actions.shape,
        model.action_horizon,
        float(np.max(np.abs(actual - standalone_actions))),
    )

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

"""Expose RLinf checkpoint weights through OpenPI's native Policy interface."""

from __future__ import annotations

import copy
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


def warmup_policy(policy: Any, observation: dict, **kwargs: Any) -> dict:
    """Run first inference with a live elapsed-time display and stop it on exit."""
    stream = sys.stderr
    start = time.monotonic()
    stopped = threading.Event()

    def display() -> None:
        elapsed = time.monotonic() - start
        stream.write(
            f"\r[WARMUP] First inference / compilation: {elapsed:.1f}s elapsed"
        )
        stream.flush()

    def update() -> None:
        while not stopped.wait(1.0):
            display()

    display()
    timer = threading.Thread(target=update, name="ur-warmup-timer", daemon=True)
    timer.start()
    completed = False
    try:
        result = policy.infer(observation, **kwargs)
        completed = True
        return result
    finally:
        stopped.set()
        timer.join()
        elapsed = time.monotonic() - start
        status = "Completed" if completed else "Stopped before completion"
        stream.write(f"\r[WARMUP] {status} after {elapsed:.2f}s".ljust(90) + "\n")
        stream.flush()


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
    if config["robot"].get("direct_action_horizon", False):
        raise ValueError(
            "Keep robot.direct_action_horizon=false to preserve the trained horizon"
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
    return OmegaConf.create(network)


class NativeModelAdapter(torch.nn.Module):
    """Adapt OpenPI's (device, observation) sampler call to the RLinf network."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    @torch.inference_mode()
    def sample_actions(
        self, device: str, observation: Any, **kwargs: Any
    ) -> torch.Tensor:
        from rlinf.models.embodiment.openpi.modules.model import Observation

        del device
        return self.model.sample_actions(
            Observation.from_observation_like(observation), **kwargs
        )


def create_policy(config: dict[str, Any]) -> Any:
    """Load RLinf weights and retain VLA-Precision's native OpenPI transforms."""
    from openpi import transforms
    from openpi.models.pi0_config import Pi0Config
    from openpi.policies.policy import Policy
    from openpi.training.config import ModelTransformFactory

    from rlinf.models.embodiment.openpi import get_model
    from rlinf.utils.logging import get_logger

    from .ur_policy import UR5eInputs, UR5eOutputs

    cfg = model_config_from_yaml(config)
    torch.set_num_threads(int(config["model"].get("torch_num_threads", 4)))
    get_logger().info(
        "Loading RLinf checkpoint for native VLA-Precision UR inference: %s",
        cfg.model_path,
    )
    model = get_model(cfg)
    _configure_sampler_compile(model, _torch_compile_setting(config))
    native_config = Pi0Config(
        pi05=bool(cfg.pi05),
        action_dim=int(cfg.openpi.model_action_dim),
        action_horizon=int(cfg.openpi.action_horizon),
        max_token_len=int(cfg.openpi.max_token_len),
        paligemma_variant=str(cfg.openpi.paligemma_variant),
        action_expert_variant=str(cfg.openpi.action_expert_variant),
    )
    model_transforms = ModelTransformFactory()(native_config)
    norm_stats = load_norm_stats(config["model"]["norm_stats_path"])
    return Policy(
        NativeModelAdapter(model),
        transforms=[
            transforms.InjectDefaultPrompt(None),
            UR5eInputs(model_type=native_config.model_type),
            transforms.Normalize(norm_stats, use_quantiles=bool(cfg.pi05)),
            *model_transforms.inputs,
        ],
        output_transforms=[
            *model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=bool(cfg.pi05)),
            UR5eOutputs(),
        ],
        sample_kwargs={"num_steps": int(config["model"]["sample_steps"])},
        is_pytorch=True,
        pytorch_device=str(config["model"].get("device", "cuda:0")),
    )


def check_model(config: dict[str, Any]) -> None:
    """Compare native Policy and RLinf env I/O with fixed noise, without hardware."""
    from rlinf.utils.logging import get_logger

    policy = create_policy(config)
    model = policy._model.model
    rng = np.random.default_rng(42)
    obs = {
        "observation/state": rng.normal(0, 0.01, 19).astype(np.float32),
        "observation/image": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
        "prompt": config["task"]["description"],
    }
    noise = rng.normal(size=(model.action_horizon, model.action_dim)).astype(np.float32)
    native_actions = warmup_policy(policy, obs, noise=noise)["actions"]
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
        native_actions.shape != (model.action_horizon, 7)
        or not np.isfinite(native_actions).all()
    ):
        raise ValueError(
            f"Expected finite ({model.action_horizon}, 7) actions, got {native_actions.shape}"
        )
    expected = native_actions[: len(actual)]
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    get_logger().info(
        "Native Policy GPU check passed: full action shape=%s, chunk=%s, max abs diff=%.8g; no hardware connected",
        native_actions.shape,
        actual.shape,
        float(np.max(np.abs(actual - expected))),
    )

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
import dataclasses
import pathlib

import numpy as np
import openpi.models.model as _model
import openpi.transforms as transforms
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory

DEFAULT_OBS_INDICES = tuple(range(1, 20))


def _parse_ur_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = image.transpose(1, 2, 0)
    return image


@dataclasses.dataclass(frozen=True)
class URInputs(transforms.DataTransformFn):
    """Map raw UR TCP observations and delta actions to OpenPI inputs."""

    action_dim: int = 32

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["observation/state"])
        if state.shape[-1] > self.action_dim:
            raise ValueError(
                f"Expected at most {self.action_dim} state dimensions, got {state.shape[-1]}"
            )
        base = _parse_ur_image(data["observation/image"])
        wrist = _parse_ur_image(data["observation/wrist_image"])
        result = {
            "state": state,
            "image": {
                "base_0_rgb": base,
                "left_wrist_0_rgb": wrist,
                "right_wrist_0_rgb": np.zeros_like(base),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_,
            },
        }
        if "actions" in data:
            actions = np.asarray(data["actions"])
            if actions.shape[-1] != 7:
                raise ValueError(
                    f"Expected 7 UR action dimensions, got {actions.shape}"
                )
            result["actions"] = transforms.pad_to_dim(actions, self.action_dim)
        if "prompt" in data:
            result["prompt"] = data["prompt"]
        return result


@dataclasses.dataclass(frozen=True)
class UROutputs(transforms.DataTransformFn):
    """Return the seven UR delta-pose and gripper action dimensions."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])[..., :7]}


@dataclasses.dataclass(frozen=True)
class LeRobotURDataConfig(DataConfigFactory):
    """UR5e LeRobot data with configurable one-based raw state indices."""

    obs_indices: tuple[int, ...] = DEFAULT_OBS_INDICES
    default_prompt: str | None = None

    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        base = self.create_base_config(assets_dirs, model_config)
        return dataclasses.replace(
            base,
            action_sequence_keys=("action",),
            repack_transforms=transforms.Group(
                inputs=[
                    transforms.RepackTransform(
                        {
                            "observation/image": "observation.images.exterior_image",
                            "observation/wrist_image": "observation.images.wrist_image",
                            "observation/state": "observation.state",
                            "actions": "action",
                            "prompt": "task",
                        }
                    )
                ]
            ),
            data_transforms=transforms.Group(
                inputs=[URInputs(model_config.action_dim)],
                outputs=[UROutputs()],
            ),
            model_transforms=ModelTransformFactory(default_prompt=self.default_prompt)(
                model_config
            ),
        )

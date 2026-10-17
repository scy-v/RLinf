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

import json
import os
import pathlib
from collections.abc import Iterator
from typing import Literal

import numpy as np
import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms
import tqdm
import tyro
from openpi.training.config import DataConfig, TrainConfig

from rlinf.data.storage.lerobot import resolve_lerobot_dataset_root
from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
from rlinf.utils.logging import get_logger

logger = get_logger()


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {
            k: v
            for k, v in x.items()
            if not np.issubdtype(np.asarray(v).dtype, np.str_)
        }


def create_torch_dataloader(
    data_config: DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.TorchDataLoader, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(
        data_config, action_horizon, model_config
    )
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(
        data_config, action_horizon, batch_size, shuffle=False
    )
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_parquet_batches(
    dataset_root: pathlib.Path, data_config: DataConfig, config: TrainConfig
) -> Iterator[dict[str, np.ndarray]]:
    """Compute numeric inputs from local LeRobot parquet using the configured transforms.

    This statistics-only reader accepts v2/v3 storage and clamps action windows
    at episode boundaries. Images and task text are placeholders; only numeric
    state/action transforms contribute to the statistics.
    """
    import pyarrow.parquet as pq

    files = sorted((dataset_root / "data").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"Expected data parquet files under {dataset_root}")
    info = json.loads((dataset_root / "meta/info.json").read_text())
    image_keys = {
        key
        for key, feature in info["features"].items()
        if feature.get("dtype") in ("image", "video")
    }
    columns = [key for key in pq.read_schema(files[0]).names if key not in image_keys]
    table = pq.read_table(files, columns=columns)
    arrays = {}
    for key in columns:
        values = np.asarray(table[key].to_pylist())
        if values.dtype.kind in "iufb":
            if not np.isfinite(values).all():
                raise ValueError(f"Expected finite dataset values in {key}")
            if values.dtype.kind == "f":
                values = values.astype(np.float64)
        arrays[key] = values
    episodes = arrays["episode_index"]
    frames = arrays["frame_index"]
    count = len(episodes)
    if (
        not count
        or count != info["total_frames"]
        or len(np.unique(episodes)) != info["total_episodes"]
    ):
        raise ValueError(
            "Expected nonempty dataset frames and episodes matching meta/info.json"
        )
    for key in data_config.action_sequence_keys:
        if key not in arrays:
            raise ValueError(
                f"Expected configured action sequence column {key!r}, got {columns}"
            )
    ends = np.empty(count, dtype=np.int64)
    boundaries = np.r_[0, np.flatnonzero(np.diff(episodes)) + 1, count]
    if len(boundaries) - 1 != info["total_episodes"]:
        raise ValueError("Expected each episode to occupy one contiguous frame range")
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        if not np.array_equal(frames[start:end], np.arange(end - start)):
            raise ValueError(
                f"Expected ordered frames starting at zero in episode {episodes[start]}"
            )
        ends[start:end] = end - 1
    transform = transforms.compose(
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            RemoveStrings(),
        ]
    )
    images = {}
    for key in image_keys:
        feature = info["features"][key]
        shape = (3, 2, 2) if (feature.get("shape") or [None])[0] == 3 else (2, 2, 3)
        images[key] = np.zeros(shape, dtype=np.uint8)
    for start in range(0, count, config.batch_size):
        samples = []
        for index in range(start, min(start + config.batch_size, count)):
            sample = {key: values[index] for key, values in arrays.items()}
            window = np.minimum(
                index + np.arange(config.model.action_horizon), ends[index]
            )
            for key in data_config.action_sequence_keys:
                sample[key] = arrays[key][window]
            sample.update(images)
            sample.setdefault("prompt", "")
            sample.setdefault("task", "")
            samples.append(transform(sample))
        yield {
            key: np.stack([sample[key] for sample in samples])
            for key in ("state", "actions")
        }


def main(
    config_name: str,
    repo_id: str,
    obs_indices: str | None = None,
    output_dir: str | None = None,
    data_backend: Literal["auto", "lerobot", "parquet"] = "auto",
) -> None:
    """Compute stats with optional comma-separated, one-based observation indices."""
    dataset_root = resolve_lerobot_dataset_root(repo_id)
    if not (dataset_root / "meta" / "info.json").is_file():
        raise FileNotFoundError(
            f"LeRobot dataset not found for repo_id={repo_id!r} at {dataset_root}. "
            "Pass a local dataset path or a cached Hugging Face repo id."
        )
    indices = obs_indices if obs_indices is not None else os.getenv("OBS_INDICES")
    data_kwargs = None
    if indices is not None:
        data_kwargs = {"obs_indices": tuple(int(i.strip()) for i in indices.split(","))}
    config = get_openpi_config(config_name, repo_id=repo_id, data_kwargs=data_kwargs)
    data_config = config.data.create(config.assets_dirs, config.model)

    info = json.loads((dataset_root / "meta/info.json").read_text())
    use_parquet = data_backend == "parquet" or (
        data_backend == "auto"
        and str(info.get("codebase_version", "")).startswith("v3.")
    )
    if use_parquet:
        if data_config.rlds_data_dir is not None:
            raise ValueError(
                "Expected a LeRobot config for the parquet backend, got RLDS"
            )
        data_loader = create_parquet_batches(dataset_root, data_config, config)
        num_batches = (
            info["total_frames"] + config.batch_size - 1
        ) // config.batch_size
    elif data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config,
            config.model.action_horizon,
            config.batch_size,
            config.model,
            config.num_workers,
        )

    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in stats:
            stats[key].update(np.asarray(batch[key]))
    norm_stats = {key: value.get_statistics() for key, value in stats.items()}
    output_path = (
        pathlib.Path(output_dir).expanduser()
        if output_dir
        else pathlib.Path(config.assets_dirs) / data_config.repo_id
    )
    logger.info("Writing stats to %s", output_path)
    normalize.save(output_path, norm_stats)
    selection = getattr(config.data, "obs_indices", None)
    metadata = {
        "config_name": config_name,
        "dataset_root": str(dataset_root),
        "robot_type": info.get("robot_type"),
        "obs_indices": list(selection) if selection is not None else None,
        "total_frames": info["total_frames"],
        "total_episodes": info["total_episodes"],
        "action_horizon": config.model.action_horizon,
        "model_action_dim": config.model.action_dim,
        "data_backend": "parquet" if use_parquet else "lerobot",
    }
    for repack in data_config.repack_transforms.inputs:
        if not isinstance(repack, transforms.RepackTransform):
            continue
        for target, field in (
            ("observation/state", "state_names"),
            ("actions", "action_names"),
        ):
            source = repack.structure.get(target)
            if not isinstance(source, str):
                continue
            names = info["features"].get(source, {}).get("names")
            if isinstance(names, list):
                metadata[field] = (
                    [names[i - 1] for i in selection]
                    if target == "observation/state" and selection is not None
                    else names
                )
    (output_path / "norm_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )


if __name__ == "__main__":
    tyro.cli(main)

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

"""Repack numeric NPZ files losslessly, preserving the numpy.load interface."""

import argparse
import hashlib
import io
import json
import os
import tempfile
import zipfile
from pathlib import Path

import numpy as np


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repack(path):
    """Verify all array bytes before atomically replacing a smaller archive."""
    path = Path(path)
    before_size, before_hash = path.stat().st_size, file_hash(path)
    fd, name = tempfile.mkstemp(prefix=".repack-", suffix=".npz", dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        with np.load(path, allow_pickle=False) as original:
            with zipfile.ZipFile(
                temporary, "w", compression=zipfile.ZIP_LZMA
            ) as archive:
                for key in original.files:
                    array = original[key]
                    if array.ndim > 1:
                        array = np.asfortranarray(array)
                    stream = io.BytesIO()
                    np.save(stream, array, allow_pickle=False)
                    archive.writestr(key + ".npy", stream.getvalue())
            with np.load(temporary, allow_pickle=False) as packed:
                if original.files != packed.files:
                    raise ValueError(f"NPZ keys changed: {path}")
                for key in original.files:
                    left, right = original[key], packed[key]
                    if (
                        left.shape != right.shape
                        or left.dtype != right.dtype
                        or left.tobytes(order="C") != right.tobytes(order="C")
                    ):
                        raise ValueError(
                            f"Array changed during compression: {path}:{key}"
                        )
        replaced = temporary.stat().st_size < before_size
        if file_hash(path) != before_hash:
            raise RuntimeError(f"Source changed while compressing: {path}")
        if replaced:
            temporary.chmod(path.stat().st_mode & 0o777)
            os.replace(temporary, path)
        return {
            "path": str(path),
            "before_bytes": before_size,
            "after_bytes": path.stat().st_size,
            "before_sha256": before_hash,
            "after_sha256": file_hash(path),
            "replaced": replaced,
            "verification": "same keys, shapes, dtypes and C-order array bytes",
            "encoding": "ZIP_LZMA; multidimensional NPY arrays in Fortran order",
        }
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", type=Path, nargs="+")
    args = parser.parse_args()
    for source in args.paths:
        print(json.dumps(repack(source), ensure_ascii=False), flush=True)

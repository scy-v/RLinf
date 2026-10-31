#!/usr/bin/env bash
# Run from the repository root. No robot connections or global cache eviction.
set -euo pipefail
exec "${RLT_BENCH_PYTHON:-.venv/bin/python}" - "$@" <<'PY'
import ctypes
import datetime
import json
import logging
import mmap
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import yaml

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger('startup_cache')
root = Path.cwd()
warm_runs = int(sys.argv[1]) if len(sys.argv) > 1 else 2
if warm_runs < 1:
    raise ValueError('Expected at least one cached fresh-process run')
config_path = root / 'evaluations/realworld/vla_precision_ur.yaml'
config = yaml.safe_load(config_path.read_text())
checkpoint_root = Path(config['model']['checkpoint_dir'])
checkpoint = next(p for p in (
    checkpoint_root / 'model_state_dict/full_weights.pt',
    checkpoint_root / 'actor/model_state_dict/full_weights.pt',
    checkpoint_root / 'full_weights.pt',
) if p.is_file())
run_id = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
output = root / 'rltoken-setup/support/benchmarks/20261031/startup_cache' / run_id
output.mkdir(parents=True)
(output / 'config.yaml').write_text(config_path.read_text())
cache_root = Path(tempfile.mkdtemp(prefix='rlt-inference-cache-'))
for name in ('inductor', 'triton', 'cuda'):
    (cache_root / name).mkdir()
libc = ctypes.CDLL(None, use_errno=True)
libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_ubyte)]
libc.mincore.restype = ctypes.c_int


def residency():
    size = checkpoint.stat().st_size
    count = (size + mmap.PAGESIZE - 1) // mmap.PAGESIZE
    pages = (ctypes.c_ubyte * count)()
    with checkpoint.open('rb') as stream:
        with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_COPY) as mapping:
            address = ctypes.addressof(ctypes.c_char.from_buffer(mapping))
            if libc.mincore(address, size, pages) != 0:
                raise OSError(ctypes.get_errno(), 'mincore failed')
    cached = sum(page & 1 for page in pages)
    return {'cached_pages': cached, 'pages': count, 'cached_fraction': cached / count}


child = r'''
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
import yaml
from evaluations.realworld.vla_precision_ur.policy import create_policy, warmup_policy

output = Path(sys.argv[1])
config = yaml.safe_load(Path(sys.argv[2]).read_text())
iteration = 0

def record(event, **fields):
    with output.open('a') as stream:
        stream.write(json.dumps({'event': event, 'iteration': iteration, **fields}) + '\n')

torch.manual_seed(1234)
policy = create_policy(config, timing_recorder=record)
observation = {
    'observation/state': np.zeros(20, dtype=np.float32),
    'observation/image': np.zeros((224, 224, 3), dtype=np.uint8),
    'observation/wrist_image': np.zeros((224, 224, 3), dtype=np.uint8),
    'prompt': config['task']['description'],
}
for iteration in (1, 2, 3, 4):
    result = warmup_policy(policy, observation, timing_recorder=record)
    assert result['actions'].shape == (int(policy.model.action_chunk), 7)
    assert np.isfinite(result['actions']).all()
'''

env = dict(os.environ)
env.update({
    'PYTHONPATH': str(root),
    'TORCHINDUCTOR_CACHE_DIR': str(cache_root / 'inductor'),
    'TRITON_CACHE_DIR': str(cache_root / 'triton'),
    'CUDA_CACHE_PATH': str(cache_root / 'cuda'),
    'TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE': '0',
    'TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE': '0',
    'TORCHINDUCTOR_BUNDLED_AUTOTUNE_REMOTE_CACHE': '0',
})
manifest = {
    'checkpoint': str(checkpoint.resolve()),
    'checkpoint_bytes': checkpoint.stat().st_size,
    'cache_root': str(cache_root),
    'conditions': {
        'input': 'Synthetic zero images, two 224x224 RGB cameras, raw state dim 20',
        'robot_connected': False,
        'processes': f'{warm_runs + 1} fresh Python processes; four predictions in each',
        'cold': 'Checkpoint file pages evicted with file-scoped POSIX_FADV_DONTNEED; independent empty Inductor/Triton/CUDA cache directories',
        'warm': 'Checkpoint and compiler caches populated by preceding run',
        'limits': 'System libraries, tokenizer assets and device firmware caches are not evicted; no global drop_caches; storage-device internal caches are uncontrolled',
    },
    'runs': [],
}
manifest_path = output / 'manifest.json'
manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')

before = residency()
with checkpoint.open('rb') as stream:
    os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
after = residency()
manifest['eviction'] = {'before': before, 'after': after}
manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
if after['cached_pages'] != 0:
    raise RuntimeError(f"Cold condition not achieved: {after}")
log.info('Cold checkpoint verified: zero resident pages; output=%s', output)
for name in ['cold', *(f'warm_{index}' for index in range(1, warm_runs + 1))]:
    cached_before = residency()
    if name != 'cold' and cached_before['cached_fraction'] < 0.999:
        # Populate only this checkpoint if memory pressure reclaimed its pages.
        with checkpoint.open('rb') as stream:
            while stream.read(32 * 1024 * 1024):
                pass
        cached_before = residency()
    cache_files_before = sum(p.is_file() for p in cache_root.rglob('*'))
    if name == 'cold' and cache_files_before:
        raise RuntimeError('Compiler cache was not empty before the cold run')
    log.info('Starting %s: resident=%.2f%% compiler_cache_files=%d', name, cached_before['cached_fraction'] * 100, cache_files_before)
    started = time.perf_counter()
    with (output / f'{name}.log').open('w') as stream:
        result = subprocess.run(
            [sys.executable, '-c', child, str(output / f'{name}.jsonl'), str(output / 'config.yaml')],
            env=env, cwd=root, stdout=stream, stderr=subprocess.STDOUT,
        )
    record = {
        'name': name, 'returncode': result.returncode,
        'process_wall_sec': time.perf_counter() - started,
        'residency_before': cached_before, 'residency_after': residency(),
        'compiler_cache_files_before': cache_files_before,
    }
    manifest['runs'].append(record)
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    log.info('Finished %s: code=%d elapsed=%.2fs', name, result.returncode, record['process_wall_sec'])
    if result.returncode:
        raise RuntimeError(f'{name} failed; inspect {output / (name + ".log")}')
log.info('Completed controlled cache comparison: %s', output)
PY

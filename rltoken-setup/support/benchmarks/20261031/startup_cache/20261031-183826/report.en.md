# Inference startup cache comparison

## Conditions

RTX 4090 D; Stage 1 step-30000 checkpoint; BF16; four CPU threads; max-autotune; 10 denoising iterations, 20 predicted actions, 10 returned actions. The VLA-Precision adapter calls the shared RLinf OpenPI loader and sampler. Inputs are synthetic zero RGB images from two 224x224 cameras and a zero state. No cameras or robot are connected. Ray startup and communication are outside this test. Each of the three fresh processes makes four predictions.

The cold run evicts only this checkpoint's clean file pages with POSIX_FADV_DONTNEED, verifies zero residency using mincore, and uses empty isolated Inductor, Triton and CUDA driver cache directories. Both warm runs start with 100% checkpoint residency and reuse the populated compiler directories. System libraries, tokenizer assets and storage-device internal caches are uncontrolled; this is not a fully cold operating-system reboot. Stage boundaries synchronize the initialized GPU. There is one cold observation and two warm observations, rather than a distribution of repeated cold boots.

## Results

| Seconds | Cold | Warm process 1 | Warm process 2 |
|---|---:|---:|---:|
| Load | 111.990 | 17.666 | 17.658 |
| Load + first warmup | 446.819 | 51.922 | 41.071 |
| Steady call median (calls 3/4) | 0.065 | 0.065 | 0.065 |
| First warmup | 334.830 | 34.255 | 23.413 |

| Load stage, seconds | Cold | Warm 1 | Warm 2 | Source |
|---|---:|---:|---:|---|
| Import RLinf model factory | 0.9767 | 0.9139 | 0.9207 | [evaluations/realworld/vla_precision_ur/policy.py:194](/home/suchenyu/RLinf-rltoken/evaluations/realworld/vla_precision_ur/policy.py:194) |
| Resolve configuration | 0.0062 | 0.0030 | 0.0029 | [evaluations/realworld/vla_precision_ur/policy.py:197](/home/suchenyu/RLinf-rltoken/evaluations/realworld/vla_precision_ur/policy.py:197) |
| Import model implementation | 0.0066 | 0.0045 | 0.0045 | [rlinf/models/embodiment/openpi/__init__.py:47](/home/suchenyu/RLinf-rltoken/rlinf/models/embodiment/openpi/__init__.py:47) |
| Construct and initialize network | 12.9683 | 12.4240 | 12.4030 | [rlinf/models/embodiment/openpi/__init__.py:139](/home/suchenyu/RLinf-rltoken/rlinf/models/embodiment/openpi/__init__.py:139) |
| Set up transforms and tokenizer | 1.2918 | 1.0773 | 1.0652 | [rlinf/models/embodiment/openpi/__init__.py:430](/home/suchenyu/RLinf-rltoken/rlinf/models/embodiment/openpi/__init__.py:430) |
| Read metadata and map checkpoint | 2.1913 | 0.0119 | 0.0118 | [rlinf/models/embodiment/openpi/checkpoint.py:150](/home/suchenyu/RLinf-rltoken/rlinf/models/embodiment/openpi/checkpoint.py:150) |
| Normalize parameter keys | 0.0002 | 0.0002 | 0.0002 | [rlinf/models/embodiment/openpi/checkpoint.py:154](/home/suchenyu/RLinf-rltoken/rlinf/models/embodiment/openpi/checkpoint.py:154) |
| Copy mapped weights into model | 91.8389 | 0.7144 | 0.7199 | [rlinf/models/embodiment/openpi/checkpoint.py:164](/home/suchenyu/RLinf-rltoken/rlinf/models/embodiment/openpi/checkpoint.py:164) |
| Freeze and cast parameters | 1.1056 | 1.0124 | 1.0024 | [rlinf/models/embodiment/openpi/__init__.py:281](/home/suchenyu/RLinf-rltoken/rlinf/models/embodiment/openpi/__init__.py:281) |
| Transfer model to GPU | 1.1525 | 1.0714 | 1.1057 | [evaluations/realworld/vla_precision_ur/policy.py:205](/home/suchenyu/RLinf-rltoken/evaluations/realworld/vla_precision_ur/policy.py:205) |
| Register compiler wrapper | 0.0016 | 0.0015 | 0.0015 | [evaluations/realworld/vla_precision_ur/policy.py:207](/home/suchenyu/RLinf-rltoken/evaluations/realworld/vla_precision_ur/policy.py:207) |
| Other work and diagnostic output | 0.4499 | 0.4319 | 0.4204 | Load total minus top-level stages |

| Seconds | Cold | Warm process 1 | Warm process 2 |
|---|---:|---:|---:|
| Frame compilation (inclusive) | 284.393 | 16.761 | 16.322 |
| Backend (within frame compilation) | 269.193 | 1.558 | 1.581 |
| Runtime autotuning (inside first sampling call) | 50.336 | 17.336 | 6.936 |

| First prediction stage, seconds | Cold | Warm 1 | Warm 2 | Source |
|---|---:|---:|---:|---|
| inference.input | 0.00206 | 0.00199 | 0.00192 | [rlinf/models/embodiment/openpi/env_io.py:143](/home/suchenyu/RLinf-rltoken/rlinf/models/embodiment/openpi/env_io.py:143) |
| inference.sample | 334.82598 | 34.25194 | 23.40981 | [rlinf/models/embodiment/openpi/env_io.py:160](/home/suchenyu/RLinf-rltoken/rlinf/models/embodiment/openpi/env_io.py:160) |
| inference.decode | 0.00035 | 0.00036 | 0.00035 | [rlinf/models/embodiment/openpi/env_io.py:164](/home/suchenyu/RLinf-rltoken/rlinf/models/embodiment/openpi/env_io.py:164) |
| inference.to_cpu | 0.00003 | 0.00003 | 0.00003 | [evaluations/realworld/vla_precision_ur/policy.py:166](/home/suchenyu/RLinf-rltoken/evaluations/realworld/vla_precision_ur/policy.py:166) |

Frame, backend and runtime compiler metrics are explanatory measurements inside first-inference wall time. Backend time is included in frame compilation; neither should be added again to the warmup total.

Cold checkpoint application reads about 15.27 GiB from storage while copying mapped tensor data. Both warm runs charge zero storage reads to checkpoint mapping/application, reducing application from 91.84 seconds to about 0.72 seconds. Network construction still takes about 12.4 seconds in each fresh warm process.

The cold compiler records one FX cache miss; each warm process records one hit. Warm processes still spend about 16 seconds processing the compilation frame, plus runtime tuning. Subsequent calls in the same process take about 65 ms after the second call's additional graph preparation.

The raw cold log contains 23 backend algorithm-selection benchmark groups, reporting 70.44 seconds of benchmarking within backend compilation. For example, mm(968x2048, 2048x16384) selected a candidate running in about 0.4762 ms after comparing alternative implementations. This backend search differs from runtime Triton tuning; these nested measurements must not be summed indiscriminately.

## Evidence and sources

See report.md for the complete source mapping and first-call stage details, source_map.json for machine-readable locations, manifest.json for cache residency, summary.json for all derived numbers, config.yaml for the exact configuration, and cold/warm_1/warm_2 JSONL and log files for raw records. Reproduce from the repository root with the parent directory's run.sh. Existing user compiler caches and checkpoint contents remain unchanged.

## Additional cached fresh-process runs

| Cached fresh process | Load, s | Frame compile, s | Runtime tuning, s | First warmup, s |
|---|---:|---:|---:|---:|
| 1 | 17.67 | 16.76 | 17.34 | 34.26 |
| 2 | 17.66 | 16.32 | 6.94 | 23.41 |
| 3 | 17.92 | 16.39 | 9.42 | 25.97 |
| 4 | 17.86 | 16.82 | 10.48 | 27.47 |

All four runs hit the FX compiler cache. Runs 3 and 4 do not become progressively faster; most variation lies in runtime tuning. Within a process, the second prediction takes about 98–106 ms and calls 3/4 take about 64–69 ms. Reproduce a cold run plus four cached runs with `bash rltoken-setup/support/benchmarks/20261031/startup_cache/run.sh 4`. Raw additional records are warm_3.jsonl and warm_4.jsonl.

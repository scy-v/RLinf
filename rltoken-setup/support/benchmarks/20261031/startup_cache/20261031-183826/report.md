# 推理启动缓存对照

## 测试条件

在 RTX 4090 D 上使用 VLA-Precision 适配器，调用两套推理共用的 RLinf OpenPI 模型构建、权重加载与采样代码。checkpoint 为 Stage 1 step 30000，BF16，CPU 线程数 4，`max-autotune`，10 次去噪、预测 20 步、返回 10 步。输入为两张 224×224 全零 RGB 图像及全零 state；未连接相机或机械臂。未测试 Ray 启动、通信或物理执行。三次均为全新 Python 进程，各调用四次推理。

冷缓存测试仅使用 `POSIX_FADV_DONTNEED` 驱逐这个 checkpoint 的干净文件页，以 `mincore` 验证驻留率为零；Inductor、Triton 和 CUDA driver 编译缓存指向独立空目录。后两次复用该目录和已读入的 checkpoint 文件页，启动前驻留率均为 100%。没有全局清缓存；系统库、tokenizer 资源和存储设备内部缓存未清。因此“冷缓存”指模型文件页和指定的编译缓存，不代表整台机器重启后的全部资源冷态。

计时保留 load 和首次 warmup 的原有边界。每个启动阶段结束时同步已初始化的 GPU；该测量方式与上一轮诊断一致。父子阶段、编译内部指标不可重复相加。以下是一次冷启动和两次缓存启动的原始观测，不是多次冷启动的统计分布。

## 总体结果

| 阶段（秒） | 无文件／编译缓存 | 缓存后第一次新进程 | 缓存后第二次新进程 |
|---|---:|---:|---:|
| Load 总计 | 111.990 | 17.666 | 17.658 |
| 首次 warmup | 334.830 | 34.255 | 23.413 |
| Load＋首次 warmup | 446.819 | 51.922 | 41.071 |
| 同进程第二次预测 | 0.106 | 0.098 | 0.099 |
| 同进程第三、四次预测中位数 | 0.065 | 0.065 | 0.065 |

## Load：耗时和源码

| 加载阶段（秒） | 冷缓存 | 热缓存 1 | 热缓存 2 | 源码 |
|---|---:|---:|---:|---|
| policy.imports：导入 RLinf OpenPI 加载器 | 0.9767 | 0.9139 | 0.9207 | [evaluations/realworld/vla_precision_ur/policy.py:194](/home/suchenyu/RLinf-rltoken/evaluations/realworld/vla_precision_ur/policy.py:194) |
| policy.config：解析模型配置 | 0.0062 | 0.0030 | 0.0029 | [evaluations/realworld/vla_precision_ur/policy.py:197](/home/suchenyu/RLinf-rltoken/evaluations/realworld/vla_precision_ur/policy.py:197) |
| model.imports：导入模型实现 | 0.0066 | 0.0045 | 0.0045 | [rlinf/models/embodiment/openpi/__init__.py:47](/home/suchenyu/RLinf-rltoken/rlinf/models/embodiment/openpi/__init__.py:47) |
| model.construct：构建模型并初始化参数；网络组装见 pi0.py:90 | 12.9683 | 12.4240 | 12.4030 | [rlinf/models/embodiment/openpi/__init__.py:139](/home/suchenyu/RLinf-rltoken/rlinf/models/embodiment/openpi/__init__.py:139) |
| model.transforms：准备输入输出变换、tokenizer 和归一化资源 | 1.2918 | 1.0773 | 1.0652 | [rlinf/models/embodiment/openpi/__init__.py:430](/home/suchenyu/RLinf-rltoken/rlinf/models/embodiment/openpi/__init__.py:430) |
| checkpoint.read：读取 checkpoint 元数据并映射文件 | 2.1913 | 0.0119 | 0.0118 | [rlinf/models/embodiment/openpi/checkpoint.py:150](/home/suchenyu/RLinf-rltoken/rlinf/models/embodiment/openpi/checkpoint.py:150) |
| checkpoint.keys：参数键名归一化和旧格式适配 | 0.0002 | 0.0002 | 0.0002 | [rlinf/models/embodiment/openpi/checkpoint.py:154](/home/suchenyu/RLinf-rltoken/rlinf/models/embodiment/openpi/checkpoint.py:154) |
| checkpoint.apply：将映射的权重复制进模型，触发数据页实际读取 | 91.8389 | 0.7144 | 0.7199 | [rlinf/models/embodiment/openpi/checkpoint.py:164](/home/suchenyu/RLinf-rltoken/rlinf/models/embodiment/openpi/checkpoint.py:164) |
| model.freeze_and_precision：冻结参数和转换参数精度 | 1.1056 | 1.0124 | 1.0024 | [rlinf/models/embodiment/openpi/__init__.py:281](/home/suchenyu/RLinf-rltoken/rlinf/models/embodiment/openpi/__init__.py:281) |
| model.to_device：将模型搬到 GPU | 1.1525 | 1.0714 | 1.1057 | [evaluations/realworld/vla_precision_ur/policy.py:205](/home/suchenyu/RLinf-rltoken/evaluations/realworld/vla_precision_ur/policy.py:205) |
| model.compile_setup：包装 torch.compile；尚不执行首次图编译 | 0.0016 | 0.0015 | 0.0015 | [evaluations/realworld/vla_precision_ur/policy.py:207](/home/suchenyu/RLinf-rltoken/evaluations/realworld/vla_precision_ur/policy.py:207) |
| 其余工作及计时开销 | 0.4499 | 0.4319 | 0.4204 | 由 load 总计减去根阶段耗时，包含诊断输出 |

| checkpoint 条件 | 冷缓存 | 热缓存 1 | 热缓存 2 |
|---|---:|---:|---:|
| 启动前文件页驻留率 | 0.00% | 100.00% | 100.00% |
| checkpoint.read 存储读取量（GiB） | 0.0088 | 0.0000 | 0.0000 |
| checkpoint.apply 存储读取量（GiB） | 15.2658 | 0.0000 | 0.0000 |

冷加载的主要耗时位于 `checkpoint.apply`。这里将内存映射的权重复制进模型，触发约 15.27 GiB 的存储读取，而不是仅在 `torch.load` 返回映射对象时读取。热加载时读取量为零，该阶段降至不足一秒。模型构建仍约 13 秒：文件缓存不复用新进程中的模型对象，也不免去参数分配和初始化。

## Warmup：耗时和源码

| 首次预测阶段（秒） | 冷缓存 | 热缓存 1 | 热缓存 2 | 源码 |
|---|---:|---:|---:|---|
| inference.input | 0.00206 | 0.00199 | 0.00192 | [rlinf/models/embodiment/openpi/env_io.py:143](/home/suchenyu/RLinf-rltoken/rlinf/models/embodiment/openpi/env_io.py:143) |
| inference.sample | 334.82598 | 34.25194 | 23.40981 | [rlinf/models/embodiment/openpi/env_io.py:160](/home/suchenyu/RLinf-rltoken/rlinf/models/embodiment/openpi/env_io.py:160) |
| inference.decode | 0.00035 | 0.00036 | 0.00035 | [rlinf/models/embodiment/openpi/env_io.py:164](/home/suchenyu/RLinf-rltoken/rlinf/models/embodiment/openpi/env_io.py:164) |
| inference.to_cpu | 0.00003 | 0.00003 | 0.00003 | [evaluations/realworld/vla_precision_ur/policy.py:166](/home/suchenyu/RLinf-rltoken/evaluations/realworld/vla_precision_ur/policy.py:166) |

| 阶段（秒） | 无文件／编译缓存 | 缓存后第一次新进程 | 缓存后第二次新进程 |
|---|---:|---:|---:|
| 编译帧总耗时（包含后端） | 284.393 | 16.761 | 16.322 |
| 其中：后端编译 | 269.193 | 1.558 | 1.581 |
| 运行时内核调优 | 50.336 | 17.336 | 6.936 |
| FX 编译缓存命中／未命中 | 0/1 | 1/0 | 1/0 |

编译帧总耗时包含后端编译，后端又包含代码生成、候选算法编译和比较等工作。运行时内核调优也位于首次采样调用内；这些数值均为解释性指标，不能叠加到 warmup 总时间。FX 缓存命中后，新进程仍需要重新处理计算图并做运行时准备，因此首次调用仍需数十秒。同进程后续预测复用已就绪的状态，第三、四次约 65 ms。

冷缓存日志中的真实例子是 `mm(968x2048, 2048x16384)`：候选内核的最快执行耗时约 0.4762 ms，首次启动仍需要编译、测试多个候选。候选算法比较位于后端编译中，另有运行时 Triton 配置调优；两者不能混为同一项，也不能与父阶段重复相加。

模型实际预测的工作是：图像／文本／state 变换，视觉语言前缀与 KV cache 计算，10 次动作专家去噪，输出反归一化、裁剪及 CPU 搬运。完整采样器被编译时，按一次调用计时，避免插入计时操作拆分计算图。

| 预热部分 | 源码 |
|---|---|
| inference.input：输入变换、tokenization 和设备搬运 | [rlinf/models/embodiment/openpi/env_io.py:143](/home/suchenyu/RLinf-rltoken/rlinf/models/embodiment/openpi/env_io.py:143) |
| inference.sample：动作采样；前缀和去噪循环见 pi0.py:522 | [rlinf/models/embodiment/openpi/env_io.py:160](/home/suchenyu/RLinf-rltoken/rlinf/models/embodiment/openpi/env_io.py:160) |
| inference.decode：输出反归一化和裁剪 | [rlinf/models/embodiment/openpi/env_io.py:164](/home/suchenyu/RLinf-rltoken/rlinf/models/embodiment/openpi/env_io.py:164) |
| inference.to_cpu：动作复制回 CPU | [evaluations/realworld/vla_precision_ur/policy.py:166](/home/suchenyu/RLinf-rltoken/evaluations/realworld/vla_precision_ur/policy.py:166) |
| compiler.frame：TorchDynamo 首次图处理和编译入口 | [.venv/lib/python3.11/site-packages/torch/_dynamo/convert_frame.py:1490](/home/suchenyu/RLinf-rltoken/.venv/lib/python3.11/site-packages/torch/_dynamo/convert_frame.py:1490) |
| compiler.backend：TorchInductor 后端编译入口 | [.venv/lib/python3.11/site-packages/torch/_inductor/compile_fx.py:2497](/home/suchenyu/RLinf-rltoken/.venv/lib/python3.11/site-packages/torch/_inductor/compile_fx.py:2497) |
| compiler.autotune：TorchInductor 比较候选实现并选择内核；调优日志输出见同文件 4374 行 | [.venv/lib/python3.11/site-packages/torch/_inductor/select_algorithm.py:3119](/home/suchenyu/RLinf-rltoken/.venv/lib/python3.11/site-packages/torch/_inductor/select_algorithm.py:3119) |
| compiler.runtime_autotune：Triton 运行时配置调优；进程内首次调用可再次执行 | [.venv/lib/python3.11/site-packages/torch/_inductor/runtime/triton_heuristics.py:1173](/home/suchenyu/RLinf-rltoken/.venv/lib/python3.11/site-packages/torch/_inductor/runtime/triton_heuristics.py:1173) |

## 复查

运行脚本为上一级的 `run.sh`，需在仓库根目录执行。脚本只做离线预测，定向驱逐所选 checkpoint 文件页，并为每次实验创建独立编译缓存目录。原始数据见 `cold.jsonl`、`warm_1.jsonl`、`warm_2.jsonl`，完整终端输出见同名 `.log`，缓存条件见 `manifest.json`，汇总见 `summary.json`，配置快照见 `config.yaml`。

## 缓存后第三、第四次新进程补测

| 新进程缓存启动 | Load（秒） | 图处理与编译合计（秒） | 运行时调优（秒） | 首次 warmup（秒） |
|---|---:|---:|---:|---:|
| 1 | 17.67 | 16.76 | 17.34 | 34.26 |
| 2 | 17.66 | 16.32 | 6.94 | 23.41 |
| 3 | 17.92 | 16.39 | 9.42 | 25.97 |
| 4 | 17.86 | 16.82 | 10.48 | 27.47 |

四次均命中 FX 编译缓存，第三、第四次并未继续变快。主要波动出现在运行时调优，不能将第二次比第一次快理解为每次启动都会递减。同进程第二次预测约 98–106 ms，第三、第四次约 64–69 ms。用 `bash rltoken-setup/support/benchmarks/20261031/startup_cache/run.sh 4` 可重新运行一组冷启动及四组缓存启动；原始补测记录见 `warm_3.jsonl`、`warm_4.jsonl`。

# UR5e RLToken 技术说明

数据处理、权重转换、兼容性与验证记录。日常启动见[训练命令](../训练命令.md)，[英文说明](TRAINING_NOTES.en.md)。

通过 `obs_indices` 选择模型使用的原始 state 维度，默认取前 19 维。先计算与选择对应的归一化统计，再将同一选择用于 SFT 和第二阶段。

## 数据和归一化

```bash
source ~/RLinf/rltoken-setup/activate.sh
DATA="$HOME/.cache/huggingface/lerobot/scylearning/insert_black_rack_into_blue_slot_20260916_v01"
OUT="$PWD/rltoken-setup/outputs/ur5e_insert_black_rack_state19"
python toolkits/lerobot/calculate_norm_stats.py \
  --config-name pi05_ur_state --repo-id "$DATA" \
  --obs-indices 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 --output-dir "$OUT"
```

归一化脚本按数据格式选择读取方式：v3 默认读取数值 parquet；v2 默认使用现有 LeRobot 加载器，也可传 `--data-backend parquet` 使用通用数值读取器。读取器使用配置中的字段映射和 action 序列字段，与机械臂型号无关。

此命令读取全部 200 条轨迹、67,740 帧，生成 `$OUT/norm_stats.json` 和 `norm_metadata.json`。原始 LeRobot 文件保持原样。UR 统计直接读取 parquet，支持本数据集的 v3.0 格式，并包含不足一个 batch 的末尾帧。action 按 20 步窗口统计，超出轨迹末尾时重复最后一个 action，与 OpenPI 的窗口语义一致。

默认 state 是 TCP 位姿 6 维、速度 6 维、力/力矩 6 维和 `gripper_raw_position`，共 19 维。action 原样使用数据集的 6 维增量位姿和夹爪位置，补零到模型的 32 维；不再做一次差分。

使用环境变量时，同样明确指定本数据集使用的完整 1–19 维：

```bash
OBS_INDICES=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 python toolkits/lerobot/calculate_norm_stats.py \
  --config-name pi05_ur_state --repo-id "$DATA" \
  --output-dir "$PWD/rltoken-setup/outputs/ur5e_insert_black_rack_state19"
```

编号从 **1** 开始，与 `~/openpi` 的实现一致。也可传 `--obs-indices 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19`；命令行优先于环境变量，均未设置时使用前 19 维。允许重排，拒绝空列表、重复、非正数、越界和超过 32 个输入维度。

## 第一阶段 SFT

使用 `examples/sft/config/ur_rlt_stage1_sft_openpi_pi05.yaml`。本机已将 OpenPI 默认缓存中的 JAX `pi05_base` 转换为 RLinf 使用的 PyTorch 格式，`actor.model.model_path` 已指向 `/home/suchenyu/RLinf/rltoken-setup/outputs/pi05_base_openpi`，其中包含 `model.safetensors` 和 `config.json`。数据路径、默认 1–19 维统计路径和两路相机也已填写。

在新机器上，准备好前面的归一化结果和 OpenPI 缓存权重后，执行以下转换；输出 horizon 与训练配置一致，都是 20：

```bash
JAX_PLATFORMS=cpu python -m rlinf.utils.ckpt_convertor.openpi.convert --mode jax_to_openpi \
  --input-model "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_base" \
  --input-norm-stats "$OUT/norm_stats.json" \
  --output-model "$PWD/rltoken-setup/outputs/pi05_base_openpi" \
  --output-norm-stats "$PWD/rltoken-setup/outputs/pi05_base_openpi/scylearning/insert_black_rack_into_blue_slot_20260916_v01/norm_stats.json" \
  --action-dim 32 --action-horizon 20 --max-token-len 200
```

这一步转换已有权重并复制归一化文件，不下载或重新训练模型。权重保存在本机忽略目录中。

维度选择保存在配置中，便于 Ray worker 使用同一设置：

```yaml
actor:
  model:
    openpi_data:
      obs_indices: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
      norm_stats_path: /home/suchenyu/RLinf/rltoken-setup/outputs/ur5e_insert_black_rack_state19/norm_stats.json
    openpi:
      config_name: pi05_ur_state
      num_images_in_input: 2
```

归一化、SFT 和 rollout 共用通用的 `SelectState` 维度选择变换，再进入各机械臂的数据适配。UR 文件负责字段、双相机和 action 表示；Franka 等配置也可通过 `openpi_data.obs_indices` 启用同一变换，未设置时保留各自原有行为。外部相机映射为 `base_0_rgb`，腕部相机映射为 `left_wrist_0_rgb`。更换维度或顺序后，重新计算归一化并更新路径；保留统计旁的 `norm_metadata.json`，加载时会校验维度顺序。

UR 安装目标固定使用与你的 `~/openpi` 相同的 LeRobot 提交 `d6ea3bbce0fc8c75138f02639ac4ceaf12a67829`（包版本 0.3.4，数据格式 v3.0）。OpenPI 的导入兼容层会使用 `lerobot.datasets.lerobot_dataset`；UR 将数据样本自带的 `task` 文本映射为 prompt，读取 `meta/tasks.parquet`，默认任务文本为 `Insert the black rack into the blue slot`。

本机已逐张量核对转换结果：将本地 OpenPI 转换器的结果映射到 RLinf 参数布局后，全部 667 个基础权重张量完全一致。RLToken 新增模块从头初始化。

当前配置使用四张 A800，每卡 micro batch 为 2，global batch 为 32，每次参数更新包含 4 次梯度累积。`activate.sh` 将 Hugging Face Arrow 缓存放在独立的 `outputs/hf-datasets-cache` 中，避免与其他 `datasets` 版本生成的缓存混用；原始 LeRobot 数据位置不变。启动方式沿用作者的 SFT 脚本：

```bash
bash examples/sft/run_vla_sft.sh ur_rlt_stage1_sft_openpi_pi05
```

训练配置使用 GPU 0–3，总步数 30000，每 5000 步保存权重。`runner.resume_dir` 当前仍指向 `20261017-22:41:05` 这次训练的第 5000 步完整 checkpoint；执行同一命令将续训，从头训练需先将该字段设为 `null`。`20261018-22:40:04` 这次训练已生成第 30000 步权重，路径为 `/data/suchenyu/RLinf/logs/20261018-22:40:04-ur_rlt_stage1_sft_openpi_pi05/ur_rlt_stage1_sft_openpi_pi05/checkpoints/global_step_30000/actor/model_state_dict/full_weights.pt`。`~/RLinf/logs` 是该数据盘日志目录的软链接。

当前使用 `is_lora: False`、`train_expert_only: False` 和作者原有的联合 SFT 前向。`no_shard` 与作者第一阶段配置一致，各卡保留完整参数、梯度和优化器状态。`gradient_checkpointing: False` 也与作者配置一致；作者原有的 `Pi0.sft_forward()` 每次前向还会调用 `gradient_checkpointing_disable()`。

### 与作者 Franka 第一阶段的核对

核对基线为上游提交 `3d0470015d98919d48ac65fad3be13253c1ee8ba`。22 个训练入口、模型与更新相关文件逐字节一致，包括 `run_vla_sft.sh`、`train_vla_sft.py`、`pi0.py`、`rlt_token_transformer.py`、SFT 数据加载器和基础 SFT worker；另在包含保存逻辑改动的文件中核对了 9 个方法，数据加载、损失入口、优化器构建和更新方法均未改变。实际训练保存的 TensorBoard 配置与当前训练 YAML 的设置一致，差异只有启动脚本生成的日志目录。

PI₀.₅ 主干、RLT encoder/decoder、损失 `rlt_loss + 1.0 * vla_loss`、RLT 特征维度 2048、层数 2、头数 8、prefix 长度 1024、mask 和联合训练设置均保持原版。学习率峰值 `1e-5`、AdamW 参数、梯度裁剪 1、warmup 1000、cosine 类型、最低学习率 `2.5e-6` 也相同。以下差异需要保留在复现说明中：

| 项目 | 作者 Franka | 当前 UR |
|---|---|---|
| 卡数、micro / global batch | 1 卡，16 / 64 | 4 卡，2 / 32 |
| 有效相机输入 | 单相机 | 外部与腕部双相机 |
| 任务与 state | Franka 数据适配 | UR 数据适配与第 1–19 维筛选 |
| 最大步数与 LR 调度总长度 | 20000 | 30000，cosine 衰减周期随之延长 |
| 保存间隔 | 2000 | 5000 |
| 保存内容 | 权重及完整训练状态 | 仅权重，不能完整续训新 checkpoint |
| 终端日志 | 每步 | 每 10 步；TensorBoard 仍每步 |
| 启动状态 | 默认从头 | 当前配置从第 5000 步续训 |
| 梯度检查点配置 | False | True；两者 SFT 前向均主动关闭 |

UR 还固定了支持数据格式 v3 的 LeRobot 提交，并增加了对应归一化统计兼容路径；这属于数据读取适配。安装环境中 `rlinf-openpi==0.1.1` 的 54 个 Python 文件、LeRobot 的 221 个 Python 文件均与各自安装记录哈希一致。输入筛选专项 17 项、日志专项 8 项、CPU checkpoint 保存专项 2 项通过；本次核对未重新启动 GPU 训练。

因此，准确表述是“沿用作者的训练算法，适配 UR 数据并采用上述训练设置差异”，不能表述为“整个训练实现只增加了 obs_indices”或“除 batch/部署外参数完全相同”。

当前 OpenPI 模型不能直接通过 `is_lora: True` 切换到可用的 LoRA 训练：通用 PEFT 默认目标会匹配到不支持的 `ModuleList`，原生 `*_lora` Gemma 变体的 attention LoRA 也尚未实现。需要先明确适配 VLM、action expert 及 RLToken 的训练范围并验证；当前命令使用全参模式。

## 第二阶段

在完成 UR 驱动、任务与部署配置后，设置 feature model 的同一数据配置，并将 actor 的 `proprio_dim` 设为所选维度数量。默认示例为：

```yaml
rollout:
  rlt_feature_model:
    model_path: /第一阶段checkpoint/actor
    openpi_data:
      repo_id: scylearning/insert_black_rack_into_blue_slot_20260916_v01
      obs_indices: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
      norm_stats_path: /home/suchenyu/RLinf/rltoken-setup/outputs/ur5e_insert_black_rack_state19/norm_stats.json
    openpi:
      config_name: pi05_ur_state
      num_images_in_input: 2
      state_indices: []
actor:
  model:
    proprio_dim: 19
```

第二阶段的 VLA 输入和 MLP 的原始 proprio 使用同一选择。`openpi.state_indices` 是既有的另一套零基索引；使用 `openpi_data.obs_indices` 时保持为空，以免重复裁剪；两个后端都会拒绝同时配置两套索引。环境须提供完整的原始 48 维 state，顺序与数据集一致。

现有 `realworld_rlt_stage2_ac_mlp.yaml` 的硬件和任务仍属于 Franka；上面的片段只说明 UR 模型字段，不能直接把现有 Franka 配置用于 UR 真机。未运行真机控制或完整训练。

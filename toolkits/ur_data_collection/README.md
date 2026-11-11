# UR5e RECAP rollout 数采

这里将 `~/ur_data_collection/lerobot_ur5e_keyteleop` 的 UR 机器人、键盘遥操作与 LeRobot v3 数采迁入当前 RLinf 环境，并加入第一阶段 PI05 模型 rollout。原始文件及 SHA256 见 [provenance.json](provenance.json)；源目录未修改。

## 使用

在仓库根目录执行：

```bash
bash recap-setup/support/install-ur-rollout.sh
bash toolkits/ur_data_collection/run_rollout.sh
```

参数集中在 [ur5e_recap_rollout.yaml](config/ur5e_recap_rollout.yaml)。模型在当前进程加载并预热，然后连接设备；默认使用 5,000 步 SFT 权重、19 维状态、7 维动作、预测 20 步并执行其中 10 步，RTC 关闭。机器人控制参数、状态与动作字段顺序、双相机设置和 15 FPS 沿用原数采配置。

启动后先按回车确认，直接自动复位到初始位姿，再按回车开始采集。后续轨迹保存或丢弃后先等待回车，确认后才启用手动复位：运动键可操作机器人，右方向键进入自动复位，回车开始下一条。复位不写入数据。运动键沿用原 `s/w、d/a、q/e、r/t、g/f、b/v`，`o/l` 开／合夹爪，`/` 切换精细步长。

键盘结束采用独立的结果判定阶段：采集中在 0.5 秒内连按两次句号键，执行完当前 chunk 后停止运动，再询问本条结果。选择结果并完成保存或丢弃之后，才进入复位。

| 生效阶段 | 按键 | 行为 |
|---|---|---|
| 采集 | 按住运动键或 `o/l` | 当前动作直接由人工接管，记录 `intervene_flag=true`；松开后继续原 chunk 的模型动作 |
| 采集 | 0.5 秒内连按两次句号键 | 请求结束当前轨迹，完成当前 chunk 后进入结果判定 |
| 结果判定 | 0.5 秒内双击空格 | 按成功保存 |
| 结果判定 | 0.5 秒内双击 F | 按失败保存 |
| 结果判定 | 0.5 秒内双击 D | 丢弃当前轨迹 |
| 手动复位 | 右方向键 | 结束手动调整，进入自动复位 |
| 所有阶段 | Esc / Ctrl+C | 退出，丢弃尚未保存的 episode，释放设备 |

各阶段都有终端提示，识别到按键后打印确认。结果判定期间不下发动作；D/F 在采集接管和手动复位期间仍是原运动键，不会判失败或丢弃。切换阶段会清空双击历史和遥操作事件，仍按住的运动键必须松开重按后才生效；纯遥操作 record 入口不启用这套结果判定按键。

`rollout.max_chunks=45` 是单条实际执行的 chunk 次数上限，包括含人工接管动作的 chunk；仅达到上限时自动按失败保存，不询问结果；若同一 chunk 已收到 `..` 请求，则优先进入人工判定。默认每个 chunk 为 10 个动作，45 次最多执行 450 个动作。`record.task.num_episodes=100` 表示本次新增条数。默认 `storage.root=null`，每次新采集在 `storage.output_dir`（`logs/ur_recap_rollout/round_001`）下创建独立时间戳目录，终端打印完整路径。续采设置 `record.task.resume=true`，并将 `storage.root` 指向已有数据集的时间戳目录，校验原 schema 与 FPS 后追加。下一轮修改 `repo_id` 与 `storage.output_dir` 中的轮次，完成采集后再标注。

## 相对原数采增加的逻辑

[rollout_loop.py](rollout_loop.py) 每次推理取得一个 chunk，逐步读取新 observation，选模型动作或人工动作，再调用原 `send_action`，将其返回动作与执行前 observation 一起写入 `add_frame`。保存的是实际下发的动作命令，不是模型未执行的预测，也不是动作执行后的反馈位姿。

例如预测 `a1…a10`，人工只接管第 1–3 步，则保存 `h1,h2,h3,a4,…,a10`。每步都推进 chunk 下标，释放接管后继续剩余模型动作；第 4 步后的 observation 来自经过人工干预后的实际机器人。键盘结束请求和 chunk 次数上限都不截断当前 chunk。人工结束后，数据留在未提交的 episode 缓冲区，结果判定完成前不能保存；选择丢弃、退出及异常会放弃整条未保存数据。

模型或人工动作选定后，夹爪命令按 `record.robot.close_threshold`（默认 0.7）二值化：小于阈值为 0，否则为 1，再下发并记录。`record.robot.gripper_reverse` 仍仅由驱动转换硬件方向，保存逻辑开／合命令；observation 中测得的夹爪位置保留连续值。

完整状态、`action`、双相机视频和 LeRobot 自动字段沿用原格式，额外增加两列：

| 字段 | 写入语义 |
|---|---|
| `is_success` | episode 结束后统一回填每帧，同一条全部为成功或失败；属于独立 feature，不混入 `observation.state` |
| `intervene_flag` | 每帧实际动作是否由人工接管提供 |

不要求人工填写 `done`、`reward` 或 `return`。每次启动的模型、控制和数据配置记录在 `meta/rollout_from_episode_XXXXXX.json`，便于追溯该批次的 checkpoint。

`rollout.record_timing` 默认 `true`：采集循环正常结束、Esc 或异常退出时，在该配置 JSON 中补充 `timing`。耗时四舍五入到整秒，保存为 `33min 25s` 或 `22s`；记录带时区的 `started_at`／`ended_at`、`total_duration`，以及 `record_duration`（推理、执行和逐帧记录）、`reset_duration`、`wait_duration`（等待回车）、`review_duration`（结果判定）、`save_duration`（每条保存／丢弃处理）、`other_duration`（提示及状态切换等开销）。时间范围从首次复位确认前到采集循环退出，包含未完成或丢弃轨迹消耗的时间，不包含模型加载、预热、设备退出清理及回收站确认；续采每次单独记录。设为 `false` 时只保存配置，不写 `timing`。

[robot/ur5e.py](robot/ur5e.py) 保留原控制计算、`get_observation`、`send_action` 与复位数学；迁移补充 RTDE、相机、串口和夹爪线程退出清理。[policy.py](policy.py) 只负责按 SFT 顺序选状态及准备双相机输入，权重加载、归一化和动作解码复用当前 RLinf PI05 推理实现。

退出时同一 JSON 的 `collection_summary` 保存当前数据集的自主／人工干预成功、失败和总条数，包含续采前已有轨迹；任意一帧接管即计入人工干预，未保存或丢弃的轨迹不计入。该统计不受 `record_timing` 开关影响。

## RECAP 数据衔接

Step 1 使用 [ur_recap_compute_returns.yaml](../../examples/offline_rl/config/ur_recap_compute_returns.yaml)，先将 rollout 的 `dataset_path: ???` 替换为终端打印的实际时间戳目录；多批数据分别加入 `train_data_paths`。该步骤将原 SFT 数据按成功示教处理，rollout 按 `is_success` 处理，结果写为 `meta/returns_ur_round_001.parquet`。每步 reward 为 −1，成功末帧为 0、失败末帧为 −300；无需在采集阶段写这些值。

本次补齐 v3 任务文本索引、跨 parquet 的完整 episode 回报计算，以及单元素布尔数组读取；附加回报文件不会被误声明成原始 parquet 的 feature。训练固定长度 action chunk 由数据加载器从逐帧动作组装，episode 尾部按加载器规则补齐并返回 padding mask，采集端不伪造未来动作。

当前已完成第一阶段模型 rollout、Step 1 对接和 Step 2 的 UR 图像／任务文本训练适配，入口见[训练命令](../../recap-setup/训练命令.md)。Step 3–4 的 UR 数据映射与人工干预样本 advantage 处理仍待接入；写入 `intervene_flag` 本身不会使现有 Step 3 自动将干预样本标为正 advantage。

## 验证

[测试文件](../../tests/unit_tests/test_ur_data_collection.py) 使用假硬件和真实 LeRobot 写盘，覆盖接管后继续原 chunk、成功与失败、chunk 次数上限、双击时限、阶段隔离、主动丢弃、视频读取、训练 chunk 补齐、续采、复位及连接失败清理。实际 GPU 权重验证与未覆盖范围见[测试进展](../../recap-setup/测试进展.md)。

采集进度在终端同一行显示已完成 chunk 数／上限，以及模型控制或人工接管状态，运动按键不回显。按 Esc 可提前结束本批采集：当前未保存轨迹丢弃，之前已保存轨迹保留，等待回车阶段同样有效。Ctrl+C 或报错后先停止设备、丢弃未保存缓存并关闭数据集，再询问是否将当前整个数据集（含此前已保存轨迹）移入回收站；输入 `y` 才移入回收站，其他输入保留，询问期间再次 Ctrl+C 或输入关闭也保留。Esc 正常提前退出保留已保存数据，不触发删除询问；保存过程中等待写盘结束。

手动复位使用独立的 `rollout.reset_teleop`：force 平移／旋转为 0.045／0.075，position 平移／旋转为 0.009／0.03，均为当前采集粗调步长的 1.5 倍。复位阶段 `/` 不切换精细步长，返回采集后恢复原粗调／精细配置；自动复位速度不受此设置影响。

UR5e 真机评估
==============

本页介绍如何在 UR5e、大寰 PGI/PGE 夹爪和两台 RealSense 相机上部署 RLinf OpenPI checkpoint。UR 专用实现对齐本机 VLA-Precision 的观测、动作和回合控制，模型采样与 RTC 重新规划继续使用 RLinf。

准备环境
--------

在 GPU 和机器人控制机上安装 OpenPI、UR 依赖，再激活环境。分机部署的 placement 配置参考 :doc:`realworld`。

.. code-block:: bash

   bash requirements/install.sh embodied --model openpi --env ur \
       --venv .venv --no-root --install-rlinf
   source rltoken-setup/activate.sh

该命令创建模型和硬件运行环境。UR 安装项包含 RTDE 1.6.3、串口夹爪和 RealSense 依赖。

配置硬件和模型路径
------------------

在 ``evaluations/realworld/realworld_ur5e_eval_pi05_sft.yaml`` 中设置硬件和 checkpoint。``rollout.model.model_path`` 直接指向包含 ``model_state_dict/full_weights.pt`` 的训练结果 ``actor`` 目录；``rollout.model.openpi_data.norm_stats_path`` 指向配套的 ``checkpoints/norm_stats.json``。模型工厂直接加载完整权重，``runner.ckpt_path`` 保持 null，避免重复加载。UR 配置将 ``precision`` 和 ``embed_dtype`` 均设为 ``bf16``。OpenPI 加载器保留归一化、视觉输入层和动作输出头等参数的 FP32 精度，主干与嵌入使用 BF16，与 Franka 默认混合精度一致。

默认机械臂 IP 为 ``192.168.201.11``，夹爪串口为 ``/dev/ur5e_left_gripper``，外部相机为 ``348522072391``，腕部相机为 ``944622073455``。两路相机均使用 640×480、15 FPS。夹爪力度为 60、速度为 100、原始行程为 0–1000；模型夹爪动作大于 0.7 时打开，否则关闭。观测继续使用实测原始位置除以 1000。

任务和运动设置位于 ``examples/embodiment/config/env/realworld_ur5e_sft_env.yaml``。固定复位 TCP 为 ``[0.389694, 0.107973, 0.382018, -3.14, 0.0, -1.57]``，前三项是基坐标系位置，单位为米，后三项为 Euler XYZ，单位为弧度。随机复位关闭，工作空间边界继续生效。力控选择向量为 ``[1, 1, 1, 0, 0, 1]``，负载为 1.601 kg，重心为 ``[0.011, -0.002, 0.052]`` 米。核对实物设置后，将 UR 控制器置于远程控制模式。

运行同步或 RTC 评估
-------------------

使用 UR 专用入口，在指定回合数结束后正常退出。激活环境后执行：

.. code-block:: bash

   bash evaluations/realworld/run_ur5e_eval.sh
   # 重叠执行动作和模型重新规划：
   bash evaluations/realworld/run_ur5e_eval.sh runner.rtc.enabled=true

启动脚本在仓库内真实的 ``results/`` 目录下创建带时间戳的目录。若未设置 ``RAY_TMPDIR``，Ray 临时文件存放在 ``/tmp/ray/`` 下，使用短路径以避免 Unix domain socket 路径长度超限。训练 checkpoint 继续保存在 ``logs/`` 下。同步模式执行完当前动作块后才请求下一次预测；RTC 沿用 RLinf 的引导采样和重新规划调度，UR worker 额外处理评估回合数结束。两种模式均使用 10 步去噪、预测 20 个动作、每块执行 3 个动作。默认配置在回合运行期间启用持续 PD 跟踪。设置 ``cluster.node_groups[0].hardware.configs[0].impedance.continuous_tracking=false`` 后，每个动作计算并发送一次 PD 力，沿用独立 VLA-Precision 推理后端的更新方式。

UR 评估和独立 VLA-Precision 入口均只传入机器人 IP 来创建 RTDE 连接，使用 SDK 根据机器人确定的默认通信频率（UR5e 为 500 Hz）。``impedance.frequency: 100.0`` 仅设置启用持续跟踪时的更新频率，不再缩放力矩，也不控制动作节奏。旋转增益直接作用于力矩，随附配置使用 ``kp_rotation: 20.0``、``kd_rotation: 4.0``，等效于原来的 2000/400 除以 100；独立入口对应名称为 ``kp_rot``、``kd_rot``。迁移自定义旧配置时，先将两个旋转增益分别除以原配置频率，之后调整更新频率时保持增益不变；独立入口省略增益参数时使用 32/6.4，保持原 4000/800 除以 125 的默认输出。动作节奏由 ``env.eval.override_cfg.step_frequency`` 控制。

UR 逐动作控制使用同一次读取的位姿和速度计算目标及阻抗力。

分布式评估配置通过 ``env.eval.override_cfg.enable_camera_player: true`` 默认开启相机预览，设为 ``false`` 可关闭窗口。模型图像使用 RGB；UR 预览将自己的图像副本转换成 BGR 后交给 OpenCV。预览随观测刷新而更新。

查看启动耗时和循环频率
----------------------

分布式 UR 入口先加载模型、复位机器人，再使用真实观测完成一次预热推理，最后等待 Enter。预热动作会被丢弃，编译时间不占用首回合的超时预算。``[LOAD]`` 和 ``[WARMUP]`` 每秒报告累计耗时；加载、预热、推理、循环和回合的常规计时始终显示；当 ``env.eval.ur_evaluation.save_logs`` 为 true 时，也会保存加载、预热、推理和循环完成后的耗时。

``env.eval.ur_evaluation.save_logs: true``\ （默认值）时，分布式入口会在 ``results/rlinf-ur/logs/all_logs`` 下写入与 VLA 相同形式的 YAML 推理记录，以及汇总计时 JSONL。上一层日志目录中的 ``latest.yaml`` 和 ``latest.jsonl`` 指向同一次运行。``env.eval.ur_evaluation.log_dir`` 可修改该目录；每次运行的配置、指标和原始执行记录仍保存在 ``results/rlinf-ur/runs`` 下。每条 YAML 推理记录包含输入 ``state``、``actions``、``actions_delta``、提示词、回合信息和全局 ``infer_index``；为兼容原有格式，``infer_time`` 保留相同编号。使用此编号关联环境和推理计时，``component`` 区分执行与推理来源。详细动作计时还包含块内的 ``action_index``。加载和预热是独立事件，不对应正式预测记录。``episode_boundary: true`` 标记可能包含复位和人工等待的环境样本。RTC 的预测记录另含 ``chunk_id``；``rtc_step`` 汇总覆盖一个时间窗口，不对应单次推理。终端循环摘要最多每秒输出一次，同步模式的块间推理等待仍会降低整体动作频率。

设置 ``env.eval.ur_evaluation.action_timing_enabled=true`` 后，终端会打印逐动作详细计时；开启日志保存时，也会将这些字段写入汇总计时 JSONL 和每次运行的 ``ur5e-actions.jsonl``。其中 ``control_sec`` 和 ``control_hz`` 包含动作处理和补足等待，``observation_sec`` 则在之后单独测量。开关默认为 false，不影响常规计时。``save_logs: false`` 会关闭 UR 动作与状态、计时和回合结果文件的保存，但不影响终端输出；Ray 日志和录像有各自的控制方式。开启保存时，即使关闭详细动作计时，也会保存动作和状态记录。

设置 ``continuous_tracking: true`` 后，``[PD]`` 消息会单独报告后台发力频率。评估 wrapper 在执行动作时约每秒收集一次统计，并在回合结束、复位和关闭时收集剩余数据。开启日志保存后，统计以 ``component: controller``、``event: force_mode`` 写入汇总计时 JSONL，独立于 ``action_timing_enabled``。``target_hz`` 是配置频率，``measured_hz`` 根据成功完成的 ``forceMode`` 调用间隔计算。``send_interval_sec_mean`` 和 ``send_interval_sec_max`` 分别表示平均和最长发送间隔；``work_overruns`` 统计读状态、算力和发送的耗时超过配置周期的次数；``held_target_samples`` 统计沿用上一次目标的更新次数。控制线程只累计计数，不写文件；回合运行期间的推理等待也会累计，在 wrapper 下一次收集时写出。预热和等待回车期间暂停力控，频率统计排除复位和等待期间的暂停时间。这些数据测量 SDK 调用时序，不代表 RTDE 数据包的新鲜度或硬实时保证。

控制回合并读取结果
------------------

每次复位后，在机器人控制机的实体键盘上按一次回车开始。0.5 秒内双击回车记录失败并复位，双击空格记录成功；每回合 60 秒超时。默认记录 20 个回合结果，最后一个结果记录后复位并退出。UR wrapper 会屏蔽已结束动作块中剩余的动作；RTC 会等待并丢弃旧预测，再开始新回合。

评估配置设置了 ``impedance.start_paused: true``，连接和复位后保持力控暂停，模型预热和等待回车期间也不启动力控。按回车后才以当前实测位姿为起点启用力控；成功、失败或超时后，先停止力控，再记录结果和复位。后续每个回合都需要再次按回车，最后一次回零也保持力控暂停。初始化和回合复位仍按配置执行 ``moveL`` 回零动作。

存在多个键盘设备时，在启动 Ray 前指定设备：

.. code-block:: bash

   export RLINF_KEYBOARD_DEVICE=/dev/input/by-id/your-keyboard-event-kbd

选择当前用户可读的实体键盘 event 设备。SSH 终端中输入的文本不等同于该设备的按键事件。

在 ``env.eval.ur_evaluation`` 下设置 ``num_episodes``、``episode_timeout_sec``、``double_press_interval`` 和 ``wait_for_start``。每次回合结束后，``ur5e-evaluation.json`` 保存状态、成功标记、耗时、实际执行动作数和累计成功率；录像和 runner 指标使用同一个日志目录。``max_steps_per_rollout_epoch`` 另外限制总执行步数。

核对观测和动作语义
------------------

部署时保留与训练数据相同的表示。UR 适配先去除控制器的 TCP offset，在 EE 上计算相对观测和局部动作，再加回 offset 得到要发送的 TCP 目标。位姿相对于复位时的实测 EE 位姿；RTDE 线速度、角速度、力和力矩保持原始基坐标系表示。经过 Euler 转换后的 state 共 19 维：位姿 6、速度 6、力和力矩 6、实测夹爪开度 1。

相机图像保留完整视野：UR 将 RealSense 的 BGR 图像转为 RGB，按 OpenPI 的 PIL 缩放方式补边到 224×224；只在 UR 中替代通用的中心正方形裁剪。

局部 EE 动作使用 ``p_target = p + R * delta_p`` 和 ``R_target = R * EulerXYZ(delta_r)``。UR 自己处理该变换，外层只应用 Euler 观测 wrapper；``use_relative_frame`` 为 false，避免重复执行通用 adjoint 变换。微小动作保持在基坐标系逐轴判断位移阈值 0.0001 米，整体判断旋转阈值 0.001 弧度，将静止轴锁存在保持目标上。

复位依次打开夹爪、停止 force mode、调用 ``servoStop`` 和 ``moveL``、重新设置负载、等待 0.25 秒、清零力传感器、再等待 0.50 秒，最后允许接收下一个动作。控制器保留 0.048 米位置误差限制和 4.0 m/s 速度误差限制，``rotation_error_clip_degrees: null`` 关闭旋转误差裁剪。这些行为均属于 UR 实现，Franka 的通用环境、变换和控制逻辑保持原样。

验证范围
--------

假 SDK 测试覆盖非零 TCP offset 下的局部动作合成、原始速度输入、微小动作保持、二值夹爪、复位顺序和力传感器清零。键盘测试覆盖成功、失败复位、超时和旧动作块取消。可使用合成观测验证 checkpoint 加载和同步、RTC 的 GPU 推理，不连接硬件。这些检查不衡量任务成功率、接触效果或真机时序，仍需在配置的 UR 工位验证。

与 VLA-Precision 执行流程对照
----------------------------------------

若需将机器人执行与上述 RLinf 分布式路径对照，可使用移植的 VLA-Precision 后端。``evaluations/realworld/vla_precision_ur/`` 包含从本机 VLA-Precision 复制的单臂 UR 主循环、记录、图像处理和 UR policy 变换。``provenance.json`` 记录来源版本、文件哈希和保留方法，文件哈希也包含来源工作区已有的修改；运行时不依赖原仓库。

该路径使用独立配置 ``evaluations/realworld/vla_precision_ur.yaml``。硬件、相机、任务、力控和回合设置沿用本地部署配置。对比动作节奏时，应对齐 ``robot.action_fps`` 和 ``env.eval.override_cfg.step_frequency``。输出目录为 ``results/vla-precision-ur``。``model.checkpoint_dir`` 指向给定的 RLinf actor checkpoint，``model.norm_stats_path`` 指向配套统计量；``model.network`` 记录训练网络结构，包括预测 20 步、32 维内部动作、7 维物理动作和启用 RLT。推理与分布式 UR 入口一样采用 BF16/FP32 混合精度，``precision`` 和 ``embed_dtype`` 均设为 ``bf16``。统计量包含 19 维 state 和补齐后的 32 维 action，机器人仅接收前 7 维动作。

独立 policy 使用 RLinf 模型已安装的输入和输出变换。两条 UR 入口均调用共享的 RLinf 观测转换，包括 CPU 图像归一化、NHWC 图像布局及 float32 state。独立适配器直接返回 RLinf 解码后的动作块，不再重复转换观测或解码输出。两条入口均由 ``openpi.action_horizon`` 控制预测长度（默认 20），由 ``openpi.action_chunk`` 控制返回和执行长度（默认 3），``num_action_chunks`` 引用 ``openpi.action_chunk``。分布式入口修改 ``rollout.model.openpi.action_chunk``，独立入口修改 ``model.network.openpi.action_chunk``；机器人执行返回的整个动作块。

独立入口始终打印加载、预热和推理循环的常规计时，``record.save_logs: true``\ （默认值）时，还会将测量结果保存到实验目录下的 ``logs/all_logs/timing_<timestamp>.jsonl``。在独立 YAML 中设置 ``record.action_timing_enabled: true``，还会打印动作细分耗时，开启日志保存时也会写入文件：状态读取、目标转换（包括 TCP 偏移读取）、阻抗计算、指令发送、夹爪目标更新、其余处理和补足等待。文件中的耗时单位为秒，终端动作耗时单位为毫秒。动作诊断开关默认为 false，不影响常规终端计时。``record.save_logs: false`` 会关闭计时、动作与输入状态、回合记录和汇总图的文件保存，不影响终端输出和录像。

独立入口的两个日志文件都有快捷链接：``logs/latest.yaml`` 指向输入状态、动作块和回合结果，``logs/latest.jsonl`` 指向计时。每轮计时的 ``infer_index`` 与 YAML 推理记录一致；详细动作计时还用 ``action_index`` 标识块内动作。常规循环计时在回合复位和人工等待之前保存。写入线程忙时，预测记录会排队，不会被丢弃。

移植的 UR 入口还支持通过 ``model.network.openpi.torch_compile`` 选择编译模式。设为 ``max-autotune`` 可参考 OpenPI 的 PyTorch 设置，用 ``torch.compile`` 包装完整的 ``sample_actions`` 方法。首次推理会在现有的回合前预热阶段完成编译。加载和预热期间每秒原地更新累计耗时，完成或中断时输出最终时长。``--check-model`` 的首次推理也使用同一计时显示。输入形状变化可能触发重新编译，编译后的浮点输出也可能与原路径存在差异。多组输入的离线检查发现，编译后夹爪输出的最大差异约为 0.684，因此选择该模式时还需验证任务行为。布尔值保持原有含义：``false`` 关闭局部算子编译，``true`` 编译局部算子并保留 Python 采样循环。其他可选字符串为 ``default``、``reduce-overhead`` 和 ``max-autotune-no-cudagraphs``。分布式 UR 入口单独使用 ``rollout.sampler_compile_mode`` 选择完整采样器编译，当前同样设为 ``max-autotune``。设为 ``rollout.sampler_compile_mode=null`` 可仅保留原有局部算子编译设置。通用模型加载器保持原样。

启动诊断保留原有 load 和 warmup 总耗时，并在同一计时 JSONL 文件中增加 ``startup_stage``、``startup_summary`` 和 ``startup_compile`` 记录。加载阶段分别统计模型构建、变换资源准备、checkpoint 映射、参数复制、精度转换和设备搬运，各阶段附带进程 CPU 时间、Linux 存储读取字节数和缺页次数。内存映射 checkpoint 的磁盘读取可能发生在 ``checkpoint.apply``，而非 ``checkpoint.read``；比较两次运行时需同时查看这两个阶段。读取字节增量为零表示该阶段没有计入本进程的存储读取，不能据此确定每个字节来自哪一级缓存。

预热阶段分别统计输入处理、动作采样和输出解码。编译记录包含编译帧处理、运行时内核调优、CUDA graph 准备及可用的编译缓存命中计数；这些是采样调用内部的解释性指标，不能再加到采样总耗时上。完整编译的采样器作为一次调用计时，避免计时操作拆分编译图。在线 RLT 还分别统计前缀编码、RL token 提取、参考动作采样和 MLP 路由；``runner.timing_mode=detailed`` 时，各 worker 在 ``runner.logger.log_path`` 下写入 ``startup_timing/<component>_<rank>.jsonl``。启动计时在阶段边界同步已初始化的加速器；常规控制和训练迭代保留已有计时行为。未启用合成观测预热时，在线路径统计首次实际推理。

激活现有环境后，先检查依赖和路径，再启动。可选的 GPU 检查使用合成观测和固定扩散噪声，对比独立适配器与 RLinf worker 的模型输入输出；两个检查命令均不连接硬件：

.. code-block:: bash

   source rltoken-setup/activate.sh
   bash evaluations/realworld/run_ur5e_vla_precision.sh --check
   bash evaluations/realworld/run_ur5e_vla_precision.sh --check-model
   # 连接硬件、复位、预热模型，然后等待 Enter：
   bash evaluations/realworld/run_ur5e_vla_precision.sh
   # 选择另一份独立 YAML：
   bash evaluations/realworld/run_ur5e_vla_precision.sh --config /path/to/ur.yaml

该入口在单进程内运行 OpenPI 原生 ``Policy.infer()``，通过适配层调用 RLinf 网络并加载其 checkpoint。RGB 相机读取、224 × 224 补边、state 拼接、位姿增量、夹爪线程、力控计算、终端键盘控制和退出保存视频提示均沿用 VLA-Precision。键盘输入来自当前终端，SSH 终端也可使用；双击 Enter 记录失败并复位，双击空格记录成功。默认仍为 20 回合，每回合最长 60 秒。

VLA-Precision 预测完整动作序列，同步执行前 3 个动作。独立入口默认设置 ``robot.force_mode.continuous_tracking: false``，每个动作发送一次力控指令。在 YAML 中将其设为 ``true``，即可启用本地后台控制器：前台按 ``robot.action_fps`` 更新绝对 TCP 目标，后台按 ``robot.force_mode.rtde_freq`` 读取当前状态、沿用原 PD 公式重算力并调用 ``forceMode``。每个目标首次发送成功后才确认接收，不等待机械臂实际到达目标。等待下一轮推理时，后台继续跟踪最后的绝对目标。

独立后台控制器在按回车启动 episode 后才运行，成功、失败、超时、复位和关闭时停止。复位先等待线程退出，再回位、设置载荷和清零力传感器；预热和回合间等待期间力控保持关闭。控制器失败会中止执行，停止失败时不继续回位或清零。原有零输出过滤仍会为被过滤的动作禁用力控选择轴。``[PD]`` 日志及保存的 ``controller`` 计时事件记录成功发送的实际频率。持续跟踪模式下，逐动作的 ``send_ms`` 表示提交目标并等待确认的时间，PD 计算在后台完成。这个选项在移植后的独立推理后端内实现，不调用 RLinf 机器人控制器。

分布式 UR 入口通过自身的 impedance 配置独立选择逐动作更新或持续跟踪。机械臂和夹爪控制机与 RLinf 环境位于同一节点时，连接直接位于环境进程内。夹爪串口 I/O 使用后台线程，同步动作块的完整相机观测在最后一个动作后刷新。移植路径也保留来源代码未使用 RLinf 工作空间裁剪的行为，不开启 RTC。来源 YAML 中的 ``model.infer_fps`` 未被硬件主循环读取，因此此处省略。

固定合成输入下输出一致，只能验证模型输入输出适配；真机对照用于检查相机、控制和执行时序。两条路径仍共享 RLinf 网络实现，因此仅凭本次对照不能排除其与官方 OpenPI 模型之间的差异，也不能确定 checkpoint 本身的质量。Franka 及通用环境、控制器均未修改。

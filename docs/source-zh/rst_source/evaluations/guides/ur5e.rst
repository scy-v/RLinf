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

启动脚本在仓库内真实的 ``results/`` 目录下创建带时间戳的目录。若未设置 ``RAY_TMPDIR``，Ray 临时文件存放在 ``results/ray/`` 下。训练 checkpoint 继续保存在 ``logs/`` 下。同步模式执行完当前动作块后才请求下一次预测；RTC 沿用 RLinf 的引导采样和重新规划调度，UR worker 额外处理评估回合数结束。两种模式均使用 10 步去噪、预测 20 个动作、每块执行 3 个动作。模型推理期间，100 Hz 后台控制器持续跟踪最后一个绝对 TCP 目标。

控制回合并读取结果
------------------

每次复位后，在机器人控制机的实体键盘上按一次回车开始。0.5 秒内双击回车记录失败并复位，双击空格记录成功；每回合 60 秒超时。默认记录 20 个回合结果，最后一个结果记录后复位并退出。UR wrapper 会屏蔽已结束动作块中剩余的动作；RTC 会等待并丢弃旧预测，再开始新回合。

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

复位依次打开夹爪、停止 force mode、调用 ``servoStop`` 和 ``moveL``、重新设置负载、等待 0.25 秒、清零力传感器、再等待 0.50 秒，最后恢复目标跟踪。控制器保留 0.048 米位置误差限制和 4.0 m/s 速度误差限制，``rotation_error_clip_degrees: null`` 关闭旋转误差裁剪。这些行为均属于 UR 实现，Franka 的通用环境、变换和控制逻辑保持原样。

验证范围
--------

假 SDK 测试覆盖非零 TCP offset 下的局部动作合成、原始速度输入、微小动作保持、二值夹爪、复位顺序和力传感器清零。键盘测试覆盖成功、失败复位、超时和旧动作块取消。可使用合成观测验证 checkpoint 加载和同步、RTC 的 GPU 推理，不连接硬件。这些检查不衡量任务成功率、接触效果或真机时序，仍需在配置的 UR 工位验证。

与 VLA-Precision 执行流程对照
----------------------------------------

若需将机器人执行与上述 RLinf 分布式路径对照，可使用移植的 VLA-Precision 后端。``evaluations/realworld/vla_precision_ur/`` 包含从本机 VLA-Precision 复制的单臂 UR 主循环、记录、图像处理和 UR policy 变换。``provenance.json`` 记录来源版本、文件哈希和保留方法，文件哈希也包含来源工作区已有的修改；运行时不依赖原仓库。

该路径使用独立配置 ``evaluations/realworld/vla_precision_ur.yaml``。硬件、相机、任务、力控和回合设置与来源 YAML 一致，输出目录为 ``results/vla-precision-ur``。``model.checkpoint_dir`` 指向给定的 RLinf actor checkpoint，``model.norm_stats_path`` 指向配套统计量；``model.network`` 记录训练网络结构，包括预测 20 步、32 维内部动作、7 维物理动作和启用 RLT。推理与分布式 UR 入口一样采用 BF16/FP32 混合精度，``precision`` 和 ``embed_dtype`` 均设为 ``bf16``。统计量包含 19 维 state 和补齐后的 32 维 action，机器人仅接收前 7 维动作。

移植的 UR 入口还支持通过 ``model.network.openpi.torch_compile`` 选择编译模式。设为 ``max-autotune`` 可参考 OpenPI 的 PyTorch 设置，用 ``torch.compile`` 包装完整的 ``sample_actions`` 方法。首次推理会在现有的回合前预热阶段完成编译。预热期间每秒原地更新累计耗时，完成或中断时输出最终时长。``--check-model`` 的首次推理也使用同一计时显示。输入形状变化可能触发重新编译，编译后的浮点输出也可能与原路径存在差异。多组输入的离线检查发现，编译后夹爪输出的最大差异约为 0.684，因此选择该模式时还需验证任务行为。布尔值保持原有含义：``false`` 关闭局部算子编译，``true`` 编译局部算子并保留 Python 采样循环。其他可选字符串为 ``default``、``reduce-overhead`` 和 ``max-autotune-no-cudagraphs``。该扩展仅作用于移植的 UR 入口；通用模型加载器和分布式 UR/RTC 入口保持原有设置。

激活现有环境后，先检查依赖和路径，再启动。可选的 GPU 检查使用合成观测和固定扩散噪声，对比 OpenPI 原生 policy 与 RLinf 模型输入输出；两个检查命令均不连接硬件：

.. code-block:: bash

   source rltoken-setup/activate.sh
   bash evaluations/realworld/run_ur5e_vla_precision.sh --check
   bash evaluations/realworld/run_ur5e_vla_precision.sh --check-model
   # 连接硬件、复位、预热模型，然后等待 Enter：
   bash evaluations/realworld/run_ur5e_vla_precision.sh
   # 选择另一份独立 YAML：
   bash evaluations/realworld/run_ur5e_vla_precision.sh --config /path/to/ur.yaml

该入口在单进程内运行 OpenPI 原生 ``Policy.infer()``，通过适配层调用 RLinf 网络并加载其 checkpoint。RGB 相机读取、224 × 224 补边、state 拼接、位姿增量、夹爪线程、力控计算、终端键盘控制和退出保存视频提示均沿用 VLA-Precision。键盘输入来自当前终端，SSH 终端也可使用；双击 Enter 记录失败并复位，双击空格记录成功。默认仍为 20 回合，每回合最长 60 秒。

VLA-Precision 预测完整动作序列，同步执行前 3 个动作。它在动作执行时发送力控指令，模型推理期间没有 Python 后台控制线程刷新目标，这与上文 RLinf 控制器不同。移植路径也保留来源代码未使用 RLinf 工作空间裁剪的行为，不开启 RTC。来源 YAML 中的 ``model.infer_fps`` 未被硬件主循环读取，因此此处省略。这些行为用于形成对照，不改变通用机器人代码。

固定合成输入下输出一致，只能验证模型输入输出适配；真机对照用于检查相机、控制和执行时序。两条路径仍共享 RLinf 网络实现，因此仅凭本次对照不能排除其与官方 OpenPI 模型之间的差异，也不能确定 checkpoint 本身的质量。Franka 及通用环境、控制器均未修改。

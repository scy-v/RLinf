UR5e 真机评估
==============

本页介绍如何在 UR5e、大寰 PGI/PGE 串口夹爪和两台 RealSense 相机上部署 RLinf 的 OpenPI checkpoint。推理循环、机器人调度和相机驱动使用 RLinf 现有实现。硬件依赖参考了 `VLA-Precision <https://github.com/scy-v/VLA-Precision/tree/41edacde80d8c9d8146d610202299d71b0e79f7c>`_，没有引入其 HTTP 服务或 Actor 循环；UR 驱动参考其笛卡尔 PD 阻抗计算与 force-mode 执行。

安装与配置
----------

机器人控制机安装 UR 依赖；同时运行模型的机器还需安装 OpenPI。已有训练正在运行时，使用独立的虚拟环境。

.. code-block:: bash

   bash requirements/install.sh embodied --env ur --venv .ur-robot-venv
   # 同一台机器运行机器人控制和模型：
   bash requirements/install.sh embodied --model openpi --env ur --venv .ur-eval-venv

现有 UR 安装项提供 ``ur-rtde==1.6.3``、``pydhgripper==1.0.2``、串口和 RealSense 依赖。VLA-Precision 固定 RTDE 1.6.2；本次沿用 RLinf 已有的 1.6.3，不安装 Franka SDK。

``evaluations/realworld/realworld_ur5e_eval_pi05_sft.yaml`` 和 ``examples/embodiment/config/env/realworld_ur5e_sft_env.yaml`` 的默认硬件数值参考上述固定版本的 VLA-Precision 单 UR 原生推理 ``ur.yaml``。机械臂 IP 为 ``192.168.201.11``。外部相机 ``348522072391`` 映射到 ``exterior_image``，腕部相机 ``944622073455`` 映射到 ``wrist_image``；两者沿用 RLinf RealSense 默认的 640×480、15 FPS。初始 TCP 为 ``[0.392232, 0.104898, 0.335162, -3.14, 0.0, -1.57]``，使用基坐标系 XYZ 和 Euler XYZ，单位为米和弧度。PGI 串口为 ``/dev/ur5e_left_gripper``，力度 80、速度 100、原始位置范围 80–1000。负载质量为 1.601 kg，重心为 ``[0.011, -0.002, 0.052]`` 米。

工作空间按本任务配置为：X 为 0–0.90 米、Y 为 −0.45–0.45 米、Z 为 0–0.60 米；Euler XYZ 上下限以初始姿态为中心，分别偏移 ±30°、±30°、±90° 后转换为绝对弧度值。UR 使用自己的 TCP 复位实现，不做周期性关节复位；``enable_random_reset: true`` 和 ``reset_pose_range: [0.03, 0.03, 0.03, 0, 0, 0]`` 使 XYZ 各均匀随机 ±3 cm，姿态固定。范围的前三项单位是米，后三项是角度，沿用 VLA-Precision 的约定；初始位姿本身仍使用米和弧度。``target_ee_pose`` 与复位位姿一致，作为安全裁剪的姿态参考；键盘 wrapper 会屏蔽基于目标位姿的回合结束判定。任务描述、数据集和归一化文件继续使用本次黑色架子插槽任务的训练配置。

工作空间边界 ``ee_pose_limit_min``、``ee_pose_limit_max`` 各包含六个值：基坐标系 XYZ（米）和 Euler XYZ（弧度）。它们用于裁剪模型动作产生的 TCP 目标，限制位置和姿态，不代表随机复位范围。UR 的 TCP 复位路径使用配置的复位位姿及随机范围；这些边界不构成对所有物理运动的轨迹保护。当前默认值已填写；修改范围时保留跨越 −π 的连续角度区间，留空不会自动关闭限制。

默认在同一台机器运行 GPU 推理和机器人环境。分机部署时，按 :doc:`realworld` 配置 Ray 集群，将 ``env`` 放到机器人节点组，无需更换推理循环。UR 需要处于远程控制模式。``UR5eArm`` 接收绝对 XYZ + XYZW 目标，由后台循环以标称 100Hz 读取实测位姿和速度，计算截断后的笛卡尔 PD 力/力矩，转换到配置的力控任务坐标系，再调用 ``forceMode``。两次策略请求之间持续保持最近目标。增益、误差截断和旋转力矩除以控制频率的计算沿用 VLA-Precision 的运行控制公式。关节复位先停止 force mode，执行 ``moveJ``，再从实测位姿恢复阻抗跟踪；正常 TCP 动作不使用 ``moveL``。

运行策略
--------

激活评估环境，按 :doc:`realworld` 配置 Ray，然后通过 RLinf 入口运行第一阶段策略：

.. code-block:: bash

   source .ur-eval-venv/bin/activate
   bash evaluations/run_eval.sh realworld_ur5e_eval_pi05_sft

环境标称执行频率为 100Hz，对应参考实现的 ``action_fps``。每次模型调用采用 10 步去噪、执行 3 个动作，参考其原生推理配置；预测长度仍为 20，与本次 RLT 训练模型一致。``rollout.model.model_path`` 指向转换后的 OpenPI 基座，``runner.ckpt_path`` 加载 ``20261018-22:40:04`` 这次训练第 30000 步的完整第一阶段权重；部署到其他机器时在 YAML 中更新这些路径。归一化文件、两路图像和从 1 开始的 state 索引与本机 UR SFT 配置一致。state 按 TCP 位姿、速度、力、力矩和归一化实测夹爪位置排列，经 Euler wrapper 后为 19 维。夹爪动作为 [0, 1] 的绝对开度，0 关闭、1 打开，驱动负责限制设备行程。

RTC 使用同一个入口配置，在 YAML 中设置 ``runner.rtc.enabled: true`` 即可开启；设为 ``false`` 则使用同步推理，也是默认值。模型端的引导开关跟随该字段。RTC 复用现有 workers，重叠执行动作和重新规划，并处理动作块衔接。任务环境由 ``UR5eEnv-v1`` 选择，提供通用的复位和执行逻辑，与是否开启 RTC 独立。

键盘复位
--------

在机器人控制机上，首次复位完成后按一次回车开始推理。运行期间，0.5 秒内连续按两次回车即可结束当前回合，回到配置的初始位姿；每次复位完成后，都会等待另一次单次回车再开始推理。该操作只表示人工复位，不计为任务成功；长按回车的重复事件不会算作双击。使用环境 worker 所在机器连接的实体键盘，SSH 终端输入不会产生这些设备事件。存在多个键盘时，在启动 Ray 前选择有读取权限的设备：

.. code-block:: bash

   export RLINF_KEYBOARD_DEVICE=/dev/input/by-id/your-keyboard-event-kbd

``keyboard_reward_wrapper: double_enter_reset`` 选择通用的 ``KeyboardResetWrapper``，``keyboard_wrapper_kwargs.wait_for_start: true`` 启用开始前的等待，开始前通过 ``observe()`` 重新读取机器人和相机观测，不执行动作。``keyboard_wrapper_kwargs.double_press_interval`` 设置双击间隔；``auto_reset: true`` 让 ``RealWorldEnv`` 调用机器人自己的复位实现。wrapper 屏蔽目标位姿判定和单回合步数超时。``max_steps_per_rollout_epoch: 30000`` 仍限制整次评估的总执行步数，键盘复位会在该预算内开始新回合。同步推理会丢弃当前动作块中剩余的动作；RTC 会丢弃旧的待返回预测，从复位后的观测重新开始。这套控制独立于机器人驱动和 RTC 开关。

需要确认的动作语义
------------------

``UR5eEnv`` 与 ``FrankaEnv`` 分别继承公共的 ``CartesianEnv``，共用动作执行、观测和复位逻辑；各自实现硬件构建和夹爪行为。两者使用现有 ``RelativeFrame``、Euler wrappers 和相同的 TCP 增量合成方式。启用相对坐标变换时，RLinf 先用当前位姿的 adjoint 矩阵转换六维动作，再将位移加到实测 TCP 位置，将 Euler 增量旋转左乘到实测姿态。

VLA-Precision 的 UR 控制器使用 ``p_next = p + R.apply(delta_p)`` 和 ``R_next = R * EulerXYZ(delta_r)``。这种局部位姿合成与 RLinf 基于 adjoint 的六维向量转换存在差异。reset-relative 观测使用的坐标系也需要与采集数据核对；维度名称相同不代表物理语义相同。本次按要求保留 RLinf 的算法，真机执行前需结合采集代码讨论这些差异。

验证范围
--------

键盘测试通过管道模拟 evdev 设备，让实际监听器分别驱动 dummy UR、Franka 环境，覆盖长按过滤、按键时间戳、动作块取消和回合统计。假 SDK 测试覆盖笛卡尔 PD 力/力矩、误差截断、力控坐标变换、没有新策略请求时的目标保持、关节复位时 force mode 的停止与恢复、错误传播、连接释放、PGI 单位、RLinf 增量复用及 19 维模型观测。CI 中的 ``ur5e_mock_sac_cnn`` 使用原有 embodied 流程运行假硬件。这些测试不能证明真机成功率、接触控制效果或实际时序，仍需 UR5e 和相机验证。

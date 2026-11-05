# 本地 VLA-Precision：UR 锁轴与 Replay 路径核对

范围：只读检查 `/home/suchenyu/VLA-Precision` 当前源码和同名插槽任务配置，未启动机器人或核对历史运行配置。不能据此声称上游所有版本或历史训练都相同。

## 结论

当前力控锁轴没有自动变成策略输出或 Replay 的 r/p 清零：

| 设置 | 当前值 | 作用 |
|---|---|---|
| `robot_server.left_arm.selection_vector` | `[1,1,1,0,0,1]` | 控制坐标系的力控选择轴 |
| `robot.delta_action_mask` | `[1,1,1,1,1,1]` | 动作增量各维掩码，当前全保留 |
| server 返回的 `delta_action_mask` | `[1,1,1,1,1,1]` | robot 客户端会用它覆盖先读取的配置掩码 |

策略生成动作后，UR客户端做clip和动作掩码，将处理后的指令复制成`replay_chunk`并返回；环境将其放入`info.executed_action`，actor再写入Replay。这里的`executed_action`不是根据实测前后姿态计算的位移。因当前掩码全为1，策略输出的非零r/p仍可入库；人工干预另有按控制选择轴变换的路径，不能混同于策略输出处理。

策略输出还经过OpenPI反归一化，因此其r/p的物理尺度受数据统计量影响。不能仅凭缺少清零就认定它与RLToken MLP具有同样大的r/p偏移，更不能认定历史实验已发生相同故障。

## 源码定位

相对上述本地仓库根目录：

- `configs/stage2/tasks/insert_black_rack_into_blue_slot.yaml`：动作掩码与力控选择轴分别配置。
- `src/vla_precision/robotics/servers/ur/server.py:get_server_config`：返回全1的动作掩码，并说明控制坐标系选择轴不能直接当TCP局部动作掩码。
- `src/vla_precision/robotics/robots/ur.py:_load_server_configuration`、`_mask_action`、`execute_action_chunk`：加载掩码、处理指令、返回Replay指令。
- `src/vla_precision/robotics/environments/robot.py:finish_action_chunk_step`：返回`executed_action`和`delta_action_mask`。
- `src/vla_precision/acob_stream/actor.py`：接收上述字段，处理干预后将`replay_actions`存为trajectory的`actions`。
- `src/vla_precision/acob/agent.py:sample_actions`、`integrations/openpi/adapter.py:policy_output_actions`：策略输出经OpenPI变换，未将forceMode选择轴作为输出清零规则。

核对结果是“当前配置未启用动作层清零”，而非“代码不支持动作掩码”。尤其不能只修改任务YAML的`delta_action_mask`就认定有效：当前server返回值还会覆盖它。

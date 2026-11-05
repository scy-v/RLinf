# 梯度范数问题检索与恢复记录

2026-11-03，只读查询 RLinf/RLinf 的 GitHub issues 搜索接口，未限制 open／closed，因此包括已关闭 issue 和已合并 PR。

| 查询关键词 | 返回条数 |
|---|---:|
| `"clip_grad_norm"` | 0 |
| `"grad_norm" "critic"` | 20 |
| `"gradient clipping"` | 13 |
| `"fsdp_sac_policy_worker"` | 1 |

在本次命中的标题和正文中，没有找到明确针对 actor／critic 残留梯度混入同一范数的报告；这不证明该问题从未被讨论。原始返回见 [JSON](gradient_issue_search.json)。

相关 [PR #1558](https://github.com/RLinf/RLinf/pull/1558) 修复多卡分片范数未跨所有 shard 归约的问题，不是此次参数组混合问题。[Issue #504](https://github.com/RLinf/RLinf/issues/504) 讨论方差计算的自由度警告，同样不匹配。

按用户要求，撤回额外的 optimizer.zero_grad 操作及候选梯度隔离测试，保留原始整模型范数统计和裁剪行为。worker 和测试文件已与本地修改前基线逐字节核对一致，保留既有计时功能。恢复后 45 项相关 CPU／假硬件回归通过，Ruff 与空白检查通过；未连接机械臂。

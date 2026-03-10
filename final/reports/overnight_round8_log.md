# Overnight Round 8 Log (2026-03-09)

## Hypothesis
若机制重配主要由“JSON payload 本身”驱动，则 `system_json_pad` 与 `user_json_pad` 的机制变化幅度应接近；否则应呈现 user 位点特异差异。

## Experimental Design
- 扩展注入模式：`system_json_pad`（与 `user_json_pad` 同 payload，不同角色位点）。
- 全量重跑 shift robustness（v4）：`orig + user_json_pad + system_json_pad + user_pad_short`。
- 新增可复现脚本：
  - `experiments/analyze_toolcall_shift_deltas.py`（paired bootstrap + common intersection）
  - `experiments/analyze_toolcall_stratum_transitions.py`（orig/system/user 分层迁移）
- 进一步运行 `system_json_pad` node ablation + stratification，并做三方节点重分配对比。

## Results
1. 组级对比（paired bootstrap, n=139）：
- `user_json_pad - orig`:
  - `all_heads drop_full_nec`: CI `[-0.2354, -0.1985]`
  - `all_mlps drop_full_nec`: CI `[+0.1905, +0.2436]`
- `system_json_pad - orig`:
  - `all_heads drop_full_nec`: CI `[-0.0641, -0.0471]`
  - `all_mlps drop_full_nec`: CI `[+0.0584, +0.1491]`
- `user_json_pad - system_json_pad`:
  - `all_heads drop_full_nec`: CI `[-0.1792, -0.1315]`
  - `all_mlps drop_full_nec`: CI `[+0.0880, +0.1434]`

2. 节点级三方差异（drop_full_nec）：
- `user_json - system_json` 显著上升：
  - `MLP27` CI `[+0.1189, +0.1486]`
  - `MLP20` CI `[+0.0684, +0.0806]`
- `user_json - system_json` 显著下降：
  - `L21H12` CI `[-0.0640, -0.0500]`
  - `L24H6` CI `[-0.0525, -0.0338]`

3. 分层迁移轨迹：
- `orig -> system_json`: 8/10 节点保持原层级（变化较小）。
- `system_json -> user_json`: 出现集中迁移，典型为
  - `L17H8/L21H1/L21H12` 从 backbone 降至 weak/redundant；
  - `MLP27` 从 weak/redundant 升至 backbone。

## Interpretation
- 新证据支持“user 通道特异机制重配”，而非“任意 JSON 注入都同等重配”。
- 这使分组策略从“静态语义标签”升级为“条件化机制分组”成为必要：
  - stable backbone 需要显式绑定分布条件。

## Decision
- 在最终结论中明确两层结论：
  - function-level robust（高置信）
  - mechanism-level non-stationary and context-dependent（高置信）
- 后续优先：将 grouping 评估升级为“condition-aware grouping”，并给出跨条件失效边界。

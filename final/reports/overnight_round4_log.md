# Overnight Round 4 Log (2026-03-08)

## Hypothesis
若节点分层是稳健机制而非阈值偶然，提升 `gap_min`（更严格样本筛选）后分层排序应基本保持。

## Experimental Design
- 在 `gap_min=0.8` 下重跑全量节点消融（同 Round 1 方法）。
- 与 `gap_min=0.5` 逐节点对比 `drop_full_nec/drop_full_suff/node_suff/node_nec`。

## Results
- `gap_min=0.8` 有效样本 `138`（vs `139`）。
- 关键指标几乎不变（逐节点 delta 量级约 `1e-3`）。
- 主干排序保持：`L21H12 > L17H8 > L21H1 > L23H6`。
- MLP 的弱/负必要性结论保持不变。

## Interpretation
- 节点分层结论对 `gap` 过滤阈值具有高稳定性，不是边界样本驱动的脆弱现象。

## Decision
- 将“阈值敏感性低”作为支持证据写入最终结论；下一步可转向外部分布迁移验证。

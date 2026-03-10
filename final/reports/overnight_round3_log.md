# Overnight Round 3 Log (2026-03-08)

## Hypothesis
最终分组应同时反映“结构稳定”与“因果必要”，单轴分组会混淆主干与冗余分量。

## Experimental Design
- 合并 Round 1 的节点必要性结果与 Round 2 的结构频率结果。
- 定义双阈值分层：
  - 稳定阈值：`stability_freq >= 0.70`
  - 必要性阈值：`drop_full_nec_median >= 0.02`
- 生成二维散点分层图与分层表。

## Results
- `stable_necessary_backbone`:
  - `L21H12, L17H8, L21H1, L23H6`
- `stable_but_weak_or_redundant`:
  - `L20H5, L24H6, MLP22, MLP27`
- `unstable_weak`:
  - `MLP11, MLP20`

## Interpretation
- 头部主干 (`L17H8/L21H1/L21H12/L23H6`) 同时满足“稳定 + 必要”。
- `L20H5/L24H6` 更像稳定辅助路由，必要性低于主干。
- `MLP22/MLP27` 虽结构稳定，但在 full-circuit 必要性上偏弱或负，适合归入“稳定辅助/冗余写入器”。

## Decision For Next Round
- 最终论文叙事采用双轴分层，而非单一语义标签。
- 后续实验优先验证该分层在外部分布/新样本上的可迁移性。

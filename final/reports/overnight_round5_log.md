# Overnight Round 5 Log (2026-03-08)

## Hypothesis
若“共性电路”不仅拟合训练样本，还具有可迁移性，则在 held-out 子集上仍应显著优于随机同规模节点集。

## Experimental Design
- 3 个随机划分（seed: 11/22/33），每次 70% 样本建 core、30% 样本测试。
- 在测试集评估：
  - train-derived core 的 `suff/nec` 中位数
  - 随机同规模基线 `suff` 中位数
  - `core - random` 中位差

## Results
- 跨 split 中位汇总：
  - `test_suff_median = 0.943`
  - `test_nec_median  = 0.963`
  - `test_random_suff_median = 0.396`
  - `test_global_minus_random_median = 0.485`
- train-derived core 与参考全局 core 的节点 Jaccard 中位数：`0.90`。

## Interpretation
- 在未参与建模的 held-out 样本上，core 仍明显优于随机对照，支持“机制可迁移”而非过拟合单样本。

## Decision
- 将“held-out transfer 显著优于随机”纳入高置信证据层。
- 下一步可继续扩展到更强分布偏移（如新模板或人工对抗样本）。

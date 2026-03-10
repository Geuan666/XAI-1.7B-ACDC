# Overnight Round 9 Log (2026-03-09)

## Hypothesis
如果采用条件化分层（condition-aware grouping），应在对应条件上提升 backbone 的必要性指标，并在非对应条件上出现可解释的性能代价。

## Experimental Design
- 基于已得到的 stratification 分别构造两套分组：
  - `useraware_strata`（来自 user-json 节点分层）
  - `systemaware_strata`（来自 system-json 节点分层）
- 分别在对应条件下重跑 group-level shift robustness：
  - `orig + user_json_pad`（useraware）
  - `orig + system_json_pad`（systemaware）
- 新增汇总脚本：
  - `experiments/analyze_toolcall_condition_aware_grouping.py`

## Results
1. user-aware 在目标条件上有收益，但伴随明显跨条件代价：
- `user_json_pad` 上 backbone `drop_full_nec`：
  - static: `0.0962`
  - user-aware: `0.1345`（`+0.0383`）
- 配对 bootstrap（n=139）显示该增益稳定：
  - `drop_full_nec` delta CI `[+0.0122, +0.0606]`
  - `drop_full_suff` delta CI `[+0.0114, +0.0366]`
- `orig` 上同一 user-aware backbone 明显退化：
  - 相对 static 下降 `-0.2499`
- 配对 bootstrap（n=139）：
  - `drop_full_nec` delta CI `[-0.2623, -0.2222]`
  - `drop_full_suff` delta CI `[-0.1964, -0.1724]`

2. system-aware 未在目标条件上超过 static：
- `system_json_pad` 上 backbone `drop_full_nec`：
  - static: `0.2083`
  - system-aware: `0.1739`（`-0.0344`）
- `orig` 上代价中等：相对 static `-0.0667`。
- 配对 bootstrap（n=139）进一步确认均为显著下降：
  - `system_json` 上 `drop_full_nec` delta CI `[-0.0455, -0.0364]`
  - `orig` 上 `drop_full_nec` delta CI `[-0.0667, -0.0600]`

3. 解释性结论：
- 条件化分组并非自动更好；它对分组构建策略（尤其节点数量/约束）非常敏感。
- 当前 user-aware 组只有 2 个节点，带来 target 提升但泛化性差，存在过拟合风险。

## Interpretation
- “condition-aware grouping”是必要方向，但必须加入结构约束（如最小节点数、正则化、跨条件一致性约束），否则会牺牲跨条件可解释性。

## Decision
- 将 Round 9 作为保留负结果：
  - user-aware: target gain + cross-condition collapse
  - system-aware: no target gain
- 下一轮应转向“约束化条件分组”而非直接使用无约束 strata。

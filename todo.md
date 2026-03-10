# TODO: 电路鲁棒性与开放探索（2026-03-09 夜间迭代版）

## 0. 已完成进展（本轮实测）

- [x] 审阅 `final/` 全部图表、报告、表格，确认当前主结论成立但存在三类风险：
  1) discovery/evaluation 数据泄漏风险（同一批样本既发现又验证）；
  2) 阈值敏感性未系统量化；
  3) “通用电路”与“多机制并存”边界未充分刻画。
- [x] 新增脚本：`Automatic-Circuit-Discovery/experiments/cross_validate_toolcall_core.py`
  - 功能：train/test 分裂发现 + held-out suff/nec + 同规模随机对照。
- [x] 运行 6-split CV（基线阈值 n0.50/e0.35）：
  - `test_suff_median_mean = 0.9330`
  - `test_nec_median_mean = 0.9514`
  - `test_delta_vs_random_mean = 0.8006`
  - `core_pairwise_jaccard_mean = 0.8952`
- [x] 运行阈值敏感性 CV sweep（`n0.40/e0.25`, `0.45/0.30`, `0.55/0.40`, `0.60/0.45`）：
  - 汇总：`Automatic-Circuit-Discovery/experiments/results/toolcall_crossval_sweep_summary.csv`
  - tradeoff 图：`Automatic-Circuit-Discovery/experiments/results/toolcall_crossval_tradeoff.png`
- [x] 对最佳 held-out fidelity 候选（`n0.40/e0.25`）做全流程重跑：
  - aggregate replay：`global_suff=0.9661`, `global_nec=0.9747`, `global-random=0.6742`
  - 输出目录：`Automatic-Circuit-Discovery/experiments/results/toolcall_q1_q164_semantic_roles_n0.40_e0.25`
- [x] 手工节点消融（`toolcall_manual_core_ablation.json`）确认：
  - 12-node 扩展 core 指标最高；
  - 去掉 `MLP19` 或 `MLP25` 均显著掉点；
  - 去掉两者后退回原 10-node 水平。
- [x] cluster-sim 阈值稳定性检查：
  - `sim_th=0.35 -> 7 clusters`
  - `sim_th=0.45 -> 16 clusters`
  - `sim_th=0.55 -> 75 clusters`
  - 结论：聚类结构对相似阈值敏感，不能直接当“固定机制个数”证据。
- [x] overnight_round2 best-config 全流程补齐（含 contrast tracing）：
  - best config: `node=0.55, edge=0.40`（8-node core）。
  - contrast tracing 已补齐：`contrast_token_trace_report.json`（139 样本，contrast pos=133 一致）。
- [x] P1 边稳定性修剪实测（保留负结果）：
  - 新脚本：`experiments/prune_toolcall_core_by_path_stability.py`。
  - 按规则 `edge_ratio_median > 0.03` 且 `positive_frac >= 0.90` 并施加 Input->Output 路径约束后：
    - core `8/20 -> 7/17`，删除 `L20H5` 分支；
    - replay 从 `suff/nec=0.9036/0.9241` 降到 `0.8615/0.8667`；
    - role-group full-core 从 `0.9070/0.9206` 降到 `0.8644/0.8679`。
  - 更严格阈值 `edge_ratio_min=0.05`（6-node）进一步下降到 `0.8491/0.8525`。
  - 结论：该修剪目前不应替代 best 8-node 主结果。
- [x] P2 cluster-aware transfer（含 overlap-confound）补齐：
  - 新增脚本能力：`evaluate_toolcall_cluster_transfer.py` 支持 compact source 构造、source-overlap 矩阵、overlap-vs-transfer 相关性。
  - best aggregate baseline：`cluster_jaccard_mean=0.561`, `cross/within suff median=0.985`。
  - compact top-2 探针：`cluster_jaccard_mean=0.444`, `cross/within suff median=0.957`。
  - 低重叠对子集（`J<=0.34`, `n=30`）仍保持高迁移（`suff_vs_within_median=0.978`）。
  - 同时存在 pair 级波动边界（`suff_vs_within_ratio` 约 `0.666~1.445`），不支持“完全单一 universal”强表述。

## 1. 当前结论分层（更新）

- 高置信：
  - 存在可迁移的强核心机制（held-out suff/nec 明显高于随机对照）。
  - L20/L21/L24 + MLP27 主干机制稳定。
- 中置信：
  - 扩展 12-node core 提升恢复/必要性，但引入额外辅助通路，解释复杂度上升。
- 低置信（边界）：
  - “单一且完全统一全部样本”的强版本结论仍不足；当前更稳妥是“共享主干 + 簇对簇方向性专化”。

## 2. 下一阶段最高优先级（P0）

- [x] 构建“双轨结论”并统一证据模板：
  - 轨道 A（紧凑解释）：10-node compact core（高可解释性）。
  - 轨道 B（高保真恢复）：12-node extended core（更高 suff/nec）。
- [x] 统一对比表必须包含：
  - `n_nodes`, `n_edges`, held-out `suff/nec/min/hmean`, `delta_vs_random`, cross-split jaccard。
- [ ] 给出 paper-ready 的主叙事判定：
  - 若审稿偏“机制简洁”：主文用 compact，extended 放补充；
  - 若审稿偏“恢复性能”：主文用 extended，同时强调复杂度代价。

## 3. 方法学强化（P1）

- [x] 基于 path-patching 稳定性做边级修剪：
  - 规则：`edge_ratio_median > 0.03` 且 `positive_frac >= 0.90`（trimmed）。
  - 增补路径约束后形成 7-node/17-edge 候选，但 fidelity 明显下降。
- [x] 修剪后重跑 replay + role-group，确认存在性能退化（负结果已记录）。

## 4. 异质性与必要性边界（P2）

- [x] 做 cluster-aware 验证：
  - 训练某 cluster core，测试到其他 cluster，量化跨簇迁移衰减。
  - 输出“机制共享度矩阵”（用于回答是否 truly universal）。
- [x] 将负结果并入正式结论（不再隐藏）。

## 5. 图表质量门槛（P3）

- [ ] 每张图必须回答：
  1) 支撑了哪个命题？
  2) 统计不确定性是否可见？
  3) 是否有强对照？
- [ ] 不满足以上任一条：重画或降级到附录。

## 6. 本轮新增关键工件索引

- `Automatic-Circuit-Discovery/experiments/cross_validate_toolcall_core.py`
- `Automatic-Circuit-Discovery/experiments/results/toolcall_q1_q164_crossval_v1/`
- `Automatic-Circuit-Discovery/experiments/results/toolcall_q1_q164_crossval_n0.40_e0.25/`
- `Automatic-Circuit-Discovery/experiments/results/toolcall_crossval_sweep_summary.csv`
- `Automatic-Circuit-Discovery/experiments/results/toolcall_crossval_tradeoff.png`
- `Automatic-Circuit-Discovery/experiments/results/toolcall_q1_q164_aggregate_n0.40_e0.25/`
- `Automatic-Circuit-Discovery/experiments/results/toolcall_q1_q164_semantic_roles_n0.40_e0.25/`
- `Automatic-Circuit-Discovery/experiments/results/toolcall_manual_core_ablation.json`
- `final/tables/compact_vs_extended_summary.csv`
- `final/figures/compact_vs_extended_overview.png`

## 7. 下一个单一最高优先动作（Next 1）

- [ ] 推进 P3：按“命题-不确定性-强对照”三条门槛重审现有全部图表，清理不达标图并补充一页主文级 claim-boundary panel。

# TODO: 电路鲁棒性与开放探索（Agent 自主迭代版）

## 0. 对齐摘要（当前共识）

- 现有电路结果“有进展但证据还不够硬”。
- 核心问题 1：电路鲁棒性仍欠缺，充分性与必要性的论证还不令人满意。
- 核心问题 2：节点语义分组的可信度不足，现有实验不能充分支撑分组结论。
- 开放假设：可能存在更好的分组方式；也可能当前任务本身不存在稳定共性电路。
- 研究目标：在论文语境下，尽最大可能验证并逼近“通用电路”结论，同时接受反例和负结果。

## 1. 总目标（What）

- 目标 A：系统增强现有电路的鲁棒性证据链。
- 目标 B：更严格验证电路的充分性与必要性。
- 目标 C：重审并验证节点语义分组是否真实反映因果功能。
- 目标 D：开展更广泛的可解释性探索，挖掘新的、可发表的发现。

## 2. 工作规则（Rules, 不预设实现）

- 每一轮迭代必须显式记录：
  - 假设
  - 实验设计
  - 结果
  - 解释
  - 下一步决策
- 对不支持假设的结果，必须保留并分析，不做“只报喜不报忧”。
- 允许推翻已有结论，包括“通用电路可能不成立”的结论。
- 结论必须可复现：命令、参数、数据切片、随机种子、版本信息可追溯。
- 优先复用已有脚本/结果；若新增脚本，应服务于提高探索效率与证据质量。

## 3. 证据标准（Definition of Done）

- 鲁棒性：关键结论在多扰动、多样本切片、多随机种子下趋势稳定。
- 充分性：仅保留候选电路时，核心行为保持在可接受范围。
- 必要性：去除候选关键节点/边后，目标行为出现显著且可重复退化。
- 语义分组有效性：分组与因果贡献相匹配，且能区分语义机制与统计伪相关。
- 泛化性：核心结论在不同子任务或输入分布上尽量成立。

## 4. 六小时自主探索协议（Timebox）

- 连续探索窗口：9 小时。
- 建议循环粒度：每 30 到 60 分钟为一轮。
- 每轮输出至少两条：
  - 本轮最重要发现
  - 下轮单一最高优先级动作
- 若连续两轮无明显增益，必须主动切换假设或评估维度。
- 中途（约第 3 小时）进行一次方向收敛：保留高价值路线，停止低收益路线。
- 9 小时结束时形成阶段报告：
  - 高/中/低置信结论分层
  - 支持证据与反证并列
  - 当前“通用电路”判断（成立/部分成立/不成立）
  - 下一阶段最小可执行计划

## 5. 待办清单（仅列任务，不限方法）

- [x] 建立并统一“鲁棒性 + 充分性 + 必要性”评估看板。
- [x] 对现有候选电路开展多维度压力测试，定位最脆弱环节。
- [x] 针对语义分组提出并比较多种可检验方案，识别最可信解释。
- [x] 检查是否存在跨样本可迁移的稳定子电路结构。
- [x] 系统记录负结果与边界条件，明确结论成立范围。
- [x] 挖掘额外可解释性发现并评估论文价值（新现象、新机制、新反例）。
- [x] 输出阶段性研究日志与可复现工件索引。

## 6. 研究纪律（必须遵守）

- 证据优先于叙事，不为“好看结论”牺牲严谨性。
- 负结果与不确定性必须进入正式结论。
- 清晰区分：观察现象、因果证据、推测解释。

## 7. Round 1 执行记录（2026-03-08）

### 7.1 已完成实验

- 节点级因果消融（全量 139 样本）：`node alone` + `full minus node`
  - 输出：`final/reports/node_ablation_report.json`
- 鲁棒性切片看板（gap/长度/hard-easy/随机半切）
  - 输出：`final/reports/robustness_dashboard_report.json`
- 数据驱动分组对比评估（与语义分组同评估协议）
  - 输出：`final/reports/role_group_data_driven_report.json`
- 阶段研究日志
  - 输出：`final/reports/overnight_round1_log.md`

### 7.2 本轮关键结论（供下一轮直接使用）

- `L17H8/L21H1/L21H12` 构成更强“必要性主干”；其 `drop_full_nec` 排名前列。
- MLP 组中 `MLP20/MLP22/MLP27` 在 full-circuit 中呈负或弱必要性（存在冗余/干扰信号）。
- `full_core` 的 suff/nec 在主要切片中整体稳定（中位数约 0.89~0.92）。

### 7.3 Round 2 最高优先级动作（单一主线）

- 主线：验证“主干可迁移性”而不是只在当前分布成立。
- 动作：
  - 在不同 `gap` 阈值、不同样本子集、不同聚合阈值下重建 core 并重放评估；
  - 比较 core 节点交并比 + suff/nec 变化；
  - 明确“稳定共性子电路”的最小成立范围与失效边界。

## 8. Round 2 执行记录（2026-03-08）

### 8.1 已完成实验

- 聚合超参数稳定性 sweep（27 组）：
  - `gap_min ∈ {0.4,0.5,0.6}`
  - `core_node_th ∈ {0.45,0.50,0.55}`
  - `core_edge_th ∈ {0.30,0.35,0.40}`
  - 输出：`final/reports/core_stability_report.json`

### 8.2 本轮关键结论

- 跨参数的核心结构稳定性较高：
  - node Jaccard mean/median = `0.870 / 0.909`
  - edge Jaccard mean/median = `0.853 / 0.852`
- 频率 >= 0.70 的稳定节点共 8 个，且均在多数变体中重复出现。
- 与 Round 1 对照后确认：存在“结构稳定但必要性偏弱”节点（典型 MLP），需要在结论中单独分层。

### 8.3 Round 3 最高优先级动作（单一主线）

- 生成“结构稳定性 × 因果必要性”二维分层报告，并据此重写最终分组叙事与图表排序。

## 9. Round 3 执行记录（2026-03-08）

### 9.1 已完成实验

- 完成节点双轴分层（结构频率 × 因果必要性）：
  - 输出：`final/reports/node_stratification_report.json`
  - 图：`final/figures/node_stratification_scatter.png`

### 9.2 本轮关键结论

- 明确识别 `stable_necessary_backbone`（4 个关键头）与 `stable_but_weak_or_redundant`（含部分 MLP）。
- 当前最稳妥的论文叙事不再是“所有核心节点同等关键”，而是“主干 + 辅助/冗余分层”。

### 9.3 Round 4 最高优先级动作（单一主线）

- 在新的样本分布（或更严格子集）上复验 Round 3 分层是否保持不变，验证可迁移性边界。

## 10. Round 4 执行记录（2026-03-08）

### 10.1 已完成实验

- 更严格样本过滤复验：`gap_min=0.8` 下重跑节点消融。
  - 输出：`final/reports/node_ablation_gap08_report.json`
  - 对比表：`final/tables/node_ablation_gap08_vs_gap05.csv`

### 10.2 本轮关键结论

- 分层结果对 gap 阈值高度稳定，节点排序和正负必要性方向基本不变。
- “head 主干必要 + 部分 MLP 冗余/辅助”结论未被阈值变化推翻。

### 10.3 Round 5 最高优先级动作（单一主线）

- 在外部分布或新构造样本上复验 `stable_necessary_backbone`，量化可迁移性与失效边界。

## 11. Round 5 执行记录（2026-03-08）

### 11.1 已完成实验

- Held-out transfer（3 个随机划分）：
  - 70% 样本建 core，30% 样本测试；
  - 与随机同规模节点集做对照；
  - 输出：`final/reports/core_transfer_report.json`

### 11.2 本轮关键结论

- 测试集中位（跨 split）：
  - `suff=0.943`, `nec=0.963`, `random_suff=0.396`, `core-random=0.485`
- train-derived core 与参考 core 的节点 Jaccard 中位数约 `0.90`。
- 说明电路并非仅拟合建模子集，具有较强可迁移性。

### 11.3 Round 6 最高优先级动作（单一主线）

- 设计更强分布偏移或对抗样本，继续验证当前“稳定主干 + 辅助冗余”分层是否仍成立。

## 12. Round 6 执行记录（2026-03-09）

### 12.1 已完成实验

- 分布偏移鲁棒性评估（同向注入中性上下文）：
  - 模式：`orig / user_pad_short / user_json_pad`
  - 输出：`final/reports/shift_robustness_report.json`
  - Bootstrap 对比：`final/tables/shift_robustness_mode_delta_bootstrap.csv`

### 12.2 本轮关键结论

- `full_core` 在 shift 下依旧强（function-level robustness 持续成立）。
- 但机制层面发生显著重配：
  - `all_heads` 组内必要性下降；
  - `all_mlps` 组内必要性显著上升；
  - `stable_necessary_backbone` 仍为正必要，但幅度下降。
- 这给出关键边界条件：电路有效性可稳定，但内部实现路径并非固定不变。

### 12.3 Round 7 最高优先级动作（单一主线）

- 在 shift 样本上做 node-level ablation，定位“机制重配”具体落到哪些节点，并更新分组规则。

## 13. Round 7 执行记录（2026-03-09）

### 13.1 已完成实验

- 在 `user_json_pad` shift 上完成 node-level ablation（`n=139`）。
  - 输出：`final/reports/node_ablation_user_json_report.json`
  - 表：`final/tables/node_ablation_user_json_summary.csv`
- 完成 `orig vs user_json` 配对重分配 bootstrap 与主图：
  - 表：`final/tables/node_reallocation_user_json_vs_orig_delta_summary.csv`
  - 图：`final/figures/node_reallocation_user_json_vs_orig_barh.png`
- 完成 user-json 条件下 stratification：
  - 输出：`final/reports/node_stratification_user_json_report.json`
  - 表：`final/tables/node_stratification_user_json.csv`

### 13.2 本轮关键结论

- user-json 条件下出现显著节点级必要性重排：
  - 上升最强：`MLP27`, `MLP20`；
  - 下降最强：`L21H12`, `L24H6`。
- 原始“head backbone”为主的叙事不再全局成立，必须升级为条件化机制叙事。

### 13.3 Round 8 最高优先级动作（单一主线）

- 加入强对照（`system_json_pad`）以区分“payload 结构效应”与“user 位点特异效应”。

## 14. Round 8 执行记录（2026-03-09）

### 14.1 已完成实验

- 扩展并全量重跑 shift robustness（v4）：
  - 模式：`orig / user_json_pad / system_json_pad / user_pad_short`
  - 输出：`final/reports/shift_robustness_v4_report.json`
  - 对比：`final/tables/shift_robustness_v4_mode_delta_bootstrap.csv`
- 在 `system_json_pad` 上完成 node-level ablation + stratification：
  - 输出：`final/reports/node_ablation_system_json_report.json`
  - 输出：`final/reports/node_stratification_system_json_report.json`
- 完成三方节点重分配与分层迁移分析：
  - 表：`final/tables/node_reallocation_system_vs_orig_delta_summary.csv`
  - 表：`final/tables/node_reallocation_user_vs_system_delta_summary.csv`
  - 表：`final/tables/node_stratum_trajectories.csv`
  - 图：`final/figures/node_stratum_trajectories_heatmap.png`

### 14.2 本轮关键结论

- `system_json_pad` 相比 `user_json_pad` 触发的机制重配明显更弱。
- `user_json_pad - system_json_pad` 仍有显著对比（CI 排除 0）：
  - heads 必要性继续下降；
  - MLP 必要性继续上升。
- 说明重配不是“任意 JSON 注入”即可复现，存在 user 通道特异性。

### 14.3 Round 9 最高优先级动作（单一主线）

- 基于 `orig/system_json/user_json` 三条件，正式构建并验证 condition-aware grouping，输出跨条件失效边界矩阵。

## 15. Round 9 执行记录（2026-03-09）

### 15.1 已完成实验

- 基于 user/system 条件分层，分别构建 `useraware` 与 `systemaware` 分组并重跑 group-level shift robustness：
  - `final/reports/shift_robustness_useraware_v1_report.json`
  - `final/reports/shift_robustness_systemaware_v1_report.json`
- 新增条件化分组汇总分析：
  - `final/reports/condition_aware_grouping_report.json`
  - `final/tables/condition_aware_backbone_comparison.csv`
  - `final/tables/condition_aware_grouping_delta_bootstrap.csv`

### 15.2 本轮关键结论

- `useraware` 在目标条件 `user_json` 上有提升（backbone `drop_full_nec` 增加），但在 `orig` 上出现明显退化，存在过拟合风险。
- `systemaware` 未在目标条件 `system_json` 超过 static 分组。
- 配对 bootstrap 显示上述增益/退化均稳定（CI 排除 0），因此该负结果可信。
- 结论：condition-aware 方向必要，但“无约束分组重定义”不足以给出稳定改进。

### 15.3 Round 10 最高优先级动作（单一主线）

- 设计“约束化条件分组”（最小节点数 + 跨条件正则），验证是否能同时保留 target 增益并控制 off-target 退化。

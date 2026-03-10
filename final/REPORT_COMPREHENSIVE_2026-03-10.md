# Tool-Call Circuit 最终综合报告（2026-03-10）

> 本报告目标：在 `final/` 内完成一次“去重/对齐/补全/解释”后的单文档交付，确保你只看这一份即可追踪全局脉络、指标定义、图表结论与风险点。

## 0. 数据审计结论（先回答“有没有冲突/缺失”）

- 审计范围：`final/` 全量文件（143 个），其中图 53、表 49、JSON 报告 27。
- 重复审计：同名冲突 0，同内容重复组 0。
- 引用审计：`reports/*.json` 的 `artifacts` 引用全部已指向 `final/` 且可落盘访问（outside=0, missing=0）。
- 版本并存说明：`shift_robustness` 同时包含 `v1/v4/useraware/systemaware`，属于“方法迭代并保留历史”，不是冲突；本报告以 `v4 + condition-aware` 为最新主线，v1 作为对照。

## 1. 统一指标字典（所有图表共用）

```text
obj(x) = logit_x("<tool_call>") - logit_x(distractor)
gap    = obj(clean) - obj(corrupt)
suff   = (obj(corrupt + clean_patch) - obj(corrupt)) / gap
nec    = (obj(clean) - obj(clean + corrupt_patch)) / gap
drop_full_* = full_* - minus_*
edge_ratio  = source_ratio - blocked_ratio
```

- `obj`：`logit(<tool_call>) - logit(distractor)`。
- `gap`：`obj(clean) - obj(corrupt)`，仅在 `gap > gap_min` 样本上做后续统计。
- `suff_ratio`：`(obj(corrupt+clean_patch)-obj(corrupt))/gap`，越高表示“足以恢复”。
- `nec_ratio`：`(obj(clean)-obj(clean+corrupt_patch))/gap`，越高表示“必要性”强。
- `full_*`：在完整 core 电路上测得的 suff/nec 指标。
- `node_suff_ratio / node_nec_ratio`：只保留单节点（或单组）时的 suff/nec。
- `minus_node_*`：从 full_core 中移除该节点（或该组）后的 suff/nec。
- `drop_full_suff / drop_full_nec`：`full - minus_node`，越大表示移除后退化越明显。
- `delta_median`：配对条件差（shifted - baseline）的中位数（按样本配对）。
- `delta_median_boot_mean / ci_lo / ci_hi`：bootstrap 对中位差的估计与95%置信区间；CI 跨 0 代表证据不足。
- `source_ratio / blocked_ratio / edge_ratio`：边级路径 patch：`edge_ratio = source_ratio - blocked_ratio`。
- `positive_frac`：边在样本中 `edge_ratio>0` 的比例。
- `frequency`：节点/边在超参数变体中出现频率（稳定性）。
- `node_jaccard / edge_jaccard`：两套 core 的重叠度，用于稳定性比较。
- `stratum`：按 `stability_freq >= 0.7` 与 `drop_full_nec_median >= 0.02` 分层。

## 2. 主线结论（图串联）

### 2.1 全局 core 先成立，再看细节

- 样本：总计 `164`，有效 replay `139`。
- 全局 core：`10` 节点 / `23` 边（阈值 node=0.5, edge=0.35）。
- replay：core suff 中位 `0.907`，nec 中位 `0.924`，随机同规模 suff 中位 `0.328`，差值 `0.583`。

![Global Core](context/final_circuit_global_core.png)

### 2.2 节点语义：读侧与写侧分工可复现

- Tool-tag 读侧最强：`L24H6`（tool_call_tags_ratio 中位 `0.443`）。
- User-query 读侧最强：`L20H5`（user_block_ratio 中位 `0.185`）。
- 写侧上，`target_logit_delta` 与 `distractor_logit_delta` 联合区分 Booster 与 Suppressor。

![Semantic Read](figures/semantic_read_causal_heatmap.png)

![Semantic Attention Delta](figures/semantic_attention_delta_heatmap.png)

![Semantic Write](figures/semantic_write_target_delta.png)

### 2.3 组级因果：heads 主干，MLP 协同

- `full_core`：suff/nec 中位 `0.907` / `0.924`。
- `all_heads`：drop_full_nec 中位 `0.440`。
- `all_mlps`：drop_full_nec 中位 `-0.127`（可为负，表示冗余/替代效应）。

![Role Group Causal](figures/role_group_causal_heatmap.png)

![Role Group Suff](figures/role_group_sufficiency.png)

![Role Group Drop](figures/role_group_necessity_drop.png)

### 2.4 边级中介：路径可定位

- trimmed 最强边：`MLP27->Residual Output: <tool_call>`，edge_ratio 中位 `0.375`，positive_frac `1.000`。
- 结合 bar/heatmap 可看到 Input/中层 Router/Writer 到 Output 的中介链。

![Edge Heatmap Trimmed](figures/path_patch_edge_heatmap_trimmed.png)

![Edge Bar Trimmed](figures/path_patch_edge_bar_trimmed.png)

### 2.5 稳健性与迁移：功能稳，机制会重分配

- v4 显著上升（示例）：`user_json_pad-orig / all_mlps / drop_full_nec_abs` = `1.800`。
- v4 显著下降（示例）：`user_pad_short-orig / all_heads / drop_full_nec_abs` = `-2.191`。
- user_json vs orig：上升 `MLP27` Δ=0.141；下降 `L21H12` Δ=-0.077。

![Shift v4 Drop-NEC](figures/shift_robustness_v4_drop_nec_heatmap.png)

![Node Reallocation User-vs-Orig](figures/node_reallocation_user_json_vs_orig_barh.png)

### 2.6 条件自适应分组：目标收益与跨域代价并存

- user-aware 在 user_json 目标域收益（drop_full_nec）≈ `0.038`。
- user-aware 在 orig 域代价（off-target penalty）≈ `-0.250`。
- system-aware 在 system_json 目标域收益 ≈ `-0.034`。
- 因此 condition-aware 不是“无条件更好”，而是“按目标域优化会牺牲跨域泛化”。

![Shift UserAware Drop-NEC](figures/shift_robustness_useraware_v1_drop_nec_heatmap.png)

![Shift SystemAware Drop-NEC](figures/shift_robustness_systemaware_v1_drop_nec_heatmap.png)

## 3. 全量图片逐项解释（53/53）

### 图：`context/final_circuit_global_core.png`
该图用于对应模块的可视化诊断，详见同名表与 report.json 指标。

![final_circuit_global_core.png](context/final_circuit_global_core.png)

### 图：`figures/contrast_token_trace.png`
位置级 tracing：有效样本 139，contrast 位置众数=133；早层恢复峰值明显高于后层。

![contrast_token_trace.png](figures/contrast_token_trace.png)

### 图：`figures/core_stability_edge_frequency.png`
边稳定性图：pairwise edge Jaccard 中位=0.852。

![core_stability_edge_frequency.png](figures/core_stability_edge_frequency.png)

### 图：`figures/core_stability_edge_jaccard_heatmap.png`
边稳定性图：pairwise edge Jaccard 中位=0.852。

![core_stability_edge_jaccard_heatmap.png](figures/core_stability_edge_jaccard_heatmap.png)

### 图：`figures/core_stability_node_frequency.png`
节点稳定性图：pairwise node Jaccard 中位=0.909。

![core_stability_node_frequency.png](figures/core_stability_node_frequency.png)

### 图：`figures/core_stability_node_jaccard_heatmap.png`
节点稳定性图：pairwise node Jaccard 中位=0.909。

![core_stability_node_jaccard_heatmap.png](figures/core_stability_node_jaccard_heatmap.png)

### 图：`figures/core_transfer_splits.png`
train/test split transfer：跨分割重建 core 后在 test 上仍保持高 suff/nec，并显著优于随机同规模。

![core_transfer_splits.png](figures/core_transfer_splits.png)

### 图：`figures/final_circuit.png`
节点级最终电路。core=10 节点，主干包含 L17H8/L20H5/L21H1/L21H12/L24H6 与 MLP27。

![final_circuit.png](figures/final_circuit.png)

### 图：`figures/final_circuit_coarse.png`
语义粗粒度电路，强调读侧（Reader/Router）与写侧（Writer/Suppressor）分工。

![final_circuit_coarse.png](figures/final_circuit_coarse.png)

### 图：`figures/final_circuit_edge_path_patching.png`
边级因果验证后的结构图；trimmed 最高中介边 `MLP27->Residual Output: <tool_call>`，中位 edge_ratio=0.375。

![final_circuit_edge_path_patching.png](figures/final_circuit_edge_path_patching.png)

### 图：`figures/final_circuit_semantic.png`
在最终电路上叠加语义角色标签（读/写/抑制）后的视图。

![final_circuit_semantic.png](figures/final_circuit_semantic.png)

### 图：`figures/node_ablation_heatmap.png`
orig 条件节点消融图：最关键节点 `L21H12`，drop_full_nec 中位=0.083。

![node_ablation_heatmap.png](figures/node_ablation_heatmap.png)

### 图：`figures/node_ablation_system_json_heatmap.png`
`system_json_pad` 条件节点消融图：最关键节点 `L21H12`，drop_full_nec 中位=0.067。

![node_ablation_system_json_heatmap.png](figures/node_ablation_system_json_heatmap.png)

### 图：`figures/node_ablation_user_json_heatmap.png`
`user_json_pad` 条件节点消融图：最关键节点 `MLP27`，drop_full_nec 中位=0.130。

![node_ablation_user_json_heatmap.png](figures/node_ablation_user_json_heatmap.png)

### 图：`figures/node_alone_suff.png`
orig 条件节点消融图：最关键节点 `L21H12`，drop_full_nec 中位=0.083。

![node_alone_suff.png](figures/node_alone_suff.png)

### 图：`figures/node_alone_suff_system_json.png`
system_json 条件单节点 suff/必要性退化图，显示 backbone 在系统提示增强下的再分配。

![node_alone_suff_system_json.png](figures/node_alone_suff_system_json.png)

### 图：`figures/node_alone_suff_user_json.png`
user_json 条件单节点 suff/必要性退化图，MLP27 相对贡献上升。

![node_alone_suff_user_json.png](figures/node_alone_suff_user_json.png)

### 图：`figures/node_drop_full_nec.png`
orig 条件节点消融图：最关键节点 `L21H12`，drop_full_nec 中位=0.083。

![node_drop_full_nec.png](figures/node_drop_full_nec.png)

### 图：`figures/node_drop_full_nec_system_json.png`
system_json 条件单节点 suff/必要性退化图，显示 backbone 在系统提示增强下的再分配。

![node_drop_full_nec_system_json.png](figures/node_drop_full_nec_system_json.png)

### 图：`figures/node_drop_full_nec_user_json.png`
user_json 条件单节点 suff/必要性退化图，MLP27 相对贡献上升。

![node_drop_full_nec_user_json.png](figures/node_drop_full_nec_user_json.png)

### 图：`figures/node_reallocation_system_vs_orig_barh.png`
system_json-orig 重分配：上升最多 `MLP20` Δ=0.043，下降最多 `L21H12` Δ=-0.019。

![node_reallocation_system_vs_orig_barh.png](figures/node_reallocation_system_vs_orig_barh.png)

### 图：`figures/node_reallocation_user_json_vs_orig_barh.png`
user_json-orig 重分配：上升最多 `MLP27` Δ=0.141，下降最多 `L21H12` Δ=-0.077。

![node_reallocation_user_json_vs_orig_barh.png](figures/node_reallocation_user_json_vs_orig_barh.png)

### 图：`figures/node_reallocation_user_vs_system_barh.png`
user_json-system_json 重分配：上升最多 `MLP27` Δ=0.140，下降最多 `L21H12` Δ=-0.058。

![node_reallocation_user_vs_system_barh.png](figures/node_reallocation_user_vs_system_barh.png)

### 图：`figures/node_stratification_scatter.png`
orig 分层散点：strata 计数 {'stable_necessary_backbone': 4, 'stable_but_weak_or_redundant': 4, 'unstable_weak': 2}。

![node_stratification_scatter.png](figures/node_stratification_scatter.png)

### 图：`figures/node_stratification_scatter_system_json.png`
system_json 分层散点：strata 计数 {'stable_but_weak_or_redundant': 5, 'stable_necessary_backbone': 3, 'unstable_but_necessary': 1, 'unstable_weak': 1}。

![node_stratification_scatter_system_json.png](figures/node_stratification_scatter_system_json.png)

### 图：`figures/node_stratification_scatter_user_json.png`
user_json 分层散点：strata 计数 {'stable_but_weak_or_redundant': 6, 'stable_necessary_backbone': 2, 'unstable_but_necessary': 1, 'unstable_weak': 1}。

![node_stratification_scatter_user_json.png](figures/node_stratification_scatter_user_json.png)

### 图：`figures/node_stratum_trajectories_heatmap.png`
节点跨条件分层轨迹热图：直观看到 backbone 与 weak 节点在 orig/system/user 之间迁移。

![node_stratum_trajectories_heatmap.png](figures/node_stratum_trajectories_heatmap.png)

### 图：`figures/path_patch_edge_bar_full.png`
边级路径 patch 结果图。trimmed 下 `MLP27->Residual Output: <tool_call>` 最强，中位 edge_ratio=0.375，positive_frac=1.000。

![path_patch_edge_bar_full.png](figures/path_patch_edge_bar_full.png)

### 图：`figures/path_patch_edge_bar_trimmed.png`
边级路径 patch 结果图。trimmed 下 `MLP27->Residual Output: <tool_call>` 最强，中位 edge_ratio=0.375，positive_frac=1.000。

![path_patch_edge_bar_trimmed.png](figures/path_patch_edge_bar_trimmed.png)

### 图：`figures/path_patch_edge_heatmap_full.png`
边级路径 patch 结果图。trimmed 下 `MLP27->Residual Output: <tool_call>` 最强，中位 edge_ratio=0.375，positive_frac=1.000。

![path_patch_edge_heatmap_full.png](figures/path_patch_edge_heatmap_full.png)

### 图：`figures/path_patch_edge_heatmap_trimmed.png`
边级路径 patch 结果图。trimmed 下 `MLP27->Residual Output: <tool_call>` 最强，中位 edge_ratio=0.375，positive_frac=1.000。

![path_patch_edge_heatmap_trimmed.png](figures/path_patch_edge_heatmap_trimmed.png)

### 图：`figures/robustness_gap_bin_groups.png`
按 gap 分位（Q1/Qmid/Q4）聚合的组级退化趋势图。

![robustness_gap_bin_groups.png](figures/robustness_gap_bin_groups.png)

### 图：`figures/robustness_slice_heatmap.png`
跨切片稳健性热图；对 gap/长度/难度切片比较 full_core 与子组退化。

![robustness_slice_heatmap.png](figures/robustness_slice_heatmap.png)

### 图：`figures/role_group_causal_heatmap.png`
语义分组评估：`all_heads` drop_full_nec 中位=0.440，full_core suff/nec=0.907/0.924。

![role_group_causal_heatmap.png](figures/role_group_causal_heatmap.png)

### 图：`figures/role_group_data_driven_causal_heatmap.png`
数据驱动分组评估：`all_heads` drop_full_nec 中位=0.440。

![role_group_data_driven_causal_heatmap.png](figures/role_group_data_driven_causal_heatmap.png)

### 图：`figures/role_group_data_driven_necessity_drop.png`
数据驱动分组评估：`all_heads` drop_full_nec 中位=0.440。

![role_group_data_driven_necessity_drop.png](figures/role_group_data_driven_necessity_drop.png)

### 图：`figures/role_group_data_driven_sufficiency.png`
数据驱动分组评估：`all_heads` drop_full_nec 中位=0.440。

![role_group_data_driven_sufficiency.png](figures/role_group_data_driven_sufficiency.png)

### 图：`figures/role_group_necessity_drop.png`
语义分组评估：`all_heads` drop_full_nec 中位=0.440，full_core suff/nec=0.907/0.924。

![role_group_necessity_drop.png](figures/role_group_necessity_drop.png)

### 图：`figures/role_group_sufficiency.png`
语义分组评估：`all_heads` drop_full_nec 中位=0.440，full_core suff/nec=0.907/0.924。

![role_group_sufficiency.png](figures/role_group_sufficiency.png)

### 图：`figures/semantic_attention_delta_heatmap.png`
clean-corrupt 注意力质量增量图，定位模型在关键位置集合上的重分配。

![semantic_attention_delta_heatmap.png](figures/semantic_attention_delta_heatmap.png)

### 图：`figures/semantic_read_causal_heatmap.png`
读侧因果热图：`L24H6` 的 tool_call_tags_ratio 中位=0.443；`L20H5` 的 user_block_ratio 中位=0.185。

![semantic_read_causal_heatmap.png](figures/semantic_read_causal_heatmap.png)

### 图：`figures/semantic_write_target_delta.png`
写侧 `delta logit(target/distractor)`，用于区分 Booster 与 Suppressor 角色。

![semantic_write_target_delta.png](figures/semantic_write_target_delta.png)

### 图：`figures/shift_robustness_drop_nec_heatmap.png`
基础 shift 鲁棒性图（orig/user_pad_short/user_json_pad）展示功能稳健与机制重分配并存。

![shift_robustness_drop_nec_heatmap.png](figures/shift_robustness_drop_nec_heatmap.png)

### 图：`figures/shift_robustness_nec_heatmap.png`
基础 shift 鲁棒性图（orig/user_pad_short/user_json_pad）展示功能稳健与机制重分配并存。

![shift_robustness_nec_heatmap.png](figures/shift_robustness_nec_heatmap.png)

### 图：`figures/shift_robustness_suff_heatmap.png`
基础 shift 鲁棒性图（orig/user_pad_short/user_json_pad）展示功能稳健与机制重分配并存。

![shift_robustness_suff_heatmap.png](figures/shift_robustness_suff_heatmap.png)

### 图：`figures/shift_robustness_systemaware_v1_drop_nec_heatmap.png`
system-aware 分组下的 shift 鲁棒性图，用于隔离 role-position 与 payload 影响。

![shift_robustness_systemaware_v1_drop_nec_heatmap.png](figures/shift_robustness_systemaware_v1_drop_nec_heatmap.png)

### 图：`figures/shift_robustness_systemaware_v1_nec_heatmap.png`
system-aware 分组下的 shift 鲁棒性图，用于隔离 role-position 与 payload 影响。

![shift_robustness_systemaware_v1_nec_heatmap.png](figures/shift_robustness_systemaware_v1_nec_heatmap.png)

### 图：`figures/shift_robustness_systemaware_v1_suff_heatmap.png`
system-aware 分组下的 shift 鲁棒性图，用于隔离 role-position 与 payload 影响。

![shift_robustness_systemaware_v1_suff_heatmap.png](figures/shift_robustness_systemaware_v1_suff_heatmap.png)

### 图：`figures/shift_robustness_useraware_v1_drop_nec_heatmap.png`
user-aware 分组下的 shift 鲁棒性图，重点看 user_json 目标收益与 orig 旁路代价。

![shift_robustness_useraware_v1_drop_nec_heatmap.png](figures/shift_robustness_useraware_v1_drop_nec_heatmap.png)

### 图：`figures/shift_robustness_useraware_v1_nec_heatmap.png`
user-aware 分组下的 shift 鲁棒性图，重点看 user_json 目标收益与 orig 旁路代价。

![shift_robustness_useraware_v1_nec_heatmap.png](figures/shift_robustness_useraware_v1_nec_heatmap.png)

### 图：`figures/shift_robustness_useraware_v1_suff_heatmap.png`
user-aware 分组下的 shift 鲁棒性图，重点看 user_json 目标收益与 orig 旁路代价。

![shift_robustness_useraware_v1_suff_heatmap.png](figures/shift_robustness_useraware_v1_suff_heatmap.png)

### 图：`figures/shift_robustness_v4_drop_nec_heatmap.png`
v4 多条件 shift 鲁棒性热图（orig/system_json/user_json/user_pad_short），用于跨模式对齐比较。

![shift_robustness_v4_drop_nec_heatmap.png](figures/shift_robustness_v4_drop_nec_heatmap.png)

### 图：`figures/shift_robustness_v4_nec_heatmap.png`
v4 多条件 shift 鲁棒性热图（orig/system_json/user_json/user_pad_short），用于跨模式对齐比较。

![shift_robustness_v4_nec_heatmap.png](figures/shift_robustness_v4_nec_heatmap.png)

### 图：`figures/shift_robustness_v4_suff_heatmap.png`
v4 多条件 shift 鲁棒性热图（orig/system_json/user_json/user_pad_short），用于跨模式对齐比较。

![shift_robustness_v4_suff_heatmap.png](figures/shift_robustness_v4_suff_heatmap.png)

## 4. 全量表格逐项解释（49/49）

### 表：`tables/condition_aware_backbone_comparison.csv`
- 规模：8 行，8 列。
- 主指标 `drop_full_nec_median`：`full_nec - minus_nec` 的中位数，越大越关键。
- 结果：最高 `mode=user_pad_short` (drop_full_nec_median=0.274)；最低 `mode=orig` (drop_full_nec_median=0.017)。
- 解读：该表用于比较静态分组与条件自适应分组的目标收益与跨域代价。

- 列：`mode, grouping, n_nodes, n_samples, suff_median, nec_median, drop_full_suff_median, drop_full_nec_median`

### 表：`tables/condition_aware_grouping_delta_bootstrap.csv`
- 规模：8 行，7 列。
- 结果：最高 `mode=user_json_pad` (delta_median_boot_mean=0.038)；最低 `mode=orig` (delta_median_boot_mean=-0.237)。
- 解读：该表用于比较静态分组与条件自适应分组的目标收益与跨域代价。

- 列：`comparison, mode, metric, delta_median_boot_mean, ci_lo, ci_hi, n`

### 表：`tables/core_stability_edge_frequency.csv`
- 规模：28 行，2 列。
- 主指标 `frequency`：跨变体出现频率（稳定性）。
- 结果：最高 `edge=L17H8->L21H12` (frequency=1.000)；最低 `edge=L24H6->MLP25` (frequency=0.111)。
- 解读：该表评估超参扰动下的结构稳定性，频率/Jaccard 越高越稳。

- 列：`edge, frequency`

### 表：`tables/core_stability_node_frequency.csv`
- 规模：11 行，2 列。
- 主指标 `frequency`：跨变体出现频率（稳定性）。
- 结果：最高 `node=L17H8` (frequency=1.000)；最低 `node=MLP25` (frequency=0.333)。
- 解读：该表评估超参扰动下的结构稳定性，频率/Jaccard 越高越稳。

- 列：`node, frequency`

### 表：`tables/core_stability_pairwise.csv`
- 规模：351 行，4 列。
- 主指标 `node_jaccard`：节点集合重叠度。
- 结果：`node_jaccard` 范围 [0.727, 1.000]。
- 解读：该表评估超参扰动下的结构稳定性，频率/Jaccard 越高越稳。

- 列：`variant_a, variant_b, node_jaccard, edge_jaccard`

### 表：`tables/core_stability_variants.csv`
- 规模：27 行，11 列。
- 结果：最高 `variant=gap0.60_n0.55_e0.30` (gap_min=0.600)；最低 `variant=gap0.40_n0.45_e0.30` (gap_min=0.400)。
- 解读：该表评估超参扰动下的结构稳定性，频率/Jaccard 越高越稳。

- 列：`variant, gap_min, core_node_th, core_edge_th, n_records_total, n_records_active, total_weight, n_core_nodes, n_core_edges, core_nodes, core_edges`

### 表：`tables/core_transfer_splits.csv`
- 规模：3 行，13 列。
- 结果：最高 `split=seed33` (seed=33.000)；最低 `split=seed11` (seed=11.000)。

- 列：`split, seed, train_n, test_n, train_total_weight, core_node_count, core_nodes, core_vs_reference_jaccard, test_suff_median, test_nec_median, test_random_suff_median, test_global_minus_random_median, test_replay_samples`

### 表：`tables/node_ablation_gap08_per_sample.csv`
- 规模：1380 行，12 列。
- 结果：最高 `node=L17H8` (q_index=164.000)；最低 `node=L21H1` (q_index=1.000)。

- 列：`q_index, node, node_role, gap, full_suff_ratio, full_nec_ratio, node_suff_ratio, node_nec_ratio, minus_node_suff_ratio, minus_node_nec_ratio, drop_full_suff, drop_full_nec`

### 表：`tables/node_ablation_gap08_summary.csv`
- 规模：10 行，19 列。
- 主指标 `drop_full_nec_median`：`full_nec - minus_nec` 的中位数，越大越关键。
- 结果：最高 `node=L21H12` (drop_full_nec_median=0.083)；最低 `node=MLP20` (drop_full_nec_median=-0.037)。
- 解读：该表用于识别单节点必要性与冗余性，通常以 `drop_full_nec_median` 排序作为 backbone 依据。

- 列：`node, node_role, n_samples, node_suff_median, node_suff_ci_lo, node_suff_ci_hi, node_nec_median, node_nec_ci_lo, node_nec_ci_hi, minus_node_suff_median, minus_node_nec_median, drop_full_suff_median, drop_full_suff_ci_lo, drop_full_suff_ci_hi, drop_full_nec_median, drop_full_nec_ci_lo, drop_full_nec_ci_hi, rank_drop_full_nec, rank_node_suff`

### 表：`tables/node_ablation_gap08_vs_gap05.csv`
- 规模：10 行，13 列。
- 结果：最高 `node=L21H12` (drop_full_nec_median_gap05=0.083)；最低 `node=MLP20` (drop_full_nec_median_gap05=-0.037)。

- 列：`node, drop_full_nec_median_gap05, drop_full_suff_median_gap05, node_suff_median_gap05, node_nec_median_gap05, drop_full_nec_median_gap08, drop_full_suff_median_gap08, node_suff_median_gap08, node_nec_median_gap08, drop_full_nec_median_delta_08_minus_05, drop_full_suff_median_delta_08_minus_05, node_suff_median_delta_08_minus_05, node_nec_median_delta_08_minus_05`

### 表：`tables/node_ablation_per_sample.csv`
- 规模：1390 行，12 列。
- 结果：最高 `node=L24H6` (q_index=164.000)；最低 `node=MLP22` (q_index=1.000)。

- 列：`q_index, node, node_role, gap, full_suff_ratio, full_nec_ratio, node_suff_ratio, node_nec_ratio, minus_node_suff_ratio, minus_node_nec_ratio, drop_full_suff, drop_full_nec`

### 表：`tables/node_ablation_summary.csv`
- 规模：10 行，19 列。
- 主指标 `drop_full_nec_median`：`full_nec - minus_nec` 的中位数，越大越关键。
- 结果：最高 `node=L21H12` (drop_full_nec_median=0.083)；最低 `node=MLP20` (drop_full_nec_median=-0.037)。
- 解读：该表用于识别单节点必要性与冗余性，通常以 `drop_full_nec_median` 排序作为 backbone 依据。

- 列：`node, node_role, n_samples, node_suff_median, node_suff_ci_lo, node_suff_ci_hi, node_nec_median, node_nec_ci_lo, node_nec_ci_hi, minus_node_suff_median, minus_node_nec_median, drop_full_suff_median, drop_full_suff_ci_lo, drop_full_suff_ci_hi, drop_full_nec_median, drop_full_nec_ci_lo, drop_full_nec_ci_hi, rank_drop_full_nec, rank_node_suff`

### 表：`tables/node_ablation_system_json_per_sample.csv`
- 规模：1390 行，12 列。
- 结果：最高 `node=L24H6` (q_index=164.000)；最低 `node=MLP22` (q_index=1.000)。

- 列：`q_index, node, node_role, gap, full_suff_ratio, full_nec_ratio, node_suff_ratio, node_nec_ratio, minus_node_suff_ratio, minus_node_nec_ratio, drop_full_suff, drop_full_nec`

### 表：`tables/node_ablation_system_json_summary.csv`
- 规模：10 行，19 列。
- 主指标 `drop_full_nec_median`：`full_nec - minus_nec` 的中位数，越大越关键。
- 结果：最高 `node=L21H12` (drop_full_nec_median=0.067)；最低 `node=MLP27` (drop_full_nec_median=-0.011)。
- 解读：该表用于识别单节点必要性与冗余性，通常以 `drop_full_nec_median` 排序作为 backbone 依据。

- 列：`node, node_role, n_samples, node_suff_median, node_suff_ci_lo, node_suff_ci_hi, node_nec_median, node_nec_ci_lo, node_nec_ci_hi, minus_node_suff_median, minus_node_nec_median, drop_full_suff_median, drop_full_suff_ci_lo, drop_full_suff_ci_hi, drop_full_nec_median, drop_full_nec_ci_lo, drop_full_nec_ci_hi, rank_drop_full_nec, rank_node_suff`

### 表：`tables/node_ablation_user_json_delta_bootstrap_vs_orig.csv`
- 规模：10 行，7 列。
- 结果：最高 `node=MLP27` (n=139.000)；最低 `node=MLP27` (n=139.000)。

- 列：`node, n, delta_drop_full_nec_mean, ci_lo, ci_hi, orig_median, json_median`

### 表：`tables/node_ablation_user_json_per_sample.csv`
- 规模：1390 行，12 列。
- 结果：最高 `node=L24H6` (q_index=164.000)；最低 `node=MLP22` (q_index=1.000)。

- 列：`q_index, node, node_role, gap, full_suff_ratio, full_nec_ratio, node_suff_ratio, node_nec_ratio, minus_node_suff_ratio, minus_node_nec_ratio, drop_full_suff, drop_full_nec`

### 表：`tables/node_ablation_user_json_summary.csv`
- 规模：10 行，19 列。
- 主指标 `drop_full_nec_median`：`full_nec - minus_nec` 的中位数，越大越关键。
- 结果：最高 `node=MLP27` (drop_full_nec_median=0.130)；最低 `node=L24H6` (drop_full_nec_median=-0.042)。
- 解读：该表用于识别单节点必要性与冗余性，通常以 `drop_full_nec_median` 排序作为 backbone 依据。

- 列：`node, node_role, n_samples, node_suff_median, node_suff_ci_lo, node_suff_ci_hi, node_nec_median, node_nec_ci_lo, node_nec_ci_hi, minus_node_suff_median, minus_node_nec_median, drop_full_suff_median, drop_full_suff_ci_lo, drop_full_suff_ci_hi, drop_full_nec_median, drop_full_nec_ci_lo, drop_full_nec_ci_hi, rank_drop_full_nec, rank_node_suff`

### 表：`tables/node_reallocation_system_vs_orig_delta_summary.csv`
- 规模：10 行，9 列。
- 主指标 `delta_median`：配对条件中位差（shifted-baseline）。
- 结果：最高 `node=MLP20` (delta_median=0.043)；最低 `node=L21H12` (delta_median=-0.019)。
- 解读：该表量化节点在不同提示条件下的相对贡献迁移。

- 列：`node, node_role, n_pairs, baseline_median, shifted_median, delta_median, delta_mean, delta_ci_lo, delta_ci_hi`

### 表：`tables/node_reallocation_user_json_vs_orig_delta_summary.csv`
- 规模：10 行，9 列。
- 主指标 `delta_median`：配对条件中位差（shifted-baseline）。
- 结果：最高 `node=MLP27` (delta_median=0.141)；最低 `node=L21H12` (delta_median=-0.077)。
- 解读：该表量化节点在不同提示条件下的相对贡献迁移。

- 列：`node, node_role, n_pairs, baseline_median, shifted_median, delta_median, delta_mean, delta_ci_lo, delta_ci_hi`

### 表：`tables/node_reallocation_user_vs_system_delta_summary.csv`
- 规模：10 行，9 列。
- 主指标 `delta_median`：配对条件中位差（shifted-baseline）。
- 结果：最高 `node=MLP27` (delta_median=0.140)；最低 `node=L21H12` (delta_median=-0.058)。
- 解读：该表量化节点在不同提示条件下的相对贡献迁移。

- 列：`node, node_role, n_pairs, baseline_median, shifted_median, delta_median, delta_mean, delta_ci_lo, delta_ci_hi`

### 表：`tables/node_roles.csv`
- 规模：10 行，4 列。
- 结果：最高 `node=L21H12` (full_ratio_median=0.423)；最低 `node=MLP11` (full_ratio_median=0.100)。
- 解读：该表用于把因果数值映射为可解释语义角色（reader/router/writer/suppressor）。

- 列：`node, role, full_ratio_median, target_logit_delta_median`

### 表：`tables/node_stratification.csv`
- 规模：10 行，21 列。
- 主指标 `drop_full_nec_median`：`full_nec - minus_nec` 的中位数，越大越关键。
- 结果：最高 `node=L21H12` (drop_full_nec_median=0.083)；最低 `node=MLP20` (drop_full_nec_median=-0.037)。

- 列：`node, node_role, n_samples, node_suff_median, node_suff_ci_lo, node_suff_ci_hi, node_nec_median, node_nec_ci_lo, node_nec_ci_hi, minus_node_suff_median, minus_node_nec_median, drop_full_suff_median, drop_full_suff_ci_lo, drop_full_suff_ci_hi, drop_full_nec_median, drop_full_nec_ci_lo, drop_full_nec_ci_hi, rank_drop_full_nec, rank_node_suff, stability_freq, stratum`

### 表：`tables/node_stratification_system_json.csv`
- 规模：10 行，21 列。
- 主指标 `drop_full_nec_median`：`full_nec - minus_nec` 的中位数，越大越关键。
- 结果：最高 `node=L21H12` (drop_full_nec_median=0.067)；最低 `node=MLP27` (drop_full_nec_median=-0.011)。

- 列：`node, node_role, n_samples, node_suff_median, node_suff_ci_lo, node_suff_ci_hi, node_nec_median, node_nec_ci_lo, node_nec_ci_hi, minus_node_suff_median, minus_node_nec_median, drop_full_suff_median, drop_full_suff_ci_lo, drop_full_suff_ci_hi, drop_full_nec_median, drop_full_nec_ci_lo, drop_full_nec_ci_hi, rank_drop_full_nec, rank_node_suff, stability_freq, stratum`

### 表：`tables/node_stratification_user_json.csv`
- 规模：10 行，21 列。
- 主指标 `drop_full_nec_median`：`full_nec - minus_nec` 的中位数，越大越关键。
- 结果：最高 `node=MLP27` (drop_full_nec_median=0.130)；最低 `node=L24H6` (drop_full_nec_median=-0.042)。

- 列：`node, node_role, n_samples, node_suff_median, node_suff_ci_lo, node_suff_ci_hi, node_nec_median, node_nec_ci_lo, node_nec_ci_hi, minus_node_suff_median, minus_node_nec_median, drop_full_suff_median, drop_full_suff_ci_lo, drop_full_suff_ci_hi, drop_full_nec_median, drop_full_nec_ci_lo, drop_full_nec_ci_hi, rank_drop_full_nec, rank_node_suff, stability_freq, stratum`

### 表：`tables/node_stratum_trajectories.csv`
- 规模：10 行，4 列。
- 解读：该表用于展示节点在稳定性-必要性分层体系中的跨条件迁移。

- 列：`node, orig, system_json_pad, user_json_pad`

### 表：`tables/path_patch_edge_per_sample.csv`
- 规模：3197 行，8 列。
- 结果：最高 `edge=L21H1->MLP27` (q_index=164.000)；最低 `edge=L20H5->L21H12` (q_index=1.000)。
- 解读：该表是边级中介因果证据，`edge_ratio_median` 与 `positive_frac` 联合判断边可靠性。

- 列：`q_index, edge, source, target, source_ratio, blocked_ratio, edge_ratio, gap`

### 表：`tables/path_patch_edge_summary_full.csv`
- 规模：23 行，9 列。
- 主指标 `edge_ratio_median`：边级中介效应中位数（source-blocked）。
- 结果：最高 `edge=MLP27->Residual Output: <tool_call>` (edge_ratio_median=0.379)；最低 `edge=L17H8->L20H5` (edge_ratio_median=0.030)。
- 解读：该表是边级中介因果证据，`edge_ratio_median` 与 `positive_frac` 联合判断边可靠性。

- 列：`edge, n_samples, source_ratio_median, blocked_ratio_median, edge_ratio_median, edge_ratio_mean, edge_ratio_ci_lo, edge_ratio_ci_hi, positive_frac`

### 表：`tables/path_patch_edge_summary_trimmed.csv`
- 规模：23 行，9 列。
- 主指标 `edge_ratio_median`：边级中介效应中位数（source-blocked）。
- 结果：最高 `edge=MLP27->Residual Output: <tool_call>` (edge_ratio_median=0.375)；最低 `edge=L17H8->L20H5` (edge_ratio_median=0.031)。
- 解读：该表是边级中介因果证据，`edge_ratio_median` 与 `positive_frac` 联合判断边可靠性。

- 列：`edge, n_samples, source_ratio_median, blocked_ratio_median, edge_ratio_median, edge_ratio_mean, edge_ratio_ci_lo, edge_ratio_ci_hi, positive_frac`

### 表：`tables/robustness_gap_bins.csv`
- 规模：20 行，7 列。
- 主指标 `drop_full_nec_median`：`full_nec - minus_nec` 的中位数，越大越关键。
- 结果：最高 `group=all_heads` (drop_full_nec_median=0.467)；最低 `group=all_mlps` (drop_full_nec_median=-0.203)。

- 列：`gap_bin, group, n_samples, suff_median, nec_median, drop_full_suff_median, drop_full_nec_median`

### 表：`tables/robustness_slice_groups.csv`
- 规模：90 行，9 列。
- 主指标 `drop_full_nec_median`：`full_nec - minus_nec` 的中位数，越大越关键。
- 结果：最高 `group=all_heads` (drop_full_nec_median=0.467)；最低 `group=all_mlps` (drop_full_nec_median=-0.203)。

- 列：`group, n_samples, suff_median, nec_median, suff_minus_median, nec_minus_median, drop_full_suff_median, drop_full_nec_median, slice`

### 表：`tables/robustness_slice_summary.csv`
- 规模：10 行，9 列。
- 结果：最高 `slice=all` (n_samples=139.000)；最低 `slice=hard_cases` (n_samples=16.000)。

- 列：`n_samples, full_core_suff_median, full_core_nec_median, format_router_drop_nec_median, query_reader_drop_nec_median, tool_tag_reader_drop_nec_median, all_heads_drop_nec_median, all_mlps_drop_nec_median, slice`

### 表：`tables/role_group_data_driven_summary.csv`
- 规模：6 行，15 列。
- 主指标 `drop_full_nec_median`：`full_nec - minus_nec` 的中位数，越大越关键。
- 结果：最高 `group=all_heads` (drop_full_nec_median=0.440)；最低 `group=all_mlps` (drop_full_nec_median=-0.127)。

- 列：`group, n_samples, n_nodes, suff_median, suff_ci_lo, suff_ci_hi, nec_median, nec_ci_lo, nec_ci_hi, drop_full_suff_median, drop_full_suff_ci_lo, drop_full_suff_ci_hi, drop_full_nec_median, drop_full_nec_ci_lo, drop_full_nec_ci_hi`

### 表：`tables/role_group_old_vs_data_driven.csv`
- 规模：15 行，7 列。
- 主指标 `drop_full_nec_median`：`full_nec - minus_nec` 的中位数，越大越关键。
- 结果：最高 `group=all_heads` (drop_full_nec_median=0.440)；最低 `group=all_mlps` (drop_full_nec_median=-0.127)。

- 列：`scheme, group, n_nodes, suff_median, nec_median, drop_full_suff_median, drop_full_nec_median`

### 表：`tables/role_group_per_sample.csv`
- 规模：1251 行，10 列。
- 结果：最高 `group=all_mlps` (q_index=164.000)；最低 `group=tool_tag_reader` (q_index=1.000)。

- 列：`q_index, group, n_nodes, suff_ratio, nec_ratio, suff_minus_ratio, nec_minus_ratio, delta_full_suff_drop, delta_full_nec_drop, gap`

### 表：`tables/role_group_summary.csv`
- 规模：9 行，15 列。
- 主指标 `drop_full_nec_median`：`full_nec - minus_nec` 的中位数，越大越关键。
- 结果：最高 `group=all_heads` (drop_full_nec_median=0.440)；最低 `group=all_mlps` (drop_full_nec_median=-0.127)。

- 列：`group, n_samples, n_nodes, suff_median, suff_ci_lo, suff_ci_hi, nec_median, nec_ci_lo, nec_ci_hi, drop_full_suff_median, drop_full_suff_ci_lo, drop_full_suff_ci_hi, drop_full_nec_median, drop_full_nec_ci_lo, drop_full_nec_ci_hi`

### 表：`tables/semantic_node_metrics.csv`
- 规模：1390 行，29 列。
- 结果：最高 `node=MLP22` (q_index=164.000)；最低 `node=MLP11` (q_index=1.000)。
- 解读：该表用于把因果数值映射为可解释语义角色（reader/router/writer/suppressor）。

- 列：`q_index, node, node_type, gap, full_ratio, top1_ratio, top3_ratio, contrast_ratio, tool_call_tags_ratio, tools_block_ratio, user_block_ratio, attn_contrast_clean, attn_tool_call_tags_clean, attn_tools_block_clean, attn_user_block_clean, attn_recent_32_clean, attn_prefix_16_clean, attn_contrast_corrupt, attn_tool_call_tags_corrupt, attn_tools_block_corrupt, attn_user_block_corrupt, attn_recent_32_corrupt, attn_prefix_16_corrupt, top1_pos, top1_token, top1_category, target_logit_delta, distractor_logit_delta, top_positive_tokens`

### 表：`tables/shift_robustness_common_intersection_summary.csv`
- 规模：15 行，8 列。
- 主指标 `drop_full_nec_median`：`full_nec - minus_nec` 的中位数，越大越关键。
- 结果：最高 `group=all_heads` (drop_full_nec_median=0.436)；最低 `group=all_mlps` (drop_full_nec_median=-0.125)。

- 列：`mode, group, n_samples, gap_aug_median, suff_median, nec_median, drop_full_nec_median, drop_full_nec_abs_median`

### 表：`tables/shift_robustness_mode_delta_bootstrap.csv`
- 规模：20 行，7 列。
- 结果：最高 `group=all_mlps` (delta_median_boot_mean=1.804)；最低 `group=all_heads` (delta_median_boot_mean=-2.121)。
- 解读：该表是跨条件机制重分配证据核心，CI 不跨 0 的行可视为显著变化。

- 列：`contrast, group, metric, delta_median_boot_mean, ci_lo, ci_hi, n`

### 表：`tables/shift_robustness_per_sample.csv`
- 规模：1985 行，13 列。
- 结果：最高 `group=all_mlps` (q_index=164.000)；最低 `group=full_core` (q_index=1.000)。

- 列：`mode, q_index, group, n_nodes, gap_aug, full_suff_ratio, full_nec_ratio, suff_ratio, nec_ratio, suff_minus_ratio, nec_minus_ratio, drop_full_suff, drop_full_nec`

### 表：`tables/shift_robustness_summary.csv`
- 规模：15 行，11 列。
- 主指标 `drop_full_nec_median`：`full_nec - minus_nec` 的中位数，越大越关键。
- 结果：最高 `group=all_heads` (drop_full_nec_median=0.436)；最低 `group=all_mlps` (drop_full_nec_median=-0.111)。

- 列：`mode, group, n_samples, n_nodes, gap_aug_median, full_suff_median, full_nec_median, suff_median, nec_median, drop_full_suff_median, drop_full_nec_median`

### 表：`tables/shift_robustness_systemaware_v1_per_sample.csv`
- 规模：1390 行，13 列。
- 结果：最高 `group=all_heads` (q_index=164.000)；最低 `group=full_core` (q_index=1.000)。

- 列：`mode, q_index, group, n_nodes, gap_aug, full_suff_ratio, full_nec_ratio, suff_ratio, nec_ratio, suff_minus_ratio, nec_minus_ratio, drop_full_suff, drop_full_nec`

### 表：`tables/shift_robustness_systemaware_v1_summary.csv`
- 规模：10 行，11 列。
- 主指标 `drop_full_nec_median`：`full_nec - minus_nec` 的中位数，越大越关键。
- 结果：最高 `group=all_heads` (drop_full_nec_median=0.436)；最低 `group=all_mlps` (drop_full_nec_median=-0.111)。

- 列：`mode, group, n_samples, n_nodes, gap_aug_median, full_suff_median, full_nec_median, suff_median, nec_median, drop_full_suff_median, drop_full_nec_median`

### 表：`tables/shift_robustness_useraware_v1_per_sample.csv`
- 规模：1390 行，13 列。
- 结果：最高 `group=all_heads` (q_index=164.000)；最低 `group=full_core` (q_index=1.000)。

- 列：`mode, q_index, group, n_nodes, gap_aug, full_suff_ratio, full_nec_ratio, suff_ratio, nec_ratio, suff_minus_ratio, nec_minus_ratio, drop_full_suff, drop_full_nec`

### 表：`tables/shift_robustness_useraware_v1_summary.csv`
- 规模：10 行，11 列。
- 主指标 `drop_full_nec_median`：`full_nec - minus_nec` 的中位数，越大越关键。
- 结果：最高 `group=all_heads` (drop_full_nec_median=0.436)；最低 `group=all_mlps` (drop_full_nec_median=-0.111)。

- 列：`mode, group, n_samples, n_nodes, gap_aug_median, full_suff_median, full_nec_median, suff_median, nec_median, drop_full_suff_median, drop_full_nec_median`

### 表：`tables/shift_robustness_v4_common_intersection_summary.csv`
- 规模：20 行，7 列。
- 主指标 `drop_full_nec_median`：`full_nec - minus_nec` 的中位数，越大越关键。
- 结果：最高 `group=all_heads` (drop_full_nec_median=0.436)；最低 `group=all_mlps` (drop_full_nec_median=-0.125)。

- 列：`mode, group, n_common_samples, suff_median, nec_median, drop_full_nec_median, drop_full_nec_abs_median`

### 表：`tables/shift_robustness_v4_mode_delta_bootstrap.csv`
- 规模：40 行，7 列。
- 结果：最高 `group=all_mlps` (delta_median_boot_mean=1.800)；最低 `group=all_heads` (delta_median_boot_mean=-2.191)。
- 解读：该表是跨条件机制重分配证据核心，CI 不跨 0 的行可视为显著变化。

- 列：`contrast, group, metric, delta_median_boot_mean, ci_lo, ci_hi, n`

### 表：`tables/shift_robustness_v4_per_sample.csv`
- 规模：2680 行，13 列。
- 结果：最高 `group=stable_but_weak_or_redundant` (q_index=164.000)；最低 `group=full_core` (q_index=1.000)。

- 列：`mode, q_index, group, n_nodes, gap_aug, full_suff_ratio, full_nec_ratio, suff_ratio, nec_ratio, suff_minus_ratio, nec_minus_ratio, drop_full_suff, drop_full_nec`

### 表：`tables/shift_robustness_v4_summary.csv`
- 规模：20 行，11 列。
- 主指标 `drop_full_nec_median`：`full_nec - minus_nec` 的中位数，越大越关键。
- 结果：最高 `group=all_heads` (drop_full_nec_median=0.436)；最低 `group=all_mlps` (drop_full_nec_median=-0.111)。

- 列：`mode, group, n_samples, n_nodes, gap_aug_median, full_suff_median, full_nec_median, suff_median, nec_median, drop_full_suff_median, drop_full_nec_median`

### 表：`tables/stratum_transition_counts.csv`
- 规模：10 行，5 列。
- 结果：`count` 范围 [1.000, 4.000]。
- 解读：该表用于展示节点在稳定性-必要性分层体系中的跨条件迁移。

- 列：`from_condition, to_condition, from_stratum, to_stratum, count`

## 5. 关键风险与后续建议

- `suff/nec` 可 >1 或 <0（归一化与非线性叠加导致），解释时应同时看 CI 与跨条件一致性。
- `user-aware` 与 `system-aware` 的跨域 penalty 已被量化，部署时应先确定目标域，再选分组。
- `reports/*.json` 的 `artifacts` 已全部对齐到 `final/`，后续新增结果建议保持该约束，避免路径漂移。

## 6. 一句话总括

功能层面：core 电路在多切片与多条件下保持高 suff/nec；机制层面：shift 会引发 heads↔MLPs 的可量化重分配，且这种重分配在 user_json 条件下最显著。

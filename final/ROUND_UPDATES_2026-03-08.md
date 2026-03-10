# Round Updates (2026-03-08)

## New Methodology

1. Node-level causal ablation (`node alone`, `full minus node`) on all 139 valid samples.
2. Slice-based robustness dashboard (gap/length/hard-easy/random split).
3. Hyperparameter stability sweep for global-core extraction (27 variants).
4. Two-axis node stratification: structural stability vs causal necessity.
5. Distribution-shift robustness with controlled neutral context injection.

## Main Findings

1. Core-level suff/nec remains strong and stable across major slices.
2. Head backbone nodes (`L17H8`, `L21H1`, `L21H12`, `L23H6`) are both stable and causally necessary.
3. Several MLP nodes are structurally stable but weak/negative in full-circuit necessity (redundant/supportive rather than backbone).
4. Grouping is better expressed as layered strata, not flat semantic labels.
5. Under shift, full-core behavior remains robust while head/MLP contribution rebalances, indicating mechanism reallocation.

## Added Artifacts

- Reports:
  - `final/reports/node_ablation_report.json`
  - `final/reports/robustness_dashboard_report.json`
  - `final/reports/role_group_data_driven_report.json`
  - `final/reports/core_stability_report.json`
  - `final/reports/node_stratification_report.json`
- Tables:
  - `final/tables/node_ablation_summary.csv`
  - `final/tables/robustness_slice_summary.csv`
  - `final/tables/role_group_data_driven_summary.csv`
  - `final/tables/core_stability_variants.csv`
  - `final/tables/node_stratification.csv`
- Figures:
  - `final/figures/node_ablation_heatmap.png`
  - `final/figures/robustness_slice_heatmap.png`
  - `final/figures/core_stability_node_jaccard_heatmap.png`
  - `final/figures/node_stratification_scatter.png`
  - `final/figures/shift_robustness_drop_nec_heatmap.png`

## Incremental Update (2026-03-09)

### New Methodology

1. Shifted node-level ablation (`user_json_pad`) with paired bootstrap delta vs orig.
2. Strong control shift (`system_json_pad`) to isolate role-position effect from payload effect.
3. Reproducible paired-bootstrap script for shift-mode deltas (`analyze_toolcall_shift_deltas.py`).
4. Cross-condition stratum transition analysis (`orig -> system_json_pad -> user_json_pad`).
5. Condition-aware grouping validation (user-aware/system-aware) with explicit off-target checks.

### Main Findings

1. Function-level robustness remains strong under both `system_json_pad` and `user_json_pad`.
2. Mechanism-level reallocation is much stronger under `user_json_pad`:
  - `user_json_pad - orig`: heads drop and MLPs rise with wide CI separation.
  - `user_json_pad - system_json_pad`: heads still drop and MLPs still rise (CI excludes 0).
3. Node-level shifts are concentrated:
  - strongest up-shifts: `MLP27`, `MLP20`
  - strongest down-shifts: `L21H12`, `L24H6`
4. Condition-aware grouping is necessary:
  - several backbone heads in orig/system conditions become weak/redundant in user-json condition.
5. Condition-aware grouping is not automatically better:
  - user-aware improves target (`user_json`) but has large off-target penalty on `orig`.
  - system-aware does not outperform static grouping on `system_json`.

### Added Artifacts

- Reports:
  - `final/reports/overnight_round7_log.md`
  - `final/reports/overnight_round8_log.md`
  - `final/reports/overnight_round9_log.md`
  - `final/reports/shift_robustness_v4_report.json`
  - `final/reports/shift_robustness_v4_delta_report.json`
  - `final/reports/node_ablation_user_json_report.json`
  - `final/reports/node_ablation_system_json_report.json`
  - `final/reports/stratum_transition_report.json`
  - `final/reports/condition_aware_grouping_report.json`
  - `final/reports/condition_aware_grouping_delta_report.json`
- Tables:
  - `final/tables/shift_robustness_v4_mode_delta_bootstrap.csv`
  - `final/tables/shift_robustness_v4_common_intersection_summary.csv`
  - `final/tables/node_reallocation_user_json_vs_orig_delta_summary.csv`
  - `final/tables/node_reallocation_system_vs_orig_delta_summary.csv`
  - `final/tables/node_reallocation_user_vs_system_delta_summary.csv`
  - `final/tables/node_stratum_trajectories.csv`
  - `final/tables/stratum_transition_counts.csv`
  - `final/tables/condition_aware_backbone_comparison.csv`
  - `final/tables/condition_aware_grouping_delta_bootstrap.csv`
- Figures:
  - `final/figures/node_reallocation_user_json_vs_orig_barh.png`
  - `final/figures/node_reallocation_system_vs_orig_barh.png`
  - `final/figures/node_reallocation_user_vs_system_barh.png`
  - `final/figures/node_stratum_trajectories_heatmap.png`
  - `final/figures/shift_robustness_v4_drop_nec_heatmap.png`
  - `final/figures/shift_robustness_useraware_v1_drop_nec_heatmap.png`

## Incremental Update (2026-03-10)

### Update Type

Result packaging sync only (no new rerun): supplement missing detail-level artifacts into `final/` so downstream review can rely on `final/` alone.

Additional normalization: all `final/reports/*.json` `artifacts` paths were remapped to files under `final/` (instead of upstream `experiments/results`), ensuring path-consistency for standalone consumption.

### Added Artifacts

- Reports:
  - `final/reports/proposed_groups_data_driven.json`
- Tables:
  - `final/tables/path_patch_edge_per_sample.csv`
  - `final/tables/role_group_per_sample.csv`
  - `final/tables/node_ablation_per_sample.csv`
  - `final/tables/node_ablation_gap08_per_sample.csv`
  - `final/tables/node_ablation_user_json_per_sample.csv`
  - `final/tables/node_ablation_system_json_per_sample.csv`
  - `final/tables/shift_robustness_useraware_v1_per_sample.csv`
  - `final/tables/shift_robustness_systemaware_v1_per_sample.csv`
- Figures:
  - `final/figures/final_circuit_semantic.png`
  - `final/figures/shift_robustness_useraware_v1_nec_heatmap.png`
  - `final/figures/shift_robustness_useraware_v1_suff_heatmap.png`
  - `final/figures/shift_robustness_systemaware_v1_nec_heatmap.png`
  - `final/figures/shift_robustness_systemaware_v1_suff_heatmap.png`

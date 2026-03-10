# ROUND2 Update (2026-03-09)

## Scope
- Continued from overnight_round2 checkpoint.
- Finished missing contrast-token tracing stage for best config.
- Ran stability-based edge pruning with replay + role-group re-evaluation.
- Ran cluster-transfer reruns with overlap-confound diagnostics and compact source constructions.
- Synced best/pruned artifacts into `final/` under explicit filenames.

## Commands executed
```bash
# 1) Complete missing contrast tracing
python experiments/trace_toolcall_contrast_token.py \
  --input-root experiments/results/toolcall_q1_q164 \
  --output-root experiments/results/overnight_round2/semantic_best_n0.55_e0.4 \
  --model-path /root/data/Qwen/Qwen3-1.7B \
  --device cuda --gap-min 0.5

# 2) Path-stability pruning + replay
python experiments/prune_toolcall_core_by_path_stability.py \
  --aggregate-summary experiments/results/overnight_round2/aggregate_best_n0.55_e0.4/global_core_summary.json \
  --path-patch-report experiments/results/overnight_round2/semantic_best_n0.55_e0.4/path_patch_edge_report.json \
  --semantic-report experiments/results/overnight_round2/semantic_best_n0.55_e0.4/semantic_roles_report.json \
  --output-root experiments/results/overnight_round2/stability_pruned_best_n0.55_e0.4 \
  --edge-ratio-min 0.03 --positive-frac-min 0.90 \
  --path-metrics trimmed --enforce-input-output-path \
  --run-replay --input-root experiments/results/toolcall_q1_q164 \
  --model-path /root/data/Qwen/Qwen3-1.7B --device cuda --replay-random 2

# 3) Re-evaluate role groups on pruned core
MPLBACKEND=Agg python experiments/evaluate_toolcall_role_groups.py \
  --input-root experiments/results/toolcall_q1_q164 \
  --semantic-report experiments/results/overnight_round2/stability_pruned_best_n0.55_e0.4/semantic_roles_report_pruned.json \
  --aggregate-summary experiments/results/overnight_round2/stability_pruned_best_n0.55_e0.4/global_core_summary_pruned.json \
  --output-root experiments/results/overnight_round2/stability_pruned_best_n0.55_e0.4/semantic_role_eval \
  --model-path /root/data/Qwen/Qwen3-1.7B --device cuda --gap-min 0.5

# 4) Cluster transfer baseline rerun (with overlap diagnostics)
python experiments/evaluate_toolcall_cluster_transfer.py \
  --input-root experiments/results/toolcall_q1_q164 \
  --aggregate-summary experiments/results/overnight_round2/aggregate_best_n0.55_e0.4/global_core_summary.json \
  --output-root experiments/results/overnight_round2/cluster_transfer_best_n0.55_e0.4_v2 \
  --model-path /root/data/Qwen/Qwen3-1.7B --device cuda \
  --source-mode cluster_summary --n-random 1

# 5) Compact low-overlap probe (deterministic top-k)
python experiments/evaluate_toolcall_cluster_transfer.py \
  --input-root experiments/results/toolcall_q1_q164 \
  --aggregate-summary experiments/results/overnight_round2/aggregate_best_n0.55_e0.4/global_core_summary.json \
  --output-root experiments/results/overnight_round2/cluster_transfer_best_n0.55_e0.4_compact_top2_det \
  --model-path /root/data/Qwen/Qwen3-1.7B --device cuda \
  --source-mode compact_support --compact-topk-nodes 2 --n-random 1
```

## Key outcomes

### A) Best config (n0.55/e0.4) is fully complete now
- Contrast tracing done on 139 samples.
- `contrast_position_mode = 133` with histogram `133:139`.
- Contrast early-layer peak remains strong (L1 median recovery ~1.164), while `<tool_call>`/prefix token patching stays near zero.

### B) Stability pruning result (negative but informative)
- Rule: `edge_ratio_median > 0.03` and `positive_frac >= 0.90` on trimmed path-patch summary.
- With Input->Output path constraint:
  - Core shrank from `8 nodes / 20 edges` to `7 nodes / 17 edges`.
  - Removed node: `L20H5`.
  - Removed edges: `L17H8->L20H5`, `L20H5->L21H12`, `L20H5->L21H1`.

#### Replay comparison
- Best 8-node:
  - `suff_median = 0.9036`
  - `nec_median = 0.9241`
  - `global_minus_random_median = 0.6825`
- Pruned 7-node:
  - `suff_median = 0.8615`
  - `nec_median = 0.8667`
  - `global_minus_random_median = 0.6389`
- Stricter 6-node (`edge_ratio_min=0.05`):
  - `suff_median = 0.8491`
  - `nec_median = 0.8525`
  - `global_minus_random_median = 0.6534`

#### Role-group full-core comparison
- Best 8-node: `suff=0.9070`, `nec=0.9206`
- Pruned 7-node: `suff=0.8644`, `nec=0.8679`

Conclusion: this pruning simplifies topology but causes clear causal-fidelity loss; keep as boundary/negative evidence, not as replacement for main best-core result.

### C) P2 transfer-overlap boundary (new)
- Added overlap-aware diagnostics in transfer script:
  - source overlap matrix (`cluster_source_overlap.csv/.png`);
  - overlap-vs-transfer paired table + correlation (`cluster_transfer_overlap_pairs.csv`, `cluster_transfer_overlap_correlation.json`);
  - deterministic top-k tie-break for compact source construction.
- Best aggregate (`n0.55/e0.4`) main numbers:
  - baseline sources: `Jaccard_mean=0.561`, `cross/within suff median=0.985`.
  - compact top-2 sources: `Jaccard_mean=0.444`, `cross/within suff median=0.957`.
  - low-overlap subset (`J<=0.34`, `n=30`) keeps high retention:
    - `suff_vs_within_median=0.978`, `mean=0.991`.
  - overlap-transfer correlation remains weak (near 0 in this partition).
- Boundary (negative evidence retained):
  - pair-level spread is non-trivial (`suff_vs_within_ratio` ~`0.666` to `1.445` in top-2 probe),
  - so “single fully universal circuit” remains too strong;
  - safer claim is shared trunk + directional cluster-pair specialization.

## New synced artifacts in final/
- `context/global_core_summary_best_n0.55_e0.4.json`
- `context/final_circuit_global_core_best_n0.55_e0.4.png`
- `reports/contrast_token_trace_report_best_n0.55_e0.4.json`
- `figures/contrast_token_trace_best_n0.55_e0.4.png`
- `reports/role_group_report_best_n0.55_e0.4.json`
- `tables/role_group_summary_best_n0.55_e0.4.csv`
- `reports/path_patch_edge_report_best_n0.55_e0.4.json`
- `tables/path_patch_edge_summary_trimmed_best_n0.55_e0.4.csv`
- `figures/path_patch_edge_heatmap_trimmed_best_n0.55_e0.4.png`
- `context/global_core_summary_pruned_stability_n0.55_e0.4.json`
- `reports/global_core_replay_pruned_stability_n0.55_e0.4.json`
- `context/final_circuit_global_core_pruned_stability_n0.55_e0.4.png`
- `tables/path_stability_edge_decisions_n0.55_e0.4.csv`
- `tables/path_stability_edge_decisions_n0.55_e0.4_er0.05.csv`
- `reports/role_group_report_pruned_stability_n0.55_e0.4.json`
- `tables/role_group_summary_pruned_stability_n0.55_e0.4.csv`
- `figures/role_group_causal_heatmap_pruned_stability_n0.55_e0.4.png`
- `tables/core_tradeoff_with_pruning.csv`
- `tables/stability_threshold_structural_sweep.csv`
- `tables/cluster_transfer_overlap_confound_summary.csv`
- `tables/cluster_transfer_low_overlap_probe.csv`
- `tables/cluster_transfer_pair_extremes_best_top2.csv`
- `figures/cluster_transfer_overlap_boundary.png`
- `figures/cluster_transfer_overlap_scatter_best_summary.png`
- `figures/cluster_transfer_overlap_scatter_best_top2.png`

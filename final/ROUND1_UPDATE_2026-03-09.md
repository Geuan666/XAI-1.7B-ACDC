# Round-1 Overnight Update (2026-03-09)

## What was added

- Cross-validated held-out evaluation script:
  - `Automatic-Circuit-Discovery/experiments/cross_validate_toolcall_core.py`
- New held-out results:
  - `Automatic-Circuit-Discovery/experiments/results/toolcall_q1_q164_crossval_v1/`
  - threshold sweep dirs: `..._crossval_n0.40_e0.25`, `..._n0.45_e0.30`, `..._n0.55_e0.40`, `..._n0.60_e0.45`
- Sweep summary + tradeoff figure:
  - `Automatic-Circuit-Discovery/experiments/results/toolcall_crossval_sweep_summary.csv`
  - `Automatic-Circuit-Discovery/experiments/results/toolcall_crossval_tradeoff.png`
- Full rerun for higher-fidelity core (n0.40/e0.25):
  - `Automatic-Circuit-Discovery/experiments/results/toolcall_q1_q164_aggregate_n0.40_e0.25/`
  - `Automatic-Circuit-Discovery/experiments/results/toolcall_q1_q164_semantic_roles_n0.40_e0.25/`
- Manual node ablation report:
  - `Automatic-Circuit-Discovery/experiments/results/toolcall_manual_core_ablation.json`

## Key numbers

### Held-out CV (baseline core thresholds n0.50/e0.35, 6 splits)
- test suff median (mean across splits): **0.9330**
- test nec median (mean across splits): **0.9514**
- test delta vs random (mean across splits): **0.8006**

### Aggregate replay comparison
- Original 10-node core:
  - suff **0.9074**, nec **0.9241**, global-random **0.5833**
- Extended 12-node core (n0.40/e0.25):
  - suff **0.9661**, nec **0.9747**, global-random **0.6742**

### Manual node ablation
- `new_12` is best overall.
- Dropping `MLP19` or `MLP25` both reduces quality.
- Dropping both returns to original 10-node behavior.

## Interpretation

- Evidence for cross-sample transfer is now much stronger because discovery and evaluation were split.
- There is a real tradeoff: compact 10-node core is simpler; 12-node core has clearly higher causal fidelity.
- Cluster heterogeneity is sensitive to similarity threshold, so strong “single universal mechanism” claims remain risky without additional cluster-aware transfer tests.

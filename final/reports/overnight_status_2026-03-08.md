# Overnight Status (2026-03-09)

## Completed Iterations

- Round 1: Node-level ablation + slice robustness dashboard + data-driven group re-eval.
- Round 2: 27-variant hyperparameter sweep for core stability.
- Round 3: Stability×necessity 2D node stratification.
- Round 4: Strict-filter (`gap_min=0.8`) robustness recheck.
- Round 5: Held-out transfer test (train/test split, random baseline control).
- Round 6: Distribution-shift robustness with neutral context injection (`user_pad_short`, `user_json_pad`).
- Round 7: Shifted node-level ablation (`user_json_pad`) + paired node reallocation bootstrap.
- Round 8: Strong control (`system_json_pad`) + mode/node/stratum transition analysis.
- Round 9: Condition-aware grouping validation (`useraware` / `systemaware`) with explicit off-target checks.

## Highest-Confidence Findings

1. Function-level robustness remains high across conditions:
- `full_core` retains strong suff/nec on `orig`, `system_json_pad`, and `user_json_pad`.

2. Mechanism-level non-stationarity is now high-confidence:
- `user_json_pad - orig`: `all_heads drop_full_nec` CI `[-0.2354, -0.1985]`, `all_mlps` CI `[+0.1905, +0.2436]`.
- `user_json_pad - system_json_pad`: `all_heads` CI `[-0.1792, -0.1315]`, `all_mlps` CI `[+0.0880, +0.1434]`.

3. `system_json_pad` induces substantially milder reallocation than `user_json_pad`:
- `system_json_pad - orig`: `all_heads` CI `[-0.0641, -0.0471]`, `all_mlps` CI `[+0.0584, +0.1491]`.

4. Node-level reallocation is concentrated and reproducible:
- strongest positive shifts under user-json: `MLP27`, `MLP20`.
- strongest negative shifts under user-json: `L21H12`, `L24H6`.

5. Stratum trajectories show condition-sensitive backbone:
- `orig -> system_json_pad`: mostly stable.
- `system_json_pad -> user_json_pad`: multiple head backbone nodes demoted; `MLP27` promoted to backbone.
6. Condition-aware grouping gives mixed evidence (important negative result):
- user-aware grouping improves `user_json` target metric but collapses on `orig` (large off-target penalty).
- system-aware grouping does not beat static grouping on `system_json`.
- paired bootstrap confirms these shifts are stable (all relevant CI exclude 0).

## Current Best Grouping Narrative

- Keep two-layer conclusion:
  - `function-level`: robust core behavior.
  - `mechanism-level`: context-dependent allocation.
- Prefer reporting both static and condition-aware views, and explicitly mark condition-aware instability/overfitting risk.

## Next Action

- Build constrained condition-aware grouping (minimum node budget + cross-condition regularization) and re-evaluate whether target gains survive without severe off-target collapse.

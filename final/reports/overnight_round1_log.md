# Overnight Round 1 Log (2026-03-08)

## Hypothesis
1. Current semantic grouping may over-credit MLP groups; node-level necessity may reveal a different causal structure.
2. Full-core robustness may vary significantly across gap/length slices.

## Experimental Design
1. Node-level causal ablation (`node alone`, `full minus node`) on all valid 139 samples.
2. Slice-based robustness dashboard over role-group and edge-path metrics:
- gap quartiles
- prompt length quartiles
- hard/easy subsets
- random half-splits
3. Re-evaluate role groups using data-driven grouping induced by node-ablation summary.

## Results
1. Node-level necessity ranking (`drop_full_nec_median`) shows strongest necessity concentrated in:
- `L21H12` (0.083)
- `L17H8` (0.040)
- `L21H1` (0.036)
2. Multiple MLP nodes are weak or negative in full-circuit necessity:
- `MLP27` (-0.023)
- `MLP22` (-0.020)
- `MLP20` (-0.037)
3. Data-driven groups separate a clear backbone from redundant/interfering block:
- `essential_backbone`: `L17H8,L21H1,L21H12` with `drop_full_nec_median=0.203`
- `redundant_or_interfering`: `MLP20,MLP22,MLP27` with `drop_full_nec_median=-0.121`
4. Robustness slices keep the main story stable:
- `full_core_suff_median` remains ~0.89-0.92 across major slices.
- `format_router`/head necessity remains positive and dominant.
- `all_mlps_drop_nec_median` remains negative in every slice examined.

## Interpretation
1. Evidence now supports a stronger claim: head backbone is causally central; MLP block is partly supportive but not uniformly necessary.
2. Original semantic grouping is useful descriptively, but causally cleaner grouping should separate essential-head backbone from MLP redundancy.

## Decision For Next Round
1. Keep full-core claim (strong), but revise grouping claim to emphasize backbone vs redundant/supportive components.
2. Next run should stress-test this revised grouping under additional perturbations (seeded subsampling, threshold sweeps, and transfer checks).

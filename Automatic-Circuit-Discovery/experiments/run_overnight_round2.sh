#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/data/XAI-1.7B-ACDCcopy/Automatic-Circuit-Discovery"
cd "$ROOT"

OUT_BASE="experiments/results/overnight_round2"
mkdir -p "$OUT_BASE"

CV_SUMMARY="$OUT_BASE/cv_runs_summary.csv"
echo "run_tag,core_node_th,core_edge_th,seed,splits,core_size_median,core_pairwise_jaccard_mean,test_suff_median_mean,test_nec_median_mean,test_delta_vs_random_mean,score" > "$CV_SUMMARY"

configs=(
  "0.40 0.25"
  "0.45 0.30"
  "0.50 0.35"
  "0.55 0.40"
)
seeds=(901 1901 2901)

for cfg in "${configs[@]}"; do
  node_th=$(echo "$cfg" | awk '{print $1}')
  edge_th=$(echo "$cfg" | awk '{print $2}')
  for seed in "${seeds[@]}"; do
    tag="n${node_th}_e${edge_th}_s${seed}"
    out="$OUT_BASE/cv_${tag}"
    echo "[overnight] CV start: $tag"

    python experiments/cross_validate_toolcall_core.py \
      --input-root experiments/results/toolcall_q1_q164 \
      --full-aggregate-summary experiments/results/toolcall_q1_q164_aggregate/global_core_summary.json \
      --output-root "$out" \
      --model-path /root/data/Qwen/Qwen3-1.7B \
      --device cuda \
      --gap-min 0.5 \
      --ap-discount 0.7 \
      --core-node-th "$node_th" \
      --core-edge-th "$edge_th" \
      --splits 8 \
      --train-frac 0.7 \
      --random-controls 5 \
      --seed "$seed"

    python - <<PY
import json
from pathlib import Path
report=Path("$out/crossval_report.json")
d=json.loads(report.read_text())
a=d["aggregate_summary"]
score=float(a["test_suff_median_mean"])+float(a["test_nec_median_mean"])+float(a["test_delta_vs_random_mean"])-0.04*float(a["core_size_median"])
row=[
    "$tag",
    "$node_th",
    "$edge_th",
    "$seed",
    str(a["n_splits"]),
    str(a["core_size_median"]),
    str(a["core_pairwise_jaccard_mean"]),
    str(a["test_suff_median_mean"]),
    str(a["test_nec_median_mean"]),
    str(a["test_delta_vs_random_mean"]),
    str(score),
]
with open("$CV_SUMMARY","a",encoding="utf-8") as f:
    f.write(",".join(row)+"\n")
print("[overnight] CV done:", "$tag", "score=", score)
PY
  done
done

BEST_JSON="$OUT_BASE/best_config.json"
python - <<PY
import csv, json
from collections import defaultdict
p="$CV_SUMMARY"
rows=list(csv.DictReader(open(p,encoding="utf-8")))
agg=defaultdict(list)
for r in rows:
    k=(r['core_node_th'], r['core_edge_th'])
    agg[k].append(float(r['score']))
best=max(agg.items(), key=lambda kv: sum(kv[1])/len(kv[1]))
(best_node,best_edge),scores=best
out={
    'best_node_th':float(best_node),
    'best_edge_th':float(best_edge),
    'avg_score':sum(scores)/len(scores),
    'n_runs':len(scores),
}
open("$BEST_JSON","w",encoding='utf-8').write(json.dumps(out,indent=2,ensure_ascii=False))
print('[overnight] best config:',out)
PY

BEST_NODE=$(python - <<'PY'
import json
print(json.load(open('/root/data/XAI-1.7B-ACDCcopy/Automatic-Circuit-Discovery/experiments/results/overnight_round2/best_config.json'))['best_node_th'])
PY
)
BEST_EDGE=$(python - <<'PY'
import json
print(json.load(open('/root/data/XAI-1.7B-ACDCcopy/Automatic-Circuit-Discovery/experiments/results/overnight_round2/best_config.json'))['best_edge_th'])
PY
)

BEST_TAG="n${BEST_NODE}_e${BEST_EDGE}"
AGG_OUT="$OUT_BASE/aggregate_best_${BEST_TAG}"
SEM_OUT="$OUT_BASE/semantic_best_${BEST_TAG}"

python experiments/aggregate_toolcall_circuits.py \
  --input-root experiments/results/toolcall_q1_q164 \
  --output-root "$AGG_OUT" \
  --model-path /root/data/Qwen/Qwen3-1.7B \
  --device cuda \
  --core-node-th "$BEST_NODE" \
  --core-edge-th "$BEST_EDGE" \
  --replay-random 6 \
  --seed 123

python experiments/analyze_toolcall_semantic_roles.py \
  --input-root experiments/results/toolcall_q1_q164 \
  --aggregate-summary "$AGG_OUT/global_core_summary.json" \
  --output-root "$SEM_OUT" \
  --model-path /root/data/Qwen/Qwen3-1.7B \
  --device cuda \
  --gap-min 0.5

python experiments/evaluate_toolcall_role_groups.py \
  --input-root experiments/results/toolcall_q1_q164 \
  --semantic-report "$SEM_OUT/semantic_roles_report.json" \
  --aggregate-summary "$AGG_OUT/global_core_summary.json" \
  --output-root "$SEM_OUT" \
  --model-path /root/data/Qwen/Qwen3-1.7B \
  --device cuda \
  --gap-min 0.5 \
  --bootstrap 1000 \
  --seed 123

python experiments/path_patch_toolcall_edges.py \
  --input-root experiments/results/toolcall_q1_q164 \
  --aggregate-summary "$AGG_OUT/global_core_summary.json" \
  --output-root "$SEM_OUT" \
  --model-path /root/data/Qwen/Qwen3-1.7B \
  --device cuda \
  --gap-min 0.5 \
  --trim-frac 0.10 \
  --bootstrap 1000 \
  --seed 123

python experiments/trace_toolcall_contrast_token.py \
  --input-root experiments/results/toolcall_q1_q164 \
  --output-root "$SEM_OUT" \
  --model-path /root/data/Qwen/Qwen3-1.7B \
  --device cuda \
  --gap-min 0.5

echo "[overnight] finished all tasks. outputs under $OUT_BASE"

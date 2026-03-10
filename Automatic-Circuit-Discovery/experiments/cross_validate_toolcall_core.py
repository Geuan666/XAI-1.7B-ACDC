#!/usr/bin/env python3
"""
Cross-validated evaluation for global tool-call circuits.

Goal:
- Discover a consensus core on train samples only.
- Evaluate sufficiency/necessity on held-out test samples.
- Compare against same-size random node controls.

This script strengthens causal evidence by reducing train-test leakage.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

from experiments.aggregate_toolcall_circuits import (
    SampleRecord,
    aggregate_supports,
    load_sample_records,
    pick_consensus_edges,
    pick_consensus_nodes,
    safe_float,
)
from experiments.launch_toolcall_qwen3_q85 import (
    collect_clean_cache_cpu,
    evaluate_on_base_with_source,
    load_hooked_qwen3,
)


def median(xs: Sequence[float]) -> float:
    vals = [float(x) for x in xs if isinstance(x, (int, float)) and math.isfinite(float(x))]
    return float(np.median(vals)) if vals else float("nan")


def mean(xs: Sequence[float]) -> float:
    vals = [float(x) for x in xs if isinstance(x, (int, float)) and math.isfinite(float(x))]
    return float(np.mean(vals)) if vals else float("nan")


def harmonic_mean(x: float, y: float, eps: float = 1e-8) -> float:
    if x <= 0 or y <= 0:
        return 0.0
    return float(2.0 * x * y / max(eps, x + y))


def jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return float(len(sa & sb) / len(sa | sb))


def stratified_split(
    records: Sequence[SampleRecord], train_frac: float, seed: int
) -> Tuple[List[SampleRecord], List[SampleRecord]]:
    rng = random.Random(seed)
    gaps = [safe_float(r.summary.get("gap"), 0.0) for r in records]
    if len(gaps) < 8:
        shuffled = list(records)
        rng.shuffle(shuffled)
        cut = max(1, int(round(train_frac * len(shuffled))))
        cut = min(len(shuffled) - 1, cut)
        return shuffled[:cut], shuffled[cut:]

    q1, q2, q3 = np.quantile(np.array(gaps), [0.25, 0.50, 0.75]).tolist()
    bins: Dict[int, List[SampleRecord]] = defaultdict(list)
    for r in records:
        g = safe_float(r.summary.get("gap"), 0.0)
        if g <= q1:
            b = 0
        elif g <= q2:
            b = 1
        elif g <= q3:
            b = 2
        else:
            b = 3
        bins[b].append(r)

    train: List[SampleRecord] = []
    test: List[SampleRecord] = []
    for b in sorted(bins):
        items = bins[b]
        rng.shuffle(items)
        cut = int(round(train_frac * len(items)))
        cut = max(1, min(len(items) - 1, cut)) if len(items) >= 2 else len(items)
        train.extend(items[:cut])
        test.extend(items[cut:])

    if not train or not test:
        shuffled = list(records)
        rng.shuffle(shuffled)
        cut = max(1, int(round(train_frac * len(shuffled))))
        cut = min(len(shuffled) - 1, cut)
        return shuffled[:cut], shuffled[cut:]

    return sorted(train, key=lambda r: r.q_index), sorted(test, key=lambda r: r.q_index)


def evaluate_records(
    model,
    records: Sequence[SampleRecord],
    patch_nodes: Sequence[str],
    node_universe: Sequence[str],
    random_controls: int,
    seed: int,
    desc: str,
) -> List[Dict[str, object]]:
    rng = random.Random(seed)
    patch_nodes = list(dict.fromkeys(patch_nodes))
    node_universe = list(dict.fromkeys(node_universe))
    patch_size = len(patch_nodes)
    patch_set = set(patch_nodes)

    rows: List[Dict[str, object]] = []
    for rec in tqdm(records, desc=desc, dynamic_ncols=True):
        q_index = rec.q_index
        summary = rec.summary
        clean_path = Path(str(summary.get("clean_prompt", ""))).resolve()
        corrupt_path = Path(str(summary.get("corrupt_prompt", ""))).resolve()
        if not clean_path.exists() or not corrupt_path.exists():
            continue

        clean_text = clean_path.read_text(encoding="utf-8")
        corrupt_text = corrupt_path.read_text(encoding="utf-8")

        clean_tokens = model.to_tokens(clean_text, prepend_bos=False)
        corrupt_tokens = model.to_tokens(corrupt_text, prepend_bos=False)
        if clean_tokens.shape != corrupt_tokens.shape:
            continue

        target_token = int(summary.get("target_token_id"))
        distractor_token = int(summary.get("distractor_token_id"))
        clean_obj = safe_float(summary.get("clean_obj"), float("nan"))
        corrupt_obj = safe_float(summary.get("corrupt_obj"), float("nan"))
        gap = safe_float(summary.get("gap"), float("nan"))
        if not (math.isfinite(clean_obj) and math.isfinite(corrupt_obj) and math.isfinite(gap)):
            continue
        if abs(gap) <= 1e-8:
            continue

        clean_cache_cpu = collect_clean_cache_cpu(model, clean_tokens)
        corrupt_cache_cpu = collect_clean_cache_cpu(model, corrupt_tokens)

        suff_obj = evaluate_on_base_with_source(
            model=model,
            base_tokens=corrupt_tokens,
            source_cache_cpu=clean_cache_cpu,
            patch_nodes=patch_nodes,
            target_token=target_token,
            distractor_token=distractor_token,
        )
        nec_obj = evaluate_on_base_with_source(
            model=model,
            base_tokens=clean_tokens,
            source_cache_cpu=corrupt_cache_cpu,
            patch_nodes=patch_nodes,
            target_token=target_token,
            distractor_token=distractor_token,
        )

        suff = (suff_obj - corrupt_obj) / gap
        nec = (clean_obj - nec_obj) / gap

        pool = [n for n in node_universe if n not in patch_set]
        if len(pool) < patch_size:
            pool = list(node_universe)

        rand_suffs: List[float] = []
        if random_controls > 0 and pool:
            k = min(patch_size, len(pool))
            for _ in range(random_controls):
                rand_nodes = rng.sample(pool, k=k)
                rand_obj = evaluate_on_base_with_source(
                    model=model,
                    base_tokens=corrupt_tokens,
                    source_cache_cpu=clean_cache_cpu,
                    patch_nodes=rand_nodes,
                    target_token=target_token,
                    distractor_token=distractor_token,
                )
                rand_suffs.append((rand_obj - corrupt_obj) / gap)

        row = {
            "q_index": q_index,
            "gap": gap,
            "suff_ratio": suff,
            "nec_ratio": nec,
            "min_ratio": min(suff, nec),
            "harmonic_ratio": harmonic_mean(max(0.0, suff), max(0.0, nec)),
            "rand_suff_mean": mean(rand_suffs),
            "rand_suff_median": median(rand_suffs),
            "rand_suff_n": len(rand_suffs),
            "n_patch_nodes": patch_size,
        }
        rows.append(row)

        model.reset_hooks()
        gc.collect()
        torch.cuda.empty_cache()

    return rows


def summarize_rows(rows: Sequence[Dict[str, object]]) -> Dict[str, float]:
    suff = [safe_float(r.get("suff_ratio"), float("nan")) for r in rows]
    nec = [safe_float(r.get("nec_ratio"), float("nan")) for r in rows]
    mins = [safe_float(r.get("min_ratio"), float("nan")) for r in rows]
    hms = [safe_float(r.get("harmonic_ratio"), float("nan")) for r in rows]
    rand = [safe_float(r.get("rand_suff_mean"), float("nan")) for r in rows]
    return {
        "n_samples": len(rows),
        "suff_median": median(suff),
        "suff_mean": mean(suff),
        "nec_median": median(nec),
        "nec_mean": mean(nec),
        "min_median": median(mins),
        "harmonic_median": median(hms),
        "rand_suff_mean_median": median(rand),
        "global_minus_random_median": median(suff) - median(rand),
    }


def write_rows_csv(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    if not rows:
        return
    fields = [
        "split_id",
        "subset",
        "q_index",
        "gap",
        "suff_ratio",
        "nec_ratio",
        "min_ratio",
        "harmonic_ratio",
        "rand_suff_mean",
        "rand_suff_median",
        "rand_suff_n",
        "n_patch_nodes",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_cv_box(split_metrics: Sequence[Dict[str, object]], out_path: Path) -> None:
    test_suff = [safe_float(s["test"]["suff_median"], float("nan")) for s in split_metrics]
    test_nec = [safe_float(s["test"]["nec_median"], float("nan")) for s in split_metrics]
    test_rand = [safe_float(s["test"]["rand_suff_mean_median"], float("nan")) for s in split_metrics]

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.boxplot([test_suff, test_nec, test_rand], tick_labels=["Test suff", "Test nec", "Test rand suff"], showmeans=True)
    ax.set_ylabel("Median ratio across held-out samples")
    ax.set_title("Cross-Validation: Held-out Circuit Performance")
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-validated tool-call core evaluation.")
    parser.add_argument("--input-root", type=str, default="experiments/results/toolcall_q1_q164")
    parser.add_argument(
        "--full-aggregate-summary",
        type=str,
        default="experiments/results/toolcall_q1_q164_aggregate/global_core_summary.json",
    )
    parser.add_argument("--output-root", type=str, default="experiments/results/toolcall_q1_q164_crossval")
    parser.add_argument("--model-path", type=str, default="/root/data/Qwen/Qwen3-1.7B")
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--gap-min", type=float, default=0.5)
    parser.add_argument("--ap-discount", type=float, default=0.7)
    parser.add_argument("--core-node-th", type=float, default=0.50)
    parser.add_argument("--core-edge-th", type=float, default=0.35)
    parser.add_argument("--min-nodes", type=int, default=8)
    parser.add_argument("--max-core-nodes", type=int, default=18)

    parser.add_argument("--splits", type=int, default=6)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--random-controls", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--eval-train", action="store_true")
    args = parser.parse_args()

    input_root = Path(args.input_root).resolve()
    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    full_core_nodes: List[str] = []
    full_summary_path = Path(args.full_aggregate_summary).resolve()
    if full_summary_path.exists():
        full_summary = json.loads(full_summary_path.read_text(encoding="utf-8"))
        full_core_nodes = [str(x) for x in full_summary.get("core_nodes", [])]

    all_records = load_sample_records(input_root, gap_min=args.gap_min, ap_discount=args.ap_discount)
    records = [r for r in all_records if r.weight > 0]
    if len(records) < 10:
        raise ValueError(f"Not enough usable records under {input_root}. Got {len(records)}.")

    node_universe = sorted({n for r in records for n in r.nodes})
    if not node_universe:
        raise ValueError("Empty node universe.")

    model, tokenizer = load_hooked_qwen3(args.model_path, device=args.device, dtype=torch.bfloat16)
    _ = tokenizer

    split_metrics: List[Dict[str, object]] = []
    per_sample_rows: List[Dict[str, object]] = []
    split_node_rows: List[Dict[str, object]] = []
    split_core_nodes: List[List[str]] = []

    for split_id in range(args.splits):
        split_seed = args.seed + 1000 * split_id
        train_records, test_records = stratified_split(records, train_frac=args.train_frac, seed=split_seed)

        node_support, edge_support, _, _, _, _ = aggregate_supports(train_records)
        core_nodes = pick_consensus_nodes(
            node_support=node_support,
            node_threshold=args.core_node_th,
            min_nodes=args.min_nodes,
            max_nodes=args.max_core_nodes,
        )
        core_edges = pick_consensus_edges(
            nodes=core_nodes,
            edge_support=edge_support,
            node_support=node_support,
            edge_threshold=args.core_edge_th,
            min_edges=max(10, len(core_nodes)),
        )

        split_core_nodes.append(core_nodes)
        for node in core_nodes:
            split_node_rows.append(
                {
                    "split_id": split_id,
                    "node": node,
                    "in_full_core": int(node in set(full_core_nodes)),
                }
            )

        test_rows = evaluate_records(
            model=model,
            records=test_records,
            patch_nodes=core_nodes,
            node_universe=node_universe,
            random_controls=args.random_controls,
            seed=split_seed + 17,
            desc=f"Split {split_id} test",
        )
        for r in test_rows:
            r["split_id"] = split_id
            r["subset"] = "test"
        per_sample_rows.extend(test_rows)
        test_summary = summarize_rows(test_rows)

        train_summary: Dict[str, float]
        if args.eval_train:
            train_rows = evaluate_records(
                model=model,
                records=train_records,
                patch_nodes=core_nodes,
                node_universe=node_universe,
                random_controls=max(1, args.random_controls // 2),
                seed=split_seed + 33,
                desc=f"Split {split_id} train",
            )
            for r in train_rows:
                r["split_id"] = split_id
                r["subset"] = "train"
            per_sample_rows.extend(train_rows)
            train_summary = summarize_rows(train_rows)
        else:
            train_summary = {
                "n_samples": len(train_records),
                "suff_median": float("nan"),
                "nec_median": float("nan"),
                "rand_suff_mean_median": float("nan"),
                "global_minus_random_median": float("nan"),
            }

        split_metrics.append(
            {
                "split_id": split_id,
                "seed": split_seed,
                "n_train": len(train_records),
                "n_test": len(test_records),
                "core_nodes": core_nodes,
                "core_edges": core_edges,
                "core_size": len(core_nodes),
                "jaccard_vs_full_core": jaccard(core_nodes, full_core_nodes) if full_core_nodes else float("nan"),
                "train": train_summary,
                "test": test_summary,
            }
        )

    # Aggregate across splits.
    core_jaccards: List[float] = []
    for i in range(len(split_core_nodes)):
        for j in range(i + 1, len(split_core_nodes)):
            core_jaccards.append(jaccard(split_core_nodes[i], split_core_nodes[j]))

    node_freq: Dict[str, int] = defaultdict(int)
    for nodes in split_core_nodes:
        for n in nodes:
            node_freq[n] += 1

    aggregate_summary = {
        "n_records_total": len(records),
        "n_splits": len(split_metrics),
        "core_size_median": median([m["core_size"] for m in split_metrics]),
        "core_pairwise_jaccard_mean": mean(core_jaccards),
        "core_pairwise_jaccard_median": median(core_jaccards),
        "jaccard_vs_full_core_mean": mean([safe_float(m["jaccard_vs_full_core"], float("nan")) for m in split_metrics]),
        "test_suff_median_mean": mean([safe_float(m["test"]["suff_median"], float("nan")) for m in split_metrics]),
        "test_nec_median_mean": mean([safe_float(m["test"]["nec_median"], float("nan")) for m in split_metrics]),
        "test_delta_vs_random_mean": mean(
            [safe_float(m["test"]["global_minus_random_median"], float("nan")) for m in split_metrics]
        ),
        "node_frequency": dict(sorted(node_freq.items(), key=lambda x: (-x[1], x[0]))),
    }

    (out_root / "crossval_report.json").write_text(
        json.dumps(
            {
                "config": vars(args),
                "full_aggregate_summary": str(full_summary_path),
                "split_metrics": split_metrics,
                "aggregate_summary": aggregate_summary,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Split-level compact CSV.
    split_csv_rows: List[Dict[str, object]] = []
    for m in split_metrics:
        split_csv_rows.append(
            {
                "split_id": m["split_id"],
                "seed": m["seed"],
                "n_train": m["n_train"],
                "n_test": m["n_test"],
                "core_size": m["core_size"],
                "jaccard_vs_full_core": m["jaccard_vs_full_core"],
                "test_suff_median": m["test"]["suff_median"],
                "test_nec_median": m["test"]["nec_median"],
                "test_min_median": m["test"]["min_median"],
                "test_harmonic_median": m["test"]["harmonic_median"],
                "test_rand_suff_mean_median": m["test"]["rand_suff_mean_median"],
                "test_global_minus_random_median": m["test"]["global_minus_random_median"],
            }
        )
    with (out_root / "crossval_split_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split_id",
                "seed",
                "n_train",
                "n_test",
                "core_size",
                "jaccard_vs_full_core",
                "test_suff_median",
                "test_nec_median",
                "test_min_median",
                "test_harmonic_median",
                "test_rand_suff_mean_median",
                "test_global_minus_random_median",
            ],
        )
        writer.writeheader()
        writer.writerows(split_csv_rows)

    with (out_root / "crossval_core_nodes.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split_id", "node", "in_full_core"])
        writer.writeheader()
        writer.writerows(split_node_rows)

    write_rows_csv(per_sample_rows, out_root / "crossval_per_sample.csv")
    plot_cv_box(split_metrics, out_root / "crossval_test_boxplot.png")

    print(f"[done] wrote {out_root / 'crossval_report.json'}")
    print(f"[done] splits={len(split_metrics)} test_suff_mean={aggregate_summary['test_suff_median_mean']:.4f}")


if __name__ == "__main__":
    main()

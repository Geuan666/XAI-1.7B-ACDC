#!/usr/bin/env python3
"""
Held-out transfer evaluation for tool-call global core.

For each split seed:
1) build core on train subset,
2) replay train-derived core on disjoint test subset,
3) compare against random same-size baseline on test.

This directly tests whether the discovered circuit transfers beyond the
samples used to construct it.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence, Set

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.aggregate_toolcall_circuits import (
    aggregate_supports,
    load_sample_records,
    pick_consensus_edges,
    pick_consensus_nodes,
    replay_global_circuit,
)


def parse_int_list(raw: str) -> List[int]:
    out: List[int] = []
    for x in raw.split(","):
        x = x.strip()
        if not x:
            continue
        out.append(int(x))
    return out


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def apply_plot_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )


def save_transfer_plot(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    apply_plot_style()
    labels = [str(r["split"]) for r in rows]
    suff = [float(r["test_suff_median"]) for r in rows]
    nec = [float(r["test_nec_median"]) for r in rows]
    rand = [float(r["test_random_suff_median"]) for r in rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(8.2, 1.1 * len(labels)), 4.8), constrained_layout=True)
    ax.plot(x, suff, marker="o", linewidth=2.0, label="test suff (train core)")
    ax.plot(x, nec, marker="o", linewidth=2.0, label="test nec (train core)")
    ax.plot(x, rand, marker="o", linewidth=1.8, label="test random suff baseline")
    ax.axhline(0.0, color="#444444", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Median ratio")
    ax.set_title("Held-out Transfer Across Split Seeds")
    ax.legend(loc="best")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Held-out transfer evaluation for tool-call core.")
    parser.add_argument("--input-root", type=str, default="experiments/results/toolcall_q1_q164")
    parser.add_argument(
        "--reference-summary",
        type=str,
        default="experiments/results/toolcall_q1_q164_aggregate/global_core_summary.json",
    )
    parser.add_argument("--output-root", type=str, default="experiments/results/toolcall_q1_q164_core_transfer")
    parser.add_argument("--model-path", type=str, default="/root/data/Qwen/Qwen3-1.7B")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gap-min", type=float, default=0.5)
    parser.add_argument("--ap-discount", type=float, default=0.7)
    parser.add_argument("--core-node-th", type=float, default=0.5)
    parser.add_argument("--core-edge-th", type=float, default=0.35)
    parser.add_argument("--min-nodes", type=int, default=8)
    parser.add_argument("--max-core-nodes", type=int, default=18)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--split-seeds", type=str, default="11,22,33")
    parser.add_argument("--replay-random", type=int, default=1)
    args = parser.parse_args()

    input_root = Path(args.input_root).resolve()
    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    records = load_sample_records(root=input_root, gap_min=args.gap_min, ap_discount=args.ap_discount)
    if not records:
        raise ValueError(f"No records under {input_root}")

    reference_core: List[str] = []
    ref_path = Path(args.reference_summary).resolve()
    if ref_path.exists():
        ref = json.loads(ref_path.read_text(encoding="utf-8"))
        reference_core = [str(n) for n in ref.get("core_nodes", [])]

    q_all = [int(r.q_index) for r in records if float(r.weight) > 0.0]
    if len(q_all) < 40:
        raise ValueError("Too few active samples for held-out transfer.")
    split_seeds = parse_int_list(args.split_seeds)
    if not split_seeds:
        raise ValueError("No split seeds provided.")

    rows: List[Dict[str, object]] = []
    for seed in split_seeds:
        rng = random.Random(seed)
        q_perm = list(q_all)
        rng.shuffle(q_perm)
        cut = int(len(q_perm) * float(args.train_frac))
        cut = min(max(cut, 10), len(q_perm) - 10)
        q_train = set(q_perm[:cut])
        q_test = set(q_perm[cut:])
        train_records = [r for r in records if int(r.q_index) in q_train]
        test_records = [r for r in records if int(r.q_index) in q_test]

        ns, es, _, _, _, tw = aggregate_supports(train_records)
        core_nodes = pick_consensus_nodes(
            node_support=ns,
            node_threshold=args.core_node_th,
            min_nodes=args.min_nodes,
            max_nodes=args.max_core_nodes,
        )
        _core_edges = pick_consensus_edges(
            nodes=core_nodes,
            edge_support=es,
            node_support=ns,
            edge_threshold=args.core_edge_th,
            min_edges=max(10, len(core_nodes)),
        )
        replay = replay_global_circuit(
            records=test_records,
            global_nodes=core_nodes,
            model_path=args.model_path,
            device=args.device,
            n_random=args.replay_random,
            seed=seed + 1000,
        )

        row = {
            "split": f"seed{seed}",
            "seed": seed,
            "train_n": len(q_train),
            "test_n": len(q_test),
            "train_total_weight": tw,
            "core_nodes": core_nodes,
            "core_node_count": len(core_nodes),
            "core_vs_reference_jaccard": jaccard(set(core_nodes), set(reference_core)) if reference_core else float("nan"),
            "test_suff_median": float(replay.get("global_suff_ratio_median", float("nan"))),
            "test_nec_median": float(replay.get("global_nec_ratio_median", float("nan"))),
            "test_random_suff_median": float(replay.get("random_suff_ratio_mean_median", float("nan"))),
            "test_global_minus_random_median": float(replay.get("global_minus_random_median", float("nan"))),
            "test_replay_samples": int(replay.get("ran_samples", 0)),
        }
        rows.append(row)

    # Save row table.
    out_csv = out_root / "core_transfer_splits.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "split",
                "seed",
                "train_n",
                "test_n",
                "train_total_weight",
                "core_node_count",
                "core_nodes",
                "core_vs_reference_jaccard",
                "test_suff_median",
                "test_nec_median",
                "test_random_suff_median",
                "test_global_minus_random_median",
                "test_replay_samples",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r["split"],
                    r["seed"],
                    r["train_n"],
                    r["test_n"],
                    r["train_total_weight"],
                    r["core_node_count"],
                    "|".join(r["core_nodes"]),
                    r["core_vs_reference_jaccard"],
                    r["test_suff_median"],
                    r["test_nec_median"],
                    r["test_random_suff_median"],
                    r["test_global_minus_random_median"],
                    r["test_replay_samples"],
                ]
            )

    save_transfer_plot(rows=rows, out_path=out_root / "core_transfer_splits.png")

    def _med(key: str) -> float:
        vals = [float(r[key]) for r in rows if isinstance(r.get(key), (int, float))]
        return float(np.median(vals)) if vals else float("nan")

    report = {
        "n_splits": len(rows),
        "gap_min": args.gap_min,
        "train_frac": args.train_frac,
        "split_seeds": split_seeds,
        "aggregate_params": {
            "core_node_th": args.core_node_th,
            "core_edge_th": args.core_edge_th,
            "min_nodes": args.min_nodes,
            "max_core_nodes": args.max_core_nodes,
            "ap_discount": args.ap_discount,
        },
        "summary_medians": {
            "test_suff_median": _med("test_suff_median"),
            "test_nec_median": _med("test_nec_median"),
            "test_random_suff_median": _med("test_random_suff_median"),
            "test_global_minus_random_median": _med("test_global_minus_random_median"),
            "core_vs_reference_jaccard_median": _med("core_vs_reference_jaccard"),
        },
        "splits": rows,
        "artifacts": {
            "splits_csv": str(out_csv),
            "splits_png": str(out_root / "core_transfer_splits.png"),
        },
    }
    report_path = out_root / "core_transfer_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[done] output root: {out_root}")
    print(f"[done] report: {report_path}")


if __name__ == "__main__":
    main()

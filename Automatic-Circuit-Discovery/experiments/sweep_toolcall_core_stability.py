#!/usr/bin/env python3
"""
Sweep aggregation hyperparameters and quantify global-core stability.

This script reuses aggregate_toolcall_circuits utilities to test whether
the discovered core remains stable under different:
- gap filters,
- node support thresholds,
- edge support thresholds.

Outputs:
- per-variant core summary CSV/JSON
- pairwise Jaccard matrix tables
- node/edge frequency across variants
- stable-core proposal from frequency criterion
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from itertools import combinations, product
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.aggregate_toolcall_circuits import (
    aggregate_supports,
    load_sample_records,
    pick_consensus_edges,
    pick_consensus_nodes,
)
from experiments.launch_toolcall_qwen3_q85 import node_layer


def parse_list(raw: str) -> List[float]:
    vals: List[float] = []
    for x in raw.split(","):
        x = x.strip()
        if not x:
            continue
        vals.append(float(x))
    return vals


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def edge_key(e: Tuple[str, str]) -> str:
    return f"{e[0]}->{e[1]}"


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


def save_frequency_bar(freq: Dict[str, float], title: str, out_path: Path) -> None:
    if not freq:
        return
    items = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    labels = [k for k, _ in items]
    vals = [v for _, v in items]
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(max(9.5, 0.45 * len(labels)), 4.8), constrained_layout=True)
    ax.bar(np.arange(len(labels)), vals, color="#2a6f97", edgecolor="#1f1f1f", linewidth=0.8)
    ax.axhline(0.7, color="#8b0000", linewidth=1.2, linestyle="--", label="stable>=0.70")
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Frequency across variants")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_pairwise_heatmap(
    labels: Sequence[str],
    matrix: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    if matrix.size == 0:
        return
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(0.55 * len(labels) + 2.8, 0.55 * len(labels) + 2.4))
    im = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=70, ha="right")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cbar.set_label("Jaccard")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def clean_variant_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.=-]+", "_", name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep tool-call core stability across aggregation hyperparameters.")
    parser.add_argument("--input-root", type=str, default="experiments/results/toolcall_q1_q164")
    parser.add_argument("--output-root", type=str, default="experiments/results/toolcall_q1_q164_core_stability")
    parser.add_argument("--gap-list", type=str, default="0.4,0.5,0.6")
    parser.add_argument("--core-node-th-list", type=str, default="0.45,0.50,0.55")
    parser.add_argument("--core-edge-th-list", type=str, default="0.30,0.35,0.40")
    parser.add_argument("--ap-discount", type=float, default=0.7)
    parser.add_argument("--min-nodes", type=int, default=8)
    parser.add_argument("--max-core-nodes", type=int, default=18)
    parser.add_argument("--stable-node-freq", type=float, default=0.7)
    parser.add_argument("--stable-edge-freq", type=float, default=0.7)
    args = parser.parse_args()

    input_root = Path(args.input_root).resolve()
    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    gap_list = parse_list(args.gap_list)
    node_th_list = parse_list(args.core_node_th_list)
    edge_th_list = parse_list(args.core_edge_th_list)
    if not gap_list or not node_th_list or not edge_th_list:
        raise ValueError("All sweep lists must be non-empty.")

    variant_rows: List[Dict[str, object]] = []
    variant_nodes: Dict[str, Set[str]] = {}
    variant_edges: Dict[str, Set[str]] = {}

    for gap_min, node_th, edge_th in product(gap_list, node_th_list, edge_th_list):
        label = clean_variant_name(f"gap{gap_min:.2f}_n{node_th:.2f}_e{edge_th:.2f}")
        records = load_sample_records(root=input_root, gap_min=gap_min, ap_discount=args.ap_discount)
        if not records:
            continue
        node_support, edge_support, _, _, _, total_w = aggregate_supports(records)
        if not node_support:
            continue

        core_nodes = pick_consensus_nodes(
            node_support=node_support,
            node_threshold=node_th,
            min_nodes=args.min_nodes,
            max_nodes=args.max_core_nodes,
        )
        core_edges = pick_consensus_edges(
            nodes=core_nodes,
            edge_support=edge_support,
            node_support=node_support,
            edge_threshold=edge_th,
            min_edges=max(10, len(core_nodes)),
        )
        active_records = [r for r in records if float(r.weight) > 0.0]
        variant_rows.append(
            {
                "variant": label,
                "gap_min": gap_min,
                "core_node_th": node_th,
                "core_edge_th": edge_th,
                "n_records_total": len(records),
                "n_records_active": len(active_records),
                "total_weight": total_w,
                "n_core_nodes": len(core_nodes),
                "n_core_edges": len(core_edges),
                "core_nodes": core_nodes,
                "core_edges": core_edges,
            }
        )
        variant_nodes[label] = set(core_nodes)
        variant_edges[label] = set(edge_key((a, b)) for a, b in core_edges)

    if not variant_rows:
        raise ValueError("No valid variants were produced.")

    # Save variant summary.
    variant_rows_sorted = sorted(variant_rows, key=lambda x: str(x["variant"]))
    variant_csv = out_root / "core_stability_variants.csv"
    with variant_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "variant",
                "gap_min",
                "core_node_th",
                "core_edge_th",
                "n_records_total",
                "n_records_active",
                "total_weight",
                "n_core_nodes",
                "n_core_edges",
                "core_nodes",
                "core_edges",
            ]
        )
        for r in variant_rows_sorted:
            w.writerow(
                [
                    r["variant"],
                    r["gap_min"],
                    r["core_node_th"],
                    r["core_edge_th"],
                    r["n_records_total"],
                    r["n_records_active"],
                    r["total_weight"],
                    r["n_core_nodes"],
                    r["n_core_edges"],
                    "|".join(r["core_nodes"]),
                    "|".join(edge_key(e) for e in r["core_edges"]),
                ]
            )

    labels = [str(r["variant"]) for r in variant_rows_sorted]
    n = len(labels)
    node_j = np.zeros((n, n), dtype=np.float64)
    edge_j = np.zeros((n, n), dtype=np.float64)
    pair_rows: List[Dict[str, object]] = []
    for i in range(n):
        for j in range(n):
            a, b = labels[i], labels[j]
            node_j[i, j] = jaccard(variant_nodes[a], variant_nodes[b])
            edge_j[i, j] = jaccard(variant_edges[a], variant_edges[b])
    for i, j in combinations(range(n), 2):
        pair_rows.append(
            {
                "variant_a": labels[i],
                "variant_b": labels[j],
                "node_jaccard": float(node_j[i, j]),
                "edge_jaccard": float(edge_j[i, j]),
            }
        )

    pair_csv = out_root / "core_stability_pairwise.csv"
    pd_rows = sorted(pair_rows, key=lambda x: (x["node_jaccard"], x["edge_jaccard"]), reverse=True)
    with pair_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["variant_a", "variant_b", "node_jaccard", "edge_jaccard"])
        w.writeheader()
        w.writerows(pd_rows)

    node_counter: Dict[str, int] = defaultdict(int)
    edge_counter: Dict[str, int] = defaultdict(int)
    for label in labels:
        for nname in variant_nodes[label]:
            node_counter[nname] += 1
        for ename in variant_edges[label]:
            edge_counter[ename] += 1

    denom = max(1, len(labels))
    node_freq = {k: v / denom for k, v in node_counter.items()}
    edge_freq = {k: v / denom for k, v in edge_counter.items()}
    node_freq = dict(sorted(node_freq.items(), key=lambda x: (x[1], -node_layer(x[0])), reverse=True))
    edge_freq = dict(sorted(edge_freq.items(), key=lambda x: x[1], reverse=True))

    stable_nodes = sorted(
        [nname for nname, freq in node_freq.items() if freq >= args.stable_node_freq],
        key=lambda nname: (node_layer(nname), 0 if nname.startswith("MLP") else 1, nname),
    )
    stable_edges = sorted([ename for ename, freq in edge_freq.items() if freq >= args.stable_edge_freq])

    # Frequency tables.
    node_freq_csv = out_root / "core_stability_node_frequency.csv"
    with node_freq_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["node", "frequency"])
        for k, v in node_freq.items():
            w.writerow([k, v])

    edge_freq_csv = out_root / "core_stability_edge_frequency.csv"
    with edge_freq_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["edge", "frequency"])
        for k, v in edge_freq.items():
            w.writerow([k, v])

    save_frequency_bar(
        freq=node_freq,
        title="Core Node Frequency Across Sweep Variants",
        out_path=out_root / "core_stability_node_frequency.png",
    )
    save_frequency_bar(
        freq=edge_freq,
        title="Core Edge Frequency Across Sweep Variants",
        out_path=out_root / "core_stability_edge_frequency.png",
    )
    save_pairwise_heatmap(
        labels=labels,
        matrix=node_j,
        title="Node Jaccard Across Variants",
        out_path=out_root / "core_stability_node_jaccard_heatmap.png",
    )
    save_pairwise_heatmap(
        labels=labels,
        matrix=edge_j,
        title="Edge Jaccard Across Variants",
        out_path=out_root / "core_stability_edge_jaccard_heatmap.png",
    )

    summary = {
        "input_root": str(input_root),
        "grid": {
            "gap_list": gap_list,
            "core_node_th_list": node_th_list,
            "core_edge_th_list": edge_th_list,
            "ap_discount": args.ap_discount,
            "min_nodes": args.min_nodes,
            "max_core_nodes": args.max_core_nodes,
        },
        "n_variants": len(labels),
        "pairwise_node_jaccard_mean": float(np.mean([r["node_jaccard"] for r in pair_rows])) if pair_rows else float("nan"),
        "pairwise_node_jaccard_median": float(np.median([r["node_jaccard"] for r in pair_rows])) if pair_rows else float("nan"),
        "pairwise_edge_jaccard_mean": float(np.mean([r["edge_jaccard"] for r in pair_rows])) if pair_rows else float("nan"),
        "pairwise_edge_jaccard_median": float(np.median([r["edge_jaccard"] for r in pair_rows])) if pair_rows else float("nan"),
        "stable_thresholds": {
            "node_freq": args.stable_node_freq,
            "edge_freq": args.stable_edge_freq,
        },
        "stable_nodes": stable_nodes,
        "stable_edges": stable_edges,
        "artifacts": {
            "variants_csv": str(variant_csv),
            "pairwise_csv": str(pair_csv),
            "node_frequency_csv": str(node_freq_csv),
            "edge_frequency_csv": str(edge_freq_csv),
            "node_frequency_png": str(out_root / "core_stability_node_frequency.png"),
            "edge_frequency_png": str(out_root / "core_stability_edge_frequency.png"),
            "node_jaccard_heatmap_png": str(out_root / "core_stability_node_jaccard_heatmap.png"),
            "edge_jaccard_heatmap_png": str(out_root / "core_stability_edge_jaccard_heatmap.png"),
        },
    }
    report_path = out_root / "core_stability_report.json"
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[done] output root: {out_root}")
    print(f"[done] variants: {len(labels)}")
    print(f"[done] report: {report_path}")


if __name__ == "__main__":
    main()

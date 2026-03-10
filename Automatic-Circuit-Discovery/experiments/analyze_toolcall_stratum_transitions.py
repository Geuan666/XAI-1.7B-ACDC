#!/usr/bin/env python3
"""
Compare node stratification states across multiple conditions.

Outputs:
- per-node stratum trajectory table
- transition-count table between adjacent conditions
- compact categorical heatmap
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap


STRATUM_ORDER = [
    "stable_necessary_backbone",
    "stable_but_weak_or_redundant",
    "unstable_but_necessary",
    "unstable_weak",
]
STRATUM_TO_ID = {s: i for i, s in enumerate(STRATUM_ORDER)}


def parse_input_specs(spec: str) -> List[Tuple[str, Path]]:
    out: List[Tuple[str, Path]] = []
    parts = [x.strip() for x in spec.split(",") if x.strip()]
    for p in parts:
        if ":" not in p:
            raise ValueError(f"Bad input spec: {p}")
        label, path = p.split(":", 1)
        label = label.strip()
        csv_path = Path(path.strip()).resolve()
        if not label:
            raise ValueError(f"Bad label in spec: {p}")
        out.append((label, csv_path))
    if len(out) < 2:
        raise ValueError("Need at least two labeled inputs.")
    return out


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze node stratum transitions across conditions.")
    parser.add_argument(
        "--inputs",
        type=str,
        default=(
            "orig:experiments/results/toolcall_q1_q164_node_stratification_v1/node_stratification.csv,"
            "system_json_pad:experiments/results/toolcall_q1_q164_node_stratification_system_json_v1/node_stratification.csv,"
            "user_json_pad:experiments/results/toolcall_q1_q164_node_stratification_user_json_v1/node_stratification.csv"
        ),
        help="Comma-separated label:path specs.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="experiments/results/toolcall_q1_q164_stratum_transition_v1",
    )
    args = parser.parse_args()

    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    input_specs = parse_input_specs(args.inputs)
    label_to_map: Dict[str, Dict[str, str]] = {}
    all_nodes = set()
    for label, csv_path in input_specs:
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        df = pd.read_csv(csv_path)
        if "node" not in df.columns or "stratum" not in df.columns:
            raise ValueError(f"Missing required columns in {csv_path}")
        mapping = {str(r["node"]): str(r["stratum"]) for _, r in df.iterrows()}
        label_to_map[label] = mapping
        all_nodes.update(mapping.keys())

    labels = [label for label, _ in input_specs]
    nodes = sorted(all_nodes)

    node_rows: List[Dict[str, object]] = []
    for n in nodes:
        row: Dict[str, object] = {"node": n}
        for label in labels:
            row[label] = label_to_map[label].get(n, "missing")
        node_rows.append(row)

    node_csv = out_root / "node_stratum_trajectories.csv"
    with node_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["node"] + labels)
        writer.writeheader()
        for r in node_rows:
            writer.writerow(r)

    transition_rows: List[Dict[str, object]] = []
    for i in range(len(labels) - 1):
        src = labels[i]
        dst = labels[i + 1]
        counts: Dict[Tuple[str, str], int] = {}
        for n in nodes:
            s = str(label_to_map[src].get(n, "missing"))
            t = str(label_to_map[dst].get(n, "missing"))
            counts[(s, t)] = counts.get((s, t), 0) + 1
        for (s, t), c in sorted(counts.items(), key=lambda x: (-x[1], x[0][0], x[0][1])):
            transition_rows.append(
                {
                    "from_condition": src,
                    "to_condition": dst,
                    "from_stratum": s,
                    "to_stratum": t,
                    "count": int(c),
                }
            )

    trans_csv = out_root / "stratum_transition_counts.csv"
    with trans_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "from_condition",
                "to_condition",
                "from_stratum",
                "to_stratum",
                "count",
            ],
        )
        writer.writeheader()
        for r in transition_rows:
            writer.writerow(r)

    # Categorical heatmap.
    matrix = np.full((len(nodes), len(labels)), fill_value=-1, dtype=np.int64)
    for ri, n in enumerate(nodes):
        for ci, label in enumerate(labels):
            s = str(label_to_map[label].get(n, "missing"))
            matrix[ri, ci] = STRATUM_TO_ID.get(s, -1)

    cmap = ListedColormap(["#1b9e77", "#d95f02", "#7570b3", "#999999", "#f0f0f0"])
    plot_mat = matrix.copy()
    plot_mat[plot_mat < 0] = 4

    apply_plot_style()
    fig, ax = plt.subplots(figsize=(1.5 * len(labels) + 3.2, 0.52 * len(nodes) + 1.8), constrained_layout=True)
    im = ax.imshow(plot_mat, cmap=cmap, aspect="auto", vmin=0, vmax=4)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(nodes)))
    ax.set_yticklabels(nodes)
    ax.set_title("Node Stratum Trajectories Across Conditions")
    ax.set_xlabel("Condition")
    ax.set_ylabel("Node")
    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cbar.set_ticks([0, 1, 2, 3, 4])
    cbar.set_ticklabels(
        [
            "stable_necessary_backbone",
            "stable_but_weak_or_redundant",
            "unstable_but_necessary",
            "unstable_weak",
            "missing",
        ]
    )
    fig_path = out_root / "node_stratum_trajectories_heatmap.png"
    fig.savefig(fig_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    report = {
        "labels": labels,
        "n_nodes": len(nodes),
        "stratum_order": STRATUM_ORDER,
        "artifacts": {
            "node_stratum_trajectories_csv": str(node_csv),
            "stratum_transition_counts_csv": str(trans_csv),
            "node_stratum_trajectories_heatmap_png": str(fig_path),
        },
    }
    report_path = out_root / "stratum_transition_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[done] output root: {out_root}")
    print(f"[done] report: {report_path}")


if __name__ == "__main__":
    main()

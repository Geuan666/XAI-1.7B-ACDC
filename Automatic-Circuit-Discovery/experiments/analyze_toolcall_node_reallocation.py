#!/usr/bin/env python3
"""
Paired node-level mechanism reallocation analysis between two ablation runs.

Typical use:
- baseline: orig prompts
- shifted:  user_json_pad prompts

The script aligns (q_index, node) entries, computes paired deltas on selected
metric(s), estimates bootstrap CI, and renders a publication-friendly delta
figure for direct mechanism-shift evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import DefaultDict, Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def finite(values: Iterable[float]) -> List[float]:
    out: List[float] = []
    for x in values:
        if isinstance(x, (int, float)) and math.isfinite(float(x)):
            out.append(float(x))
    return out


def med(values: Iterable[float]) -> float:
    vals = finite(values)
    return float(median(vals)) if vals else float("nan")


def bootstrap_median_ci(values: Sequence[float], n_boot: int, seed: int) -> Dict[str, float]:
    vals = finite(values)
    if not vals:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    rng = random.Random(seed)
    n = len(vals)
    boots: List[float] = []
    for _ in range(n_boot):
        sample = [vals[rng.randrange(n)] for __ in range(n)]
        boots.append(float(np.median(sample)))
    boots.sort()
    lo_idx = max(0, int(0.025 * n_boot))
    hi_idx = min(n_boot - 1, int(0.975 * n_boot))
    return {
        "mean": float(np.mean(boots)),
        "lo": float(boots[lo_idx]),
        "hi": float(boots[hi_idx]),
    }


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


def read_metric_rows(
    csv_path: Path,
    metric: str,
) -> Dict[Tuple[int, str], Dict[str, object]]:
    out: Dict[Tuple[int, str], Dict[str, object]] = {}
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                q_index = int(row["q_index"])
                node = str(row["node"])
                value = float(row[metric])
            except Exception:
                continue
            if not math.isfinite(value):
                continue
            key = (q_index, node)
            out[key] = {
                "q_index": q_index,
                "node": node,
                "node_role": str(row.get("node_role", "")),
                "value": value,
            }
    return out


def save_delta_barh(
    rows: Sequence[Dict[str, object]],
    baseline_label: str,
    shifted_label: str,
    metric: str,
    out_path: Path,
) -> None:
    if not rows:
        return
    labels = [str(r["node"]) for r in rows]
    y = np.arange(len(rows))
    delta = np.array([float(r["delta_median"]) for r in rows], dtype=np.float64)
    lo = np.array([float(r["delta_ci_lo"]) for r in rows], dtype=np.float64)
    hi = np.array([float(r["delta_ci_hi"]) for r in rows], dtype=np.float64)
    err = np.vstack([np.maximum(0.0, delta - lo), np.maximum(0.0, hi - delta)])
    colors = ["#c0392b" if d > 0 else "#2c7fb8" for d in delta]

    apply_plot_style()
    fig, ax = plt.subplots(figsize=(9.8, max(4.5, 0.58 * len(rows) + 1.8)), constrained_layout=True)
    ax.barh(y, delta, color=colors, edgecolor="#1f1f1f", linewidth=0.7, alpha=0.92)
    ax.errorbar(delta, y, xerr=err, fmt="none", ecolor="#1f1f1f", elinewidth=1.0, capsize=3)
    ax.axvline(0.0, color="#444444", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(f"Delta median ({shifted_label} - {baseline_label})")
    ax.set_ylabel("Node")
    ax.set_title(f"Node Mechanism Reallocation ({metric})")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired node-level mechanism reallocation analysis.")
    parser.add_argument(
        "--baseline-per-sample",
        type=str,
        default="experiments/results/toolcall_q1_q164_node_ablation_v1/node_ablation_per_sample.csv",
    )
    parser.add_argument(
        "--shifted-per-sample",
        type=str,
        default="experiments/results/toolcall_q1_q164_node_ablation_user_json_v1/node_ablation_per_sample.csv",
    )
    parser.add_argument("--baseline-label", type=str, default="orig")
    parser.add_argument("--shifted-label", type=str, default="user_json_pad")
    parser.add_argument("--metric", type=str, default="drop_full_nec")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--output-root",
        type=str,
        default="experiments/results/toolcall_q1_q164_node_reallocation_v1",
    )
    args = parser.parse_args()

    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    baseline_csv = Path(args.baseline_per_sample).resolve()
    shifted_csv = Path(args.shifted_per_sample).resolve()

    base = read_metric_rows(baseline_csv, metric=args.metric)
    shift = read_metric_rows(shifted_csv, metric=args.metric)
    common_keys = sorted(set(base.keys()) & set(shift.keys()))
    if not common_keys:
        raise ValueError("No common (q_index, node) rows between baseline and shifted inputs.")

    by_node_delta: DefaultDict[str, List[float]] = defaultdict(list)
    by_node_base: DefaultDict[str, List[float]] = defaultdict(list)
    by_node_shift: DefaultDict[str, List[float]] = defaultdict(list)
    node_roles: Dict[str, str] = {}

    common_q_indices = set()
    for key in common_keys:
        q_index, node = key
        b = float(base[key]["value"])
        s = float(shift[key]["value"])
        if not (math.isfinite(b) and math.isfinite(s)):
            continue
        d = s - b
        by_node_delta[node].append(d)
        by_node_base[node].append(b)
        by_node_shift[node].append(s)
        node_roles[node] = str(base[key].get("node_role", "")) or str(shift[key].get("node_role", ""))
        common_q_indices.add(q_index)

    summary_rows: List[Dict[str, object]] = []
    for i, node in enumerate(sorted(by_node_delta.keys())):
        deltas = by_node_delta[node]
        base_vals = by_node_base[node]
        shift_vals = by_node_shift[node]
        ci = bootstrap_median_ci(deltas, n_boot=args.bootstrap, seed=args.seed + 17 * (i + 1))
        summary_rows.append(
            {
                "node": node,
                "node_role": node_roles.get(node, ""),
                "n_pairs": len(deltas),
                "baseline_median": med(base_vals),
                "shifted_median": med(shift_vals),
                "delta_median": med(deltas),
                "delta_mean": float(np.mean(deltas)) if deltas else float("nan"),
                "delta_ci_lo": float(ci["lo"]),
                "delta_ci_hi": float(ci["hi"]),
            }
        )

    summary_rows.sort(key=lambda r: float(r["delta_median"]), reverse=True)

    summary_csv = out_root / "node_reallocation_delta_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "node",
                "node_role",
                "n_pairs",
                "baseline_median",
                "shifted_median",
                "delta_median",
                "delta_mean",
                "delta_ci_lo",
                "delta_ci_hi",
            ]
        )
        for r in summary_rows:
            writer.writerow(
                [
                    r["node"],
                    r["node_role"],
                    r["n_pairs"],
                    r["baseline_median"],
                    r["shifted_median"],
                    r["delta_median"],
                    r["delta_mean"],
                    r["delta_ci_lo"],
                    r["delta_ci_hi"],
                ]
            )

    fig_path = out_root / "node_reallocation_delta_barh.png"
    save_delta_barh(
        rows=summary_rows,
        baseline_label=args.baseline_label,
        shifted_label=args.shifted_label,
        metric=args.metric,
        out_path=fig_path,
    )

    report = {
        "metric": args.metric,
        "baseline_label": args.baseline_label,
        "shifted_label": args.shifted_label,
        "n_common_rows": len(common_keys),
        "n_common_q_indices": len(common_q_indices),
        "common_q_indices": sorted(common_q_indices),
        "bootstrap": args.bootstrap,
        "seed": args.seed,
        "summary_rows": summary_rows,
        "artifacts": {
            "summary_csv": str(summary_csv),
            "delta_barh_png": str(fig_path),
        },
    }
    report_path = out_root / "node_reallocation_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[done] output root: {out_root}")
    print(f"[done] report: {report_path}")


if __name__ == "__main__":
    main()

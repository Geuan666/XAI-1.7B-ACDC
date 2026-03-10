#!/usr/bin/env python3
"""
Stratify core nodes by structural stability and causal necessity.

Inputs:
- node ablation summary (drop_full_nec etc.)
- core stability node frequency table

Outputs:
- merged stratification CSV/JSON
- scatter figure for paper-ready interpretation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def assign_stratum(freq: float, nec: float, freq_th: float, nec_th: float) -> str:
    stable = float(freq) >= freq_th
    necessary = float(nec) >= nec_th
    if stable and necessary:
        return "stable_necessary_backbone"
    if stable and not necessary:
        return "stable_but_weak_or_redundant"
    if (not stable) and necessary:
        return "unstable_but_necessary"
    return "unstable_weak"


def main() -> None:
    parser = argparse.ArgumentParser(description="Stratify tool-call nodes by stability and necessity.")
    parser.add_argument(
        "--node-ablation-summary",
        type=str,
        default="experiments/results/toolcall_q1_q164_node_ablation_v1/node_ablation_summary.csv",
    )
    parser.add_argument(
        "--node-frequency",
        type=str,
        default="experiments/results/toolcall_q1_q164_core_stability_v1/core_stability_node_frequency.csv",
    )
    parser.add_argument("--output-root", type=str, default="experiments/results/toolcall_q1_q164_node_stratification")
    parser.add_argument("--stable-freq-th", type=float, default=0.70)
    parser.add_argument("--necessary-drop-th", type=float, default=0.02)
    args = parser.parse_args()

    ablation_path = Path(args.node_ablation_summary).resolve()
    freq_path = Path(args.node_frequency).resolve()
    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if not ablation_path.exists():
        raise FileNotFoundError(ablation_path)
    if not freq_path.exists():
        raise FileNotFoundError(freq_path)

    ab = pd.read_csv(ablation_path)
    fr = pd.read_csv(freq_path).rename(columns={"frequency": "stability_freq"})
    merged = ab.merge(fr, on="node", how="left")
    merged["stability_freq"] = merged["stability_freq"].fillna(0.0)
    merged["stratum"] = merged.apply(
        lambda r: assign_stratum(
            freq=float(r["stability_freq"]),
            nec=float(r["drop_full_nec_median"]),
            freq_th=float(args.stable_freq_th),
            nec_th=float(args.necessary_drop_th),
        ),
        axis=1,
    )
    merged = merged.sort_values(by=["stability_freq", "drop_full_nec_median"], ascending=[False, False]).reset_index(drop=True)

    out_csv = out_root / "node_stratification.csv"
    merged.to_csv(out_csv, index=False)

    # Scatter figure.
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(8.8, 6.2), constrained_layout=True)
    palette: Dict[str, str] = {
        "stable_necessary_backbone": "#1b9e77",
        "stable_but_weak_or_redundant": "#d95f02",
        "unstable_but_necessary": "#7570b3",
        "unstable_weak": "#999999",
    }
    for stratum, dfg in merged.groupby("stratum"):
        ax.scatter(
            dfg["stability_freq"].values,
            dfg["drop_full_nec_median"].values,
            s=80,
            label=stratum,
            color=palette.get(stratum, "#444444"),
            edgecolors="#1f1f1f",
            linewidths=0.7,
            alpha=0.92,
        )
        for r in dfg.itertuples(index=False):
            ax.text(float(r.stability_freq) + 0.008, float(r.drop_full_nec_median) + 0.001, str(r.node), fontsize=9)

    ax.axvline(float(args.stable_freq_th), color="#444444", linestyle="--", linewidth=1.2)
    ax.axhline(float(args.necessary_drop_th), color="#444444", linestyle="--", linewidth=1.2)
    ax.set_xlim(-0.02, 1.05)
    ymin = float(np.nanmin(merged["drop_full_nec_median"].values)) - 0.03
    ymax = float(np.nanmax(merged["drop_full_nec_median"].values)) + 0.03
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Structural stability frequency across sweep variants")
    ax.set_ylabel("Causal necessity in full circuit (drop_full_nec_median)")
    ax.set_title("Node Stratification: Stability vs Necessity")
    ax.legend(loc="best")
    scatter_png = out_root / "node_stratification_scatter.png"
    fig.savefig(scatter_png, dpi=220, bbox_inches="tight")
    plt.close(fig)

    stratum_counts = (
        merged.groupby("stratum")["node"].apply(list).to_dict()
        if not merged.empty
        else {}
    )
    report = {
        "stable_freq_th": args.stable_freq_th,
        "necessary_drop_th": args.necessary_drop_th,
        "n_nodes": int(len(merged)),
        "strata": {k: [str(x) for x in v] for k, v in stratum_counts.items()},
        "artifacts": {
            "node_stratification_csv": str(out_csv),
            "node_stratification_scatter_png": str(scatter_png),
        },
    }
    report_path = out_root / "node_stratification_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[done] output root: {out_root}")
    print(f"[done] report: {report_path}")


if __name__ == "__main__":
    main()

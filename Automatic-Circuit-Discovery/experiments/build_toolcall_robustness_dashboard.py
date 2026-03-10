#!/usr/bin/env python3
"""
Build a robustness dashboard for tool-call circuit evidence.

Inputs:
- role-group per-sample causal table
- role-group report (for group names)
- optional edge path-patching per-sample table
- optional summary root (to derive prompt-length slices)

Outputs:
- slice-wise robustness tables (CSV/JSON)
- gap-bin group-necessity table
- publication-friendly summary figures
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def finite(xs: Iterable[float]) -> List[float]:
    out: List[float] = []
    for x in xs:
        if isinstance(x, (int, float)) and math.isfinite(float(x)):
            out.append(float(x))
    return out


def med(xs: Iterable[float]) -> float:
    vals = finite(xs)
    return float(np.median(vals)) if vals else float("nan")


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


def infer_prompt_lengths(input_root: Path) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for q_dir in sorted(input_root.glob("q[0-9][0-9][0-9]")):
        sp = q_dir / "summary.json"
        if not sp.exists():
            continue
        try:
            summary = json.loads(sp.read_text(encoding="utf-8"))
            q = int(summary.get("q_index", -1))
            cp = Path(summary["clean_prompt"])
            if q <= 0 or not cp.exists():
                continue
            out[q] = len(cp.read_text(encoding="utf-8"))
        except Exception:
            continue
    return out


def build_slices(
    full_df: pd.DataFrame,
    prompt_lens: Dict[int, int],
    seed: int,
) -> Dict[str, List[int]]:
    rng = np.random.default_rng(seed)
    q_all = sorted(int(x) for x in full_df["q_index"].unique().tolist())
    if not q_all:
        return {"all": []}
    slices: Dict[str, List[int]] = {"all": q_all}

    gap_by_q = {int(r.q_index): float(r.gap) for r in full_df.itertuples(index=False)}
    gaps = np.array([gap_by_q[q] for q in q_all], dtype=np.float64)
    q25, q75 = np.quantile(gaps, [0.25, 0.75])
    slices["gap_q1"] = [q for q in q_all if gap_by_q[q] <= q25]
    slices["gap_q4"] = [q for q in q_all if gap_by_q[q] >= q75]
    slices["gap_mid"] = [q for q in q_all if q25 < gap_by_q[q] < q75]

    hard = full_df[(full_df["suff_ratio"] < 0.85) | (full_df["nec_ratio"] < 0.85)]["q_index"].astype(int).tolist()
    easy = full_df[(full_df["suff_ratio"] >= 0.95) & (full_df["nec_ratio"] >= 0.95)]["q_index"].astype(int).tolist()
    slices["hard_cases"] = sorted(set(hard))
    slices["easy_cases"] = sorted(set(easy))

    arr = np.array(q_all, dtype=np.int32)
    rng.shuffle(arr)
    half = len(arr) // 2
    slices["random_half_a"] = sorted(int(x) for x in arr[:half].tolist())
    slices["random_half_b"] = sorted(int(x) for x in arr[half:].tolist())

    if prompt_lens:
        common = [q for q in q_all if q in prompt_lens]
        if len(common) >= 20:
            lens = np.array([prompt_lens[q] for q in common], dtype=np.float64)
            l25, l75 = np.quantile(lens, [0.25, 0.75])
            slices["prompt_len_q1"] = [q for q in common if prompt_lens[q] <= l25]
            slices["prompt_len_q4"] = [q for q in common if prompt_lens[q] >= l75]

    return {k: v for k, v in slices.items() if v}


def summarise_slice(
    role_df: pd.DataFrame,
    q_subset: Sequence[int],
    groups: Sequence[str],
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    subset = role_df[role_df["q_index"].isin(list(q_subset))]
    group_rows: List[Dict[str, object]] = []
    for g in groups:
        dfg = subset[subset["group"] == g]
        if dfg.empty:
            continue
        group_rows.append(
            {
                "group": g,
                "n_samples": int(dfg["q_index"].nunique()),
                "suff_median": med(dfg["suff_ratio"]),
                "nec_median": med(dfg["nec_ratio"]),
                "suff_minus_median": med(dfg["suff_minus_ratio"]),
                "nec_minus_median": med(dfg["nec_minus_ratio"]),
                "drop_full_suff_median": med(dfg["delta_full_suff_drop"]),
                "drop_full_nec_median": med(dfg["delta_full_nec_drop"]),
            }
        )
    row_by_group = {str(r["group"]): r for r in group_rows}
    full = row_by_group.get("full_core", {})
    summary = {
        "n_samples": int(len(set(int(x) for x in q_subset))),
        "full_core_suff_median": float(full.get("suff_median", float("nan"))),
        "full_core_nec_median": float(full.get("nec_median", float("nan"))),
        "format_router_drop_nec_median": float(row_by_group.get("format_router", {}).get("drop_full_nec_median", float("nan"))),
        "query_reader_drop_nec_median": float(row_by_group.get("query_reader", {}).get("drop_full_nec_median", float("nan"))),
        "tool_tag_reader_drop_nec_median": float(row_by_group.get("tool_tag_reader", {}).get("drop_full_nec_median", float("nan"))),
        "all_heads_drop_nec_median": float(row_by_group.get("all_heads", {}).get("drop_full_nec_median", float("nan"))),
        "all_mlps_drop_nec_median": float(row_by_group.get("all_mlps", {}).get("drop_full_nec_median", float("nan"))),
    }
    return group_rows, summary


def summarise_edge_slice(edge_df: pd.DataFrame, q_subset: Sequence[int]) -> Dict[str, object]:
    if edge_df.empty:
        return {}
    dfe = edge_df[edge_df["q_index"].isin(list(q_subset))]
    if dfe.empty:
        return {}
    by_edge = dfe.groupby("edge")["edge_ratio"].median().sort_values(ascending=False)
    medians = by_edge.to_dict()
    abs_signal = [abs(float(v)) for v in medians.values() if math.isfinite(float(v))]
    weak_frac = float(np.mean([1.0 if float(v) < 0.05 else 0.0 for v in medians.values()])) if medians else float("nan")
    top5 = [{"edge": str(k), "median": float(v)} for k, v in by_edge.head(5).items()]
    bot5 = [{"edge": str(k), "median": float(v)} for k, v in by_edge.tail(5).items()]
    return {
        "n_edge_rows": int(len(dfe)),
        "n_unique_edges": int(len(by_edge)),
        "edge_abs_signal_median": med(abs_signal),
        "edge_abs_signal_mean": float(np.mean(abs_signal)) if abs_signal else float("nan"),
        "edge_weak_frac_median_lt_0p05": weak_frac,
        "top5_edges": top5,
        "bottom5_edges": bot5,
    }


def save_slice_heatmap(slice_df: pd.DataFrame, out_path: Path) -> None:
    cols = [
        "full_core_suff_median",
        "full_core_nec_median",
        "format_router_drop_nec_median",
        "query_reader_drop_nec_median",
        "tool_tag_reader_drop_nec_median",
        "all_heads_drop_nec_median",
        "all_mlps_drop_nec_median",
    ]
    use = slice_df.copy()
    use = use[use["slice"] != "all"].copy()
    if use.empty:
        use = slice_df.copy()
    matrix = use[cols].to_numpy(dtype=np.float64)
    rows = use["slice"].tolist()

    apply_plot_style()
    fig, ax = plt.subplots(figsize=(1.25 * len(cols) + 1.2, 0.65 * len(rows) + 2.2))
    vals = matrix[np.isfinite(matrix)]
    vmax = float(np.percentile(np.abs(vals), 98.0)) if vals.size > 0 else 1.0
    vmax = max(vmax, 1e-6)
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=28, ha="right")
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(rows)
    ax.set_title("Robustness Dashboard by Slice")
    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cbar.set_label("Median value")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_gap_bin_plot(gap_bin_df: pd.DataFrame, groups: Sequence[str], out_path: Path) -> None:
    apply_plot_style()
    bins = ["gap_Q1", "gap_Q2", "gap_Q3", "gap_Q4"]
    fig, ax = plt.subplots(figsize=(9.4, 5.0), constrained_layout=True)
    x = np.arange(len(bins))
    palette = {
        "format_router": "#1b9e77",
        "query_reader": "#d95f02",
        "tool_tag_reader": "#7570b3",
        "all_heads": "#1f78b4",
        "all_mlps": "#e31a1c",
    }
    for g in groups:
        ys: List[float] = []
        for b in bins:
            sel = gap_bin_df[(gap_bin_df["gap_bin"] == b) & (gap_bin_df["group"] == g)]
            ys.append(float(sel["drop_full_nec_median"].iloc[0]) if not sel.empty else float("nan"))
        ax.plot(x, ys, marker="o", linewidth=2.0, label=g, color=palette.get(g))
    ax.axhline(0.0, color="#444444", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(bins)
    ax.set_ylabel("drop_full_nec median")
    ax.set_title("Group Necessity Across Gap Bins")
    ax.legend(loc="best")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build tool-call robustness dashboard.")
    parser.add_argument(
        "--role-per-sample",
        type=str,
        default="experiments/results/toolcall_q1_q164_semantic_roles_v3/role_group_per_sample.csv",
    )
    parser.add_argument(
        "--role-report",
        type=str,
        default="experiments/results/toolcall_q1_q164_semantic_roles_v3/role_group_report.json",
    )
    parser.add_argument(
        "--edge-per-sample",
        type=str,
        default="experiments/results/toolcall_q1_q164_semantic_roles_v3/path_patch_edge_per_sample.csv",
    )
    parser.add_argument("--input-root", type=str, default="experiments/results/toolcall_q1_q164")
    parser.add_argument("--output-root", type=str, default="experiments/results/toolcall_q1_q164_robustness_dashboard")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    role_per_sample = Path(args.role_per_sample).resolve()
    role_report_path = Path(args.role_report).resolve()
    edge_per_sample = Path(args.edge_per_sample).resolve()
    input_root = Path(args.input_root).resolve()
    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if not role_per_sample.exists():
        raise FileNotFoundError(f"Missing role per-sample CSV: {role_per_sample}")
    if not role_report_path.exists():
        raise FileNotFoundError(f"Missing role report: {role_report_path}")

    role_df = pd.read_csv(role_per_sample)
    needed_cols = {
        "q_index",
        "group",
        "suff_ratio",
        "nec_ratio",
        "suff_minus_ratio",
        "nec_minus_ratio",
        "delta_full_suff_drop",
        "delta_full_nec_drop",
        "gap",
    }
    if not needed_cols.issubset(set(role_df.columns)):
        missing = sorted(needed_cols - set(role_df.columns))
        raise ValueError(f"role_per_sample missing columns: {missing}")

    role_report = json.loads(role_report_path.read_text(encoding="utf-8"))
    ext_groups = list(dict(role_report.get("extended_groups", {})).keys())
    groups = ["full_core"] + [g for g in ext_groups if g in set(role_df["group"].unique())]
    groups = [g for g in groups if g in set(role_df["group"].unique())]

    edge_df = pd.read_csv(edge_per_sample) if edge_per_sample.exists() else pd.DataFrame()
    prompt_lens = infer_prompt_lengths(input_root=input_root) if input_root.exists() else {}

    full_df = role_df[role_df["group"] == "full_core"].copy()
    if full_df.empty:
        raise ValueError("No full_core rows in role_per_sample.")
    slices = build_slices(full_df=full_df, prompt_lens=prompt_lens, seed=args.seed)

    slice_group_rows: List[Dict[str, object]] = []
    slice_summary_rows: List[Dict[str, object]] = []
    edge_summary: Dict[str, Dict[str, object]] = {}
    for slice_name, q_subset in slices.items():
        g_rows, s_row = summarise_slice(role_df=role_df, q_subset=q_subset, groups=groups)
        for r in g_rows:
            r2 = dict(r)
            r2["slice"] = slice_name
            slice_group_rows.append(r2)
        s_row["slice"] = slice_name
        slice_summary_rows.append(s_row)
        edge_summary[slice_name] = summarise_edge_slice(edge_df=edge_df, q_subset=q_subset)

    slice_summary_df = pd.DataFrame(slice_summary_rows).sort_values(by="n_samples", ascending=False)
    slice_group_df = pd.DataFrame(slice_group_rows).sort_values(by=["slice", "group"])
    slice_summary_csv = out_root / "robustness_slice_summary.csv"
    slice_group_csv = out_root / "robustness_slice_groups.csv"
    slice_summary_df.to_csv(slice_summary_csv, index=False)
    slice_group_df.to_csv(slice_group_csv, index=False)

    # Gap-bin diagnostics for key groups.
    q_gap = full_df[["q_index", "gap"]].drop_duplicates().copy()
    q_gap["gap_bin"] = pd.qcut(q_gap["gap"], q=4, labels=["gap_Q1", "gap_Q2", "gap_Q3", "gap_Q4"])
    gap_bin_df = role_df.merge(q_gap[["q_index", "gap_bin"]], on="q_index", how="inner")
    key_groups = [g for g in ["format_router", "query_reader", "tool_tag_reader", "all_heads", "all_mlps"] if g in set(gap_bin_df["group"].unique())]
    gap_bin_rows: List[Dict[str, object]] = []
    for b in ["gap_Q1", "gap_Q2", "gap_Q3", "gap_Q4"]:
        for g in key_groups:
            dfg = gap_bin_df[(gap_bin_df["gap_bin"] == b) & (gap_bin_df["group"] == g)]
            gap_bin_rows.append(
                {
                    "gap_bin": b,
                    "group": g,
                    "n_samples": int(dfg["q_index"].nunique()),
                    "suff_median": med(dfg["suff_ratio"]),
                    "nec_median": med(dfg["nec_ratio"]),
                    "drop_full_suff_median": med(dfg["delta_full_suff_drop"]),
                    "drop_full_nec_median": med(dfg["delta_full_nec_drop"]),
                }
            )
    gap_bin_out = pd.DataFrame(gap_bin_rows)
    gap_bin_csv = out_root / "robustness_gap_bins.csv"
    gap_bin_out.to_csv(gap_bin_csv, index=False)

    # Figures.
    save_slice_heatmap(slice_df=slice_summary_df, out_path=out_root / "robustness_slice_heatmap.png")
    save_gap_bin_plot(gap_bin_df=gap_bin_out, groups=key_groups, out_path=out_root / "robustness_gap_bin_groups.png")

    report = {
        "n_total_samples": int(full_df["q_index"].nunique()),
        "slices": {k: [int(x) for x in v] for k, v in slices.items()},
        "groups": groups,
        "edge_summary_by_slice": edge_summary,
        "artifacts": {
            "slice_summary_csv": str(slice_summary_csv),
            "slice_groups_csv": str(slice_group_csv),
            "gap_bin_csv": str(gap_bin_csv),
            "slice_heatmap_png": str(out_root / "robustness_slice_heatmap.png"),
            "gap_bin_groups_png": str(out_root / "robustness_gap_bin_groups.png"),
        },
    }
    report_path = out_root / "robustness_dashboard_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[done] output root: {out_root}")
    print(f"[done] report: {report_path}")


if __name__ == "__main__":
    main()

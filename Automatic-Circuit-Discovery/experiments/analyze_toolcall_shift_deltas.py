#!/usr/bin/env python3
"""
Paired bootstrap analysis for shift-robustness per-sample outputs.

Given `shift_robustness_per_sample.csv`, this script computes:
1) mode-contrast paired deltas with bootstrap CI;
2) common-intersection summaries across selected modes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


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


def parse_contrast(spec: str) -> Tuple[str, str]:
    if "-" not in spec:
        raise ValueError(f"Bad contrast spec: {spec}")
    left, right = spec.split("-", 1)
    left = left.strip()
    right = right.strip()
    if not left or not right:
        raise ValueError(f"Bad contrast spec: {spec}")
    return left, right


def paired_deltas(
    df: pd.DataFrame,
    mode_a: str,
    mode_b: str,
    group: str,
    metric: str,
) -> np.ndarray:
    sub = df[(df["group"] == group) & (df["mode"].isin([mode_a, mode_b]))][["mode", "q_index", metric]]
    if sub.empty:
        return np.array([], dtype=np.float64)
    pivot = sub.pivot_table(index="q_index", columns="mode", values=metric, aggfunc="first")
    if mode_a not in pivot.columns or mode_b not in pivot.columns:
        return np.array([], dtype=np.float64)
    pivot = pivot.dropna(subset=[mode_a, mode_b])
    if pivot.empty:
        return np.array([], dtype=np.float64)
    return (pivot[mode_a].to_numpy(dtype=np.float64) - pivot[mode_b].to_numpy(dtype=np.float64))


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired bootstrap deltas for shift robustness outputs.")
    parser.add_argument(
        "--per-sample-csv",
        type=str,
        default="experiments/results/toolcall_q1_q164_shift_robustness_v4/shift_robustness_per_sample.csv",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="experiments/results/toolcall_q1_q164_shift_robustness_v4",
    )
    parser.add_argument(
        "--contrasts",
        type=str,
        default="user_json_pad-orig,system_json_pad-orig,user_json_pad-system_json_pad,user_pad_short-orig",
    )
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    per_csv = Path(args.per_sample_csv).resolve()
    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(per_csv)
    for col in [
        "q_index",
        "gap_aug",
        "suff_ratio",
        "nec_ratio",
        "drop_full_suff",
        "drop_full_nec",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["drop_full_nec_abs"] = df["drop_full_nec"] * df["gap_aug"]

    contrast_specs = [c.strip() for c in args.contrasts.split(",") if c.strip()]
    groups = [
        "full_core",
        "all_heads",
        "all_mlps",
        "stable_necessary_backbone",
        "stable_but_weak_or_redundant",
    ]

    rows: List[Dict[str, object]] = []
    for i, c in enumerate(contrast_specs):
        mode_a, mode_b = parse_contrast(c)

        # Full-core behavior metrics.
        for j, metric in enumerate(["suff_ratio", "nec_ratio"]):
            delta = paired_deltas(df, mode_a=mode_a, mode_b=mode_b, group="full_core", metric=metric)
            ci = bootstrap_median_ci(delta.tolist(), n_boot=args.bootstrap, seed=args.seed + 101 * i + 7 * j)
            rows.append(
                {
                    "contrast": f"{mode_a}-{mode_b}",
                    "group": "full_core",
                    "metric": metric,
                    "delta_median_boot_mean": float(ci["mean"]),
                    "ci_lo": float(ci["lo"]),
                    "ci_hi": float(ci["hi"]),
                    "n": int(len(delta)),
                }
            )

        # Group-level necessity reallocations.
        for g in groups[1:]:
            for j, metric in enumerate(["drop_full_nec", "drop_full_nec_abs"]):
                delta = paired_deltas(df, mode_a=mode_a, mode_b=mode_b, group=g, metric=metric)
                ci = bootstrap_median_ci(delta.tolist(), n_boot=args.bootstrap, seed=args.seed + 1009 * i + 37 * j + len(g))
                rows.append(
                    {
                        "contrast": f"{mode_a}-{mode_b}",
                        "group": g,
                        "metric": metric,
                        "delta_median_boot_mean": float(ci["mean"]),
                        "ci_lo": float(ci["lo"]),
                        "ci_hi": float(ci["hi"]),
                        "n": int(len(delta)),
                    }
                )

    delta_csv = out_root / "shift_robustness_mode_delta_bootstrap.csv"
    with delta_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "contrast",
                "group",
                "metric",
                "delta_median_boot_mean",
                "ci_lo",
                "ci_hi",
                "n",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # Common-intersection summary across all modes mentioned in contrasts.
    contrast_modes = sorted(
        {
            m
            for c in contrast_specs
            for m in parse_contrast(c)
        }
    )
    full_core_mode_q: Dict[str, set] = {}
    for m in contrast_modes:
        q_set = set(df[(df["mode"] == m) & (df["group"] == "full_core")]["q_index"].dropna().astype(int).tolist())
        full_core_mode_q[m] = q_set
    common_q = set.intersection(*full_core_mode_q.values()) if full_core_mode_q else set()

    common_rows: List[Dict[str, object]] = []
    for m in contrast_modes:
        md = df[(df["mode"] == m) & (df["q_index"].isin(common_q))]
        for g in groups:
            gd = md[md["group"] == g]
            if gd.empty:
                continue
            common_rows.append(
                {
                    "mode": m,
                    "group": g,
                    "n_common_samples": int(gd["q_index"].nunique()),
                    "suff_median": med(gd["suff_ratio"].tolist()),
                    "nec_median": med(gd["nec_ratio"].tolist()),
                    "drop_full_nec_median": med(gd["drop_full_nec"].tolist()),
                    "drop_full_nec_abs_median": med(gd["drop_full_nec_abs"].tolist()),
                }
            )

    common_csv = out_root / "shift_robustness_common_intersection_summary.csv"
    with common_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "mode",
                "group",
                "n_common_samples",
                "suff_median",
                "nec_median",
                "drop_full_nec_median",
                "drop_full_nec_abs_median",
            ],
        )
        writer.writeheader()
        for r in common_rows:
            writer.writerow(r)

    report = {
        "contrasts": contrast_specs,
        "contrast_modes": contrast_modes,
        "n_common_q_indices": len(common_q),
        "common_q_indices": sorted(int(x) for x in common_q),
        "bootstrap": args.bootstrap,
        "seed": args.seed,
        "artifacts": {
            "mode_delta_bootstrap_csv": str(delta_csv),
            "common_intersection_summary_csv": str(common_csv),
        },
    }
    report_path = out_root / "shift_robustness_delta_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[done] output root: {out_root}")
    print(f"[done] report: {report_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Paired bootstrap comparison: static grouping vs condition-aware grouping.

Compares `stable_necessary_backbone` on matched q-index samples.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


def bootstrap_median_ci(values: Sequence[float], n_boot: int, seed: int) -> Dict[str, float]:
    vals = [float(v) for v in values if math.isfinite(float(v))]
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


def paired_deltas(
    base_df: pd.DataFrame,
    alt_df: pd.DataFrame,
    mode: str,
    metric: str,
) -> np.ndarray:
    b = base_df[(base_df["mode"] == mode) & (base_df["group"] == "stable_necessary_backbone")][["q_index", metric]]
    a = alt_df[(alt_df["mode"] == mode) & (alt_df["group"] == "stable_necessary_backbone")][["q_index", metric]]
    merged = a.merge(b, on="q_index", how="inner", suffixes=("_alt", "_base"))
    merged = merged.dropna(subset=[f"{metric}_alt", f"{metric}_base"])
    if merged.empty:
        return np.array([], dtype=np.float64)
    return merged[f"{metric}_alt"].to_numpy(dtype=np.float64) - merged[f"{metric}_base"].to_numpy(dtype=np.float64)


def load_per_sample(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in ["q_index", "drop_full_nec", "drop_full_suff", "suff_ratio", "nec_ratio"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired bootstrap: static vs condition-aware grouping.")
    parser.add_argument(
        "--static-per-sample",
        type=str,
        default="experiments/results/toolcall_q1_q164_shift_robustness_v4/shift_robustness_per_sample.csv",
    )
    parser.add_argument(
        "--useraware-per-sample",
        type=str,
        default="experiments/results/toolcall_q1_q164_shift_robustness_useraware_v1/shift_robustness_per_sample.csv",
    )
    parser.add_argument(
        "--systemaware-per-sample",
        type=str,
        default="experiments/results/toolcall_q1_q164_shift_robustness_systemaware_v1/shift_robustness_per_sample.csv",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="experiments/results/toolcall_q1_q164_condition_aware_grouping_v1",
    )
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    static_df = load_per_sample(Path(args.static_per_sample).resolve())
    user_df = load_per_sample(Path(args.useraware_per_sample).resolve())
    system_df = load_per_sample(Path(args.systemaware_per_sample).resolve())

    eval_specs: List[Tuple[str, pd.DataFrame, str]] = [
        ("useraware_vs_static@user_json_pad", user_df, "user_json_pad"),
        ("useraware_vs_static@orig", user_df, "orig"),
        ("systemaware_vs_static@system_json_pad", system_df, "system_json_pad"),
        ("systemaware_vs_static@orig", system_df, "orig"),
    ]

    rows: List[Dict[str, object]] = []
    for i, (name, alt_df, mode) in enumerate(eval_specs):
        for j, metric in enumerate(["drop_full_nec", "drop_full_suff"]):
            d = paired_deltas(static_df, alt_df, mode=mode, metric=metric)
            ci = bootstrap_median_ci(d.tolist(), n_boot=args.bootstrap, seed=args.seed + 101 * i + 13 * j)
            rows.append(
                {
                    "comparison": name,
                    "mode": mode,
                    "metric": metric,
                    "delta_median_boot_mean": float(ci["mean"]),
                    "ci_lo": float(ci["lo"]),
                    "ci_hi": float(ci["hi"]),
                    "n": int(len(d)),
                }
            )

    out_csv = out_root / "condition_aware_grouping_delta_bootstrap.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["comparison", "mode", "metric", "delta_median_boot_mean", "ci_lo", "ci_hi", "n"],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    report = {
        "bootstrap": args.bootstrap,
        "seed": args.seed,
        "rows": rows,
        "artifacts": {
            "condition_aware_grouping_delta_bootstrap_csv": str(out_csv),
        },
    }
    report_path = out_root / "condition_aware_grouping_delta_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[done] output root: {out_root}")
    print(f"[done] report: {report_path}")


if __name__ == "__main__":
    main()

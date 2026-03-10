#!/usr/bin/env python3
"""
Compare static grouping vs condition-aware grouping on shift robustness summaries.

Focus metric:
- stable_necessary_backbone drop_full_nec_median
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def load_backbone_rows(summary_csv: Path, grouping_label: str) -> List[Dict[str, object]]:
    df = pd.read_csv(summary_csv)
    df = df[df["group"] == "stable_necessary_backbone"].copy()
    rows: List[Dict[str, object]] = []
    for _, r in df.iterrows():
        rows.append(
            {
                "grouping": grouping_label,
                "mode": str(r["mode"]),
                "n_nodes": int(r["n_nodes"]),
                "n_samples": int(r["n_samples"]),
                "suff_median": float(r["suff_median"]),
                "nec_median": float(r["nec_median"]),
                "drop_full_nec_median": float(r["drop_full_nec_median"]),
                "drop_full_suff_median": float(r["drop_full_suff_median"]),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze condition-aware grouping benefits.")
    parser.add_argument(
        "--static-summary",
        type=str,
        default="experiments/results/toolcall_q1_q164_shift_robustness_v4/shift_robustness_summary.csv",
    )
    parser.add_argument(
        "--useraware-summary",
        type=str,
        default="experiments/results/toolcall_q1_q164_shift_robustness_useraware_v1/shift_robustness_summary.csv",
    )
    parser.add_argument(
        "--systemaware-summary",
        type=str,
        default="experiments/results/toolcall_q1_q164_shift_robustness_systemaware_v1/shift_robustness_summary.csv",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="experiments/results/toolcall_q1_q164_condition_aware_grouping_v1",
    )
    args = parser.parse_args()

    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    rows.extend(load_backbone_rows(Path(args.static_summary).resolve(), "static_orig_strata"))
    rows.extend(load_backbone_rows(Path(args.useraware_summary).resolve(), "useraware_strata"))
    rows.extend(load_backbone_rows(Path(args.systemaware_summary).resolve(), "systemaware_strata"))

    rows = sorted(rows, key=lambda x: (str(x["mode"]), str(x["grouping"])))

    table_csv = out_root / "condition_aware_backbone_comparison.csv"
    with table_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "mode",
                "grouping",
                "n_nodes",
                "n_samples",
                "suff_median",
                "nec_median",
                "drop_full_suff_median",
                "drop_full_nec_median",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # Aggregate key gains/penalties vs static baseline.
    by_mode_grouping: Dict[str, Dict[str, Dict[str, object]]] = {}
    for r in rows:
        mode = str(r["mode"])
        grp = str(r["grouping"])
        by_mode_grouping.setdefault(mode, {})[grp] = r

    def gain(mode: str, grouping: str, metric: str) -> float:
        base = by_mode_grouping.get(mode, {}).get("static_orig_strata")
        alt = by_mode_grouping.get(mode, {}).get(grouping)
        if base is None or alt is None:
            return float("nan")
        return float(alt[metric]) - float(base[metric])

    summary = {
        "target_userjson_gain_drop_full_nec": gain("user_json_pad", "useraware_strata", "drop_full_nec_median"),
        "target_systemjson_gain_drop_full_nec": gain("system_json_pad", "systemaware_strata", "drop_full_nec_median"),
        "offtarget_orig_penalty_useraware_drop_full_nec": gain("orig", "useraware_strata", "drop_full_nec_median"),
        "offtarget_orig_penalty_systemaware_drop_full_nec": gain("orig", "systemaware_strata", "drop_full_nec_median"),
    }

    report = {
        "rows": rows,
        "summary": summary,
        "artifacts": {
            "condition_aware_backbone_comparison_csv": str(table_csv),
        },
    }
    report_path = out_root / "condition_aware_grouping_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[done] output root: {out_root}")
    print(f"[done] report: {report_path}")


if __name__ == "__main__":
    main()

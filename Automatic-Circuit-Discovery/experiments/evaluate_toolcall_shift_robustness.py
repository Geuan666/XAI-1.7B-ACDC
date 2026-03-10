#!/usr/bin/env python3
"""
Robustness under prompt distribution shift via neutral context injection.

For each clean/corrupt pair, we inject identical neutral text to both sides
to create controlled distribution shift (position/content perturbation) while
preserving the clean-vs-corrupt contrast.

We then evaluate selected node groups:
- full_core
- stable_necessary_backbone
- stable_but_weak_or_redundant
- all_heads
- all_mlps

Outputs:
- per-mode per-sample CSV
- summary CSV/JSON with median suff/nec and full-minus-group drops
- compact figures for mode-wise comparison
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

from experiments.launch_toolcall_qwen3_q85 import (
    evaluate_on_base_with_source,
    load_hooked_qwen3,
    objective_from_logits,
)


def finite(xs: Iterable[float]) -> List[float]:
    out: List[float] = []
    for x in xs:
        if isinstance(x, (int, float)) and math.isfinite(float(x)):
            out.append(float(x))
    return out


def med(xs: Iterable[float]) -> float:
    vals = finite(xs)
    return float(median(vals)) if vals else float("nan")


def parse_layer(node: str) -> int:
    if node.startswith("MLP"):
        return int(node[3:])
    if node.startswith("L") and "H" in node:
        return int(node[1:].split("H", 1)[0])
    raise ValueError(f"Unknown node name: {node}")


def collect_node_cache_cpu(model, tokens: torch.Tensor, nodes: Sequence[str]) -> Dict[str, torch.Tensor]:
    names = set()
    for node in nodes:
        if node.startswith("MLP"):
            names.add(f"blocks.{int(node[3:])}.hook_mlp_out")
        else:
            layer = parse_layer(node)
            names.add(f"blocks.{layer}.attn.hook_z")
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=lambda n: n in names)
    return {k: v.detach().cpu() for k, v in cache.items()}


def inject_after_last(text: str, anchor: str, payload: str) -> str:
    idx = text.rfind(anchor)
    if idx < 0:
        return text
    pos = idx + len(anchor)
    return text[:pos] + payload + text[pos:]


def augment_prompt(text: str, mode: str) -> str:
    payload_short = (
        "Context note: This metadata is not part of the target function logic.\n"
        "Please ignore this note when deciding actual behavior.\n"
    )
    payload_long = payload_short * 4
    payload_newline = ("\n#\n" * 24)
    payload_json = ("{\"meta\":\"context\"}\n" * 12)

    if mode == "orig":
        return text
    if mode == "user_pad_short":
        return inject_after_last(text, "<|im_start|>user\n", payload_short)
    if mode == "user_pad_long":
        return inject_after_last(text, "<|im_start|>user\n", payload_long)
    if mode == "user_newline_pad":
        return inject_after_last(text, "<|im_start|>user\n", payload_newline)
    if mode == "user_json_pad":
        return inject_after_last(text, "<|im_start|>user\n", payload_json)
    if mode == "system_json_pad":
        return inject_after_last(text, "<|im_start|>system\n", payload_json)
    if mode == "system_pad_long":
        return inject_after_last(text, "<|im_start|>system\n", payload_long)
    raise ValueError(f"Unknown mode: {mode}")


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


def save_mode_group_heatmap(df, value_col: str, title: str, out_path: Path) -> None:
    if df.empty:
        return
    pivot = df.pivot(index="group", columns="mode", values=value_col)
    row_order = sorted(
        pivot.index.tolist(),
        key=lambda g: (0 if g == "full_core" else 1, g),
    )
    col_order = sorted(
        pivot.columns.tolist(),
        key=lambda m: (0 if m == "orig" else 1, m),
    )
    matrix = pivot.loc[row_order, col_order].to_numpy(dtype=np.float64)

    apply_plot_style()
    fig, ax = plt.subplots(figsize=(1.2 * max(4, len(col_order)) + 2.0, 0.55 * max(5, len(row_order)) + 2.0))
    vals = matrix[np.isfinite(matrix)]
    vmax = float(np.percentile(np.abs(vals), 98.0)) if vals.size > 0 else 1.0
    vmax = max(vmax, 1e-6)
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(col_order)))
    ax.set_xticklabels(col_order, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(row_order)))
    ax.set_yticklabels(row_order)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cbar.set_label(value_col)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate robustness under neutral prompt shift.")
    parser.add_argument("--input-root", type=str, default="experiments/results/toolcall_q1_q164")
    parser.add_argument(
        "--semantic-report",
        type=str,
        default="experiments/results/toolcall_q1_q164_semantic_roles_v3/semantic_roles_report.json",
    )
    parser.add_argument(
        "--stratification-report",
        type=str,
        default="experiments/results/toolcall_q1_q164_node_stratification_v1/node_stratification_report.json",
    )
    parser.add_argument("--output-root", type=str, default="experiments/results/toolcall_q1_q164_shift_robustness_v1")
    parser.add_argument("--model-path", type=str, default="/root/data/Qwen/Qwen3-1.7B")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--modes", type=str, default="orig,user_pad_short,user_pad_long,system_pad_long")
    parser.add_argument("--gap-min-select", type=float, default=0.5, help="Select samples by original summary gap.")
    parser.add_argument("--gap-min-aug", type=float, default=0.5, help="Evaluate only when augmented gap exceeds this.")
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all selected samples.")
    parser.add_argument("--q-start", type=int, default=1)
    parser.add_argument("--q-end", type=int, default=10000)
    args = parser.parse_args()

    input_root = Path(args.input_root).resolve()
    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    sem = json.loads(Path(args.semantic_report).resolve().read_text(encoding="utf-8"))
    core_nodes = [str(n) for n in sem.get("core_nodes", [])]
    if not core_nodes:
        raise ValueError("No core_nodes in semantic report.")
    core_nodes = sorted(core_nodes, key=lambda n: (parse_layer(n), 0 if n.startswith("MLP") else 1, n))

    strat = json.loads(Path(args.stratification_report).resolve().read_text(encoding="utf-8"))
    strata = dict(strat.get("strata", {}))
    stable_backbone = [str(n) for n in strata.get("stable_necessary_backbone", []) if n in set(core_nodes)]
    stable_weak = [str(n) for n in strata.get("stable_but_weak_or_redundant", []) if n in set(core_nodes)]
    all_heads = [n for n in core_nodes if n.startswith("L")]
    all_mlps = [n for n in core_nodes if n.startswith("MLP")]

    groups: Dict[str, List[str]] = {
        "full_core": core_nodes,
        "stable_necessary_backbone": stable_backbone,
        "stable_but_weak_or_redundant": stable_weak,
        "all_heads": all_heads,
        "all_mlps": all_mlps,
    }
    groups = {k: v for k, v in groups.items() if v}

    mode_list = [x.strip() for x in args.modes.split(",") if x.strip()]
    if not mode_list:
        raise ValueError("No modes specified.")

    # Select samples from original summaries.
    q_dirs = sorted(input_root.glob("q[0-9][0-9][0-9]"))
    sample_infos: List[Tuple[int, Dict[str, object], Path, Path]] = []
    for q_dir in q_dirs:
        sp = q_dir / "summary.json"
        if not sp.exists():
            continue
        s = json.loads(sp.read_text(encoding="utf-8"))
        q_index = int(s.get("q_index", -1))
        if q_index < args.q_start or q_index > args.q_end:
            continue
        gap = float(s.get("gap", float("nan")))
        if not math.isfinite(gap) or gap <= args.gap_min_select:
            continue
        cp = Path(s["clean_prompt"])
        rp = Path(s["corrupt_prompt"])
        if not cp.exists() or not rp.exists():
            continue
        sample_infos.append((q_index, s, cp, rp))
    if args.max_samples > 0:
        sample_infos = sample_infos[: args.max_samples]
    if not sample_infos:
        raise ValueError("No samples selected.")

    model, tokenizer = load_hooked_qwen3(args.model_path, device=args.device, dtype=torch.bfloat16)
    model.set_use_attn_result(False)
    model.set_use_split_qkv_input(False)
    model.set_use_hook_mlp_in(False)
    target_token = tokenizer("<tool_call>", add_special_tokens=False)["input_ids"][0]

    per_rows: List[Dict[str, object]] = []
    skipped_by_mode: Dict[str, List[int]] = defaultdict(list)
    analyzed_by_mode: Dict[str, List[int]] = defaultdict(list)

    for mode in mode_list:
        pbar = tqdm(sample_infos, desc=f"Shift eval [{mode}]", dynamic_ncols=True)
        for q_index, summary, clean_path, corrupt_path in pbar:
            clean_text = augment_prompt(clean_path.read_text(encoding="utf-8"), mode=mode)
            corrupt_text = augment_prompt(corrupt_path.read_text(encoding="utf-8"), mode=mode)
            clean_tokens = model.to_tokens(clean_text, prepend_bos=False)
            corrupt_tokens = model.to_tokens(corrupt_text, prepend_bos=False)
            if clean_tokens.shape != corrupt_tokens.shape:
                skipped_by_mode[mode].append(q_index)
                continue

            try:
                with torch.no_grad():
                    clean_logits = model(clean_tokens)
                    corrupt_logits = model(corrupt_tokens)
            except torch.OutOfMemoryError:
                skipped_by_mode[mode].append(q_index)
                model.reset_hooks()
                gc.collect()
                torch.cuda.empty_cache()
                continue

            distractor = int(torch.argmax(corrupt_logits[0, -1]).item())
            if distractor == target_token:
                top2 = torch.topk(corrupt_logits[0, -1], k=2).indices.tolist()
                distractor = int(top2[1]) if len(top2) > 1 else distractor

            clean_obj = float(objective_from_logits(clean_logits, target_token, distractor).item())
            corrupt_obj = float(objective_from_logits(corrupt_logits, target_token, distractor).item())
            gap_aug = clean_obj - corrupt_obj
            if not math.isfinite(gap_aug) or gap_aug <= args.gap_min_aug:
                skipped_by_mode[mode].append(q_index)
                continue

            try:
                clean_cache = collect_node_cache_cpu(model, clean_tokens, core_nodes)
                corrupt_cache = collect_node_cache_cpu(model, corrupt_tokens, core_nodes)
            except torch.OutOfMemoryError:
                skipped_by_mode[mode].append(q_index)
                model.reset_hooks()
                gc.collect()
                torch.cuda.empty_cache()
                continue

            def suff_ratio(nodes_to_patch: Sequence[str]) -> float:
                obj = evaluate_on_base_with_source(
                    model=model,
                    base_tokens=corrupt_tokens,
                    source_cache_cpu=clean_cache,
                    patch_nodes=nodes_to_patch,
                    target_token=target_token,
                    distractor_token=distractor,
                )
                return (obj - corrupt_obj) / gap_aug

            def nec_ratio(nodes_to_patch: Sequence[str]) -> float:
                obj = evaluate_on_base_with_source(
                    model=model,
                    base_tokens=clean_tokens,
                    source_cache_cpu=corrupt_cache,
                    patch_nodes=nodes_to_patch,
                    target_token=target_token,
                    distractor_token=distractor,
                )
                return (clean_obj - obj) / gap_aug

            try:
                full_suff = suff_ratio(groups["full_core"])
                full_nec = nec_ratio(groups["full_core"])
            except torch.OutOfMemoryError:
                skipped_by_mode[mode].append(q_index)
                model.reset_hooks()
                gc.collect()
                torch.cuda.empty_cache()
                continue

            for g, members in groups.items():
                try:
                    g_suff = suff_ratio(members)
                    g_nec = nec_ratio(members)
                except torch.OutOfMemoryError:
                    continue
                minus_nodes = [n for n in core_nodes if n not in set(members)]
                if minus_nodes:
                    try:
                        minus_suff = suff_ratio(minus_nodes)
                        minus_nec = nec_ratio(minus_nodes)
                    except torch.OutOfMemoryError:
                        minus_suff = float("nan")
                        minus_nec = float("nan")
                else:
                    minus_suff = float("nan")
                    minus_nec = float("nan")
                per_rows.append(
                    {
                        "mode": mode,
                        "q_index": q_index,
                        "group": g,
                        "n_nodes": len(members),
                        "gap_aug": gap_aug,
                        "full_suff_ratio": full_suff,
                        "full_nec_ratio": full_nec,
                        "suff_ratio": g_suff,
                        "nec_ratio": g_nec,
                        "suff_minus_ratio": minus_suff,
                        "nec_minus_ratio": minus_nec,
                        "drop_full_suff": full_suff - minus_suff if math.isfinite(minus_suff) else float("nan"),
                        "drop_full_nec": full_nec - minus_nec if math.isfinite(minus_nec) else float("nan"),
                    }
                )

            analyzed_by_mode[mode].append(q_index)
            model.reset_hooks()
            del clean_cache
            del corrupt_cache
            gc.collect()
            torch.cuda.empty_cache()

    per_csv = out_root / "shift_robustness_per_sample.csv"
    with per_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "mode",
                "q_index",
                "group",
                "n_nodes",
                "gap_aug",
                "full_suff_ratio",
                "full_nec_ratio",
                "suff_ratio",
                "nec_ratio",
                "suff_minus_ratio",
                "nec_minus_ratio",
                "drop_full_suff",
                "drop_full_nec",
            ],
        )
        w.writeheader()
        for r in per_rows:
            w.writerow(r)

    # Summary.
    by_mode_group: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for r in per_rows:
        by_mode_group[(str(r["mode"]), str(r["group"]))].append(r)

    summary_rows: List[Dict[str, object]] = []
    for mode in sorted(mode_list, key=lambda m: (0 if m == "orig" else 1, m)):
        for g in sorted(groups.keys(), key=lambda x: (0 if x == "full_core" else 1, x)):
            grp = by_mode_group.get((mode, g), [])
            if not grp:
                continue
            suff_vals = [float(x["suff_ratio"]) for x in grp]
            nec_vals = [float(x["nec_ratio"]) for x in grp]
            drop_s = [float(x["drop_full_suff"]) for x in grp]
            drop_n = [float(x["drop_full_nec"]) for x in grp]
            full_s = [float(x["full_suff_ratio"]) for x in grp]
            full_n = [float(x["full_nec_ratio"]) for x in grp]
            summary_rows.append(
                {
                    "mode": mode,
                    "group": g,
                    "n_samples": len(grp),
                    "n_nodes": int(grp[0]["n_nodes"]),
                    "gap_aug_median": med([float(x["gap_aug"]) for x in grp]),
                    "full_suff_median": med(full_s),
                    "full_nec_median": med(full_n),
                    "suff_median": med(suff_vals),
                    "nec_median": med(nec_vals),
                    "drop_full_suff_median": med(drop_s),
                    "drop_full_nec_median": med(drop_n),
                }
            )

    summary_csv = out_root / "shift_robustness_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "mode",
                "group",
                "n_samples",
                "n_nodes",
                "gap_aug_median",
                "full_suff_median",
                "full_nec_median",
                "suff_median",
                "nec_median",
                "drop_full_suff_median",
                "drop_full_nec_median",
            ]
        )
        for r in summary_rows:
            w.writerow(
                [
                    r["mode"],
                    r["group"],
                    r["n_samples"],
                    r["n_nodes"],
                    r["gap_aug_median"],
                    r["full_suff_median"],
                    r["full_nec_median"],
                    r["suff_median"],
                    r["nec_median"],
                    r["drop_full_suff_median"],
                    r["drop_full_nec_median"],
                ]
            )

    import pandas as pd  # local use only for pivot/plot

    sdf = pd.DataFrame(summary_rows)
    save_mode_group_heatmap(
        df=sdf,
        value_col="suff_median",
        title="Shift Robustness: Sufficiency by Group and Mode",
        out_path=out_root / "shift_robustness_suff_heatmap.png",
    )
    save_mode_group_heatmap(
        df=sdf,
        value_col="nec_median",
        title="Shift Robustness: Necessity by Group and Mode",
        out_path=out_root / "shift_robustness_nec_heatmap.png",
    )
    save_mode_group_heatmap(
        df=sdf,
        value_col="drop_full_nec_median",
        title="Shift Robustness: Drop-Full-Nec by Group and Mode",
        out_path=out_root / "shift_robustness_drop_nec_heatmap.png",
    )

    report = {
        "modes": mode_list,
        "groups": groups,
        "n_selected_samples": len(sample_infos),
        "n_analyzed_by_mode": {k: len(set(v)) for k, v in analyzed_by_mode.items()},
        "skipped_q_indices_by_mode": {k: sorted(set(v)) for k, v in skipped_by_mode.items()},
        "gap_min_select": args.gap_min_select,
        "gap_min_aug": args.gap_min_aug,
        "summary_rows": summary_rows,
        "artifacts": {
            "per_sample_csv": str(per_csv),
            "summary_csv": str(summary_csv),
            "suff_heatmap_png": str(out_root / "shift_robustness_suff_heatmap.png"),
            "nec_heatmap_png": str(out_root / "shift_robustness_nec_heatmap.png"),
            "drop_nec_heatmap_png": str(out_root / "shift_robustness_drop_nec_heatmap.png"),
        },
    }
    report_path = out_root / "shift_robustness_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[done] output root: {out_root}")
    print(f"[done] report: {report_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Node-level causal ablation analysis for the tool-call consensus circuit.

For each valid clean/corrupt pair:
1) evaluate full consensus core (suff/nec),
2) evaluate each node alone (suff/nec),
3) evaluate full-minus-node (suff/nec),
4) quantify each node's indispensability inside the full circuit.

This script is intended to stress-test role grouping claims:
- which nodes are truly necessary,
- which nodes are mainly supportive/redundant,
- whether some nodes appear counterproductive in full-circuit context.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Sequence

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


def parse_layer(node: str) -> int:
    if node.startswith("MLP"):
        return int(node[3:])
    m = re.fullmatch(r"L(\d+)H(\d+)", node)
    if m is None:
        raise ValueError(f"Unknown node name: {node}")
    return int(m.group(1))


def finite(xs: Iterable[float]) -> List[float]:
    out: List[float] = []
    for x in xs:
        if isinstance(x, (int, float)) and math.isfinite(float(x)):
            out.append(float(x))
    return out


def med(xs: Iterable[float]) -> float:
    vals = finite(xs)
    return float(median(vals)) if vals else float("nan")


def mean(xs: Iterable[float]) -> float:
    vals = finite(xs)
    return float(np.mean(vals)) if vals else float("nan")


def bootstrap_ci(values: Sequence[float], n_boot: int, seed: int) -> Dict[str, float]:
    vals = finite(values)
    if not vals:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    rng = random.Random(seed)
    n = len(vals)
    boot: List[float] = []
    for _ in range(n_boot):
        sample = [vals[rng.randrange(n)] for __ in range(n)]
        boot.append(float(np.median(sample)))
    boot.sort()
    lo_idx = max(0, int(0.025 * n_boot))
    hi_idx = min(n_boot - 1, int(0.975 * n_boot))
    return {
        "mean": float(np.mean(boot)),
        "lo": float(boot[lo_idx]),
        "hi": float(boot[hi_idx]),
    }


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
    payload_json = ("{\"meta\":\"context\"}\n" * 12)
    if mode == "orig":
        return text
    if mode == "user_pad_short":
        return inject_after_last(text, "<|im_start|>user\n", payload_short)
    if mode == "user_pad_long":
        return inject_after_last(text, "<|im_start|>user\n", payload_long)
    if mode == "user_json_pad":
        return inject_after_last(text, "<|im_start|>user\n", payload_json)
    if mode == "system_json_pad":
        return inject_after_last(text, "<|im_start|>system\n", payload_json)
    if mode == "system_pad_long":
        return inject_after_last(text, "<|im_start|>system\n", payload_long)
    raise ValueError(f"Unknown augment mode: {mode}")


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


def save_heatmap(
    matrix: np.ndarray,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    title: str,
    out_path: Path,
    percentile_clip: float = 98.0,
) -> None:
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(1.2 * max(6, len(col_labels)), 0.6 * max(5, len(row_labels)) + 2.0))
    vals = matrix[np.isfinite(matrix)]
    if vals.size == 0:
        vmax = 1.0
    else:
        vmax = float(np.percentile(np.abs(vals), percentile_clip))
        vmax = max(vmax, 1e-6)
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=28, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title(f"{title}\nSymmetric clipping at ±{vmax:.3f}")
    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cbar.set_label("Median ratio (center=0)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_bar(
    labels: Sequence[str],
    values: Sequence[float],
    cis: Sequence[Dict[str, float]],
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    apply_plot_style()
    x = np.arange(len(labels))
    y = np.array(values, dtype=np.float64)
    lo = np.array([float(c.get("lo", np.nan)) for c in cis], dtype=np.float64)
    hi = np.array([float(c.get("hi", np.nan)) for c in cis], dtype=np.float64)
    yerr = np.vstack([np.maximum(0.0, y - lo), np.maximum(0.0, hi - y)])

    fig, ax = plt.subplots(figsize=(max(9.5, 0.7 * len(labels)), 5.0), constrained_layout=True)
    ax.bar(x, y, color="#2a6f97", edgecolor="#1f1f1f", linewidth=0.8)
    ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor="#1f1f1f", elinewidth=1.1, capsize=3)
    ax.axhline(0.0, color="#444444", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def derive_grouping(summary_rows: Sequence[Dict[str, object]]) -> Dict[str, List[str]]:
    rows = [r for r in summary_rows if isinstance(r.get("node"), str)]
    if not rows:
        return {}
    drop_vals = np.array(
        [float(r.get("drop_full_nec_median", float("nan"))) for r in rows if math.isfinite(float(r.get("drop_full_nec_median", float("nan"))))],
        dtype=np.float64,
    )
    alone_vals = np.array(
        [float(r.get("node_suff_median", float("nan"))) for r in rows if math.isfinite(float(r.get("node_suff_median", float("nan"))))],
        dtype=np.float64,
    )
    drop_hi = float(np.percentile(drop_vals, 70)) if drop_vals.size > 0 else 0.10
    drop_lo = float(np.percentile(drop_vals, 30)) if drop_vals.size > 0 else 0.02
    alone_hi = float(np.percentile(alone_vals, 70)) if alone_vals.size > 0 else 0.30

    groups: Dict[str, List[str]] = defaultdict(list)
    for r in rows:
        n = str(r["node"])
        drop_n = float(r.get("drop_full_nec_median", float("nan")))
        alone_s = float(r.get("node_suff_median", float("nan")))
        if math.isfinite(drop_n) and drop_n >= drop_hi:
            groups["essential_backbone"].append(n)
        elif math.isfinite(drop_n) and drop_n <= min(0.0, drop_lo):
            if math.isfinite(alone_s) and alone_s >= alone_hi:
                groups["parallel_writer_redundant_in_full"].append(n)
            else:
                groups["redundant_or_interfering"].append(n)
        elif math.isfinite(alone_s) and alone_s >= alone_hi:
            groups["standalone_supporter"].append(n)
        else:
            groups["contextual_supporter"].append(n)

    for g in list(groups.keys()):
        groups[g] = sorted(groups[g], key=lambda n: (parse_layer(n), 0 if n.startswith("MLP") else 1, n))
    return dict(groups)


def main() -> None:
    parser = argparse.ArgumentParser(description="Node-level causal ablation for tool-call circuit.")
    parser.add_argument("--input-root", type=str, default="experiments/results/toolcall_q1_q164")
    parser.add_argument(
        "--semantic-report",
        type=str,
        default="experiments/results/toolcall_q1_q164_semantic_roles_v3/semantic_roles_report.json",
    )
    parser.add_argument("--model-path", type=str, default="/root/data/Qwen/Qwen3-1.7B")
    parser.add_argument("--output-root", type=str, default="experiments/results/toolcall_q1_q164_node_ablation")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gap-min", type=float, default=0.5)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all valid samples.")
    parser.add_argument("--q-start", type=int, default=1)
    parser.add_argument("--q-end", type=int, default=10_000)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--augment-mode", type=str, default="orig")
    parser.add_argument("--gap-min-aug", type=float, default=0.5)
    args = parser.parse_args()

    input_root = Path(args.input_root).resolve()
    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    sem_report = json.loads(Path(args.semantic_report).resolve().read_text(encoding="utf-8"))
    core_nodes = list(sem_report.get("core_nodes", []))
    node_roles = {
        str(k): str(v.get("role", ""))
        for k, v in dict(sem_report.get("node_summary", {})).items()
        if isinstance(v, dict)
    }
    if not core_nodes:
        raise ValueError("No core_nodes found in semantic report.")
    core_nodes = sorted(core_nodes, key=lambda n: (parse_layer(n), 0 if n.startswith("MLP") else 1, n))
    core_set = set(core_nodes)

    model, tokenizer = load_hooked_qwen3(args.model_path, device=args.device, dtype=torch.bfloat16)
    model.set_use_attn_result(False)
    model.set_use_split_qkv_input(False)
    model.set_use_hook_mlp_in(False)
    target_token = tokenizer("<tool_call>", add_special_tokens=False)["input_ids"][0]

    q_dirs = sorted(input_root.glob("q[0-9][0-9][0-9]"))
    sample_infos: List[tuple[int, Dict[str, object], Path, Path]] = []
    for q_dir in q_dirs:
        sp = q_dir / "summary.json"
        if not sp.exists():
            continue
        s = json.loads(sp.read_text(encoding="utf-8"))
        q_index = int(s.get("q_index", -1))
        if q_index < args.q_start or q_index > args.q_end:
            continue
        gap = float(s.get("gap", float("nan")))
        if not math.isfinite(gap) or gap <= args.gap_min:
            continue
        cp = Path(s["clean_prompt"])
        rp = Path(s["corrupt_prompt"])
        if not cp.exists() or not rp.exists():
            continue
        sample_infos.append((q_index, s, cp, rp))
    if args.max_samples > 0:
        sample_infos = sample_infos[: args.max_samples]
    if not sample_infos:
        raise ValueError("No valid samples selected.")

    rows: List[Dict[str, object]] = []
    analyzed: List[int] = []
    skipped: List[int] = []

    pbar = tqdm(sample_infos, desc="Node ablation", dynamic_ncols=True)
    for q_index, summary, clean_prompt, corrupt_prompt in pbar:
        clean_text = augment_prompt(clean_prompt.read_text(encoding="utf-8"), mode=args.augment_mode)
        corrupt_text = augment_prompt(corrupt_prompt.read_text(encoding="utf-8"), mode=args.augment_mode)
        clean_tokens = model.to_tokens(clean_text, prepend_bos=False)
        corrupt_tokens = model.to_tokens(corrupt_text, prepend_bos=False)
        if clean_tokens.shape != corrupt_tokens.shape:
            skipped.append(q_index)
            continue

        try:
            with torch.no_grad():
                clean_logits = model(clean_tokens)
                corrupt_logits = model(corrupt_tokens)
        except torch.OutOfMemoryError:
            skipped.append(q_index)
            model.reset_hooks()
            gc.collect()
            torch.cuda.empty_cache()
            continue

        distractor = int(summary.get("distractor_token_id", -1))
        if distractor < 0 or distractor >= corrupt_logits.shape[-1]:
            distractor = int(torch.argmax(corrupt_logits[0, -1]).item())
        if distractor == target_token:
            top2 = torch.topk(corrupt_logits[0, -1], k=2).indices.tolist()
            distractor = int(top2[1]) if len(top2) > 1 else int(torch.argmax(corrupt_logits[0, -1]).item())

        clean_obj = float(objective_from_logits(clean_logits, target_token, distractor).item())
        corrupt_obj = float(objective_from_logits(corrupt_logits, target_token, distractor).item())
        gap = clean_obj - corrupt_obj
        if not math.isfinite(gap) or gap <= args.gap_min_aug:
            skipped.append(q_index)
            continue

        try:
            clean_cache = collect_node_cache_cpu(model, clean_tokens, nodes=core_nodes)
            corrupt_cache = collect_node_cache_cpu(model, corrupt_tokens, nodes=core_nodes)
        except torch.OutOfMemoryError:
            skipped.append(q_index)
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
            return (obj - corrupt_obj) / gap

        def nec_ratio(nodes_to_patch: Sequence[str]) -> float:
            obj = evaluate_on_base_with_source(
                model=model,
                base_tokens=clean_tokens,
                source_cache_cpu=corrupt_cache,
                patch_nodes=nodes_to_patch,
                target_token=target_token,
                distractor_token=distractor,
            )
            return (clean_obj - obj) / gap

        try:
            full_suff = suff_ratio(core_nodes)
            full_nec = nec_ratio(core_nodes)
        except torch.OutOfMemoryError:
            skipped.append(q_index)
            model.reset_hooks()
            gc.collect()
            torch.cuda.empty_cache()
            continue

        for node in core_nodes:
            minus_nodes = [n for n in core_nodes if n != node]
            try:
                node_suff = suff_ratio([node])
                node_nec = nec_ratio([node])
                minus_suff = suff_ratio(minus_nodes) if minus_nodes else float("nan")
                minus_nec = nec_ratio(minus_nodes) if minus_nodes else float("nan")
            except torch.OutOfMemoryError:
                continue

            rows.append(
                {
                    "q_index": q_index,
                    "node": node,
                    "node_role": node_roles.get(node, ""),
                    "gap": gap,
                    "full_suff_ratio": full_suff,
                    "full_nec_ratio": full_nec,
                    "node_suff_ratio": node_suff,
                    "node_nec_ratio": node_nec,
                    "minus_node_suff_ratio": minus_suff,
                    "minus_node_nec_ratio": minus_nec,
                    "drop_full_suff": full_suff - minus_suff if math.isfinite(minus_suff) else float("nan"),
                    "drop_full_nec": full_nec - minus_nec if math.isfinite(minus_nec) else float("nan"),
                }
            )

        analyzed.append(q_index)
        model.reset_hooks()
        del clean_cache
        del corrupt_cache
        gc.collect()
        torch.cuda.empty_cache()

    # Per-sample table.
    per_sample_csv = out_root / "node_ablation_per_sample.csv"
    with per_sample_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "q_index",
                "node",
                "node_role",
                "gap",
                "full_suff_ratio",
                "full_nec_ratio",
                "node_suff_ratio",
                "node_nec_ratio",
                "minus_node_suff_ratio",
                "minus_node_nec_ratio",
                "drop_full_suff",
                "drop_full_nec",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Aggregate summary.
    by_node: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for r in rows:
        by_node[str(r["node"])].append(r)

    summary_rows: List[Dict[str, object]] = []
    for i, node in enumerate(core_nodes):
        grp = by_node.get(node, [])
        node_suff = [float(x["node_suff_ratio"]) for x in grp]
        node_nec = [float(x["node_nec_ratio"]) for x in grp]
        minus_suff = [float(x["minus_node_suff_ratio"]) for x in grp]
        minus_nec = [float(x["minus_node_nec_ratio"]) for x in grp]
        drop_s = [float(x["drop_full_suff"]) for x in grp]
        drop_n = [float(x["drop_full_nec"]) for x in grp]
        summary_rows.append(
            {
                "node": node,
                "node_role": node_roles.get(node, ""),
                "n_samples": len(grp),
                "node_suff_median": med(node_suff),
                "node_suff_mean": mean(node_suff),
                "node_suff_ci": bootstrap_ci(node_suff, n_boot=args.bootstrap, seed=args.seed + 11 * (i + 1)),
                "node_nec_median": med(node_nec),
                "node_nec_mean": mean(node_nec),
                "node_nec_ci": bootstrap_ci(node_nec, n_boot=args.bootstrap, seed=args.seed + 101 + 11 * (i + 1)),
                "minus_node_suff_median": med(minus_suff),
                "minus_node_nec_median": med(minus_nec),
                "drop_full_suff_median": med(drop_s),
                "drop_full_suff_ci": bootstrap_ci(drop_s, n_boot=args.bootstrap, seed=args.seed + 201 + 11 * (i + 1)),
                "drop_full_nec_median": med(drop_n),
                "drop_full_nec_ci": bootstrap_ci(drop_n, n_boot=args.bootstrap, seed=args.seed + 301 + 11 * (i + 1)),
            }
        )

    # Ranking and export-friendly columns.
    drop_rank = {
        r["node"]: rank + 1
        for rank, r in enumerate(sorted(summary_rows, key=lambda x: float(x.get("drop_full_nec_median", float("-inf"))), reverse=True))
    }
    standalone_rank = {
        r["node"]: rank + 1
        for rank, r in enumerate(sorted(summary_rows, key=lambda x: float(x.get("node_suff_median", float("-inf"))), reverse=True))
    }
    for r in summary_rows:
        r["rank_drop_full_nec"] = int(drop_rank[r["node"]])
        r["rank_node_suff"] = int(standalone_rank[r["node"]])

    summary_csv = out_root / "node_ablation_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "node",
                "node_role",
                "n_samples",
                "node_suff_median",
                "node_suff_ci_lo",
                "node_suff_ci_hi",
                "node_nec_median",
                "node_nec_ci_lo",
                "node_nec_ci_hi",
                "minus_node_suff_median",
                "minus_node_nec_median",
                "drop_full_suff_median",
                "drop_full_suff_ci_lo",
                "drop_full_suff_ci_hi",
                "drop_full_nec_median",
                "drop_full_nec_ci_lo",
                "drop_full_nec_ci_hi",
                "rank_drop_full_nec",
                "rank_node_suff",
            ]
        )
        for r in summary_rows:
            w.writerow(
                [
                    r["node"],
                    r["node_role"],
                    r["n_samples"],
                    r["node_suff_median"],
                    r["node_suff_ci"]["lo"],
                    r["node_suff_ci"]["hi"],
                    r["node_nec_median"],
                    r["node_nec_ci"]["lo"],
                    r["node_nec_ci"]["hi"],
                    r["minus_node_suff_median"],
                    r["minus_node_nec_median"],
                    r["drop_full_suff_median"],
                    r["drop_full_suff_ci"]["lo"],
                    r["drop_full_suff_ci"]["hi"],
                    r["drop_full_nec_median"],
                    r["drop_full_nec_ci"]["lo"],
                    r["drop_full_nec_ci"]["hi"],
                    r["rank_drop_full_nec"],
                    r["rank_node_suff"],
                ]
            )

    # Plots.
    ordered = sorted(summary_rows, key=lambda x: float(x.get("drop_full_nec_median", float("-inf"))), reverse=True)
    labels = [str(r["node"]) for r in ordered]
    heat_cols = [
        "node_suff_median",
        "node_nec_median",
        "minus_node_suff_median",
        "minus_node_nec_median",
        "drop_full_suff_median",
        "drop_full_nec_median",
    ]
    heat_matrix = np.array(
        [[float(r[c]) for c in heat_cols] for r in ordered],
        dtype=np.float64,
    )
    save_heatmap(
        matrix=heat_matrix,
        row_labels=labels,
        col_labels=heat_cols,
        title="Node Causal Contribution Summary",
        out_path=out_root / "node_ablation_heatmap.png",
    )
    save_bar(
        labels=labels,
        values=[float(r["drop_full_nec_median"]) for r in ordered],
        cis=[r["drop_full_nec_ci"] for r in ordered],
        title="Node Necessity in Full Circuit (drop_full_nec)",
        ylabel="Delta necessity ratio",
        out_path=out_root / "node_drop_full_nec.png",
    )
    save_bar(
        labels=labels,
        values=[float(r["node_suff_median"]) for r in ordered],
        cis=[r["node_suff_ci"] for r in ordered],
        title="Node Standalone Sufficiency (node alone)",
        ylabel="Recovery ratio vs gap",
        out_path=out_root / "node_alone_suff.png",
    )

    proposed_groups = derive_grouping(summary_rows)
    proposed_groups = {
        k: [n for n in v if n in core_set]
        for k, v in proposed_groups.items()
        if v
    }

    report = {
        "n_input_samples": len(sample_infos),
        "n_analyzed_samples": len(set(analyzed)),
        "analyzed_q_indices": sorted(set(analyzed)),
        "skipped_q_indices": sorted(set(skipped)),
        "gap_min": args.gap_min,
        "augment_mode": args.augment_mode,
        "gap_min_aug": args.gap_min_aug,
        "bootstrap": args.bootstrap,
        "seed": args.seed,
        "core_nodes": core_nodes,
        "summary_rows": summary_rows,
        "proposed_groups_data_driven": proposed_groups,
        "artifacts": {
            "per_sample_csv": str(per_sample_csv),
            "summary_csv": str(summary_csv),
            "heatmap_png": str(out_root / "node_ablation_heatmap.png"),
            "drop_full_nec_png": str(out_root / "node_drop_full_nec.png"),
            "node_alone_suff_png": str(out_root / "node_alone_suff.png"),
        },
    }
    report_path = out_root / "node_ablation_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[done] output root: {out_root}")
    print(f"[done] analyzed samples: {len(set(analyzed))}")
    print(f"[done] report: {report_path}")


if __name__ == "__main__":
    main()

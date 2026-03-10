#!/usr/bin/env python3
"""
Cluster-aware transfer evaluation for tool-call circuits.

Given an aggregate summary with cluster-level circuits, this script evaluates:
1) each source cluster circuit on each target cluster sample subset;
2) sufficiency / necessity / delta-vs-random metrics;
3) cross-cluster transfer decay relative to within-cluster performance.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib
import numpy as np
import torch
from tqdm.auto import tqdm

from experiments.aggregate_toolcall_circuits import (
    INPUT_NODE,
    OUTPUT_NODE,
    aggregate_supports,
    load_sample_records,
    pick_consensus_edges,
    pick_consensus_nodes,
)
from experiments.launch_toolcall_qwen3_q85 import (
    collect_clean_cache_cpu,
    evaluate_on_base_with_source,
    load_hooked_qwen3,
    node_layer,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@dataclass
class SourceCircuit:
    name: str
    nodes: List[str]
    edges: List[Tuple[str, str]]
    source_cluster_id: int | None
    cluster_size: int
    build_mode: str


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
    cmap: str = "viridis",
    center: float | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    apply_plot_style()
    fig, ax = plt.subplots(
        figsize=(max(8.5, 1.0 * len(col_labels)), max(4.5, 0.7 * len(row_labels) + 1.8)),
        constrained_layout=True,
    )
    if center is None:
        im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        if vmin is None or vmax is None:
            finite_vals = matrix[np.isfinite(matrix)]
            if finite_vals.size == 0:
                bound = 1.0
            else:
                bound = float(np.nanpercentile(np.abs(finite_vals - center), 95))
                bound = max(bound, 1e-6)
            vmin = center - bound
            vmax = center + bound
        im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title(title)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if math.isfinite(float(v)):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="white", fontsize=8.5)

    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("Median")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def sample_random_nodes(universe: Sequence[str], k: int, rng: random.Random) -> List[str]:
    if k <= 0:
        return []
    if k >= len(universe):
        return list(universe)
    return rng.sample(list(universe), k)


def parse_cluster_assignments(path: Path) -> Dict[int, int]:
    out: Dict[int, int] = {}
    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            q = int(row["q_index"])
            c = int(row["cluster_id"])
            out[q] = c
    return out


def sort_nodes(nodes: Iterable[str]) -> List[str]:
    return sorted(set(nodes), key=lambda n: (node_layer(n), 0 if n.startswith("MLP") else 1, n))


def node_jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def pearson_corr(xs: Sequence[float], ys: Sequence[float]) -> float:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(pairs) < 3:
        return float("nan")
    x = np.asarray([p[0] for p in pairs], dtype=np.float64)
    y = np.asarray([p[1] for p in pairs], dtype=np.float64)
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std <= 1e-12 or y_std <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def rank_values(vals: Sequence[float]) -> List[float]:
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    out = [0.0] * len(vals)
    i = 0
    rank = 1.0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg_rank = (rank + rank + (j - i)) / 2.0
        for k in range(i, j + 1):
            out[order[k]] = avg_rank
        rank += (j - i + 1)
        i = j + 1
    return out


def spearman_corr(xs: Sequence[float], ys: Sequence[float]) -> float:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(pairs) < 3:
        return float("nan")
    x = [p[0] for p in pairs]
    y = [p[1] for p in pairs]
    xr = rank_values(x)
    yr = rank_values(y)
    return pearson_corr(xr, yr)


def save_overlap_scatter(
    *,
    x: Sequence[float],
    y_left: Sequence[float],
    y_right: Sequence[float],
    out_path: Path,
) -> None:
    apply_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), constrained_layout=True)
    configs = [
        (axes[0], y_left, "Suff / Within", "suff_vs_within_ratio"),
        (axes[1], y_right, "Delta / Within", "delta_vs_within_ratio"),
    ]
    for ax, y, title, ylab in configs:
        pts = [(a, b) for a, b in zip(x, y) if math.isfinite(float(a)) and math.isfinite(float(b))]
        if pts:
            xv = np.asarray([p[0] for p in pts], dtype=np.float64)
            yv = np.asarray([p[1] for p in pts], dtype=np.float64)
            ax.scatter(xv, yv, s=40, alpha=0.8, color="#1f77b4", edgecolors="none")
            if len(xv) >= 2 and float(np.std(xv)) > 1e-8:
                m, b = np.polyfit(xv, yv, 1)
                grid = np.linspace(0.0, 1.0, 100)
                ax.plot(grid, m * grid + b, color="#d62728", linewidth=1.6)
        ax.axhline(1.0, linestyle="--", linewidth=1.0, color="#666666")
        ax.set_xlim(-0.03, 1.03)
        ax.set_xlabel("Source/target node Jaccard")
        ax.set_ylabel(ylab)
        ax.set_title(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_sources_from_cluster_summaries(
    agg_summary: Dict[str, object],
    min_cluster_size: int,
    include_global: bool,
) -> List[SourceCircuit]:
    cluster_summaries = list(agg_summary.get("cluster_summaries", []))
    out: List[SourceCircuit] = []
    for c in cluster_summaries:
        cid = int(c["cluster_id"])
        size = int(c["size"])
        if size < min_cluster_size:
            continue
        nodes = sort_nodes([str(n) for n in c.get("nodes", [])])
        edges = [(str(a), str(b)) for a, b in c.get("edges", [])]
        if not nodes:
            continue
        out.append(
            SourceCircuit(
                name=f"cluster_{cid}",
                nodes=nodes,
                edges=edges,
                source_cluster_id=cid,
                cluster_size=size,
                build_mode="cluster_summary",
            )
        )

    if include_global:
        global_nodes = sort_nodes([str(n) for n in agg_summary.get("core_nodes", [])])
        global_edges = [(str(a), str(b)) for a, b in agg_summary.get("core_edges", [])]
        if global_nodes:
            out.append(
                SourceCircuit(
                    name="global_core",
                    nodes=global_nodes,
                    edges=global_edges,
                    source_cluster_id=None,
                    cluster_size=len(global_nodes),
                    build_mode="global_aggregate",
                )
            )
    return out


def build_sources_from_compact_support(
    *,
    records: Sequence[object],
    q_to_cluster: Dict[int, int],
    min_cluster_size: int,
    include_global: bool,
    agg_summary: Dict[str, object],
    compact_node_th: float,
    compact_edge_th: float,
    compact_min_nodes: int,
    compact_max_nodes: int,
    compact_min_edges: int,
    compact_topk_nodes: int,
) -> List[SourceCircuit]:
    cluster_to_records: Dict[int, List[object]] = defaultdict(list)
    for rec in records:
        q = int(rec.q_index)
        if q in q_to_cluster:
            cluster_to_records[int(q_to_cluster[q])].append(rec)

    out: List[SourceCircuit] = []
    cluster_items = sorted(cluster_to_records.items(), key=lambda x: len(x[1]), reverse=True)
    for cid, members in cluster_items:
        if len(members) < min_cluster_size:
            continue
        node_support, edge_support, _, _, _, _ = aggregate_supports(members)
        if not node_support:
            continue

        if compact_topk_nodes > 0:
            ranked = sorted(
                node_support.items(),
                key=lambda x: (-float(x[1]), node_layer(str(x[0])), str(x[0])),
            )
            nodes = sort_nodes([n for n, _ in ranked[:compact_topk_nodes]])
        else:
            nodes = pick_consensus_nodes(
                node_support=node_support,
                node_threshold=compact_node_th,
                min_nodes=compact_min_nodes,
                max_nodes=compact_max_nodes,
            )
        if not nodes:
            continue

        edges = pick_consensus_edges(
            nodes=nodes,
            edge_support=edge_support,
            node_support=node_support,
            edge_threshold=compact_edge_th,
            min_edges=max(compact_min_edges, len(nodes)),
        )
        connected_nodes = sort_nodes(
            [
                n
                for a, b in edges
                for n in (a, b)
                if n not in {INPUT_NODE, OUTPUT_NODE}
            ]
        )
        if connected_nodes:
            nodes = connected_nodes

        out.append(
            SourceCircuit(
                name=f"cluster_{cid}",
                nodes=nodes,
                edges=edges,
                source_cluster_id=cid,
                cluster_size=len(members),
                build_mode="compact_support",
            )
        )

    if include_global:
        global_nodes = sort_nodes([str(n) for n in agg_summary.get("core_nodes", [])])
        global_edges = [(str(a), str(b)) for a, b in agg_summary.get("core_edges", [])]
        if global_nodes:
            out.append(
                SourceCircuit(
                    name="global_core",
                    nodes=global_nodes,
                    edges=global_edges,
                    source_cluster_id=None,
                    cluster_size=len(global_nodes),
                    build_mode="global_aggregate",
                )
            )
    return out


def build_target_labels(
    sources: Sequence[SourceCircuit],
    include_all_valid: bool,
) -> List[str]:
    labels: List[str] = []
    for s in sources:
        if s.source_cluster_id is not None:
            lbl = f"cluster_{s.source_cluster_id}"
            if lbl not in labels:
                labels.append(lbl)
    if include_all_valid:
        labels.append("all_valid")
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate cross-cluster transfer for tool-call circuits.")
    parser.add_argument("--input-root", type=str, default="experiments/results/toolcall_q1_q164")
    parser.add_argument(
        "--aggregate-summary",
        type=str,
        default="experiments/results/overnight_round2/aggregate_best_n0.55_e0.4/global_core_summary.json",
    )
    parser.add_argument(
        "--cluster-assignments",
        type=str,
        default="",
        help="Optional cluster_assignments.csv path; if empty, use aggregate artifact path.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="experiments/results/overnight_round2/cluster_transfer_best_n0.55_e0.4",
    )
    parser.add_argument("--model-path", type=str, default="/root/data/Qwen/Qwen3-1.7B")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gap-min", type=float, default=0.5)
    parser.add_argument("--q-start", type=int, default=1)
    parser.add_argument("--q-end", type=int, default=10_000)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all eligible samples.")
    parser.add_argument("--min-cluster-size", type=int, default=8)
    parser.add_argument("--min-target-samples", type=int, default=6)
    parser.add_argument(
        "--source-mode",
        type=str,
        choices=["cluster_summary", "compact_support"],
        default="cluster_summary",
        help="How to build source cluster circuits for transfer evaluation.",
    )
    parser.add_argument("--compact-node-th", type=float, default=0.65)
    parser.add_argument("--compact-edge-th", type=float, default=0.45)
    parser.add_argument("--compact-min-nodes", type=int, default=4)
    parser.add_argument("--compact-max-nodes", type=int, default=8)
    parser.add_argument("--compact-min-edges", type=int, default=6)
    parser.add_argument(
        "--compact-topk-nodes",
        type=int,
        default=0,
        help="If >0, override compact node threshold and directly use top-k node support per cluster.",
    )
    parser.add_argument("--include-global", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-all-valid-target", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--n-random", type=int, default=1, help="Random controls per sample per source.")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    input_root = Path(args.input_root).resolve()
    agg_path = Path(args.aggregate_summary).resolve()
    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    agg = json.loads(agg_path.read_text(encoding="utf-8"))
    cluster_assignment_path = (
        Path(args.cluster_assignments).resolve()
        if args.cluster_assignments.strip()
        else Path(str(agg.get("artifacts", {}).get("cluster_assignments_csv", ""))).resolve()
    )
    if not cluster_assignment_path.exists():
        raise FileNotFoundError(f"cluster assignments not found: {cluster_assignment_path}")
    q_to_cluster = parse_cluster_assignments(cluster_assignment_path)

    ap_discount = float(agg.get("weight_params", {}).get("ap_discount", 0.7))
    records = load_sample_records(root=input_root, gap_min=args.gap_min, ap_discount=ap_discount)
    records = [r for r in records if args.q_start <= int(r.q_index) <= args.q_end]

    # Keep only records with valid cluster id and valid objective/gap metadata.
    filtered = []
    for rec in records:
        q = int(rec.q_index)
        if q not in q_to_cluster:
            continue
        s = rec.summary
        gap = float(s.get("gap", float("nan")))
        clean_obj = float(s.get("clean_obj", float("nan")))
        corrupt_obj = float(s.get("corrupt_obj", float("nan")))
        target_token = int(s.get("target_token_id", -1))
        distractor = int(s.get("distractor_token_id", -1))
        if not (
            math.isfinite(gap)
            and math.isfinite(clean_obj)
            and math.isfinite(corrupt_obj)
            and gap > args.gap_min
            and target_token >= 0
            and distractor >= 0
        ):
            continue
        clean_prompt = Path(str(s.get("clean_prompt", "")))
        corrupt_prompt = Path(str(s.get("corrupt_prompt", "")))
        if not clean_prompt.exists() or not corrupt_prompt.exists():
            continue
        filtered.append(rec)

    if args.max_samples > 0:
        filtered = filtered[: args.max_samples]
    if not filtered:
        raise ValueError("No valid samples after filtering.")

    if args.source_mode == "cluster_summary":
        sources = build_sources_from_cluster_summaries(
            agg_summary=agg,
            min_cluster_size=args.min_cluster_size,
            include_global=args.include_global,
        )
    else:
        sources = build_sources_from_compact_support(
            records=filtered,
            q_to_cluster=q_to_cluster,
            min_cluster_size=args.min_cluster_size,
            include_global=args.include_global,
            agg_summary=agg,
            compact_node_th=args.compact_node_th,
            compact_edge_th=args.compact_edge_th,
            compact_min_nodes=args.compact_min_nodes,
            compact_max_nodes=args.compact_max_nodes,
            compact_min_edges=args.compact_min_edges,
            compact_topk_nodes=args.compact_topk_nodes,
        )
    if not sources:
        raise ValueError("No source circuits found after source construction.")

    target_labels = build_target_labels(sources, include_all_valid=args.include_all_valid_target)
    if not target_labels:
        raise ValueError("No target labels built.")

    # Filter target labels with too few samples.
    target_counts: Dict[str, int] = defaultdict(int)
    for rec in filtered:
        lbl = f"cluster_{q_to_cluster[int(rec.q_index)]}"
        target_counts[lbl] += 1
    if args.include_all_valid_target:
        target_counts["all_valid"] = len(filtered)

    kept_targets = [
        t
        for t in target_labels
        if t == "all_valid" or target_counts.get(t, 0) >= args.min_target_samples
    ]
    target_labels = kept_targets
    if not target_labels:
        raise ValueError("No target labels remain after min-target-samples filtering.")

    # For random controls, use all discovered nodes from valid samples.
    universe_nodes = sorted({n for rec in filtered for n in rec.nodes})
    rng = random.Random(args.seed)

    model, _ = load_hooked_qwen3(args.model_path, device=args.device, dtype=torch.bfloat16)
    model.set_use_attn_result(False)
    model.set_use_split_qkv_input(False)
    model.set_use_hook_mlp_in(False)

    rows: List[Dict[str, object]] = []
    skipped_q: List[int] = []
    processed_q: List[int] = []

    pbar = tqdm(filtered, desc="Cluster transfer", dynamic_ncols=True)
    for rec in pbar:
        q = int(rec.q_index)
        s = rec.summary
        cluster_lbl = f"cluster_{q_to_cluster[q]}"
        if cluster_lbl not in target_labels and "all_valid" not in target_labels:
            continue

        clean_prompt = Path(str(s["clean_prompt"]))
        corrupt_prompt = Path(str(s["corrupt_prompt"]))
        clean_text = clean_prompt.read_text(encoding="utf-8")
        corrupt_text = corrupt_prompt.read_text(encoding="utf-8")
        clean_tokens = model.to_tokens(clean_text, prepend_bos=False)
        corrupt_tokens = model.to_tokens(corrupt_text, prepend_bos=False)
        if clean_tokens.shape != corrupt_tokens.shape:
            skipped_q.append(q)
            continue

        gap = float(s["gap"])
        clean_obj = float(s["clean_obj"])
        corrupt_obj = float(s["corrupt_obj"])
        target_token = int(s["target_token_id"])
        distractor = int(s["distractor_token_id"])

        try:
            clean_cache = collect_clean_cache_cpu(model, clean_tokens)
            corrupt_cache = collect_clean_cache_cpu(model, corrupt_tokens)
        except torch.OutOfMemoryError:
            skipped_q.append(q)
            model.reset_hooks()
            gc.collect()
            torch.cuda.empty_cache()
            continue

        try:
            for src in sources:
                nodes = src.nodes
                k = len(nodes)

                suff_obj = evaluate_on_base_with_source(
                    model=model,
                    base_tokens=corrupt_tokens,
                    source_cache_cpu=clean_cache,
                    patch_nodes=nodes,
                    target_token=target_token,
                    distractor_token=distractor,
                )
                suff_ratio = (suff_obj - corrupt_obj) / gap

                nec_obj = evaluate_on_base_with_source(
                    model=model,
                    base_tokens=clean_tokens,
                    source_cache_cpu=corrupt_cache,
                    patch_nodes=nodes,
                    target_token=target_token,
                    distractor_token=distractor,
                )
                nec_ratio = (clean_obj - nec_obj) / gap

                random_suffs: List[float] = []
                for _ in range(max(0, args.n_random)):
                    rnd_nodes = sample_random_nodes(universe=universe_nodes, k=k, rng=rng)
                    rnd_obj = evaluate_on_base_with_source(
                        model=model,
                        base_tokens=corrupt_tokens,
                        source_cache_cpu=clean_cache,
                        patch_nodes=rnd_nodes,
                        target_token=target_token,
                        distractor_token=distractor,
                    )
                    random_suffs.append((rnd_obj - corrupt_obj) / gap)
                rnd_mean = float(np.mean(random_suffs)) if random_suffs else float("nan")
                delta = suff_ratio - rnd_mean if math.isfinite(rnd_mean) else float("nan")

                # Cluster-specific target row.
                if cluster_lbl in target_labels:
                    rows.append(
                        {
                            "q_index": q,
                            "source": src.name,
                            "source_build_mode": src.build_mode,
                            "source_cluster_id": src.source_cluster_id,
                            "source_cluster_size": src.cluster_size,
                            "target": cluster_lbl,
                            "source_nodes": k,
                            "suff_ratio": suff_ratio,
                            "nec_ratio": nec_ratio,
                            "random_suff_mean": rnd_mean,
                            "delta_vs_random": delta,
                            "gap": gap,
                        }
                    )
                # Global target row.
                if "all_valid" in target_labels:
                    rows.append(
                        {
                            "q_index": q,
                            "source": src.name,
                            "source_build_mode": src.build_mode,
                            "source_cluster_id": src.source_cluster_id,
                            "source_cluster_size": src.cluster_size,
                            "target": "all_valid",
                            "source_nodes": k,
                            "suff_ratio": suff_ratio,
                            "nec_ratio": nec_ratio,
                            "random_suff_mean": rnd_mean,
                            "delta_vs_random": delta,
                            "gap": gap,
                        }
                    )
        except torch.OutOfMemoryError:
            skipped_q.append(q)
        finally:
            processed_q.append(q)
            model.reset_hooks()
            del clean_cache
            del corrupt_cache
            gc.collect()
            torch.cuda.empty_cache()

    per_sample_csv = out_root / "cluster_transfer_per_sample.csv"
    per_fields = [
        "q_index",
        "source",
        "source_build_mode",
        "source_cluster_id",
        "source_cluster_size",
        "target",
        "source_nodes",
        "suff_ratio",
        "nec_ratio",
        "random_suff_mean",
        "delta_vs_random",
        "gap",
    ]
    with per_sample_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=per_fields)
        w.writeheader()
        w.writerows(rows)

    by_pair: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for r in rows:
        by_pair[(str(r["source"]), str(r["target"]))].append(r)

    source_labels = [s.name for s in sources]
    source_by_name = {s.name: s for s in sources}
    summary_rows: List[Dict[str, object]] = []
    for src in source_labels:
        for tgt in target_labels:
            grp = by_pair.get((src, tgt), [])
            suff_vals = [float(x["suff_ratio"]) for x in grp]
            nec_vals = [float(x["nec_ratio"]) for x in grp]
            rnd_vals = [float(x["random_suff_mean"]) for x in grp]
            del_vals = [float(x["delta_vs_random"]) for x in grp]
            src_meta = source_by_name[src]
            summary_rows.append(
                {
                    "source": src,
                    "source_build_mode": src_meta.build_mode,
                    "source_cluster_id": src_meta.source_cluster_id,
                    "source_cluster_size": src_meta.cluster_size,
                    "target": tgt,
                    "n_samples": len(grp),
                    "source_nodes": int(grp[0]["source_nodes"]) if grp else 0,
                    "suff_median": med(suff_vals),
                    "suff_mean": mean(suff_vals),
                    "nec_median": med(nec_vals),
                    "nec_mean": mean(nec_vals),
                    "random_suff_median": med(rnd_vals),
                    "delta_vs_random_median": med(del_vals),
                    "delta_vs_random_mean": mean(del_vals),
                }
            )

    # Relative-to-within transfer ratio for cluster sources.
    within_lookup: Dict[str, float] = {}
    within_delta_lookup: Dict[str, float] = {}
    for src in source_labels:
        if src.startswith("cluster_"):
            self_row = next((r for r in summary_rows if r["source"] == src and r["target"] == src), None)
            within_lookup[src] = float(self_row["suff_median"]) if self_row else float("nan")
            within_delta_lookup[src] = float(self_row["delta_vs_random_median"]) if self_row else float("nan")

    for r in summary_rows:
        src = str(r["source"])
        within = within_lookup.get(src, float("nan"))
        within_delta = within_delta_lookup.get(src, float("nan"))
        suff = float(r["suff_median"])
        delta = float(r["delta_vs_random_median"])
        r["suff_vs_within_ratio"] = suff / within if math.isfinite(within) and abs(within) > 1e-8 else float("nan")
        r["delta_vs_within_ratio"] = (
            delta / within_delta if math.isfinite(within_delta) and abs(within_delta) > 1e-8 else float("nan")
        )

    summary_csv = out_root / "cluster_transfer_summary.csv"
    summary_fields = [
        "source",
        "source_build_mode",
        "source_cluster_id",
        "source_cluster_size",
        "target",
        "n_samples",
        "source_nodes",
        "suff_median",
        "suff_mean",
        "nec_median",
        "nec_mean",
        "random_suff_median",
        "delta_vs_random_median",
        "delta_vs_random_mean",
        "suff_vs_within_ratio",
        "delta_vs_within_ratio",
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        w.writerows(summary_rows)

    # Transfer decay table.
    decay_rows: List[Dict[str, object]] = []
    cluster_targets = [t for t in target_labels if t.startswith("cluster_")]
    for src in source_labels:
        if not src.startswith("cluster_"):
            continue
        within = next((r for r in summary_rows if r["source"] == src and r["target"] == src), None)
        if within is None:
            continue
        cross = [
            r for r in summary_rows
            if r["source"] == src and r["target"] in cluster_targets and r["target"] != src
        ]
        cross_suff = [float(r["suff_median"]) for r in cross]
        cross_delta = [float(r["delta_vs_random_median"]) for r in cross]
        within_suff = float(within["suff_median"])
        within_delta = float(within["delta_vs_random_median"])
        cross_suff_mean = mean(cross_suff)
        cross_delta_mean = mean(cross_delta)
        decay_rows.append(
            {
                "source": src,
                "within_suff_median": within_suff,
                "cross_suff_median_mean": cross_suff_mean,
                "cross_over_within_suff": (
                    cross_suff_mean / within_suff if math.isfinite(within_suff) and abs(within_suff) > 1e-8 else float("nan")
                ),
                "within_delta_median": within_delta,
                "cross_delta_median_mean": cross_delta_mean,
                "cross_over_within_delta": (
                    cross_delta_mean / within_delta
                    if math.isfinite(within_delta) and abs(within_delta) > 1e-8
                    else float("nan")
                ),
            }
        )

    decay_csv = out_root / "cluster_transfer_decay.csv"
    decay_fields = [
        "source",
        "within_suff_median",
        "cross_suff_median_mean",
        "cross_over_within_suff",
        "within_delta_median",
        "cross_delta_median_mean",
        "cross_over_within_delta",
    ]
    with decay_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=decay_fields)
        w.writeheader()
        w.writerows(decay_rows)

    sources_json = out_root / "cluster_transfer_sources.json"
    sources_payload = [
        {
            "name": s.name,
            "source_cluster_id": s.source_cluster_id,
            "cluster_size": s.cluster_size,
            "build_mode": s.build_mode,
            "n_nodes": len(s.nodes),
            "n_edges": len(s.edges),
            "nodes": s.nodes,
            "edges": [[a, b] for a, b in s.edges],
        }
        for s in sources
    ]
    sources_json.write_text(json.dumps(sources_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # Overlap diagnostics to explicitly quantify the transfer overlap confound.
    overlap_rows: List[Dict[str, object]] = []
    overlap_matrix = np.full((len(source_labels), len(source_labels)), np.nan, dtype=np.float64)
    for i, src_a in enumerate(source_labels):
        nodes_a = source_by_name[src_a].nodes
        set_a = set(nodes_a)
        for j, src_b in enumerate(source_labels):
            nodes_b = source_by_name[src_b].nodes
            set_b = set(nodes_b)
            jac = node_jaccard(nodes_a, nodes_b)
            overlap_matrix[i, j] = jac
            if j < i:
                continue
            overlap_rows.append(
                {
                    "source_a": src_a,
                    "source_b": src_b,
                    "source_a_nodes": len(nodes_a),
                    "source_b_nodes": len(nodes_b),
                    "intersection": len(set_a & set_b),
                    "union": len(set_a | set_b),
                    "jaccard": jac,
                }
            )

    overlap_csv = out_root / "cluster_source_overlap.csv"
    overlap_fields = [
        "source_a",
        "source_b",
        "source_a_nodes",
        "source_b_nodes",
        "intersection",
        "union",
        "jaccard",
    ]
    with overlap_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=overlap_fields)
        w.writeheader()
        w.writerows(overlap_rows)

    save_heatmap(
        matrix=overlap_matrix,
        row_labels=source_labels,
        col_labels=source_labels,
        title="Source Circuit Node Overlap (Jaccard)",
        out_path=out_root / "cluster_source_overlap_heatmap.png",
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
    )

    transfer_overlap_rows: List[Dict[str, object]] = []
    for r in summary_rows:
        src = str(r["source"])
        tgt = str(r["target"])
        if not src.startswith("cluster_") or not tgt.startswith("cluster_") or src == tgt:
            continue
        if src not in source_by_name or tgt not in source_by_name:
            continue
        transfer_overlap_rows.append(
            {
                "source": src,
                "target": tgt,
                "n_samples": int(r["n_samples"]),
                "source_nodes": int(r["source_nodes"]),
                "target_nodes": len(source_by_name[tgt].nodes),
                "source_target_jaccard": node_jaccard(source_by_name[src].nodes, source_by_name[tgt].nodes),
                "suff_median": float(r["suff_median"]),
                "delta_vs_random_median": float(r["delta_vs_random_median"]),
                "suff_vs_within_ratio": float(r["suff_vs_within_ratio"]),
                "delta_vs_within_ratio": float(r["delta_vs_within_ratio"]),
            }
        )

    overlap_transfer_csv = out_root / "cluster_transfer_overlap_pairs.csv"
    overlap_transfer_fields = [
        "source",
        "target",
        "n_samples",
        "source_nodes",
        "target_nodes",
        "source_target_jaccard",
        "suff_median",
        "delta_vs_random_median",
        "suff_vs_within_ratio",
        "delta_vs_within_ratio",
    ]
    with overlap_transfer_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=overlap_transfer_fields)
        w.writeheader()
        w.writerows(transfer_overlap_rows)

    x_overlap = [float(r["source_target_jaccard"]) for r in transfer_overlap_rows]
    y_suff_rel = [float(r["suff_vs_within_ratio"]) for r in transfer_overlap_rows]
    y_delta_rel = [float(r["delta_vs_within_ratio"]) for r in transfer_overlap_rows]
    overlap_corr = {
        "n_pairs": len(transfer_overlap_rows),
        "pearson_jaccard_vs_suff_within_ratio": pearson_corr(x_overlap, y_suff_rel),
        "spearman_jaccard_vs_suff_within_ratio": spearman_corr(x_overlap, y_suff_rel),
        "pearson_jaccard_vs_delta_within_ratio": pearson_corr(x_overlap, y_delta_rel),
        "spearman_jaccard_vs_delta_within_ratio": spearman_corr(x_overlap, y_delta_rel),
    }
    overlap_corr_json = out_root / "cluster_transfer_overlap_correlation.json"
    overlap_corr_json.write_text(json.dumps(overlap_corr, ensure_ascii=False, indent=2), encoding="utf-8")

    save_overlap_scatter(
        x=x_overlap,
        y_left=y_suff_rel,
        y_right=y_delta_rel,
        out_path=out_root / "cluster_transfer_overlap_scatter.png",
    )

    # Matrices + plots.
    mat_suff = np.full((len(source_labels), len(target_labels)), np.nan, dtype=np.float64)
    mat_delta = np.full_like(mat_suff, np.nan)
    mat_rel = np.full_like(mat_suff, np.nan)
    for i, src in enumerate(source_labels):
        for j, tgt in enumerate(target_labels):
            row = next((r for r in summary_rows if r["source"] == src and r["target"] == tgt), None)
            if row is None:
                continue
            mat_suff[i, j] = float(row["suff_median"])
            mat_delta[i, j] = float(row["delta_vs_random_median"])
            mat_rel[i, j] = float(row["suff_vs_within_ratio"])

    save_heatmap(
        matrix=mat_suff,
        row_labels=source_labels,
        col_labels=target_labels,
        title="Cluster Transfer Matrix: Sufficiency Median",
        out_path=out_root / "cluster_transfer_suff_heatmap.png",
        cmap="viridis",
    )
    save_heatmap(
        matrix=mat_delta,
        row_labels=source_labels,
        col_labels=target_labels,
        title="Cluster Transfer Matrix: Delta vs Random (Median)",
        out_path=out_root / "cluster_transfer_delta_heatmap.png",
        cmap="RdBu_r",
        center=0.0,
    )
    save_heatmap(
        matrix=mat_rel,
        row_labels=source_labels,
        col_labels=target_labels,
        title="Cluster Transfer: Sufficiency / Within-Cluster Sufficiency",
        out_path=out_root / "cluster_transfer_relative_suff_heatmap.png",
        cmap="magma",
        vmin=0.0,
        vmax=max(1.2, float(np.nanpercentile(mat_rel[np.isfinite(mat_rel)], 95)) if np.isfinite(mat_rel).any() else 1.2),
    )

    report = {
        "input_root": str(input_root),
        "aggregate_summary": str(agg_path),
        "cluster_assignments": str(cluster_assignment_path),
        "gap_min": args.gap_min,
        "min_cluster_size": args.min_cluster_size,
        "min_target_samples": args.min_target_samples,
        "source_mode": args.source_mode,
        "compact_params": {
            "node_th": args.compact_node_th,
            "edge_th": args.compact_edge_th,
            "min_nodes": args.compact_min_nodes,
            "max_nodes": args.compact_max_nodes,
            "min_edges": args.compact_min_edges,
            "topk_nodes": args.compact_topk_nodes,
        },
        "n_random": args.n_random,
        "seed": args.seed,
        "n_records_filtered": len(filtered),
        "n_records_processed": len(set(processed_q)),
        "skipped_q_indices": sorted(set(skipped_q)),
        "sources": [
            {
                "name": s.name,
                "source_cluster_id": s.source_cluster_id,
                "cluster_size": s.cluster_size,
                "build_mode": s.build_mode,
                "n_nodes": len(s.nodes),
                "n_edges": len(s.edges),
            }
            for s in sources
        ],
        "targets": [{"label": t, "n_samples": int(target_counts.get(t, 0))} for t in target_labels],
        "decay_rows": decay_rows,
        "overlap_correlation": overlap_corr,
        "artifacts": {
            "per_sample_csv": str(per_sample_csv),
            "summary_csv": str(summary_csv),
            "decay_csv": str(decay_csv),
            "sources_json": str(sources_json),
            "source_overlap_csv": str(overlap_csv),
            "source_overlap_heatmap_png": str(out_root / "cluster_source_overlap_heatmap.png"),
            "overlap_pairs_csv": str(overlap_transfer_csv),
            "overlap_correlation_json": str(overlap_corr_json),
            "overlap_scatter_png": str(out_root / "cluster_transfer_overlap_scatter.png"),
            "suff_heatmap_png": str(out_root / "cluster_transfer_suff_heatmap.png"),
            "delta_heatmap_png": str(out_root / "cluster_transfer_delta_heatmap.png"),
            "relative_suff_heatmap_png": str(out_root / "cluster_transfer_relative_suff_heatmap.png"),
        },
    }
    (out_root / "cluster_transfer_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[done] output root: {out_root}")
    print(f"[done] processed samples: {len(set(processed_q))}")
    print(f"[done] sources={len(source_labels)} targets={len(target_labels)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Cross-sample circuit aggregation for tool-call circuits discovered on q1..q164.

Inputs:
- Per-sample summaries under experiments/results/toolcall_q1_q164/qXXX/summary.json

Outputs:
- node/edge support tables
- global core and relaxed consensus circuits
- cluster assignments and cluster-level circuits
- functional replay of global core on all samples
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
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

from experiments.launch_toolcall_qwen3_q85 import (
    collect_clean_cache_cpu,
    draw_circuit,
    evaluate_on_base_with_source,
    load_hooked_qwen3,
    node_layer,
)

INPUT_NODE = "Input Embed"
OUTPUT_NODE = "Residual Output: <tool_call>"


@dataclass
class SampleRecord:
    q_index: int
    path: Path
    summary: Dict[str, object]
    weight: float
    nodes: List[str]
    edges: List[Tuple[str, str]]


def is_finite_num(v: object) -> bool:
    return isinstance(v, (int, float)) and math.isfinite(float(v))


def safe_float(v: object, default: float = 0.0) -> float:
    return float(v) if is_finite_num(v) else default


def compute_sample_weight(summary: Dict[str, object], gap_min: float, ap_discount: float) -> float:
    gap = safe_float(summary.get("gap"), 0.0)
    if gap <= gap_min:
        return 0.0

    det = max(0.0, min(1.0, safe_float(summary.get("detailed_ratio_vs_gap"), 0.0)))
    nec = max(0.0, min(1.0, safe_float(summary.get("necessity_ratio_vs_gap"), 0.0)))
    w = det * nec
    ap_mode = str(summary.get("ap_mode", "full"))
    if ap_mode != "full":
        w *= ap_discount
    return w


def load_sample_records(root: Path, gap_min: float, ap_discount: float) -> List[SampleRecord]:
    records: List[SampleRecord] = []
    for q_dir in sorted(root.glob("q[0-9][0-9][0-9]")):
        summary_path = q_dir / "summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        q_index = int(summary.get("q_index"))
        nodes = list(summary.get("detailed_nodes", []))
        raw_edges = summary.get("detailed_edges", [])
        edges = [(str(a), str(b)) for a, b in raw_edges]
        weight = compute_sample_weight(summary, gap_min=gap_min, ap_discount=ap_discount)
        records.append(
            SampleRecord(
                q_index=q_index,
                path=q_dir,
                summary=summary,
                weight=weight,
                nodes=nodes,
                edges=edges,
            )
        )
    records.sort(key=lambda r: r.q_index)
    return records


def aggregate_supports(
    records: Sequence[SampleRecord],
) -> Tuple[
    Dict[str, float],
    Dict[Tuple[str, str], float],
    Dict[str, float],
    Dict[Tuple[str, str], float],
    Dict[str, List[float]],
    float,
]:
    node_wsum: Dict[str, float] = defaultdict(float)
    edge_wsum: Dict[Tuple[str, str], float] = defaultdict(float)
    node_cnt: Dict[str, float] = defaultdict(float)
    edge_cnt: Dict[Tuple[str, str], float] = defaultdict(float)
    node_norm_scores: Dict[str, List[float]] = defaultdict(list)
    total_w = 0.0

    for rec in records:
        w = rec.weight
        if w <= 0:
            continue
        total_w += w
        uniq_nodes = set(rec.nodes)
        uniq_edges = set(rec.edges)
        for n in uniq_nodes:
            node_wsum[n] += w
            node_cnt[n] += 1.0
        for e in uniq_edges:
            edge_wsum[e] += w
            edge_cnt[e] += 1.0

        gap = abs(safe_float(rec.summary.get("gap"), 0.0))
        if gap > 1e-8:
            for d in rec.summary.get("top_node_scores", []):
                name = str(d.get("name"))
                score = safe_float(d.get("score"), 0.0)
                node_norm_scores[name].append(score / gap)

    if total_w <= 0:
        return {}, {}, {}, {}, {}, 0.0

    node_support = {k: v / total_w for k, v in node_wsum.items()}
    edge_support = {k: v / total_w for k, v in edge_wsum.items()}
    return node_support, edge_support, node_cnt, edge_cnt, node_norm_scores, total_w


def write_node_table(
    out_path: Path,
    node_support: Dict[str, float],
    node_cnt: Dict[str, float],
    node_norm_scores: Dict[str, List[float]],
    n_samples: int,
) -> None:
    rows = []
    for n, s in node_support.items():
        vals = node_norm_scores.get(n, [])
        rows.append(
            {
                "node": n,
                "layer": node_layer(n),
                "support_weighted": s,
                "support_unweighted": node_cnt.get(n, 0.0) / max(1, n_samples),
                "median_score_norm": float(np.median(vals)) if vals else float("nan"),
                "mean_score_norm": float(np.mean(vals)) if vals else float("nan"),
            }
        )
    rows.sort(key=lambda x: x["support_weighted"], reverse=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "node",
                "layer",
                "support_weighted",
                "support_unweighted",
                "median_score_norm",
                "mean_score_norm",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_edge_table(
    out_path: Path,
    edge_support: Dict[Tuple[str, str], float],
    edge_cnt: Dict[Tuple[str, str], float],
    n_samples: int,
) -> None:
    rows = []
    for (a, b), s in edge_support.items():
        rows.append(
            {
                "source": a,
                "target": b,
                "support_weighted": s,
                "support_unweighted": edge_cnt.get((a, b), 0.0) / max(1, n_samples),
            }
        )
    rows.sort(key=lambda x: x["support_weighted"], reverse=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["source", "target", "support_weighted", "support_unweighted"],
        )
        writer.writeheader()
        writer.writerows(rows)


def dedup_edges(edges: Iterable[Tuple[str, str]]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    seen = set()
    for e in edges:
        if e not in seen:
            out.append(e)
            seen.add(e)
    return out


def pick_consensus_nodes(
    node_support: Dict[str, float],
    node_threshold: float,
    min_nodes: int,
    max_nodes: int,
) -> List[str]:
    candidates = [(n, s) for n, s in node_support.items() if s >= node_threshold]
    candidates.sort(key=lambda x: x[1], reverse=True)
    if len(candidates) < min_nodes:
        all_nodes = sorted(node_support.items(), key=lambda x: x[1], reverse=True)
        candidates = all_nodes[:min_nodes]
    nodes = [n for n, _ in candidates[:max_nodes]]
    return sorted(set(nodes), key=lambda n: (node_layer(n), 0 if n.startswith("MLP") else 1, n))


def repair_consensus_graph(
    nodes: Sequence[str],
    edges: Sequence[Tuple[str, str]],
    node_support: Dict[str, float],
    edge_support: Dict[Tuple[str, str], float],
) -> List[Tuple[str, str]]:
    if not nodes:
        return [(INPUT_NODE, OUTPUT_NODE)]

    nodes_sorted = sorted(nodes, key=lambda n: (node_layer(n), n))
    edges = dedup_edges(edges)
    valid_nodes = set(nodes_sorted)

    def valid_edge(e: Tuple[str, str]) -> bool:
        a, b = e
        if a == INPUT_NODE and b in valid_nodes:
            return True
        if b == OUTPUT_NODE and a in valid_nodes:
            return True
        return a in valid_nodes and b in valid_nodes and node_layer(a) <= node_layer(b)

    edges = [e for e in edges if valid_edge(e)]

    if not any(a == INPUT_NODE for a, _ in edges):
        earliest_layer = min(node_layer(n) for n in nodes_sorted)
        earliest = [n for n in nodes_sorted if node_layer(n) == earliest_layer]
        earliest = sorted(earliest, key=lambda n: node_support.get(n, 0.0), reverse=True)
        for n in earliest[:2]:
            edges.append((INPUT_NODE, n))

    if not any(b == OUTPUT_NODE for _, b in edges):
        latest = sorted(
            nodes_sorted,
            key=lambda n: (node_layer(n), node_support.get(n, 0.0)),
            reverse=True,
        )
        for n in latest[:3]:
            edges.append((n, OUTPUT_NODE))

    edges = dedup_edges(edges)

    # Ensure every node is reachable from input.
    adjacency: Dict[str, List[str]] = defaultdict(list)
    for a, b in edges:
        adjacency[a].append(b)

    reached = {INPUT_NODE}
    queue = [INPUT_NODE]
    while queue:
        cur = queue.pop(0)
        for nxt in adjacency.get(cur, []):
            if nxt not in reached:
                reached.add(nxt)
                queue.append(nxt)

    for n in nodes_sorted:
        if n in reached:
            continue
        preds = [p for p in nodes_sorted if p in reached and node_layer(p) <= node_layer(n)]
        if preds:
            best = max(
                preds,
                key=lambda p: (
                    edge_support.get((p, n), 0.0),
                    node_support.get(p, 0.0),
                ),
            )
            edges.append((best, n))
        else:
            edges.append((INPUT_NODE, n))
        reached.add(n)

    edges = dedup_edges(edges)

    # Ensure out-degree for all non-output nodes.
    outdeg: Dict[str, int] = defaultdict(int)
    for a, _ in edges:
        outdeg[a] += 1

    for n in [INPUT_NODE] + list(nodes_sorted):
        if outdeg.get(n, 0) > 0:
            continue
        later = [m for m in nodes_sorted if node_layer(m) > node_layer(n)]
        if later:
            best = max(
                later,
                key=lambda m: (
                    edge_support.get((n, m), 0.0),
                    node_support.get(m, 0.0),
                ),
            )
            edges.append((n, best))
        else:
            edges.append((n, OUTPUT_NODE))

    return dedup_edges(edges)


def pick_consensus_edges(
    nodes: Sequence[str],
    edge_support: Dict[Tuple[str, str], float],
    node_support: Dict[str, float],
    edge_threshold: float,
    min_edges: int,
) -> List[Tuple[str, str]]:
    node_set = set(nodes)
    edges = []
    for (a, b), s in edge_support.items():
        if s < edge_threshold:
            continue
        if a == INPUT_NODE and b in node_set:
            edges.append((a, b))
        elif b == OUTPUT_NODE and a in node_set:
            edges.append((a, b))
        elif a in node_set and b in node_set:
            edges.append((a, b))

    if len(edges) < min_edges:
        ranked = sorted(edge_support.items(), key=lambda x: x[1], reverse=True)
        for (a, b), _ in ranked:
            if a == INPUT_NODE and b in node_set:
                edges.append((a, b))
            elif b == OUTPUT_NODE and a in node_set:
                edges.append((a, b))
            elif a in node_set and b in node_set:
                edges.append((a, b))
            if len(dedup_edges(edges)) >= min_edges:
                break

    return repair_consensus_graph(
        nodes=nodes,
        edges=dedup_edges(edges),
        node_support=node_support,
        edge_support=edge_support,
    )


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / max(1, union)


def cluster_samples(records: Sequence[SampleRecord], sim_threshold: float) -> Dict[int, int]:
    active = [r for r in records if r.weight > 0]
    ordered = sorted(active, key=lambda r: r.weight, reverse=True)
    clusters: List[Dict[str, object]] = []
    assignment: Dict[int, int] = {}
    sets = {r.q_index: set(r.nodes) for r in ordered}

    for rec in ordered:
        best_idx = None
        best_sim = -1.0
        rec_set = sets[rec.q_index]
        for idx, c in enumerate(clusters):
            medoid_q = int(c["medoid"])
            sim = jaccard(rec_set, sets[medoid_q])
            if sim >= sim_threshold and sim > best_sim:
                best_sim = sim
                best_idx = idx
        if best_idx is None:
            clusters.append({"medoid": rec.q_index, "members": [rec.q_index]})
            assignment[rec.q_index] = len(clusters) - 1
        else:
            clusters[best_idx]["members"].append(rec.q_index)
            assignment[rec.q_index] = best_idx

    # Update medoid once.
    for idx, c in enumerate(clusters):
        members: List[int] = list(c["members"])
        best_q = members[0]
        best_avg = -1.0
        for q in members:
            sims = [jaccard(sets[q], sets[other]) for other in members]
            avg = float(np.mean(sims))
            if avg > best_avg:
                best_avg = avg
                best_q = q
        c["medoid"] = best_q
        for q in members:
            assignment[q] = idx

    # Assign zero-weight samples to nearest cluster by node Jaccard.
    if clusters:
        medoid_sets = {idx: sets[int(c["medoid"])] for idx, c in enumerate(clusters)}
        for rec in records:
            if rec.q_index in assignment:
                continue
            rec_set = set(rec.nodes)
            sims = [(idx, jaccard(rec_set, medoid_sets[idx])) for idx in medoid_sets]
            best_idx, _ = max(sims, key=lambda x: x[1])
            assignment[rec.q_index] = best_idx
    else:
        for rec in records:
            assignment[rec.q_index] = 0

    return assignment


def sample_random_nodes(universe: Sequence[str], k: int, rng: random.Random) -> List[str]:
    if k <= 0:
        return []
    if k >= len(universe):
        return list(universe)
    return rng.sample(list(universe), k)


def replay_global_circuit(
    records: Sequence[SampleRecord],
    global_nodes: Sequence[str],
    model_path: str,
    device: str,
    n_random: int,
    seed: int,
) -> Dict[str, object]:
    if not global_nodes:
        return {"ran_samples": 0, "note": "empty global nodes"}

    model, tokenizer = load_hooked_qwen3(model_path, device=device, dtype=torch.bfloat16)
    _ = tokenizer  # tokenizer not needed directly, kept for symmetry

    rng = random.Random(seed)
    universe = sorted({n for r in records for n in r.nodes}, key=lambda n: (node_layer(n), n))
    k = len(global_nodes)

    per_sample = []
    for rec in tqdm(records, desc="Replay global core", dynamic_ncols=True):
        summary = rec.summary
        clean_prompt = Path(str(summary.get("clean_prompt")))
        corrupt_prompt = Path(str(summary.get("corrupt_prompt")))
        target = int(summary.get("target_token_id"))
        distractor = int(summary.get("distractor_token_id"))
        clean_obj = safe_float(summary.get("clean_obj"), float("nan"))
        corrupt_obj = safe_float(summary.get("corrupt_obj"), float("nan"))
        gap = safe_float(summary.get("gap"), float("nan"))

        if not clean_prompt.exists() or not corrupt_prompt.exists() or not math.isfinite(gap) or abs(gap) <= 1e-8:
            continue

        try:
            clean_text = clean_prompt.read_text(encoding="utf-8")
            corrupt_text = corrupt_prompt.read_text(encoding="utf-8")
            clean_tokens = model.to_tokens(clean_text, prepend_bos=False)
            corrupt_tokens = model.to_tokens(corrupt_text, prepend_bos=False)
            if clean_tokens.shape != corrupt_tokens.shape:
                continue

            clean_cache = collect_clean_cache_cpu(model, clean_tokens)
            corrupt_cache = collect_clean_cache_cpu(model, corrupt_tokens)

            global_obj = evaluate_on_base_with_source(
                model=model,
                base_tokens=corrupt_tokens,
                source_cache_cpu=clean_cache,
                patch_nodes=global_nodes,
                target_token=target,
                distractor_token=distractor,
            )
            global_ratio = (global_obj - corrupt_obj) / gap

            clean_with_global_corrupt = evaluate_on_base_with_source(
                model=model,
                base_tokens=clean_tokens,
                source_cache_cpu=corrupt_cache,
                patch_nodes=global_nodes,
                target_token=target,
                distractor_token=distractor,
            )
            global_nec_ratio = (clean_obj - clean_with_global_corrupt) / gap

            rand_ratios = []
            for _ in range(max(0, n_random)):
                rnd_nodes = sample_random_nodes(universe=universe, k=k, rng=rng)
                rnd_obj = evaluate_on_base_with_source(
                    model=model,
                    base_tokens=corrupt_tokens,
                    source_cache_cpu=clean_cache,
                    patch_nodes=rnd_nodes,
                    target_token=target,
                    distractor_token=distractor,
                )
                rand_ratios.append((rnd_obj - corrupt_obj) / gap)

            per_sample.append(
                {
                    "q_index": rec.q_index,
                    "weight": rec.weight,
                    "global_suff_ratio": global_ratio,
                    "global_nec_ratio": global_nec_ratio,
                    "random_suff_ratio_mean": float(np.mean(rand_ratios)) if rand_ratios else float("nan"),
                    "random_suff_ratio_std": float(np.std(rand_ratios)) if rand_ratios else float("nan"),
                }
            )
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
        finally:
            gc.collect()
            torch.cuda.empty_cache()
            model.reset_hooks()

    def finite_vals(key: str) -> List[float]:
        vals = [float(x[key]) for x in per_sample if is_finite_num(x.get(key))]
        return [v for v in vals if math.isfinite(v)]

    suff_vals = finite_vals("global_suff_ratio")
    nec_vals = finite_vals("global_nec_ratio")
    rand_vals = finite_vals("random_suff_ratio_mean")
    effect_vals = [a - b for a, b in zip(suff_vals, rand_vals)] if rand_vals else []

    replay = {
        "ran_samples": len(per_sample),
        "n_random": n_random,
        "global_suff_ratio_median": float(np.median(suff_vals)) if suff_vals else float("nan"),
        "global_suff_ratio_mean": float(np.mean(suff_vals)) if suff_vals else float("nan"),
        "global_nec_ratio_median": float(np.median(nec_vals)) if nec_vals else float("nan"),
        "global_nec_ratio_mean": float(np.mean(nec_vals)) if nec_vals else float("nan"),
        "random_suff_ratio_mean_median": float(np.median(rand_vals)) if rand_vals else float("nan"),
        "random_suff_ratio_mean_mean": float(np.mean(rand_vals)) if rand_vals else float("nan"),
        "global_minus_random_median": float(np.median(effect_vals)) if effect_vals else float("nan"),
        "global_minus_random_mean": float(np.mean(effect_vals)) if effect_vals else float("nan"),
        "per_sample": per_sample,
    }
    return replay


def write_cluster_assignments(
    out_path: Path,
    records: Sequence[SampleRecord],
    assignment: Dict[int, int],
) -> None:
    counts: Dict[int, int] = defaultdict(int)
    for c in assignment.values():
        counts[c] += 1
    rows = []
    for rec in records:
        c = assignment.get(rec.q_index, 0)
        rows.append(
            {
                "q_index": rec.q_index,
                "cluster_id": c,
                "cluster_size": counts[c],
                "weight": rec.weight,
            }
        )
    rows.sort(key=lambda x: (x["cluster_id"], x["q_index"]))

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["q_index", "cluster_id", "cluster_size", "weight"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate cross-sample tool-call circuits.")
    parser.add_argument(
        "--input-root",
        type=str,
        default="experiments/results/toolcall_q1_q164",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="experiments/results/toolcall_q1_q164_aggregate",
    )
    parser.add_argument("--model-path", type=str, default="/root/data/Qwen/Qwen3-1.7B")
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--gap-min", type=float, default=0.5)
    parser.add_argument("--ap-discount", type=float, default=0.7)

    parser.add_argument("--core-node-th", type=float, default=0.50)
    parser.add_argument("--core-edge-th", type=float, default=0.35)
    parser.add_argument("--relaxed-node-th", type=float, default=0.25)
    parser.add_argument("--relaxed-edge-th", type=float, default=0.15)
    parser.add_argument("--min-nodes", type=int, default=8)
    parser.add_argument("--max-core-nodes", type=int, default=18)
    parser.add_argument("--max-relaxed-nodes", type=int, default=26)

    parser.add_argument("--cluster-sim-th", type=float, default=0.45)
    parser.add_argument("--cluster-min-size", type=int, default=8)
    parser.add_argument("--cluster-max-plots", type=int, default=8)

    parser.add_argument("--replay-random", type=int, default=2)
    parser.add_argument("--skip-replay", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_root = Path(args.input_root).resolve()
    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    records = load_sample_records(
        root=input_root,
        gap_min=args.gap_min,
        ap_discount=args.ap_discount,
    )
    if not records:
        raise ValueError(f"No per-sample summaries found in {input_root}")

    node_support, edge_support, node_cnt, edge_cnt, node_norm_scores, total_w = aggregate_supports(records)
    n_samples = len(records)

    write_node_table(
        out_path=out_root / "node_support.csv",
        node_support=node_support,
        node_cnt=node_cnt,
        node_norm_scores=node_norm_scores,
        n_samples=n_samples,
    )
    write_edge_table(
        out_path=out_root / "edge_support.csv",
        edge_support=edge_support,
        edge_cnt=edge_cnt,
        n_samples=n_samples,
    )

    sample_rows = [
        {
            "q_index": r.q_index,
            "weight": r.weight,
            "gap": safe_float(r.summary.get("gap"), float("nan")),
            "detailed_ratio_vs_gap": safe_float(r.summary.get("detailed_ratio_vs_gap"), float("nan")),
            "necessity_ratio_vs_gap": safe_float(r.summary.get("necessity_ratio_vs_gap"), float("nan")),
            "ap_mode": str(r.summary.get("ap_mode", "full")),
        }
        for r in records
    ]
    with (out_root / "sample_weights.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "q_index",
                "weight",
                "gap",
                "detailed_ratio_vs_gap",
                "necessity_ratio_vs_gap",
                "ap_mode",
            ],
        )
        writer.writeheader()
        writer.writerows(sample_rows)

    core_nodes = pick_consensus_nodes(
        node_support=node_support,
        node_threshold=args.core_node_th,
        min_nodes=args.min_nodes,
        max_nodes=args.max_core_nodes,
    )
    core_edges = pick_consensus_edges(
        nodes=core_nodes,
        edge_support=edge_support,
        node_support=node_support,
        edge_threshold=args.core_edge_th,
        min_edges=max(10, len(core_nodes)),
    )
    draw_circuit(
        nodes=core_nodes,
        edges=core_edges,
        out_path=out_root / "final_circuit_global_core.png",
        title="Global Core Circuit (Cross-sample Consensus)",
    )

    relaxed_nodes = pick_consensus_nodes(
        node_support=node_support,
        node_threshold=args.relaxed_node_th,
        min_nodes=args.min_nodes + 2,
        max_nodes=args.max_relaxed_nodes,
    )
    relaxed_edges = pick_consensus_edges(
        nodes=relaxed_nodes,
        edge_support=edge_support,
        node_support=node_support,
        edge_threshold=args.relaxed_edge_th,
        min_edges=max(14, len(relaxed_nodes)),
    )
    draw_circuit(
        nodes=relaxed_nodes,
        edges=relaxed_edges,
        out_path=out_root / "final_circuit_global_relaxed.png",
        title="Global Relaxed Circuit (Cross-sample Consensus)",
    )

    assignment = cluster_samples(records=records, sim_threshold=args.cluster_sim_th)
    write_cluster_assignments(
        out_path=out_root / "cluster_assignments.csv",
        records=records,
        assignment=assignment,
    )

    # Cluster-level core circuits.
    cluster_to_q: Dict[int, List[int]] = defaultdict(list)
    for q, c in assignment.items():
        cluster_to_q[c].append(q)
    cluster_sorted = sorted(cluster_to_q.items(), key=lambda x: len(x[1]), reverse=True)

    cluster_summaries = []
    plotted = 0
    for c_id, q_list in cluster_sorted:
        members = [r for r in records if r.q_index in set(q_list)]
        if len(members) < args.cluster_min_size:
            continue
        ns, es, _, _, _, tw = aggregate_supports(members)
        c_nodes = pick_consensus_nodes(
            node_support=ns,
            node_threshold=max(0.35, args.core_node_th - 0.1),
            min_nodes=min(args.min_nodes, 6),
            max_nodes=14,
        )
        c_edges = pick_consensus_edges(
            nodes=c_nodes,
            edge_support=es,
            node_support=ns,
            edge_threshold=max(0.20, args.core_edge_th - 0.12),
            min_edges=max(8, len(c_nodes)),
        )

        cluster_summaries.append(
            {
                "cluster_id": c_id,
                "size": len(members),
                "total_weight": tw,
                "nodes": c_nodes,
                "edges": c_edges,
            }
        )

        if plotted < args.cluster_max_plots:
            draw_circuit(
                nodes=c_nodes,
                edges=c_edges,
                out_path=out_root / f"final_circuit_cluster_{c_id:02d}.png",
                title=f"Cluster {c_id} Core Circuit (n={len(members)})",
            )
            plotted += 1

    replay: Dict[str, object]
    should_replay = not args.skip_replay
    if should_replay and args.device.startswith("cuda") and not torch.cuda.is_available():
        should_replay = False
        replay = {
            "ran_samples": 0,
            "note": "Skipped replay because CUDA device is unavailable in current session.",
        }
    elif should_replay:
        replay = replay_global_circuit(
            records=records,
            global_nodes=core_nodes,
            model_path=args.model_path,
            device=args.device,
            n_random=args.replay_random,
            seed=args.seed,
        )
    else:
        replay = {"ran_samples": 0, "note": "Replay skipped by --skip-replay."}

    (out_root / "global_core_replay.json").write_text(
        json.dumps(replay, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = {
        "input_root": str(input_root),
        "n_samples": n_samples,
        "total_weight": total_w,
        "weight_params": {"gap_min": args.gap_min, "ap_discount": args.ap_discount},
        "core_thresholds": {"node": args.core_node_th, "edge": args.core_edge_th},
        "relaxed_thresholds": {"node": args.relaxed_node_th, "edge": args.relaxed_edge_th},
        "core_nodes": core_nodes,
        "core_edges": core_edges,
        "relaxed_nodes": relaxed_nodes,
        "relaxed_edges": relaxed_edges,
        "n_clusters": len(cluster_to_q),
        "cluster_sizes": {str(k): len(v) for k, v in cluster_to_q.items()},
        "cluster_summaries": cluster_summaries,
        "replay_summary": {
            "ran_samples": replay.get("ran_samples"),
            "global_suff_ratio_median": replay.get("global_suff_ratio_median"),
            "global_nec_ratio_median": replay.get("global_nec_ratio_median"),
            "random_suff_ratio_mean_median": replay.get("random_suff_ratio_mean_median"),
            "global_minus_random_median": replay.get("global_minus_random_median"),
        },
        "artifacts": {
            "node_support_csv": str(out_root / "node_support.csv"),
            "edge_support_csv": str(out_root / "edge_support.csv"),
            "sample_weights_csv": str(out_root / "sample_weights.csv"),
            "cluster_assignments_csv": str(out_root / "cluster_assignments.csv"),
            "global_core_png": str(out_root / "final_circuit_global_core.png"),
            "global_relaxed_png": str(out_root / "final_circuit_global_relaxed.png"),
            "global_replay_json": str(out_root / "global_core_replay.json"),
        },
    }
    (out_root / "global_core_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[done] aggregate outputs: {out_root}")
    print(f"[done] core nodes={len(core_nodes)} edges={len(core_edges)}")
    print(f"[done] replay samples={replay.get('ran_samples', 0)}")


if __name__ == "__main__":
    main()

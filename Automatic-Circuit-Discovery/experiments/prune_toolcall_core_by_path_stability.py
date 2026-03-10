#!/usr/bin/env python3
"""
Prune consensus core edges by path-patching stability metrics.

This script:
1) filters aggregate core edges by path-patch stability thresholds;
2) optionally enforces Input->Output path connectivity (which can remove nodes);
3) optionally reruns replay metrics on the pruned node set;
4) writes pruned aggregate/semantic reports plus an edge decision table.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from experiments.launch_toolcall_qwen3_q85 import draw_circuit

INPUT_NODE = "Input Embed"
OUTPUT_NODE = "Residual Output: <tool_call>"


def edge_label(src: str, dst: str) -> str:
    return f"{src}->{dst}"


def dfs(start: str, adj: Dict[str, List[str]]) -> Set[str]:
    stack = [start]
    seen: Set[str] = {start}
    while stack:
        cur = stack.pop()
        for nxt in adj.get(cur, []):
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen


def build_adjacency(edges: Sequence[Tuple[str, str]]) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    fwd: Dict[str, List[str]] = defaultdict(list)
    rev: Dict[str, List[str]] = defaultdict(list)
    for src, dst in edges:
        fwd[src].append(dst)
        rev[dst].append(src)
    return fwd, rev


def keep_edges_by_threshold(
    core_edges: Sequence[Tuple[str, str]],
    path_rows: Sequence[Dict[str, object]],
    edge_ratio_min: float,
    positive_frac_min: float,
) -> Tuple[List[Tuple[str, str]], List[Dict[str, object]]]:
    stats: Dict[str, Dict[str, object]] = {}
    for row in path_rows:
        key = str(row.get("edge", ""))
        if key:
            stats[key] = row

    kept: List[Tuple[str, str]] = []
    decisions: List[Dict[str, object]] = []
    for src, dst in core_edges:
        key = edge_label(src, dst)
        row = stats.get(key)
        ratio = float(row.get("edge_ratio_median")) if row else float("nan")
        pos = float(row.get("positive_frac")) if row else float("nan")
        src_ratio = float(row.get("source_ratio_median")) if row else float("nan")
        blk_ratio = float(row.get("blocked_ratio_median")) if row else float("nan")

        keep = bool(
            row is not None
            and ratio > edge_ratio_min
            and pos >= positive_frac_min
        )
        reason = ""
        if row is None:
            reason = "missing_path_metric"
        elif ratio <= edge_ratio_min:
            reason = f"edge_ratio<= {edge_ratio_min:.4f}"
        elif pos < positive_frac_min:
            reason = f"positive_frac< {positive_frac_min:.4f}"

        if keep:
            kept.append((src, dst))

        decisions.append(
            {
                "edge": key,
                "src": src,
                "dst": dst,
                "edge_ratio_median": ratio,
                "positive_frac": pos,
                "source_ratio_median": src_ratio,
                "blocked_ratio_median": blk_ratio,
                "keep_after_threshold": int(keep),
                "drop_reason": reason,
                "keep_final": int(keep),  # may be overwritten by path constraint
            }
        )
    return kept, decisions


def enforce_input_output_paths(
    core_nodes: Sequence[str],
    candidate_edges: Sequence[Tuple[str, str]],
) -> Tuple[List[str], List[Tuple[str, str]], Set[str], Set[str]]:
    fwd, rev = build_adjacency(candidate_edges)
    reachable_from_input = dfs(INPUT_NODE, fwd)
    can_reach_output = dfs(OUTPUT_NODE, rev)

    keep_node_set = {
        n for n in core_nodes if n in reachable_from_input and n in can_reach_output
    }
    pruned_nodes = [n for n in core_nodes if n in keep_node_set]

    pruned_edges: List[Tuple[str, str]] = []
    for src, dst in candidate_edges:
        if src == INPUT_NODE and dst in keep_node_set:
            pruned_edges.append((src, dst))
        elif src in keep_node_set and dst in keep_node_set:
            pruned_edges.append((src, dst))
        elif src in keep_node_set and dst == OUTPUT_NODE:
            pruned_edges.append((src, dst))

    return pruned_nodes, pruned_edges, reachable_from_input, can_reach_output


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def maybe_run_replay(
    *,
    input_root: Path,
    pruned_nodes: Sequence[str],
    model_path: str,
    device: str,
    gap_min: float,
    ap_discount: float,
    n_random: int,
    seed: int,
) -> Dict[str, object]:
    from experiments.aggregate_toolcall_circuits import load_sample_records, replay_global_circuit

    records = load_sample_records(root=input_root, gap_min=gap_min, ap_discount=ap_discount)
    replay = replay_global_circuit(
        records=records,
        global_nodes=pruned_nodes,
        model_path=model_path,
        device=device,
        n_random=n_random,
        seed=seed,
    )
    return replay


def main() -> None:
    parser = argparse.ArgumentParser(description="Prune tool-call core by path-patch stability.")
    parser.add_argument("--aggregate-summary", type=str, required=True)
    parser.add_argument("--path-patch-report", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--semantic-report", type=str, default="")
    parser.add_argument("--edge-ratio-min", type=float, default=0.03)
    parser.add_argument("--positive-frac-min", type=float, default=0.90)
    parser.add_argument(
        "--path-metrics",
        type=str,
        choices=["trimmed", "full"],
        default="trimmed",
        help="Use edge_summary_trimmed or edge_summary_full from path-patch report.",
    )
    parser.add_argument(
        "--enforce-input-output-path",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep only nodes on Input->...->Output paths after threshold filtering.",
    )
    parser.add_argument("--run-replay", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--input-root", type=str, default="experiments/results/toolcall_q1_q164")
    parser.add_argument("--model-path", type=str, default="/root/data/Qwen/Qwen3-1.7B")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gap-min", type=float, default=0.5)
    parser.add_argument("--ap-discount", type=float, default=0.7)
    parser.add_argument("--replay-random", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    agg_path = Path(args.aggregate_summary).resolve()
    ppr_path = Path(args.path_patch_report).resolve()
    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    agg = json.loads(agg_path.read_text(encoding="utf-8"))
    ppr = json.loads(ppr_path.read_text(encoding="utf-8"))

    core_nodes = [str(x) for x in agg.get("core_nodes", [])]
    core_edges = [tuple(e) for e in agg.get("core_edges", [])]
    if not core_nodes or not core_edges:
        raise ValueError("aggregate summary must contain non-empty core_nodes/core_edges.")

    rows_key = "edge_summary_trimmed" if args.path_metrics == "trimmed" else "edge_summary_full"
    path_rows = ppr.get(rows_key, [])
    if not path_rows:
        raise ValueError(f"path patch report has no `{rows_key}` rows.")

    kept_edges_thresh, decisions = keep_edges_by_threshold(
        core_edges=core_edges,
        path_rows=path_rows,
        edge_ratio_min=args.edge_ratio_min,
        positive_frac_min=args.positive_frac_min,
    )

    if args.enforce_input_output_path:
        pruned_nodes, pruned_edges, reach_from_input, reach_to_output = enforce_input_output_paths(
            core_nodes=core_nodes,
            candidate_edges=kept_edges_thresh,
        )
    else:
        nodes_from_edges = {
            n
            for src, dst in kept_edges_thresh
            for n in (src, dst)
            if n not in {INPUT_NODE, OUTPUT_NODE}
        }
        pruned_nodes = [n for n in core_nodes if n in nodes_from_edges]
        pruned_set = set(pruned_nodes)
        pruned_edges = [
            (src, dst)
            for src, dst in kept_edges_thresh
            if (src == INPUT_NODE and dst in pruned_set)
            or (src in pruned_set and dst in pruned_set)
            or (src in pruned_set and dst == OUTPUT_NODE)
        ]
        reach_from_input = set()
        reach_to_output = set()

    pruned_edge_set = set(pruned_edges)
    for d in decisions:
        src = str(d["src"])
        dst = str(d["dst"])
        d["keep_final"] = int((src, dst) in pruned_edge_set)
        if int(d["keep_after_threshold"]) and not d["keep_final"]:
            d["drop_reason"] = "dropped_by_input_output_path"

    if not pruned_nodes:
        raise ValueError("Pruning removed all nodes. Relax thresholds or disable path enforcement.")
    if not pruned_edges:
        raise ValueError("Pruning removed all edges. Relax thresholds.")

    draw_circuit(
        nodes=pruned_nodes,
        edges=pruned_edges,
        out_path=out_root / "final_circuit_global_core_pruned.png",
        title="Global Core Circuit (Path-Stability Pruned)",
    )

    replay: Dict[str, object] = {"ran_samples": 0, "note": "Replay not requested."}
    if args.run_replay:
        replay = maybe_run_replay(
            input_root=Path(args.input_root).resolve(),
            pruned_nodes=pruned_nodes,
            model_path=args.model_path,
            device=args.device,
            gap_min=args.gap_min,
            ap_discount=args.ap_discount,
            n_random=args.replay_random,
            seed=args.seed,
        )
        (out_root / "global_core_replay_pruned.json").write_text(
            json.dumps(replay, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    pruned_summary = dict(agg)
    pruned_summary["core_nodes"] = pruned_nodes
    pruned_summary["core_edges"] = pruned_edges
    if args.run_replay:
        pruned_summary["replay_summary"] = {
            "ran_samples": replay.get("ran_samples"),
            "global_suff_ratio_median": replay.get("global_suff_ratio_median"),
            "global_nec_ratio_median": replay.get("global_nec_ratio_median"),
            "random_suff_ratio_mean_median": replay.get("random_suff_ratio_mean_median"),
            "global_minus_random_median": replay.get("global_minus_random_median"),
        }

    pruned_summary["path_stability_pruning"] = {
        "aggregate_summary_path": str(agg_path),
        "path_patch_report_path": str(ppr_path),
        "path_metrics_source": rows_key,
        "edge_ratio_min": args.edge_ratio_min,
        "positive_frac_min": args.positive_frac_min,
        "enforce_input_output_path": bool(args.enforce_input_output_path),
        "core_nodes_before": core_nodes,
        "core_nodes_after": pruned_nodes,
        "core_edges_before": core_edges,
        "core_edges_after": pruned_edges,
        "dropped_nodes": [n for n in core_nodes if n not in set(pruned_nodes)],
        "dropped_edges": [[s, t] for (s, t) in core_edges if (s, t) not in set(pruned_edges)],
        "reachability": {
            "reachable_from_input": sorted(reach_from_input),
            "can_reach_output": sorted(reach_to_output),
        },
    }
    (out_root / "global_core_summary_pruned.json").write_text(
        json.dumps(pruned_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    decision_fields = [
        "edge",
        "src",
        "dst",
        "edge_ratio_median",
        "positive_frac",
        "source_ratio_median",
        "blocked_ratio_median",
        "keep_after_threshold",
        "keep_final",
        "drop_reason",
    ]
    write_csv(out_root / "path_stability_edge_decisions.csv", decisions, decision_fields)

    if args.semantic_report.strip():
        sem_path = Path(args.semantic_report).resolve()
        sem = json.loads(sem_path.read_text(encoding="utf-8"))
        sem["core_nodes"] = pruned_nodes
        sem["core_nodes_source"] = str(sem_path)
        sem["path_stability_pruning"] = pruned_summary["path_stability_pruning"]
        (out_root / "semantic_roles_report_pruned.json").write_text(
            json.dumps(sem, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(f"[done] wrote pruned summary: {out_root / 'global_core_summary_pruned.json'}")
    print(f"[done] core nodes: {len(core_nodes)} -> {len(pruned_nodes)}")
    print(f"[done] core edges: {len(core_edges)} -> {len(pruned_edges)}")


if __name__ == "__main__":
    main()

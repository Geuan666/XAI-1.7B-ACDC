#!/usr/bin/env python3
"""
Batch circuit mining for tool-call behavior on all q* prompt pairs.

This script reuses the same ACDC-inspired workflow as launch_toolcall_qwen3_q85.py,
but runs it for many samples in one process (model loaded once on GPU).
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import traceback
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from tqdm.auto import tqdm

from experiments.launch_toolcall_qwen3_q85 import (
    assert_no_dead_ends,
    build_edges,
    collect_clean_cache_cpu,
    compute_ap_head_gain,
    compute_ap_head_gain_lowmem,
    compute_ct_head_gain,
    compute_mlp_gain,
    draw_circuit,
    evaluate_on_base_with_source,
    load_hooked_qwen3,
    objective_from_logits,
    pick_nodes,
    plot_head_heatmap,
    plot_probe,
)


def discover_q_indices(pair_dir: Path) -> List[int]:
    clean = {int(p.stem.split("-q")[-1]) for p in pair_dir.glob("prompt-clean-q*.txt")}
    corrupt = {int(p.stem.split("-q")[-1]) for p in pair_dir.glob("prompt-corrupted-q*.txt")}
    return sorted(clean & corrupt)


def parse_q_list(q_list_raw: str) -> List[int]:
    out: List[int] = []
    for chunk in q_list_raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo_s, hi_s = chunk.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            if lo > hi:
                lo, hi = hi, lo
            out.extend(list(range(lo, hi + 1)))
        else:
            out.append(int(chunk))
    return sorted(set(out))


def save_summary_csv(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    if not rows:
        return
    fieldnames = [
        "q_index",
        "status",
        "clean_obj",
        "corrupt_obj",
        "gap",
        "detailed_ratio_vs_gap",
        "necessity_ratio_vs_gap",
        "rough_ratio_vs_gap",
        "rough_necessity_ratio_vs_gap",
        "probe_head",
        "target_token_str",
        "distractor_token_str",
        "error",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def run_one_q(
    q_index: int,
    pair_dir: Path,
    out_dir: Path,
    model,
    tokenizer,
) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)

    clean_path = pair_dir / f"prompt-clean-q{q_index}.txt"
    corrupt_path = pair_dir / f"prompt-corrupted-q{q_index}.txt"
    if not clean_path.exists() or not corrupt_path.exists():
        raise FileNotFoundError(f"Missing prompt files: {clean_path} / {corrupt_path}")

    clean_text = clean_path.read_text(encoding="utf-8")
    corrupt_text = corrupt_path.read_text(encoding="utf-8")

    clean_tokens = model.to_tokens(clean_text, prepend_bos=False)
    corrupt_tokens = model.to_tokens(corrupt_text, prepend_bos=False)
    if clean_tokens.shape != corrupt_tokens.shape:
        raise ValueError(f"Clean/corrupt token shapes differ: {clean_tokens.shape} vs {corrupt_tokens.shape}")

    with torch.no_grad():
        clean_logits = model(clean_tokens)
        corrupt_logits = model(corrupt_tokens)

    target_token = tokenizer.encode("<tool_call>", add_special_tokens=False)[0]
    distractor_token = int(torch.argmax(corrupt_logits[0, -1]).item())

    clean_obj = float(objective_from_logits(clean_logits, target_token, distractor_token).item())
    corrupt_obj = float(objective_from_logits(corrupt_logits, target_token, distractor_token).item())
    gap = clean_obj - corrupt_obj

    clean_cache_cpu = collect_clean_cache_cpu(model, clean_tokens)

    ct_head = compute_ct_head_gain(
        model=model,
        corrupt_tokens=corrupt_tokens,
        clean_cache_cpu=clean_cache_cpu,
        target_token=target_token,
        distractor_token=distractor_token,
        corrupt_obj=corrupt_obj,
    )

    mlp_gain = compute_mlp_gain(
        model=model,
        corrupt_tokens=corrupt_tokens,
        clean_cache_cpu=clean_cache_cpu,
        target_token=target_token,
        distractor_token=distractor_token,
        corrupt_obj=corrupt_obj,
    )

    ap_mode = "full"
    try:
        ap_head = compute_ap_head_gain(
            model=model,
            corrupt_tokens=corrupt_tokens,
            clean_cache_cpu=clean_cache_cpu,
            target_token=target_token,
            distractor_token=distractor_token,
        )
    except torch.OutOfMemoryError:
        gc.collect()
        torch.cuda.empty_cache()
        ap_mode = "lowmem"
        try:
            ap_head = compute_ap_head_gain_lowmem(
                model=model,
                corrupt_tokens=corrupt_tokens,
                clean_cache_cpu=clean_cache_cpu,
                target_token=target_token,
                distractor_token=distractor_token,
            )
        except torch.OutOfMemoryError:
            gc.collect()
            torch.cuda.empty_cache()
            ap_mode = "zero_fallback"
            ap_head = torch.zeros_like(ct_head)
    model.reset_hooks()

    detailed_nodes, rough_nodes, score_table = pick_nodes(ct_head, ap_head, mlp_gain)
    score_lookup = {s.name: s.score for s in score_table}
    detailed_edges = build_edges(detailed_nodes, score_lookup=score_lookup, max_parents=2)
    rough_edges = build_edges(rough_nodes, score_lookup=score_lookup, max_parents=2)
    assert_no_dead_ends(detailed_nodes, detailed_edges)
    assert_no_dead_ends(rough_nodes, rough_edges)

    detailed_obj = evaluate_on_base_with_source(
        model=model,
        base_tokens=corrupt_tokens,
        source_cache_cpu=clean_cache_cpu,
        patch_nodes=detailed_nodes,
        target_token=target_token,
        distractor_token=distractor_token,
    )
    detailed_ratio = (detailed_obj - corrupt_obj) / gap if abs(gap) > 1e-8 else float("nan")

    rough_obj = evaluate_on_base_with_source(
        model=model,
        base_tokens=corrupt_tokens,
        source_cache_cpu=clean_cache_cpu,
        patch_nodes=rough_nodes,
        target_token=target_token,
        distractor_token=distractor_token,
    )
    rough_ratio = (rough_obj - corrupt_obj) / gap if abs(gap) > 1e-8 else float("nan")

    corrupt_cache_cpu = collect_clean_cache_cpu(model, corrupt_tokens)
    clean_with_detailed_corrupted = evaluate_on_base_with_source(
        model=model,
        base_tokens=clean_tokens,
        source_cache_cpu=corrupt_cache_cpu,
        patch_nodes=detailed_nodes,
        target_token=target_token,
        distractor_token=distractor_token,
    )
    necessity_drop = clean_obj - clean_with_detailed_corrupted
    necessity_ratio = necessity_drop / gap if abs(gap) > 1e-8 else float("nan")

    clean_with_rough_corrupted = evaluate_on_base_with_source(
        model=model,
        base_tokens=clean_tokens,
        source_cache_cpu=corrupt_cache_cpu,
        patch_nodes=rough_nodes,
        target_token=target_token,
        distractor_token=distractor_token,
    )
    rough_necessity_drop = clean_obj - clean_with_rough_corrupted
    rough_necessity_ratio = rough_necessity_drop / gap if abs(gap) > 1e-8 else float("nan")

    head_scores = [s for s in score_table if s.name.startswith("L")]
    probe_head = head_scores[0].name if head_scores else "L0H0"

    corrupt_with_probe = evaluate_on_base_with_source(
        model=model,
        base_tokens=corrupt_tokens,
        source_cache_cpu=clean_cache_cpu,
        patch_nodes=[probe_head],
        target_token=target_token,
        distractor_token=distractor_token,
    )
    clean_with_probe_corrupt = evaluate_on_base_with_source(
        model=model,
        base_tokens=clean_tokens,
        source_cache_cpu=corrupt_cache_cpu,
        patch_nodes=[probe_head],
        target_token=target_token,
        distractor_token=distractor_token,
    )

    plot_head_heatmap(
        ap_head.numpy(),
        "Attribution Patching Head Heatmap (AP)",
        out_dir / "ap_head_heatmap.png",
    )
    plot_head_heatmap(
        ct_head.numpy(),
        "Causal Tracing Head Heatmap (CT)",
        out_dir / "ct_head_heatmap.png",
    )
    plot_probe(
        out_path=out_dir / f"{probe_head}_probe.png",
        head_name=probe_head,
        corrupt_obj=corrupt_obj,
        corrupt_with_head=corrupt_with_probe,
        clean_obj=clean_obj,
        clean_with_head_corrupt=clean_with_probe_corrupt,
    )
    draw_circuit(
        nodes=detailed_nodes,
        edges=detailed_edges,
        out_path=out_dir / "final_circuit_detailed.png",
        title="Detailed Circuit (Tool-call Decision)",
    )
    draw_circuit(
        nodes=rough_nodes,
        edges=rough_edges,
        out_path=out_dir / "final_circuit.png",
        title="Simplified Circuit (Tool-call Decision)",
    )

    summary = {
        "q_index": q_index,
        "clean_prompt": str(clean_path),
        "corrupt_prompt": str(corrupt_path),
        "target_token_id": target_token,
        "target_token_str": tokenizer.decode([target_token]),
        "distractor_token_id": distractor_token,
        "distractor_token_str": tokenizer.decode([distractor_token]),
        "clean_obj": clean_obj,
        "corrupt_obj": corrupt_obj,
        "gap": gap,
        "detailed_obj": detailed_obj,
        "detailed_ratio_vs_gap": detailed_ratio,
        "rough_obj": rough_obj,
        "rough_ratio_vs_gap": rough_ratio,
        "clean_with_detailed_corrupted": clean_with_detailed_corrupted,
        "necessity_drop": necessity_drop,
        "necessity_ratio_vs_gap": necessity_ratio,
        "clean_with_rough_corrupted": clean_with_rough_corrupted,
        "rough_necessity_drop": rough_necessity_drop,
        "rough_necessity_ratio_vs_gap": rough_necessity_ratio,
        "probe_head": probe_head,
        "ap_mode": ap_mode,
        "probe_corrupt_with_head": corrupt_with_probe,
        "probe_clean_with_head_corrupt": clean_with_probe_corrupt,
        "detailed_nodes": detailed_nodes,
        "detailed_edges": detailed_edges,
        "rough_nodes": rough_nodes,
        "rough_edges": rough_edges,
        "top_node_scores": [
            {"name": s.name, "score": s.score, "ct": s.ct, "ap": s.ap}
            for s in score_table
        ],
        "artifacts": {
            "ap_head_heatmap": str(out_dir / "ap_head_heatmap.png"),
            "ct_head_heatmap": str(out_dir / "ct_head_heatmap.png"),
            "probe": str(out_dir / f"{probe_head}_probe.png"),
            "final_circuit_detailed": str(out_dir / "final_circuit_detailed.png"),
            "final_circuit": str(out_dir / "final_circuit.png"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch mine tool-call circuits on Qwen3.")
    parser.add_argument("--pair-dir", type=str, default="/root/data/XAI-1.7B-ACDC/pair")
    parser.add_argument("--model-path", type=str, default="/root/data/Qwen/Qwen3-1.7B")
    parser.add_argument("--out-root", type=str, default="experiments/results/toolcall_q1_q164")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--q-list", type=str, default="")
    parser.add_argument("--q-min", type=int, default=1)
    parser.add_argument("--q-max", type=int, default=164)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()

    pair_dir = Path(args.pair_dir).resolve()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    discovered = discover_q_indices(pair_dir)
    if args.q_list.strip():
        targets = parse_q_list(args.q_list)
    else:
        targets = [q for q in discovered if args.q_min <= q <= args.q_max]
    targets = [q for q in targets if q in discovered]
    if not targets:
        raise ValueError("No valid q indices to run.")

    model, tokenizer = load_hooked_qwen3(args.model_path, device=args.device, dtype=torch.bfloat16)

    rows: List[Dict[str, object]] = []
    log_path = out_root / "batch_progress.jsonl"
    pbar = tqdm(targets, desc="Batch q", dynamic_ncols=True)
    for q in pbar:
        out_dir = out_root / f"q{q:03d}"
        summary_path = out_dir / "summary.json"
        if args.resume and summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                row = {
                    "q_index": q,
                    "status": "skipped_resume",
                    "clean_obj": summary.get("clean_obj"),
                    "corrupt_obj": summary.get("corrupt_obj"),
                    "gap": summary.get("gap"),
                    "detailed_ratio_vs_gap": summary.get("detailed_ratio_vs_gap"),
                    "necessity_ratio_vs_gap": summary.get("necessity_ratio_vs_gap"),
                    "rough_ratio_vs_gap": summary.get("rough_ratio_vs_gap"),
                    "rough_necessity_ratio_vs_gap": summary.get("rough_necessity_ratio_vs_gap"),
                    "probe_head": summary.get("probe_head"),
                    "target_token_str": summary.get("target_token_str"),
                    "distractor_token_str": summary.get("distractor_token_str"),
                    "error": "",
                }
            except Exception as ex:
                row = {"q_index": q, "status": "resume_read_failed", "error": repr(ex)}
            rows.append(row)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            continue

        try:
            summary = run_one_q(
                q_index=q,
                pair_dir=pair_dir,
                out_dir=out_dir,
                model=model,
                tokenizer=tokenizer,
            )
            row = {
                "q_index": q,
                "status": "ok",
                "clean_obj": summary["clean_obj"],
                "corrupt_obj": summary["corrupt_obj"],
                "gap": summary["gap"],
                "detailed_ratio_vs_gap": summary["detailed_ratio_vs_gap"],
                "necessity_ratio_vs_gap": summary["necessity_ratio_vs_gap"],
                "rough_ratio_vs_gap": summary["rough_ratio_vs_gap"],
                "rough_necessity_ratio_vs_gap": summary["rough_necessity_ratio_vs_gap"],
                "probe_head": summary["probe_head"],
                "target_token_str": summary["target_token_str"],
                "distractor_token_str": summary["distractor_token_str"],
                "error": "",
            }
            rows.append(row)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            pbar.set_postfix(
                q=q,
                det=f"{summary['detailed_ratio_vs_gap']:.3f}",
                nec=f"{summary['necessity_ratio_vs_gap']:.3f}",
            )
        except Exception as ex:  # noqa: BLE001
            err = traceback.format_exc()
            row = {
                "q_index": q,
                "status": "error",
                "clean_obj": "",
                "corrupt_obj": "",
                "gap": "",
                "detailed_ratio_vs_gap": "",
                "necessity_ratio_vs_gap": "",
                "rough_ratio_vs_gap": "",
                "rough_necessity_ratio_vs_gap": "",
                "probe_head": "",
                "target_token_str": "",
                "distractor_token_str": "",
                "error": repr(ex),
            }
            rows.append(row)
            err_dir = out_root / f"q{q:03d}"
            err_dir.mkdir(parents=True, exist_ok=True)
            (err_dir / "error.txt").write_text(err, encoding="utf-8")
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if args.stop_on_error:
                raise
        finally:
            gc.collect()
            torch.cuda.empty_cache()
            model.reset_hooks()

    save_summary_csv(rows, out_root / "batch_summary.csv")
    (out_root / "batch_summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    ok = sum(1 for r in rows if r.get("status") in {"ok", "skipped_resume"})
    bad = sum(1 for r in rows if r.get("status") not in {"ok", "skipped_resume"})
    print(f"[done] total={len(rows)} ok_or_skipped={ok} error={bad}")
    print(f"[done] outputs root: {out_root}")


if __name__ == "__main__":
    main()

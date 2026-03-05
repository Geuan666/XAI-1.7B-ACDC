#!/usr/bin/env python3
"""
Causal tracing for the clean/corrupt contrast token (Write vs State).

For each sample, patch one residual state (layer l, position p) from clean into corrupt,
and measure objective recovery. Aggregates median recovery curves across samples.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

from experiments.launch_toolcall_qwen3_q85 import load_hooked_qwen3, objective_from_logits


def find_diff_pos(ids_a: Sequence[int], ids_b: Sequence[int]) -> List[int]:
    return [i for i, (x, y) in enumerate(zip(ids_a, ids_b)) if x != y]


def med(xs: Sequence[float]) -> float:
    vals = [float(x) for x in xs if isinstance(x, (int, float)) and math.isfinite(float(x))]
    return float(median(vals)) if vals else float("nan")


def mean(xs: Sequence[float]) -> float:
    vals = [float(x) for x in xs if isinstance(x, (int, float)) and math.isfinite(float(x))]
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
            "axes.titlesize": 16,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace contrast-token causal effect across layers.")
    parser.add_argument("--input-root", type=str, default="experiments/results/toolcall_q1_q164")
    parser.add_argument("--output-root", type=str, default="experiments/results/toolcall_q1_q164_semantic_roles_v2")
    parser.add_argument("--model-path", type=str, default="/root/data/Qwen/Qwen3-1.7B")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gap-min", type=float, default=0.5)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means use all valid samples.")
    parser.add_argument("--q-start", type=int, default=1)
    parser.add_argument("--q-end", type=int, default=10_000)
    parser.add_argument(
        "--recompute-baseline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Recompute clean/corrupt objectives with the current run configuration.",
    )
    args = parser.parse_args()

    input_root = Path(args.input_root).resolve()
    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_hooked_qwen3(args.model_path, device=args.device, dtype=torch.bfloat16)
    model.set_use_attn_result(False)
    model.set_use_split_qkv_input(False)
    model.set_use_hook_mlp_in(False)
    n_layers = int(model.cfg.n_layers)

    target_token = tokenizer("<tool_call>", add_special_tokens=False)["input_ids"][0]
    tool_call_pos = 105  # fixed in this dataset
    prefix_pos = 0
    contrast_pos_counter: Counter[int] = Counter()

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
        if not math.isfinite(gap) or gap <= args.gap_min:
            continue
        clean_prompt = Path(s["clean_prompt"])
        corrupt_prompt = Path(s["corrupt_prompt"])
        if not clean_prompt.exists() or not corrupt_prompt.exists():
            continue
        sample_infos.append((q_index, s, clean_prompt, corrupt_prompt))
    if args.max_samples > 0:
        sample_infos = sample_infos[: args.max_samples]
    if not sample_infos:
        raise ValueError("No valid samples selected.")

    layer_vals_contrast: Dict[int, List[float]] = {l: [] for l in range(n_layers)}
    layer_vals_toolcall: Dict[int, List[float]] = {l: [] for l in range(n_layers)}
    layer_vals_prefix: Dict[int, List[float]] = {l: [] for l in range(n_layers)}
    skipped: List[int] = []

    pbar = tqdm(sample_infos, desc="Contrast trace", dynamic_ncols=True)
    for q_index, summary, clean_prompt, corrupt_prompt in pbar:
        clean_text = clean_prompt.read_text(encoding="utf-8")
        corrupt_text = corrupt_prompt.read_text(encoding="utf-8")
        clean_tokens = model.to_tokens(clean_text, prepend_bos=False)
        corrupt_tokens = model.to_tokens(corrupt_text, prepend_bos=False)
        if clean_tokens.shape != corrupt_tokens.shape:
            skipped.append(q_index)
            continue

        ids_clean = [int(x) for x in clean_tokens[0].tolist()]
        ids_corrupt = [int(x) for x in corrupt_tokens[0].tolist()]
        diff_pos = find_diff_pos(ids_clean, ids_corrupt)
        if len(diff_pos) != 1:
            skipped.append(q_index)
            continue
        contrast_pos = int(diff_pos[0])

        clean_obj = float(summary.get("clean_obj", float("nan")))
        corrupt_obj = float(summary.get("corrupt_obj", float("nan")))
        gap = float(summary.get("gap", clean_obj - corrupt_obj))
        if not (math.isfinite(clean_obj) and math.isfinite(corrupt_obj) and math.isfinite(gap)) or abs(gap) < 1e-8:
            skipped.append(q_index)
            continue

        try:
            with torch.no_grad():
                clean_logits = model(clean_tokens)
                corrupt_logits = model(corrupt_tokens)
            distractor = int(summary.get("distractor_token_id", -1))
            if distractor < 0 or distractor >= corrupt_logits.shape[-1]:
                distractor = int(torch.argmax(corrupt_logits[0, -1]).item())
            if distractor == target_token:
                top2 = torch.topk(corrupt_logits[0, -1], k=2).indices.tolist()
                distractor = int(top2[1]) if len(top2) > 1 else int(torch.argmax(corrupt_logits[0, -1]).item())

            if args.recompute_baseline:
                clean_obj = float(objective_from_logits(clean_logits, target_token, distractor).item())
                corrupt_obj = float(objective_from_logits(corrupt_logits, target_token, distractor).item())
                gap = clean_obj - corrupt_obj
                if not math.isfinite(gap) or abs(gap) < 1e-8:
                    skipped.append(q_index)
                    continue
                if gap <= args.gap_min:
                    skipped.append(q_index)
                    continue

        except torch.OutOfMemoryError:
            skipped.append(q_index)
            model.reset_hooks()
            torch.cuda.empty_cache()
            continue

        contrast_pos_counter[contrast_pos] += 1

        for l in range(n_layers):
            try:
                name = f"blocks.{l}.hook_resid_pre"
                with torch.no_grad():
                    _, clean_cache = model.run_with_cache(clean_tokens, names_filter=lambda n, t=name: n == t)
                clean_layer = clean_cache[name][0].detach().cpu()
                del clean_cache

                def patch_one_position(pos: int) -> float:
                    clean_vec = clean_layer[pos, :].to(corrupt_tokens.device)

                    def hook_fn(resid: torch.Tensor, hook):  # noqa: ANN001
                        resid = resid.clone()
                        resid[:, pos, :] = clean_vec.to(dtype=resid.dtype)
                        return resid

                    with torch.no_grad():
                        logits = model.run_with_hooks(corrupt_tokens, fwd_hooks=[(name, hook_fn)])
                    obj = float(objective_from_logits(logits, target_token, distractor).item())
                    return (obj - corrupt_obj) / gap

                layer_vals_contrast[l].append(patch_one_position(contrast_pos))
                if tool_call_pos < clean_tokens.shape[1]:
                    layer_vals_toolcall[l].append(patch_one_position(tool_call_pos))
                if prefix_pos < clean_tokens.shape[1]:
                    layer_vals_prefix[l].append(patch_one_position(prefix_pos))
                del clean_layer
            except torch.OutOfMemoryError:
                break

        model.reset_hooks()
        torch.cuda.empty_cache()

    layers = list(range(n_layers))
    contrast_med = [med(layer_vals_contrast[l]) for l in layers]
    toolcall_med = [med(layer_vals_toolcall[l]) for l in layers]
    prefix_med = [med(layer_vals_prefix[l]) for l in layers]
    contrast_mean = [mean(layer_vals_contrast[l]) for l in layers]

    apply_plot_style()
    fig, ax = plt.subplots(figsize=(11.5, 5.6), constrained_layout=True)
    ax.plot(layers, contrast_med, label="Patch contrast token (Write/State)", color="#1f77b4", linewidth=2.2)
    ax.plot(layers, toolcall_med, label="Patch <tool_call> token", color="#d62728", linewidth=1.8, alpha=0.85)
    ax.plot(layers, prefix_med, label="Patch prefix token (pos 0)", color="#2ca02c", linewidth=1.8, alpha=0.85)
    ax.axhline(0.0, color="#444444", linewidth=1.0)
    ax.set_xlabel("Layer (hook_resid_pre)")
    ax.set_ylabel("Recovery ratio vs gap")
    ax.set_title("Causal Trace by Position: Recovery Curve Across Layers")
    ax.legend(loc="best")
    fig.savefig(out_root / "contrast_token_trace.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    report = {
        "n_samples": len(sample_infos),
        "skipped_q_indices": skipped,
        "contrast_position_mode": int(max(contrast_pos_counter, key=contrast_pos_counter.get)) if contrast_pos_counter else -1,
        "contrast_position_hist": {str(k): int(v) for k, v in sorted(contrast_pos_counter.items())},
        "tool_call_position": tool_call_pos,
        "prefix_position": prefix_pos,
        "layers": layers,
        "contrast_recovery_median": contrast_med,
        "contrast_recovery_mean": contrast_mean,
        "toolcall_recovery_median": toolcall_med,
        "prefix_recovery_median": prefix_med,
        "artifacts": {
            "trace_png": str(out_root / "contrast_token_trace.png"),
        },
    }
    (out_root / "contrast_token_trace_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[done] wrote {out_root / 'contrast_token_trace_report.json'}")


if __name__ == "__main__":
    main()

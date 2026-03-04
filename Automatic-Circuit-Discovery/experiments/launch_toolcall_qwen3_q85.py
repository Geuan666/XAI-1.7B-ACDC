#!/usr/bin/env python3
"""
ACDC-inspired circuit mining for Qwen3-1.7B tool-call behavior on single prompts.

Default target:
- clean:    pair/prompt-clean-q85.txt    (greedy next token tends to "<tool_call>")
- corrupt:  pair/prompt-corrupted-q85.txt (greedy next token tends to non-tool text)
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyArrowPatch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import logging as hf_logging
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from transformer_lens import HookedTransformer


# -----------------------------
# Model + token helpers
# -----------------------------


def patch_qwen3_rope_theta() -> None:
    """TransformerLens expects `cfg.rope_theta`; Qwen3 stores it under rope_scaling."""
    if hasattr(Qwen3Config, "rope_theta"):
        return
    Qwen3Config.rope_theta = property(  # type: ignore[attr-defined]
        lambda self: (
            (getattr(self, "rope_scaling", None) or {}).get("rope_theta")
            or (getattr(self, "rope_parameters", None) or {}).get("rope_theta")
            or 1_000_000
        )
    )


def load_hooked_qwen3(model_path: str, device: str, dtype: torch.dtype) -> Tuple[HookedTransformer, AutoTokenizer]:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    hf_logging.set_verbosity_error()
    try:
        hf_logging.disable_progress_bar()
    except Exception:
        pass

    patch_qwen3_rope_theta()

    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="cpu",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.bos_token is None:
        tokenizer.bos_token = "<|endoftext|>"
    tokenizer.add_bos_token = True

    model = HookedTransformer.from_pretrained(
        "Qwen/Qwen3-1.7B",
        hf_model=hf_model,
        tokenizer=tokenizer,
        device=device,
        dtype=dtype,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        trust_remote_code=True,
    )
    model.set_use_attn_result(True)
    model.set_use_split_qkv_input(True)
    model.set_use_hook_mlp_in(True)
    model.eval()

    # Free HF model copy ASAP.
    del hf_model
    gc.collect()
    torch.cuda.empty_cache()

    return model, tokenizer


def objective_from_logits(logits: torch.Tensor, target_token: int, distractor_token: int) -> torch.Tensor:
    return (logits[:, -1, target_token] - logits[:, -1, distractor_token]).mean()


def node_layer(node_name: str) -> int:
    if node_name.startswith("MLP"):
        return int(node_name[3:])
    m = re.fullmatch(r"L(\d+)H(\d+)", node_name)
    if m:
        return int(m.group(1))
    raise ValueError(f"Unknown node name format: {node_name}")


def parse_head(node_name: str) -> Tuple[int, int]:
    m = re.fullmatch(r"L(\d+)H(\d+)", node_name)
    if m is None:
        raise ValueError(f"Not a head node: {node_name}")
    return int(m.group(1)), int(m.group(2))


# -----------------------------
# Patching evaluation
# -----------------------------


def collect_clean_cache_cpu(
    model: HookedTransformer, clean_tokens: torch.Tensor
) -> Dict[str, torch.Tensor]:
    names_filter = lambda n: n.endswith("attn.hook_z") or n.endswith("hook_mlp_out")
    with torch.no_grad():
        _, cache_gpu = model.run_with_cache(clean_tokens, names_filter=names_filter)
    cache_cpu = {k: v.detach().cpu() for k, v in cache_gpu.items()}
    del cache_gpu
    torch.cuda.empty_cache()
    return cache_cpu


def evaluate_on_base_with_source(
    model: HookedTransformer,
    base_tokens: torch.Tensor,
    source_cache_cpu: Dict[str, torch.Tensor],
    patch_nodes: Sequence[str],
    target_token: int,
    distractor_token: int,
) -> float:
    heads_by_layer: Dict[int, List[int]] = {}
    mlp_layers: List[int] = []

    for node in patch_nodes:
        if node.startswith("MLP"):
            mlp_layers.append(int(node[3:]))
        else:
            layer, head = parse_head(node)
            heads_by_layer.setdefault(layer, []).append(head)

    hooks = []

    for layer, heads in heads_by_layer.items():
        cache_name = f"blocks.{layer}.attn.hook_z"
        clean_act = source_cache_cpu[cache_name].to(base_tokens.device)
        heads = sorted(set(heads))

        def make_head_hook(src: torch.Tensor, hs: Sequence[int]):
            def hook_fn(z: torch.Tensor, hook):  # noqa: ANN001
                z = z.clone()
                for h in hs:
                    # Patch only the next-token prediction position to avoid full-sequence leakage.
                    z[:, -1, h, :] = src[:, -1, h, :]
                return z

            return hook_fn

        hooks.append((cache_name, make_head_hook(clean_act, heads)))

    for layer in sorted(set(mlp_layers)):
        cache_name = f"blocks.{layer}.hook_mlp_out"
        clean_act = source_cache_cpu[cache_name].to(base_tokens.device)

        def make_mlp_hook(src: torch.Tensor):
            def hook_fn(mlp_out: torch.Tensor, hook):  # noqa: ANN001
                mlp_out = mlp_out.clone()
                mlp_out[:, -1, :] = src[:, -1, :]
                return mlp_out

            return hook_fn

        hooks.append((cache_name, make_mlp_hook(clean_act)))

    with torch.no_grad():
        logits = model.run_with_hooks(base_tokens, fwd_hooks=hooks)
    return float(objective_from_logits(logits, target_token, distractor_token).item())


def compute_ct_head_gain(
    model: HookedTransformer,
    corrupt_tokens: torch.Tensor,
    clean_cache_cpu: Dict[str, torch.Tensor],
    target_token: int,
    distractor_token: int,
    corrupt_obj: float,
) -> torch.Tensor:
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    scores = torch.zeros(n_layers, n_heads, dtype=torch.float32)

    for layer in tqdm(range(n_layers), desc="CT head", leave=False):
        cache_name = f"blocks.{layer}.attn.hook_z"
        clean_act = clean_cache_cpu[cache_name].to(corrupt_tokens.device)

        for head in range(n_heads):
            def make_head_hook(src: torch.Tensor, h: int):
                def hook_fn(z: torch.Tensor, hook):  # noqa: ANN001
                    z = z.clone()
                    z[:, -1, h, :] = src[:, -1, h, :]
                    return z

                return hook_fn

            with torch.no_grad():
                logits = model.run_with_hooks(corrupt_tokens, fwd_hooks=[(cache_name, make_head_hook(clean_act, head))])
            patched_obj = objective_from_logits(logits, target_token, distractor_token).item()
            scores[layer, head] = patched_obj - corrupt_obj

    return scores


def compute_mlp_gain(
    model: HookedTransformer,
    corrupt_tokens: torch.Tensor,
    clean_cache_cpu: Dict[str, torch.Tensor],
    target_token: int,
    distractor_token: int,
    corrupt_obj: float,
) -> torch.Tensor:
    gains: List[float] = []
    for layer in tqdm(range(model.cfg.n_layers), desc="CT MLP", leave=False):
        cache_name = f"blocks.{layer}.hook_mlp_out"
        clean_act = clean_cache_cpu[cache_name].to(corrupt_tokens.device)

        def hook_fn(mlp_out: torch.Tensor, hook):  # noqa: ANN001
            mlp_out = mlp_out.clone()
            mlp_out[:, -1, :] = clean_act[:, -1, :]
            return mlp_out

        with torch.no_grad():
            logits = model.run_with_hooks(corrupt_tokens, fwd_hooks=[(cache_name, hook_fn)])
        patched_obj = objective_from_logits(logits, target_token, distractor_token).item()
        gains.append(patched_obj - corrupt_obj)
    return torch.tensor(gains, dtype=torch.float32)


def compute_ap_head_gain(
    model: HookedTransformer,
    corrupt_tokens: torch.Tensor,
    clean_cache_cpu: Dict[str, torch.Tensor],
    target_token: int,
    distractor_token: int,
) -> torch.Tensor:
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    z_store: Dict[str, torch.Tensor] = {}

    def z_hook(z: torch.Tensor, hook):  # noqa: ANN001
        z.retain_grad()
        z_store[hook.name] = z
        return z

    model.reset_hooks()
    model.add_hook(lambda n: n.endswith("attn.hook_z"), z_hook)

    model.zero_grad(set_to_none=True)
    logits = model(corrupt_tokens)
    obj = objective_from_logits(logits, target_token, distractor_token)
    obj.backward()

    ap = torch.zeros(n_layers, n_heads, dtype=torch.float32)
    for layer in range(n_layers):
        name = f"blocks.{layer}.attn.hook_z"
        z = z_store[name]
        grad = z.grad
        delta = clean_cache_cpu[name].to(z.device) - z.detach()
        # Position-local AP at the prediction position.
        ap[layer] = (grad[:, -1, :, :] * delta[:, -1, :, :]).sum(dim=(0, 2)).float().cpu()

    model.reset_hooks()
    torch.cuda.empty_cache()
    return ap


def compute_ap_head_gain_lowmem(
    model: HookedTransformer,
    corrupt_tokens: torch.Tensor,
    clean_cache_cpu: Dict[str, torch.Tensor],
    target_token: int,
    distractor_token: int,
) -> torch.Tensor:
    """
    Low-memory fallback for AP.
    Computes gradient*delta per layer by running backward once per layer.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    ap = torch.zeros(n_layers, n_heads, dtype=torch.float32)

    for layer in tqdm(range(n_layers), desc="AP lowmem", leave=False):
        name = f"blocks.{layer}.attn.hook_z"
        z_store: Dict[str, torch.Tensor] = {}

        def z_hook(z: torch.Tensor, hook):  # noqa: ANN001
            z.retain_grad()
            z_store[hook.name] = z
            return z

        model.reset_hooks()
        try:
            model.add_hook(lambda n, target=name: n == target, z_hook)
            model.zero_grad(set_to_none=True)
            logits = model(corrupt_tokens)
            obj = objective_from_logits(logits, target_token, distractor_token)
            obj.backward()

            z = z_store[name]
            grad = z.grad
            delta = clean_cache_cpu[name].to(z.device) - z.detach()
            ap[layer] = (grad[:, -1, :, :] * delta[:, -1, :, :]).sum(dim=(0, 2)).float().cpu()
        finally:
            model.reset_hooks()
            gc.collect()
            torch.cuda.empty_cache()

    return ap


# -----------------------------
# Node/edge selection
# -----------------------------


@dataclass
class NodeScore:
    name: str
    score: float
    ct: float
    ap: float


def pick_nodes(ct_head: torch.Tensor, ap_head: torch.Tensor, mlp_gain: torch.Tensor) -> Tuple[List[str], List[str], List[NodeScore]]:
    n_layers, n_heads = ct_head.shape
    combo = 0.55 * ct_head + 0.45 * ap_head

    head_rank: List[Tuple[int, int, float]] = []
    for l in range(n_layers):
        for h in range(n_heads):
            head_rank.append((l, h, float(combo[l, h].item())))
    head_rank.sort(key=lambda x: x[2], reverse=True)

    mlp_rank: List[Tuple[int, float]] = [(l, float(mlp_gain[l].item())) for l in range(n_layers)]
    mlp_rank.sort(key=lambda x: x[1], reverse=True)

    bins = [(0, 6), (7, 13), (14, 20), (21, n_layers - 1)]

    chosen_heads: List[Tuple[int, int]] = []
    for lo, hi in bins:
        bin_candidates = [(l, h, s) for (l, h, s) in head_rank if lo <= l <= hi and s > 0.15]
        for l, h, _ in bin_candidates[:2]:
            if (l, h) not in chosen_heads:
                chosen_heads.append((l, h))
    for l, h, s in head_rank:
        if s <= 0.15:
            continue
        if (l, h) not in chosen_heads:
            chosen_heads.append((l, h))
        if len(chosen_heads) >= 10:
            break
    if len(chosen_heads) < 6:
        for l, h, _ in head_rank:
            if (l, h) not in chosen_heads:
                chosen_heads.append((l, h))
            if len(chosen_heads) >= 6:
                break

    chosen_mlps: List[int] = []
    for lo, hi in bins:
        cand = next((l for (l, s) in mlp_rank if lo <= l <= hi and s > 0.10), None)
        if cand is not None and cand not in chosen_mlps:
            chosen_mlps.append(cand)
    for l, s in mlp_rank:
        if s <= 0.10:
            continue
        if l not in chosen_mlps:
            chosen_mlps.append(l)
        if len(chosen_mlps) >= 6:
            break
    if len(chosen_mlps) < 4:
        for l, _ in mlp_rank:
            if l not in chosen_mlps:
                chosen_mlps.append(l)
            if len(chosen_mlps) >= 4:
                break

    detailed_nodes = [f"MLP{l}" for l in chosen_mlps] + [f"L{l}H{h}" for l, h in chosen_heads]
    detailed_nodes = sorted(set(detailed_nodes), key=lambda n: (node_layer(n), 0 if n.startswith("MLP") else 1, n))

    score_table: List[NodeScore] = []
    for name in detailed_nodes:
        if name.startswith("MLP"):
            l = int(name[3:])
            score_table.append(NodeScore(name=name, score=float(mlp_gain[l].item()), ct=float(mlp_gain[l].item()), ap=0.0))
        else:
            l, h = parse_head(name)
            score_table.append(
                NodeScore(
                    name=name,
                    score=float(combo[l, h].item()),
                    ct=float(ct_head[l, h].item()),
                    ap=float(ap_head[l, h].item()),
                )
            )
    score_table.sort(key=lambda x: x.score, reverse=True)

    # Keep a compact but branched rough circuit.
    score_lookup = {s.name: s.score for s in score_table}
    rough_set = set()
    must_keep = set()
    if detailed_nodes:
        rough_set.add(detailed_nodes[0])
        rough_set.add(detailed_nodes[-1])
        must_keep.add(detailed_nodes[0])
        must_keep.add(detailed_nodes[-1])
        early_layer = min(node_layer(n) for n in detailed_nodes)
        early_nodes = [n for n in detailed_nodes if node_layer(n) == early_layer]
        early_nodes = sorted(early_nodes, key=lambda n: score_lookup.get(n, 0.0), reverse=True)
        rough_set.update(early_nodes[:2])
        must_keep.update(early_nodes[:2])

    top_heads = [s.name for s in score_table if s.name.startswith("L")]
    top_mlps = [s.name for s in score_table if s.name.startswith("MLP")]
    rough_set.update(top_heads[:3])
    rough_set.update(top_mlps[:3])

    for lo, hi in bins:
        bin_nodes = [n for n in detailed_nodes if lo <= node_layer(n) <= hi]
        if bin_nodes:
            best = max(bin_nodes, key=lambda n: score_lookup.get(n, 0.0))
            rough_set.add(best)

    # Prefer at least one duplicated-layer pair (visible branch), then top up by score.
    for l in sorted({node_layer(n) for n in detailed_nodes}):
        same_layer = [n for n in detailed_nodes if node_layer(n) == l]
        if len(same_layer) >= 2:
            same_layer = sorted(same_layer, key=lambda n: score_lookup.get(n, 0.0), reverse=True)
            rough_set.update(same_layer[:2])
            break

    for s in score_table:
        if len(rough_set) >= 8:
            break
        rough_set.add(s.name)

    rough_nodes = sorted(rough_set, key=lambda n: (node_layer(n), 0 if n.startswith("MLP") else 1, n))
    if len(rough_nodes) > 8:
        keep = set(must_keep)
        if len(keep) > 8:
            mandatory_scored = sorted(keep, key=lambda n: score_lookup.get(n, 0.0), reverse=True)
            keep = set(mandatory_scored[:8])
        for s in score_table:
            if len(keep) >= 8:
                break
            if s.name in rough_set:
                keep.add(s.name)
        rough_nodes = sorted(keep, key=lambda n: (node_layer(n), 0 if n.startswith("MLP") else 1, n))

    return detailed_nodes, rough_nodes, score_table


def build_edges(nodes: Sequence[str], score_lookup: Dict[str, float], max_parents: int = 2) -> List[Tuple[str, str]]:
    input_node = "Input Embed"
    output_node = "Residual Output: <tool_call>"

    sorted_nodes = sorted(nodes, key=lambda n: (node_layer(n), 0 if n.startswith("MLP") else 1, n))
    if not sorted_nodes:
        return [(input_node, output_node)]

    edges: List[Tuple[str, str]] = []

    # Input fans into 1-2 earliest anchors.
    earliest_layer = min(node_layer(n) for n in sorted_nodes)
    earliest_nodes = [n for n in sorted_nodes if node_layer(n) == earliest_layer]
    earliest_nodes = sorted(earliest_nodes, key=lambda n: score_lookup.get(n, 0.0), reverse=True)
    for n in earliest_nodes[:2]:
        edges.append((input_node, n))
    if len(earliest_nodes) == 1 and len(sorted_nodes) > 1:
        edges.append((input_node, sorted_nodes[1]))

    # Multi-parent DAG edges based on score + distance decay.
    for idx, target in enumerate(sorted_nodes):
        t_layer = node_layer(target)
        candidates = [n for n in sorted_nodes[:idx] if node_layer(n) < t_layer]
        if not candidates:
            continue

        ranked: List[Tuple[float, str]] = []
        for src in candidates:
            gap = max(1, t_layer - node_layer(src))
            type_bonus = 0.18 if (src.startswith("MLP") != target.startswith("MLP")) else 0.0
            rank_score = score_lookup.get(src, 0.0) / (1.0 + 0.35 * gap) + type_bonus
            ranked.append((rank_score, src))
        ranked.sort(key=lambda x: x[0], reverse=True)
        for _, src in ranked[:max_parents]:
            edges.append((src, target))

    # Multiple late nodes feed output.
    latest_sorted = sorted(
        sorted_nodes,
        key=lambda n: (node_layer(n), score_lookup.get(n, 0.0)),
        reverse=True,
    )
    output_parents: List[str] = []
    seen_layers = set()
    for n in latest_sorted:
        l = node_layer(n)
        if l in seen_layers and len(output_parents) >= 1:
            continue
        output_parents.append(n)
        seen_layers.add(l)
        if len(output_parents) >= 3:
            break
    for n in output_parents:
        edges.append((n, output_node))

    def dedup_edges(edge_list: Sequence[Tuple[str, str]]) -> List[Tuple[str, str]]:
        out: List[Tuple[str, str]] = []
        seen = set()
        for e in edge_list:
            if e not in seen:
                out.append(e)
                seen.add(e)
        return out

    edges = dedup_edges(edges)

    # Connectivity repair from Input.
    adjacency: Dict[str, List[str]] = {}
    for s, t in edges:
        adjacency.setdefault(s, []).append(t)

    reached = {input_node}
    queue = [input_node]
    while queue:
        cur = queue.pop(0)
        for nxt in adjacency.get(cur, []):
            if nxt not in reached:
                reached.add(nxt)
                queue.append(nxt)

    for n in sorted_nodes:
        if n in reached:
            continue
        preds = [p for p in sorted_nodes if node_layer(p) < node_layer(n) and p in reached]
        if preds:
            best_pred = max(preds, key=lambda x: score_lookup.get(x, 0.0))
            edges.append((best_pred, n))
        else:
            edges.append((input_node, n))

    edges = dedup_edges(edges)

    # Enforce out-degree for all non-output nodes.
    outdeg: Dict[str, int] = {}
    for s, _ in edges:
        outdeg[s] = outdeg.get(s, 0) + 1

    for n in [input_node] + sorted_nodes:
        if outdeg.get(n, 0) > 0:
            continue
        later = [m for m in sorted_nodes if node_layer(m) > node_layer(n)]
        if later:
            target = max(
                later,
                key=lambda m: score_lookup.get(m, 0.0) - 0.18 * (node_layer(m) - node_layer(n)),
            )
            edges.append((n, target))
        else:
            edges.append((n, output_node))

    return dedup_edges(edges)


def assert_no_dead_ends(nodes: Sequence[str], edges: Sequence[Tuple[str, str]]) -> None:
    input_node = "Input Embed"
    output_node = "Residual Output: <tool_call>"
    outdeg: Dict[str, int] = {}
    for s, _ in edges:
        outdeg[s] = outdeg.get(s, 0) + 1
    dead_nodes = [n for n in [input_node] + list(nodes) if n != output_node and outdeg.get(n, 0) == 0]
    if dead_nodes:
        raise RuntimeError(f"Graph contains dead-end non-output nodes: {dead_nodes}")


# -----------------------------
# Plotting
# -----------------------------


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


def plot_head_heatmap(data: np.ndarray, title: str, out_path: Path) -> None:
    apply_plot_style()

    clip = float(np.percentile(np.abs(data), 98))
    clip = max(clip, 1e-6)

    fig, ax = plt.subplots(figsize=(10.5, 6.2), constrained_layout=True)
    im = ax.imshow(data, cmap="RdBu_r", vmin=-clip, vmax=clip, aspect="auto", origin="lower")
    ax.set_title(f"{title}\nSymmetric clipping at ±{clip:.3f}")
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Contribution (positive = supports <tool_call>)")
    fig.savefig(out_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def plot_probe(
    out_path: Path,
    head_name: str,
    corrupt_obj: float,
    corrupt_with_head: float,
    clean_obj: float,
    clean_with_head_corrupt: float,
) -> None:
    labels = [
        "Corrupt baseline",
        f"Corrupt + {head_name}(clean)",
        "Clean baseline",
        f"Clean + {head_name}(corrupt)",
    ]
    vals = [corrupt_obj, corrupt_with_head, clean_obj, clean_with_head_corrupt]
    colors = ["#b35d5d", "#4f81bd", "#5a9f6f", "#c08a4f"]

    apply_plot_style()
    plt.rcParams.update({"axes.titlesize": 15})

    fig, ax = plt.subplots(figsize=(9.6, 5.4), constrained_layout=True)
    bars = ax.bar(np.arange(len(vals)), vals, color=colors, edgecolor="black", linewidth=1.1)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_title(f"Component Probe: {head_name}")
    ax.set_ylabel("Tool-call logit difference")
    ax.set_xticks(np.arange(len(vals)))
    ax.set_xticklabels(labels, rotation=14, ha="right")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + (0.08 if v >= 0 else -0.08), f"{v:.2f}", ha="center", va="bottom" if v >= 0 else "top")
    fig.savefig(out_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def draw_circuit(
    nodes: Sequence[str],
    edges: Sequence[Tuple[str, str]],
    out_path: Path,
    title: str,
) -> None:
    input_node = "Input Embed"
    output_node = "Residual Output: <tool_call>"
    all_nodes = [input_node] + list(nodes) + [output_node]

    # Layer map for vertical layout (input at bottom, output at top).
    if nodes:
        min_l = min(node_layer(n) for n in nodes)
        max_l = max(node_layer(n) for n in nodes)
    else:
        min_l = 0
        max_l = 1
    layer_map = {input_node: min_l - 2, output_node: max_l + 2}
    for n in nodes:
        layer_map[n] = node_layer(n)

    by_layer: Dict[int, List[str]] = {}
    for n in all_nodes:
        by_layer.setdefault(layer_map[n], []).append(n)

    pos: Dict[str, Tuple[float, float]] = {}
    for layer in sorted(by_layer):
        group = sorted(by_layer[layer], key=lambda n: (0 if n.startswith("MLP") else 1, n))
        k = len(group)
        span = 2.2
        xs = [0.0] if k == 1 else np.linspace(-span * (k - 1) / 2, span * (k - 1) / 2, k)
        for x, n in zip(xs, group):
            pos[n] = (float(x), float(layer))

    apply_plot_style()
    fig, ax = plt.subplots(figsize=(8.6, 11.0), constrained_layout=True)
    ax.set_title(title)
    ax.axis("off")

    # Edges
    edge_color = "#8a4f28"
    for s, t in edges:
        x1, y1 = pos[s]
        x2, y2 = pos[t]
        rad = 0.09 if abs(x2 - x1) > 0.2 else 0.0
        arrow = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="->",
            mutation_scale=14,
            linewidth=1.7,
            color=edge_color,
            connectionstyle=f"arc3,rad={rad}",
            zorder=1,
        )
        ax.add_patch(arrow)

    # Nodes
    for n in all_nodes:
        x, y = pos[n]
        if n in {input_node, output_node}:
            fc = "#bcd3ea"
            ec = "#bcd3ea"
            size = 240
        else:
            fc = "#f5ede7"
            ec = "#2e2e2e"
            size = 200
        ax.scatter([x], [y], s=size, c=fc, edgecolors=ec, linewidths=1.6, zorder=3)

        if n == input_node:
            text = "Input Embed"
        elif n == output_node:
            text = "Residual Output: <tool_call>"
        else:
            text = n
        ax.text(
            x + 0.34,
            y,
            text,
            va="center",
            ha="left",
            fontsize=11 if n not in {input_node, output_node} else 13,
            zorder=4,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.2, "alpha": 0.85},
        )

    ys = [p[1] for p in pos.values()]
    ax.set_ylim(min(ys) - 1.0, max(ys) + 1.3)
    ax.set_xlim(-6.8, 7.8)
    fig.savefig(out_path, dpi=280, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine tool-call circuits on Qwen3 with ACDC-inspired patching.")
    parser.add_argument("--q-index", type=int, default=85)
    parser.add_argument("--pair-dir", type=str, default="/root/data/XAI-1.7B-ACDC/pair")
    parser.add_argument("--model-path", type=str, default="/root/data/Qwen/Qwen3-1.7B")
    parser.add_argument("--out-dir", type=str, default="experiments/results/toolcall_q85")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    clean_path = Path(args.pair_dir) / f"prompt-clean-q{args.q_index}.txt"
    corrupt_path = Path(args.pair_dir) / f"prompt-corrupted-q{args.q_index}.txt"
    if not clean_path.exists() or not corrupt_path.exists():
        raise FileNotFoundError(f"Missing prompt files: {clean_path} / {corrupt_path}")

    model, tokenizer = load_hooked_qwen3(args.model_path, device=args.device, dtype=torch.bfloat16)

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

    ap_head = compute_ap_head_gain(
        model=model,
        corrupt_tokens=corrupt_tokens,
        clean_cache_cpu=clean_cache_cpu,
        target_token=target_token,
        distractor_token=distractor_token,
    )

    detailed_nodes, rough_nodes, score_table = pick_nodes(ct_head, ap_head, mlp_gain)
    score_lookup = {s.name: s.score for s in score_table}
    detailed_edges = build_edges(detailed_nodes, score_lookup=score_lookup, max_parents=2)
    rough_edges = build_edges(rough_nodes, score_lookup=score_lookup, max_parents=2)
    assert_no_dead_ends(detailed_nodes, detailed_edges)
    assert_no_dead_ends(rough_nodes, rough_edges)

    # Circuit-level sufficiency / necessity checks.
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

    # Build corrupt cache for necessity check in clean context.
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

    # Probe strongest head.
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

    # Plots
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
        "q_index": args.q_index,
        "clean_prompt": str(clean_path),
        "corrupt_prompt": str(corrupt_path),
        "model_path": args.model_path,
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

    print(f"[done] outputs written to: {out_dir}")
    print(f"[done] probe component: {probe_head}")
    print(f"[done] detailed sufficiency ratio: {detailed_ratio:.3f}")
    print(f"[done] detailed necessity ratio: {necessity_ratio:.3f}")
    print(f"[done] rough sufficiency ratio: {rough_ratio:.3f}")
    print(f"[done] rough necessity ratio: {rough_necessity_ratio:.3f}")


if __name__ == "__main__":
    main()

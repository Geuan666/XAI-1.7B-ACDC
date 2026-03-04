"""Minimal torchtyping compatibility shim.

This project uses torchtyping annotations in several legacy modules. Newer
TransformerLens stacks require newer typeguard versions that conflict with the
published torchtyping package. For runtime behavior here, we only need
annotation compatibility, not torchtyping's runtime checks.
"""

from __future__ import annotations

import torch


class _TensorType:
    """Subscription-compatible stand-in for torchtyping.TensorType."""

    def __class_getitem__(cls, _item):
        return torch.Tensor


TensorType = _TensorType
TT = _TensorType


def patch_typeguard() -> None:
    """No-op replacement for torchtyping.patch_typeguard."""

    return None


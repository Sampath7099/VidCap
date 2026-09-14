"""LoRA from raw tensor ops — no `peft`. Targets the frozen LLM's projections."""
import math

import torch
import torch.nn as nn

from .config import LORA_TARGETS


class LoRALinear(nn.Module):
    """Wraps a frozen GPT-2 Conv1D / nn.Linear. B is zero-init, so init output == base output."""

    def __init__(self, base, r=8, alpha=16, dropout=0.0):
        super().__init__()
        self.base = base.requires_grad_(False)
        # Conv1D stores weight as (in, out); nn.Linear as (out, in).
        d_in, d_out = base.weight.shape if base.weight.ndim == 2 and _is_conv1d(base) \
            else (base.weight.shape[1], base.weight.shape[0])
        # Match the base layer's dtype/device. Qwen2.5 checkpoints carry bfloat16 in their
        # config, and transformers honours it by default, so a hardcoded fp32 adapter
        # dies in the matmul with "expected m1 and m2 to have the same dtype".
        w = base.weight
        self.A = nn.Parameter(torch.empty(d_in, r, dtype=w.dtype, device=w.device))
        self.B = nn.Parameter(torch.zeros(r, d_out, dtype=w.dtype, device=w.device))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = alpha / r
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + (self.drop(x) @ self.A @ self.B) * self.scale


def _is_conv1d(m):
    return type(m).__name__ == "Conv1D"


def apply_lora(model, targets=LORA_TARGETS, r=8, alpha=16, dropout=0.0):
    """Freeze `model`, then wrap every submodule whose name ends in a target. Returns count."""
    model.requires_grad_(False)
    hits = [(n, m) for n, m in model.named_modules() if n.split(".")[-1] in targets]
    for name, mod in hits:
        parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
        setattr(parent, name.split(".")[-1], LoRALinear(mod, r, alpha, dropout))
    return len(hits)


def lora_parameters(model):
    return [p for n, p in model.named_parameters() if n.endswith((".A", ".B"))]

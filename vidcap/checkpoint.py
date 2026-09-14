"""Resumable checkpointing. Kaggle kills sessions at 12h — every training script uses this."""
from pathlib import Path

import torch

from .config import CKPT


def save(name, step, model, optimizer=None, **extra):
    p = Path(CKPT) / f"{name}.pt"
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".pt.tmp")
    torch.save({"step": step, "model": model.state_dict(),
                "optim": optimizer.state_dict() if optimizer is not None else None,
                **extra}, tmp)
    tmp.replace(p)  # atomic — a kill mid-write can't corrupt the last good checkpoint
    return p


def load(name, model=None, optimizer=None, map_location="cpu"):
    """Returns the checkpoint dict (models/optimizers restored in place), or None if absent."""
    p = Path(CKPT) / f"{name}.pt"
    if not p.exists():
        return None
    ck = torch.load(p, map_location=map_location, weights_only=False)
    if model is not None:
        model.load_state_dict(ck["model"])
    if optimizer is not None and ck.get("optim") is not None:
        optimizer.load_state_dict(ck["optim"])
    return ck

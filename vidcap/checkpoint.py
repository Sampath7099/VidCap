"""Resumable checkpointing. Kaggle kills sessions at 12h — every training script uses this.

Checkpoints hold ONLY trainable state. The frozen decoder is ~3.2GB and is rebuilt from HF on
load; storing it three times filled /kaggle/working and made torch.save die mid-write.
"""
from pathlib import Path

import torch

from .config import CKPT


def _state(model):
    """Trainable-only state when the model offers it, else everything (plain nn.Modules)."""
    fn = getattr(model, "trainable_state_dict", None)
    return fn() if callable(fn) else model.state_dict()


def _expected_missing(model, key):
    """Keys a trainable-only checkpoint legitimately omits: the frozen decoder's own weights."""
    fn = getattr(model, "_is_frozen_llm_key", None)
    return bool(fn(key)) if callable(fn) else False


def save(name, step, model, optimizer=None, **extra):
    p = Path(CKPT) / f"{name}.pt"
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".pt.tmp")
    torch.save({"step": step, "model": _state(model),
                "optim": optimizer.state_dict() if optimizer is not None else None,
                **extra}, tmp)
    tmp.replace(p)  # atomic — a kill mid-write can't corrupt the last good checkpoint
    return p


def restore(model, ck, where="checkpoint"):
    """Load trainable state, refusing anything that does not match this architecture.

    Missing frozen-decoder keys are expected. Any other missing key, any unexpected key, or a
    shape mismatch means the checkpoint was written by a different architecture — say so in
    three readable lines rather than dumping a wall of state_dict keys.
    """
    saved = {k: v for k, v in (ck.get("arch") or ck.get("args") or {}).items()
             if k in ("connector", "n_prefix", "lora_r", "stage", "k", "blind")}

    def reject(detail):
        return RuntimeError(
            f"{where} was written by a different architecture and cannot be used.\n"
            f"  saved with: {saved or 'unknown (older checkpoint)'}\n"
            f"  fix: delete it, or pass --name <new-name> to train from scratch.\n"
            f"  {detail}")

    try:
        info = model.load_state_dict(ck["model"], strict=False)
    except RuntimeError as e:                      # shape mismatch always raises
        raise reject(f"torch said: {str(e).splitlines()[0]}") from None
    missing = [k for k in info.missing_keys if not _expected_missing(model, k)]
    if missing or info.unexpected_keys:
        raise reject(f"{len(missing)} unmatched, {len(info.unexpected_keys)} unexpected "
                     f"(e.g. {(missing + list(info.unexpected_keys))[:3]})")
    return model


def load(name, model=None, optimizer=None, map_location="cpu"):
    """Returns the checkpoint dict (models/optimizers restored in place), or None if absent."""
    p = Path(CKPT) / f"{name}.pt"
    if not p.exists():
        return None
    ck = torch.load(p, map_location=map_location, weights_only=False)
    if model is not None:
        restore(model, ck, where=str(p))
    if optimizer is not None and ck.get("optim") is not None:
        optimizer.load_state_dict(ck["optim"])
    return ck

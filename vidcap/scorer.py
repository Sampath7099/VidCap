"""Frame-relevance scorer — the project's headline component.

Predicts, from a frame's cached SigLIP embedding ALONE, how caption-relevant that frame is.
Captions are used only to build training targets; the scorer never sees text, at train or test
time, so there is no chicken-and-egg problem at inference.

Trained with a listwise ranking loss, not regression: SigLIP embeddings are anisotropic (a red
and a blue frame sit at 0.924 cosine), so absolute similarity carries almost no signal — only
the ordering of frames within a clip does.
"""
import numpy as np
import torch
import torch.nn as nn

from .config import VISION_DIM


class FrameScorer(nn.Module):
    """(B, N, D) frame embeddings -> (B, N) relevance scores. Deliberately small: it sees one
    frame at a time, so extra capacity buys memorisation, not generalisation."""

    def __init__(self, d=VISION_DIM, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d),                      # same reason as the connector: raw SigLIP is
            nn.Linear(d, hidden), nn.GELU(),      # large-magnitude and anisotropic
            nn.Linear(hidden, hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, f):
        return self.net(f).squeeze(-1)


def clip_targets(emb, caps):
    """Per-frame relevance target: mean cosine similarity to this clip's caption embeddings.

    Standardised within the clip, because only relative order is meaningful — the absolute
    values sit in a narrow band and differ per clip.
    """
    e = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8)
    c = caps / np.maximum(np.linalg.norm(caps, axis=1, keepdims=True), 1e-8)
    t = (e @ c.T).mean(1)
    return (t - t.mean()) / max(t.std(), 1e-6)


def listwise_loss(pred, target, mask):
    """ListNet: cross-entropy between the softmax of predicted and target scores over a clip.

    Listwise rather than pairwise — it uses the whole ranking at once and needs no pair
    sampling. Padding is masked to -inf so it cannot attract probability mass.
    """
    neg = torch.finfo(pred.dtype).min
    pred = pred.masked_fill(~mask, neg)
    target = target.masked_fill(~mask, neg)
    return -(torch.softmax(target, -1) * torch.log_softmax(pred, -1)).sum(-1).mean()


@torch.no_grad()
def spearman(pred, target, mask):
    """Mean within-clip Spearman correlation — the scorer's correctness metric."""
    out = []
    for p, t, m in zip(pred, target, mask):
        p, t = p[m], t[m]
        if p.numel() < 3:
            continue
        rp = p.argsort().argsort().float()
        rt = t.argsort().argsort().float()
        rp, rt = rp - rp.mean(), rt - rt.mean()
        d = (rp.norm() * rt.norm()).clamp_min(1e-8)
        out.append(((rp * rt).sum() / d).item())
    return float(np.mean(out)) if out else float("nan")

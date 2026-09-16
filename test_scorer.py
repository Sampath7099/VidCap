"""Scorer gates: python test_scorer.py"""
import os
import tempfile

import numpy as np

TMP = tempfile.mkdtemp(prefix="vidcap_scorer_")
os.environ["VIDCAP_DATA"], os.environ["VIDCAP_OUT"] = f"{TMP}/data", f"{TMP}/out"

import torch  # noqa: E402
from vidcap.config import VISION_DIM  # noqa: E402
from vidcap.scorer import (FrameScorer, clip_targets, listwise_loss,  # noqa: E402
                           spearman)


def test_targets_rank_by_caption_similarity():
    """The target must rank caption-aligned frames highest — it is the supervision signal,
    so if it is wrong everything downstream learns the wrong thing."""
    rng = np.random.default_rng(0)
    cap = rng.standard_normal((3, VISION_DIM)).astype(np.float32)
    emb = rng.standard_normal((20, VISION_DIM)).astype(np.float32)
    planted = [2, 9, 15]
    for i in planted:
        emb[i] = cap.mean(0) * 4.0
    t = clip_targets(emb, cap)
    assert sorted(np.argsort(-t)[:3].tolist()) == planted, np.argsort(-t)[:5]
    # standardised within the clip: only order is meaningful, so scale must be removed
    assert abs(t.mean()) < 1e-4 and abs(t.std() - 1.0) < 1e-3, (t.mean(), t.std())
    print(f"targets ok (found planted {planted}, standardised)")


def test_spearman_endpoints():
    p = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    m = torch.ones(1, 4, dtype=torch.bool)
    assert spearman(p, p, m) > 0.99, "identical ranking must be +1"
    assert spearman(p, -p, m) < -0.99, "reversed ranking must be -1"
    print("spearman ok (+1 identical, -1 reversed)")


def test_loss_ignores_padding():
    """Clips have different pool sizes, so batches are padded. Padding must not affect the loss
    or short clips would train against phantom frames."""
    torch.manual_seed(0)
    p = torch.randn(1, 6)
    t = torch.randn(1, 6)
    m = torch.tensor([[True] * 4 + [False] * 2])
    full = listwise_loss(p[:, :4], t[:, :4], m[:, :4])
    padded = listwise_loss(p, t, m)
    assert torch.allclose(full, padded, atol=1e-5), (full.item(), padded.item())
    print("padding mask ok (loss unchanged by padded slots)")


def test_scorer_learns_a_ranking():
    """End-to-end: on data where relevance IS a fixed direction in embedding space, the scorer
    must recover the ordering from the embedding alone. If this fails the head cannot learn."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    d = 64
    direction = rng.standard_normal(d).astype(np.float32)
    F, T = [], []
    for _ in range(64):
        e = rng.standard_normal((24, d)).astype(np.float32)
        t = e @ direction
        F.append(e); T.append((t - t.mean()) / t.std())
    F = torch.from_numpy(np.stack(F)).float()
    T = torch.from_numpy(np.stack(T)).float()
    M = torch.ones(F.shape[:2], dtype=torch.bool)

    model = FrameScorer(d=d, hidden=64)
    before = spearman(model(F), T, M)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(300):
        loss = listwise_loss(model(F), T, M)
        opt.zero_grad(); loss.backward(); opt.step()
    after = spearman(model(F), T, M)
    assert after > 0.9, f"scorer failed to learn a recoverable ranking: {before:.3f} -> {after:.3f}"
    print(f"scorer learns ranking ok (spearman {before:+.3f} -> {after:+.3f})")


def test_scorer_sees_no_text():
    """The scorer must be usable at inference with no caption anywhere — that is the whole
    premise. Its forward signature takes embeddings only."""
    m = FrameScorer(d=32, hidden=16)
    out = m(torch.randn(2, 11, 32))
    assert out.shape == (2, 11), out.shape
    print("scorer interface ok (embeddings in, per-frame scores out, no text)")


if __name__ == "__main__":
    test_targets_rank_by_caption_similarity()
    test_spearman_endpoints()
    test_loss_ignores_padding()
    test_scorer_learns_a_ranking()
    test_scorer_sees_no_text()
    print(f"\nscorer gates passed ({TMP})")

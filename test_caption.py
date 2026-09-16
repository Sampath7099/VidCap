"""End-to-end demo gates (context.md requirement #1): python test_caption.py"""
import os
import pathlib
import tempfile

import numpy as np

TMP = tempfile.mkdtemp(prefix="vidcap_caption_")
os.environ["VIDCAP_DATA"], os.environ["VIDCAP_OUT"] = f"{TMP}/data", f"{TMP}/out"

import cv2  # noqa: E402
import torch  # noqa: E402
from scripts.caption import caption_from_pool  # noqa: E402
from vidcap.config import MIN_POOL_FRAMES  # noqa: E402
from vidcap.data import uniform_indices  # noqa: E402
from vidcap.model import VideoCaptioner  # noqa: E402
from vidcap.video import sample_frames  # noqa: E402

D = 64


def make_video(path, seconds=4, fps=30, size=96):
    """A moving bright square, so frames genuinely differ over time."""
    path.parent.mkdir(parents=True, exist_ok=True)
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (size, size))
    n = seconds * fps
    for i in range(n):
        f = np.zeros((size, size, 3), np.uint8)
        x = int((size - 16) * i / max(n - 1, 1))
        f[40:56, x:x + 16] = 255
        w.write(f)
    w.release()


def test_real_video_decodes_to_a_pool():
    """The demo is the only path that touches raw video. It must turn a file on disk into a
    candidate pool bigger than the largest frame budget, with real timestamps."""
    p = pathlib.Path(TMP) / "clip.mp4"
    make_video(p, seconds=4)
    frames, times = sample_frames(p, fps=3.0)
    assert len(frames) >= MIN_POOL_FRAMES, f"pool too small: {len(frames)}"
    assert len(frames) == len(times)
    assert frames.ndim == 4 and frames.shape[-1] == 3, frames.shape
    assert times[-1] > times[0], "timestamps must increase"
    print(f"video -> pool ok ({len(frames)} frames, {times[-1]:.1f}s)")


def test_missing_video_does_not_crash():
    frames, times = sample_frames(pathlib.Path(TMP) / "nope.mp4", fps=3.0)
    assert len(frames) == 0 and len(times) == 0, "undecodable video must return empty, not raise"
    print("undecodable video ok (empty pool, no crash)")


def test_caption_from_pool_end_to_end():
    """Pool -> selector -> connector -> decoder -> a real string. Uses distilgpt2 so this runs
    on a laptop; the only thing it cannot check is caption quality."""
    torch.manual_seed(0)
    m = VideoCaptioner(llm_name="distilgpt2", d_vis=D, n_prefix=4, connector="meanpool",
                       lora_r=0).eval()
    emb = np.random.randn(37, D).astype(np.float32)
    cap, idx = caption_from_pool(emb, 4, lambda e, k, r: uniform_indices(len(e), k), m)
    assert isinstance(cap, str), type(cap)
    assert len(idx) == 4 and max(idx) < 37, idx
    print(f"pool -> caption ok (picked {idx}, said {cap[:40]!r})")


def test_empty_pool_is_handled():
    m = VideoCaptioner(llm_name="distilgpt2", d_vis=D, n_prefix=4, connector="meanpool",
                       lora_r=0).eval()
    cap, idx = caption_from_pool(np.zeros((0, D), np.float32), 4,
                                 lambda e, k, r: uniform_indices(len(e), k), m)
    assert cap == "" and idx == [], (cap, idx)
    print("empty pool ok (no caption, no crash)")


def test_budget_larger_than_pool():
    """A 1-second clip can yield fewer frames than K. The demo must still produce a caption."""
    m = VideoCaptioner(llm_name="distilgpt2", d_vis=D, n_prefix=4, connector="meanpool",
                       lora_r=0).eval()
    emb = np.random.randn(3, D).astype(np.float32)
    cap, idx = caption_from_pool(emb, 8, lambda e, k, r: uniform_indices(len(e), k), m)
    assert len(idx) == 8 and isinstance(cap, str), (idx, cap)
    print(f"short pool ok (3 frames padded to K=8)")


if __name__ == "__main__":
    test_real_video_decodes_to_a_pool()
    test_missing_video_does_not_crash()
    test_caption_from_pool_end_to_end()
    test_empty_pool_is_handled()
    test_budget_larger_than_pool()
    print(f"\ncaption gates passed ({TMP})")

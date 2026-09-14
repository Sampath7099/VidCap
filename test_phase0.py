"""Phase 0 self-check: python test_phase0.py  (builds a synthetic video, no downloads needed)."""
import json
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np

TMP = tempfile.mkdtemp(prefix="vidcap_t0_")
os.environ["VIDCAP_DATA"], os.environ["VIDCAP_OUT"] = f"{TMP}/data", f"{TMP}/out"

import torch  # noqa: E402
from vidcap import checkpoint, datasets  # noqa: E402
from vidcap.config import FRAME_SIZE, VISION_DIM  # noqa: E402
from vidcap.encoder import cache_video, load_cached, normalize  # noqa: E402
from vidcap.video import sample_frames  # noqa: E402


def make_video(path, seconds=6, fps=30, size=(320, 240)):
    """Colour changes every second, so 1fps sampling must return `seconds` distinct frames."""
    path.parent.mkdir(parents=True, exist_ok=True)
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(seconds * fps):
        f = np.zeros((size[1], size[0], 3), np.uint8)
        f[..., i // fps % 3] = 255
        w.write(f)
    w.release()


def test_sampling():
    v = Path(TMP) / "data/msrvtt/video0.mp4"
    make_video(v, seconds=6)
    frames, times = sample_frames(v, fps=1.0, min_frames=0)
    assert len(frames) == len(times) == 6, f"expected 6 frames, got {len(frames)}"
    assert np.all(np.diff(times) > 0.9), f"timestamps not ~1s apart: {times}"
    # source (240px short side) is below FRAME_SIZE: must NOT be upscaled, that only blurs
    assert min(frames[0].shape[:2]) == 240, f"small source was upscaled: {frames[0].shape}"

    # a source above FRAME_SIZE must be downscaled to exactly it
    big = Path(TMP) / "data/msrvtt/video_big.mp4"
    make_video(big, seconds=2, size=(1280, 720))
    bf, _ = sample_frames(big, fps=1.0, min_frames=0)
    assert min(bf[0].shape[:2]) == FRAME_SIZE, f"not resized to {FRAME_SIZE}: {bf[0].shape}"
    # pool floor: a short clip must still yield > the largest frame budget (16)
    frames, _ = sample_frames(v, fps=1.0, min_frames=32)
    assert len(frames) >= 32, f"pool floor not applied: {len(frames)} frames from a 6s clip"
    # long video must be capped AND still reach the end (uniform subsample, not truncation)
    frames, times = sample_frames(v, fps=30.0, max_frames=10)
    assert len(frames) == 10 and times[-1] > 5.0, f"tail lost on cap: {times}"
    print("sampling ok")


def test_cache_resumable():
    from vidcap.encoder import load_vision
    model, proc, device = load_vision()
    v = Path(TMP) / "data/msrvtt/video0.mp4"
    p, n = cache_video("msrvtt", "video0", v, model, proc, device)
    emb, times = load_cached("msrvtt", "video0")
    # msrvtt pools at 3fps with a 32-frame floor -> a 6s clip gives well over the K=16 budget
    assert emb.shape[0] >= 32 and emb.shape[1] == VISION_DIM and emb.dtype == np.float32, emb.shape
    assert len(times) == len(emb)
    mtime = p.stat().st_mtime_ns
    _, n2 = cache_video("msrvtt", "video0", v, model, proc, device)  # re-run must skip
    assert n2 == n and p.stat().st_mtime_ns == mtime, "re-run recomputed instead of resuming"
    # The pool must span the clip. Compare WITHIN-colour-block similarity against ACROSS-block:
    # an absolute cosine threshold is not portable, because CLIP/SigLIP embeddings are anisotropic
    # (everything sits in a narrow cone, so even unrelated images score ~0.9). The relative
    # gap is the encoder-independent signal.
    sim = normalize(emb) @ normalize(emb).T
    block = (times.astype(int) % 3)                      # synthetic video cycles r/g/b each second
    same = block[:, None] == block[None, :]
    eye = np.eye(len(emb), dtype=bool)
    within, across = sim[same & ~eye].mean(), sim[~same].mean()
    assert within > across + 0.01, \
        f"pool does not distinguish frames: within-block {within:.3f} vs across {across:.3f}"
    assert times[-1] > 4.0, f"pool does not reach the clip's end: {times[-1]}"
    print(f"  within-block sim {within:.3f} > across-block {across:.3f}")
    print(f"cache ok ({n} frames, resumable, dim={emb.shape[1]})")


def test_msrvtt_loader_and_leakage():
    root = Path(TMP) / "data/msrvtt"
    make_video(root / "video7010.mp4", seconds=2)
    (root / "info.videodatainfo.json").write_text(json.dumps({
        "videos": [{"video_id": "video0", "split": "train"},
                   {"video_id": "video7010", "split": "test"}],
        "sentences": [{"video_id": "video0", "caption": "a man is talking"},
                      {"video_id": "video7010", "caption": "a dog runs"}]}))
    recs = datasets.msrvtt(root)
    assert len(recs) == 2, recs
    assert {r["split"] for r in recs} == {"train", "test"}
    assert recs[0]["captions"] == ["a man is talking"]
    assert datasets.verify_no_leakage(recs) == {"test": 1, "train": 1}
    # the gate must actually fire when a test id leaks into train
    leaked = recs + [{"video_id": "video7010", "path": None, "split": "train", "captions": []}]
    try:
        datasets.verify_no_leakage(leaked)
        raise SystemExit("FAIL: leakage went undetected")
    except AssertionError:
        pass
    print("loader + leakage gate ok")


def test_msrvtt_frozen_in_time_naming():
    """The Frozen-in-Time zip ships MSR_VTT.json, not *videodatainfo*.json."""
    root = Path(TMP) / "data/msrvtt_fit"
    make_video(root / "videos/all/video9.mp4", seconds=2)
    (root / "annotation").mkdir(parents=True, exist_ok=True)
    (root / "annotation/MSR_VTT.json").write_text(json.dumps({
        "videos": [{"video_id": "video9", "split": "train"}],
        "sentences": [{"video_id": "video9", "caption": "a cat sits"}]}))
    recs = datasets.msrvtt(root)
    assert len(recs) == 1 and recs[0]["captions"] == ["a cat sits"], recs

    # COCO-style schema (annotations/image_id) with no per-video split field: captions
    # must still parse, and the split must fall back to the official id ranges.
    coco = Path(TMP) / "data/msrvtt_coco"
    for vid in ("video3", "video6600", "video8000"):
        make_video(coco / f"videos/all/{vid}.mp4", seconds=2)
    (coco / "annotation").mkdir(parents=True, exist_ok=True)
    (coco / "annotation/MSR_VTT.json").write_text(json.dumps({
        "videos": [{"video_id": v} for v in ("video3", "video6600", "video8000")],
        "annotations": [{"image_id": v, "caption": f"cap {v}"}
                        for v in ("video3", "video6600", "video8000")]}))
    recs = datasets.msrvtt(coco)
    got = {r["video_id"]: r["split"] for r in recs}
    assert got == {"video3": "train", "video6600": "val", "video8000": "test"}, got

    # a schema we genuinely can't parse must name itself, not return an empty list
    weird = Path(TMP) / "data/msrvtt_weird"
    make_video(weird / "video0.mp4", seconds=2)
    (weird / "MSR_VTT.json").write_text(json.dumps({"totally": "different"}))
    try:
        datasets.msrvtt(weird)
        raise SystemExit("FAIL: unparseable schema returned silently")
    except ValueError as e:
        assert "totally" in str(e), e
    # a missing annotation must fail loudly, not return an empty dataset
    bare = Path(TMP) / "data/msrvtt_bare"
    make_video(bare / "video0.mp4", seconds=2)
    try:
        datasets.msrvtt(bare)
        raise SystemExit("FAIL: missing annotation returned silently")
    except FileNotFoundError:
        pass
    print("frozen-in-time naming + missing-annotation gate ok")


def test_checkpoint_resume():
    m = torch.nn.Linear(4, 2)
    opt = torch.optim.Adam(m.parameters(), lr=0.1)
    m(torch.randn(3, 4)).sum().backward()
    opt.step()  # real step -> Adam moment buffers exist to restore
    checkpoint.save("t0", step=7, model=m, optimizer=opt, note="hi")

    m2, opt2 = torch.nn.Linear(4, 2), None
    ck = checkpoint.load("t0", m2)
    assert ck["step"] == 7 and ck["note"] == "hi"
    assert torch.equal(m.weight, m2.weight), "weights not restored"
    opt2 = torch.optim.Adam(m2.parameters(), lr=0.1)
    checkpoint.load("t0", m2, opt2)
    assert opt2.state_dict()["state"], "optimizer state not restored — resume would be wrong"
    assert checkpoint.load("nope") is None
    print("checkpoint resume ok")


def test_checkpoint_rejects_wrong_architecture():
    """Changing --connector/--n-prefix makes an old checkpoint structurally incompatible.
    It must say so in one readable line, not dump a wall of state_dict keys."""
    a = torch.nn.Linear(4, 2)
    checkpoint.save("arch", 1, a, args={"connector": "resampler", "n_prefix": 8})
    try:
        checkpoint.load("arch", torch.nn.Linear(4, 8))   # different shape = different arch
        raise SystemExit("FAIL: incompatible checkpoint loaded silently")
    except RuntimeError as e:
        assert "different architecture" in str(e) and "resampler" in str(e), e
    print("checkpoint architecture guard ok")


if __name__ == "__main__":
    test_sampling()
    test_cache_resumable()
    test_msrvtt_loader_and_leakage()
    test_msrvtt_frozen_in_time_naming()
    test_checkpoint_resume()
    test_checkpoint_rejects_wrong_architecture()
    print(f"\nall phase 0 checks passed  ({TMP})")

"""ActivityNet paragraph-eval gates: python test_activitynet.py"""
import json
import os
import pathlib
import tempfile

import numpy as np

TMP = tempfile.mkdtemp(prefix="vidcap_anet_")
os.environ["VIDCAP_DATA"], os.environ["VIDCAP_OUT"] = f"{TMP}/data", f"{TMP}/out"

import torch  # noqa: E402
from scripts.evaluate import uniform_sel  # noqa: E402
from scripts.evaluate_paragraphs import (FIXED, caption_segments, paired_ci,  # noqa: E402
                                         paragraph, timelines)
from vidcap import datasets  # noqa: E402
from vidcap.metrics import cider_d  # noqa: E402
from vidcap.model import VideoCaptioner  # noqa: E402

D = 64


def test_loader_reads_val_paragraphs_in_time_order():
    root = pathlib.Path(TMP) / "data/activitynet"
    (root / "videos").mkdir(parents=True, exist_ok=True)
    for v in ("v_aaa.mp4", "v_bbb.mkv", "v_train.mp4"):
        (root / "videos" / v).write_bytes(b"")
    (root / "val_1.json").write_text(json.dumps({
        "v_aaa": {"duration": 30, "timestamps": [[10, 20], [0, 9]],
                  "sentences": [" He jumps.", "A man runs."]},
        "v_bbb": {"duration": 9, "timestamps": [[0, 9]], "sentences": ["A dog barks."]},
        "v_gone": {"duration": 9, "timestamps": [[0, 9]], "sentences": ["No video for this."]}}))
    (root / "val_2.json").write_text(json.dumps({
        "v_aaa": {"duration": 30, "timestamps": [[0, 30]], "sentences": ["Someone exercises."]}}))
    (root / "train.json").write_text(json.dumps({
        "v_train": {"duration": 9, "timestamps": [[0, 9]], "sentences": ["Must not load."]}}))
    recs = {r["video_id"]: r for r in datasets.activitynet()}
    assert set(recs) == {"aaa", "bbb"}, f"only val videos that exist: {set(recs)}"
    assert recs["aaa"]["paragraphs"] == ["A man runs. He jumps.", "Someone exercises."], recs["aaa"]
    assert recs["bbb"]["paragraphs"] == ["A dog barks."] and recs["aaa"]["split"] == "val"
    assert len(recs["aaa"]["captions"]) == 3
    print("activitynet loader ok (val only, time-ordered, one paragraph per annotation file)")


def test_loader_fails_loudly_without_annotations():
    empty = pathlib.Path(TMP) / "empty_anet"
    empty.mkdir()
    try:
        datasets.activitynet(empty)
    except SystemExit as e:
        assert "val_1.json" in str(e), e
        print("missing-annotation gate ok")
        return
    raise AssertionError("an ActivityNet root with no val JSONs must fail loudly")


def test_per_item_cider_matches_corpus():
    hyps = ["a man is running on a track", "a dog barks at a cat", "people dance"]
    refs = [["a man runs around a track"], ["a dog is barking"], ["a group of people dance", "people are dancing"]]
    items = cider_d(hyps, refs, per_item=True)
    assert len(items) == 3 and abs(np.mean(items) - cider_d(hyps, refs)) < 1e-12
    print(f"per-video CIDEr ok (mean {np.mean(items):.4f} == corpus)")


def test_paired_ci():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    same = paired_ci(a, a)
    assert same == {"mean": 0.0, "lo": 0.0, "hi": 0.0}, same
    up = paired_ci(a + 1.0, a)
    assert up["mean"] == 1.0 and up["lo"] == 1.0 and up["hi"] == 1.0, up
    noisy = paired_ci(a + np.array([1.0, -1.0, 1.0, -1.0]), a)
    assert noisy["lo"] < 0 < noisy["hi"], f"a zero-mean difference must straddle 0: {noisy}"
    print("paired bootstrap ok (exact on constant shifts, straddles 0 on noise)")


def test_paragraph():
    ev = [{"caption": "a man runs"}, {"caption": "he jumps."}, {"caption": "  "}]
    assert paragraph(ev) == "a man runs. he jumps.", paragraph(ev)
    print("paragraph assembly ok")


def test_batching_keeps_order_and_timelines_tile():
    torch.manual_seed(0)
    m = VideoCaptioner(llm_name="distilgpt2", d_vis=D, n_prefix=4, connector="meanpool",
                       lora_r=0).eval()
    rng = np.random.default_rng(0)
    recs = [{"video_id": f"v{i}"} for i in range(3)]
    pools = [(rng.normal(size=(n, D)).astype(np.float32), np.arange(n, dtype=np.float32))
             for n in (20, 47, 9)]
    jobs = [(r, e, s, s + 5) for r, (e, _) in zip(recs, pools) for s in range(0, len(e) - 5, 5)]
    one_by_one = caption_segments(m, jobs, 2, uniform_sel, "cpu", batch=1)
    assert caption_segments(m, jobs, 2, uniform_sel, "cpu", batch=4) == one_by_one, \
        "batching across videos must not reorder or change captions"
    tl = timelines(recs, pools, m, 2, uniform_sel, "cpu", FIXED)
    assert len(tl) == 3
    for ev, (emb, t) in zip(tl, pools):
        assert ev[0]["range"][0] == 0 and ev[-1]["range"][1] == len(emb), ev
        assert all(a["range"][1] == b["range"][0] for a, b in zip(ev, ev[1:])), "must tile"
    assert len(tl[1]) <= 4, "47 s in fixed 15 s windows is at most 4 events"
    print(f"batched captions ok (identical to one-by-one; {sum(map(len, tl))} events tile 3 videos)")


if __name__ == "__main__":
    test_loader_reads_val_paragraphs_in_time_order()
    test_loader_fails_loudly_without_annotations()
    test_per_item_cider_matches_corpus()
    test_paired_ci()
    test_paragraph()
    test_batching_keeps_order_and_timelines_tile()
    print(f"\nactivitynet gates passed ({TMP})")

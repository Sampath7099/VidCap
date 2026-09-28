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
    assert recs["aaa"]["events"] == [(0.0, 9.0, "A man runs."), (10.0, 20.0, "He jumps.")],         "events come from the first annotation file, time-ordered, with timestamps"
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


def test_spread_topk_does_not_bunch():
    from scripts.evaluate_budget import spread_topk
    s = np.zeros(120)
    s[50:56] = [9, 10, 8, 7, 9, 6]                  # one exciting burst
    s[10], s[100] = 5, 4                             # two lesser peaks elsewhere
    got = spread_topk(s, 3, 15)
    assert got == [10, 51, 100], f"expected the burst's best plus the other peaks, got {got}"
    plain = sorted(np.argsort(-s)[:3].tolist())
    assert max(plain) - min(plain) < 6, "fixture must show that plain top-k bunches"
    g = spread_topk(s, 8, 15)
    assert len(g) == 8 and len(set(g)) == 8 and g == sorted(g), f"gap too strict must top up: {g}"
    assert spread_topk(np.ones(3), 4, 1) == [0, 1, 2, 2], "short pool pads like uniform_indices"
    print(f"spread top-k ok ({got}; plain top-k would bunch at {plain})")


def test_budget_paragraphs():
    from scripts.evaluate_budget import paragraphs_for, uniform_budget
    rng = np.random.default_rng(1)
    emb = rng.normal(size=(60, D)).astype(np.float32)
    assert uniform_budget(emb, None, 4) == [0, 20, 39, 59]
    torch.manual_seed(0)
    m = VideoCaptioner(llm_name="distilgpt2", d_vis=D, n_prefix=4, connector="meanpool",
                       lora_r=0).eval()
    recs = [{"video_id": "a"}, {"video_id": "b"}]
    pools = [(emb, np.arange(60, dtype=np.float32)), (emb[:30], np.arange(30, dtype=np.float32))]
    paras = paragraphs_for(recs, pools, m, 4, uniform_budget, "cpu", batch=3)
    assert len(paras) == 2 and all(isinstance(x, str) for x in paras)
    assert paras == paragraphs_for(recs, pools, m, 4, uniform_budget, "cpu", batch=100),         "batch size must not change which caption lands in which paragraph"
    odd = paragraphs_for(recs, pools, m, 3, lambda e, t, k, r: [1, 5, 9], "cpu")
    assert len(odd) == 2, "odd budgets must pad the last pair, not crash"
    print("budget paragraphs ok (K/2 captions per video, batch-invariant, odd K handled)")


def test_segment_best_keeps_coverage():
    from scripts.evaluate_budget import by_segment
    from vidcap.timeline import pair_frames, segment_best
    s = np.zeros(120)
    s[50:56] = [9, 10, 8, 7, 9, 6]                  # the burst plain top-k would spend it all on
    s[10], s[100] = 5, 4
    got = segment_best(s, 4)
    assert got == [10, 51, 60, 100], got
    assert [g // 30 for g in got] == [0, 1, 2, 3], "exactly one frame per quarter"
    assert segment_best(np.ones(3), 4) == [0, 1, 2, 2]
    assert by_segment(lambda e, r: None)(np.zeros((40, 2)), None, 4) == [0, 13, 26, 39],         "no score -> uniform"
    assert pair_frames([1, 5, 9]) == [[1, 5], [9, 9]] and pair_frames([2, 4]) == [[2, 4]]
    print(f"segment best ok ({got}: one per quarter, best within each)")


def test_event_oracle_picks_inside_each_event():
    from scripts.evaluate_budget import make_event_oracle
    from vidcap.encoder import cache_path
    rng = np.random.default_rng(2)
    emb = rng.normal(size=(60, D)).astype(np.float32)
    times = np.arange(60, dtype=np.float32)
    sents = ["first thing", "second thing", "third thing"]
    targets = [7, 33, 52]                            # the frame each sentence "describes"
    p = cache_path("activitynet", "_captions")
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, emb=emb[targets], video_id=np.array(["x"] * 3), text=np.array(sents))
    rec = {"video_id": "x", "events": [(0, 20, sents[0]), (25, 40, sents[1]), (45, 59, sents[2])]}
    sel = make_event_oracle("activitynet")
    assert sel(emb, times, 3, rec) == targets, sel(emb, times, 3, rec)
    two = sel(emb, times, 2, rec)
    assert two == [7, 52], f"k < events: spread over events, first and last -> {two}"
    six = sel(emb, times, 6, rec)
    assert len(six) == 6 and len(set(six)) == 6 and {7, 33, 52} <= set(six), six
    assert all(any(a <= times[i] <= b for a, b, _ in rec["events"]) for i in six), "outside events"
    assert sel(emb, times, 3, {"video_id": "unknown"}) == [0, 30, 59], "no events -> uniform"
    print(f"event oracle ok (one frame per event, on its sentence's frame; k>events stays inside)")


if __name__ == "__main__":
    test_spread_topk_does_not_bunch()
    test_budget_paragraphs()
    test_segment_best_keeps_coverage()
    test_event_oracle_picks_inside_each_event()
    test_loader_reads_val_paragraphs_in_time_order()
    test_loader_fails_loudly_without_annotations()
    test_per_item_cider_matches_corpus()
    test_paired_ci()
    test_paragraph()
    test_batching_keeps_order_and_timelines_tile()
    print(f"\nactivitynet gates passed ({TMP})")

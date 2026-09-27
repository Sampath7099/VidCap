"""Video timeline / summary / Q&A gates: python test_watch.py"""
import os
import tempfile

import numpy as np

TMP = tempfile.mkdtemp(prefix="vidcap_watch_")
os.environ["VIDCAP_DATA"], os.environ["VIDCAP_OUT"] = f"{TMP}/data", f"{TMP}/out"

import torch  # noqa: E402
from scripts.evaluate import uniform_sel  # noqa: E402
from scripts.watch import (adapter, answer, best_event, build_timeline, chat,  # noqa: E402
                           summarize_timeline)
from vidcap.model import VideoCaptioner  # noqa: E402
from vidcap.timeline import (fmt_time, merge_repeats, segment, source_event,  # noqa: E402
                             timeline_text)

D = 64
rng = np.random.default_rng(0)


def scenes(*lengths):
    """One random direction per scene plus small per-frame noise, 1 fps."""
    emb = np.concatenate([np.tile(rng.normal(size=D), (n, 1)) + 0.05 * rng.normal(size=(n, D))
                          for n in lengths]).astype(np.float32)
    return emb, np.arange(len(emb), dtype=np.float32)


def tiny():
    torch.manual_seed(0)
    return VideoCaptioner(llm_name="distilgpt2", d_vis=D, n_prefix=4, connector="meanpool",
                          lora_r=0).eval()


def test_segment_finds_the_cut_and_covers_every_frame():
    emb, t = scenes(10, 12)
    sp = segment(emb, t)
    assert sp == [(0, 10), (10, 22)], f"expected one cut at the scene change, got {sp}"
    assert segment(np.zeros((0, D), np.float32), np.zeros(0)) == []
    print("scene cut ok (cut exactly at frame 10, full coverage)")


def test_segment_respects_min_and_max_length():
    emb, t = scenes(60)                            # one static minute
    sp = segment(emb, t, max_len=15.0)
    assert len(sp) == 4 and sp[0][0] == 0 and sp[-1][1] == 60, sp
    assert all(a[1] == b[0] for a, b in zip(sp, sp[1:])), f"gaps or overlaps: {sp}"
    emb, t = scenes(2, 20)                          # a 2 s flash is too short to be a scene
    assert all(t[e - 1] - t[s] >= 2.0 for s, e in segment(emb, t)), segment(emb, t)
    print(f"scene lengths ok (60 s static -> {len(sp)} spans <= 15 s; no sub-min_len scene)")


def test_merge_repeats():
    ev = [{"start": 0, "end": 5, "caption": "A man talks.", "frames": [1], "range": [0, 5]},
          {"start": 5, "end": 9, "caption": "a man talks", "frames": [6], "range": [5, 9]},
          {"start": 9, "end": 12, "caption": "a dog runs", "frames": [10], "range": [9, 12]}]
    m = merge_repeats(ev)
    assert [e["caption"] for e in m] == ["A man talks.", "a dog runs"], m
    assert m[0]["end"] == 9 and m[0]["frames"] == [1, 6] and m[0]["range"] == [0, 9], m[0]
    assert ev[0]["end"] == 5, "merge must not mutate its input"
    assert fmt_time(0) == "0:00" and fmt_time(125.4) == "2:05"
    assert timeline_text(m).splitlines()[1] == "[0:09-0:12] a dog runs"
    print("merge + formatting ok (repeat merged, ranges joined, input untouched)")


def test_source_event_pins_answer_to_time():
    ev = [{"caption": "a man is cooking in a kitchen"}, {"caption": "a woman is eating at a table"},
          {"caption": "a man and a woman are talking"}]
    assert source_event("A woman eats at a table.", ev) is ev[1]
    assert source_event("The man cooks in the kitchen", ev) is ev[0]
    assert source_event("football", ev) is None, "no shared words must give no time, not a guess"
    assert source_event("anything", []) is None
    print("answer -> event ok (time comes from the matched caption, none when nothing matches)")


def test_build_timeline_end_to_end():
    m = tiny()
    emb, t = scenes(10, 12, 8)
    ev = build_timeline(emb, t, m, uniform_sel, 2, "cpu")
    assert ev and ev[0]["start"] == 0 and ev[-1]["end"] == t[-1], ev
    assert ev[0]["range"][0] == 0 and ev[-1]["range"][1] == len(emb)
    assert all(a["range"][1] == b["range"][0] for a, b in zip(ev, ev[1:])), "scenes must tile"
    for e in ev:
        assert isinstance(e["caption"], str) and "\n" not in e["caption"], "one line per event"
        assert all(e["range"][0] <= f < e["range"][1] for f in e["frames"]), "frame outside scene"
    print(f"timeline ok ({len(ev)} events tile the video, frames inside their scene)")


def test_summary_and_timeline_answer_are_text():
    m = tiny()
    one = [{"start": 0, "end": 5, "caption": "a man is cooking", "frames": [1], "range": [0, 5]}]
    assert summarize_timeline(m, one) == "A man is cooking.", "one event must not go through the LLM"
    assert summarize_timeline(m, []) == "(no caption)"
    two = one + [{"start": 5, "end": 9, "caption": "a woman eats", "frames": [6], "range": [5, 9]}]
    assert isinstance(summarize_timeline(m, two), str)
    assert isinstance(chat(m, "sys", "user", max_new_tokens=5), str)
    emb, _ = scenes(5, 4)
    a = answer("who eats?", two, emb, m, None, None, None, "cpu")
    assert a["frames"] is None and isinstance(a["timeline"]["answer"], str), a
    print("summary + timeline answer ok (single event verbatim, no QA ckpt -> timeline only)")


def test_adapter_swap_restores_exactly():
    m = tiny()
    orig = {k: v.clone() for k, v in m.trainable_state_dict().items()}
    qa = {k: v + 1.0 for k, v in orig.items()}
    with adapter(m, {"model": qa}, {"model": orig}):
        assert all(torch.equal(m.trainable_state_dict()[k], qa[k]) for k in qa), "QA weights not in"
    assert all(torch.equal(m.trainable_state_dict()[k], orig[k]) for k in orig), "not restored"
    try:
        with adapter(m, {"model": qa}, {"model": orig}):
            raise ValueError
    except ValueError:
        pass
    assert all(torch.equal(m.trainable_state_dict()[k], orig[k]) for k in orig), \
        "an error while answering must still restore the captioner"
    print("adapter swap ok (QA weights in, captioner restored, even on error)")


def test_best_event_matches_the_question():
    emb, _ = scenes(10, 10, 10)
    ev = [{"range": [0, 10]}, {"range": [10, 20]}, {"range": [20, 30]}]
    assert best_event(ev, emb, emb[25] + 0.01) is ev[2]
    assert best_event(ev, emb, emb[3]) is ev[0]
    print("question -> scene ok (picks the scene holding the matching frame)")


if __name__ == "__main__":
    test_segment_finds_the_cut_and_covers_every_frame()
    test_segment_respects_min_and_max_length()
    test_merge_repeats()
    test_source_event_pins_answer_to_time()
    test_build_timeline_end_to_end()
    test_summary_and_timeline_answer_are_text()
    test_adapter_swap_restores_exactly()
    test_best_event_matches_the_question()
    print(f"\nwatch gates passed ({TMP})")

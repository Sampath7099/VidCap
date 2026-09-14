"""Metrics / data / decoding gates: python test_pipeline.py"""
import math
import os
import tempfile

import numpy as np

TMP = tempfile.mkdtemp(prefix="vidcap_pipe_")
os.environ["VIDCAP_DATA"], os.environ["VIDCAP_OUT"] = f"{TMP}/data", f"{TMP}/out"

import torch  # noqa: E402
from vidcap import metrics  # noqa: E402
from vidcap.config import VISION_DIM  # noqa: E402
from vidcap.data import VideoCaptionDataset, make_collate, uniform_indices  # noqa: E402


def test_bleu():
    # exact match -> 1.0 (all precisions 1, no brevity penalty)
    h = ["a man is riding a horse on the beach"]
    assert abs(metrics.bleu(h, [h]) - 1.0) < 1e-9, metrics.bleu(h, [h])
    # no shared 4-gram -> 0
    assert metrics.bleu(["the cat sat on a mat"], [["dogs run through tall wet grass"]]) == 0.0
    # hand-computable: hyp is a 4-word prefix of an 8-word ref.
    # every n-gram matches, so p_n = 1; BP = exp(1 - 8/4)
    hy, rf = ["a b c d"], [["a b c d e f g h"]]
    assert abs(metrics.bleu(hy, rf) - math.exp(1 - 8 / 4)) < 1e-9, metrics.bleu(hy, rf)
    # repetition must be clipped, not rewarded
    assert metrics.bleu(["a a a a a a"], [["a b c d e f"]]) < 0.3
    print("bleu ok")


def test_rouge_l():
    h = ["a man is cooking"]
    assert abs(metrics.rouge_l(h, [h]) - 1.0) < 1e-9
    assert metrics.rouge_l(["x y z"], [["a b c"]]) == 0.0
    # LCS is order-sensitive: same words, wrong order scores lower than exact
    assert metrics.rouge_l(["cooking is man a"], [["a man is cooking"]]) < 1.0
    # takes the best reference, not the first
    assert metrics.rouge_l(["a man is cooking"], [["totally unrelated", "a man is cooking"]]) > 0.99
    print("rouge_l ok")


def test_cider():
    refs = [["a man is playing guitar"], ["a dog runs in the park"],
            ["a woman is cooking food"], ["a car drives down the road"]]
    exact = [r[0] for r in refs]
    wrong = ["completely unrelated nonsense here"] * 4
    s_exact, s_wrong = metrics.cider_d(exact, refs), metrics.cider_d(wrong, refs)
    assert s_exact > s_wrong, (s_exact, s_wrong)
    assert s_wrong < 0.1, s_wrong
    # idf must down-weight ngrams common to every reference: a caption made only of the
    # most common word should score far below an exact match
    common = [["a a a a"], ["a a a a"], ["a a a a"], ["a a a a"]]
    assert metrics.cider_d(["a a a a"], common) < s_exact
    print(f"cider ok (exact {s_exact:.2f} vs wrong {s_wrong:.3f})")


def test_uniform_indices():
    assert uniform_indices(20, 4) == [0, 6, 13, 19]
    assert uniform_indices(8, 8) == list(range(8))
    assert uniform_indices(3, 5) == [0, 1, 2, 2, 2]   # pads, never crashes, never wrong length
    assert len(uniform_indices(1, 8)) == 8
    print("uniform_indices ok")


def _fake_cache(dataset, ids, n_frames=40):
    from vidcap.encoder import cache_path
    for v in ids:
        p = cache_path(dataset, v)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            np.savez_compressed(f, emb=np.random.randn(n_frames, VISION_DIM).astype(np.float32),
                                times=np.arange(n_frames, dtype=np.float32))


def test_selector_protocol():
    """Every entry in evaluate.SELECTORS is called as sel(emb, k) by eval_batches, so all
    of them must take the pool array — not a pool size. Registering a size-taking function
    here raised 'truth value of an array is ambiguous' only once that arm actually ran."""
    from scripts.evaluate import SELECTORS
    emb = np.random.randn(20, 16).astype(np.float32)
    for name, sel in SELECTORS.items():
        idx = sel(emb, 4)
        assert len(idx) == 4, f"{name} returned {len(idx)} indices for k=4"
        assert all(isinstance(i, int) and 0 <= i < len(emb) for i in idx), f"{name}: {idx}"
        short = sel(emb[:2], 4)          # pool smaller than the budget must still pad to k
        assert len(short) == 4, f"{name} gave {len(short)} for a 2-frame pool"
    print(f"selector protocol ok ({', '.join(SELECTORS)})")


def test_dataset_and_collate():
    from transformers import AutoTokenizer
    from vidcap.config import LLM_MODEL
    ids = [f"video{i}" for i in range(6)]
    _fake_cache("msrvtt", ids)
    recs = [{"video_id": v, "path": None, "split": "train",
             "captions": [f"caption {v} one", f"caption {v} two"]} for v in ids]

    ds = VideoCaptionDataset("msrvtt", recs, k=8)
    frames, cap = ds[0]
    assert frames.shape == (8, VISION_DIM), frames.shape
    assert cap.startswith("caption video0")

    tok = AutoTokenizer.from_pretrained(LLM_MODEL)
    tok.pad_token = tok.pad_token or tok.eos_token
    f, i, m = make_collate(tok)([ds[j] for j in range(4)])
    assert f.shape == (4, 8, VISION_DIM) and i.shape[0] == 4 and i.shape == m.shape
    # every row must end in eos, or the model never learns to stop
    assert all(row[mask.bool()][-1].item() == tok.eos_token_id for row, mask in zip(i, m)), \
        "collate dropped the eos token"
    # eval mode must be deterministic (same caption every call) for reproducible metrics
    de = VideoCaptionDataset("msrvtt", recs, k=8, train=False)
    assert de[0][1] == de[0][1] == recs[0]["captions"][0]
    print(f"dataset+collate ok (batch {tuple(f.shape)}, tokens {tuple(i.shape)})")


def test_beam1_equals_greedy():
    """Phase 7 exit criterion."""
    from vidcap.decode import beam_search, greedy
    from vidcap.model import VideoCaptioner
    torch.manual_seed(0)
    m = VideoCaptioner(connector="resampler", lora_r=0).eval()
    frames = torch.randn(2, 8, VISION_DIM)
    g = greedy(m, frames, max_new_tokens=10)
    b1 = beam_search(m, frames, beam=1, max_new_tokens=10)
    assert g == b1, f"beam=1 != greedy:\n  greedy {g}\n  beam1  {b1}"
    b4 = beam_search(m, frames, beam=4, max_new_tokens=10)
    print(f"beam1==greedy ok\n  greedy: {g[0]!r}\n  beam4:  {b4[0]!r}")


if __name__ == "__main__":
    test_bleu()
    test_rouge_l()
    test_cider()
    test_uniform_indices()
    test_selector_protocol()
    test_dataset_and_collate()
    test_beam1_equals_greedy()
    print(f"\npipeline gates passed ({TMP})")

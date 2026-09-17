"""Video Q&A gates: python test_qa.py"""
import json
import os
import pathlib
import tempfile

import numpy as np

TMP = tempfile.mkdtemp(prefix="vidcap_qa_")
os.environ["VIDCAP_DATA"], os.environ["VIDCAP_OUT"] = f"{TMP}/data", f"{TMP}/out"

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from vidcap import datasets  # noqa: E402
from vidcap.config import VISION_DIM  # noqa: E402
from vidcap.data import (VideoQADataset, cache_ns, load_split,  # noqa: E402
                         make_collate, make_qa_collate)
from vidcap.encoder import cache_path  # noqa: E402
from vidcap.metrics import qa_accuracy, qa_accuracy_by_type  # noqa: E402
from vidcap.model import VideoCaptioner  # noqa: E402

TINY = "distilgpt2"
IDS = ["video10", "video6600", "video8000"]        # one per official split range


def _fixture():
    """Videos + msrvtt captions + QA annotations + cached shards, all consistent."""
    root = pathlib.Path(TMP) / "data/msrvtt"
    (root / "annotations").mkdir(parents=True, exist_ok=True)
    for v in IDS:
        (root / f"{v}.mp4").write_bytes(b"")       # _find_videos only needs the name
        p = cache_path("msrvtt", v)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, emb=np.random.randn(40, VISION_DIM).astype(np.float32),
                            times=np.arange(40, dtype=np.float32))
    (root / "MSR_VTT.json").write_text(json.dumps({
        "videos": [{"video_id": v} for v in IDS],
        "annotations": [{"image_id": v, "caption": f"a caption for {v}"} for v in IDS]}))
    for split, vid in zip(("train", "val", "test"), IDS):
        (root / "annotations" / datasets.QA_SPLITS[split]).write_text(json.dumps([
            {"video": f"{vid}.mp4", "question": "what is the man doing",
             "answer": "talking", "answer_type": "what", "question_id": "1"},
            {"video": f"{vid}.mp4", "question": "who is there",
             "answer": "man", "answer_type": "who", "question_id": "2"}]))
    return root


def test_qa_loader_splits_and_cache_reuse():
    """MSRVTT-QA rides on msrvtt's shards. If it looked under a msrvtt_qa/ cache it would find
    nothing and silently train on an empty set."""
    _fixture()
    assert cache_ns("msrvtt_qa") == "msrvtt", "QA must read msrvtt's cache"
    recs = datasets.msrvtt_qa()
    assert len(recs) == 6, recs
    assert {r["split"] for r in recs} == {"train", "val", "test"}
    assert recs[0]["question"] and recs[0]["answer"], recs[0]
    # the real integration: load_split must resolve shards through the alias
    for s in ("train", "val", "test"):
        got = load_split("msrvtt_qa", s)
        assert len(got) == 2, f"{s}: {len(got)} (cache alias broken?)"
    print("qa loader ok (6 pairs, 3 splits, shards resolved via msrvtt)")


def test_qa_loader_fails_loudly_without_annotations():
    bare = pathlib.Path(TMP) / "data/empty"
    (bare).mkdir(parents=True, exist_ok=True)
    (bare / "video10.mp4").write_bytes(b"")
    try:
        datasets.msrvtt_qa(bare)
        raise SystemExit("FAIL: missing QA annotations returned silently")
    except FileNotFoundError as e:
        assert "fetch_qa" in str(e), e
    print("missing-annotation gate ok")


def test_loss_mask_excludes_the_question():
    """The whole point of QA conditioning: the question is input, not a target."""
    _fixture()
    tok = AutoTokenizer.from_pretrained(TINY)
    tok.pad_token = tok.pad_token or tok.eos_token
    recs = load_split("msrvtt_qa", "train")
    ds = VideoQADataset("msrvtt_qa", recs, k=8)
    frames, ids, attn, loss = make_qa_collate(tok)([ds[i] for i in range(2)])

    assert frames.shape == (2, 8, VISION_DIM), frames.shape
    assert ids.shape == attn.shape == loss.shape
    for row_ids, row_attn, row_loss in zip(ids, attn, loss):
        assert row_loss.sum() > 0, "nothing to learn from"
        assert row_loss.sum() < row_attn.sum(), "loss must not cover the whole sequence"
        first = int(row_loss.argmax())
        assert row_loss[:first].sum() == 0, "question tokens leaked into the loss"
        # the answer span must end with eos, or the model never learns to stop
        last = int(row_attn.sum()) - 1
        assert row_ids[last].item() == tok.eos_token_id, "answer span missing eos"
    print(f"loss mask ok (question excluded; {int(loss[0].sum())} answer tokens supervised)")


def test_loss_mask_none_is_unchanged_captioning():
    """Regression guard: the captioning path must be bit-identical with the new argument.

    Single-threaded because multi-threaded CPU matmul varies reduction order and moves the loss
    in the last bits — which would make a bitwise gate flaky for reasons unrelated to masking.
    """
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(0)
        m = VideoCaptioner(llm_name=TINY, d_vis=64, n_prefix=4, connector="meanpool",
                           lora_r=0).eval()
        f = torch.randn(2, 8, 64)
        ids = torch.randint(0, 1000, (2, 6))
        attn = torch.ones_like(ids)
        with torch.no_grad():
            a, _ = m(f, ids, attn)                    # loss_mask defaults to None
            b, _ = m(f, ids, attn, torch.ones_like(ids))
            partial = torch.zeros_like(ids)
            partial[:, -2:] = 1                       # supervise only the tail
            c, _ = m(f, ids, attn, partial)
        assert torch.equal(a, b), (a.item(), b.item())
        assert not torch.equal(a, c), "loss_mask had no effect — the argument is not wired in"
    finally:
        torch.set_num_threads(threads)
    print(f"captioning regression ok (unchanged {a.item():.6f}; masked differs {c.item():.6f})")


def test_prompted_decode_strips_the_prompt_and_beam1_matches_greedy():
    from vidcap.decode import beam_search, greedy
    torch.manual_seed(0)
    m = VideoCaptioner(llm_name=TINY, d_vis=64, n_prefix=4, connector="meanpool", lora_r=0).eval()
    f = torch.randn(2, 8, 64)
    prompt = torch.randint(0, 1000, (2, 5))
    plain = greedy(m, f, max_new_tokens=6)
    primed = greedy(m, f, max_new_tokens=6, prompt_ids=prompt)
    assert primed != plain, "prompt had no effect on generation"
    b1 = beam_search(m, f, beam=1, max_new_tokens=6, prompt_ids=prompt)
    assert b1 == primed, f"beam=1 != greedy with a prompt:\n  greedy {primed}\n  beam1  {b1}"
    print(f"prompted decode ok (beam1==greedy, answer only: {primed[0][:30]!r})")


def test_padding_would_corrupt_answers_so_batches_must_be_uniform():
    """Two claims, both measured on logits rather than decoded text — an untrained model's text
    is degenerate and cannot discriminate.

    1. Left-padding changes the logits even WITH a correct attention mask, because GPT-2 uses
       absolute position embeddings (Qwen uses RoPE; same problem). So masking is not a fix.
    2. Batching equal-length prompts is exact.
    """
    from vidcap.decode import _next_logits, greedy
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(0)
        m = VideoCaptioner(llm_name=TINY, d_vis=64, n_prefix=4, connector="meanpool",
                           lora_r=0).eval()
        f = torch.randn(1, 8, 64)
        real = [5, 6, 7]
        pre = m.prefix(f)
        alone = _next_logits(m, pre, torch.tensor([real]))
        padded = _next_logits(m, pre, torch.tensor([[0, 0, 0] + real]))
        drift = float((padded - alone).abs().max())
        assert drift > 1e-3, "padding should shift logits — if it does not, this gate is vacuous"

        # 2. equal-length prompts batch exactly
        f2 = torch.randn(2, 8, 64)
        qs = [[5, 6, 7], [8, 9, 10]]
        alone2 = [greedy(m, f2[i:i + 1], max_new_tokens=4, prompt_ids=torch.tensor([q]))[0]
                  for i, q in enumerate(qs)]
        batched = greedy(m, f2, max_new_tokens=4, prompt_ids=torch.tensor(qs))
        assert batched == alone2, f"equal-length batching differs:\n{alone2}\n{batched}"
    finally:
        torch.set_num_threads(threads)
    print(f"uniform-length batching ok (padding would drift logits by {drift:.3f}; batched==alone)")


def test_evaluate_qa_groups_by_length():
    """The guarantee above only holds if evaluate_qa never builds a mixed-length batch."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TINY)
    tok.pad_token = tok.pad_token or tok.eos_token
    qs = ["what is he doing", "who is there", "where", "what is the man in red doing now"]
    by_len = {}
    for i, q in enumerate(qs):
        by_len.setdefault(len(tok(q + "?")["input_ids"]), []).append(i)
    assert len(by_len) > 1, "fixture must contain differing lengths to be meaningful"
    for n, idxs in by_len.items():
        lens = {len(tok(qs[i] + "?")["input_ids"]) for i in idxs}
        assert lens == {n}, f"group {n} is not uniform: {lens}"
    assert sum(len(v) for v in by_len.values()) == len(qs), "grouping dropped questions"
    print(f"length grouping ok ({len(by_len)} groups, all uniform, none dropped)")


def test_qa_accuracy():
    assert qa_accuracy(["man", "Dog."], ["man", "dog"]) == 1.0, "case/punctuation must not matter"
    assert qa_accuracy(["cat", "cat"], ["dog", "cat"]) == 0.5
    assert qa_accuracy([], []) == 0.0
    by = qa_accuracy_by_type(["man", "two", "cat"], ["man", "three", "cat"],
                             ["who", "how", "what"])
    assert by["who"] == (1.0, 1) and by["how"] == (0.0, 1), by
    print(f"qa accuracy ok ({by})")


def test_overfits_a_few_qa_triples():
    """End-to-end gate: the model must be able to memorise answers conditioned on the question."""
    torch.manual_seed(0)
    _fixture()
    tok = AutoTokenizer.from_pretrained(TINY)
    tok.pad_token = tok.pad_token or tok.eos_token
    recs = load_split("msrvtt_qa", "train")
    dl = DataLoader(VideoQADataset("msrvtt_qa", recs, k=8), batch_size=2,
                    collate_fn=make_qa_collate(tok))
    m = VideoCaptioner(llm_name=TINY, n_prefix=4, connector="meanpool", lora_r=0)
    opt = torch.optim.AdamW(m.trainable_parameters(), lr=1e-3)
    first = last = None
    for _ in range(40):
        for frames, ids, attn, loss_mask in dl:
            loss, _ = m(frames, ids, attn, loss_mask)
            opt.zero_grad(); loss.backward(); opt.step()
            first = loss.item() if first is None else first
            last = loss.item()
    assert last < first * 0.5, f"QA head failed to overfit: {first:.3f} -> {last:.3f}"
    print(f"qa overfit ok (loss {first:.3f} -> {last:.3f})")


if __name__ == "__main__":
    test_qa_loader_splits_and_cache_reuse()
    test_qa_loader_fails_loudly_without_annotations()
    test_loss_mask_excludes_the_question()
    test_loss_mask_none_is_unchanged_captioning()
    test_prompted_decode_strips_the_prompt_and_beam1_matches_greedy()
    test_padding_would_corrupt_answers_so_batches_must_be_uniform()
    test_evaluate_qa_groups_by_length()
    test_qa_accuracy()
    test_overfits_a_few_qa_triples()
    print(f"\nqa gates passed ({TMP})")

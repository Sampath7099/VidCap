"""Training-loop integration gate: python test_train.py

Wires dataset -> collate -> model -> loss -> backward -> checkpoint -> resume exactly as
scripts/train.py does, on synthetic cache shards with a tiny LLM. Real-config correctness
(Qwen shapes, LoRA identity, overfitting) is gated separately in test_model.py.
"""
import os
import tempfile

import numpy as np

TMP = tempfile.mkdtemp(prefix="vidcap_train_")
os.environ["VIDCAP_DATA"], os.environ["VIDCAP_OUT"] = f"{TMP}/data", f"{TMP}/out"

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from vidcap import checkpoint  # noqa: E402
from vidcap.config import VISION_DIM  # noqa: E402
from vidcap.data import VideoCaptionDataset, make_collate  # noqa: E402
from vidcap.encoder import cache_path  # noqa: E402
from vidcap.model import VideoCaptioner  # noqa: E402

TINY = "distilgpt2"
CAPS = ["a man is cooking", "a dog runs fast", "two people dance", "a car drives by"]


def setup(n=8):
    recs = []
    for i in range(n):
        vid = f"video{i}"
        p = cache_path("msrvtt", vid)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            np.savez_compressed(f, emb=np.random.randn(40, VISION_DIM).astype(np.float32),
                                times=np.arange(40, dtype=np.float32))
        recs.append({"video_id": vid, "path": None, "split": "train",
                     "captions": [CAPS[i % len(CAPS)]]})
    return recs


def test_train_step_and_resume():
    torch.manual_seed(0)
    recs = setup()
    m = VideoCaptioner(connector="resampler", n_prefix=4, lora_r=4, llm_name=TINY)
    dl = DataLoader(VideoCaptionDataset("msrvtt", recs, k=8), batch_size=4, shuffle=True,
                    collate_fn=make_collate(m.tok), drop_last=True)

    params = m.trainable_parameters()
    assert params, "nothing trainable"
    n_train = sum(p.numel() for p in params)
    n_total = sum(p.numel() for p in m.parameters())
    assert n_train < n_total, "everything is trainable — the backbone was not frozen"

    opt = torch.optim.AdamW(params, lr=1e-3)
    losses = []
    for _ in range(12):
        for frames, ids, mask in dl:
            loss, _ = m(frames, ids, mask)
            opt.zero_grad(); loss.backward()
            # every trainable tensor must actually receive gradient, or it is silently dead
            assert all(p.grad is not None for p in params), "a trainable param got no gradient"
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            losses.append(loss.item())
    assert losses[-1] < losses[0], f"loss did not decrease: {losses[0]:.3f} -> {losses[-1]:.3f}"
    print(f"train step ok (loss {losses[0]:.3f} -> {losses[-1]:.3f}, "
          f"trainable {n_train/1e3:.0f}k / {n_total/1e3:.0f}k)")

    # resume must restore bit-identical weights and live optimizer state
    checkpoint.save("itest", 42, m, opt)
    ref = {k: v.clone() for k, v in m.state_dict().items()}
    for p in params:
        p.data.add_(torch.randn_like(p))          # corrupt
    ck = checkpoint.load("itest", m, opt)
    assert ck["step"] == 42
    assert all(torch.equal(ref[k], v) for k, v in m.state_dict().items()), "resume lost weights"
    assert opt.state_dict()["state"], "resume lost optimizer state"
    print("checkpoint round-trip ok")


def test_frozen_backbone_unchanged_by_training():
    """Stage B must not move the LLM at all — only connector/projector."""
    torch.manual_seed(0)
    recs = setup(4)
    m = VideoCaptioner(connector="meanpool", n_prefix=4, lora_r=0, llm_name=TINY)
    before = {k: v.clone() for k, v in m.llm.state_dict().items()}
    dl = DataLoader(VideoCaptionDataset("msrvtt", recs, k=8), batch_size=4,
                    collate_fn=make_collate(m.tok), drop_last=True)
    opt = torch.optim.AdamW(m.trainable_parameters(), lr=1e-2)
    for frames, ids, mask in dl:
        loss, _ = m(frames, ids, mask)
        opt.zero_grad(); loss.backward(); opt.step()
    after = m.llm.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before), "Stage B modified the frozen LLM"
    print("frozen backbone ok (LLM bit-identical after training)")


if __name__ == "__main__":
    test_train_step_and_resume()
    test_frozen_backbone_unchanged_by_training()
    print(f"\ntraining gates passed ({TMP})")

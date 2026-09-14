"""Vanilla-captioner gates: python test_model.py

Light by default: everything that does not specifically need the real decoder runs on a tiny
LLM. Only the LoRA-target gate loads Qwen, because target module names differ per architecture
and that is exactly what it must catch.

  VIDCAP_HEAVY=1 python test_model.py    # also run the real-config overfit (GPU/Kaggle)

Loading Qwen2.5-1.5B in fp32 costs ~6.2GB each time. Do not instantiate VideoCaptioner in a
loop — that is what hung a 15GB laptop with zram swap.
"""
import os

import torch

from vidcap.config import LLM_MODEL, VISION_DIM
from vidcap.lora import apply_lora, lora_parameters
from vidcap.model import CONNECTORS, Projector, VideoCaptioner, uniform_select

torch.manual_seed(0)
D = VISION_DIM
TINY = "distilgpt2"  # small but real; tiny-gpt2 has hidden_size=2 and cannot learn
HEAVY = os.environ.get("VIDCAP_HEAVY") == "1"


def require_ram(gb, what):
    """Fail loudly instead of thrashing zram into a system hang."""
    try:
        with open("/proc/meminfo") as f:
            avail = next(int(l.split()[1]) for l in f if l.startswith("MemAvailable")) / 2**20
    except Exception:
        return
    if avail < gb:
        raise SystemExit(f"need ~{gb:.1f}GB free for {what}, only {avail:.1f}GB available. "
                         f"Close something or run this on Kaggle.")


def test_lora_targets_match_real_decoder():
    """Must use the REAL decoder: LoRA target names are architecture-specific, and a target
    that matches nothing yields zero adapters while every other check still passes."""
    from transformers import AutoModelForCausalLM
    require_ram(7.0, f"{LLM_MODEL} in fp32")
    base = AutoModelForCausalLM.from_pretrained(LLM_MODEL).eval()
    ids = torch.randint(0, 1000, (2, 12))
    with torch.no_grad():
        before = base(ids).logits.clone()

    n = apply_lora(base, r=8)
    assert n > 0, f"no LoRA targets matched {LLM_MODEL} — adapters would train nothing"
    with torch.no_grad():
        assert torch.equal(before, base(ids).logits), "zero-init LoRA perturbed the base model"

    for p in lora_parameters(base):
        if p.dim() == 2 and p.abs().sum() == 0:
            p.data.normal_(0, 0.02)
    with torch.no_grad():
        assert not torch.equal(before, base(ids).logits), "LoRA has no effect — it is dead"
    assert all(not p.requires_grad for k, p in base.named_parameters()
               if not k.endswith((".A", ".B"))), "base weights left trainable"
    print(f"lora ok ({n} adapters on {LLM_MODEL}, base frozen, zero-init exact)")
    del base


def test_connectors_shapes():
    """No LLM needed — this is a tensor-shape contract on connector + projector."""
    f = torch.randn(3, 12, D)
    for name, cls in CONNECTORS.items():
        conn = cls() if name == "meanpool" else cls(D)
        expand = 8 if name == "meanpool" else 1
        proj = Projector(D, 1536, expand)
        pre = proj(conn(f))
        assert pre.shape[0] == 3 and pre.shape[2] == 1536, pre.shape
        assert pre.shape[1] > 0
        print(f"  {name}: {tuple(f.shape)} -> {tuple(pre.shape)}")
    print("connectors ok")


def test_uniform_select():
    e = torch.arange(20).float().unsqueeze(1).repeat(1, D)
    sel, idx = uniform_select(e, 4)
    assert idx == [0, 6, 13, 19], idx
    assert sel.shape == (4, D)
    sel2, idx2 = uniform_select(e[:3], 8)
    assert idx2 == [0, 1, 2] and sel2.shape == (3, D)
    print(f"uniform_select ok (k=4 of 20 -> {idx}; k=8 of 3 -> {idx2})")


def test_blind_control_ignores_video():
    m = VideoCaptioner(connector="resampler", lora_r=0, llm_name=TINY, blind=True).eval()
    with torch.no_grad():
        a, b = m.prefix(torch.randn(2, 8, D)), m.prefix(torch.randn(2, 8, D))
    assert torch.equal(a, b), "blind model still leaks visual information"
    m.blind = False
    with torch.no_grad():
        a, b = m.prefix(torch.randn(2, 8, D)), m.prefix(torch.randn(2, 8, D))
    assert not torch.equal(a, b), "sighted model ignores its frames"
    print("blind control ok")


def test_overfits_tiny_batch(llm_name=TINY):
    """Masking, labels and the gradient path end to end."""
    m = VideoCaptioner(connector="resampler", n_prefix=8, lora_r=8, llm_name=llm_name)
    caps = ["a man is cooking pasta", "a dog runs on grass",
            "two people are dancing", "a car drives at night"]
    frames = torch.randn(4, 8, D)
    b = m.tok(caps, return_tensors="pt", padding=True)
    opt = torch.optim.AdamW(m.trainable_parameters(), lr=1e-3)

    first = None
    for step in range(300):
        loss, _ = m(frames, b["input_ids"], b["attention_mask"])
        opt.zero_grad(); loss.backward(); opt.step()
        first = first if first is not None else loss.item()
        if loss.item() < first * 0.2:
            break
    assert loss.item() < first * 0.25, f"did not overfit: {first:.3f} -> {loss.item():.3f}"
    print(f"overfit ok on {llm_name} ({step+1} steps, loss {first:.3f} -> {loss.item():.3f})")
    del m


if __name__ == "__main__":
    test_connectors_shapes()
    test_uniform_select()
    test_blind_control_ignores_video()
    test_overfits_tiny_batch()
    test_lora_targets_match_real_decoder()
    if HEAVY:
        require_ram(9.0, "real-config overfit")
        test_overfits_tiny_batch(LLM_MODEL)
    else:
        print("(skipped real-config overfit; VIDCAP_HEAVY=1 to run it, best on GPU)")
    print("\nvanilla captioner gates passed.")

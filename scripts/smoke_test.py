"""Phase 0 gate: both frozen backbones load and produce sane output. No training."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from vidcap.encoder import embed_images, embed_texts, load_vision, normalize
from vidcap.config import LLM_MODEL

import numpy as np


def check_vision():
    model, proc, device = load_vision()
    # A red square and a blue square, described correctly and incorrectly.
    red = np.zeros((224, 224, 3), np.uint8); red[..., 0] = 220
    blue = np.zeros((224, 224, 3), np.uint8); blue[..., 2] = 220
    img = normalize(embed_images(model, proc, device, [red, blue]))
    txt = normalize(embed_texts(model, proc, device, ["a red image", "a blue image"]))
    sim = img @ txt.T
    assert sim[0, 0] > sim[0, 1] and sim[1, 1] > sim[1, 0], f"vision-text alignment wrong:\n{sim}"
    print(f"vision ok on {device}  dim={img.shape[1]}  diag-sim={np.diag(sim).round(3)}")


def check_llm():
    tok = AutoTokenizer.from_pretrained(LLM_MODEL)
    llm = AutoModelForCausalLM.from_pretrained(LLM_MODEL).eval()
    ids = tok("A man is playing", return_tensors="pt").input_ids
    with torch.no_grad():
        out = llm.generate(ids, max_new_tokens=8, do_sample=False)
    n_attn = sum(1 for n, _ in llm.named_modules() if n.endswith("attn.c_attn"))
    assert n_attn > 0, "no attention projections found — LoRA (Phase 3) needs these"
    print(f"{LLM_MODEL} ok  {sum(p.numel() for p in llm.parameters())/1e6:.0f}M params  "
          f"{n_attn} LoRA-targetable projections")
    print(f"  sample: {tok.decode(out[0], skip_special_tokens=True)!r}")


if __name__ == "__main__":
    check_vision()
    check_llm()
    print("\nPhase 0 backbones ready.")

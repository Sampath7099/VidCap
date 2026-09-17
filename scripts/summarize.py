"""Video in, short paragraph out.

  python -m scripts.summarize clip.mp4
  python -m scripts.summarize clip.mp4 --ask "what colour is the car?" --k 4

Composed, not generated: the caption supplies the main sentence, and a fixed probe set is put to
the Q&A model to add facts it does not already state.

Honest about what this is — MSR-VTT's ~20 captions per clip are near-paraphrases, not
complementary sentences, so there is NO multi-sentence ground truth to train or score against.
This is a qualitative deliverable. Do not attach a number to it.
"""
import argparse
from pathlib import Path

import torch

from scripts.caption import caption_from_pool
from scripts.evaluate import load_model, make_learned, uniform_sel
from vidcap.config import POOL_FPS
from vidcap.decode import greedy
from vidcap.encoder import embed_images, load_vision
from vidcap.metrics import _norm_answer
from vidcap.video import sample_frames

# Chosen to match MSRVTT-QA's answer types (what 68%, who 28%, then how/where), so the model is
# asked the kind of question it was actually trained on.
PROBES = [
    ("who", "who is in the video?"),
    ("where", "where is this happening?"),
    ("what", "what is the main object?"),
    ("how many", "how many people are there?"),
]


def ask(model, frames, question, max_new_tokens=6):
    q = question.strip()
    ids = model.tok(q if q.endswith("?") else q + "?", return_tensors="pt")["input_ids"]
    return greedy(model, frames, max_new_tokens=max_new_tokens,
                  prompt_ids=ids.to(frames.device))[0].strip()


def summarize(emb, k, selector, cap_model, qa_model, device, probes=PROBES, extra=()):
    """-> (caption, [(label, answer)]). Answers already implied by the caption are dropped, so
    the summary adds information rather than restating it."""
    caption, idx = caption_from_pool(emb, k, selector, cap_model, device)
    seen = set(_norm_answer(caption).split())

    frames = torch.from_numpy(emb[idx]).float()[None].to(device)
    facts = []
    for label, q in list(probes) + [(q, q) for q in extra]:
        a = ask(qa_model, frames, q)
        norm = _norm_answer(a)
        if not norm or norm in seen:
            continue                      # already said by the caption, or empty
        seen.add(norm)
        facts.append((label, a))
    return caption, facts, idx


def render(caption, facts):
    # A caption with no letters or digits (an undertrained model can emit just "." ) is not a
    # sentence — say so rather than rendering a lone full stop.
    c = caption.strip()
    out = c[:1].upper() + c[1:] if any(ch.isalnum() for ch in c) else "(no caption)"
    if not out.endswith("."):
        out += "."
    if facts:
        out += " " + " ".join(f"{label.capitalize()}: {a}." for label, a in facts)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--ckpt", default="stageB", help="captioning checkpoint")
    ap.add_argument("--qa-ckpt", default="qaB", help="Q&A checkpoint")
    ap.add_argument("--scorer", default=None, help="use learned frame selection")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--fps", type=float, default=POOL_FPS["msrvtt"])
    ap.add_argument("--ask", action="append", default=[], help="extra question; repeatable")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cap_model, _ = load_model(args.ckpt, device)
    qa_model, meta = load_model(args.qa_ckpt, device)
    if meta.get("task") != "qa":
        print(f"warning: --qa-ckpt '{args.qa_ckpt}' was trained for "
              f"task={meta.get('task', 'unknown')!r}; answers will be unreliable")
    selector = make_learned(args.scorer, device) if args.scorer else uniform_sel
    vis, proc, _ = load_vision(device)

    for path in args.videos:
        p = Path(path)
        if not p.exists():
            print(f"{p}: no such file")
            continue
        frames, times = sample_frames(p, fps=args.fps)
        if len(frames) == 0:
            print(f"{p}: could not decode")
            continue
        emb = embed_images(vis, proc, device, frames)
        caption, facts, idx = summarize(emb, args.k, selector, cap_model, qa_model,
                                        device, extra=args.ask)
        picked = ", ".join(f"{times[i]:.1f}s" for i in idx)
        print(f"\n{p.name}  ({len(frames)} frames, {times[-1]:.1f}s, K={args.k} at [{picked}])")
        print(f"  {render(caption, facts)}")


if __name__ == "__main__":
    main()

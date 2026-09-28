"""Watch a video: timeline of what happened, a summary, then questions about it.

  python -m scripts.watch clip.mp4                         # timeline + summary, then ask away
  python -m scripts.watch clip.mp4 --ask "what is the man holding?" --json out.json
  python -m scripts.watch clip.mp4 --budget 8          # 8 frames for the whole video, in pairs

Each scene is captioned from frames the learned selector picks. The frozen Qwen2.5-Instruct
decoder, used as a plain text model, writes the summary and answers questions over the timeline;
the Q&A adapter answers from the frames of the scene the question best matches.
"""
import argparse
import json
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from scripts.caption import caption_from_pool
from scripts.evaluate import load_model, make_learned, make_learned_scores, uniform_sel
from scripts.summarize import ask
from vidcap import checkpoint
from vidcap.data import uniform_indices
from vidcap.decode import greedy
from vidcap.encoder import embed_images, embed_texts, load_vision, normalize
from vidcap.timeline import (fmt_time, merge_repeats, pair_frames, segment, segment_best,
                             source_event, timeline_text)
from vidcap.video import sample_frames

SUMMARY_PROMPT = (
    "You summarise videos from a timeline of automatically generated scene captions. The captions "
    "are short and may be generic or partly wrong. Write 2-4 sentences describing what happens, "
    "in order. Use only what the timeline says; do not invent names, objects or events.")
# Tested on Qwen2.5-1.5B: asked to cite times it copied the wrong range; a numbered-event format
# was worse. The worked example lifts content accuracy, and the time is attached in code instead.
NOT_FOUND = "not in the video"
QA_PROMPT = (
    "You answer questions about a video from its timeline. Each line is [start-end] followed by "
    "what happens then. Reply with one short sentence based on the timeline, and do not mention "
    f"times. If nothing in the timeline relates to the question, reply: {NOT_FOUND}\n\n"
    "Example timeline:\n[0:00-0:05] a dog runs in a park\n[0:05-0:09] a boy throws a ball\n"
    "Q: what does the boy do? A: The boy throws a ball.\n"
    "Q: what happens before the boy throws the ball? A: A dog runs in a park.\n"
    f"Q: what is the weather? A: {NOT_FOUND}")


def build_timeline(emb, times, model, selector, k, device):
    """-> merged events [{start, end, caption, frames, range}], one caption per scene.
    frames are the selected pool indices; range is the scene's [start, end) in the pool."""
    events = []
    for s, e in segment(emb, times):
        cap, idx = caption_from_pool(emb[s:e], k, selector, model, device)
        end = float(times[e]) if e < len(times) else float(times[-1])
        events.append({"start": float(times[s]), "end": end, "caption": " ".join(cap.split()),
                       "frames": sorted({s + int(i) for i in idx}), "range": [s, e]})
    return merge_repeats(events)


def build_budget_timeline(emb, times, model, idx, device):
    """One caption per time-ordered pair of the budget frames. Each event runs from its pair to
    the next, so events still tile the video for question routing."""
    pairs = pair_frames(sorted(set(int(i) for i in idx)))
    f = torch.from_numpy(np.ascontiguousarray(np.stack([emb[p] for p in pairs]))).float().to(device)
    caps = greedy(model, f)
    starts = [0] + [p[0] for p in pairs[1:]]
    ends = starts[1:] + [len(emb)]
    return merge_repeats([
        {"start": float(times[a]), "end": float(times[b]) if b < len(times) else float(times[-1]),
         "caption": " ".join(c.split()), "frames": sorted(set(p)), "range": [a, b]}
        for p, c, a, b in zip(pairs, caps, starts, ends)])


@torch.no_grad()
def chat(model, system, user, max_new_tokens=120):
    """The captioner's frozen decoder as a plain instruction-following text model."""
    tok = model.tok
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    text = (tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            if tok.chat_template else f"{system}\n\n{user}\n\n")
    ids = tok(text, return_tensors="pt").to(model.llm.device)
    out = model.llm.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
    return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def summarize_timeline(model, events):
    """One event needs no summarising — rewriting it only invites the LLM to embellish."""
    if len(events) <= 1:
        c = events[0]["caption"].strip() if events else ""
        return (c[:1].upper() + c[1:] + ".") if c else "(no caption)"
    return chat(model, SUMMARY_PROMPT, "Timeline:\n" + timeline_text(events))


@contextmanager
def adapter(model, ck, restore_ck):
    """Swap in another checkpoint's connector/projector, then put the original back.
    Both tasks share one frozen Qwen, so two tasks cost one decoder's RAM, not two."""
    checkpoint.restore(model, ck)
    try:
        yield model
    finally:
        checkpoint.restore(model, restore_ck)


def best_event(events, emb, q_emb):
    """The event holding the frame most similar to the question text (SigLIP space)."""
    sim = normalize(emb) @ normalize(q_emb[None])[0]
    return max(events, key=lambda ev: sim[ev["range"][0]:ev["range"][1]].max())


def ask_timeline(model, events, question):
    """-> {when, answer}; when is the event the answer came from, found in code, or None."""
    a = chat(model, QA_PROMPT, f"Timeline:\n{timeline_text(events)}\nQ: {question} A:", 40)
    # It writes times anyway ("appears at 0:07", wrong); the code-attached one is the real one.
    a = re.sub(r"\s*\b(?:at|from|around)?\s*\d+:\d{2}(?:\s*-\s*\d+:\d{2})?", "", a).strip()
    ev = None if NOT_FOUND in a.lower() else source_event(a, events)
    return {"when": f"{fmt_time(ev['start'])}-{fmt_time(ev['end'])}" if ev else None, "answer": a}


def answer(question, events, emb, model, qa_ck, cap_ck, vision, device):
    """-> {question, frames: {when, answer} | None, timeline: {when, answer}}."""
    out = {"question": question, "frames": None,
           "timeline": ask_timeline(model, events, question)}
    if qa_ck is not None:
        ev = best_event(events, emb, embed_texts(*vision, device, [question])[0])
        f = torch.from_numpy(np.ascontiguousarray(emb[ev["frames"]])).float()[None].to(device)
        with adapter(model, qa_ck, cap_ck):
            a = ask(model, f, question)
        out["frames"] = {"when": f"{fmt_time(ev['start'])}-{fmt_time(ev['end'])}", "answer": a}
    return out


def show_answer(a):
    for src in ("frames", "timeline"):
        if a[src]:
            when = f"[{a[src]['when']}]" if a[src]["when"] else ""
            print(f"  {src:8s} {when:13s} {a[src]['answer']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--ckpt", default="stageB", help="captioning checkpoint")
    ap.add_argument("--qa-ckpt", default="qaB", help="Q&A checkpoint; frame answers skipped if absent")
    ap.add_argument("--scorer", default="scorer", help="frame scorer; 'none' for uniform frames")
    ap.add_argument("--k", type=int, default=2, help="frames per scene (learned wins most at 1-2)")
    ap.add_argument("--budget", type=int, default=0,
                    help="K frames for the WHOLE video instead of scenes; 8 was the best measured "
                         "long-video setting on ActivityNet")
    ap.add_argument("--budget-select", choices=("uniform", "seg-learned"), default="uniform",
                    help="how --budget picks frames: evenly spaced, or the scorer's best per slice")
    ap.add_argument("--fps", type=float, default=1.0, help="candidate-pool density")
    ap.add_argument("--max-frames", type=int, default=128,
                    help="pool cap; SigLIP is ~3 s/frame on CPU, so this bounds the wait")
    ap.add_argument("--ask", action="append", default=[], help="question; repeatable")
    ap.add_argument("--no-chat", action="store_true", help="do not prompt for questions")
    ap.add_argument("--json", default=None, help="write timeline, summary and answers here")
    args = ap.parse_args()

    p = Path(args.video)
    if not p.exists():
        raise SystemExit(f"{p}: no such file")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()

    model, _ = load_model(args.ckpt, device)
    cap_ck = checkpoint.load(args.ckpt, map_location=device)
    qa_ck = checkpoint.load(args.qa_ckpt, map_location=device)
    if qa_ck is None:
        print(f"note: no '{args.qa_ckpt}' checkpoint — answering from the timeline only")
    elif qa_ck.get("task") != "qa":
        raise SystemExit(f"--qa-ckpt '{args.qa_ckpt}' is a {qa_ck.get('task', 'caption')} checkpoint")
    selector = uniform_sel if args.scorer == "none" else make_learned(args.scorer, device)
    vis, proc, _ = load_vision(device)

    frames, times = sample_frames(p, fps=args.fps, max_frames=args.max_frames)
    if len(frames) == 0:
        raise SystemExit(f"{p}: could not decode")
    print(f"{p.name}: {times[-1]:.0f}s, embedding {len(frames)} frames ...", flush=True)
    emb = embed_images(vis, proc, device, frames)
    del frames

    if args.budget:
        idx = (uniform_indices(len(emb), args.budget) if args.budget_select == "uniform" else
               segment_best(make_learned_scores(args.scorer, device)(emb), args.budget))
        events = build_budget_timeline(emb, times, model, idx, device)
    else:
        events = build_timeline(emb, times, model, selector, args.k, device)
    print(f"\nTimeline ({len(events)} events)\n{timeline_text(events)}")
    summary = summarize_timeline(model, events)
    print(f"\nSummary\n  {summary}\n\n[{time.time() - t0:.0f}s]")

    def qa(q):
        a = answer(q, events, emb, model, qa_ck, cap_ck, (vis, proc), device)
        show_answer(a)
        return a

    answers = []
    for q in args.ask:
        print(f"\n> {q}")
        answers.append(qa(q))
    if not args.no_chat and sys.stdin.isatty():
        print("\nAsk about the video (empty line to quit).")
        while q := input("> ").strip():
            answers.append(qa(q))

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"video": str(p), "events": events, "summary": summary, "answers": answers},
                      f, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()

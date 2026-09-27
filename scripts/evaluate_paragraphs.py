"""Long-video paragraphs on ActivityNet Captions: does frame selection help watch.py's timeline?

  python -m scripts.build_cache activitynet --limit 300
  python -m scripts.evaluate_paragraphs --ckpt stageB --scorer scorer --summaries

Every arm gets the same videos and the same frame budget per segment. The timeline paragraph is
the scene captions in order; the summary is watch.py's Qwen rewrite of that timeline. Both are
scored against the two human reference paragraphs (val_1, val_2) per video.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.evaluate import load_model, make_learned, make_oracle, motion_indices, uniform_sel
from scripts.watch import summarize_timeline
from vidcap.config import OUT
from vidcap.data import load_split
from vidcap.decode import greedy
from vidcap.encoder import load_cached
from vidcap.metrics import cider_d, evaluate
from vidcap.timeline import merge_repeats, segment

FIXED = {"z": float("inf"), "ratio": float("inf")}   # no scene cuts: plain 15 s windows


def caption_segments(model, jobs, k, select, device, batch=32):
    """jobs [(rec, emb, s, e)] -> one caption per job, in order. Batched across videos: one clip
    at a time made this ~16x slower than the budget-curve eval it has to sit beside."""
    caps = []
    for i in range(0, len(jobs), batch):
        frames = [torch.from_numpy(np.ascontiguousarray(emb[s:e][select(emb[s:e], k, rec)])).float()
                  for rec, emb, s, e in jobs[i:i + batch]]
        caps += greedy(model, torch.stack(frames).to(device))
    return [" ".join(c.split()) for c in caps]


def timelines(recs, pools, model, k, select, device, cuts):
    """-> per video, merged events [{start, end, caption, frames, range}] — watch.py's format."""
    jobs, owner = [], []
    for vi, (rec, (emb, times)) in enumerate(zip(recs, pools)):
        for s, e in segment(emb, times, **cuts):
            jobs.append((rec, emb, s, e))
            owner.append(vi)
    caps = caption_segments(model, jobs, k, select, device)
    out = [[] for _ in recs]
    for (rec, emb, s, e), vi, cap in zip(jobs, owner, caps):
        times = pools[vi][1]
        out[vi].append({"start": float(times[s]),
                        "end": float(times[e]) if e < len(times) else float(times[-1]),
                        "caption": cap, "frames": [], "range": [s, e]})
    return [merge_repeats(ev) for ev in out]


def paragraph(events):
    return " ".join(e["caption"].rstrip(".") + "." for e in events if e["caption"].strip())


def paired_ci(a, b, n_boot=2000, seed=0):
    """Mean of a - b over videos and its 95% paired-bootstrap interval."""
    d = np.asarray(a) - np.asarray(b)
    idx = np.random.default_rng(seed).integers(0, len(d), (n_boot, len(d)))
    lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
    return {"mean": float(d.mean()), "lo": float(lo), "hi": float(hi)}


def score(hyps, refs):
    s = evaluate(hyps, refs)
    s["per_video_cider"] = cider_d(hyps, refs, per_item=True)
    s["sentences"] = float(np.mean([h.count(".") for h in hyps]))
    s["words"] = float(np.mean([len(h.split()) for h in hyps]))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="stageB")
    ap.add_argument("--scorer", default="scorer")
    ap.add_argument("--k", type=int, default=2, help="frames per segment, as in watch.py")
    ap.add_argument("--summaries", action="store_true", help="also score Qwen summaries (slow)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    recs = load_split("activitynet", "val")   # build_cache --limit already spread the subset
    if not recs:
        raise SystemExit("no cached activitynet shards — run scripts/build_cache.py activitynet")
    pools = [load_cached("activitynet", r["video_id"]) for r in recs]
    refs = [r["paragraphs"] for r in recs]
    print(f"{len(recs)} videos, median {np.median([t[-1] for _, t in pools]):.0f}s, "
          f"{np.mean([len(r['paragraphs']) for r in recs]):.2f} reference paragraphs each")

    model, _ = load_model(args.ckpt, device)
    arms = {"fixed+uniform": (uniform_sel, FIXED), "scenes+uniform": (uniform_sel, {}),
            "scenes+motion": (motion_indices, {}),
            "scenes+learned": (make_learned(args.scorer, device), {}),
            "scenes+oracle": (make_oracle("activitynet"), {})}

    p = Path(args.out) if args.out else OUT / "eval_activitynet_paragraphs.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    res = {"n": len(recs), "k": args.k, "arms": {}, "examples": {}, "paired": {}}

    def save():
        with open(p, "w") as f:
            json.dump(res, f, indent=2)

    tls = {}
    for name, (sel, cuts) in arms.items():
        tls[name] = timelines(recs, pools, model, args.k, sel, device, cuts)
        hyps = [paragraph(ev) for ev in tls[name]]
        res["arms"][name] = score(hyps, refs)
        res["examples"][name] = hyps[:3]
        s = res["arms"][name]
        print(f"{name:15s} CIDEr-D {s['CIDEr-D']:.4f}  BLEU-4 {s['BLEU-4']:.4f}  "
              f"ROUGE-L {s['ROUGE-L']:.4f}  ({s['sentences']:.1f} sent, {s['words']:.0f} words)",
              flush=True)
        save()

    if args.summaries:
        for name in ("scenes+uniform", "scenes+learned"):
            hyps = [summarize_timeline(model, ev) for ev in tls[name]]
            res["arms"][name + " summary"] = score(hyps, refs)
            res["examples"][name + " summary"] = hyps[:3]
            s = res["arms"][name + " summary"]
            print(f"{name + ' summary':23s} CIDEr-D {s['CIDEr-D']:.4f}  BLEU-4 {s['BLEU-4']:.4f}  "
                  f"({s['sentences']:.1f} sent, {s['words']:.0f} words)", flush=True)
            save()

    a = res["arms"]
    for x, y in [("scenes+learned", "scenes+uniform"), ("scenes+learned", "scenes+motion"),
                 ("scenes+uniform", "fixed+uniform"), ("scenes+oracle", "scenes+uniform"),
                 ("scenes+learned summary", "scenes+uniform summary")]:
        if x in a and y in a:
            ci = paired_ci(a[x]["per_video_cider"], a[y]["per_video_cider"])
            res["paired"][f"{x} - {y}"] = ci
            print(f"CIDEr-D {x} - {y}: {ci['mean']:+.4f}  95% CI [{ci['lo']:+.4f}, {ci['hi']:+.4f}]")
    res["references"] = refs[:3]
    save()
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()

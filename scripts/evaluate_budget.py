"""Whole-video frame budget on ActivityNet: K frames for the entire video, then a paragraph.

  python -m scripts.evaluate_budget --budgets 4,8

Per-scene selection (evaluate_paragraphs.py) gave selection nothing to do: a scene's frames are
near-duplicates, and even the oracle could not beat uniform. Here one budget covers the whole
~2-minute pool, so it decides WHICH moments get described — the setting where uniform sampling
can step over a short event. Chosen frames are captioned in time-ordered pairs, so K frames give
K/2 sentences (8 -> 4, close to the references' ~3.7).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.evaluate import (_unit, load_model, make_learned_scores, make_oracle_scores,
                              motion_scores)
from scripts.evaluate_paragraphs import paired_ci, paragraph, score
from vidcap.config import CACHE, OUT
from vidcap.data import load_split, uniform_indices
from vidcap.decode import greedy
from vidcap.encoder import load_cached
from vidcap.timeline import merge_repeats, pair_frames, segment_best


def spread_topk(scores, k, min_gap):
    """Highest-scoring k indices at least min_gap apart, sorted. Plain top-k on a per-frame
    scorer spends the whole budget on one burst: its best frames are neighbours."""
    n = len(scores)
    if n <= k:
        return uniform_indices(n, k)
    picked = []
    for i in np.argsort(-np.asarray(scores), kind="stable"):
        if all(abs(int(i) - j) >= min_gap for j in picked):
            picked.append(int(i))
            if len(picked) == k:
                break
    for i in uniform_indices(n, k):          # gap too strict for this pool: top up evenly
        if len(picked) == k:
            break
        if i not in picked:
            picked.append(int(i))
    return sorted(picked)


def uniform_budget(emb, times, k, rec=None):
    return uniform_indices(len(emb), k)


def by_spread(score_fn):
    """Global top-k with a min gap of half the uniform spacing (the first budget experiment)."""
    def select(emb, times, k, rec=None):
        s = score_fn(emb, rec)
        return uniform_indices(len(emb), k) if s is None else             spread_topk(s, k, max(1, len(emb) // (2 * k)))
    return select


def by_segment(score_fn):
    def select(emb, times, k, rec=None):
        s = score_fn(emb, rec)
        return uniform_indices(len(emb), k) if s is None else segment_best(s, k)
    return select


def make_event_oracle(dataset):
    """CEILING for coverage: one frame per annotated event — the one closest to THAT event's
    sentence. The video-level oracle ranks frames against all sentences pooled, i.e. rewards
    typicality, and could not beat uniform at K>=4; this one knows where each event is."""
    z = np.load(CACHE / dataset / "_captions.npz")
    text = {(str(v), str(t)): e for v, t, e in zip(z["video_id"], z["text"], z["emb"])}

    def select(emb, times, k, rec):
        ev = [(a, b, text.get((str(rec["video_id"]), t))) for a, b, t in rec.get("events", [])]
        ev = [x for x in ev if x[2] is not None]
        if not ev or len(emb) <= k:
            return uniform_indices(len(emb), k)
        slots = ([ev[i] for i in np.linspace(0, len(ev) - 1, k).round().astype(int)]
                 if k <= len(ev) else [ev[i % len(ev)] for i in range(k)])
        u, picked = _unit(emb), []
        for a, b, t in slots:
            inside = np.flatnonzero((times >= a) & (times <= b))
            if len(inside) == 0:                  # event shorter than the frame spacing
                inside = np.array([int(np.argmin(np.abs(times - (a + b) / 2)))])
            order = inside[np.argsort(-(u[inside] @ _unit(t[None])[0]))]
            picked.append(next((int(i) for i in order if int(i) not in picked), int(order[0])))
        return sorted(picked)

    return select


def paragraphs_for(recs, pools, model, k, select, device, batch=32):
    """-> one paragraph per video: its K chosen frames captioned in consecutive pairs."""
    jobs, owner, when = [], [], []
    for vi, (rec, (emb, times)) in enumerate(zip(recs, pools)):
        for pair in pair_frames(select(emb, times, k, rec)):
            jobs.append(emb[pair])
            owner.append(vi)
            when.append((float(times[pair[0]]), float(times[pair[-1]])))
    caps = []
    for i in range(0, len(jobs), batch):
        f = torch.from_numpy(np.ascontiguousarray(np.stack(jobs[i:i + batch]))).float().to(device)
        caps += [" ".join(c.split()) for c in greedy(model, f)]
    events = [[] for _ in recs]
    for vi, cap, (s, e) in zip(owner, caps, when):
        events[vi].append({"start": s, "end": e, "caption": cap, "frames": [], "range": [0, 0]})
    return [paragraph(merge_repeats(ev)) for ev in events]


ALL_ARMS = ("uniform", "motion", "learned", "oracle",
            "seg-motion", "seg-learned", "seg-oracle", "event-oracle")
COMPARE = [("learned", "uniform"), ("motion", "uniform"), ("oracle", "uniform"),
           ("learned", "motion"), ("seg-learned", "uniform"), ("seg-motion", "uniform"),
           ("seg-oracle", "uniform"), ("event-oracle", "uniform"), ("seg-learned", "learned"),
           ("seg-learned", "seg-motion"), ("event-oracle", "seg-learned")]


def build_arms(names, scorer, device):
    learned = make_learned_scores(scorer, device) if any("learned" in n for n in names) else None
    oracle = make_oracle_scores("activitynet") if any(n in ("oracle", "seg-oracle") for n in names) else None
    make = {"uniform": lambda: uniform_budget,
            "motion": lambda: by_spread(motion_scores), "learned": lambda: by_spread(learned),
            "oracle": lambda: by_spread(oracle), "seg-motion": lambda: by_segment(motion_scores),
            "seg-learned": lambda: by_segment(learned), "seg-oracle": lambda: by_segment(oracle),
            "event-oracle": lambda: make_event_oracle("activitynet")}
    unknown = [n for n in names if n not in make]
    if unknown:
        raise SystemExit(f"unknown arms {unknown}; choose from {', '.join(ALL_ARMS)}")
    return {n: make[n]() for n in names}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="stageB")
    ap.add_argument("--scorer", default="scorer")
    ap.add_argument("--budgets", default="4,8")
    ap.add_argument("--arms", default=",".join(ALL_ARMS))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    recs = load_split("activitynet", "val")
    if not recs:
        raise SystemExit("no cached activitynet shards — attach the saved cache_activitynet")
    pools = [load_cached("activitynet", r["video_id"]) for r in recs]
    refs = [r["paragraphs"] for r in recs]
    print(f"{len(recs)} videos, median pool {np.median([len(e) for e, _ in pools]):.0f} frames")

    model, _ = load_model(args.ckpt, device)
    arms = build_arms([a.strip() for a in args.arms.split(",") if a.strip()], args.scorer, device)

    p = Path(args.out) if args.out else OUT / "eval_activitynet_budget.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    res = {"n": len(recs), "budgets": {}}

    def save():
        with open(p, "w") as f:
            json.dump(res, f, indent=2)

    for k in [int(b) for b in args.budgets.split(",")]:
        r = res["budgets"][str(k)] = {"arms": {}, "paired": {}, "examples": {}}
        for name, select in arms.items():
            hyps = paragraphs_for(recs, pools, model, k, select, device)
            r["arms"][name] = score(hyps, refs)
            r["examples"][name] = hyps[:3]
            s = r["arms"][name]
            print(f"K={k:<2d} {name:12s} CIDEr-D {s['CIDEr-D']:.4f}  BLEU-4 {s['BLEU-4']:.4f}  "
                  f"ROUGE-L {s['ROUGE-L']:.4f}  ({s['sentences']:.1f} sent, {s['words']:.0f} words)",
                  flush=True)
            save()
        a = r["arms"]
        for x, y in COMPARE:
            if x in a and y in a:
                ci = paired_ci(a[x]["per_video_cider"], a[y]["per_video_cider"])
                r["paired"][f"{x} - {y}"] = ci
                print(f"  K={k} CIDEr-D {x} - {y}: {ci['mean']:+.4f}  "
                      f"95% CI [{ci['lo']:+.4f}, {ci['hi']:+.4f}]")
        save()
    res["references"] = refs[:3]
    save()
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()

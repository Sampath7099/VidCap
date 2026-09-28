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

from scripts.evaluate import load_model, make_learned_scores, make_oracle_scores, motion_scores
from scripts.evaluate_paragraphs import paired_ci, paragraph, score
from vidcap.config import OUT
from vidcap.data import load_split, uniform_indices
from vidcap.decode import greedy
from vidcap.encoder import load_cached
from vidcap.timeline import merge_repeats


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


def budget_select(score_fn, emb, k, rec):
    """score_fn None = uniform. min_gap is half the uniform spacing, so no method can put two
    frames closer than uniform would put two half-steps."""
    s = None if score_fn is None else score_fn(emb, rec)
    if s is None:
        return uniform_indices(len(emb), k)
    return spread_topk(s, k, max(1, len(emb) // (2 * k)))


def paragraphs_for(recs, pools, model, k, score_fn, device, batch=32):
    """-> one paragraph per video: its K chosen frames captioned in consecutive pairs."""
    jobs, owner, when = [], [], []
    for vi, (rec, (emb, times)) in enumerate(zip(recs, pools)):
        idx = budget_select(score_fn, emb, k, rec)
        for a in range(0, len(idx), 2):
            pair = idx[a:a + 2]
            jobs.append(emb[pair + pair[-1:] * (2 - len(pair))])   # odd K: repeat the last frame
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="stageB")
    ap.add_argument("--scorer", default="scorer")
    ap.add_argument("--budgets", default="4,8")
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
    arms = {"uniform": None, "motion": motion_scores,
            "learned": make_learned_scores(args.scorer, device),
            "oracle": make_oracle_scores("activitynet")}

    p = Path(args.out) if args.out else OUT / "eval_activitynet_budget.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    res = {"n": len(recs), "budgets": {}}

    def save():
        with open(p, "w") as f:
            json.dump(res, f, indent=2)

    for k in [int(b) for b in args.budgets.split(",")]:
        r = res["budgets"][str(k)] = {"arms": {}, "paired": {}, "examples": {}}
        for name, fn in arms.items():
            hyps = paragraphs_for(recs, pools, model, k, fn, device)
            r["arms"][name] = score(hyps, refs)
            r["examples"][name] = hyps[:3]
            s = r["arms"][name]
            print(f"K={k:<2d} {name:8s} CIDEr-D {s['CIDEr-D']:.4f}  BLEU-4 {s['BLEU-4']:.4f}  "
                  f"ROUGE-L {s['ROUGE-L']:.4f}  ({s['sentences']:.1f} sent, {s['words']:.0f} words)",
                  flush=True)
            save()
        a = r["arms"]
        for x, y in [("learned", "uniform"), ("motion", "uniform"), ("oracle", "uniform"),
                     ("learned", "motion")]:
            ci = paired_ci(a[x]["per_video_cider"], a[y]["per_video_cider"])
            r["paired"][f"{x} - {y}"] = ci
            print(f"  K={k} CIDEr-D {x} - {y}: {ci['mean']:+.4f}  95% CI [{ci['lo']:+.4f}, {ci['hi']:+.4f}]")
        save()
    res["references"] = refs[:3]
    save()
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()

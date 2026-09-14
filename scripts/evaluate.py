"""Quality-vs-frame-budget curves: uniform vs motion heuristic vs learned scorer.

  python -m scripts.evaluate --ckpt stageC --budgets 2,4,8,16
  python -m scripts.evaluate --ckpt stageC --ckpt-blind blind    # sighted vs blind control

The headline result. Writes JSON so the README figure is regenerable from one file.
"""
import argparse
import json

import numpy as np
import torch

from vidcap import checkpoint
from vidcap.config import OUT
from vidcap.data import eval_batches, load_split, uniform_indices
from vidcap.decode import beam_search, greedy
from vidcap.metrics import evaluate
from vidcap.model import VideoCaptioner


def motion_indices(emb, k):
    """Classical baseline: pick frames with the largest change from the previous frame.
    Operates on embedding deltas (cached) rather than raw pixels — same idea, no video decode."""
    if len(emb) <= k:
        return uniform_indices(len(emb), k)
    e = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8)
    d = np.concatenate([[0.0], 1.0 - (e[1:] * e[:-1]).sum(1)])  # cosine distance to previous
    return sorted(np.argsort(-d)[:k].tolist())


def uniform_sel(emb, k):
    """Adapter: uniform_indices takes a pool size, the selector protocol takes the pool."""
    return uniform_indices(len(emb), k)


# Every entry MUST take (emb, k) — eval_batches calls them as sel(emb, k). Registering
# uniform_indices directly here passed an array where it expected an int.
SELECTORS = {"uniform": uniform_sel, "motion": motion_indices}


def load_model(name, device):
    ck = checkpoint.load(name, map_location=device)
    if ck is None:
        raise SystemExit(f"no checkpoint '{name}' — train it first")
    a = ck.get("args", {})
    m = VideoCaptioner(connector=a.get("connector", "resampler"),
                       n_prefix=a.get("n_prefix", 8), lora_r=a.get("lora_r", 8),
                       blind=a.get("blind", False)).to(device).eval()
    m.load_state_dict(ck["model"], strict=False)
    return m, a


def run(model, dataset, recs, select, k, device, beam=1, limit_batches=None):
    hyps, refs = [], []
    for bi, (frames, rs) in enumerate(eval_batches(dataset, recs, k=k, select=select)):
        if limit_batches and bi >= limit_batches:
            break
        frames = frames.to(device)
        out = greedy(model, frames) if beam <= 1 else beam_search(model, frames, beam=beam)
        hyps += out
        refs += rs
    return evaluate(hyps, refs), hyps, refs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ckpt-blind", default=None, help="blind control checkpoint to compare against")
    ap.add_argument("--dataset", default="msrvtt")
    ap.add_argument("--split", default="test")
    ap.add_argument("--root", default=None)
    ap.add_argument("--budgets", default="2,4,8,16")
    ap.add_argument("--beam", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    recs = load_split(args.dataset, args.split, args.root, args.limit)
    if not recs:
        raise SystemExit("no cached shards for this split")
    print(f"{len(recs)} {args.dataset}/{args.split} clips")

    model, _ = load_model(args.ckpt, device)
    budgets = [int(b) for b in args.budgets.split(",")]
    results = {"dataset": args.dataset, "split": args.split, "n": len(recs),
               "beam": args.beam, "curves": {}}

    for sel_name, sel in SELECTORS.items():
        results["curves"][sel_name] = {}
        for k in budgets:
            scores, hyps, _ = run(model, args.dataset, recs, sel, k, device, args.beam)
            results["curves"][sel_name][k] = scores
            print(f"{sel_name:8s} K={k:<3d} " +
                  "  ".join(f"{m} {v:.4f}" for m, v in scores.items()) +
                  f"   e.g. {hyps[0]!r}")

    if args.ckpt_blind:
        blind, _ = load_model(args.ckpt_blind, device)
        k = max(budgets)
        b_scores, _, _ = run(blind, args.dataset, recs, uniform_sel, k, device, args.beam)
        results["blind"] = b_scores
        s = results["curves"]["uniform"][k]
        print(f"\nblind control @K={k}: " + "  ".join(f"{m} {v:.4f}" for m, v in b_scores.items()))
        gap = {m: s[m] - b_scores[m] for m in s}
        print("sighted - blind: " + "  ".join(f"{m} {v:+.4f}" for m, v in gap.items()))
        if gap["CIDEr-D"] <= 0:
            print("\n*** THE MODEL IS IGNORING THE VIDEO. Frame selection cannot matter until\n"
                  "    this is fixed — do not interpret the curves above. ***")

    p = args.out or (OUT / f"eval_{args.dataset}_{args.split}.json")
    p.parent.mkdir(parents=True, exist_ok=True) if hasattr(p, "parent") else None
    with open(p, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()

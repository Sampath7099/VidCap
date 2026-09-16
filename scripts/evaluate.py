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
from vidcap.config import CACHE, OUT
from vidcap.data import eval_batches, load_split, uniform_indices
from vidcap.decode import beam_search, greedy
from vidcap.metrics import evaluate
from vidcap.model import VideoCaptioner


def _unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


def motion_indices(emb, k, rec=None):
    """Classical baseline: pick frames with the largest change from the previous frame.
    Operates on embedding deltas (cached) rather than raw pixels — same idea, no video decode."""
    if len(emb) <= k:
        return uniform_indices(len(emb), k)
    e = _unit(emb)
    d = np.concatenate([[0.0], 1.0 - (e[1:] * e[:-1]).sum(1)])  # cosine distance to previous
    return sorted(np.argsort(-d)[:k].tolist())


def make_oracle(dataset):
    """CEILING CONTROL — picks the K frames most similar to the clip's ground-truth captions.

    It reads the test caption, so it is not a usable method; it is the upper bound a learned
    scorer is trying to approximate without that caption. Oracle ~= uniform means the data has
    no selection headroom at this budget and no scorer can win, which is worth knowing before
    building one. Oracle >> uniform means the headroom is real and the open question is only
    whether it can be found from pixels alone.
    """
    z = np.load(CACHE / dataset / "_captions.npz")
    by_vid = {}
    for v, e in zip(z["video_id"], z["emb"]):
        by_vid.setdefault(str(v), []).append(e)
    by_vid = {v: _unit(np.stack(e)) for v, e in by_vid.items()}

    def oracle(emb, k, rec=None):
        caps = by_vid.get(str(rec["video_id"])) if rec else None
        if caps is None or len(emb) <= k:
            return uniform_indices(len(emb), k)
        score = (_unit(emb) @ caps.T).mean(1)      # mean similarity over this clip's captions
        return sorted(np.argsort(-score)[:k].tolist())

    return oracle


def make_learned(name, device):
    """THE CONTRIBUTION — top-K by predicted relevance, from pixels alone.

    Same shape as the oracle but without the caption, so the gap between them is exactly how
    much of the ceiling the scorer actually recovers.
    """
    from vidcap.scorer import FrameScorer
    model = FrameScorer().to(device).eval()
    if checkpoint.load(name, model, map_location=device) is None:
        raise SystemExit(f"no scorer checkpoint '{name}' — run scripts/train_scorer.py first")

    @torch.no_grad()
    def learned(emb, k, rec=None):
        if len(emb) <= k:
            return uniform_indices(len(emb), k)
        s = model(torch.from_numpy(emb).float()[None].to(device))[0].cpu().numpy()
        return sorted(np.argsort(-s)[:k].tolist())

    return learned


def uniform_sel(emb, k, rec=None):
    """Adapter: uniform_indices takes a pool size, the selector protocol takes the pool."""
    return uniform_indices(len(emb), k)


# Every entry MUST take (emb, k, rec) — eval_batches calls them that way. rec carries the
# video_id the oracle needs; the other arms ignore it.
SELECTORS = {"uniform": uniform_sel, "motion": motion_indices}   # + "oracle", built per-dataset


def load_model(name, device):
    ck = checkpoint.load(name, map_location=device)
    if ck is None:
        raise SystemExit(f"no checkpoint '{name}' — train it first")
    # "arch" is the architecture actually built (train.arch_of). Fall back to raw args only for
    # checkpoints written before that existed — and default lora_r to 0, since stage B has no
    # adapters regardless of what --lora-r said.
    a = ck.get("arch") or ck.get("args", {})
    m = VideoCaptioner(connector=a.get("connector", "meanpool"),
                       n_prefix=a.get("n_prefix", 8), lora_r=a.get("lora_r", 0),
                       blind=a.get("blind", False)).to(device).eval()
    # One shared rule (checkpoint.restore): frozen-decoder keys are legitimately absent, anything
    # else missing or unexpected means the checkpoint does not describe this model. strict=False
    # alone would silently report metrics computed from randomly-initialised weights.
    try:
        checkpoint.restore(m, ck, where=f"checkpoint '{name}'")
    except RuntimeError as e:
        raise SystemExit(str(e)) from None
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
    ap.add_argument("--scorer", default=None,
                    help="scorer checkpoint name; adds the learned selection arm")
    ap.add_argument("--oracle", action="store_true",
                    help="add the cheating ceiling arm (uses ground-truth captions)")
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

    selectors = dict(SELECTORS)
    if args.scorer:
        selectors["learned"] = make_learned(args.scorer, device)
    if args.oracle:
        selectors["oracle"] = make_oracle(args.dataset)
    for sel_name, sel in selectors.items():
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

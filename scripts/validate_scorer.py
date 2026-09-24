"""Validate the frame scorer against HUMAN importance labels (TVSum / SumMe).

  python -m scripts.build_cache tvsum && python -m scripts.build_cache summe
  python -m scripts.validate_scorer --datasets tvsum,summe

This answers the sharpest criticism of the headline result. The scorer is trained on
SigLIP-to-caption similarity and the captioner consumes SigLIP embeddings, so the two are aligned
by construction — a sceptic will say the selector only agrees with itself. TVSum and SumMe carry
real per-frame importance annotated by humans who never saw SigLIP, so correlation here is the
one measurement that cannot be circular.

Reported against two nulls, because a bare correlation is unreadable: `motion`, the classical
frame-difference heuristic, and `random`. Beating random says the scorer found something;
beating motion says it found something a cheap heuristic does not already give you.
"""
import argparse
import json

import cv2
import numpy as np
import torch

from vidcap import checkpoint
from vidcap.config import OUT
from vidcap.datasets import summe, tvsum
from vidcap.encoder import cache_path
from vidcap.scorer import FrameScorer, spearman

LOADERS = {"tvsum": tvsum, "summe": summe}


def frame_indices(path, times, n_scores):
    """Pool timestamps (seconds) -> indices into the per-frame human score array.

    TVSum and SumMe are BOTH per-frame (verified: len(scores) == frame count exactly), so this
    is a plain time*fps mapping for both — see DATASETS.md. Clipped because the decoder's
    reported frame count and the annotation length can differ by a frame or two.
    """
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if not fps or fps != fps or fps <= 0:
        fps = 30.0
    return np.clip((np.asarray(times) * fps).round().astype(int), 0, n_scores - 1)


def motion_scores(emb):
    """Classical null: cosine distance to the previous frame. Same signal as evaluate.py's
    motion arm, as a per-frame score rather than a top-K selection."""
    e = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8)
    return np.concatenate([[0.0], 1.0 - (e[1:] * e[:-1]).sum(1)])


def rho(pred, human):
    """Within-video Spearman, via the same implementation the scorer's own metric uses."""
    p = torch.from_numpy(np.asarray(pred, np.float32))[None]
    h = torch.from_numpy(np.asarray(human, np.float32))[None]
    return spearman(p, h, torch.ones_like(p, dtype=torch.bool))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="tvsum,summe")
    ap.add_argument("--scorer", default="scorer")
    ap.add_argument("--out", default=str(OUT / "validate_scorer.json"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model = FrameScorer().eval()
    if checkpoint.load(args.scorer, model) is None:
        raise SystemExit(f"no scorer checkpoint '{args.scorer}' — run scripts/train_scorer.py")
    rng = np.random.default_rng(args.seed)

    report = {}
    for name in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        records, scores = LOADERS[name]()
        per_video, missing = [], 0
        for r in records:
            p = cache_path(name, r["video_id"])
            if not p.exists():
                missing += 1
                continue
            z = np.load(p)
            emb, times = z["emb"], z["times"]
            if len(emb) < 3:
                continue
            human = scores[r["video_id"]][frame_indices(r["path"], times, len(scores[r["video_id"]]))]
            if human.std() < 1e-6:            # a flat annotation has no ranking to predict
                continue
            with torch.no_grad():
                pred = model(torch.from_numpy(emb).float()).numpy()
            per_video.append({
                "video_id": r["video_id"],
                "n_frames": int(len(emb)),
                "learned": rho(pred, human),
                "motion": rho(motion_scores(emb), human),
                "random": rho(rng.standard_normal(len(emb)), human),
            })

        if not per_video:
            print(f"{name}: no cached videos — run: python -m scripts.build_cache {name}")
            continue
        agg = {arm: {"mean": float(np.mean([v[arm] for v in per_video])),
                     "std": float(np.std([v[arm] for v in per_video]))}
               for arm in ("learned", "motion", "random")}
        report[name] = {"n_videos": len(per_video), "n_missing_cache": missing,
                        "aggregate": agg, "per_video": per_video}
        print(f"\n{name}: {len(per_video)} videos ({missing} not cached)")
        for arm, a in agg.items():
            print(f"  {arm:8s} mean Spearman vs human  {a['mean']:+.4f}  (sd {a['std']:.4f})")

    if report:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

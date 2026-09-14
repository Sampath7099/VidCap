"""Stage 0: video -> candidate frame pool -> cached CLIP embeddings. Resumable; safe to re-run.

  python -m scripts.build_cache msrvtt
  python -m scripts.build_cache msrvtt --limit 50 --root data/msrvtt
"""
import argparse
import sys

import numpy as np
from tqdm import tqdm

from vidcap.encoder import cache_path, cache_video, embed_texts, load_vision
from vidcap.config import CACHE, DATASETS
from vidcap.datasets import LOADERS, verify_no_leakage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=DATASETS)
    ap.add_argument("--root", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    records = LOADERS[args.dataset](args.root)
    print(f"{args.dataset}: {len(records)} videos found | splits {verify_no_leakage(records)}")
    if args.limit:
        records = records[:args.limit]
    if not records:
        sys.exit("no videos found — check --root / DATA path")

    model, proc, device = load_vision()
    print(f"vision encoder on {device}")

    # mininterval: Kaggle's committed runs capture stdout to a file where \r doesn't
    # collapse, so a default-rate tqdm writes thousands of lines into the log.
    done, empty = 0, []
    for r in tqdm(records, desc="caching", unit="vid", mininterval=30):
        if cache_path(args.dataset, r["video_id"]).exists() and not args.overwrite:
            done += 1
            continue
        _, n = cache_video(args.dataset, r["video_id"], r["path"], model, proc, device, args.overwrite)
        if n == 0:
            empty.append(r["video_id"])
        done += 1

    # Caption text embeddings: the Phase 1 relevance labels (text is a label source, never a model input).
    caps = [(r["video_id"], c) for r in records for c in r["captions"]]
    if caps:
        emb = embed_texts(model, proc, device, [c for _, c in caps])
        out = CACHE / args.dataset / "_captions.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, emb=emb.astype(np.float32),
                            video_id=np.array([v for v, _ in caps]),
                            text=np.array([c for _, c in caps]))
        print(f"cached {len(caps)} caption embeddings -> {out}")

    print(f"done: {done} cached, {len(empty)} undecodable {empty[:10]}")


if __name__ == "__main__":
    main()

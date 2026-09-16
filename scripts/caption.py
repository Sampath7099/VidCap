"""Video in, caption out — the end-to-end usability gate (context.md requirement #1).

  python -m scripts.caption clip.mp4
  python -m scripts.caption clip.mp4 --k 1 --select learned,uniform     # side-by-side
  python -m scripts.caption holdout/*.mp4 --k 2 --beam 4

This is the only path that touches raw video at inference; everything else in the repo runs on
cached embeddings. Frames are sampled at the same 3 fps the cache was built with, so the model
sees the pool density it was trained on.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from scripts.evaluate import load_model, make_learned, motion_indices, uniform_sel
from vidcap.config import POOL_FPS
from vidcap.decode import beam_search, greedy
from vidcap.encoder import embed_images, load_vision
from vidcap.video import sample_frames


def caption_from_pool(emb, k, selector, model, device="cpu", beam=1):
    """(N, D) candidate-frame embeddings -> (caption, chosen indices).

    Separated from video decoding so it is testable without loading SigLIP.
    """
    if len(emb) == 0:
        return "", []
    idx = selector(emb, k, None)
    f = torch.from_numpy(np.ascontiguousarray(emb[idx])).float()[None].to(device)
    with torch.no_grad():
        out = greedy(model, f) if beam <= 1 else beam_search(model, f, beam=beam)
    return out[0], idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+", help="paths to video files")
    ap.add_argument("--ckpt", default="stageB")
    ap.add_argument("--scorer", default="scorer")
    ap.add_argument("--k", type=int, default=4, help="frame budget")
    ap.add_argument("--select", default="learned",
                    help="comma-separated: learned, uniform, motion — several compares them")
    ap.add_argument("--fps", type=float, default=POOL_FPS["msrvtt"],
                    help="candidate-pool density; defaults to what the cache was built with")
    ap.add_argument("--beam", type=int, default=1)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_model(args.ckpt, device)

    wanted = [s.strip() for s in args.select.split(",") if s.strip()]
    selectors = {}
    for name in wanted:
        if name == "uniform":
            selectors[name] = uniform_sel
        elif name == "motion":
            selectors[name] = motion_indices
        elif name == "learned":
            selectors[name] = make_learned(args.scorer, device)
        else:
            raise SystemExit(f"unknown selector {name!r} (learned, uniform, motion)")

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

        print(f"\n{p.name}  ({len(frames)} candidate frames, {times[-1]:.1f}s)")
        for name, sel in selectors.items():
            cap, idx = caption_from_pool(emb, args.k, sel, model, device, args.beam)
            picked = ", ".join(f"{times[i]:.1f}s" for i in idx)
            print(f"  {name:8s} K={args.k}  [{picked}]")
            print(f"  {'':8s}      {cap!r}")


if __name__ == "__main__":
    main()

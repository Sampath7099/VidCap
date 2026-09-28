"""Side-by-side examples: the evenly spaced frame vs the picker's frame, and what each produces.

  python -m scripts.make_examples --n 4

Captions every MSR-VTT test clip from ONE frame both ways, scores each caption against that
clip's human captions, and draws the clips where the picker's caption beats uniform's by the
widest margin. These are chosen to show the difference, not to be typical — the averages over
all 2,990 clips are in RESULTS.md.
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from scripts.evaluate import load_model, make_learned, uniform_sel
from vidcap.config import OUT, POOL_FPS
from vidcap.data import load_split
from vidcap.decode import greedy
from vidcap.encoder import load_cached
from vidcap.metrics import cider_d
from vidcap.video import sample_frames


def captions(model, pools, recs, select, device, batch=32):
    """One caption per clip from its single selected frame, in record order."""
    idx, caps = [], []
    for i in range(0, len(recs), batch):
        sel = [select(pools[j], 1, recs[j])[0] for j in range(i, min(i + batch, len(recs)))]
        f = torch.stack([torch.from_numpy(pools[j][s:s + 1]).float()
                         for j, s in zip(range(i, i + len(sel)), sel)]).to(device)
        idx += sel
        caps += [" ".join(c.split()) for c in greedy(model, f)]
    return idx, caps


def draw(examples, path):
    fig, axes = plt.subplots(len(examples), 2, figsize=(10, 3.6 * len(examples)))
    axes = np.atleast_2d(axes)
    for row, ex in zip(axes, examples):
        for ax, side, title in zip(row, ("uniform", "learned"), ("Evenly spaced frame", "VidCap's frame")):
            ax.imshow(ex[side]["image"])
            ax.set_xticks([]), ax.set_yticks([])
            ax.set_title(f"{title}  ({ex[side]['time']:.1f}s)", fontsize=10,
                         color="#555555" if side == "uniform" else "#1f5fa8", fontweight="bold")
            ax.set_xlabel(f"“{ex[side]['caption']}”", fontsize=10, wrap=True)
        row[0].annotate(f"A person described it as: “{ex['reference']}”", (0, -0.30),
                        xycoords="axes fraction", fontsize=9, style="italic", color="#333333")
    fig.tight_layout(h_pad=3.0)
    fig.savefig(path, dpi=130, bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--ckpt", default="stageB")
    ap.add_argument("--scorer", default="scorer")
    ap.add_argument("--fig", default="figures/examples.png")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    recs = load_split("msrvtt", "test")
    pools = [load_cached("msrvtt", r["video_id"])[0] for r in recs]
    keep = [i for i, p in enumerate(pools) if len(p)]
    recs, pools = [recs[i] for i in keep], [pools[i] for i in keep]
    refs = [r["captions"] for r in recs]
    model, _ = load_model(args.ckpt, device)
    print(f"{len(recs)} test clips")

    u_idx, u_cap = captions(model, pools, recs, uniform_sel, device)
    l_idx, l_cap = captions(model, pools, recs, make_learned(args.scorer, device), device)
    u_s = np.array(cider_d(u_cap, refs, per_item=True))
    l_s = np.array(cider_d(l_cap, refs, per_item=True))
    print(f"mean CIDEr-D  uniform {u_s.mean():.4f}  learned {l_s.mean():.4f}")

    examples = []
    for i in np.argsort(-(l_s - u_s)):
        if len(examples) == args.n:
            break
        if u_idx[i] == l_idx[i] or u_cap[i] == l_cap[i]:
            continue
        frames, times = sample_frames(recs[i]["path"], fps=POOL_FPS["msrvtt"])
        if len(frames) != len(pools[i]):          # decode must match the cached pool exactly
            continue
        examples.append({"video_id": recs[i]["video_id"], "reference": min(refs[i], key=len),
                         "uniform": {"image": frames[u_idx[i]], "time": float(times[u_idx[i]]),
                                     "caption": u_cap[i], "cider": float(u_s[i])},
                         "learned": {"image": frames[l_idx[i]], "time": float(times[l_idx[i]]),
                                     "caption": l_cap[i], "cider": float(l_s[i])}})
        print(f"  {recs[i]['video_id']}: '{u_cap[i]}' -> '{l_cap[i]}'  (+{l_s[i] - u_s[i]:.2f})")

    draw(examples, args.fig)
    out = args.out or OUT / "examples.json"
    with open(out, "w") as f:
        json.dump({"n_clips": len(recs), "mean_uniform": float(u_s.mean()),
                   "mean_learned": float(l_s.mean()),
                   "learned_better_share": float((l_s > u_s).mean()),
                   "examples": [{k: ({kk: vv for kk, vv in v.items() if kk != "image"}
                                     if isinstance(v, dict) else v) for k, v in ex.items()}
                                for ex in examples]}, f, indent=2)
    print(f"wrote {args.fig} and {out}")


if __name__ == "__main__":
    main()

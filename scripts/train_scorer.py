"""Stage A: train the frame-relevance scorer on cached embeddings. Minutes, not hours —
no vision or LLM forward passes, just a small MLP over vectors already on disk.

  python -m scripts.train_scorer --epochs 20
"""
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from vidcap import checkpoint
from vidcap.config import CACHE
from vidcap.data import load_split
from vidcap.encoder import cache_path
from vidcap.scorer import FrameScorer, clip_targets, listwise_loss, spearman


def caption_bank(dataset):
    """{video_id: (n_caps, D)} from the cached caption embeddings."""
    z = np.load(CACHE / dataset / "_captions.npz")
    by = {}
    for v, e in zip(z["video_id"], z["emb"]):
        by.setdefault(str(v), []).append(e)
    return {v: np.stack(e) for v, e in by.items()}


class ScorerDataset(Dataset):
    """One clip per item: (frames[N,D], target[N]). N varies, so collate pads with a mask."""

    def __init__(self, dataset, records, bank):
        self.dataset = dataset
        self.bank = bank
        # Only clips whose captions are cached — the target is undefined otherwise.
        self.records = [r for r in records if str(r["video_id"]) in bank]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        vid = str(self.records[i]["video_id"])
        emb = np.load(cache_path(self.dataset, vid))["emb"]
        return torch.from_numpy(emb).float(), torch.from_numpy(clip_targets(emb, self.bank[vid])).float()


def collate(batch):
    n = max(f.shape[0] for f, _ in batch)
    d = batch[0][0].shape[1]
    F = torch.zeros(len(batch), n, d)
    T = torch.zeros(len(batch), n)
    M = torch.zeros(len(batch), n, dtype=torch.bool)
    for i, (f, t) in enumerate(batch):
        F[i, :f.shape[0]], T[i, :t.shape[0]], M[i, :f.shape[0]] = f, t, True
    return F, T, M


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss, rho, n = 0.0, 0.0, 0
    for F, T, M in loader:
        F, T, M = F.to(device), T.to(device), M.to(device)
        p = model(F)
        loss += listwise_loss(p, T, M).item() * len(F)
        rho += spearman(p, T, M) * len(F)
        n += len(F)
    model.train()
    return loss / max(n, 1), rho / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="msrvtt")
    ap.add_argument("--root", default=None)
    ap.add_argument("--name", default="scorer")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bank = caption_bank(args.dataset)

    # Train split only. The caption bank covers every clip, so using val/test records here
    # would leak their captions into the scorer through the targets.
    tr = ScorerDataset(args.dataset, load_split(args.dataset, "train", args.root, args.limit), bank)
    va = ScorerDataset(args.dataset, load_split(args.dataset, "val", args.root, args.limit), bank)
    if not tr:
        raise SystemExit("no cached training shards — run scripts/build_cache.py first")
    print(f"{len(tr)} train / {len(va)} val clips")

    dl = DataLoader(tr, batch_size=args.bs, shuffle=True, collate_fn=collate, num_workers=2)
    vdl = DataLoader(va, batch_size=args.bs, collate_fn=collate) if len(va) else None

    model = FrameScorer().to(device)
    print(f"trainable {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    ck = checkpoint.load(args.name, model, opt, map_location=device)
    step = ck["step"] if ck else 0

    bar = tqdm(total=args.epochs * len(dl), initial=step, desc=args.name, unit="step", mininterval=30)
    for ep in range(step // max(len(dl), 1), args.epochs):
        for F, T, M in dl:
            F, T, M = F.to(device), T.to(device), M.to(device)
            loss = listwise_loss(model(F), T, M)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            bar.update(1)
        vl, rho = evaluate(model, vdl, device) if vdl else (float("nan"), float("nan"))
        print(f"== epoch {ep} | val loss {vl:.4f} | val spearman {rho:+.4f}", flush=True)
        checkpoint.save(args.name, step, model, opt, args=vars(args), val_loss=vl, spearman=rho)
    bar.close()
    print(f"saved {args.name} @ step {step}")


if __name__ == "__main__":
    main()

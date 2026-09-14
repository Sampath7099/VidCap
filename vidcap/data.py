"""Dataset over cached embedding shards. Frame selection is pluggable: uniform now, scorer later."""
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from .datasets import LOADERS
from .encoder import cache_path


def available(dataset, records):
    """Only records whose cache shard actually exists — caching may be partial or still running."""
    return [r for r in records if cache_path(dataset, r["video_id"]).exists()]


def load_split(dataset, split=None, root=None, limit=None):
    recs = LOADERS[dataset](root)
    if split:
        recs = [r for r in recs if r["split"] == split]
    recs = available(dataset, recs)
    return recs[:limit] if limit else recs


def uniform_indices(n, k):
    """Evenly spaced over the pool, both ends included. The default this project aims to beat."""
    if n <= k:
        return list(range(n)) + [n - 1] * (k - n)  # pad by repeating the last frame
    return np.linspace(0, n - 1, k).round().astype(int).tolist()


class VideoCaptionDataset(Dataset):
    """Yields (frames[K,D] float32, caption str). One random caption per epoch per clip."""

    def __init__(self, dataset, records, k=8, select=None, train=True, seed=0):
        self.dataset, self.records, self.k, self.train = dataset, records, k, train
        self.select = select or (lambda emb, k: uniform_indices(len(emb), k))
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.records)

    def pool(self, i):
        z = np.load(cache_path(self.dataset, self.records[i]["video_id"]))
        return z["emb"]

    def __getitem__(self, i):
        r = self.records[i]
        emb = self.pool(i)
        if len(emb) == 0:
            emb = np.zeros((1, emb.shape[1] if emb.ndim == 2 else 1), np.float32)
        idx = self.select(emb, self.k)
        frames = torch.from_numpy(np.ascontiguousarray(emb[idx])).float()
        caps = r["captions"] or [""]
        cap = self.rng.choice(caps) if self.train else caps[0]
        return frames, cap


def make_collate(tokenizer, max_len=32):
    """Right-pads captions; eos appended so the model learns to stop."""
    def collate(batch):
        frames = torch.stack([b[0] for b in batch])
        texts = [b[1].strip() + tokenizer.eos_token for b in batch]
        enc = tokenizer(texts, return_tensors="pt", padding=True,
                        truncation=True, max_length=max_len)
        return frames, enc["input_ids"], enc["attention_mask"]
    return collate


def eval_batches(dataset, records, k=8, select=None, batch=16):
    """Yields (frames[B,K,D], [[refs]]) — every reference caption, for metric computation."""
    sel = select or (lambda emb, k: uniform_indices(len(emb), k))
    for i in range(0, len(records), batch):
        chunk = records[i:i + batch]
        frames, refs = [], []
        for r in chunk:
            emb = np.load(cache_path(dataset, r["video_id"]))["emb"]
            if len(emb) == 0:
                continue
            frames.append(torch.from_numpy(np.ascontiguousarray(emb[sel(emb, k)])).float())
            refs.append(r["captions"])
        if frames:
            yield torch.stack(frames), refs

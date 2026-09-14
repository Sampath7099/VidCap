"""Frozen vision-language encoder + the embedding cache every later stage trains on.

SigLIP by default (stronger image-text alignment than CLIP, which the Phase-1 scorer
labels depend on). Swapping VISION_MODEL invalidates the whole cache — see config.py.
"""
import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

from .config import CACHE, DEFAULT_POOL_FPS, POOL_FPS, VISION_MODEL
from .video import sample_frames

_SIGLIP_TEXT_LEN = 64  # SigLIP is trained at a fixed text length; it needs pad-to-max, not longest


def load_vision(device=None, name=VISION_MODEL):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModel.from_pretrained(name, dtype=dtype).to(device).eval().requires_grad_(False)
    return model, AutoProcessor.from_pretrained(name), device


def _dim(model):
    return getattr(model.config, "projection_dim", None) or model.config.text_config.hidden_size


def _feat(x):
    """transformers >=4.50 returns BaseModelOutputWithPooling from get_*_features; older returns a tensor."""
    if torch.is_tensor(x):
        return x
    return x.pooler_output if getattr(x, "pooler_output", None) is not None else x.last_hidden_state[:, 0]


@torch.no_grad()
def embed_images(model, proc, device, images, batch=32):
    """images: list/array of HWC RGB uint8 -> (N, D) float32, unnormalized."""
    out = []
    for i in range(0, len(images), batch):
        px = proc(images=list(images[i:i + batch]), return_tensors="pt")["pixel_values"]
        px = px.to(device, dtype=model.dtype)
        out.append(_feat(model.get_image_features(pixel_values=px)).float().cpu())
    return torch.cat(out).numpy() if out else np.zeros((0, _dim(model)), np.float32)


@torch.no_grad()
def embed_texts(model, proc, device, texts, batch=256):
    is_siglip = "siglip" in str(type(model)).lower()
    pad = {"padding": "max_length", "max_length": _SIGLIP_TEXT_LEN} if is_siglip else {"padding": True}
    out = []
    for i in range(0, len(texts), batch):
        tok = proc(text=list(texts[i:i + batch]), return_tensors="pt",
                   truncation=True, **pad).to(device)
        out.append(_feat(model.get_text_features(**{k: v for k, v in tok.items()
                                                   if k in ("input_ids", "attention_mask")})).float().cpu())
    return torch.cat(out).numpy() if out else np.zeros((0, _dim(model)), np.float32)


def cache_path(dataset, video_id):
    return CACHE / dataset / f"{video_id}.npz"


def cache_video(dataset, video_id, video_path, model, proc, device, overwrite=False):
    """Embed one video's candidate-frame pool into the cache. Returns (path, n_frames)."""
    p = cache_path(dataset, video_id)
    if p.exists() and not overwrite:
        return p, int(np.load(p)["emb"].shape[0])
    frames, times = sample_frames(video_path, fps=POOL_FPS.get(dataset, DEFAULT_POOL_FPS))
    emb = embed_images(model, proc, device, frames)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".npz.tmp")
    with open(tmp, "wb") as f:  # file handle, else savez appends another ".npz"
        np.savez_compressed(f, emb=emb.astype(np.float32), times=times)
    tmp.replace(p)  # atomic: a killed Kaggle session never leaves a half-written shard
    return p, len(emb)


def load_cached(dataset, video_id):
    z = np.load(cache_path(dataset, video_id))
    return z["emb"], z["times"]


def normalize(x, axis=-1, eps=1e-8):
    """Cached embeddings are stored raw; L2-normalize here for cosine similarity."""
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), eps)

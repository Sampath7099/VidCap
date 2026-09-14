"""Paths and shared constants. Override roots with VIDCAP_DATA / VIDCAP_OUT."""
import os
from pathlib import Path

ON_KAGGLE = Path("/kaggle/working").exists()

DATA = Path(os.environ.get("VIDCAP_DATA", "/kaggle/input" if ON_KAGGLE else "data"))
OUT = Path(os.environ.get("VIDCAP_OUT", "/kaggle/working" if ON_KAGGLE else "out"))
CACHE = OUT / "cache"          # <dataset>/<video_id>.npz  — CLIP frame embeddings
CKPT = OUT / "checkpoints"

# Vision encoder is BAKED INTO THE CACHE — changing it means re-embedding every video.
VISION_MODEL = "google/siglip-so400m-patch14-384"
VISION_DIM = 1152
# Decoder is swappable at any time; nothing downstream is cached from it.
LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
# Attention projections LoRA wraps. Covers Qwen (q/k/v/o_proj) and GPT-2 (c_attn).
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "c_attn")

# Candidate-pool density. The pool must stay above the largest frame budget (16) or every
# selector is forced to pick the same frames and the Phase 8 comparison is vacuous.
# NOTE: density alone does NOT create selection headroom. Sampling 15s of single-shot MSR-VTT
# at 3fps yields near-duplicate frames, so any K of them carry the same gist and the expected
# selection effect there is ~0. Headroom comes from temporal diversity — long videos with
# sparse events — which is why the decisive evaluation needs ActivityNet-style data, not a
# higher fps. Treat MSR-VTT as the captioner's training set, not as the selection benchmark.
POOL_FPS = {"msrvtt": 3.0, "holdout": 2.0, "tvsum": 1.0, "summe": 1.0, "activitynet": 1.0}
DEFAULT_POOL_FPS = 1.0
MIN_POOL_FRAMES = 32           # pool floor; short clips get denser sampling to reach it
MAX_POOL_FRAMES = 256          # cap for long ActivityNet/TVSum videos; uniformly subsampled
FRAME_SIZE = 384               # shortest side kept at decode time; must be >= encoder input
                               # (SigLIP so400m is 384 — pre-resizing to 224 would upscale and blur)

DATASETS = ("msrvtt", "tvsum", "summe", "activitynet", "holdout")

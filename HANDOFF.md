# HANDOFF — read this first

Written 2026-09-25 at the end of a Linux session, for a fresh session on Windows. Everything an
assistant or a returning human needs to continue without re-deriving anything.

Three documents, different jobs — do not confuse them:

| File | What it is |
|---|---|
| **HANDOFF.md** (this) | Current state, next actions, traps. Start here. |
| [README.md](README.md) | The public write-up. Results, figures, limitations. |
| [context.md](context.md) | The lab notebook — chronological, every finding and correction. Long. Authoritative when it conflicts with a summary. |
| [PLAN.md](PLAN.md) | The original plan. Historical; several parts are superseded. |
| [DATASETS.md](DATASETS.md) | How to obtain each dataset, with the commands that actually worked. |

---

## 1. What this project is, in one paragraph

Video captioning under a frame budget. Every captioning pipeline samples frames uniformly (every
Nth frame) regardless of content. This project trains a small **frame-relevance scorer** that picks
informative frames instead, and measures rigorously whether that helps. It does: one learned frame
beats two evenly-spaced ones on caption quality. The system is a frozen SigLIP vision encoder and a
frozen Qwen2.5-1.5B decoder joined by ~62M from-scratch parameters, plus a 0.72M-parameter scorer.

---

## 2. Status: what works right now

**Working and verified:**
- End-to-end demo: video file in, caption out (`scripts/caption.py`). Runs on CPU, ~1-2 min/clip.
- Full training and evaluation pipeline on cached embeddings.
- 8 test files + a metric-equivalence check, all passing.
- BLEU-4 / ROUGE-L / CIDEr-D match `pycocoevalcap` to ~1e-11 — numbers are comparable to published
  ones.

**Written but never trained** — do not describe as a result:
- Video Q&A and composed summarisation (`--task qa`, `scripts/evaluate_qa.py`,
  `scripts/summarize.py`). Gate-tested, no checkpoint, no accuracy number.

**Not done:** Stage C LoRA; diversity-aware selection; connector and LoRA-rank ablations. None are
load-bearing.

---

## 3. The results (use these numbers)

### Headline — MSR-VTT test, 500 clips, greedy. CIDEr-D

| K | uniform | motion | **learned** | oracle | learned − uniform | % of ceiling |
|---|---|---|---|---|---|---|
| 1 | 0.4495 | 0.4572 | **0.4977** | 0.5220 | **+0.0482** | **66%** |
| 2 | 0.4893 | 0.4804 | **0.5107** | 0.5273 | +0.0214 | 56% |
| 3 | 0.5246 | 0.5136 | **0.5270** | 0.5359 | +0.0024 | 21% |
| 4 | 0.5397 | 0.5125 | 0.5302 | 0.5443 | −0.0095 | — |

**learned K=1 (0.4977) > uniform K=2 (0.4893).** On BLEU-4 and ROUGE-L learned wins at every
budget. Only 1 of 12 metric-budget cells is a loss (CIDEr at K=4), and measured run-to-run noise is
**~0.007 CIDEr**, so that loss is marginal.

Raw numbers: [results/eval_msrvtt_test.json](results/eval_msrvtt_test.json).

### Controls that make the above trustworthy

- **Blind control**: video input severed → CIDEr **0.019 vs 0.311** sighted. The vision path is real.
- **Oracle ceiling**: an arm allowed to see the ground-truth caption when picking frames. Bounds
  headroom at +0.0725 CIDEr at K=1. Not a method — it exists to give the result a scale.
- **Noise floor**: ~0.007 CIDEr between two independent scorer trainings.

### Human validation — the honest negative

TVSum/SumMe carry per-frame importance labelled by humans. Mean within-video Spearman, paired
per-video tests:

| dataset | n | learned | motion | random | learned − motion (paired) |
|---|---|---|---|---|---|
| TVSum | 50 | −0.055 | +0.106 | −0.012 | **−0.161**, CI [−0.269, −0.052] — significantly worse |
| SumMe | 25 | +0.113 | +0.078 | +0.009 | +0.035, CI [−0.077, +0.147] — not significant |

**`random` lands on zero in both**, so the harness is calibrated.

**The honest summary: across both human-labelled datasets the scorer never demonstrably beats the
motion baseline.** The finding is the *inversion* — learned ≫ motion for caption quality, motion ≥
learned for human importance. The scorer is a **caption-relevance predictor**, validated as one. It
is **not** a human-importance predictor.

Raw numbers: [results/validate_scorer.json](results/validate_scorer.json).

### Scorer correctness

Val Spearman ≈ **0.51** against the held-out proxy target. Converges in ONE epoch then mildly
overfits — `--epochs 3` is enough, 20 wastes time and ends slightly worse.

---

## 4. Claims you must NOT make

These were wrong at some point in this project's history and got corrected. Do not reintroduce them.

1. **"Fewer frames means faster."** FALSE as implemented. Measured: greedy decode is
   **4.73 / 4.53 / 4.49 / 4.56 s at K=1/2/4/8** — flat, because mean-pool collapses K frames to one
   vector and the projector expands to 8 prefix tokens regardless of K. Worse, selecting 1 frame of
   N requires running SigLIP on all N, so on a fresh video the learned path is ~17× *more*
   expensive than uniform K=2. The supported claim is about **information per frame**, not
   wall-clock. A real saving needs **two-tier selection** (cheap encoder scores the pool, expensive
   encoder runs on the selected K) — not implemented, and the best "what's next" answer.
2. **"The scorer matches human judgement of importance."** FALSE — see §3.
3. **"We beat published model X."** NOT ESTABLISHED. Current eval is 500 clips + greedy; published
   MSR-VTT numbers use the full 2,990-clip test split, usually with beam search. Run the full split
   (§6) before any comparison, and check the actual papers rather than trusting recalled numbers.
4. **"TVSum is scored per 2-second shot."** FALSE. TVSum and SumMe are BOTH per-frame — verified,
   `len(scores) == frame_count` at ratio exactly 1.000.

---

## 5. Where everything lives

| Artifact | Location | Cost to rebuild |
|---|---|---|
| All code + README + figures + result JSONs | GitHub `Sampath7099/VidCap` | — |
| MSR-VTT embedding cache (10k clips) | Kaggle Dataset | 8.5 GPU-h |
| `stageB.pt`, `blind.pt`, `scorer.pt` | Kaggle `vidcap-checkpoints` | 4.6 GPU-h |
| TVSum/SumMe cache + result JSONs + `scorer.pt` | `vidcap_gobag.zip` (44 MB, off-machine) | ~30 GPU-min |
| TVSum + SumMe raw videos | Kaggle `veerchheda/iitp-summe-tvsum` | 15 min download |
| MSR-VTT raw videos | `https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip` | ~15 min |

**Unzip `vidcap_gobag.zip` into the repo root** — it preserves relative paths (`out/checkpoints/`,
`results/`, `results_validation/`) and lands where the code expects.

**Two Kaggle accounts exist.** Datasets and notebooks live under **`sampathravikanti7099`**. A local
CLI was configured as `sampathravikanti` — a *different* account that has none of this. Check which
one you are authenticated as before any CLI upload.

---

## 6. What to do next — in order

### 6a. Full-split evaluation (~1–1.5 GPU-h) — highest value

Makes the result comparable to published work. See §7 for the full Kaggle session.

```bash
python -m scripts.evaluate --ckpt stageB --scorer scorer --oracle --budgets 1,2,3,4
```

No `--limit` → all 2,990 test clips. Afterwards regenerate the figure:

```bash
python -m scripts.plot_curves out/eval_msrvtt_test.json figures/budget_curves.png
```

and update the README table and the `n=500` caption.

### 6b. Q&A training (~2.5–3 GPU-h) — optional, gives Phase 2 a number

```bash
python -m scripts.fetch_qa
python -m scripts.train --stage B --task qa --init stageB --epochs 3
python -m scripts.evaluate_qa --ckpt qaB --limit 2000
```

`--init stageB` matters: it starts from the trained captioning connector. Published MSRVTT-QA
accuracy for models in this class is roughly 35–45%; report whatever you get honestly.

### 6c. Optional, none load-bearing

Stage C LoRA; diversity/shot-aware selection (now motivated by the TVSum failure, not speculation);
best-by-val checkpointing (the shipped scorer is the last epoch, not the best — so the headline is
*understated*); training the scorer on longer multi-shot video.

---

## 7. The Kaggle session (both jobs, one session, ~4.5 h)

**Settings:** Accelerator **GPU**, Internet **ON**.
**Add Input:** your MSR-VTT embedding-cache dataset, and `vidcap-checkpoints` (must contain
`stageB.pt` *and* `scorer.pt`).

### Cell 1 — clone + env

```python
!rm -rf /kaggle/working/VidCap
!git clone -q https://github.com/Sampath7099/VidCap.git /kaggle/working/VidCap
%cd /kaggle/working/VidCap
import os
os.environ["VIDCAP_DATA"] = "/kaggle/working/data"
os.environ["VIDCAP_OUT"] = "/kaggle/working"
!git log --oneline -1
```

`VIDCAP_DATA` / `VIDCAP_OUT` must be set **before** importing anything from `vidcap` —
`config.py` reads them at import time, and subprocesses otherwise fall back to `/kaggle/input`.

### Cell 2 — videos, cache, checkpoints (~20 min)

```python
!mkdir -p /kaggle/working/data /kaggle/working/checkpoints /kaggle/working/cache
!wget -q https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip -O /tmp/MSRVTT.zip
!unzip -q -o /tmp/MSRVTT.zip -d /kaggle/working/data/msrvtt && rm /tmp/MSRVTT.zip

import pathlib, shutil, os
root = pathlib.Path("/kaggle/input")

cache_src = next(p for p in root.rglob("*/msrvtt") if p.is_dir() and any(p.glob("*.npz")))
link = pathlib.Path("/kaggle/working/cache/msrvtt")
if link.is_symlink() or link.exists(): link.unlink()
link.symlink_to(cache_src)

for pt in root.rglob("*vidcap*/**/*.pt"):
    shutil.copy(pt, "/kaggle/working/checkpoints/")

have = os.listdir("/kaggle/working/checkpoints")
print("cache shards:", len(list(link.glob("*.npz"))))      # expect ~10000
print("captions npz:", (link / "_captions.npz").exists())  # must be True for --oracle
print("checkpoints:", have)
assert "stageB.pt" in have and "scorer.pt" in have, "missing checkpoint"
```

All four checks must pass before continuing.

### Cells 3–8

```python
# Cell 3 — full-split eval (~1-1.5 GPU-h)
!python -m scripts.evaluate --ckpt stageB --scorer scorer --oracle --budgets 1,2,3,4

# Cell 4 — save it IMMEDIATELY, do not wait for the session to end
!cp /kaggle/working/eval_msrvtt_test.json /kaggle/working/eval_full_test.json

# Cell 5 — QA annotations (~40MB)
!python -m scripts.fetch_qa

# Cell 6 — QA training (~2-2.5 GPU-h)
!python -m scripts.train --stage B --task qa --init stageB --epochs 3

# Cell 7 — QA eval (~20 min)
!python -m scripts.evaluate_qa --ckpt qaB --limit 2000

# Cell 8 — save everything
!cd /kaggle/working && zip -qr session_results.zip checkpoints/qaB.pt eval_full_test.json
```

Eval runs before QA deliberately: if the session dies you keep the number that matters.

### For the TVSum/SumMe validation instead

Attach `veerchheda/iitp-summe-tvsum`. Derive roots rather than guessing mount names — the mount
path has been `/kaggle/input/datasets/<user>/<slug>/`, not the usual `/kaggle/input/<slug>/`:

```python
tvsum_root = next(root.rglob("*anno.tsv")).parent.parent
summe_root = next(root.rglob("GT/*.mat")).parent.parent
# symlink /kaggle/working/data/{tvsum,summe} to these, then assert link.exists()
```

```bash
python -m scripts.build_cache tvsum && python -m scripts.build_cache summe
python -m scripts.validate_scorer --datasets tvsum,summe
```

---

## 8. Traps — every one of these cost real time

1. **`ln -sfn` to a wrong path creates a dangling symlink silently.** `rglob` over one returns
   nothing. Always `assert link.exists()`. The loaders now fail loudly, but assert anyway.
2. **Stale module cache.** After `git pull` in a live kernel, Python keeps the old module. Restart
   the kernel, or assert on source: `assert 'GT/*.mat' in inspect.getsource(datasets.summe)`.
3. **A clone into an existing directory fails**, and the cell after it runs anyway on old code.
   `rm -rf` then clone.
4. **MSR-VTT video files are required even though training reads only cached shards** — the loaders
   enumerate clips by scanning video files.
5. **Do not run `./run_tests.sh` while a cache build is running.** `test_model.py` loads Qwen at
   ~6.2 GB under a 7 G cap; with SigLIP holding RAM it gets OOM-killed and looks like a test
   failure. It passes alone.
6. **Publish results as a Kaggle Dataset before closing a session.** An accelerator change wipes
   `/kaggle/working`. A scorer was lost this way once already.
7. **CPU embedding is 3.17 s/frame** on a laptop i5 — the TVSum+SumMe cache is ~15.7 h locally vs
   ~30 min on a T4. Never build a cache locally.
8. **`*.pt`, `*.npz`, `*.zip`, `out/`, `data/` are gitignored.** Checkpoints have never been in git,
   deliberately — a committed 1.3 GB zip once blocked every push (GitHub rejects blobs >100 MB).

---

## 9. Windows setup

```bash
git clone https://github.com/Sampath7099/VidCap.git
cd VidCap
```

Then unzip `vidcap_gobag.zip` into the repo root.

**Nothing remaining requires Linux.** Both outstanding jobs run on Kaggle in a browser.

**If you want the Linux workflow back**, install **WSL2** and clone inside it. That avoids the
permissions/symlink/line-ending problems that appear when a git tree lives on NTFS. Do **not** put
the repo on a shared NTFS/exFAT partition — the executable bit on `run_tests.sh` is lost, symlinks
do not translate, and CRLF breaks shell scripts.

**Do not repartition this machine.** The Windows volumes are BitLocker-encrypted (`nvme0n1p3`,
613.7 G), so Linux cannot mount them without `dislocker`, and the ext4 root has only ~24 GB free.
The existing arrangement — code on GitHub, data on Kaggle — already is the cross-platform solution.
An exFAT USB drive covers bulk files if needed. **Share data, never code.**

**Local inference (optional):**

```bash
pip install -r requirements.txt
python -m scripts.caption yourclip.mp4 --k 2 --select learned,uniform
```

Needs `out/checkpoints/stageB.pt` and `scorer.pt`, plus ~10 GB free RAM for both models in fp32.
Python is cross-platform here; nothing in the code is Linux-specific. The `systemd-run` memory cap
used on Linux has no Windows equivalent, so close other applications if RAM is tight.

**Tests:** `run_tests.sh` is bash. On Windows run the files directly:
`python test_phase0.py`, `python test_model.py`, … or use WSL2.

---

## 10. Architecture, in brief

```
video ─► ~3fps candidate pool ─► frozen SigLIP ─► cached embeddings
                                                        │
                                   ┌────────────────────┴────────────────────┐
                                   │  frame-relevance scorer → top-K         │  ← the contribution
                                   └────────────────────┬────────────────────┘
                                                        ▼
                               connector ─► projector ─► frozen Qwen2.5-1.5B (+LoRA) ─► caption
```

| Component | Params | State |
|---|---|---|
| SigLIP so400m-384 | ~0.88B | frozen, never trained |
| Qwen2.5-1.5B-Instruct | 1.54B | frozen base |
| Connector (mean-pool, shipped) + projector | 61.95M | trained from scratch |
| Frame scorer | 0.72M | trained from scratch |
| LoRA r=8 (q,k,v,o × 28 layers) | 2.18M | Stage C, not run |

The vision encoder runs **once per video** at dataset-build time and embeddings are cached — that
is what makes every later stage cheap enough for a free T4. The 8.5 GPU-h cache is the asset the
whole project rests on.

**Key design decisions and why:**
- **Mean-pool ships**, beating the temporal transformer on a controlled run (val 3.393 vs 3.958).
  Its cost is temporal blindness — it averages the K frames, so ordering is inexpressible.
- **The scorer never sees text at inference.** Captions build the training target only, so there is
  no chicken-and-egg problem.
- **Hard top-K selection**, deliberately avoiding RL/Gumbel-softmax differentiable selection —
  the standard way this idea burns weeks on instability instead of producing a result.
- **Training uses random frame selection**, not uniform. Training on uniform and evaluating another
  selector is a confound: the connector would adapt to uniform's statistics and every other
  selector would be out-of-distribution. Random makes it selection-agnostic by construction.
- **Budgets are K=1–4, not 2/4/8/16.** MSR-VTT saturates at K=4 (0.5397/0.5376/0.5387 at K=4/8/16).
  The originally planned sweep would have measured nothing and read as "selection does not help".

**Written from scratch:** the scorer and its supervision scheme; LoRA from raw tensor ops (not
`peft`); three connectors; the fusion/prefix attention-masking scheme; greedy and beam search; the
staged training recipe and losses; the full evaluation harness; BLEU-4/ROUGE-L/CIDEr-D.

**Reused, deliberately:** frozen SigLIP, frozen Qwen2.5-1.5B, PyTorch, OpenCV.

---

## 11. Repo layout

```
vidcap/config.py       paths (VIDCAP_DATA / VIDCAP_OUT), model ids, pool fps and caps
vidcap/video.py        video file -> candidate frame pool + timestamps
vidcap/encoder.py      frozen SigLIP load, image/text embedding, atomic per-video .npz cache
vidcap/datasets.py     msrvtt / msrvtt_qa / tvsum / summe / activitynet / holdout + leakage gate
vidcap/data.py         split loading, batching, uniform/random frame index helpers
vidcap/model.py        connectors, projector, VideoCaptioner, blind control
vidcap/scorer.py       FrameScorer, listwise loss, clip_targets, spearman
vidcap/lora.py         from-scratch LoRA (no peft)
vidcap/decode.py       greedy + beam search, prompt priming for QA
vidcap/metrics.py      BLEU-4, ROUGE-L, CIDEr-D, qa_accuracy
vidcap/checkpoint.py   atomic resumable save/load

scripts/build_cache.py    Stage 0, resumable
scripts/train.py          Stage B/C, --task caption|qa
scripts/train_scorer.py   Stage A
scripts/evaluate.py       budget curves, blind + oracle arms
scripts/validate_scorer.py TVSum/SumMe vs human labels
scripts/evaluate_qa.py    MSRVTT-QA exact-match accuracy
scripts/caption.py        the demo: video in, caption out
scripts/summarize.py      composed multi-sentence summary (qualitative, no metric)
scripts/plot_curves.py    regenerates the README figure from an evaluate JSON
scripts/fetch_qa.py       MSRVTT-QA annotations
scripts/validate_metrics.py  equivalence check vs pycocoevalcap

test_*.py (8 files)    every gate; run_tests.sh runs all of them under a memory cap
```

Cache format: `out/cache/<dataset>/<video_id>.npz` → `emb (N,1152) float32` (raw, **not**
L2-normalised — normalise at use) + `times (N,)` seconds. Caption text embeddings for the scorer's
training targets: `out/cache/<dataset>/_captions.npz`.

---

## 12. How to pitch this project

> "Every video captioning pipeline samples frames uniformly. I asked whether that leaves
> performance on the table, and built the controls to answer it properly — a blind baseline to
> prove the model uses vision at all, an oracle to bound how much headroom exists, and a measured
> noise floor so I know which differences are real. One learned frame beats two uniform ones,
> recovering 66% of the achievable headroom. I also tested it against human importance labels,
> where it *doesn't* hold up — so the scorer predicts caption-relevance, not human importance, and
> I can tell you the difference."

The negative result is an asset, not damage control. Most candidates with a leaderboard number
cannot tell you whether their model is even looking at the image. Lead with the controls.

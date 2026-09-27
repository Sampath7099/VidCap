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

**Video Q&A — trained and measured 2026-09-27** (`qaB`, 1 epoch). MSRVTT-QA exact match,
5,000 test questions spread over 2,962 videos: uniform / motion / **learned** at K=1 =
0.375 / 0.372 / **0.397**; K=2 0.396 / 0.390 / 0.402; K=4 0.407 / 0.402 / 0.402. Answer-prior
floor 0.100. Same shape as captioning: learned helps at K=1, gone by K=4. Raw:
[results/evalqa_msrvtt_qa_test.json](results/evalqa_msrvtt_qa_test.json).

**`scripts/watch.py`** (2026-09-27) — video in → scene timeline (learned selector per scene) → Qwen-Instruct summary → questions answered from frames (qaB) and from the timeline. Tested with unit gates and a synthetic-video smoke run; not yet run with the real checkpoints. Needs `stageB.pt` + `scorer.pt` (+ `qaB.pt`) in `out/checkpoints/`. Details and limits: README "Watch a video" and context.md.

**Composed summaries** (`scripts/summarize.py`) — qualitative only, and weak: "How many: two" in
9 of 10 outputs is the answer prior. [results/summaries.txt](results/summaries.txt). Do not
present the summariser as validated.

**Not done:** Stage C LoRA; diversity-aware selection; connector and LoRA-rank ablations. None are
load-bearing.

---

## 3. The results (use these numbers)

### Headline — MSR-VTT test, FULL 2,990-clip split, greedy. CIDEr-D (run 2026-09-26)

| K | uniform | motion | **learned** | oracle | learned − uniform | % of ceiling |
|---|---|---|---|---|---|---|
| 1 | 0.4371 | 0.4434 | **0.5004** | 0.5459 | **+0.0633** | **58%** |
| 2 | 0.4930 | 0.4847 | **0.5154** | 0.5504 | +0.0225 | 39% |
| 3 | 0.5197 | 0.4931 | **0.5226** | 0.5537 | +0.0029 | 8% |
| 4 | 0.5301 | 0.5007 | 0.5270 | 0.5545 | −0.0031 | — |

**The K=1 gain (+0.063, ~9× noise) is the claim to lead with.** learned K=1 ≥ uniform K=2 on all
three metrics, but on CIDEr the margin is +0.0074 — AT the ~0.007 noise floor. Say "one learned
frame is worth two uniform ones", not "clearly beats". On BLEU-4 and ROUGE-L learned wins at every
budget; only CIDEr at K=4 is a loss, and it is inside the noise.

Raw numbers: [results/eval_full_test.json](results/eval_full_test.json). The older 500-clip run
(K=1 +0.0482, 66%) is [results/eval_msrvtt_test.json](results/eval_msrvtt_test.json) — superseded,
kept for history. Do not quote "66%" any more.

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
3. **"We beat published model X."** NOT ESTABLISHED. The full split is now done, but greedy only;
   published MSR-VTT numbers usually use beam search. Check the actual papers rather than trusting
   recalled numbers before any comparison.
4. **"TVSum is scored per 2-second shot."** FALSE. TVSum and SumMe are BOTH per-frame — verified,
   `len(scores) == frame_count` at ratio exactly 1.000.
5. **"One learned frame clearly beats two uniform ones."** Too strong on the full split — the CIDEr
   margin is +0.0074, at the noise floor. See §3.

---

## 5. Where everything lives

| Artifact | Location | Cost to rebuild |
|---|---|---|
| All code + README + figures + result JSONs | GitHub `Sampath7099/VidCap` | — |
| MSR-VTT embedding cache (10k clips) | Kaggle Dataset | 8.5 GPU-h |
| `stageB.pt`, `blind.pt`, `scorer.pt` | Kaggle `vidcap-checkpoints` | 4.6 GPU-h |
| `qaB.pt` (Q&A, 1 epoch) | Output of the 2026-09-27 Kaggle Q&A notebook version — publish it as a Dataset | ~5.5 GPU-h |
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

### 6a. Full-split evaluation — DONE 2026-09-26

Results in §3, [results/eval_full_test.json](results/eval_full_test.json); README and
`figures/budget_curves.png` updated. Wall-clock on a T4 including setup: roughly 4–5 h, not the
1–1.5 h originally estimated.

### 6b. Q&A training — DONE 2026-09-27 (results in §2). Recipe kept for reruns

```bash
python -m scripts.fetch_qa
python -m scripts.train --stage B --task qa --init stageB --epochs 1
python -m scripts.evaluate_qa --ckpt qaB --scorer scorer --budgets 1,2,4 --limit 5000
python -m scripts.summarize <clips...> --scorer scorer --k 4
```

**Measured: 4,658 steps/epoch at ~4.26 s/step on a T4 ≈ 5.5 h per epoch.** The old "2.5–3 GPU-h
for 3 epochs" estimate was wrong by ~7× (MSRVTT-QA has ~149k training questions vs 6.5k clips).
3 epochs = ~16.5 h, which does not fit one 12 h Kaggle session — a first attempt timed out at
40% and lost its checkpoint because checkpoints were written to `/tmp`. Run 1 epoch (loss was
already ~1.0 in epoch 2), with `$VIDCAP_OUT/checkpoints` symlinked into `/kaggle/working` so a
timeout keeps `qaB.pt` and a follow-up session can resume from it.

`--init stageB` matters: it starts from the trained captioning connector. Published MSRVTT-QA
accuracy for models in this class is roughly 35–45%; report whatever you get honestly, and say
"1 epoch".

### 6c. Optional, none load-bearing

Stage C LoRA; diversity/shot-aware selection (now motivated by the TVSum failure, not speculation);
best-by-val checkpointing (the shipped scorer is the last epoch, not the best — so the headline is
*understated*); training the scorer on longer multi-shot video.

---

## 7. The Kaggle Q&A run (committed, unattended, ~7.5 h)

Run it as **Save Version → Save & Run All (Commit)**, not in the interactive editor: a commit runs
on Kaggle's servers with the browser closed (12 h cap), an interactive session dies when idle.
A `!cmd` that fails does NOT stop a notebook, so every command goes through `run()`, which raises.

**Settings:** Accelerator **GPU** (T4 or P100; the code uses one GPU), Internet **ON**.
**Add Input:** your MSR-VTT embedding-cache dataset (~10k `.npz` **plus `_captions.npz`**) and
`vidcap-checkpoints` (`stageB.pt` and `scorer.pt`). To resume a timed-out run, also attach that version's output
(it contains `checkpoints/qaB.pt`).

Only `/kaggle/working` survives a commit, so working data goes in `/tmp` (otherwise 6 GB of video
and the cache symlink get archived) and **checkpoints are symlinked into `/kaggle/working`** so a
timeout keeps them — a previous run lost 7 h of Q&A training by checkpointing to `/tmp`.

### Cell 1 — code, paths, helper

```python
import os, sys, time, json, shutil, subprocess, pathlib

!rm -rf /tmp/VidCap
!git clone -q https://github.com/Sampath7099/VidCap.git /tmp/VidCap
%cd /tmp/VidCap
assert pathlib.Path("scripts/evaluate.py").exists(), "clone failed"

# Must be set before anything imports vidcap (config.py reads them at import time).
os.environ["VIDCAP_DATA"] = "/tmp/vidcap_data"
os.environ["VIDCAP_OUT"]  = "/tmp/vidcap_out"
DATA = pathlib.Path("/tmp/vidcap_data")
OUT  = pathlib.Path("/tmp/vidcap_out")
SAVE = pathlib.Path("/kaggle/working/results")
SAVE.mkdir(parents=True, exist_ok=True)

def run(cmd):
    """Run a shell command, stream its output, and FAIL the notebook if it fails."""
    t = time.time(); print(f"$ {cmd}", flush=True)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        print(line, end="", flush=True)
    if p.wait():
        raise RuntimeError(f"exit {p.returncode}: {cmd}")
    print(f"[done in {(time.time()-t)/60:.1f} min]", flush=True)

run("git log --oneline -1")
run("nvidia-smi --query-gpu=name,memory.total --format=csv")
```

### Cell 2 — videos, cache, checkpoints (~20 min)

```python
for d in (DATA, OUT / "cache"):
    d.mkdir(parents=True, exist_ok=True)

run("wget -q --timeout=60 --tries=5 https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip -O /tmp/MSRVTT.zip")
run(f"unzip -q -o /tmp/MSRVTT.zip -d {DATA}/msrvtt && rm /tmp/MSRVTT.zip")
n_vid = sum(1 for _ in (DATA / "msrvtt").rglob("*.mp4"))
print("videos:", n_vid); assert n_vid >= 9900, "MSR-VTT videos missing"

root = pathlib.Path("/kaggle/input")
caps = list(root.rglob("_captions.npz"))
assert len(caps) == 1, f"need exactly one _captions.npz in inputs, found: {caps}"
link = OUT / "cache" / "msrvtt"
if link.is_symlink() or link.exists():
    link.unlink()
link.symlink_to(caps[0].parent); assert link.exists(), "dangling cache symlink"
n_npz = sum(1 for _ in link.glob("*.npz"))
print("cache:", caps[0].parent, "shards:", n_npz); assert n_npz >= 9900

KEEP = pathlib.Path("/kaggle/working/checkpoints"); KEEP.mkdir(exist_ok=True)
ck = OUT / "checkpoints"
if ck.is_symlink():
    ck.unlink()
elif ck.exists():
    shutil.rmtree(ck)
ck.symlink_to(KEEP); assert ck.exists()
for name in ("stageB.pt", "scorer.pt", "qaB.pt"):   # qaB.pt only when resuming
    hits = list(root.rglob(name))
    if hits:
        shutil.copy(hits[0], KEEP / name)
assert (KEEP / "stageB.pt").exists(), "stageB.pt not found — is vidcap-checkpoints attached?"
print("checkpoints:", os.listdir(KEEP))
```

### Cells 3–5

```python
# Cell 3 — QA annotations (~40 MB)
run("python -m scripts.fetch_qa")

# Cell 4 — QA training, 1 epoch (~5.5 h at 4.26 s/step; resumes from qaB.pt if present)
run("python -m scripts.train --stage B --task qa --init stageB --epochs 1")

# Cell 5 — QA eval (~1.5 h): 5,000 questions spread over all 2,990 test videos,
# uniform / motion / learned selection at K=1,2,4. Writes straight into SAVE after every arm.
run(f"python -m scripts.evaluate_qa --ckpt qaB --scorer scorer --budgets 1,2,4 --limit 5000 "
    f"--out {SAVE}/evalqa_msrvtt_qa_test.json")

# Cell 6 — composed summaries on 10 test clips (qualitative, no metric; ~10 min)
vids = sorted((DATA / "msrvtt").rglob("video*.mp4"), key=lambda p: int(p.stem[5:]))
picks = [str(p) for p in vids if int(p.stem[5:]) in range(7010, 10000, 300)]
run(f"python -m scripts.summarize {' '.join(picks)} --scorer scorer --k 4 "
    f"> {SAVE}/summaries.txt 2>&1")
print(open(SAVE / "summaries.txt").read())
print("ALL DONE:", os.listdir(SAVE))
```

`--limit` samples evenly: `qa_test.json` is grouped by video, and its first 2,000 questions cover
only 83 of the 2,990 test videos (fixed 2026-09-27; `spread()` in `evaluate_qa.py`).

Afterwards: version → **Output** tab → download `results/`, and **New Dataset** from the output so
`qaB.pt` persists.

The caption evaluation used the same Cells 1–2 plus
`run("python -m scripts.evaluate --ckpt stageB --scorer scorer --oracle --budgets 1,2,3,4")`
(needs `scorer.pt`). It is done — do not rerun it.

### ActivityNet paragraph evaluation (committed, ~4.5 h)

Measures whether learned frame selection helps `watch.py`'s timeline and summary on long
videos, against human-written paragraphs. Same **Cell 1** as the Q&A run.

**Add Input:** `almirneto/activitynet-captions` (42 GB, ~15k videos as `videos/v_<id>.mp4|mkv`,
plus `val_1.json` / `val_2.json` — mounted, not downloaded) and `vidcap-checkpoints`
(`stageB.pt`, `scorer.pt`). The MSR-VTT cache is not needed. To resume a timed-out run, also
attach that version's output (it contains `cache_activitynet/`).

```python
# Cell 2 — data, cache folder that survives the commit, checkpoints
for d in (DATA, OUT / "cache", OUT / "checkpoints"):
    d.mkdir(parents=True, exist_ok=True)
root = pathlib.Path("/kaggle/input")
v1 = [p for p in root.rglob("val_1.json") if (p.parent / "videos").exists()]
assert len(v1) == 1, f"attach almirneto/activitynet-captions; found {v1}"
an = DATA / "activitynet"
if an.is_symlink() or an.exists():
    an.unlink()
an.symlink_to(v1[0].parent); assert an.exists()
print("activitynet videos:", sum(1 for _ in (an / "videos").iterdir()))

KEEP = pathlib.Path("/kaggle/working/cache_activitynet"); KEEP.mkdir(exist_ok=True)
for prev in root.rglob("cache_activitynet"):          # resume from an attached earlier output
    for f in prev.glob("*.npz"):
        if not (KEEP / f.name).exists():
            shutil.copy(f, KEEP / f.name)
link = OUT / "cache" / "activitynet"
if link.is_symlink() or link.exists():
    link.unlink()
link.symlink_to(KEEP); assert link.exists()
for name in ("stageB.pt", "scorer.pt"):
    hits = list(root.rglob(name)); assert hits, f"{name} missing"
    shutil.copy(hits[0], OUT / "checkpoints" / name)
print("resumed shards:", len(list(KEEP.glob("*.npz"))))
```

```python
# Cell 3 — embed 300 val videos, evenly spread (~2 h). Capped at 5 h so evaluation always runs;
# shards already written are kept, and caption embeddings are written first.
run("timeout 5h python -m scripts.build_cache activitynet --limit 300 || echo 'cache stopped at cap'")
n = len([p for p in KEEP.glob("*.npz") if p.stem != "_captions"])
print("cached videos:", n); assert n >= 100, "too few videos to evaluate"
assert (KEEP / "_captions.npz").exists(), "oracle arm needs caption embeddings"
```

```python
# Cell 4 — 5 arms + 2 summary arms, saved after each (~2.5 h)
run(f"python -m scripts.evaluate_paragraphs --summaries --out {SAVE}/eval_activitynet_paragraphs.json")
```

Arms: fixed 15 s windows + uniform (the naive pipeline); scenes + uniform / motion / **learned** /
oracle; Qwen summaries of the uniform and learned timelines. K=2 frames per segment. Per-video
CIDEr-D gives paired bootstrap CIs. Expect low absolute CIDEr: the metric's length penalty
punishes a 10-sentence timeline against ~3.7-sentence references, which is why the summaries
are scored too.

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

**Nothing remaining requires Linux.** The one outstanding job (Q&A) runs on Kaggle in a browser.

**If you want the Linux workflow back**, install **WSL2** and clone inside it. That avoids the
permissions/symlink/line-ending problems that appear when a git tree lives on NTFS. Do **not** put
the repo on a shared NTFS/exFAT partition — the executable bit on `run_tests.sh` is lost, symlinks
do not translate, and CRLF breaks shell scripts.

**Do not repartition this machine.** The Windows volumes are BitLocker-encrypted (`nvme0n1p3`,
613.7 G), so Linux cannot mount them without `dislocker`, and the ext4 root has only ~24 GB free.
The existing arrangement — code on GitHub, data on Kaggle — already is the cross-platform solution.
An exFAT USB drive covers bulk files if needed. **Share data, never code.**

**Local Python (set up 2026-09-26):** Python 3.12 is installed per-user and the repo has a
`.venv/` (CPU torch, requirements, matplotlib, pycocoevalcap). All 8 test files pass there. Use
`.venv\Scripts\python.exe` directly — bare `python` is the Microsoft Store stub. transformers 5
needs `protobuf` (now in requirements.txt; Kaggle preinstalls it).

**Local inference (optional):**

```bash
.venv\Scripts\python.exe -m scripts.caption yourclip.mp4 --k 2 --select learned,uniform
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
> noise floor so I know which differences are real. On the full MSR-VTT test split, choosing the
> single frame well adds +0.063 CIDEr — about nine times the noise — recovers 58% of the
> achievable headroom, and makes one frame worth two uniform ones. I also tested it against human importance labels,
> where it *doesn't* hold up — so the scorer predicts caption-relevance, not human importance, and
> I can tell you the difference."

The negative result is an asset, not damage control. Most candidates with a leaderboard number
cannot tell you whether their model is even looking at the image. Lead with the controls.

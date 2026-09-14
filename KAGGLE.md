# Running this on Kaggle — from zero

Kaggle gives you a free GPU for ~30 hours a week. You write code in a "Notebook" (a web page of
code boxes called **cells**) and press play on each one. Your laptop has no GPU, so all training
happens here.

Three things live in different places and it matters:

| Place | What it is | Survives? |
|---|---|---|
| `/kaggle/input/` | Datasets you attach. **Read-only.** | always there |
| `/kaggle/working/` | Your scratch space. Code + outputs go here. | **wiped when the session ends** unless you save it |
| The notebook itself | Your cells | saved automatically |

The big beginner trap: **everything in `/kaggle/working/` disappears when your session stops.**
Step 7 is how you keep it.

---

## 1. Account setup (once)

1. Sign up at kaggle.com.
2. Click your avatar (top right) → **Settings** → **Phone Verification**. Verify your number.
   **Without this you get no GPU and no internet in notebooks.** This is the single most common
   thing that blocks beginners.

---

## 2. Put the code somewhere Kaggle can reach it

Kaggle can't see your laptop. Easiest reliable route is GitHub — and you need a repo for this
project's final deliverable anyway.

The repo is already initialised and committed at `/home/sampath/Desktop/Projects/VidCap`, with
`origin` pointing at `https://github.com/Sampath7099/VidCap.git`. Make an **empty** repo of that
name on github.com (no README, no .gitignore — we already have one), then:

```bash
git push -u origin main
```

From now on, `git push` from the laptop and re-clone in the notebook to pick up changes.

*No GitHub?* Alternative: zip the folder, then on Kaggle go to **Datasets → New Dataset** and
upload the zip. Works, but you must re-upload after every code edit, which gets painful fast.

---

## 3. Get MSR-VTT

**Don't download it to your laptop.** The notebook has internet and far better bandwidth than
your house — fetch it there directly. 6.55 GB, verified live, no login:

```
https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip
```

This is the Frozen-in-Time mirror (Oxford VGG), the one most video-retrieval papers use. It
contains all 10k `video*.mp4` under `MSRVTT/videos/all/` plus `MSRVTT/annotation/MSR_VTT.json`.
Cells 3a/3b below do the fetch — nothing to click for this step.

*Fallback if that URL ever dies:* Kaggle → **Datasets** (left sidebar) → search `msrvtt`, sort by
**Most Votes**, and check the **Data** tab for **both** `.mp4` files and an annotation JSON. Many
mirrors ship only pre-extracted features, which are useless here — we need real video to extract
frames from. Two known ones: `vishnutheepb/msrvtt`, `mrandri19/msr-vtt`. Attach via **+ Add
Input** (step 4) and it appears at `/kaggle/input/<slug>/`.

---

## 4. Create the notebook

1. Kaggle → **Code** (left sidebar) → **New Notebook**.
2. Right-hand panel → **Session options**:
   - **Accelerator**: `GPU T4 x2` (or `GPU P100`)
   - **Internet**: **On** ← needed for `pip install` and `git clone`
   - **Persistence**: `Files only` is a good default

No **+ Add Input** needed — we fetch the data in Cell 3a. Internet **On** is what makes that work.

---

## 5. The cells

Paste each block into its own cell. Run with **Shift+Enter**. `!` means "run this as a terminal
command"; `%cd` changes directory.

**Cell 1 — dependencies** (torch and transformers are preinstalled; sentencepiece is not, and
SigLIP will not load without it)
```python
!pip install -q sentencepiece
```

**Cell 2 — get the code**
```python
!rm -rf /kaggle/working/VidCap
!git clone -q https://github.com/Sampath7099/VidCap.git /kaggle/working/VidCap
%cd /kaggle/working/VidCap
```

**Cell 3a — fetch MSR-VTT** (~6.55 GB, a few minutes on Kaggle's network)
```python
!wget -q --show-progress https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip -O /kaggle/working/MSRVTT.zip
```

**Cell 3b — unpack and verify.** Do not skip the verify; a truncated download looks like a
working one until the caching step quietly finds nothing.
```python
!unzip -q /kaggle/working/MSRVTT.zip -d /kaggle/working/data
!find /kaggle/working/data -name "*.mp4" | wc -l
!find /kaggle/working/data -iname "*MSR_VTT*.json" -o -iname "*videodatainfo*"
```
Expect **~10000** videos and at least one annotation JSON. If the count is near zero, the zip
is incomplete — delete it and re-run Cell 3a.

Delete the zip once unpacked, it's dead weight against the disk quota:
```python
!rm /kaggle/working/MSRVTT.zip
```

**Cell 4 — confirm both backbones load on the GPU** (~2 min, downloads ~6 GB)
```python
!python -m scripts.smoke_test
```
Expect `vision ok on cuda dim=1152` and a Qwen line. If this fails, stop and fix it — nothing
downstream can work.

**Cell 5 — cache 500 clips first.**
```python
!python -m scripts.build_cache msrvtt --root /kaggle/working/data --limit 500
```
This reads videos, samples ~3 frames/second, runs SigLIP on each, and saves small `.npz` files.
Roughly 10–20 minutes. Safe to re-run — it skips anything already done.

**Cell 6 — first real training run** (the moment of truth)
```python
!python -m scripts.train --stage B --limit 500 --epochs 3 --bs 16
```
Watch the loss. It should fall from ~8 and keep dropping.

**Cell 7 — the blind control**, trained identically but with the video zeroed out
```python
!python -m scripts.train --stage B --limit 500 --epochs 3 --bs 16 --blind --name blind
```

**Cell 8 — compare them**
```python
!python -m scripts.evaluate --ckpt stageB --ckpt-blind blind --limit 200 --budgets 8
```
**This is the gate.** If the sighted model doesn't beat blind, the model is ignoring the video
and nothing after it means anything. The script prints a loud warning if that happens.

---

## 6. If something breaks

Paste the error to me. Expect breakage on Cell 5 or 6 — that's exactly why we do 500 clips
before 10,000. Finding a bug after 20 minutes beats finding it after 3 hours.

---

## 7. Keeping your results (important)

When the session ends, `/kaggle/working/` is erased — **including the videos from Cell 3a**. Next
session you re-run Cell 3a and wait a few minutes. That's cheap; the expensive thing is the
embedding cache, which costs GPU-hours off your weekly quota. Keep that:

```python
!cd /kaggle/working && zip -qr cache.zip cache && ls -lh cache.zip
```

Then either:
- **Save Version** (top right) → the file is kept as notebook output, or
- Download it and re-upload as your own Kaggle Dataset — better, because you can attach it to
  future notebooks as `/kaggle/input/` and skip caching entirely.

---

## 8. Long runs without sitting there

Clicking cells keeps the session alive only while the browser is open, and it idles out after
inactivity.

For real training: **Save Version → Save & Run All (Commit)**. Kaggle runs the whole notebook on
its own servers — you can close the laptop. Max 12 hours per run. You'll get an email when it
finishes.

This is why every training script here checkpoints and resumes: hit the 12-hour wall, start a new
run, and it picks up where it stopped.

---

## Gotchas

- **Internet Off** → `pip install` and `git clone` both fail. Check the right panel.
- **No GPU option** → phone verification not done (step 1).
- **GPU quota** is ~30 h/week and resets weekly. Don't burn it on debugging — debug on 500 clips.
- `/kaggle/input/` is **read-only**. Never try to write there. `/kaggle/working/` is where we put
  the videos, so `--root` points there, not at `/kaggle/input`.
- `/kaggle/working/` has a ~20 GB quota. The zip (6.5) + unpacked videos (6.5) is already 13 —
  hence the `rm` after unzip. Skip it and Cell 5 can die on a full disk.
- Re-running Cell 2 wipes and re-clones the code, but leaves `cache/` and `data/` alone — deliberate.
- Restarting the session loses installed packages. Cell 1 runs again each time.

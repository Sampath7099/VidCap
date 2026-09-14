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

On your laptop, in `/home/sampath/Desktop/Projects/VidCap`:

```bash
git init && git add -A && git commit -m "VidCap: phase 0 + vanilla captioner"
```

Then make an **empty** repo on github.com (no README, no .gitignore — we already have one), and:

```bash
git remote add origin https://github.com/YOUR_USERNAME/VidCap.git
git branch -M main && git push -u origin main
```

From now on, `git push` from the laptop and re-clone in the notebook to pick up changes.

*No GitHub?* Alternative: zip the folder, then on Kaggle go to **Datasets → New Dataset** and
upload the zip. Works, but you must re-upload after every code edit, which gets painful fast.

---

## 3. Find and attach MSR-VTT

1. Kaggle → **Datasets** (left sidebar) → search `msr-vtt` or `msrvtt`.
2. Open a few results and check the **Data** tab. You want one that has **both**:
   - the video files (`.mp4`), and
   - the annotation JSON (a file with `videodatainfo` in the name)

   Some mirrors ship only annotations, or only pre-extracted features. Those won't work — we
   need actual video to extract frames from.
3. Note its name. Prefer one with more upvotes and a recent update.

Dataset slugs change and mirrors get deleted, so verify by eye rather than trusting a link.

---

## 4. Create the notebook

1. Kaggle → **Code** (left sidebar) → **New Notebook**.
2. Right-hand panel → **Session options**:
   - **Accelerator**: `GPU T4 x2` (or `GPU P100`)
   - **Internet**: **On** ← needed for `pip install` and `git clone`
   - **Persistence**: `Files only` is a good default
3. Right-hand panel → **Input** → **+ Add Input** → search your MSR-VTT dataset → **Add**.

It now appears under `/kaggle/input/<dataset-slug>/`.

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
!git clone -q https://github.com/YOUR_USERNAME/VidCap.git /kaggle/working/VidCap
%cd /kaggle/working/VidCap
```

**Cell 3 — look at what the dataset actually contains.** Do not skip this. Every mirror nests
its folders differently, and you need the real paths before anything else works.
```python
!ls /kaggle/input/
!find /kaggle/input -name "*videodatainfo*" | head
!find /kaggle/input -name "*.mp4" | head -3
!find /kaggle/input -name "*.mp4" | wc -l
```

**Cell 4 — confirm both backbones load on the GPU** (~2 min, downloads ~6 GB)
```python
!python -m scripts.smoke_test
```
Expect `vision ok on cuda dim=1152` and a Qwen line. If this fails, stop and fix it — nothing
downstream can work.

**Cell 5 — cache 500 clips first.** Use the folder from Cell 3 as `--root`.
```python
!python -m scripts.build_cache msrvtt --root /kaggle/input/YOUR-DATASET-SLUG --limit 500
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

When the session ends, `/kaggle/working/` is erased. To keep the embedding cache so you never
recompute it:

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
- `/kaggle/input/` is **read-only**. Never try to write there.
- Re-running Cell 2 wipes and re-clones the code, but leaves `cache/` alone — that's deliberate.
- Restarting the session loses installed packages. Cell 1 runs again each time.

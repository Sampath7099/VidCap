# VidCap

**Look at the right frame, not just any frame.**

When an AI describes a video, it can't afford to study every frame, so it looks at a handful.
Almost every system picks those few frames blindly and evenly: one every few seconds, whatever is
in them. That works until the one moment that matters falls between the frames it picked.

VidCap adds a small learned **frame picker** that glances at every frame and chooses the ones worth
looking at.

![evenly spaced vs VidCap frame](figures/examples.png)

On top of that picker sits a small video assistant that can:

- **caption** a clip in one sentence,
- **summarize** a longer video into a short paragraph, with a timeline of what happened, and
- **answer questions** about it.

## How it works

```
video ─► frames ─► SigLIP (frozen) ─► frame picker ─► best frames ─► bridge ─► Qwen2.5 (frozen) ─► text
```

- **SigLIP** (Google, ~0.9B parameters) turns each frame into a vector describing what's in it.
  It's used as-is, never retrained.
- **The frame picker** is ours: a tiny 0.72M-parameter network that scores every frame for how
  useful it is. It was trained to predict, from the picture alone, how well a frame matches what
  people wrote about the video.
- **The bridge** is ours too: a 62M-parameter network that translates the chosen frames into
  eight "visual words" a language model can read.
- **Qwen2.5-1.5B-Instruct** writes the caption, the summary and the answers. It's also used
  as-is. Captioning and Q&A share one copy of it and swap only the small bridge.

Everything was trained on the free Kaggle GPU tier, on 10,000 MSR-VTT clips and 149,000 MSRVTT-QA
questions.

## Try it

### 1. Set up

```bash
git clone https://github.com/Sampath7099/VidCap.git
cd VidCap
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Add the trained weights

Put these three files in `out/checkpoints/`:

| File | What it is |
|---|---|
| `stageB.pt` | the captioning bridge |
| `scorer.pt` | the frame picker |
| `qaB.pt` | the question-answering bridge (optional; adds answers read straight from the frames) |

The two big models (SigLIP and Qwen) download automatically from Hugging Face the first time you
run anything.

### 3. Open the app

```bash
python app.py
```

Then go to http://127.0.0.1:7860, upload a video and pick a tab:

- **Caption:** one sentence, from 1–4 frames. Switch between the smart picker and evenly spaced
  frames to see the difference, along with the frames it actually looked at.
- **Summarize:** a short paragraph plus a timeline of what happened.
- **Ask:** type a question. You get two answers: one read from the frames that best match your
  question, one from the timeline with a rough timestamp.

It runs on a laptop CPU, which needs about 10 GB of free RAM. The first request loads the models
(about a minute), and each new video takes a few minutes to read. With a GPU it's close to
instant.

### Prefer the command line?

```bash
python -m scripts.caption myclip.mp4 --k 1 --select learned,uniform    # compare pickers
python -m scripts.watch myvideo.mp4 --budget 8                         # summary, then ask away
python -m scripts.watch myvideo.mp4 --budget 8 --ask "what is the man holding?" --json out.json
```

## Results

In short: from a single frame, VidCap's captions score **14% higher** than the usual evenly spaced sampling, and one VidCap frame is worth two evenly spaced ones. Full numbers, examples and how each was measured are in [RESULTS.md](RESULTS.md).

## Reproducing the training

Training runs on cached SigLIP vectors, so after a one-time encoding pass everything fits on a
free GPU. See [DATASETS.md](DATASETS.md) for where to get the data.

```bash
python -m scripts.build_cache msrvtt                                   # encode frames once (~8.5 GPU-h)
python -m scripts.train --stage B --epochs 10 --bs 16                  # captioning bridge (~2.5 h)
python -m scripts.train_scorer --epochs 3                              # frame picker (minutes)
python -m scripts.fetch_qa
python -m scripts.train --stage B --task qa --init stageB --epochs 1   # Q&A bridge (~5.5 h)
python -m scripts.evaluate --ckpt stageB --scorer scorer --oracle --budgets 1,2,3,4
python -m scripts.evaluate_qa --ckpt qaB --scorer scorer --budgets 1,2,4 --limit 5000
```

The captioning metrics are written from scratch and match the official `pycocoevalcap`
implementation to 11 decimal places. Tests: run `python test_<name>.py` for any test file, or
`./run_tests.sh` on Linux.

## Project layout

```
app.py          the web app
vidcap/         model, frame picker, data loading, metrics, video timeline
scripts/        training, evaluation, and the caption / watch command-line tools
results/        raw numbers behind every table above
figures/        the plots in this README
test_*.py       tests
```

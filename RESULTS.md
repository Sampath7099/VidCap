# Results

Everything here was measured on data the models never saw during training. Raw numbers for every table are in [results/](results/).

## What the difference looks like

Each row is one MSR-VTT test clip described from a **single frame**. On the left is the frame evenly spaced sampling picks; on the right, the frame VidCap picks. These clips were picked because the difference is easy to see; the averages below cover all 2,990 test clips.

![evenly spaced vs VidCap frame](figures/examples.png)

### Captioning: one well-chosen frame is worth two evenly spaced ones

MSR-VTT test set, all 2,990 clips. The captioner is identical in every column; only the choice of
frames changes. CIDEr-D is the standard captioning score (higher is better).

![quality vs frame budget](figures/budget_curves.png)

| Frames | Evenly spaced | **VidCap picker** | Improvement |
|---|---|---|---|
| 1 | 0.437 | **0.500** | **+14%** |
| 2 | 0.493 | **0.515** | +5% |

- With a single frame, the picker recovers **58%** of the best improvement possible. To measure
  that ceiling, we let a "cheating" picker read the human-written captions before choosing frames.
- **One VidCap frame (0.500) matches two evenly spaced frames (0.493).**
- The same holds on the other standard scores: at one frame, BLEU-4 goes from 0.352 to 0.394 and
  ROUGE-L from 0.588 to 0.612.
- The gain is about **9× larger** than the variation between two independent trainings of the
  picker, so it isn't luck.

### Question answering

MSRVTT-QA, 5,000 test questions spread across the whole test set. Answers are single words.

| Frames | Evenly spaced | **VidCap picker** |
|---|---|---|
| 1 | 37.5% | **39.7%** |

For scale: always answering "man", the most common answer, gets 10%.

### Long videos

On 300 ActivityNet Captions videos (about 2 minutes each, never trained on), the picker again
recovers **58%** of the possible improvement when only two frames can describe the whole video.
The summarizer also turns a list of raw captions into a noticeably better paragraph: roughly
**+50–85%** on CIDEr-D against human-written descriptions.

### It really is watching the video

With the video input switched off, the same captioner's score falls from **0.311 to 0.019**. The
captions come from what it sees, not from guessing a typical sentence.

The picker helps most when the budget is tight (1–2 frames). Give the model four or more frames of
a short clip and evenly spaced sampling catches up.

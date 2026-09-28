"""VidCap in the browser: caption a clip, summarise a video, or ask about it.

  python app.py            # then open http://127.0.0.1:7860

Needs out/checkpoints/stageB.pt and scorer.pt; qaB.pt adds answers read from the frames.
Models load on first use and stay in memory; each uploaded video is embedded once and reused
across tabs and questions.
"""
from pathlib import Path

import gradio as gr
import torch

from scripts.caption import caption_from_pool
from scripts.evaluate import load_model, make_learned, uniform_sel
from scripts.watch import answer, build_budget_timeline, summarize_timeline
from vidcap import checkpoint
from vidcap.config import CKPT
from vidcap.data import uniform_indices
from vidcap.encoder import embed_images, load_vision
from vidcap.timeline import fmt_time, timeline_text
from vidcap.video import sample_frames

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FPS, MAX_FRAMES = 3.0, 128     # the captioner's training density; capped so CPU stays bearable
SUMMARY_FRAMES = 8             # 8 evenly spaced frames: the best long-video setting measured
_models, _video = {}, {}


def models():
    if not _models:
        if not (Path(CKPT) / "stageB.pt").exists():
            raise gr.Error(f"Missing {Path(CKPT) / 'stageB.pt'} — download it from the "
                           "vidcap-checkpoints Kaggle dataset.")
        cap, _ = load_model("stageB", DEVICE)
        vis, proc, _ = load_vision(DEVICE)
        qa = checkpoint.load("qaB", map_location=DEVICE)
        _models.update(cap=cap, cap_ck=checkpoint.load("stageB", map_location=DEVICE),
                       qa_ck=qa if qa and qa.get("task") == "qa" else None,
                       learned=make_learned("scorer", DEVICE), vision=(vis, proc))
    return _models


def video(path):
    """Frames, timestamps and SigLIP embeddings for this file, computed once."""
    if not path:
        raise gr.Error("Upload a video first.")
    if _video.get("path") != path:
        m = models()
        frames, times = sample_frames(path, fps=FPS, max_frames=MAX_FRAMES)
        if len(frames) == 0:
            raise gr.Error("Could not read that video.")
        _video.clear()
        _video.update(path=path, frames=frames, times=times,
                      emb=embed_images(*m["vision"], DEVICE, frames))
    return _video


def timeline(v):
    if "events" not in v:
        m = models()
        v["events"] = build_budget_timeline(v["emb"], v["times"], m["cap"],
                                            uniform_indices(len(v["emb"]), SUMMARY_FRAMES), DEVICE)
    return v["events"]


def do_caption(path, k, how):
    v, m = video(path), models()
    select = m["learned"] if how == "Smart (learned picker)" else uniform_sel
    cap, idx = caption_from_pool(v["emb"], int(k), select, m["cap"], DEVICE)
    shown = [(v["frames"][i], f"{fmt_time(v['times'][i])}") for i in dict.fromkeys(idx)]
    return cap[:1].upper() + cap[1:], shown


def do_summary(path):
    v, m = video(path), models()
    events = timeline(v)
    return summarize_timeline(m["cap"], events), timeline_text(events)


def do_ask(path, question):
    if not question.strip():
        raise gr.Error("Type a question.")
    v, m = video(path), models()
    a = answer(question, timeline(v), v["emb"], m["cap"], m["qa_ck"], m["cap_ck"],
               m["vision"], DEVICE)
    frames = (f"{a['frames']['answer']}  (from the frames at {a['frames']['when']})"
              if a["frames"] else "Not available — add qaB.pt to out/checkpoints/.")
    t = a["timeline"]
    return frames, t["answer"] + (f"  (around {t['when']})" if t["when"] else "")


with gr.Blocks(title="VidCap") as demo:
    gr.Markdown("# VidCap\nCaption a clip, summarise a video, or ask about it. "
                "The first run loads the models and takes a minute; on a CPU each new video "
                "takes a few minutes to read.")
    vid = gr.Video(label="Video", sources=["upload"])
    with gr.Tab("Caption"):
        with gr.Row():
            k = gr.Slider(1, 4, value=1, step=1, label="Frames to look at")
            how = gr.Radio(["Smart (learned picker)", "Evenly spaced"],
                           value="Smart (learned picker)", label="How to pick them")
        go_c = gr.Button("Caption it", variant="primary")
        out_c = gr.Textbox(label="Caption")
        gal = gr.Gallery(label="Frames it looked at", height=220, columns=4)
        go_c.click(do_caption, [vid, k, how], [out_c, gal])
    with gr.Tab("Summarize"):
        go_s = gr.Button("Summarize", variant="primary")
        out_s = gr.Textbox(label="Summary", lines=4)
        out_t = gr.Textbox(label="Timeline", lines=6)
        go_s.click(do_summary, [vid], [out_s, out_t])
    with gr.Tab("Ask"):
        q = gr.Textbox(label="Question", placeholder="What is the man holding?")
        go_q = gr.Button("Ask", variant="primary")
        out_f = gr.Textbox(label="Answer from the frames")
        out_l = gr.Textbox(label="Answer from the timeline")
        go_q.click(do_ask, [vid, q], [out_f, out_l])
        q.submit(do_ask, [vid, q], [out_f, out_l])

if __name__ == "__main__":
    demo.launch()

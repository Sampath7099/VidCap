"""Dataset loaders. Every loader returns records: {video_id, path, split, captions}.

Human-importance loaders (TVSum/SumMe) return {video_id: per-shot/per-frame score array}
and exist only to validate the scorer in Phase 1 — they are never a training input.
"""
import json
from pathlib import Path

import numpy as np

from .config import DATA

VIDEO_EXTS = (".mp4", ".webm", ".mkv", ".avi")


def _find_videos(root):
    """{stem: path} for every video under root."""
    root = Path(root)
    return {p.stem: p for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXTS}


# --- MSR-VTT: primary training + captioning-quality eval ------------------------

def _official_split(video_id):
    """MSR-VTT's published split is a contiguous id range: 0-6512 train, 6513-7009 val,
    7010-9999 test. Mirrors that drop the per-video 'split' field still follow it, so this
    reproduces the official 6513/497/2990 rather than inventing a split."""
    try:
        n = int(video_id.lower().replace("video", ""))
    except ValueError:
        return "test"
    return "train" if n < 6513 else "val" if n < 7010 else "test"


def msrvtt(root=None):
    """Official 6513/497/2990 split, from the annotation JSONs where present."""
    root = Path(root or DATA / "msrvtt")
    vids = _find_videos(root)
    caps, splits = {}, {}
    # Mirrors disagree on the filename: the official release ships *videodatainfo*.json,
    # the Frozen-in-Time zip ships MSR_VTT.json.
    anns = sorted(p for p in root.rglob("*.json")
                  if "videodatainfo" in p.name.lower() or "msr_vtt" in p.name.lower())
    if not anns:
        raise FileNotFoundError(
            f"no MSR-VTT annotation JSON under {root} "
            "(looked for *videodatainfo*.json and MSR_VTT.json)")
    for js in anns:
        d = json.loads(js.read_text())
        for v in d.get("videos", []):
            if v.get("split"):
                splits[v["video_id"]] = {"validate": "val"}.get(v["split"], v["split"])
        # ...and on the caption schema: official uses sentences/[video_id], the
        # Frozen-in-Time repack uses COCO-style annotations/[image_id]. Same content.
        for s in d.get("sentences", []) + d.get("annotations", []):
            vid, cap = s.get("video_id") or s.get("image_id"), s.get("caption")
            if vid and cap:
                caps.setdefault(vid, []).append(cap)
    if not caps:
        raise ValueError(
            f"parsed {[p.name for p in anns]} but found no captions — unexpected schema. "
            f"Top-level keys: {sorted(json.loads(anns[0].read_text()).keys())}")
    return [{"video_id": k, "path": vids[k],
             "split": splits.get(k) or _official_split(k), "captions": caps.get(k, [])}
            for k in sorted(caps) if k in vids]


# --- MSRVTT-QA: video question answering over the SAME clips as msrvtt -----------

QA_SPLITS = {"train": "qa_train.json", "val": "qa_val.json", "test": "qa_test.json"}


def msrvtt_qa(root=None):
    """One record per question: {video_id, path, split, question, answer, answer_type}.

    Splits come from the annotation filenames and coincide exactly with MSR-VTT's official video
    ranges (val is video6513-7009), so nothing needs re-caching — the shards built for captioning
    serve QA unchanged. Answers are single words; the metric is exact-match accuracy.
    """
    root = Path(root or DATA / "msrvtt")
    vids = _find_videos(root)
    recs = []
    for split, fname in QA_SPLITS.items():
        hits = sorted(root.rglob(fname))
        if not hits:
            continue
        for q in json.loads(hits[0].read_text()):
            vid = str(q.get("video", "")).rsplit(".", 1)[0]
            if vid in vids:
                recs.append({"video_id": vid, "path": vids[vid], "split": split,
                             "question": q["question"], "answer": str(q["answer"]),
                             "answer_type": q.get("answer_type", "")})
    if not recs:
        raise FileNotFoundError(
            f"no MSRVTT-QA annotations under {root} (looked for {list(QA_SPLITS.values())}). "
            "Run: python -m scripts.fetch_qa")
    return recs


# --- TVSum / SumMe: human-annotated frame importance (scorer ground truth) -------

def tvsum(root=None):
    """Records + {video_id: (n_shots,) mean importance over 20 annotators}. Shots are 2s."""
    root = Path(root or DATA / "tvsum")
    vids = _find_videos(root)
    tsv = next(root.rglob("*anno.tsv"))
    scores = {}
    for line in Path(tsv).read_text().splitlines():
        vid, _cat, anno = line.split("\t")[:3]
        scores.setdefault(vid, []).append(np.fromstring(anno, sep=",", dtype=np.float32))
    scores = {k: np.mean(v, axis=0) for k, v in scores.items()}
    recs = [{"video_id": k, "path": vids[k], "split": "eval", "captions": []}
            for k in sorted(scores) if k in vids]
    return recs, scores


def summe(root=None):
    """Records + {video_id: (n_frames,) mean human importance from GT/*.mat}."""
    from scipy.io import loadmat
    root = Path(root or DATA / "summe")
    vids = _find_videos(root)
    scores = {}
    for m in sorted(root.rglob("*.mat")):
        mat = loadmat(m)
        if "gt_score" not in mat:
            continue
        scores[m.stem] = np.asarray(mat["gt_score"], np.float32).ravel()
    recs = [{"video_id": k, "path": vids[k], "split": "eval", "captions": []}
            for k in sorted(scores) if k in vids]
    return recs, scores


# --- ActivityNet Captions: long-video stress test -------------------------------

def activitynet(root=None):
    """Whatever subset of videos is actually present locally; captions from train/val JSONs."""
    root = Path(root or DATA / "activitynet")
    vids = _find_videos(root)
    caps = {}
    for js in sorted(root.rglob("*.json")):
        d = json.loads(Path(js).read_text())
        for vid, v in d.items():
            if isinstance(v, dict) and "sentences" in v:
                caps.setdefault(vid.removeprefix("v_"), []).extend(s.strip() for s in v["sentences"])
    recs = []
    for stem, path in sorted(vids.items()):
        vid = stem.removeprefix("v_")
        if vid in caps:
            recs.append({"video_id": vid, "path": path, "split": "eval", "captions": caps[vid]})
    return recs


# --- Personal holdout: usability gate only, never scored, never trained on ------

def holdout(root=None):
    root = Path(root or DATA / "holdout")
    return [{"video_id": k, "path": v, "split": "holdout", "captions": []}
            for k, v in sorted(_find_videos(root).items())]


LOADERS = {"msrvtt": msrvtt, "msrvtt_qa": msrvtt_qa, "tvsum": lambda r=None: tvsum(r)[0],
           "summe": lambda r=None: summe(r)[0], "activitynet": activitynet, "holdout": holdout}


def verify_no_leakage(records):
    """Exit criterion: nothing in an eval/holdout split also appears as training data."""
    by_split = {}
    for r in records:
        by_split.setdefault(r["split"], set()).add(r["video_id"])
    train = by_split.get("train", set())
    for s in ("val", "test", "eval", "holdout"):
        overlap = train & by_split.get(s, set())
        assert not overlap, f"leakage: {len(overlap)} ids in both train and {s}: {sorted(overlap)[:5]}"
    return {k: len(v) for k, v in sorted(by_split.items())}

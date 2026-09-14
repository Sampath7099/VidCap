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

def msrvtt(root=None):
    """Official 6513/497/2990 split, read from the annotation JSONs (not re-derived)."""
    root = Path(root or DATA / "msrvtt")
    vids = _find_videos(root)
    caps, splits = {}, {}
    for js in sorted(root.rglob("*videodatainfo*.json")):
        d = json.loads(Path(js).read_text())
        for v in d.get("videos", []):
            splits[v["video_id"]] = {"validate": "val"}.get(v.get("split"), v.get("split", "test"))
        for s in d.get("sentences", []):
            caps.setdefault(s["video_id"], []).append(s["caption"])
    return [{"video_id": k, "path": vids[k], "split": splits.get(k, "test"), "captions": caps.get(k, [])}
            for k in sorted(caps) if k in vids]


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


LOADERS = {"msrvtt": msrvtt, "tvsum": lambda r=None: tvsum(r)[0],
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

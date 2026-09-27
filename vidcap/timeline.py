"""Long video -> scenes -> timeline of captioned events. Pure logic; no models live here."""
import numpy as np

from .encoder import normalize
from .metrics import _norm_answer


def scene_changes(emb):
    """(N, D) frame embeddings -> (N-1,) cosine distance from each frame to the next.

    Raw, not mean-centred: on TVSum, centring cut the p99/median peak ratio from 7.8x to 5.2x.
    """
    e = normalize(emb)
    return 1.0 - (e[1:] * e[:-1]).sum(1)


def segment(emb, times, min_len=3.0, max_len=15.0, z=2.0, ratio=3.0):
    """-> [(start, end)] frame ranges, end exclusive, covering every frame in order.

    A cut needs a change z std above the video's mean AND ratio x its median — z alone always
    fires somewhere, even on a static shot. Scenes stay >= min_len seconds, then any longer
    than max_len (the clip length the captioner was trained on) are split evenly.
    """
    n = len(emb)
    if n == 0:
        return []
    bounds = [0]
    if n > 2:
        d = scene_changes(emb)
        for c in np.flatnonzero(d > max(d.mean() + z * d.std(), ratio * np.median(d))) + 1:
            if times[c] - times[bounds[-1]] >= min_len and times[-1] - times[c] >= min_len:
                bounds.append(int(c))
    bounds.append(n)

    spans = []
    for s, e in zip(bounds[:-1], bounds[1:]):
        pieces = int(np.ceil((times[e - 1] - times[s]) / max_len)) or 1
        cuts = np.linspace(s, e, pieces + 1).round().astype(int)
        spans += [(int(a), int(b)) for a, b in zip(cuts[:-1], cuts[1:]) if b > a]
    return spans


def merge_repeats(events):
    """Adjacent events with the same caption become one longer event — a static shot split by
    max_len, or a false cut, should read as one thing that happened, not the same line twice."""
    out = []
    for ev in events:
        if out and _norm_answer(out[-1]["caption"]) == _norm_answer(ev["caption"]):
            out[-1] = {**out[-1], "end": ev["end"], "frames": out[-1]["frames"] + ev["frames"],
                       "range": [out[-1]["range"][0], ev["range"][1]]}
        else:
            out.append(dict(ev))
    return out


_STOP = {"a", "an", "the", "is", "are", "in", "on", "at", "of", "and", "to", "it", "with", "by"}


def _words(s):
    return set(_norm_answer(s).split()) - _STOP


def source_event(text, events):
    """The event whose caption shares the largest share of its words with text, or None if
    none share any. Pins an LLM answer to a time without trusting the LLM to copy one."""
    w = _words(text)
    best = max(events, key=lambda e: len(w & _words(e["caption"])) / (len(_words(e["caption"])) or 1),
               default=None)
    return best if best and w & _words(best["caption"]) else None


def fmt_time(sec):
    sec = int(round(sec))
    return f"{sec // 60}:{sec % 60:02d}"


def timeline_text(events):
    return "\n".join(f"[{fmt_time(e['start'])}-{fmt_time(e['end'])}] {e['caption']}" for e in events)

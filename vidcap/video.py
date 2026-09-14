"""Candidate-frame pool extraction: video file -> ~POOL_FPS RGB frames + timestamps."""
import cv2
import numpy as np

from .config import DEFAULT_POOL_FPS, FRAME_SIZE, MAX_POOL_FRAMES, MIN_POOL_FRAMES, POOL_FPS


def _resize_short(img, short=FRAME_SIZE):
    h, w = img.shape[:2]
    s = short / min(h, w)
    return img if s >= 1.0 else cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


def sample_frames(path, fps=DEFAULT_POOL_FPS, max_frames=MAX_POOL_FRAMES,
                  min_frames=MIN_POOL_FRAMES):
    """Return (frames[N,H,W,3] uint8 RGB, times[N] seconds). Empty arrays if undecodable."""
    cap = cv2.VideoCapture(str(path))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or src_fps != src_fps or src_fps <= 0:  # missing / NaN metadata
        src_fps = 30.0
    step = max(1, round(src_fps / fps))

    # Short clips: densify so the pool still exceeds the largest frame budget.
    n_total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if n_total and n_total > 0 and n_total / step < min_frames:
        step = max(1, int(n_total // min_frames))

    frames, times, idx = [], [], 0
    while True:
        if not cap.grab():  # grab() skips decode for frames we don't keep
            break
        if idx % step == 0:
            ok, bgr = cap.retrieve()
            if ok:
                frames.append(_resize_short(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
                times.append(idx / src_fps)
        idx += 1
    cap.release()

    if not frames:
        return np.zeros((0, FRAME_SIZE, FRAME_SIZE, 3), np.uint8), np.zeros(0, np.float32)
    if len(frames) > max_frames:  # uniform subsample, keeps the tail of long videos
        keep = np.linspace(0, len(frames) - 1, max_frames).round().astype(int)
        frames = [frames[i] for i in keep]
        times = [times[i] for i in keep]
    return np.stack(frames), np.asarray(times, np.float32)

"""Human-importance validation gates (scripts/validate_scorer.py): python test_validate.py"""
import os
import pathlib
import tempfile

import numpy as np

TMP = tempfile.mkdtemp(prefix="vidcap_validate_")
os.environ["VIDCAP_DATA"], os.environ["VIDCAP_OUT"] = f"{TMP}/data", f"{TMP}/out"

import cv2  # noqa: E402
from scripts.validate_scorer import frame_indices, motion_scores, rho  # noqa: E402

FPS, SECONDS = 30, 4


def make_video(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (64, 64))
    for i in range(SECONDS * FPS):
        f = np.zeros((64, 64, 3), np.uint8)
        f[20:36, (i % 48):(i % 48) + 16] = 255
        w.write(f)
    w.release()


def test_timestamps_map_to_the_right_frames():
    """The whole validation hinges on this mapping. TVSum and SumMe annotations are per-frame,
    so a pool timestamp of t seconds must land on score index round(t * fps) — an off-by-fps
    error here would silently correlate the scorer against the wrong frames and read as a null.
    """
    p = pathlib.Path(TMP) / "v.mp4"
    make_video(p)
    n_scores = SECONDS * FPS
    times = np.array([0.0, 1.0, 2.5, 3.9])
    idx = frame_indices(p, times, n_scores)
    assert list(idx) == [0, 30, 75, 117], idx
    print(f"time->frame mapping ok ({times.tolist()} -> {idx.tolist()} at {FPS}fps)")


def test_mapping_stays_in_bounds():
    """Decoder frame count and annotation length disagree by a frame or two in the real files;
    that must clip, not raise IndexError mid-run."""
    p = pathlib.Path(TMP) / "v.mp4"
    n_scores = 50
    idx = frame_indices(p, np.array([0.0, 3.9, 99.0]), n_scores)
    assert idx.max() == n_scores - 1 and idx.min() == 0, idx
    human = np.arange(n_scores)[idx]          # must not raise
    assert len(human) == 3
    print(f"out-of-range timestamps clip ok (max index {idx.max()} for {n_scores} scores)")


def test_rho_is_signed_and_calibrated():
    """rho must be +1 on a perfect ranking and -1 reversed. If it came back unsigned, a scorer
    that ranked frames exactly backwards would be reported as a success."""
    h = np.array([1.0, 5.0, 2.0, 9.0, 3.0])
    assert abs(rho(h, h) - 1.0) < 1e-6, rho(h, h)
    assert abs(rho(-h, h) + 1.0) < 1e-6, rho(-h, h)
    assert abs(rho(np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 6.0])) - 1.0) < 1e-6
    print(f"rho calibrated ok (identical {rho(h, h):+.1f}, reversed {rho(-h, h):+.1f})")


def test_motion_score_tracks_change():
    """The motion null must actually respond to change, or 'beats motion' is a vacuous claim."""
    emb = np.repeat(np.eye(4, 8, dtype=np.float32), 2, axis=0)   # changes on odd indices
    m = motion_scores(emb)
    assert m[0] == 0.0
    assert m[2] > m[1] and m[4] > m[3], m
    print(f"motion score ok (flat {m[1]:.2f} vs changed {m[2]:.2f})")


if __name__ == "__main__":
    test_timestamps_map_to_the_right_frames()
    test_mapping_stays_in_bounds()
    test_rho_is_signed_and_calibrated()
    test_motion_score_tracks_change()
    print(f"\nvalidation gates passed ({TMP})")

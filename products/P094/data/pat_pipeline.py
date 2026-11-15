"""Six-stage PAT (pointing, acquisition, tracking) frame pipeline.

This module is the SINGLE definition of the pipeline whose timing P094
compares between simulation mode and hardware mode. The identical file is
executed on the Jetson Orin Nano by the campaign and in the simulation host
by the product, so that a per-stage timing comparison is a comparison of the
same instructions and not of two implementations.

Pure numpy by necessity and by choice: the board carries numpy 2.2.6 with a
scipy 1.8.0 that cannot import against it, and no OpenCV or scikit-learn.
Pure numpy is therefore the only dependency set that can run unmodified in
both modes, which is exactly the property the comparison requires.

Every stage declares an operation count in DECLARED_OPS. That declaration is
the simulation's cost model and it is the object P094 refutes.
"""

from __future__ import annotations

import numpy as np

STAGES = ("ingest", "background", "threshold", "centroid", "gate", "track")

# Declared per-stage operation counts as a function of frame width W and the
# number of candidate regions K. These are the counts a schedulability
# analysis would derive by reading the source, which is what makes them the
# right thing to refute: they are not wrong arithmetic, they are the correct
# operation count used as a timing proxy.
def declared_ops(w: int, k: int) -> dict:
    n = w * w
    return {
        "ingest": 2.0 * n,          # cast + reshape touch every pixel twice
        "background": 4.0 * n,      # running mean update and subtract
        "threshold": 3.0 * n,       # compare, accumulate, index
        "centroid": 6.0 * float(k) * 25.0,   # 5x5 window per candidate
        "gate": 12.0 * float(k),    # fixed per-candidate predicate set
        "track": 180.0,             # fixed-size 6-state update, K-independent
    }


class RunningBackground:
    """Exponential running background with a fixed-point-free update."""

    def __init__(self, w: int, alpha: float = 0.05) -> None:
        self.alpha = float(alpha)
        self.bg = np.zeros((w, w), dtype=np.float32)
        self._primed = False

    def update(self, frame: np.ndarray) -> np.ndarray:
        if not self._primed:
            self.bg[:] = frame
            self._primed = True
        else:
            self.bg *= (1.0 - self.alpha)
            self.bg += self.alpha * frame
        return frame - self.bg


class TrackState:
    """Six-state constant-acceleration track with a fixed gain.

    Fixed gain rather than a full covariance propagation, so that the stage's
    cost is genuinely independent of the data: a stage whose declared cost is
    K-independent must actually be K-independent or the comparison measures
    the wrong thing.
    """

    def __init__(self) -> None:
        self.x = np.zeros(6, dtype=np.float64)
        self.gain = np.array([0.45, 0.45, 0.18, 0.18, 0.04, 0.04], dtype=np.float64)
        self.f = np.eye(6, dtype=np.float64)
        dt = 1.0 / 60.0
        self.f[0, 2] = dt
        self.f[1, 3] = dt
        self.f[2, 4] = dt
        self.f[3, 5] = dt
        self.f[0, 4] = 0.5 * dt * dt
        self.f[1, 5] = 0.5 * dt * dt

    def update(self, meas_xy, valid: bool) -> np.ndarray:
        self.x = self.f @ self.x
        if valid:
            resid = np.zeros(6, dtype=np.float64)
            resid[0] = meas_xy[0] - self.x[0]
            resid[1] = meas_xy[1] - self.x[1]
            resid[2] = resid[0] * 60.0 - self.x[2]
            resid[3] = resid[1] * 60.0 - self.x[3]
            self.x = self.x + self.gain * resid
        return self.x


# Gate thresholds. Deliberately set so that every branch is reachable; P093
# shipped a gate two of whose five reject reasons never fired in 268,425
# inferences, and that is recorded as a weak result. Reachability here is
# asserted by the campaign, not assumed.
GATE_MIN_SCORE = 1.35
GATE_MIN_AREA = 2
GATE_MAX_AREA = 160
GATE_BORDER = 2

REJECT_REASONS = ("no_candidate", "low_score", "area_out", "border")


def extract_candidates(resid: np.ndarray, thresh: float):
    """Threshold and group bright pixels into candidate regions.

    Single-pass numpy grouping by quantised position rather than a true
    connected-component labelling: the point of the stage is a realistic
    per-pixel cost with a K-dependent tail, and a scipy label() is not
    available on the board.
    """
    ys, xs = np.nonzero(resid > thresh)
    if ys.size == 0:
        return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.int64)
    key = (ys // 6).astype(np.int64) * 10000 + (xs // 6).astype(np.int64)
    order = np.argsort(key, kind="stable")
    key_s = key[order]
    uniq, starts, counts = np.unique(key_s, return_index=True, return_counts=True)
    ys_s = ys[order]
    xs_s = xs[order]
    reps = np.empty((uniq.size, 2), dtype=np.int64)
    for i in range(uniq.size):
        a = starts[i]
        b = a + counts[i]
        reps[i, 0] = ys_s[a:b].mean()
        reps[i, 1] = xs_s[a:b].mean()
    return reps, counts.astype(np.int64)


def centroid_of(frame: np.ndarray, cy: int, cx: int, half: int = 2):
    h, w = frame.shape
    y0 = max(0, cy - half)
    y1 = min(h, cy + half + 1)
    x0 = max(0, cx - half)
    x1 = min(w, cx + half + 1)
    win = frame[y0:y1, x0:x1].astype(np.float64)
    tot = win.sum()
    if tot <= 0.0:
        return float(cy), float(cx), 0.0
    yy = np.arange(y0, y1, dtype=np.float64)[:, None]
    xx = np.arange(x0, x1, dtype=np.float64)[None, :]
    gy = float((win * yy).sum() / tot)
    gx = float((win * xx).sum() / tot)
    peak = float(win.max())
    med = float(np.median(win)) if win.size else 0.0
    score = peak / (abs(med) + 1e-6)
    return gy, gx, score

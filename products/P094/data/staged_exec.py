"""Per-stage timed execution of the PAT pipeline.

This is the instrument. The same function body runs in hardware mode on the
Jetson and in simulation mode on the host, so a per-stage difference is a
property of the platform and not of the harness.

Timing uses time.perf_counter_ns, which on the board resolves to
clock_gettime(CLOCK_MONOTONIC) at 1 ns resolution (measured, recorded in the
campaign metadata). Each stage is bracketed individually. The bracketing cost
itself is measured by a null stage and reported, because a six-stage bracket
on a sub-millisecond stage is not free and a timing product that does not
measure its own instrument is not measuring anything.
"""

from __future__ import annotations

import time

import numpy as np

from pat_pipeline import (
    GATE_BORDER,
    GATE_MAX_AREA,
    GATE_MIN_AREA,
    GATE_MIN_SCORE,
    STAGES,
    RunningBackground,
    TrackState,
    centroid_of,
    extract_candidates,
)

_NS = time.perf_counter_ns


def bracket_overhead_ns(reps: int = 2000) -> float:
    """Measured cost of one empty start/stop bracket pair, in ns."""
    acc = 0
    for _ in range(reps):
        a = _NS()
        b = _NS()
        acc += b - a
    return acc / float(reps)


def make_frames(w: int, n_frames: int, n_targets: int, seed: int) -> np.ndarray:
    """Deterministic synthetic frame stack: drifting gaussian targets + noise.

    Generated once before timing starts so that frame synthesis is never
    inside a timed region.
    """
    rng = np.random.default_rng(seed)
    frames = rng.normal(100.0, 6.0, size=(n_frames, w, w)).astype(np.float32)
    yy = np.arange(w, dtype=np.float32)[:, None]
    xx = np.arange(w, dtype=np.float32)[None, :]
    for t in range(n_targets):
        y0 = rng.uniform(6, w - 6)
        x0 = rng.uniform(6, w - 6)
        vy = rng.uniform(-0.25, 0.25)
        vx = rng.uniform(-0.25, 0.25)
        amp = rng.uniform(55.0, 130.0)
        sig = rng.uniform(1.1, 1.9)
        for i in range(n_frames):
            cy = y0 + vy * i
            cx = x0 + vx * i
            if not (0 <= cy < w and 0 <= cx < w):
                continue
            g = amp * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sig * sig)))
            frames[i] += g.astype(np.float32)
    return frames


def run_timed(frames: np.ndarray, thresh_sigma: float = 3.0):
    """Execute the pipeline over a frame stack, timing all six stages per frame.

    Returns (per_frame_ns, census) where per_frame_ns is an (n_frames, 6)
    int64 array in stage order and census counts gate outcomes.
    """
    n_frames, w, _ = frames.shape
    bg = RunningBackground(w)
    track = TrackState()
    out = np.zeros((n_frames, len(STAGES)), dtype=np.int64)
    census = {"accept": 0, "no_candidate": 0, "low_score": 0, "area_out": 0, "border": 0}
    k_per_frame = np.zeros(n_frames, dtype=np.int64)

    for i in range(n_frames):
        raw = frames[i]

        # --- stage 0: ingest -------------------------------------------------
        t0 = _NS()
        frame = np.asarray(raw, dtype=np.float32)
        frame = frame.reshape(w, w)
        acc_ingest = float(frame[0, 0]) * 0.0
        t1 = _NS()

        # --- stage 1: background --------------------------------------------
        resid = bg.update(frame)
        t2 = _NS()

        # --- stage 2: threshold ---------------------------------------------
        sd = float(resid.std())
        thresh = thresh_sigma * sd
        reps, areas = extract_candidates(resid, thresh)
        t3 = _NS()

        # --- stage 3: centroid ----------------------------------------------
        cents = []
        for j in range(reps.shape[0]):
            cents.append(centroid_of(frame, int(reps[j, 0]), int(reps[j, 1])))
        t4 = _NS()

        # --- stage 4: gate ---------------------------------------------------
        best = None
        if not cents:
            census["no_candidate"] += 1
        else:
            for j, (gy, gx, score) in enumerate(cents):
                a = int(areas[j])
                if score < GATE_MIN_SCORE:
                    census["low_score"] += 1
                    continue
                if a < GATE_MIN_AREA or a > GATE_MAX_AREA:
                    census["area_out"] += 1
                    continue
                if (gy < GATE_BORDER or gy > w - 1 - GATE_BORDER
                        or gx < GATE_BORDER or gx > w - 1 - GATE_BORDER):
                    census["border"] += 1
                    continue
                if best is None or score > best[2]:
                    best = (gy, gx, score)
            if best is not None:
                census["accept"] += 1
        t5 = _NS()

        # --- stage 5: track --------------------------------------------------
        track.update((best[0], best[1]) if best is not None else (0.0, 0.0), best is not None)
        t6 = _NS()

        out[i, 0] = t1 - t0
        out[i, 1] = t2 - t1
        out[i, 2] = t3 - t2
        out[i, 3] = t4 - t3
        out[i, 4] = t5 - t4
        out[i, 5] = t6 - t5
        k_per_frame[i] = reps.shape[0]
        del acc_ingest

    return out, census, k_per_frame

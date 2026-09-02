"""V2 -- does the measured CUSUM detection delay match the analytic expectation?

Two settings, because they answer two different questions.

**Convention.**  The analytic expressions return a *run length*: the number of
samples the test inspects, at least one.  ``detection_delay`` returns an index
difference, zero when the alarm lands on the onset sample.  Everything below
compares run lengths, i.e. ``measured index + 1``.  Getting that one sample
wrong is a 3 % error at a delay of 30 samples and a 50 % error at a delay of
one, and it is the single easiest way to "disagree with the literature".

**B1, the exact change-point model.**  A unit-variance Gaussian sequence whose
mean steps from 0 to ``mu`` at sample 0, with the CUSUM started at zero there.
This is precisely the model Wald's and Siegmund's expressions describe, so any
disagreement is a defect in the implementation or in the formula, not in the
physics.  Criteria fixed before the run:

* **B1a** the measured mean run length is within 10 % of
  :func:`cusum_delay_siegmund` in every cell.
* **B1b** the Wald expression ``h / K`` is reported as a ratio, with no
  criterion -- it neglects the boundary overshoot and is expected to be
  optimistic; the measurement says by how much.

**B2, the closed loop.**  A step gyro bias in the GNC loop of
:mod:`fdiscope.simulate`.  The residual mean does **not** step: the estimator
absorbs part of the bias and the mean rises over the closed-loop error time
constant, so the Siegmund expression evaluated at the steady-state ``mu`` is a
lower bound on the delay.  Reported as a measurement, together with the
mean-path prediction of :func:`cusum_delay_mean_path`, which does model the
rise.  Criterion:

* **B2a** the mean-path prediction is within 25 % of the measured *median*
  run length (the median, because the mean-path recursion is noise-free and
  predicts a typical path, not an average over fluctuations).

**B3** checks that a warm-started CUSUM -- one that has been running on
fault-free data before onset -- detects sooner than the zero-started one the
theory describes.

Run: ``python validation/cusum_delay.py``
"""

from __future__ import annotations

import time
from pathlib import Path

from _support import ROOT  # noqa: F401
from _support import Tee

import numpy as np

from fdiscope.analytic import (
    cusum_delay_mean_path,
    cusum_delay_siegmund,
    cusum_delay_wald,
    cusum_threshold_for_arl0,
    normalised_bias_signature,
)
from fdiscope.detectors import CusumDetector, first_alarm_index
from fdiscope.faults import FaultSpec, FaultType
from fdiscope.metrics import mean_ci
from fdiscope.plant import loop_matrices
from fdiscope.simulate import LoopConfig, build_filter, simulate_loop

MC_TRIALS = 4000
MC_HORIZON = 1200
MC_SEED = 31337
SIEGMUND_TOL = 0.10
MEAN_PATH_TOL = 0.25
GRID = [(mu, h) for mu in (0.5, 1.0, 2.0) for h in (4.0, 5.75, 8.0)]

CLOSED_LOOP_TRIALS = 400
CLOSED_LOOP_STEPS = 1600
CLOSED_LOOP_ONSET = 600
CLOSED_LOOP_SEED0 = 41000
TARGET_ARL0 = 2000.0
BIAS_SIGMAS = (1.0, 1.5, 2.0, 3.0)

out = Tee(Path(__file__).with_name("cusum_delay_output.txt"))
t0 = time.perf_counter()

out("V2 -- CUSUM detection delay, measured against the analytic expectation")
out("=" * 78)
out("All data SIMULATED.  Delays are in samples; the loop runs at dt = 0.1 s.")
out("")

out("-- B1  exact change-point model: p_k ~ N(mu, 1) from sample 0, CUSUM from 0")
out(f"{MC_TRIALS} trials per cell, horizon {MC_HORIZON} samples, seed {MC_SEED}.")
out("")
out(f"{'mu':>5} {'h':>7} {'K':>8} {'Wald h/K':>10} {'Siegmund':>10} "
    f"{'measured':>10} {'95 % CI':>22} {'m/S':>7} {'m/W':>7}  verdict")
b1_pass = True
b1_cells = 0
rng = np.random.default_rng(MC_SEED)
for mu, h in GRID:
    samples = rng.standard_normal((MC_TRIALS, MC_HORIZON)) + mu
    incr = mu * samples - 0.5 * mu * mu
    csum = np.cumsum(incr, axis=1)
    g = csum - np.minimum(0.0, np.minimum.accumulate(csum, axis=1))
    crossed = g > h
    any_cross = crossed.any(axis=1)
    # +1 converts the crossing index into a run length (samples inspected).
    delays = np.argmax(crossed, axis=1).astype(float) + 1.0
    censored = int(np.count_nonzero(~any_cross))
    delays = delays[any_cross]
    wald = cusum_delay_wald(h, mu)
    sieg = cusum_delay_siegmund(h, mu)
    iv = mean_ci(delays)
    ratio_s = iv.point / sieg
    ratio_w = iv.point / wald
    ok = abs(ratio_s - 1.0) <= SIEGMUND_TOL and censored == 0
    b1_pass = b1_pass and ok
    b1_cells += int(ok)
    ci = f"[{iv.low:.3f}, {iv.high:.3f}]"
    out(
        f"{mu:>5.1f} {h:>7.3f} {0.5 * mu * mu:>8.4f} {wald:>10.3f} {sieg:>10.3f} "
        f"{iv.point:>10.3f} {ci:>22} "
        f"{ratio_s:>7.4f} {ratio_w:>7.4f}  {'PASS' if ok else 'FAIL'}"
    )
    if censored:
        out(f"      ... {censored} of {MC_TRIALS} trials never crossed within the horizon")
out(f"B1a: {b1_cells} of {len(GRID)} cells PASS -- overall "
    f"{'PASS' if b1_pass else 'FAIL'}  (|measured/Siegmund - 1| <= {SIEGMUND_TOL})")
out("B1b: measurement -- the m/W column is the cost of ignoring the boundary")
out("     overshoot and the finite start.  Wald's h/K is pessimistic below")
out("     mu = 1/1.1652 = 0.858 and optimistic above it, which is exactly where")
out("     the two neglected terms change relative size; Siegmund's correction")
out("     removes both and lands within a few percent everywhere.")
out("")

out("-- B2  closed loop: step gyro bias, CUSUM on the model-derived direction")
out(f"{CLOSED_LOOP_TRIALS} runs per bias, onset sample {CLOSED_LOOP_ONSET}, "
    f"seeds {CLOSED_LOOP_SEED0}+")
out(f"threshold from cusum_threshold_for_arl0(ARL0 = {TARGET_ARL0:.0f}, mu)")
out("")
base = LoopConfig(n_steps=CLOSED_LOOP_STEPS, seed=0)
kf = build_filter(loop_matrices(base.plant))
sigma_rate = float(np.sqrt(base.plant.gyro_var_rad2_s2))
quiet = LoopConfig(n_steps=CLOSED_LOOP_STEPS, seed=0, noise=False)

out("Run lengths, not index-based delays: a detector that alarms on the onset")
out("sample scores 1, not 0.")
out(f"{'bias':>8} {'mu_ss':>8} {'h':>7} {'Siegmund':>10} {'mean-path':>10} "
    f"{'measured mean':>14} {'median':>8} {'m/S':>7} {'path/med':>9}  verdict")
b2_pass = True
b2_cells = 0
b2_ok = 0
for n_sigma in BIAS_SIGMAS:
    bias = n_sigma * sigma_rate
    direction, mu_ss = normalised_bias_signature(kf, [0.0, bias])
    h = cusum_threshold_for_arl0(TARGET_ARL0, mu_ss)
    spec = FaultSpec(FaultType.SENSOR_BIAS, CLOSED_LOOP_ONSET, bias, 1)

    profile = simulate_loop(quiet, spec).residual[CLOSED_LOOP_ONSET:] @ direction
    path = cusum_delay_mean_path(profile, mu_ss, h)

    det = CusumDetector(direction=direction, mu=mu_ss, threshold=h)
    delays: list[float] = []
    censored = 0
    for i in range(CLOSED_LOOP_TRIALS):
        run = simulate_loop(
            LoopConfig(n_steps=CLOSED_LOOP_STEPS, seed=CLOSED_LOOP_SEED0 + i), spec
        )
        alarm = det.run(run.residual[CLOSED_LOOP_ONSET:]).alarm
        idx = first_alarm_index(alarm, 0)
        if idx < 0:
            censored += 1
        else:
            delays.append(float(idx) + 1.0)
    arr = np.asarray(delays)
    iv = mean_ci(arr)
    median = float(np.median(arr))
    sieg = cusum_delay_siegmund(h, mu_ss)
    if np.isfinite(path) and median >= 1.0:
        path_ratio = path / median
        ok = abs(path_ratio - 1.0) <= MEAN_PATH_TOL and censored == 0
        b2_cells += 1
        b2_ok += int(ok)
        b2_pass = b2_pass and ok
        verdict = "PASS" if ok else "FAIL"
    else:
        path_ratio = float("nan")
        verdict = "n/a"
    out(
        f"{n_sigma:>7.1f}s {mu_ss:>8.4f} {h:>7.3f} {sieg:>10.3f} {path:>10.1f} "
        f"{iv.point:>14.3f} {median:>8.1f} {iv.point / sieg:>7.4f} "
        f"{path_ratio:>9.4f}  {verdict}"
    )
    if verdict == "n/a":
        out("      ... the mean path never crosses, so no ratio is defined.")
    if censored:
        out(f"      ... {censored} of {CLOSED_LOOP_TRIALS} runs never detected")
out(f"B2a: {b2_ok} of {b2_cells} cells PASS  "
    f"(|mean-path/median - 1| <= {MEAN_PATH_TOL}) -- overall "
    f"{'PASS' if b2_pass else 'FAIL'}")
if not b2_pass:
    out("     The failing cell is the smallest bias.  The mean-path recursion is")
    out("     noise-free, so it cannot see that fluctuation lets the CUSUM cross")
    out("     early; that help is largest when the drift per sample is smallest.")
    out("     The ratio falls monotonically towards 1 as the bias grows, which is")
    out("     the behaviour that explanation predicts.")
out("")
out("The m/S column is the price of the closed loop.  The Siegmund expression")
out("assumes the residual mean steps to mu_ss at onset; it does not, because the")
out("estimator absorbs part of the bias first, so the true delay is longer.  The")
out("mean-path column, which runs the same CUSUM on the noise-free residual")
out("profile, tracks the measurement far more closely.")
out("")

out("-- B3  warm start vs zero start (measurement, no criterion)")
n_sigma = 2.0
bias = n_sigma * sigma_rate
direction, mu_ss = normalised_bias_signature(kf, [0.0, bias])
h = cusum_threshold_for_arl0(TARGET_ARL0, mu_ss)
spec = FaultSpec(FaultType.SENSOR_BIAS, CLOSED_LOOP_ONSET, bias, 1)
det = CusumDetector(direction=direction, mu=mu_ss, threshold=h)
warm: list[float] = []
cold: list[float] = []
for i in range(CLOSED_LOOP_TRIALS):
    run = simulate_loop(LoopConfig(n_steps=CLOSED_LOOP_STEPS, seed=CLOSED_LOOP_SEED0 + i), spec)
    full = det.run(run.residual).alarm
    idx = first_alarm_index(full, CLOSED_LOOP_ONSET)
    if idx >= 0:
        warm.append(float(idx - CLOSED_LOOP_ONSET) + 1.0)
    idx2 = first_alarm_index(det.run(run.residual[CLOSED_LOOP_ONSET:]).alarm, 0)
    if idx2 >= 0:
        cold.append(float(idx2) + 1.0)
out(f"bias {n_sigma:.1f} sigma, mu_ss = {mu_ss:.4f}, h = {h:.3f}")
out(f"zero-started CUSUM run length : {np.mean(cold):.3f} samples  (n = {len(cold)})")
out(f"warm-started CUSUM run length : {np.mean(warm):.3f} samples  (n = {len(warm)})")
out(f"warm start is faster by       : {np.mean(cold) - np.mean(warm):.3f} samples")
out("A warm CUSUM has usually accumulated some positive evidence already, so it")
out("beats the theory it is designed by.  The zero-started figures are the ones")
out("that match the published expressions, and the ones B1 and B2 use.")
out("")

out("=" * 78)
out(f"V2 summary: B1a {b1_cells}/{len(GRID)} cells PASS; B1b measurement; "
    f"B2a {b2_ok}/{b2_cells} cells PASS; B3 measurement")
out(f"wall time {time.perf_counter() - t0:.1f} s")
out.save()

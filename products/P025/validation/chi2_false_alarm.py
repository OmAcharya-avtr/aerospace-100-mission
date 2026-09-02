"""V1 -- does the chi-squared test's measured false-alarm rate equal its design?

Criteria, fixed before the run:

* **A1** the normalised residual of a fault-free closed loop is ``N(0, I)``:
  the time-average NIS test accepts at 95 %, and every lag-1..3 sample
  autocorrelation is inside a 4-sigma band.  Four sigma, not the 5 % band the
  function returns, because six statistics are checked at once and a single
  2-sigma excursion among six is ordinary; the 5 % band is printed alongside
  so the reader can see both.
* **A2** for each design level ``alpha`` in ``{1e-1, 1e-2, 1e-3}`` and each
  window in ``{25, 100}``, the measured false-alarm rate over
  **non-overlapping** windows has a 95 % Wilson interval containing ``alpha``.
* **A3** the same measurement over **overlapping** windows is reported as a
  ratio to ``alpha``.  No criterion: overlapping windows are correlated, and
  the point of the check is to quantify how misleading the per-sample alarm
  fraction is if it is read as the design level.
* **A4** the measured CUSUM average run length under ``H0``, with reset on
  alarm, is within a factor of 1.5 of :func:`cusum_arl0_siegmund` for three
  thresholds.

Run: ``python validation/chi2_false_alarm.py``
"""

from __future__ import annotations

import time
from pathlib import Path

from _support import ROOT  # noqa: F401
from _support import Tee

import numpy as np

from fdiscope.analytic import chi2_false_alarm_rate, chi2_threshold, cusum_arl0_siegmund
from fdiscope.detectors import CusumDetector
from fdiscope.metrics import wilson_interval
from fdiscope.residuals import nis_consistency, whiteness
from fdiscope.simulate import LoopConfig, simulate_loop

N_RUNS = 60
N_STEPS = 6000
BURN_IN = 300
SEED0 = 20000
ALPHAS = (1.0e-1, 1.0e-2, 1.0e-3)
WINDOWS = (25, 100)
CUSUM_MU = 1.0
CUSUM_THRESHOLDS = (4.0, 5.75, 8.0)
ARL0_TOLERANCE = 1.5

out = Tee(Path(__file__).with_name("chi2_false_alarm_output.txt"))
t0 = time.perf_counter()

out("V1 -- chi-squared false-alarm rate, measured against design")
out("=" * 78)
out(f"{N_RUNS} fault-free closed-loop runs of {N_STEPS} samples, burn-in {BURN_IN},")
out(f"seeds {SEED0}..{SEED0 + N_RUNS - 1}.  All data SIMULATED.")
out("")

residuals = [
    simulate_loop(LoopConfig(n_steps=N_STEPS, seed=SEED0 + i)).residual[BURN_IN:]
    for i in range(N_RUNS)
]
stacked = np.concatenate(residuals, axis=0)

out("-- A1  is the normalised residual N(0, I) and white?")
out(f"samples                 : {stacked.shape[0]}")
out(f"channel means           : {np.round(stacked.mean(axis=0), 6)}   (expect 0)")
out(f"channel std             : {np.round(stacked.std(axis=0), 6)}   (expect 1)")
check = nis_consistency(stacked)
out(
    f"mean NIS                : {check.mean_nis:.6f}  expected {check.expected:.1f}, "
    f"95 % band [{check.low:.6f}, {check.high:.6f}]"
)
rho, bound = whiteness(stacked, 3)
limit = 4.0 / np.sqrt(stacked.shape[0])
for lag in range(3):
    out(f"lag-{lag + 1} autocorrelation   : {np.round(rho[lag], 6)}   "
        f"5 % band +/-{bound:.6f}, 4-sigma +/-{limit:.6f}")
a1_pass = check.consistent and bool(np.all(np.abs(rho) < limit))
out(f"A1: {'PASS' if a1_pass else 'FAIL'}")
out("")

out("-- A2  non-overlapping windows: measured false-alarm rate vs design alpha")
out(f"{'window':>7} {'alpha':>8} {'threshold':>11} {'measured':>11} "
    f"{'95 % Wilson interval':>26} {'n':>7}  verdict")
a2_pass = True
for window in WINDOWS:
    for alpha in ALPHAS:
        threshold = chi2_threshold(alpha, window * 2)
        hits = 0
        total = 0
        for r in residuals:
            usable = (r.shape[0] // window) * window
            nis = np.sum(r[:usable] * r[:usable], axis=1).reshape(-1, window).sum(axis=1)
            hits += int(np.count_nonzero(nis > threshold))
            total += int(nis.size)
        iv = wilson_interval(hits, total)
        ok = iv.contains(alpha)
        a2_pass = a2_pass and ok
        out(
            f"{window:>7} {alpha:>8.0e} {threshold:>11.4f} {iv.point:>11.6f} "
            f"{'[' + format(iv.low, '.6f') + ', ' + format(iv.high, '.6f') + ']':>26} "
            f"{total:>7}  {'PASS' if ok else 'FAIL'}"
        )
out(f"A2: {'PASS' if a2_pass else 'FAIL'}")
out("")

out("-- A3  overlapping windows: per-sample alarm fraction, as a ratio to alpha")
out("(measurement, no pass/fail: overlapping windows are correlated, so the")
out(" per-sample alarm fraction is NOT the per-test design level)")
out(f"{'window':>7} {'alpha':>8} {'per-sample rate':>17} {'ratio to alpha':>16}")
for window in WINDOWS:
    for alpha in ALPHAS:
        threshold = chi2_threshold(alpha, window * 2)
        hits = 0
        total = 0
        for r in residuals:
            nis = np.sum(r * r, axis=1)
            csum = np.concatenate(([0.0], np.cumsum(nis)))
            stat = csum[window:] - csum[:-window]
            hits += int(np.count_nonzero(stat > threshold))
            total += int(stat.size)
        rate = hits / total
        out(f"{window:>7} {alpha:>8.0e} {rate:>17.6f} {rate / alpha:>16.3f}")
out("")

out("-- A3b round trip: chi2_false_alarm_rate(chi2_threshold(alpha, dof), dof)")
worst = 0.0
for window in WINDOWS:
    for alpha in ALPHAS:
        dof = window * 2
        back = chi2_false_alarm_rate(chi2_threshold(alpha, dof), dof)
        worst = max(worst, abs(back - alpha) / alpha)
out(f"worst relative round-trip error : {worst:.3e}   (tolerance 1e-9)")
out(f"A3b: {'PASS' if worst < 1e-9 else 'FAIL'}")
out("")

out("-- A4  CUSUM average run length under H0, reset on alarm")
out(f"{'threshold':>10} {'analytic ARL0':>15} {'measured ARL0':>15} {'ratio':>8} "
    f"{'alarms':>8}  verdict")
a4_pass = True
for h in CUSUM_THRESHOLDS:
    det = CusumDetector(direction=[1.0, 0.0], mu=CUSUM_MU, threshold=h)
    alarms = 0
    samples = 0
    for r in residuals:
        result = det.run(r, reset_on_alarm=True)
        alarms += int(np.count_nonzero(result.alarm))
        samples += int(r.shape[0])
    measured = samples / alarms if alarms else float("inf")
    analytic = cusum_arl0_siegmund(h, CUSUM_MU)
    ratio = measured / analytic
    ok = (1.0 / ARL0_TOLERANCE) <= ratio <= ARL0_TOLERANCE
    a4_pass = a4_pass and ok
    out(
        f"{h:>10.4f} {analytic:>15.2f} {measured:>15.2f} {ratio:>8.4f} "
        f"{alarms:>8}  {'PASS' if ok else 'FAIL'}"
    )
out(f"A4: {'PASS' if a4_pass else 'FAIL'}  (tolerance: factor {ARL0_TOLERANCE})")
out("")

out("=" * 78)
out(f"V1 summary: A1 {'PASS' if a1_pass else 'FAIL'}, A2 {'PASS' if a2_pass else 'FAIL'}, "
    f"A3 measurement, A3b {'PASS' if worst < 1e-9 else 'FAIL'}, A4 {'PASS' if a4_pass else 'FAIL'}")
out(f"wall time {time.perf_counter() - t0:.1f} s")
out.save()

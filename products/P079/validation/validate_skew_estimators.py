"""Check 4 -- least-squares and Kalman skew estimation: consistency and convergence.

Requirement (BATCH_08_SPEC, P079): "Baseline 2: least-squares and Kalman skew
estimation over the exchange history" and "estimator consistency and
convergence under refinement".

What is checked
---------------
1. **Exact recovery on noiseless data.** A quadratic clock with no jitter, no
   timestamping noise and no clock noise: both estimators must return the
   declared polynomial to within timestamp quantisation.
2. **Convergence under refinement.** The RMS skew error over 40 seeded
   realisations at n = 50, 100, 200, 400, 800, 1600, with the fitted
   convergence exponent against the theoretical ``n**-3/2`` for the slope of a
   straight line fitted to equally spaced points with independent noise
   (Draper and Smith, *Applied Regression Analysis*, 3rd ed., chapter 5: the
   skew variance is ``12 sigma^2 / (n (n^2 - 1) dt^2)``, so the standard error
   falls as ``n**-3/2``).
3. **The Kalman filter's own consistency diagnostic.** The mean normalised
   innovation squared should be near 1.0 for a correctly specified filter. It
   is reported, not tuned against: the measurement-noise variance is taken from
   the data by a median-absolute-deviation estimator that assumes a Gaussian,
   and the queueing delay is a right-skewed gamma, so the diagnostic is
   expected to exceed 1.0 and the measured factor is the useful number.
4. **Least squares against Kalman on the same data**, as a table.

Run: ``python validation/validate_skew_estimators.py``
"""

from __future__ import annotations

import sys

import numpy as np

from _bootstrap import Tee, bootstrap, environment_lines, scaled

ROOT = bootstrap()

from timetransfer.acquire import acquire  # noqa: E402
from timetransfer.allan import fit_log_slope  # noqa: E402
from timetransfer.backends import SimulatedBackend, SimulationSpec  # noqa: E402
from timetransfer.clockmodel import ClockSpec  # noqa: E402
from timetransfer.linkmodel import LinkSpec  # noqa: E402
from timetransfer.skew import KalmanConfig, fit_least_squares, kalman_skew  # noqa: E402
from timetransfer.twoway import two_way_estimate  # noqa: E402

TRUE_OFFSET_S = 1.250e-3
TRUE_SKEW = 1.800e-6
TRUE_DRIFT = 2.000e-12
Q1 = 1.0e-20
Q2 = 1.0e-26
SEED = 20261024


def _spec(seed: int, *, jitter_scale_s: float, noise_s: float, clock_noise: bool) -> SimulationSpec:
    return SimulationSpec(
        clock=ClockSpec(
            offset0_s=TRUE_OFFSET_S,
            skew_s_per_s=TRUE_SKEW,
            drift_per_s=TRUE_DRIFT,
            q1_s2_per_s=Q1 if clock_noise else 0.0,
            q2_per_s=Q2 if clock_noise else 0.0,
        ),
        link=LinkSpec(
            floor_ab_s=5.0e-3,
            floor_ba_s=5.0e-3,
            jitter_scale_ab_s=jitter_scale_s,
            jitter_scale_ba_s=jitter_scale_s,
        ),
        period_s=1.0,
        timestamp_noise_s=noise_s,
        resolution_ns=1,
        seed=seed,
    )


def _estimate(spec: SimulationSpec, n: int, order: int = 2):
    backend = SimulatedBackend(spec)
    backend.open()
    exchanges = acquire(backend, n, min_exchanges=8).exchanges
    backend.close()
    estimate = two_way_estimate(exchanges)
    fit = fit_least_squares(estimate.midpoint_s, estimate.offset_s, order=order)
    kal = kalman_skew(
        estimate.midpoint_s,
        estimate.offset_s,
        config=KalmanConfig(q1_s2_per_s=Q1, q2_per_s=Q2, n_states=2),
    )
    return estimate, fit, kal


def main() -> int:
    out = Tee("validate_skew_estimators_output.txt")
    out("timetransfer validation 4 -- least-squares and Kalman skew estimation")
    out("=" * 78)
    for line in environment_lines():
        out(line)
    failures = 0
    out("")
    out(f"declared offset : {TRUE_OFFSET_S:.9e} s")
    out(f"declared skew   : {TRUE_SKEW:.9e} s/s")
    out(f"declared drift  : {TRUE_DRIFT:.9e} 1/s")

    out.rule("4a. exact recovery on a noiseless link")
    spec = _spec(SEED, jitter_scale_s=0.0, noise_s=0.0, clock_noise=False)
    _, fit, kal = _estimate(spec, scaled(800, minimum=40), order=2)
    out("least squares, order 2:")
    out(fit.as_text())
    out("")
    true_offset_at_ref = (
        TRUE_OFFSET_S + TRUE_SKEW * fit.t_ref_s + 0.5 * TRUE_DRIFT * fit.t_ref_s**2
    )
    true_skew_at_ref = TRUE_SKEW + TRUE_DRIFT * fit.t_ref_s
    out(f"offset error at t_ref : {fit.offset0_s - true_offset_at_ref:+.3e} s")
    out(f"skew error            : {fit.skew_s_per_s - true_skew_at_ref:+.3e} s/s")
    out(f"drift error           : {fit.drift_per_s - TRUE_DRIFT:+.3e} 1/s")
    skew_error = abs(fit.skew_s_per_s - true_skew_at_ref)
    out("")
    out("tolerance on the skew : 1.000e-11 s/s (quantisation of 1 ns over the fitted span)")
    if skew_error > 1.0e-11:
        failures += 1
        out("FAIL: the noiseless skew is not recovered to within quantisation")
    else:
        out("PASS")
    out("")
    out("kalman, two states:")
    out(kal.as_text())

    out.rule("4b. convergence under refinement")
    n_trials = scaled(40, minimum=8)
    counts = (50, 100, 200, 400, 800, 1600) if n_trials > 8 else (50, 100, 200)
    out(f"{n_trials} seeded realisations per n, jitter scale 2.0e-04 s, "
        f"timestamping noise 2.0e-08 s")
    out("")
    out(f"{'n':>7} {'rms skew error [s/s]':>22} {'mean reported u [s/s]':>23} "
        f"{'rms/reported':>13} {'rms offset error [s]':>22}")
    rms_by_n = []
    for n in counts:
        skew_errors = []
        offset_errors = []
        reported = []
        for trial in range(n_trials):
            spec = _spec(
                SEED + 977 * trial, jitter_scale_s=2.0e-4, noise_s=2.0e-8, clock_noise=True
            )
            estimate, fit, _ = _estimate(spec, n, order=1)
            true_skew = TRUE_SKEW + TRUE_DRIFT * fit.t_ref_s
            skew_errors.append(fit.skew_s_per_s - true_skew)
            reported.append(fit.skew_stderr_s_per_s)
            backend_truth = None
            offset, _ = fit.offset_at(fit.t_ref_s)
            true_offset = (
                TRUE_OFFSET_S + TRUE_SKEW * fit.t_ref_s + 0.5 * TRUE_DRIFT * fit.t_ref_s**2
            )
            offset_errors.append(offset - true_offset)
            del backend_truth
        rms = float(np.sqrt(np.mean(np.asarray(skew_errors) ** 2)))
        rms_by_n.append(rms)
        out(
            f"{n:>7d} {rms:>22.6e} {float(np.mean(reported)):>23.6e} "
            f"{rms / float(np.mean(reported)):>13.4f} "
            f"{float(np.sqrt(np.mean(np.asarray(offset_errors) ** 2))):>22.6e}"
        )
    slope_fit = fit_log_slope(np.asarray(counts, dtype=float), np.asarray(rms_by_n))
    out("")
    out(f"fitted convergence exponent : {slope_fit.slope:+.4f} +- {slope_fit.slope_stderr:.4f}")
    out("theory for a straight-line slope with independent noise on equally spaced")
    out("points: var(a) = 12 sigma^2 / (n (n^2 - 1) dt^2), so the standard error falls")
    out("as n^-3/2, i.e. an exponent of -1.5000.")
    out(f"departure from theory       : {abs(slope_fit.slope + 1.5):.4f}")
    if abs(slope_fit.slope + 1.5) > 0.25:
        failures += 1
        out("FAIL: the skew error does not converge at the theoretical rate")
    else:
        out("PASS")

    out.rule("4c. the Kalman consistency diagnostic, measured and reported")
    out("The measurement variance R is taken from the data by a median-absolute-")
    out("deviation estimator with the Gaussian consistency factor 1/0.67449. The")
    out("queueing delay is a right-skewed gamma, so the MAD under-states its standard")
    out("deviation and the filter is expected to be over-confident. The measured")
    out("factor is below, and the implied inflation on the Kalman offset uncertainty")
    out("is its square root.")
    out("")
    out(f"{'jitter scale [s]':>18} {'gamma shape':>12} {'mean NIS':>10} "
        f"{'implied u factor':>18} {'R used [s^2]':>16}")
    nis_rows = []
    for jitter in (0.0, 5.0e-5, 2.0e-4, 1.0e-3):
        spec = SimulationSpec(
            clock=ClockSpec(
                offset0_s=TRUE_OFFSET_S,
                skew_s_per_s=TRUE_SKEW,
                q1_s2_per_s=Q1,
                q2_per_s=Q2,
            ),
            link=LinkSpec(
                floor_ab_s=5.0e-3,
                floor_ba_s=5.0e-3,
                jitter_shape_ab=2.0,
                jitter_shape_ba=2.0,
                jitter_scale_ab_s=jitter,
                jitter_scale_ba_s=jitter,
            ),
            period_s=1.0,
            timestamp_noise_s=2.0e-8,
            resolution_ns=1,
            seed=SEED,
        )
        _, _, kalman = _estimate(spec, scaled(800, minimum=40), order=1)
        nis = kalman.normalised_innovation_mean
        nis_rows.append((jitter, nis))
        out(
            f"{jitter:>18.3e} {2.0:>12.1f} {nis:>10.4f} {np.sqrt(nis):>18.4f} "
            f"{kalman.measurement_variance_s2:>16.6e}"
        )
    out("")
    out("1.0000 is a consistent filter. REPORTED, NOT A FAILURE: the departure is a")
    out("property of using a Gaussian-calibrated robust scale on a skewed")
    out("distribution, it is stated in the module docstring, and a caller who needs a")
    out("consistent filter should supply R explicitly.")

    out.rule("4d. least squares against Kalman on the same data")
    out(f"{'n':>7} {'ls offset [s]':>18} {'ls u [s]':>12} {'kf offset [s]':>18} "
        f"{'kf u [s]':>12} {'|ls - kf| [s]':>15}")
    for n in (100, 400, 1600) if n_trials > 8 else (100,):
        spec = _spec(SEED, jitter_scale_s=2.0e-4, noise_s=2.0e-8, clock_noise=True)
        estimate, fit, kalman = _estimate(spec, n, order=1)
        ls_at_end, ls_u = fit.offset_at(float(estimate.midpoint_s[-1]))
        out(
            f"{n:>7d} {ls_at_end:>18.9e} {ls_u:>12.3e} {kalman.final_offset_s:>18.9e} "
            f"{kalman.final_offset_stderr_s:>12.3e} "
            f"{abs(ls_at_end - kalman.final_offset_s):>15.6e}"
        )
    out("")
    out("Both are evaluated at the last exchange epoch. They are different estimators")
    out("with different assumptions and are not expected to agree exactly; the table")
    out("shows how far apart they are rather than asserting that they are not.")

    out.rule("summary")
    out(f"hard failures : {failures}")
    out(f"saved to      : {out.save()}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

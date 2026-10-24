"""Check 2 -- Allan deviation against closed-form noise processes, with coverage.

Requirement (BATCH_08_SPEC, P079): "Allan deviation verified against synthetic
noise processes with known slopes (white phase, white frequency, random walk)
and the fitted slopes reported against theory".

``allantools`` is not installed in this environment, so nothing here is
compared against another implementation. Each of the three processes has an
Allan variance that can be written down exactly
(:mod:`timetransfer.noise`), and that closed form is the reference:

    white PM        sigma_y(tau) = sqrt(3) * sigma_x / tau        slope -1
    white FM        sigma_y(tau) = sqrt(q1 / tau)                 slope -1/2
    random walk FM  sigma_y(tau) = sqrt(q2 * tau / 3)             slope +1/2

What is checked
---------------
1. **Fitted slopes against theory**, with the standard error of the fit.
2. **Amplitude against theory** at every tau, as a relative departure. A slope
   can be right while the constant is wrong, so both are reported.
3. **Coverage of the confidence interval**, measured by Monte Carlo over seeded
   realisations: the fraction of realisations whose interval at each tau
   contains the theoretical value. The equivalent-degrees-of-freedom formulas
   are approximations, so this is a measurement and not a check that can be
   assumed to pass. Where the achieved coverage misses the nominal it is
   reported as measured, here and in README.md.
4. **Noise-identification hit rate** of the lag-1 autocorrelation method, which
   is what chooses the degrees of freedom when the caller does not declare
   them.

Run: ``python validation/validate_allan.py``
"""

from __future__ import annotations

import sys

import numpy as np

from _bootstrap import Tee, bootstrap, environment_lines, scaled

ROOT = bootstrap()

from timetransfer.allan import allan_table, fit_log_slope, identify_noise  # noqa: E402
from timetransfer.noise import (  # noqa: E402
    NOISE_KINDS,
    synthetic_phase,
    theoretical_adev,
    theory_slope,
)

LEVELS = {"white_pm": 1.0e-9, "white_fm": 1.0e-20, "random_walk_fm": 1.0e-24}
LEVEL_UNITS = {"white_pm": "s (sigma_x)", "white_fm": "s^2/s (q1)", "random_walk_fm": "1/s (q2)"}
N_LONG = 20000
N_COVERAGE = 2048
SEED = 20261024


def main() -> int:
    out = Tee("validate_allan_output.txt")
    out("timetransfer validation 2 -- Allan deviation against closed-form noise processes")
    out("=" * 78)
    for line in environment_lines():
        out(line)
    out("")
    out("allantools is NOT installed in this environment and is NOT used. Every")
    out("reference value below is the closed form printed in timetransfer/noise.py.")
    failures = 0
    slope_rows: list[tuple[str, float, float, float, float]] = []

    out.rule("2a. fitted log-log slopes against theory")
    out(f"N = {N_LONG} phase samples, tau0 = 1 s, octave-spaced tau up to N//5")
    out("")
    out(f"{'process':>16} {'level':>12} {'units':>14} {'theory':>8} {'fitted':>10} "
        f"{'stderr':>10} {'|diff|':>10} {'r^2':>8}")
    for kind in NOISE_KINDS:
        level = LEVELS[kind]
        x = synthetic_phase(kind, N_LONG, tau0_s=1.0, level=level, seed=SEED)
        table = allan_table(x, 1.0, noise_kind=kind)
        fit = fit_log_slope(table.tau_s, table.adev)
        theory = theory_slope(kind)
        slope_rows.append((kind, theory, fit.slope, fit.slope_stderr, abs(fit.slope - theory)))
        out(
            f"{kind:>16} {level:>12.3e} {LEVEL_UNITS[kind]:>14} {theory:>8.2f} "
            f"{fit.slope:>10.4f} {fit.slope_stderr:>10.4f} "
            f"{abs(fit.slope - theory):>10.4f} {fit.r_squared:>8.5f}"
        )
    out("")
    out("tolerance on |fitted - theory| : 0.0500")
    worst_slope = max(row[4] for row in slope_rows)
    out(f"worst |fitted - theory|        : {worst_slope:.4f}")
    if worst_slope > 0.05:
        failures += 1
        out("FAIL: a fitted slope departs from theory by more than the tolerance")
    else:
        out("PASS")

    out.rule("2b. amplitude against theory at every tau")
    for kind in NOISE_KINDS:
        level = LEVELS[kind]
        x = synthetic_phase(kind, N_LONG, tau0_s=1.0, level=level, seed=SEED)
        table = allan_table(x, 1.0, noise_kind=kind)
        theory = theoretical_adev(kind, table.tau_s, level=level)
        out("")
        out(f"{kind} (level {level:.3e} {LEVEL_UNITS[kind]})")
        out(f"{'m':>7} {'tau [s]':>10} {'measured adev':>16} {'theory adev':>16} "
            f"{'rel. error':>12} {'in interval':>12}")
        inside = 0
        for point, theory_value in zip(table.points, theory, strict=True):
            covered = bool(point.lower <= theory_value <= point.upper)
            inside += int(covered)
            out(
                f"{point.m:>7d} {point.tau_s:>10.4g} {point.adev:>16.6e} "
                f"{theory_value:>16.6e} {point.adev / theory_value - 1.0:>12.5f} "
                f"{str(covered):>12}"
            )
        worst = float(np.max(np.abs(table.adev / theory - 1.0)))
        out(f"worst relative error: {worst:.5f}   intervals containing theory: "
            f"{inside}/{len(table.points)}")
        if worst > 0.40:
            failures += 1
            out("FAIL: the measured amplitude departs from theory by more than 40 per cent")

    out.rule("2c. measured coverage of the 95 per cent confidence interval")
    n_trials = scaled(300, minimum=24)
    out(f"{n_trials} seeded realisations of N = {N_COVERAGE} samples per process.")
    out("An interval counts as covering when it contains the closed-form value at that tau.")
    out("These are the simplified edf formulas of Riley, NIST SP 1065 Table 3; the")
    out("coverage below is MEASURED and is reported whether or not it reaches 0.95.")
    out("")
    out(f"{'process':>16} {'m':>5} {'tau [s]':>9} {'covered':>9} {'trials':>7} "
        f"{'measured':>10} {'nominal':>9} {'verdict':>14}")
    coverage_rows: list[tuple[str, int, float, float]] = []
    for kind in NOISE_KINDS:
        level = LEVELS[kind]
        counts: dict[int, int] = {}
        taus: dict[int, float] = {}
        for trial in range(n_trials):
            x = synthetic_phase(
                kind, N_COVERAGE, tau0_s=1.0, level=level, seed=SEED + 1000 * trial + 7
            )
            table = allan_table(x, 1.0, noise_kind=kind)
            theory = theoretical_adev(kind, table.tau_s, level=level)
            for point, value in zip(table.points, theory, strict=True):
                taus[point.m] = point.tau_s
                counts[point.m] = counts.get(point.m, 0) + int(point.lower <= value <= point.upper)
        for m in sorted(counts):
            measured = counts[m] / n_trials
            verdict = (
                "calibrated"
                if 0.90 <= measured <= 0.99
                else ("optimistic" if measured < 0.90 else "conservative")
            )
            coverage_rows.append((kind, m, measured, 0.95))
            out(
                f"{kind:>16} {m:>5d} {taus[m]:>9.4g} {counts[m]:>9d} {n_trials:>7d} "
                f"{measured:>10.4f} {0.95:>9.2f} {verdict:>14}"
            )
    worst_row = min(coverage_rows, key=lambda row: row[2])
    best_row = max(coverage_rows, key=lambda row: row[2])
    out("")
    out(f"lowest coverage  : {worst_row[2]:.4f} at {worst_row[0]} m={worst_row[1]}")
    out(f"highest coverage : {best_row[2]:.4f} at {best_row[0]} m={best_row[1]}")
    out("REPORTED, NOT A FAILURE: the edf formulas are approximations and this is the")
    out("measurement of how good they are. A coverage below 0.95 means the interval at")
    out("that tau is narrower than it should be, and a user should widen it or use more")
    out("data rather than trust the printed bound.")

    out.rule("2d. noise identification hit rate (lag-1 autocorrelation method)")
    n_ident = scaled(200, minimum=20)
    out(f"{n_ident} seeded realisations of N = {N_COVERAGE} samples per process.")
    out("")
    out(f"{'process':>16} {'hits':>7} {'trials':>7} {'rate':>8} {'mean alpha':>12} "
        f"{'theory alpha':>13}")
    theory_alpha = {"white_pm": 2.0, "white_fm": 0.0, "random_walk_fm": -2.0}
    identification_ok = True
    for kind in NOISE_KINDS:
        hits = 0
        alphas = []
        for trial in range(n_ident):
            x = synthetic_phase(
                kind, N_COVERAGE, tau0_s=1.0, level=LEVELS[kind], seed=SEED + 31 * trial
            )
            result = identify_noise(x)
            hits += int(result.kind == kind)
            alphas.append(result.alpha)
        rate = hits / n_ident
        identification_ok = identification_ok and rate >= 0.90
        out(
            f"{kind:>16} {hits:>7d} {n_ident:>7d} {rate:>8.4f} "
            f"{float(np.mean(alphas)):>12.4f} {theory_alpha[kind]:>13.1f}"
        )
    out("")
    out("threshold for a pass: every rate >= 0.90")
    if not identification_ok:
        failures += 1
        out("FAIL: the identifier misses a process it is meant to recognise")
    else:
        out("PASS")

    out.rule("summary")
    out("fitted slopes against theory:")
    for kind, theory, fitted, stderr, diff in slope_rows:
        out(f"  {kind:<16} theory {theory:+.2f}  fitted {fitted:+.4f} +- {stderr:.4f}"
            f"  |diff| {diff:.4f}")
    out("")
    out(f"hard failures : {failures}")
    out(f"saved to      : {out.save()}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

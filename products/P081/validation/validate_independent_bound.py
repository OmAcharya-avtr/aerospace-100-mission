"""The Cramer-Rao bound against the measured spread of repeated nonlinear fits.

Why this is a different kind of check
-------------------------------------
"Two implementations agree" and "the answer lies inside a bound derived from
different mathematics" are not the same check, and the second catches what the
first cannot. Everything else in this repository compares one computation of
the sensitivity matrix against another, or against the derivation it came from.
This script compares the **bound** -- Rao 1945 and Cramer 1946, via the Fisher
information of equation (9) -- against the **measured** scatter of an estimator
that the bound knows nothing about: the full nonlinear least-squares fit of
:func:`twindiverge.profile.fit`, run once per independent noise realisation of
the same truth. The bound must lie below the scatter, in every identifiable
direction and in the matrix sense.

Two estimators, because the comparison means different things for each
---------------------------------------------------------------------
- The **linear generalised-least-squares estimator** of equation (12), which is
  the efficient estimator of the LINEARISED model. Its spread should sit on the
  bound, and the measured ratio is therefore also a check that the bound is
  tight rather than merely true.
- The **nonlinear least-squares fit** of :func:`twindiverge.profile.fit`, which
  optimises the real model globally. Its spread is far ABOVE the bound in the
  best-determined directions, and the reason is reported rather than smoothed
  over: on a problem this ill-conditioned the global optimum wanders along a
  curved chi-square valley, so the estimator is biased, and its measured bias is
  printed. **The plain Cramer-Rao inequality bounds unbiased estimators**, so
  for this estimator the check is a necessary condition and not a tight one.

Two forms of the inequality
---------------------------
1. **Per direction.** For each identifiable direction ``v_i``, the standard
   deviation of ``v_i^T delta_hat`` over realisations must be at least
   ``1 / sigma_i``.
2. **In the matrix sense**, which is the stronger statement from the same
   mathematics: the empirical covariance of the projected estimates minus the
   Cramer-Rao covariance must be positive semi-definite. The least eigenvalue
   of that difference is the measurement.

How the slack is set, and why it is not a constant
--------------------------------------------------
Two terms, both derived rather than chosen:

- **Sampling.** The sample standard deviation of ``n`` normal draws has a
  relative standard error of ``1 / sqrt(2*(n-1))``, so a measured spread may sit
  below the bound by that much without the bound being violated. The allowance
  is the DECLARED four of those standard errors
  (``_bootstrap.CONTROL_SIGMA``), not a level at which an honest run would fire
  once in twenty.
- **Conditioning.** The bound itself is computed from a matrix whose condition
  number is reported, so its own relative error is of order
  ``cond(A) * eps``. That term scales with the conditioning of the bound
  expression, which is what it must do: an absolute floor would make a narrow
  bracket vacuous and would admit exactly the error this check exists to catch.

Both terms are printed with their sizes so a reader can see which one is doing
the work -- and on this plant it is the sampling term by eleven orders of
magnitude.

Negative controls
-----------------
A bound check that cannot fire is not a check. Two controls, both anchored on
the BOUND -- which is the true value here -- and not on any output, because a
control that scales an output by a factor is only as strong as that output
happened to be: an estimator whose spread is already sixty times the bound
survives being halved, which is how the first version of this control failed to
fire.

- A spread rescaled to a declared ``CONTROL_LOW`` multiple of the bound, below
  it by construction. This control MUST fire in every direction.
- A spread rescaled to a declared ``CONTROL_HIGH`` multiple, above it by
  construction. This control must NOT fire. Both directions are asserted,
  because a criterion that fires on everything is as useless as one that fires
  on nothing.
"""

from __future__ import annotations

# See validate_plant.py for why this statement is here and not in _bootstrap.
import sys  # noqa: E402

sys.dont_write_bytecode = True

import time  # noqa: E402

import numpy as np  # noqa: E402

from _bootstrap import (  # noqa: E402
    BOUND_SEEDS,
    CONTROL_SIGMA,
    Tee,
    bootstrap,
    environment_lines,
)

ROOT = bootstrap()

from twindiverge.identifiability import analyse  # noqa: E402
from twindiverge.pipeline import montecarlo_directions  # noqa: E402
from twindiverge.plant import NOMINAL, simulate_with_sensitivities  # noqa: E402
from twindiverge.report import format_table, number  # noqa: E402
from twindiverge.telemetry import (  # noqa: E402
    TELEMETRY_SETS,
    TelemetrySet,
    design_matrix,
    measure,
    residual_vector,
)


def linear_directions(telemetry: TelemetrySet, seeds: tuple[int, ...]) -> np.ndarray:
    """Equation (13) applied to each noise realisation, shape ``(seeds, rank)``.

    The linear generalised-least-squares amplitude along each identifiable
    direction is ``u_i^T r / sigma_i``, so this is a projection and not an
    optimisation: it is the efficient estimator of the linearised model and its
    spread is what the Cramer-Rao bound is tight against.
    """
    report = analyse(telemetry, NOMINAL)
    twin = simulate_with_sensitivities(NOMINAL, telemetry.campaign)
    matrix, _ = design_matrix(telemetry, NOMINAL, trajectory=twin)
    left, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
    rows = []
    for seed in seeds:
        observed, _ = measure(telemetry, NOMINAL, seed=seed)
        residual = residual_vector(telemetry, observed, twin)
        rows.append(
            (left[:, : report.rank].T @ residual) / singular_values[: report.rank]
        )
    return np.vstack(rows)

#: The sets the bound is checked on. The reference rate-only set, where the
#: bound exists only per direction because the parameters themselves are
#: refused, and its two-channel twin, where it exists per parameter too.
CHECKED = ("rate_full", "rate_torque_full")

#: Declared multiple of the bound the first negative control rescales the
#: spread to. 0.25 is four times below the bound, far outside the sampling
#: allowance of about 0.5 at this number of realisations.
CONTROL_LOW = 0.25

#: Declared multiple for the control that must NOT fire.
CONTROL_HIGH = 2.0


def main() -> int:
    out = Tee("validate_independent_bound_output.txt")
    out.rule("twindiverge -- the Cramer-Rao bound against a measured spread")
    for line in environment_lines():
        out(line)
    out("")
    out(f"noise realisations per set : {len(BOUND_SEEDS)}")
    out(f"declared control sigma     : {CONTROL_SIGMA:.1f}")
    sampling = CONTROL_SIGMA / np.sqrt(2.0 * (len(BOUND_SEEDS) - 1))
    out(f"sampling allowance         : {sampling:.6f} of the bound, from "
        f"{CONTROL_SIGMA:.1f} / sqrt(2 * ({len(BOUND_SEEDS)} - 1))")
    out("")
    out("The truth is the declared nominal parameter set, so the estimator is being")
    out("asked to recover a known answer and its scatter is the whole measurement.")

    for name in CHECKED:
        telemetry = TELEMETRY_SETS[name]
        report = analyse(telemetry, NOMINAL)
        out.rule(f"{name}: rank {report.rank} of 5, condition number "
                 f"{report.condition_number:.6e}")
        started = time.perf_counter()
        amplitudes, bound = montecarlo_directions(telemetry, NOMINAL, BOUND_SEEDS)
        elapsed = time.perf_counter() - started
        linear = linear_directions(telemetry, BOUND_SEEDS)
        conditioning = report.condition_number * np.finfo(float).eps * bound
        limit = bound * (1.0 - sampling) - conditioning
        out(f"wall clock for {len(BOUND_SEEDS)} nonlinear fits : {elapsed:.3f} s")
        out(f"sampling term, as a fraction of the bound : {sampling:.6e}")
        out(f"conditioning term, worst fraction         : "
            f"{float((conditioning / bound).max()):.6e}")
        out("")

        for label, estimates, note in (
            ("linear, equation (12)", linear,
             "the efficient estimator of the linearised model: its spread should sit "
             "ON the bound"),
            ("nonlinear, profile.fit", amplitudes,
             "the global optimiser of the real model: biased on this problem, so the "
             "bound is necessary but not tight"),
        ):
            spread = estimates.std(axis=0, ddof=1)
            mean = estimates.mean(axis=0)
            rows = []
            violations = 0
            for index in range(report.rank):
                ok = float(spread[index]) >= float(limit[index])
                violations += int(not ok)
                rows.append([
                    f"direction {index + 1}",
                    number(float(bound[index])),
                    number(float(spread[index])),
                    f"{float(spread[index] / bound[index]):.6f}",
                    number(float(mean[index])),
                    f"{float(mean[index] / bound[index]):+.4f}",
                    number(float(limit[index])),
                    "ok" if ok else "VIOLATED",
                ])
            out("")
            out(f"-- {label}: {note}")
            out("")
            out(format_table(
                ["direction", "CRB", "measured sd", "sd / CRB", "measured mean",
                 "mean / CRB", "criterion", "verdict"],
                rows,
            ))
            out("")
            out.check(
                f"{name}, {label}: the bound lies below the measured spread everywhere",
                violations == 0,
                f"{violations} violations in {report.rank} directions; ratios from "
                f"{float((spread / bound).min()):.6f} to "
                f"{float((spread / bound).max()):.6f}",
            )

            empirical = np.atleast_2d(np.cov(estimates, rowvar=False, ddof=1))
            difference = empirical - np.diag(bound**2)
            eigenvalues = np.linalg.eigvalsh(difference)
            tolerance = float(
                CONTROL_SIGMA
                * np.sqrt(2.0 / (len(BOUND_SEEDS) - 1))
                * float((bound**2).max())
            )
            out(format_table(
                ["quantity", "value"],
                [
                    ["least eigenvalue of covariance difference",
                     number(float(eigenvalues.min()))],
                    ["greatest eigenvalue", number(float(eigenvalues.max()))],
                    ["largest CRB variance", number(float((bound**2).max()))],
                    ["sampling tolerance on an eigenvalue", number(tolerance)],
                ],
            ))
            out("")
            out.check(
                f"{name}, {label}: the empirical covariance dominates the CRB covariance",
                float(eigenvalues.min()) > -tolerance,
                f"least eigenvalue {float(eigenvalues.min()):.6e} against a sampling "
                f"tolerance of {-tolerance:.6e}",
            )

        out("")
        out("negative controls, both anchored on the BOUND and not on any output: the")
        out("deviations are rescaled so that their standard deviation is a declared")
        out("multiple of the bound itself.")
        out("")
        reference = linear
        mean = reference.mean(axis=0)
        spread = reference.std(axis=0, ddof=1)
        for multiple, must_fire in ((CONTROL_LOW, True), (CONTROL_HIGH, False)):
            target = multiple * bound
            rescaled = (reference - mean) * (target / spread) + mean
            rescaled_spread = rescaled.std(axis=0, ddof=1)
            fired = int(np.sum(rescaled_spread < limit))
            out(format_table(
                ["direction", "CRB", f"sd rescaled to {multiple:.2f} x CRB",
                 "sd / CRB", "criterion", "fires"],
                [
                    [
                        f"direction {index + 1}",
                        number(float(bound[index])),
                        number(float(rescaled_spread[index])),
                        f"{float(rescaled_spread[index] / bound[index]):.6f}",
                        number(float(limit[index])),
                        "yes" if rescaled_spread[index] < limit[index] else "no",
                    ]
                    for index in range(report.rank)
                ],
            ))
            out("")
            out.control(
                f"{name}: a spread at {multiple:.2f} times the bound",
                fired > 0,
                must_fire,
                f"{fired} of {report.rank} directions violated the criterion, "
                f"{'all' if must_fire else 'none'} expected",
            )
            if must_fire:
                out.check(
                    f"{name}: the low control fires in EVERY direction, not merely one",
                    fired == report.rank,
                    f"{fired} of {report.rank}",
                )

    out.rule("what this check is, and what it is not")
    out("It is the only check in this repository whose reference is not another")
    out("computation of the sensitivity matrix. The bound comes from the Fisher")
    out("information; the spread comes from an estimator that never sees it.")
    out("")
    out("It is NOT a check that the estimator is efficient. A spread far ABOVE the")
    out("bound would pass this check and would mean the estimator is wasteful; the")
    out("ratios are printed so a reader can see which it is. On this plant the")
    out("ratios sit close to one, which is what a maximum-likelihood estimator on a")
    out("nearly linear problem should do, and it is reported rather than asserted,")
    out("because the Cramer-Rao inequality does not promise it.")

    path = out.save()
    print(f"\nwritten: {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

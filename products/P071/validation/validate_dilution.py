"""The dilution region: closed-form maximum, the non-monotonic turn, the warning.

Checks:
1. the closed-form maximum Pc over covariance scale against Alfano's
   ``R^2 / (e d^2)`` for a circular covariance, exactly;
2. the closed-form maximum against the numerical maximum of the Foster-Estes
   Pc, over a grid of geometries, with the error of the small-hard-body
   assumption measured;
3. the Pc-against-covariance-scale curve exhibited, with its single interior
   maximum located and the monotonicity on each branch asserted;
4. the measured turning point against the ``M = sqrt(2)`` criterion;
5. the library raising :class:`conjprob.dilution.DilutionWarning` on the cases
   that are on the dilution branch and not on the one that is not.
"""

from __future__ import annotations

import math
import warnings

import numpy as np

from _bootstrap import Tee, bootstrap, environment_lines

bootstrap()

from conjprob.cases import load_case  # noqa: E402
from conjprob.dilution import (  # noqa: E402
    DILUTION_MAHALANOBIS_THRESHOLD,
    DilutionWarning,
    assess_dilution,
    dilution_curve,
    max_probability,
)
from conjprob.geometry import encounter_geometry  # noqa: E402
from conjprob.report import assess_conjunction  # noqa: E402

CASES = (
    "examples/cases/leo_crossing.yaml",
    "examples/cases/large_body_tight_covariance.yaml",
    "examples/cases/far_miss_tight_covariance.yaml",
)
SCALES = np.logspace(-1.5, 2.5, 241)


def main() -> int:
    out = Tee("validate_dilution_output.txt")
    out.rule("conjprob validation: probability dilution")
    for line in environment_lines():
        out(line)
    failures = 0

    out.rule("1. closed-form maximum against Alfano's R^2 / (e d^2), circular covariance")
    out("reference: Alfano, S. (2005), Relating position uncertainty to maximum")
    out("           conjunction probability, J. Astronautical Sciences 53(2), 193-205")
    out(f"{'R [m]':>8s} {'d [m]':>9s} {'R^2/(e d^2)':>16s} {'max_probability':>16s} {'rel':>10s}")
    worst = 0.0
    for radius in (1.0, 12.0, 30.0):
        for distance in (50.0, 120.0, 1000.0):
            expected = radius**2 / (math.e * distance**2)
            value, scale = max_probability(300.0, 300.0, distance, 0.0, radius)
            worst = max(worst, abs(value / expected - 1.0))
            out(
                f"{radius:>8.3g} {distance:>9.4g} {expected:>16.10e} {value:>16.10e} "
                f"{value / expected - 1.0:>+10.2e}"
            )
    out("")
    out(f"worst relative difference {worst:.6e}; tolerance 1e-13")
    ok = worst <= 1e-13
    out(f"result: {'PASS' if ok else 'FAIL'}")
    failures += 0 if ok else 1
    out("the scale attaining the maximum is M / sqrt(2) in the same closed form")

    out.rule("2. closed form against the numerical maximum of the Foster-Estes Pc")
    out("the closed form is the small-hard-body limit, so its error grows with R/sigma")
    out(
        f"{'R/sigma_eff':>12s} {'aspect':>7s} {'closed form':>15s} {'numerical':>15s} "
        f"{'rel. err':>11s} {'scale cf':>10s} {'scale num':>10s}"
    )
    rows = []
    for ratio in (0.001, 0.01, 0.1, 0.3, 1.0):
        for aspect in (1.0, 5.0):
            sigma_x = 300.0 * math.sqrt(aspect)
            sigma_y = 300.0 / math.sqrt(aspect)
            radius = ratio * 300.0
            closed, scale_closed = max_probability(sigma_x, sigma_y, 150.0, 60.0, radius)
            assessment = assess_dilution(
                sigma_x, sigma_y, 150.0, 60.0, radius, warn=False
            )
            error = assessment.pc_max_numerical / closed - 1.0
            rows.append((ratio, error))
            out(
                f"{ratio:>12.4g} {aspect:>7.3g} {closed:>15.8e} "
                f"{assessment.pc_max_numerical:>15.8e} {error:>+11.3e} "
                f"{scale_closed:>10.5f} {assessment.scale_at_max_numerical:>10.5f}"
            )
    small = max(abs(error) for ratio, error in rows if ratio <= 0.01)
    scaled = [
        (ratio, abs(error) / ratio**2) for ratio, error in rows if 0.0 < ratio <= 0.1
    ]
    out("")
    out(f"worst relative error for R/sigma_eff <= 0.01: {small:.6e}; tolerance 1e-3")
    ok = small <= 1e-3
    out(f"result: {'PASS' if ok else 'FAIL'}")
    failures += 0 if ok else 1
    out("")
    out("the error of the closed form scales as (R/sigma_eff)^2, which is the order of the")
    out("neglected term; |relative error| / (R/sigma_eff)^2 over the rows with")
    out("R/sigma_eff <= 0.1:")
    for ratio, value in scaled:
        out(f"  R/sigma_eff {ratio:>8.4g}   ratio {value:>12.6f}")
    out("at R/sigma_eff = 1 the closed form is no longer a useful maximum and the")
    out("numerical value is the one to use; both are reported by the library")

    out.rule("3. the Pc-against-covariance-scale curve, with its non-monotonic turn")
    out("case leo-crossing; scale multiplies both principal standard deviations")
    case = load_case("examples/cases/leo_crossing.yaml")
    frame = encounter_geometry(case.primary, case.secondary).frame
    exact, approximate = dilution_curve(
        frame.sigma_x_m, frame.sigma_y_m, frame.x_bar_m, frame.y_bar_m, case.radius_m, SCALES
    )
    peak = int(np.argmax(exact))
    rising = bool(np.all(np.diff(exact[: peak + 1]) > 0.0))
    falling = bool(np.all(np.diff(exact[peak:]) < 0.0))
    out(f"{'scale':>12s} {'pc_foster':>16s} {'pc_small_body':>16s} {'branch':>10s}")
    for index in range(0, SCALES.size, 20):
        branch = "rising" if index < peak else ("peak" if index == peak else "diluting")
        out(
            f"{SCALES[index]:>12.6g} {exact[index]:>16.8e} {approximate[index]:>16.8e} "
            f"{branch:>10s}"
        )
    out("")
    out(f"maximum of the sampled curve {exact[peak]:.8e} at scale {SCALES[peak]:.6f}")
    out(f"strictly increasing left of the maximum : {rising}")
    out(f"strictly decreasing right of the maximum: {falling}")
    out(f"result: {'PASS' if rising and falling else 'FAIL'}")
    failures += 0 if rising and falling else 1

    out.rule("4. the measured turning point against the M = sqrt(2) criterion")
    out(
        "criterion: the present covariance is on the dilution branch when M < "
        f"{DILUTION_MAHALANOBIS_THRESHOLD:.10f}"
    )
    out("the measured scale at the maximum should equal M / sqrt(2)")
    out(
        f"{'M':>7s} {'M/sqrt(2)':>12s} {'measured scale':>15s} {'rel':>10s} "
        f"{'branch flag':>12s} {'sensitivity':>13s}"
    )
    worst = 0.0
    for mahalanobis in (0.2, 0.5, 1.0, 1.4142135623730951, 2.0, 5.0, 10.0):
        sigma = 300.0
        assessment = assess_dilution(
            sigma, sigma, mahalanobis * sigma, 0.0, 3.0, warn=False
        )
        expected = mahalanobis / math.sqrt(2.0)
        relative = assessment.scale_at_max_numerical / expected - 1.0
        worst = max(worst, abs(relative))
        out(
            f"{mahalanobis:>7.4f} {expected:>12.6f} "
            f"{assessment.scale_at_max_numerical:>15.6f} {relative:>+10.2e} "
            f"{str(assessment.on_dilution_branch):>12s} "
            f"{assessment.log_scale_sensitivity:>+13.4e}"
        )
    out("")
    out(f"worst relative difference in the scale at the maximum: {worst:.6e}; tolerance 1e-3")
    ok = worst <= 1e-3
    out(f"result: {'PASS' if ok else 'FAIL'}")
    failures += 0 if ok else 1
    out("the branch flag is the sign of a finite-difference derivative, so at")
    out("M = sqrt(2) exactly the derivative is within rounding of zero and the boolean")
    out("is not a promise; the assessment reports near_turning_point for that row")

    out.rule("5. the library raising the warning itself")
    out(f"{'case':<34s} {'M':>9s} {'on branch':>10s} {'warning':>9s} {'dilution factor':>16s}")
    for path in CASES:
        case = load_case(path)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            report = assess_conjunction(case.primary, case.secondary, case.radius_m)
        raised = any(issubclass(w.category, DilutionWarning) for w in caught)
        expected = report.dilution.on_dilution_branch
        out(
            f"{case.name:<34s} {report.geometry.mahalanobis:>9.5f} "
            f"{str(expected):>10s} {str(raised):>9s} "
            f"{report.dilution.dilution_factor:>16.6e}"
        )
        if raised != expected:
            failures += 1
    out("")
    out("the warning is raised by conjprob.dilution.assess_dilution, not by the caller")
    out("inspecting a flag, so a user who ignores the return value still sees it")

    out.rule("6. subnormal miss vector, a regression case")
    out("a subnormal miss component leaves the unsquared Mahalanobis hypotenuse")
    out("strictly positive while its square underflows to zero, which turned a guarded")
    out("condition in max_probability into a division by zero. Found by a property-based")
    out("test and pinned in tests/test_regressions.py.")
    tiny = 2.2250738585072014e-308
    out(f"    xbar = {tiny!r} m, sigma = 1 m")
    out(f"    hypot(xbar/sigma, 0)      = {math.hypot(tiny / 1.0, 0.0)!r}")
    out(f"    hypot(xbar/sigma, 0)**2   = {math.hypot(tiny / 1.0, 0.0) ** 2!r}")
    try:
        max_probability(1.0, 1.0, tiny, 0.0, 1.0)
    except ValueError as error:
        raised = "underflowing miss vector" in str(error)
        out(f"    ValueError raised, message names the underflow: {raised}")
    else:
        raised = False
        out("    no ValueError was raised")
    out(f"result: {'PASS' if raised else 'FAIL'}")
    failures += 0 if raised else 1

    out.rule("summary")
    out(f"checks failed: {failures}")
    out(f"written to {out.save()}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

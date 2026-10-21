"""Where the rectilinear 2-D formulation stops being valid, as measured curves.

Three separate failures of the short-encounter assumption set are measured here,
and the Monte Carlo estimators used as the reference are the same code paths the
rest of the product uses.

1. **Finite screening window.** Restricting the encounter to ``|t| <= T`` loses
   probability. The ratio is measured against ``T / t_cross`` with
   ``t_cross = sigma_eff / |v_rel|``.
2. **Curved relative motion.** A constant relative acceleration breaks the
   straight-line assumption. The error of the rectilinear 2-D Pc is measured
   against the dimensionless curvature ``kappa = |a| T^2 / (2 sigma_eff)``, and
   the kappa at which the error reaches 1 and 10 per cent is interpolated from
   the measurement.
3. **The discretisation bias of the curved estimator itself**, measured by
   refining its time grid, because a reference whose own bias is unmeasured is
   not a reference.

The measured kappa is then converted into the relative speed at which the
rectilinear assumption costs 1 per cent for a declared low Earth encounter,
which is the form an operator can act on.
"""

from __future__ import annotations

import warnings

import numpy as np

from _bootstrap import Tee, bootstrap, environment_lines

bootstrap()

from conjprob.cases import load_case  # noqa: E402
from conjprob.dilution import DilutionWarning  # noqa: E402
from conjprob.montecarlo import pc_monte_carlo_curved  # noqa: E402
from conjprob.nonlinear import (  # noqa: E402
    curvature_sweep,
    first_crossing,
    gravity_gradient_acceleration_scale,
    window_sweep,
)
from conjprob.report import assess_conjunction  # noqa: E402

CASE = "examples/cases/large_body_tight_covariance.yaml"
WINDOW_FACTORS = np.array([0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0])
WINDOW_WIDTH_FACTOR = 8.0
GEOCENTRIC_RADIUS_M = 7078137.0
SEED = 4242


def main() -> int:
    out = Tee("validate_nonlinear_output.txt")
    out.rule("conjprob validation: breakdown of the rectilinear 2-D formulation")
    for line in environment_lines():
        out(line)
    failures = 0

    case = load_case(CASE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DilutionWarning)
        report = assess_conjunction(case.primary, case.secondary, case.radius_m)
    geometry = report.geometry
    sigma_eff = geometry.sigma_effective_m
    crossing = geometry.crossing_time_s
    reference = report.analytic["foster"]
    out("")
    out(f"case                   : {case.name}")
    out(f"miss distance          : {geometry.miss_distance_m:.6f} m")
    out(f"relative speed         : {geometry.relative_speed_mps:.6f} m/s")
    out(f"effective sigma        : {sigma_eff:.6f} m")
    out(f"encounter crossing time: {crossing:.6e} s")
    out(f"rectilinear 2-D Pc     : {reference:.8e}")
    out("this case is used because its Pc is of order 0.5, so a Monte Carlo interval of")
    out("under one per cent costs 60000 samples and the curvature effect is resolvable")

    out.rule("1. finite screening window")
    out("estimator: pc_monte_carlo_window, rectilinear, exact per-sample minimum")
    unbounded, points = window_sweep(
        geometry.combined_covariance_m2,
        geometry.miss_vector_m,
        geometry.relative_velocity_mps,
        case.radius_m,
        WINDOW_FACTORS * crossing,
        sigma_effective_m=sigma_eff,
        n_samples=200_000,
        seed=SEED,
    )
    out(f"unbounded-time Monte Carlo: {unbounded.pc:.8f} ({unbounded.hits} hits)")
    out(
        f"{'T/t_cross':>10s} {'T [s]':>12s} {'pc in window':>14s} {'ratio':>10s} "
        f"{'ratio CI':>26s}"
    )
    ratios = []
    for point in points:
        ratios.append(point.ratio_to_unbounded)
        out(
            f"{point.window_over_crossing_time:>10.4f} {point.window_s:>12.6e} "
            f"{point.monte_carlo.pc:>14.8f} {point.ratio_to_unbounded:>10.6f} "
            f"[{point.ratio_ci_low:>10.6f}, {point.ratio_ci_high:>10.6f}]"
        )
    monotone = bool(np.all(np.diff(ratios) >= -1e-12))
    bounded = bool(np.all(np.array(ratios) <= 1.0 + 1e-12))
    out("")
    out(f"ratios non-decreasing in window length : {monotone}")
    out(f"ratios never exceed one                : {bounded}")
    out(f"ratio at T = 8 t_cross                 : {ratios[-1]:.6f}")
    out("the 2-D formulation integrates over all time, so it overstates the probability")
    out("of a conjunction inside a short screening window by the reciprocal of this")
    out("ratio: at half a crossing time that is a factor of "
        f"{1.0 / ratios[3]:.4f}")
    ok = monotone and bounded and abs(ratios[-1] - 1.0) <= 1e-9
    out(f"result: {'PASS' if ok else 'FAIL'}")
    failures += 0 if ok else 1

    out.rule("2. curved relative motion")
    window_s = WINDOW_WIDTH_FACTOR * crossing
    out(f"window half-width T = {WINDOW_WIDTH_FACTOR:g} t_cross = {window_s:.6e} s")
    out("acceleration applied along the miss direction, which is the worst case for Pc")
    out("estimator: pc_monte_carlo_curved, 60000 samples, 129 time nodes, 4 refinements")
    accelerations = 2.0 * sigma_eff * np.logspace(-3.0, 4.0, 15) / window_s**2
    curvature_points = curvature_sweep(
        geometry.combined_covariance_m2,
        geometry.miss_vector_m,
        geometry.relative_velocity_mps,
        case.radius_m,
        accelerations,
        window_s=window_s,
        pc_linear_2d=reference,
        sigma_effective_m=sigma_eff,
        n_samples=60_000,
        seed=SEED,
        time_nodes=129,
        refinements=4,
    )
    out(
        f"{'kappa':>11s} {'|a| [m/s^2]':>13s} {'mc pc':>11s} {'rel. err':>11s} "
        f"{'rel. err CI':>26s}"
    )
    kappa = np.array([point.curvature_parameter for point in curvature_points])
    errors = np.array([abs(point.relative_error) for point in curvature_points])
    for point in curvature_points:
        out(
            f"{point.curvature_parameter:>11.4e} {point.acceleration_mps2:>13.5e} "
            f"{point.monte_carlo.pc:>11.6f} {point.relative_error:>+11.6f} "
            f"[{point.relative_error_ci_low:>+11.6f}, {point.relative_error_ci_high:>+11.6f}]"
        )
    crossings = {}
    out("")
    for level in (0.01, 0.10):
        crossings[level] = first_crossing(kappa, errors, level)
        out(
            f"measured kappa at which the rectilinear 2-D error reaches "
            f"{level * 100:5.1f} per cent: {crossings[level]:.6e}"
        )
    growing = errors[-1] > errors[0]
    out(f"error grows with kappa over the sweep: {growing}")
    ok = growing and np.isfinite(crossings[0.01]) and np.isfinite(crossings[0.10])
    out(f"result: {'PASS' if ok else 'FAIL'}")
    failures += 0 if ok else 1

    out.rule("3. discretisation bias of the curved estimator")
    out("a time grid can only overestimate the minimum separation, so it can only")
    out("underestimate Pc; the bias is measured by refining the grid at fixed seed")
    out("geometry: the kappa = 10 point of check 2")
    acceleration = float(2.0 * sigma_eff * 10.0 / window_s**2)
    direction = geometry.miss_vector_m / np.linalg.norm(geometry.miss_vector_m)
    out(
        f"{'time nodes':>11s} {'refinements':>12s} {'hits':>7s} {'pc':>11s} "
        f"{'rel. to finest':>15s}"
    )
    results = []
    for nodes, refinements in ((9, 0), (33, 0), (129, 0), (129, 4), (513, 6)):
        result = pc_monte_carlo_curved(
            geometry.combined_covariance_m2,
            geometry.miss_vector_m,
            geometry.relative_velocity_mps,
            case.radius_m,
            window_s,
            direction * acceleration,
            n_samples=60_000,
            seed=SEED,
            time_nodes=nodes,
            refinements=refinements,
        )
        results.append((nodes, refinements, result))
    finest = results[-1][2].pc
    monotone = True
    previous = -1
    for nodes, refinements, result in results:
        out(
            f"{nodes:>11d} {refinements:>12d} {result.hits:>7d} {result.pc:>11.6f} "
            f"{result.pc / finest - 1.0:>+15.6e}"
        )
        monotone = monotone and result.hits >= previous
        previous = result.hits
    out("")
    out(f"hit count non-decreasing under refinement: {monotone}")
    out(
        f"bias of the package default (129 nodes, 4 refinements) against the finest grid: "
        f"{results[3][2].pc / finest - 1.0:+.6e}"
    )
    out("tolerance: |bias| <= 1e-3 relative")
    ok = monotone and abs(results[3][2].pc / finest - 1.0) <= 1e-3
    out(f"result: {'PASS' if ok else 'FAIL'}")
    failures += 0 if ok else 1

    out.rule("4. the measured kappa expressed as a relative speed")
    out("kappa = |a| T^2 / (2 sigma_eff) with T = w sigma_eff / |v| gives")
    out("kappa = |a| w^2 sigma_eff / (2 |v|^2), so the relative speed at which a given")
    out(
        "kappa is reached is |v| = w sqrt(|a| sigma_eff / (2 kappa)) with "
        f"w = {WINDOW_WIDTH_FACTOR:g}"
    )
    out("the acceleration is the gravity-gradient scale mu |dr| / r^3 at r = 7078137 m")
    out(
        f"{'sigma_eff [m]':>14s} {'separation [m]':>15s} {'|a| [m/s^2]':>13s} "
        f"{'v at 1 pc [m/s]':>17s} {'v at 10 pc [m/s]':>18s}"
    )
    declared = ((sigma_eff, geometry.miss_distance_m), (200.0, 300.0), (1000.0, 500.0))
    for sigma, separation in declared:
        acceleration = gravity_gradient_acceleration_scale(GEOCENTRIC_RADIUS_M, separation)
        speeds = []
        for level in (0.01, 0.10):
            value = crossings[level]
            speeds.append(
                WINDOW_WIDTH_FACTOR * np.sqrt(acceleration * sigma / (2.0 * value))
            )
        out(
            f"{sigma:>14.4f} {separation:>15.4f} {acceleration:>13.5e} "
            f"{speeds[0]:>17.6f} {speeds[1]:>18.6f}"
        )
    out("")
    out("a typical low Earth conjunction has a relative speed of several kilometres per")
    out("second, which is four orders of magnitude above these figures, so the")
    out("rectilinear assumption is excellent there. It is the slow encounter -- metres")
    out("per second and below, which is the geostationary co-location case -- where the")
    out("2-D formulation in this package should not be used, and this is the measurement")
    out("that says so rather than an assertion.")

    out.rule("summary")
    out(f"checks failed: {failures}")
    out(f"written to {out.save()}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

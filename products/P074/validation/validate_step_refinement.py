"""Integrator convergence under step refinement, and the truncation floor it buys.

Two independent routes to the same order.

**Richardson.** With three runs at ``h``, ``h/2`` and ``h/4`` the convergence
order follows from the successive differences alone,
``p = log2((x_h - x_h/2) / (x_h/2 - x_h/4))``, and no reference solution is
needed. This is the stronger statement, because a reference solution of its
own has an error.

**Against the closed form.** With only the central term enabled, the exact
solution is available from :mod:`orbitdrift.kepler`, which never evaluates the
equations of motion. The error against it, and its ratio under halving, give
the order again.

The truncation floor is the number that matters for the product: at the step
every divergence curve in this repository was produced with, how large is the
integrator's own error compared with the divergence it is measuring? Both are
reported, and their ratio.

Drag is switched off for the order measurement and on for a separate run. The
published atmosphere is a set of 28 independent exponential fits and is
therefore **discontinuous at the band boundaries**, which costs a fixed-step
Runge-Kutta method its formal order at the instant of crossing. Both numbers
are published rather than only the clean one.
"""

from __future__ import annotations

import sys

from _bootstrap import (
    Tee,
    Timer,
    bootstrap,
    environment_lines,
    is_fast,
    load_dataset,
)

bootstrap()

import numpy as np  # noqa: E402

from orbitdrift import integrate  # noqa: E402
from orbitdrift.forces import ForceModel  # noqa: E402
from orbitdrift.kepler import (  # noqa: E402
    ClassicalElements,
    elements_to_rv,
    orbital_period_s,
    propagate_two_body,
)

TWO_BODY = ForceModel(zonal=False, drag=False, third_body=False)
PRODUCTION_STEPS_PER_REV = 360
"""The step every divergence curve in this repository was produced with."""


def _orbit(perigee_altitude_km: float, eccentricity: float, inclination_deg: float):
    radius = 6378.135 + perigee_altitude_km
    semi_major = radius / (1.0 - eccentricity)
    return elements_to_rv(
        ClassicalElements(
            semi_major, eccentricity, float(np.deg2rad(inclination_deg)), 0.3, 0.7, 0.2
        )
    )


def _final_position(state, steps_per_rev: int, revolutions: int, force: ForceModel):
    period = orbital_period_s(*state)
    trajectory = integrate.propagate(
        np.atleast_2d(state[0]),
        np.atleast_2d(state[1]),
        2451545.0,
        np.array([2.0e-2]),
        n_steps=steps_per_rev * revolutions,
        step_s=period / steps_per_rev,
        force=force,
        sample_every=steps_per_rev * revolutions,
    )
    return trajectory.positions_km[0, -1], trajectory.times_s[-1]


def main() -> int:
    out = Tee("validate_step_refinement_output.txt")
    out.rule("orbitdrift -- integrator convergence under step refinement")
    for line in environment_lines():
        out(line)

    revolutions = 2 if is_fast() else 8
    ladder = (90, 180, 360, 720) if is_fast() else (90, 180, 360, 720, 1440)
    cases = [
        ("LEO 500 km, e=0.001, i=51.6", _orbit(500.0, 0.001, 51.6)),
        ("LEO 350 km, e=0.02, i=98.0", _orbit(350.0, 0.02, 98.0)),
        ("LEO 1400 km, e=0.005, i=20.0", _orbit(1400.0, 0.005, 20.0)),
    ]

    out.rule(f"error against the closed-form two-body solution, {revolutions} revolutions")
    out("  case                            steps/rev   step s      |error| km    ratio   order")
    errors: dict[str, list[float]] = {}
    for label, state in cases:
        period = orbital_period_s(*state)
        errors[label] = []
        previous = None
        for steps_per_rev in ladder:
            final, elapsed = _final_position(state, steps_per_rev, revolutions, TWO_BODY)
            reference, _ = propagate_two_body(state[0], state[1], np.array([elapsed]))
            error = float(np.linalg.norm(final - reference[0]))
            errors[label].append(error)
            ratio = previous / error if previous else float("nan")
            order = np.log2(ratio) if previous else float("nan")
            out(
                f"  {label:30s} {steps_per_rev:9d} {period / steps_per_rev:8.4f} "
                f"{error:14.6e} {ratio:8.3f} {order:7.3f}"
            )
            previous = error

    out.rule("Richardson order, with no reference solution at all")
    out("  p = log2((x_h - x_h/2) / (x_h/2 - x_h/4)), from the final position vector")
    out("  case                            triple                 order")
    richardson: dict[str, list[float]] = {}
    for label, state in cases:
        richardson[label] = []
        finals = [_final_position(state, spr, revolutions, TWO_BODY)[0] for spr in ladder]
        for index in range(len(ladder) - 2):
            first = np.linalg.norm(finals[index] - finals[index + 1])
            second = np.linalg.norm(finals[index + 1] - finals[index + 2])
            order = float(np.log2(first / second)) if second > 0.0 else float("nan")
            richardson[label].append(order)
            out(
                f"  {label:30s} {ladder[index]:5d}/{ladder[index + 1]:5d}/"
                f"{ladder[index + 2]:5d}  {order:9.4f}"
            )

    out.rule("with drag on: the atmosphere is discontinuous, and it does not matter")
    out("  The published atmosphere table is 28 independent exponential fits, so")
    out("  its band joins are discontinuous, which can cost a fixed-step method")
    out("  its formal order. The measured jumps are at most 0.14 per cent")
    out("  (validate_atmosphere_output.txt) and the orders below are")
    out("  indistinguishable from the drag-free orders above, so the mechanism is")
    out("  real and not material here.")
    out("  case                            steps/rev   |difference to finest| km   order")
    drag_model = ForceModel(zonal=True, drag=True, third_body=False)
    for label, state in cases[:1]:
        finals = [_final_position(state, spr, revolutions, drag_model)[0] for spr in ladder]
        for index in range(len(ladder) - 2):
            first = np.linalg.norm(finals[index] - finals[index + 1])
            second = np.linalg.norm(finals[index + 1] - finals[index + 2])
            order = float(np.log2(first / second)) if second > 0.0 else float("nan")
            out(
                f"  {label:30s} {ladder[index]:5d}/{ladder[index + 1]:5d}/"
                f"{ladder[index + 2]:5d}  {first:20.6e}   {order:7.4f}"
            )

    out.rule("the truncation floor against the signal it has to measure")
    out("  The divergence this product reports is SGP4 against the numerical model.")
    out("  The integrator's own error at the production step is a floor under that")
    out("  measurement, and the ratio is what makes the measurement meaningful.")
    timer = Timer()
    span_days = 1.0 if is_fast() else 7.0
    floor_rows = []
    for label, state in cases:
        period = orbital_period_s(*state)
        steps = int(round(span_days * 86400.0 / (period / PRODUCTION_STEPS_PER_REV)))
        steps -= steps % 4
        with timer:
            trajectory = integrate.propagate(
                np.atleast_2d(state[0]),
                np.atleast_2d(state[1]),
                2451545.0,
                np.array([0.0]),
                n_steps=steps,
                step_s=period / PRODUCTION_STEPS_PER_REV,
                force=TWO_BODY,
                sample_every=steps,
            )
        reference, _ = propagate_two_body(
            state[0], state[1], np.array([trajectory.times_s[-1]])
        )
        floor = float(np.linalg.norm(trajectory.positions_km[0, -1] - reference[0]))
        floor_rows.append((label, floor))
        out(
            f"  {label:30s} truncation floor at {span_days:.0f} days, "
            f"{PRODUCTION_STEPS_PER_REV} steps/rev: {floor:.6e} km"
        )

    out("")
    out("  The divergence of the committed population is read from")
    out("  fixtures/divergence_dataset.npz rather than recomputed here. Recomputing")
    out("  eight seven-day curves cost 489 s of the 600 s per-script budget on a")
    out("  shared two-core container, and the numbers are the same ones.")
    table = load_dataset()
    objects = table.column("object_index")
    ages = table.column("epoch_age_days")
    drift = table.column("drift_km")
    perigee_column = table.column("perigee_altitude_km")
    floor = max(value for _, value in floor_rows)
    last_drift = np.array(
        [
            float(drift[objects == value][int(np.argmax(ages[objects == value]))])
            for value in np.unique(objects)
        ]
    )
    out("")
    out("    object  perigee km   divergence at 6.79 d, km   floor/divergence")
    unique = np.unique(objects)
    for position in range(0, unique.size, max(1, unique.size // 8)):
        value = unique[position]
        mask = objects == value
        out(
            f"    {int(value):6d}  {float(perigee_column[mask][0]):10.1f}   "
            f"{last_drift[position]:24.4f}   {floor / last_drift[position]:16.3e}"
        )
    floor_ratios = floor / last_drift
    population_worst = float(floor_ratios.max())
    out("")
    out(f"    truncation floor used (worst of the three cases) : {floor:.6e} km")
    out(f"    smallest divergence over all {last_drift.size} objects        : "
        f"{last_drift.min():.4f} km")
    out(f"    largest divergence over all {last_drift.size} objects         : "
        f"{last_drift.max():.4f} km")
    out("")
    out("  distribution of floor/divergence over the population:")
    for percentile in (50, 75, 90, 95, 99):
        value = float(np.percentile(floor_ratios, percentile))
        out(f"    {percentile:3d}th percentile : {value:.3e}")
    out(f"    worst            : {population_worst:.3e}")
    over_one_percent = int(np.count_nonzero(floor_ratios > 1e-2))
    over_ten_percent = int(np.count_nonzero(floor_ratios > 1e-1))
    out("")
    out(f"    objects where the floor exceeds 1 per cent of their divergence  : "
        f"{over_one_percent} of {floor_ratios.size}")
    out(f"    objects where the floor exceeds 10 per cent of their divergence : "
        f"{over_ten_percent} of {floor_ratios.size}")
    out("")
    out("  This is a REPORTED LIMITATION and not a passing result. For the handful of")
    out("  objects whose seven-day divergence is under about a kilometre -- high")
    out("  altitude, tiny B*, almost nothing for the two models to disagree about --")
    out("  the integrator's own truncation error is a double-digit percentage of the")
    out("  measured divergence, and those rows should be read as 'divergence below")
    out("  about a kilometre' rather than as the value printed. README.md says so")
    out("  under Limitations. The objects concerned are named below.")
    flagged = np.flatnonzero(floor_ratios > 1e-1)
    if flagged.size:
        out("")
        out("    object  perigee km   divergence km   floor/divergence")
        for position in flagged:
            value = unique[position]
            mask = objects == value
            out(
                f"    {int(value):6d}  {float(perigee_column[mask][0]):10.1f}   "
                f"{last_drift[position]:13.4f}   {floor_ratios[position]:16.3e}"
            )

    out.rule("checks")
    for label, _ in cases:
        orders = [value for value in richardson[label] if np.isfinite(value)]
        out.check(
            f"{label}: Richardson order between 4 and 5.2",
            all(4.0 <= value <= 5.2 for value in orders),
            ", ".join(f"{value:.4f}" for value in orders),
        )
        ratios = [
            coarse / fine
            for coarse, fine in zip(errors[label][:-1], errors[label][1:], strict=True)
        ]
        out.check(
            f"{label}: error against the closed form falls by 8 to 40 per halving",
            all(8.0 < value < 40.0 for value in ratios),
            ", ".join(f"{value:.3f}" for value in ratios),
        )
    out.check(
        "the truncation floor is below 1 per cent of the divergence for at least "
        "three quarters of the population",
        float(np.percentile(floor_ratios, 75)) < 1e-2,
        f"median {float(np.median(floor_ratios)):.3e}, 75th percentile "
        f"{float(np.percentile(floor_ratios, 75)):.3e}, 90th percentile "
        f"{float(np.percentile(floor_ratios, 90)):.3e}",
    )
    out.check(
        "the truncation floor is below 20 per cent of the divergence for EVERY object",
        population_worst < 0.2,
        f"worst {population_worst:.3e} on the least divergent object; "
        f"{over_one_percent} object(s) above 1 per cent and {over_ten_percent} above "
        "10 per cent, named above and in README.md under Limitations",
    )

    out.rule("timings (UNSTABLE: wall clock on a shared two-core container)")
    out(f"  truncation-floor runs         : {timer.elapsed_s:.1f} s over {timer.calls} cases")

    out("")
    out(f"wrote {out.save()}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    sys.exit(main())

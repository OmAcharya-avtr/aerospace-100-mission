"""The numerical integrator against the closed-form two-body solution.

This is a check against an independently derived reference, not a comparison
of two implementations of the same expression. The closed form fixes the conic
once from the initial state and then only advances the mean anomaly; it never
evaluates the equations of motion, so a sign error, a wrong power of ``r`` or a
catastrophic cancellation in the force model cannot be shared between the two
sides.

The residual of Kepler's equation is reported separately, because the closed
form's own accuracy has to be established before it can be used as a reference.
"""

from __future__ import annotations

import sys

from _bootstrap import Tee, bootstrap, environment_lines, is_fast

bootstrap()

import numpy as np  # noqa: E402

from orbitdrift import integrate  # noqa: E402
from orbitdrift.constants import WGS72  # noqa: E402
from orbitdrift.forces import ForceModel  # noqa: E402
from orbitdrift.kepler import (  # noqa: E402
    ClassicalElements,
    elements_to_rv,
    mean_to_true_anomaly,
    orbital_period_s,
    propagate_two_body,
    rv_to_elements,
)

TWO_BODY = ForceModel(zonal=False, drag=False, third_body=False)


def main() -> int:
    out = Tee("validate_kepler_reference_output.txt")
    out.rule("orbitdrift -- numerical integrator against the closed-form two-body solution")
    for line in environment_lines():
        out(line)

    out.rule("step 1: the reference's own accuracy")
    out("  Kepler's equation M = E - e sin E, solved by Newton-Raphson, with the")
    out("  solution substituted back into the equation it solved. The residual is")
    out("  the defining equation evaluated independently of the iteration.")
    out("  eccentricity   worst |residual| over 64 mean anomalies")
    worst_overall = 0.0
    for eccentricity in (0.0, 1e-4, 0.01, 0.1, 0.3, 0.6, 0.9, 0.95):
        worst = 0.0
        for mean_anomaly in np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False):
            nu = mean_to_true_anomaly(float(mean_anomaly), eccentricity)
            e_anom = 2.0 * np.arctan2(
                np.sqrt(1.0 - eccentricity) * np.sin(0.5 * nu),
                np.sqrt(1.0 + eccentricity) * np.cos(0.5 * nu),
            )
            residual = (
                e_anom - eccentricity * np.sin(e_anom) - mean_anomaly
            ) % (2.0 * np.pi)
            residual = min(residual, 2.0 * np.pi - residual)
            worst = max(worst, abs(float(residual)))
        worst_overall = max(worst_overall, worst)
        out(f"  {eccentricity:12.5f}   {worst:.6e} rad")
    out.check(
        "Kepler's equation residual below 1e-12 rad for every case",
        worst_overall < 1e-12,
        f"worst {worst_overall:.3e} rad",
    )

    out.rule("step 2: the closed form conserves its invariants exactly")
    out("  No differential equation is integrated, so energy and |h| cannot drift.")
    out("  case                         revolutions   energy spread   |h| spread")
    for label, (perigee, ecc, inc) in (
        ("500 km circular-ish", (500.0, 0.001, 51.6)),
        ("350 x 1200 km", (350.0, 0.06, 98.0)),
        ("eccentric 300 x 20000 km", (300.0, 0.59, 28.0)),
    ):
        radius = WGS72.radius_km + perigee
        semi_major = radius / (1.0 - ecc)
        state = elements_to_rv(
            ClassicalElements(semi_major, ecc, float(np.deg2rad(inc)), 0.3, 0.7, 0.2)
        )
        period = orbital_period_s(*state)
        revolutions = 5
        times = np.linspace(0.0, revolutions * period, 201)
        positions, velocities = propagate_two_body(state[0], state[1], times)
        energy = 0.5 * np.sum(velocities**2, axis=1) - WGS72.mu_km3_s2 / np.linalg.norm(
            positions, axis=1
        )
        momentum = np.linalg.norm(np.cross(positions, velocities), axis=1)
        energy_spread = float(np.ptp(energy) / abs(energy[0]))
        momentum_spread = float(np.ptp(momentum) / momentum[0])
        out(
            f"  {label:28s} {revolutions:11d}   {energy_spread:13.3e}   "
            f"{momentum_spread:10.3e}"
        )
        out.check(
            f"{label}: closed-form invariants flat to 1e-13",
            energy_spread < 1e-13 and momentum_spread < 1e-13,
            f"energy {energy_spread:.3e}, |h| {momentum_spread:.3e}",
        )

    out.rule("step 3: the integrator against that reference")
    out("  RK4 with only the central term, against the closed form at the same times.")
    out("  A fixed step is a poor fit to an eccentric orbit: the perigee passage is")
    out("  fast and the apogee arc is slow, so a step chosen per revolution is far")
    out("  too coarse at perigee. The declared tolerance therefore depends on the")
    out("  eccentricity, and the third case below is in the table to show the cost")
    out("  rather than to pass a loose check. It is why the population in")
    out("  fixtures/population.tle is capped near e = 0.05, which README.md states")
    out("  under Limitations.")
    out("")
    out("  case                         steps/rev  revolutions   worst |dr| km   worst rel"
        "   declared")
    revolutions = 2 if is_fast() else 8
    for label, (perigee, ecc, inc), declared in (
        ("500 km circular-ish", (500.0, 0.001, 51.6), 1e-7),
        ("350 x 1200 km", (350.0, 0.06, 98.0), 1e-7),
        ("eccentric 300 x 20000 km", (300.0, 0.59, 28.0), 2e-6),
    ):
        radius = WGS72.radius_km + perigee
        semi_major = radius / (1.0 - ecc)
        state = elements_to_rv(
            ClassicalElements(semi_major, ecc, float(np.deg2rad(inc)), 0.3, 0.7, 0.2)
        )
        period = orbital_period_s(*state)
        for steps_per_rev in (360, 1440):
            trajectory = integrate.propagate(
                np.atleast_2d(state[0]),
                np.atleast_2d(state[1]),
                2451545.0,
                np.array([0.0]),
                n_steps=steps_per_rev * revolutions,
                step_s=period / steps_per_rev,
                force=TWO_BODY,
                sample_every=max(1, steps_per_rev // 8),
            )
            reference, _ = propagate_two_body(state[0], state[1], trajectory.times_s)
            error = np.linalg.norm(trajectory.positions_km[0] - reference, axis=-1)
            relative = error / np.linalg.norm(reference, axis=-1)
            out(
                f"  {label:28s} {steps_per_rev:9d} {revolutions:12d}   "
                f"{float(error.max()):13.6e}   {float(relative.max()):9.3e}   "
                f"{declared:8.1e}"
            )
            if steps_per_rev == 1440:
                out.check(
                    f"{label} (e={ecc:.3f}): relative error below {declared:.1e} at "
                    "1440 steps/rev",
                    float(relative.max()) < declared,
                    f"{float(relative.max()):.3e}",
                )

    out.rule("step 4: the element round trip")
    out("  State to elements to state, over a grid of conics.")
    worst_position = 0.0
    worst_velocity = 0.0
    count = 0
    rng = np.random.default_rng(97)
    for _ in range(40 if is_fast() else 400):
        semi_major = float(rng.uniform(6700.0, 45000.0))
        ecc = float(rng.uniform(0.0, 0.8))
        if semi_major * (1.0 - ecc) < 6600.0:
            continue
        elements = ClassicalElements(
            semi_major,
            ecc,
            float(rng.uniform(0.0, np.pi)),
            float(rng.uniform(0.0, 2.0 * np.pi)),
            float(rng.uniform(0.0, 2.0 * np.pi)),
            float(rng.uniform(0.0, 2.0 * np.pi)),
        )
        r, v = elements_to_rv(elements)
        back = elements_to_rv(rv_to_elements(r, v))
        worst_position = max(worst_position, float(np.max(np.abs(back[0] - r))))
        worst_velocity = max(worst_velocity, float(np.max(np.abs(back[1] - v))))
        count += 1
    out(f"  conics tested                 : {count}")
    out(f"  worst position closure        : {worst_position:.6e} km")
    out(f"  worst velocity closure        : {worst_velocity:.6e} km/s")
    out.check(
        "element round trip closes to 1e-6 km and 1e-9 km/s",
        worst_position < 1e-6 and worst_velocity < 1e-9,
        f"{worst_position:.3e} km, {worst_velocity:.3e} km/s",
    )

    out.rule("negative control")
    out("  The integrator run with the J2 term enabled must NOT match the two-body")
    out("  closed form. If it did, either the zonal term is not reaching the")
    out("  integrator or the reference is not a two-body reference.")
    state = elements_to_rv(
        ClassicalElements(WGS72.radius_km + 500.0, 0.001, float(np.deg2rad(51.6)), 0.3, 0.7, 0.2)
    )
    period = orbital_period_s(*state)
    trajectory = integrate.propagate(
        np.atleast_2d(state[0]),
        np.atleast_2d(state[1]),
        2451545.0,
        np.array([0.0]),
        n_steps=360 * 8,
        step_s=period / 360.0,
        force=ForceModel(zonal=True, drag=False, third_body=False),
        sample_every=45,
    )
    reference, _ = propagate_two_body(state[0], state[1], trajectory.times_s)
    departure = float(
        np.max(np.linalg.norm(trajectory.positions_km[0] - reference, axis=-1))
    )
    out(f"  worst departure with J2-J4 on : {departure:.4f} km")
    out.check(
        "the J2 run departs from the two-body closed form by more than 100 km",
        departure > 100.0,
        f"{departure:.3f} km against a 1e-3 km pass threshold for the two-body run",
    )

    out("")
    out(f"wrote {out.save()}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    sys.exit(main())

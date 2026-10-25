"""Energy and angular-momentum conservation on the integrator, with and without J2.

The question this script answers is not "are the invariants conserved" but
"which of the three residuals is the integrator and which is the physics". The
discrimination is made without any reference solution: halving the step must
shrink anything numerical and leave anything physical alone.

With only the central term, the specific energy, the angular momentum
magnitude and its polar component are all exact constants of motion. With the
zonal field switched on, the potential is still time-independent and still
axially symmetric about z, so energy and ``h_z`` remain exact constants, but
``|h|`` is **not**: ``J2`` exerts a torque about any equatorial axis, so the
angular momentum vector precesses about z at constant ``h_z`` while its
magnitude oscillates. A measured ``|h|`` drift is therefore the expected
answer and its size is a property of the orbit.

With drag on, the energy must decrease monotonically, which is an
independently derived sign rather than a comparison: the drag acceleration is
anti-parallel to the relative velocity for a positive ``B*``, so its power is
negative.
"""

from __future__ import annotations

import sys

from _bootstrap import Tee, Timer, bootstrap, environment_lines, is_fast

bootstrap()

import numpy as np  # noqa: E402

from orbitdrift import integrate  # noqa: E402
from orbitdrift.forces import ForceModel  # noqa: E402
from orbitdrift.kepler import ClassicalElements, elements_to_rv, orbital_period_s  # noqa: E402

TWO_BODY = ForceModel(zonal=False, drag=False, third_body=False)
ZONAL = ForceModel(zonal=True, drag=False, third_body=False)

ENERGY_TOLERANCE_AT_360 = 6e-9
"""Declared tolerance on the relative energy residual at 360 steps per revolution.

The tolerance is step-dependent, because the residual is: it falls by a factor
of about 32 per halving, measured below, which is ``h^5``. A single number
would be either vacuous at the fine steps or violated at the coarse ones, and
either way would not be a tolerance. :func:`energy_tolerance` scales this one
by ``(360 / steps_per_rev)^5``.

The value itself is three times the largest residual measured at 360 steps per
revolution over the three cases and six revolutions reported below, so it is a
bound with a stated margin rather than a number fitted to the answer.
"""


def energy_tolerance(steps_per_rev: int) -> float:
    """Declared energy-residual tolerance at a given step, from the ``h^5`` scaling."""
    return ENERGY_TOLERANCE_AT_360 * (360.0 / float(steps_per_rev)) ** 5


def _orbit(perigee_altitude_km: float, eccentricity: float, inclination_deg: float):
    radius = 6378.135 + perigee_altitude_km
    semi_major = radius / (1.0 - eccentricity)
    elements = ClassicalElements(
        semi_major, eccentricity, float(np.deg2rad(inclination_deg)), 0.3, 0.7, 0.2
    )
    return elements_to_rv(elements)


def main() -> int:
    out = Tee("validate_conservation_output.txt")
    out.rule("orbitdrift -- conservation of energy and angular momentum")
    for line in environment_lines():
        out(line)

    revolutions = 2 if is_fast() else 6
    steps = (180, 360, 720) if is_fast() else (180, 360, 720, 1440)
    cases = [
        ("LEO 500 km, e=0.001, i=51.6", _orbit(500.0, 0.001, 51.6)),
        ("LEO 350 km, e=0.02, i=98.0", _orbit(350.0, 0.02, 98.0)),
        ("LEO 1400 km, e=0.005, i=20.0", _orbit(1400.0, 0.005, 20.0)),
    ]
    if is_fast():
        cases = cases[:1]

    out.rule(f"two-body only, {revolutions} revolutions")
    out("  all three quantities are exact constants of motion")
    out("  case                            steps/rev   energy      |h|        h_z")
    two_body_energy: dict[str, list[float]] = {}
    for label, state in cases:
        period = orbital_period_s(*state)
        two_body_energy[label] = []
        for steps_per_rev in steps:
            trajectory = integrate.propagate(
                np.atleast_2d(state[0]),
                np.atleast_2d(state[1]),
                2451545.0,
                np.array([0.0]),
                n_steps=steps_per_rev * revolutions,
                step_s=period / steps_per_rev,
                force=TWO_BODY,
                sample_every=max(1, steps_per_rev // 6),
            )
            residuals = integrate.invariant_residuals(trajectory, zonal=False)
            two_body_energy[label].append(float(residuals["energy"][0]))
            out(
                f"  {label:30s} {steps_per_rev:9d}  "
                f"{float(residuals['energy'][0]):.4e} "
                f"{float(residuals['h_magnitude'][0]):.4e} "
                f"{float(residuals['h_z'][0]):.4e}"
            )

    out.rule(f"two-body plus J2, J3, J4, {revolutions} revolutions")
    out("  energy and h_z are exact constants; |h| is NOT, and must not be")
    out("  case                            steps/rev   energy      |h|        h_z")
    zonal_energy: dict[str, list[float]] = {}
    zonal_magnitude: dict[str, list[float]] = {}
    for label, state in cases:
        period = orbital_period_s(*state)
        zonal_energy[label] = []
        zonal_magnitude[label] = []
        for steps_per_rev in steps:
            trajectory = integrate.propagate(
                np.atleast_2d(state[0]),
                np.atleast_2d(state[1]),
                2451545.0,
                np.array([0.0]),
                n_steps=steps_per_rev * revolutions,
                step_s=period / steps_per_rev,
                force=ZONAL,
                sample_every=max(1, steps_per_rev // 6),
            )
            residuals = integrate.invariant_residuals(trajectory, zonal=True)
            zonal_energy[label].append(float(residuals["energy"][0]))
            zonal_magnitude[label].append(float(residuals["h_magnitude"][0]))
            out(
                f"  {label:30s} {steps_per_rev:9d}  "
                f"{float(residuals['energy'][0]):.4e} "
                f"{float(residuals['h_magnitude'][0]):.4e} "
                f"{float(residuals['h_z'][0]):.4e}"
            )

    out.rule("which residual is numerical and which is physical")
    out("  ratio of the residual at one step to the residual at half that step.")
    out("  A numerical residual shrinks; a physical amplitude does not move.")
    out("  case                            quantity      ratios under halving")
    for label, _ in cases:
        energy_ratios = [
            coarse / fine
            for coarse, fine in zip(zonal_energy[label][:-1], zonal_energy[label][1:], strict=True)
        ]
        magnitude_ratios = [
            coarse / fine
            for coarse, fine in zip(
                zonal_magnitude[label][:-1], zonal_magnitude[label][1:], strict=True
            )
        ]
        out(
            f"  {label:30s} energy       "
            + "  ".join(f"{value:8.3f}" for value in energy_ratios)
        )
        out(
            f"  {label:30s} |h|          "
            + "  ".join(f"{value:8.5f}" for value in magnitude_ratios)
        )

    out.rule("drag: the energy must decrease, and the sign is derived not compared")
    timer = Timer()
    low_state = _orbit(300.0, 0.001, 51.6)
    period = orbital_period_s(*low_state)
    with timer:
        trajectory = integrate.propagate(
            np.atleast_2d(low_state[0]),
            np.atleast_2d(low_state[1]),
            2451545.0,
            np.array([2.0e-2]),
            n_steps=360 * (4 if is_fast() else 16),
            step_s=period / 360.0,
            force=ForceModel(drag=True, third_body=False),
            sample_every=60,
        )
    energy = integrate.specific_energy_km2_s2(
        trajectory.positions_km[0], trajectory.velocities_km_s[0], zonal=True
    )
    differences = np.diff(energy)
    out(f"  samples                       : {energy.size}")
    out(f"  energy at start               : {float(energy[0]):.9f} km^2/s^2")
    out(f"  energy at end                 : {float(energy[-1]):.9f} km^2/s^2")
    out(f"  total change                  : {float(energy[-1] - energy[0]):.6e} km^2/s^2")
    out(f"  largest single increase       : {float(differences.max()):.6e} km^2/s^2")
    out.check(
        "energy decreases at every sample with drag on",
        bool(np.all(differences < 0.0)),
        f"max difference {float(differences.max()):.3e}",
    )

    out.rule("declared tolerance, which scales as h^5")
    out("  steps/rev   declared tolerance   worst measured (two-body)   worst (zonal)")
    for index, steps_per_rev in enumerate(steps):
        declared = energy_tolerance(steps_per_rev)
        worst_tb = max(two_body_energy[label][index] for label, _ in cases)
        worst_z = max(zonal_energy[label][index] for label, _ in cases)
        out(
            f"  {steps_per_rev:9d}   {declared:18.4e}   {worst_tb:25.4e}   {worst_z:13.4e}"
        )

    out.rule("checks")
    for index, steps_per_rev in enumerate(steps):
        declared = energy_tolerance(steps_per_rev)
        worst_tb = max(two_body_energy[label][index] for label, _ in cases)
        worst_z = max(zonal_energy[label][index] for label, _ in cases)
        out.check(
            f"two-body energy residual at {steps_per_rev} steps/rev below "
            f"{declared:.3e}",
            worst_tb < declared,
            f"worst {worst_tb:.4e}",
        )
        out.check(
            f"zonal energy residual at {steps_per_rev} steps/rev below {declared:.3e}",
            worst_z < declared,
            f"worst {worst_z:.4e}",
        )
    for label, _ in cases:
        ratios = [
            coarse / fine
            for coarse, fine in zip(zonal_energy[label][:-1], zonal_energy[label][1:], strict=True)
        ]
        out.check(
            f"{label}: energy residual shrinks by 20 to 50 per halving (numerical)",
            all(20.0 < value < 50.0 for value in ratios),
            ", ".join(f"{value:.2f}" for value in ratios),
        )
        magnitudes = zonal_magnitude[label]
        spread = float(np.ptp(magnitudes) / np.mean(magnitudes))
        out.check(
            f"{label}: |h| residual is step-invariant to 1e-3 (physical)",
            spread < 1e-3,
            f"relative spread {spread:.3e} over {len(magnitudes)} step sizes, "
            f"value {magnitudes[0]:.6e}",
        )

    out.rule("negative control, anchored on the true value")
    out("  The guaranteed-firing offset for a check that accepts a quantity within b")
    out("  of a reference is strictly MORE THAN 2b, measured from the true value,")
    out("  because the unperturbed quantity could already sit anywhere in [-b, b].")
    out("  Smaller offsets may or may not fire depending on where it actually sits,")
    out("  and that is the trap: a control built on an offset of b is not a control.")
    out("  Here the unperturbed residual sits near the bottom of the band, so an")
    out("  offset of 2b fires -- which is sufficient but was not guaranteed in")
    out("  advance, and the table below separates the two cases.")
    state = cases[0][1]
    period = orbital_period_s(*state)
    trajectory = integrate.propagate(
        np.atleast_2d(state[0]),
        np.atleast_2d(state[1]),
        2451545.0,
        np.array([0.0]),
        n_steps=360 * 4,
        step_s=period / 360.0,
        force=TWO_BODY,
        sample_every=36,
    )
    tolerance = energy_tolerance(360)
    clean = float(integrate.invariant_residuals(trajectory, zonal=False)["energy"][0])
    energy = integrate.specific_energy_km2_s2(
        trajectory.positions_km, trajectory.velocities_km_s, zonal=False
    )
    out("")
    out(f"  declared tolerance b          : {tolerance:.4e}")
    out(f"  unperturbed residual          : {clean:.4e} ({clean / tolerance:.4f} b)")
    out("")
    out("  offset (x b)   residual     fires?   guaranteed to fire?")
    for factor in (0.5, 1.0, 2.0, 3.0):
        target = abs(float(energy[0, 0])) * factor * tolerance
        speed = float(np.linalg.norm(trajectory.velocities_km_s[0, -1]))
        scale = np.sqrt(1.0 + 2.0 * target / speed**2)
        broken = trajectory.velocities_km_s.copy()
        broken[0, -1] *= scale
        residual = float(
            integrate.invariant_residuals(
                integrate.Trajectory(
                    trajectory.times_s,
                    trajectory.positions_km,
                    broken,
                    trajectory.step_s,
                    trajectory.force_label,
                ),
                zonal=False,
            )["energy"][0]
        )
        fired = residual > tolerance
        guaranteed = factor > 2.0
        out(
            f"  {factor:12.1f}   {residual:.4e}   "
            f"{'yes' if fired else 'no ':6s}   {'yes' if guaranteed else 'no'}"
        )
        if guaranteed:
            out.check(
                f"an offset of {factor:.1f} b fires, as it must",
                fired,
                f"residual {residual:.4e} against tolerance {tolerance:.4e}",
            )
    out("")
    out("  The 0.5 b row is the point of the whole table: it does not fire, so a")
    out("  negative control built on an offset of half the tolerance would have")
    out("  silently passed a check that was not testing anything.")

    out.rule("timings (UNSTABLE: wall clock on a shared two-core container)")
    out(f"  drag run                      : {timer.elapsed_s:.2f} s")

    out("")
    out(f"wrote {out.save()}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    sys.exit(main())

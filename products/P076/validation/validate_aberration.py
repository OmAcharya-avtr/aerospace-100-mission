"""Cross-check the four aberration routes against each other and against theory.

**Deviation from the batch specification, recorded here and in VALIDATION.md.**
The specification for P076 asks for the aberration terms to be cross-checked
against ``astropy``'s independent implementation to a stated tolerance.
``astropy`` is **not installed** in the environment this build was measured in
and installing it was not permitted, so no number anywhere in this repository
comes from ``astropy``. The header block of every output file in this directory
records ``astropy : not installed`` as measured by
``importlib.metadata.version``, not asserted.

The substitute is three independent exact implementations of the relativistic
aberration transformation, derived by three different routes:

* ``angle_relation``  -- the trigonometric relation
  ``cos(theta') = (cos(theta) - beta) / (1 - beta cos(theta))``.
* ``momentum_boost``  -- a Lorentz boost of the photon four-momentum.
* ``event_worldline`` -- a Lorentz transformation of the spacetime coordinates
  of two events on the photon worldline.

They are checked pairwise, and against four closed-form properties that follow
from the transformation itself rather than from any implementation of it:
collinear invariance, the ``arcsin(beta)`` transverse displacement, the inverse
boost, and the velocity-addition-free composition along a single axis.
"""

from __future__ import annotations

import numpy as np

from _bootstrap import Tee, bootstrap, header

bootstrap()

from pointahead.aberration import (  # noqa: E402
    ABERRATION_ROUTES,
    EXACT_ROUTES,
    aberrate,
    aberration_angle,
    lorentz_gamma,
    route_disagreement,
)
from pointahead.geometry import angle_between, norm, unit  # noqa: E402


def main() -> int:
    tee = Tee("validate_aberration_output.txt")
    header(tee, "aberration: four routes cross-checked, no astropy in this environment")
    tee("")
    tee("  astropy is named in the README alternatives table as the correct tool for")
    tee("  aberration itself. It is not installed here and no number below came from")
    tee("  it. The three exact routes replace that cross-check.")

    rng = np.random.default_rng(76076)

    tee.rule("1. pairwise disagreement between the three exact routes")
    worst_pair: dict[str, float] = {}
    worst_first_order = 0.0
    samples = 2000
    for _ in range(samples):
        direction = unit(rng.normal(size=3))
        beta = rng.normal(size=3)
        beta = beta / norm(beta) * float(rng.uniform(1.0e-9, 0.95))
        for name, value in route_disagreement(direction, beta).items():
            if "first_order" in name:
                worst_first_order = max(worst_first_order, value)
            else:
                worst_pair[name] = max(worst_pair.get(name, 0.0), value)
    tee(f"  samples                            : {samples}")
    tee("  |beta| sampled in                  : [1e-09, 0.95]")
    for name, value in sorted(worst_pair.items()):
        tee(f"  worst {name:<34s}: {value:.6e} rad")
    tee(f"  worst first_order vs exact         : {worst_first_order:.6e} rad")
    tolerance = 1.0e-13
    tee.check(
        "exact routes agree to 1e-13 rad over 2000 random directions and boosts",
        max(worst_pair.values()) < tolerance,
        f"worst {max(worst_pair.values()):.6e} rad, tolerance {tolerance:.1e} rad",
    )

    tee.rule("2. orbital-regime disagreement, the regime this package is for")
    orbital_worst: dict[str, float] = {}
    for _ in range(2000):
        direction = unit(rng.normal(size=3))
        beta = rng.normal(size=3)
        beta = beta / norm(beta) * float(rng.uniform(1.0e-6, 1.0e-4))
        for name, value in route_disagreement(direction, beta).items():
            orbital_worst[name] = max(orbital_worst.get(name, 0.0), value)
    for name, value in sorted(orbital_worst.items()):
        tee(f"  worst {name:<34s}: {value:.6e} rad")
    exact_worst = max(v for k, v in orbital_worst.items() if "first_order" not in k)
    tee.check(
        "exact routes agree to 1e-15 rad for |beta| in [1e-06, 1e-04]",
        exact_worst < 1.0e-15,
        f"worst {exact_worst:.6e} rad",
    )

    tee.rule("3. closed-form properties of the transformation itself")
    tee("  3a. a direction collinear with the boost is unchanged")
    collinear = max(
        aberration_angle(sign * np.array([1.0, 0.0, 0.0]), [magnitude, 0.0, 0.0], route)
        for sign in (1.0, -1.0)
        for magnitude in (1.0e-6, 0.1, 0.9)
        for route in EXACT_ROUTES
    )
    tee(f"      worst displacement             : {collinear:.6e} rad")
    tee.check("collinear directions are invariant", collinear < 1.0e-16)

    tee("")
    tee("  3b. a transverse direction is displaced by exactly arcsin(beta)")
    worst_transverse = 0.0
    for magnitude in (1.0e-6, 1.0e-4, 1.0e-2, 0.1, 0.5, 0.9):
        expected = float(np.arcsin(magnitude))
        for route in EXACT_ROUTES:
            measured = aberration_angle([0.0, 1.0, 0.0], [magnitude, 0.0, 0.0], route)
            worst_transverse = max(worst_transverse, abs(measured - expected) / expected)
        tee(
            f"      beta={magnitude:<8g} arcsin(beta)={expected:.12e} "
            f"measured={measured:.12e}"
        )
    tee(f"      worst relative difference      : {worst_transverse:.6e}")
    tee.check("transverse displacement is arcsin(beta) to 1e-14", worst_transverse < 1.0e-14)

    tee("")
    tee("  3c. the inverse boost recovers the original direction")
    worst_inverse = 0.0
    for _ in range(500):
        direction = unit(rng.normal(size=3))
        beta = rng.normal(size=3)
        beta = beta / norm(beta) * float(rng.uniform(1.0e-6, 0.9))
        for route in EXACT_ROUTES:
            recovered = aberrate(aberrate(direction, beta, route), -beta, route)
            worst_inverse = max(worst_inverse, angle_between(direction, recovered))
    tee(f"      worst residual                 : {worst_inverse:.6e} rad")
    tee.check("inverse boost round-trips to 1e-14 rad", worst_inverse < 1.0e-14)

    tee("")
    tee("  3d. two collinear boosts compose by relativistic velocity addition")
    tee("      beta_total = (b1 + b2) / (1 + b1 b2)")
    worst_compose = 0.0
    for b1, b2 in ((0.1, 0.2), (0.3, 0.4), (0.01, 0.9), (1.0e-5, 1.0e-5)):
        combined = (b1 + b2) / (1.0 + b1 * b2)
        for route in EXACT_ROUTES:
            stepwise = aberrate(
                aberrate([0.0, 1.0, 0.0], [b1, 0.0, 0.0], route), [b2, 0.0, 0.0], route
            )
            single = aberrate([0.0, 1.0, 0.0], [combined, 0.0, 0.0], route)
            worst_compose = max(worst_compose, angle_between(stepwise, single))
        tee(
            f"      b1={b1:<8g} b2={b2:<8g} combined={combined:.12f} "
            f"residual={angle_between(stepwise, single):.6e} rad"
        )
    tee.check("collinear composition matches velocity addition to 1e-14 rad",
              worst_compose < 1.0e-14)

    tee.rule("4. the first-order form: what the approximation costs")
    tee("  transverse direction (the second-order term vanishes, error is beta^3 / 2)")
    for magnitude in (1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1):
        beta = np.array([magnitude, 0.0, 0.0])
        error = angle_between(
            aberrate([0.0, 1.0, 0.0], beta, "first_order"),
            aberrate([0.0, 1.0, 0.0], beta, "momentum_boost"),
        )
        tee(
            f"    beta={magnitude:<10g} error={error:.6e} rad  "
            f"beta^3/2={0.5 * magnitude**3:.6e}  ratio={error / (0.5 * magnitude**3):.6f}"
        )
    tee("")
    tee("  oblique direction at 45 degrees to the boost (error is O(beta^2))")
    oblique = unit([1.0, 1.0, 0.0])
    for magnitude in (1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1):
        beta = np.array([magnitude, 0.0, 0.0])
        error = angle_between(
            aberrate(oblique, beta, "first_order"), aberrate(oblique, beta, "momentum_boost")
        )
        tee(
            f"    beta={magnitude:<10g} error={error:.6e} rad  "
            f"beta^2/4={0.25 * magnitude**2:.6e}  ratio={error / (0.25 * magnitude**2):.6f}"
        )
    orbital_beta = 7557.865206532812 / 299792458.0
    orbital_error = angle_between(
        aberrate(oblique, [orbital_beta, 0.0, 0.0], "first_order"),
        aberrate(oblique, [orbital_beta, 0.0, 0.0], "momentum_boost"),
    )
    tee("")
    tee(f"  at a 600 km orbital speed, beta     = {orbital_beta:.12e}")
    tee(f"  first-order error, oblique          = {orbital_error:.6e} rad")
    tee.check(
        "first-order aberration costs under 1e-9 rad at orbital speed",
        orbital_error < 1.0e-9,
        f"{orbital_error:.6e} rad",
    )

    tee.rule("5. gamma and the refused regimes")
    for magnitude in (0.0, 0.1, 0.6, 0.9, 0.99):
        tee(f"  gamma({magnitude:<5g})                        : {lorentz_gamma(magnitude):.12f}")
    refusals = 0
    for route in ABERRATION_ROUTES:
        try:
            aberrate([0.0, 1.0, 0.0], [1.0, 0.0, 0.0], route)
        except ValueError:
            refusals += 1
    tee(f"  routes refusing |beta| = 1          : {refusals} of {len(ABERRATION_ROUTES)}")
    tee.check("every route refuses a superluminal frame", refusals == len(ABERRATION_ROUTES))

    tee.summarise_failures()
    tee("")
    tee(f"  written: {tee.save()}")
    return 1 if tee.blocking else 0


if __name__ == "__main__":
    raise SystemExit(main())

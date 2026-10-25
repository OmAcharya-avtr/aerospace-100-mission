"""The luni-solar term against a cancellation-free expansion of the same quantity.

The point-mass third-body acceleration is written as
``mu_b (d/|d|^3 - r_b/|r_b|^3)`` with ``d = r_b - r``. The two terms nearly
cancel for ``|r| << |r_b|``: each is of order ``mu_b/|r_b|^2`` while the result
is of order ``2 mu_b |r| / |r_b|^3``, a relative cancellation of ``|r|/|r_b|``,
about 1e-4 for a low Earth orbit against the Sun. In double precision that
costs about four of sixteen digits.

A property test comparing this function with itself could not find such a loss.
The check here is against the leading term of the series expansion,
``(mu_b/|r_b|^3) (3 (r . rhat_b) rhat_b - r)``, which has no difference of
nearly equal large numbers in it at all. The two agree only to the order of the
neglected term, so the comparison is a bound on the cancellation and not an
equality: if the implemented form lost significantly more precision than the
cancellation ratio predicts, the agreement would be worse than the series
truncation allows and this check would see it.
"""

from __future__ import annotations

import sys

from _bootstrap import Tee, bootstrap, environment_lines

bootstrap()

import numpy as np  # noqa: E402

from orbitdrift.constants import MU_MOON_KM3_S2, MU_SUN_KM3_S2  # noqa: E402
from orbitdrift.forces import third_body_acceleration_km_s2  # noqa: E402
from orbitdrift.frames import moon_position_teme_km, sun_position_teme_km  # noqa: E402

FAR = np.array([1e30, 1e30, 1e30])
"""A body so distant that its contribution underflows, used to isolate one body."""


def _series(r: np.ndarray, body: np.ndarray, mu_body: float) -> np.ndarray:
    """Leading term of the expansion, with no cancellation in it."""
    distance = float(np.linalg.norm(body))
    unit = body / distance
    return (mu_body / distance**3) * (3.0 * float(np.dot(r, unit)) * unit - r)


def main() -> int:
    out = Tee("validate_third_body_output.txt")
    out.rule("orbitdrift -- the luni-solar term against a cancellation-free expansion")
    for line in environment_lines():
        out(line)

    jd = 2451545.0
    sun = sun_position_teme_km(jd)
    moon = moon_position_teme_km(jd)

    out.rule("the cancellation, quantified")
    for name, body, mu_body in (("Sun", sun, MU_SUN_KM3_S2), ("Moon", moon, MU_MOON_KM3_S2)):
        distance = float(np.linalg.norm(body))
        for radius in (6800.0, 42164.0):
            direct = mu_body / distance**2
            result = 2.0 * mu_body * radius / distance**3
            out(
                f"  {name:4s} at |r| = {radius:8.0f} km: each term ~ {direct:.3e}, "
                f"result ~ {result:.3e}, ratio {result / direct:.3e}"
            )
    out("")
    out("  The ratio is the fraction of the leading digits that survives the")
    out("  subtraction. A ratio of 1e-4 costs four decimal digits of the sixteen")
    out("  double precision carries, leaving twelve, which is ample.")

    out.rule("implemented form against the series, by body and radius")
    out("  The leading term of the expansion is first order in |r|/|r_b| and the")
    out("  next term is second order, so the leading term's RELATIVE error is first")
    out("  order: eps = |r|/|r_b|. The measured difference must therefore track eps,")
    out("  with a coefficient of order one that depends on the geometry.")
    out("")
    out("  body  |r| km    relative difference   eps = |r|/|r_b|   measured/eps")
    worst_rows = []
    for name, body, mu_body in (("Sun", sun, MU_SUN_KM3_S2), ("Moon", moon, MU_MOON_KM3_S2)):
        distance = float(np.linalg.norm(body))
        for radius in (6800.0, 10000.0, 26600.0, 42164.0):
            rng = np.random.default_rng(int(radius))
            worst = 0.0
            for _ in range(200):
                direction = rng.normal(0.0, 1.0, 3)
                direction /= np.linalg.norm(direction)
                r = radius * direction
                bodies = (body, FAR) if name == "Sun" else (FAR, body)
                measured = third_body_acceleration_km_s2(r, jd, bodies)
                series = _series(r, body, mu_body)
                worst = max(
                    worst,
                    float(np.linalg.norm(measured - series) / np.linalg.norm(series)),
                )
            predicted = radius / distance
            worst_rows.append((name, radius, worst, predicted))
            out(
                f"  {name:4s}  {radius:8.0f}   {worst:19.6e}   {predicted:15.6e}   "
                f"{worst / predicted:12.4f}"
            )

    out("")
    out("  The check is two-sided and that matters. A difference much LARGER than")
    out("  eps would mean the implemented form is losing more precision than the")
    out("  cancellation accounts for. A difference much SMALLER than eps would mean")
    out("  the series is not actually being truncated, which would make this a")
    out("  comparison of two identical expressions and no check at all.")
    for name, radius, worst, predicted in worst_rows:
        ratio = worst / predicted
        out.check(
            f"{name} at {radius:.0f} km: difference is between 0.5 and 3 times eps",
            0.5 < ratio < 3.0,
            f"measured {worst:.3e}, eps {predicted:.3e}, ratio {ratio:.4f}",
        )

    out.rule("exact cancellation at the geocentre")
    at_origin = third_body_acceleration_km_s2(np.zeros(3), jd)
    out(f"  |a| at r = 0                  : {float(np.linalg.norm(at_origin)):.3e} km/s^2")
    out.check(
        "the direct and indirect terms cancel exactly at the geocentre",
        float(np.linalg.norm(at_origin)) < 1e-18,
        f"{float(np.linalg.norm(at_origin)):.3e} km/s^2",
    )

    out.rule("magnitudes, for the record")
    out("  |r| km    |a_sun| km/s^2    |a_moon| km/s^2    two-body km/s^2")
    for radius in (6800.0, 26600.0, 42164.0):
        r = np.array([radius, 0.0, 0.0])
        solar = float(np.linalg.norm(third_body_acceleration_km_s2(r, jd, (sun, FAR))))
        lunar = float(np.linalg.norm(third_body_acceleration_km_s2(r, jd, (FAR, moon))))
        central = 398600.8 / radius**2
        out(f"  {radius:8.0f}   {solar:14.6e}   {lunar:15.6e}   {central:15.6e}")
    out("")
    out("  At low Earth orbit the luni-solar terms are of order 1e-9 km/s^2 against")
    out("  a central term of 8e-3, so they are a 1e-7 perturbation. They are")
    out("  included because the divergence this product measures is itself a small")
    out("  difference, not because they dominate anything.")

    out.rule("negative control")
    out("  The series expansion with the sign of the quadrupole term flipped must")
    out("  NOT agree, otherwise the comparison above is insensitive to the term it")
    out("  is supposed to be checking.")
    r = np.array([6800.0, 0.0, 0.0])
    distance = float(np.linalg.norm(sun))
    unit = sun / distance
    wrong = (MU_SUN_KM3_S2 / distance**3) * (-3.0 * float(np.dot(r, unit)) * unit - r)
    measured = third_body_acceleration_km_s2(r, jd, (sun, FAR))
    departure = float(np.linalg.norm(measured - wrong) / np.linalg.norm(measured))
    out(f"  relative difference with the sign flipped: {departure:.6f}")
    out.check(
        "the flipped-sign series disagrees by more than 10 per cent (control fires)",
        departure > 0.1,
        f"{departure:.6f}",
    )

    out("")
    out(f"wrote {out.save()}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    sys.exit(main())

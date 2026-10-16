"""The CubeSat orbit-average power budget, and the orbit geometry behind it.

Equations, with sources, units, assumptions and validity range
--------------------------------------------------------------
1. Orbital period of a circular orbit (Kepler's third law):

       T = 2 pi sqrt(r^3 / mu)              [s]

   with r the orbital radius in km and mu = 398600.4418 km^3/s^2 the Earth
   gravitational parameter (the WGS 84 / EGM-96 value, also the value used
   by the SGP4 implementations in wide use). Assumptions: two-body motion,
   circular orbit, point-mass Earth. Validity: a few parts in 1e-3 for low
   Earth orbit over one revolution; J2 changes the nodal period by about
   0.1 per cent at 500 km and is not modelled here.

2. Eclipse fraction under the cylindrical shadow approximation, with the
   Sun in the orbital plane (beta angle zero, the worst case):

       f_eclipse = arcsin(R_E / r) / pi     [dimensionless]

   with R_E = 6378.137 km the WGS 84 equatorial radius. Derivation: the
   shadow is a cylinder of radius R_E behind the Earth; the satellite is in
   it over a true-anomaly arc of half-width arcsin(R_E / r) centred on the
   anti-solar point, which is 2 arcsin(R_E / r) out of 2 pi. Assumptions:
   circular orbit, parallel solar rays, no penumbra, no atmospheric
   refraction, spherical Earth. Validity: beta = 0 only. At non-zero beta
   the fraction is smaller and the expression above is the upper bound; at
   high beta the orbit can be fully sunlit and the expression does not
   apply at all.

3. The rollup itself is the arithmetic of the budgetroll package, documented
   in ``src/budgetroll/aggregate.py`` with its sources.

The orbit altitude and every load in the shipped YAML are illustrative
figures chosen for this example. The 8.40 W allocation is NOT derived here
and is not claimed to be: what the geometry section validates is the
orbit-average framing that an allocation of this kind has to be built on.
"""

from __future__ import annotations

import math

from _support import CASES, Emitter

from budgetroll import (
    linear_rollup_uncertainty,
    load,
    rollup,
    rss_rollup_uncertainty,
    run_checks,
    sensitivity,
    severity_counts,
)

EARTH_RADIUS_KM = 6378.137  # WGS 84 equatorial radius
EARTH_MU_KM3_S2 = 398600.4418  # km^3/s^2
ALTITUDE_KM = 500.0


def circular_period_s(altitude_km: float) -> float:
    """Orbital period of a circular orbit in seconds."""
    r = EARTH_RADIUS_KM + altitude_km
    return 2.0 * math.pi * math.sqrt(r**3 / EARTH_MU_KM3_S2)


def eclipse_fraction(altitude_km: float) -> float:
    """Cylindrical-shadow eclipse fraction at beta = 0, dimensionless."""
    r = EARTH_RADIUS_KM + altitude_km
    return math.asin(EARTH_RADIUS_KM / r) / math.pi


def main() -> int:
    """Report the orbit geometry and the power budget rollup."""
    out = Emitter(__file__)
    out("budgetroll validation: 3U CubeSat orbit-average power budget")
    out("=" * 72)
    out("Educational-grade (validation level 1). Not flight-qualified, not certified,")
    out("not approved for operational aerospace use. Every load below is an")
    out("illustrative figure; the geometry is computed from closed-form expressions")
    out("whose sources are in this script's docstring.")
    out()

    out("1. Orbit geometry for the stated 500 km circular orbit")
    period = circular_period_s(ALTITUDE_KM)
    fraction = eclipse_fraction(ALTITUDE_KM)
    out(f"   orbital radius                   {EARTH_RADIUS_KM + ALTITUDE_KM:.3f} km")
    out(f"   period                           {period:.3f} s = {period / 60.0:.4f} min")
    out(f"   eclipse fraction at beta = 0     {fraction:.6f}")
    out(f"   eclipse duration                 {fraction * period / 60.0:.4f} min")
    out(f"   sunlit duration                  {(1 - fraction) * period / 60.0:.4f} min")
    out.check(
        "period is between 94 and 95 minutes, as expected at 500 km",
        94.0 <= period / 60.0 <= 95.0,
        f"{period / 60.0:.4f} min",
    )
    out.check(
        "eclipse fraction at beta = 0 lies in the 0.35 to 0.40 band for low Earth orbit",
        0.35 <= fraction <= 0.40,
        f"{fraction:.6f}",
    )
    # Independent cross-check: the same fraction from the half-angle geometry.
    half_angle = math.asin(EARTH_RADIUS_KM / (EARTH_RADIUS_KM + ALTITUDE_KM))
    direct = 2.0 * half_angle / (2.0 * math.pi)
    out.check(
        "the eclipse fraction agrees with the arc-length form to 1e-12",
        out.close(fraction, direct, 1.0e-12),
        f"difference {fraction - direct:+.3e}",
    )
    out()

    budget = load(CASES / "cubesat_power.yaml")
    roll = rollup(budget)
    root = roll.root

    out("2. Rollup against the hand arithmetic")
    out(f"   nominal                          {root.nominal:.4f} {budget.unit}")
    out(f"   contingency allowance            {root.allowance:.4f} {budget.unit}")
    out(f"   predicted                        {root.predicted:.4f} {budget.unit}")
    out(f"   allocation                       {root.allocation:.4f} {budget.unit}")
    out(f"   margin                           {root.margin_absolute:.4f} {budget.unit}")
    out(f"   margin fraction                  {100.0 * root.margin_fraction:.4f} %")
    out(
        f"   margin at k = {budget.k_sigma:g} sigma            "
        f"{root.margin_at_k:.6f} {budget.unit}"
    )
    for label, value, expected in (
        ("nominal is 7.1500 W", root.nominal, 7.1500),
        ("allowance is 0.7525 W", root.allowance, 0.7525),
        ("predicted is 7.9025 W", root.predicted, 7.9025),
        ("margin is 0.4975 W", root.margin_absolute, 0.4975),
    ):
        out.check(label, out.close(value, expected, 1.0e-9), f"{value:.10f}")
    out.check(
        "margin at 2 sigma is negative",
        root.margin_at_k < 0.0,
        f"{root.margin_at_k:.6f} {budget.unit}",
    )
    out()

    out("3. The four combination rules at the root")
    linear = linear_rollup_uncertainty(budget)
    rss = rss_rollup_uncertainty(budget)
    out(f"   linear (fully correlated)        {linear:.6f} {budget.unit}")
    out(f"   root-sum-square (independent)    {rss:.6f} {budget.unit}")
    out(f"   declared hierarchy at the root   {root.uncertainty:.6f} {budget.unit}")
    out(f"   exact with declared correlations {root.exact_uncertainty:.6f} {budget.unit}")
    out.check(
        "RSS does not exceed the linear sum",
        rss <= linear + 1.0e-12,
        f"{rss:.6f} <= {linear:.6f}",
    )
    out.check(
        "flat RSS is sqrt(0.1039) W",
        out.close(rss, math.sqrt(0.1039), 1.0e-12),
        f"{rss:.10f}",
    )
    out.check(
        "the declared hierarchy at the root is sqrt(0.1075) W",
        out.close(root.uncertainty, math.sqrt(0.1075), 1.0e-12),
        f"{root.uncertainty:.10f}",
    )
    out.check(
        "the exact correlated form at the root is sqrt(0.12334) W",
        out.close(root.exact_uncertainty, math.sqrt(0.12334), 1.0e-12),
        f"{root.exact_uncertainty:.10f}",
    )
    understatement = 100.0 * (root.exact_uncertainty - root.uncertainty) / root.exact_uncertainty
    out(f"   RSS understates the exact form by {understatement:.4f} %")
    out()

    out("4. Sensitivity ranking, +10 % on each leaf in turn")
    for row in sensitivity(budget, step=0.10):
        out(
            f"   {row.name[:34]:<36s}{row.predicted:8.4f}  "
            f"value {100 * row.value_share:6.2f} %  "
            f"var {100 * row.variance_share:6.2f} %  "
            f"d margin {row.margin_delta:+8.4f} {budget.unit}"
        )
    top = sensitivity(budget, step=0.10)[0]
    out.check(
        "the transmitter is the leaf that costs the most margin",
        top.id == "radio_tx",
        f"{top.id} at {top.margin_delta:+.4f} {budget.unit}",
    )
    out.check(
        "its margin delta is -0.1760 W, which is 0.10 * 1.60 * 1.10",
        out.close(top.margin_delta, -0.176, 1.0e-12),
        f"{top.margin_delta:.10f}",
    )
    out()

    out("5. Findings")
    findings = run_checks(budget)
    for finding in findings:
        out(f"   {finding}")
    counts = severity_counts(findings)
    out(f"   counts {counts}")
    out.check(
        "the declared correlation is caught: BR001 on the payload node and the root",
        sum(1 for f in findings if f.code == "BR001") == 2,
        str([f.subject for f in findings if f.code == "BR001"]),
    )
    out.check(
        "the two-sigma shortfall is caught by BR004",
        any(f.code == "BR004" for f in findings),
    )
    out.check(
        "the undeclared shared bus rail is caught by BR010",
        any(f.code == "BR010" for f in findings),
    )
    out.check(
        "counts are 4 error, 2 warning, 3 info",
        counts == {"error": 4, "warning": 2, "info": 3},
        str(counts),
    )
    return out.finish()


if __name__ == "__main__":
    raise SystemExit(main())

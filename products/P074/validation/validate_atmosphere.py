"""The declared atmosphere model: its table, its discontinuities, and its limits.

Every drag-driven divergence number in this repository rests on the piecewise
exponential model of Vallado 2013 Table 8-4, which is **static**: no solar
index, no diurnal bulge, no geomagnetic term. This script publishes what that
model actually is, measures the discontinuities at its band joins, and states
the consequence for the integrator.
"""

from __future__ import annotations

import sys

from _bootstrap import Tee, bootstrap, environment_lines

bootstrap()

import numpy as np  # noqa: E402

from orbitdrift import forces  # noqa: E402


def main() -> int:
    out = Tee("validate_atmosphere_output.txt")
    out.rule("orbitdrift -- the declared atmosphere model")
    for line in environment_lines():
        out(line)
    out("")
    out(f"  model : {forces.ATMOSPHERE_MODEL_NAME}")
    out("  source: Vallado 2013, Fundamentals of Astrodynamics and Applications,")
    out("          4th ed., Table 8-4, after CIRA-72 and the US Standard")
    out("          Atmosphere 1976")

    table = forces._ATMOSPHERE_TABLE
    out.rule("the table as implemented")
    out("  base km    rho0 kg/m^3    scale height km")
    for base, rho0, scale in table:
        out(f"  {base:7.1f}    {rho0:.6e}       {scale:11.3f}")

    out.rule("the table is reproduced exactly at each base altitude")
    worst = 0.0
    for base, rho0, _scale in table:
        measured = float(forces.atmospheric_density_kg_m3(base))
        worst = max(worst, abs(measured - rho0) / rho0)
    out(f"  worst relative departure      : {worst:.3e}")
    out.check("every base altitude returns its tabulated density", worst == 0.0, f"{worst:.3e}")

    out.rule("monotonicity")
    altitude = np.linspace(0.0, 1500.0, 150001)
    density = forces.atmospheric_density_kg_m3(altitude)
    increases = int(np.count_nonzero(np.diff(density) >= 0.0))
    out(f"  samples                       : {altitude.size}")
    out(f"  samples where density rises   : {increases}")
    out.check("density falls with height everywhere", increases == 0, f"{increases} rises")

    out.rule("the band joins are discontinuous, and by how much")
    out("  The published table is 28 independent exponential fits, not one smooth")
    out("  function, so the joins do not match exactly. The jumps are MEASURED")
    out("  below rather than assumed, and the measurement changes the conclusion:")
    out("  the largest is about 0.14 per cent, which is small enough that the")
    out("  integrator's measured convergence order with drag on is")
    out("  indistinguishable from its order with drag off (4.6978 / 4.5370 /")
    out("  4.3688 against 4.6990 / 4.5386 / 4.3701 in")
    out("  validate_step_refinement_output.txt). The order loss a discontinuous")
    out("  right-hand side can cause is therefore a real mechanism that is not")
    out("  material here, and saying so is more useful than warning about it.")
    out("")
    out("  join km   density below      density at        relative jump")
    jumps = []
    for base in table[1:, 0]:
        below = float(forces.atmospheric_density_kg_m3(base - 1e-9))
        at = float(forces.atmospheric_density_kg_m3(base))
        jump = (at - below) / below
        jumps.append(abs(jump))
        out(f"  {base:7.1f}   {below:.6e}      {at:.6e}      {jump:+.6f}")
    out("")
    out(f"  largest absolute relative jump: {max(jumps):.6f} at a band join")
    out.check(
        "no band join jumps by more than 1 per cent",
        max(jumps) < 0.01,
        f"largest {max(jumps):.6f}",
    )
    out.check(
        "the joins are genuinely discontinuous (this is a property, not a defect)",
        max(jumps) > 1e-6,
        f"largest {max(jumps):.6f}",
    )

    out.rule("what the static model cannot do")
    out("  Published solar-cycle variation of thermospheric density at 400 km is")
    out("  about an order of magnitude between solar minimum and solar maximum")
    out("  (Vallado 2013 section 8.6.2; Picone et al. 2002 for NRLMSISE-00). This")
    out("  model has no solar index at all, so it sits at one unspecified point in")
    out("  that range. The consequence for this product, stated plainly: a")
    out("  divergence curve driven by drag inherits that factor, and the curve is")
    out("  therefore a measurement of model divergence under a declared static")
    out("  atmosphere and not a prediction of what the drag will actually do.")
    out("")
    out("  altitude km    density kg/m^3    a factor of 10 either way")
    for altitude_km in (300.0, 400.0, 500.0, 600.0, 800.0, 1000.0):
        value = float(forces.atmospheric_density_kg_m3(altitude_km))
        out(f"  {altitude_km:11.1f}    {value:.6e}      {value / 10.0:.3e} to {value * 10.0:.3e}")

    out.rule("extrapolation above the table")
    out("  Above 1000 km the final band is extrapolated. The drag acceleration")
    out("  there is far below the integrator's truncation floor, so the")
    out("  extrapolation cannot affect a reported number; the magnitudes are")
    out("  published so a reader can confirm that rather than take it on trust.")
    out("  altitude km   density kg/m^3   |a_drag| km/s^2 at C_D A/m = 1e-2 m^2/kg")
    for altitude_km in (1000.0, 1500.0, 2000.0, 3000.0):
        value = float(forces.atmospheric_density_kg_m3(altitude_km))
        acceleration = 0.5 * 1e-2 * value * (7.0**2) * 1000.0
        out(f"  {altitude_km:11.1f}   {value:.6e}   {acceleration:.6e}")
    at_1000 = 0.5 * 1e-2 * float(forces.atmospheric_density_kg_m3(1000.0)) * 49.0 * 1000.0
    out("")
    out("  The 1e-2 m^2/kg drag parameter above is a high-area-to-mass object. For")
    out("  the population in fixtures/population.tle the median C_D A/m is nearer")
    out("  1e-3 m^2/kg, a tenth of that.")
    out.check(
        "drag above 1000 km is below 1e-12 km/s^2 even at a 1e-2 m^2/kg drag "
        "parameter, which is five orders below the luni-solar terms' 1e-9 and "
        "eleven below the central term",
        at_1000 < 1e-12,
        f"{at_1000:.3e} km/s^2 at 1000 km",
    )

    out("")
    out(f"wrote {out.save()}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    sys.exit(main())

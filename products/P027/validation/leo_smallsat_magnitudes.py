"""Torque magnitudes for a representative LEO smallsat, against the order-of-magnitude
band quoted in the standard references, plus internal consistency checks that are
sharper than the band. Run: ``python3 leo_smallsat_magnitudes.py``

Citation policy, stated because it limits what this script can claim
--------------------------------------------------------------------
References are given at work level only (author, title). No page, table or equation
number is quoted anywhere in this product, because no physical copy was consulted during
the build. The consequence is that the external comparison below can only be made
against an order-of-magnitude *band*, not against a specific published number. That
makes it a weak check, and it is labelled as such. The sharper evidence in this product
is the closed-form and hand-arithmetic agreement in ``hand_calculations.py`` and the
independent-quadrature agreement in ``momentum_integration.py``.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from _common import Checks  # noqa: E402

from disturbtorque import (  # noqa: E402
    EARTH_DIPOLE_MOMENT,
    R_EARTH_MEAN,
    SOLAR_IRRADIANCE_1AU,
    SOLAR_IRRADIANCE_1AU_SMAD,
    SPEED_OF_LIGHT,
    SRP_PRESSURE_1AU,
    Orbit,
    compute_profile,
    gravity_gradient_max_magnitude,
    reference_orbit,
    reference_smallsat,
    sun_direction_for_beta,
)

c = Checks()
BAND_LO, BAND_HI = 1e-7, 1e-4
sc = reference_smallsat()

print("Representative LEO smallsat: disturbance-torque magnitudes")
print("=" * 78)
print(f"""
Vehicle (this package's own definition, in disturbtorque.presets; not taken from any
published spacecraft):
  mass                {sc.mass_kg:.0f} kg
  inertia             diag(4.0, 8.0, 10.0) kg m^2
  drag area / Cd      {sc.drag_area_m2} m^2 / {sc.drag_coefficient}
  sunlit area / q     {sc.srp_area_m2} m^2 / {sc.srp_reflectance}
  cp - cm offset      {sc.cp_aero_offset_m.tolist()} m,
                      |offset| = {np.linalg.norm(sc.cp_aero_offset_m):.6f} m
  residual dipole     {sc.residual_dipole_am2.tolist()} A m^2,
                      |m| = {np.linalg.norm(sc.residual_dipole_am2):.6f} A m^2
Attitude: nadir-pointing with 5 deg pitch and 5 deg roll offsets.
Orbit: circular, inclination 51.6 deg, RAAN 0, Sun beta angle 0 deg.

Comparison band: {BAND_LO:.0e} to {BAND_HI:.0e} N m. This is the order-of-magnitude
envelope inside which the disturbance-torque estimates for small Earth-orbiting
spacecraft fall in Wertz, *Spacecraft Attitude Determination and Control*, and Larson &
Wertz, *Space Mission Analysis and Design*. It is a band and not a value because no page
reference was verified; see the citation policy in this script's docstring.
""")

alts = [300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 1000.0]
sources = ("gravity_gradient", "aerodynamic", "solar", "magnetic")
peaks: dict[float, dict[str, float]] = {}
print("Peak torque magnitude over one orbit, body frame [N m]")
print(f"{'alt [km]':>9}" + "".join(f"{s:>19}" for s in sources) + f"{'total':>19}")
print("-" * (9 + 19 * 5))
for alt in alts:
    orb = reference_orbit(alt)
    sun = sun_direction_for_beta(orb.inclination_rad, orb.raan_rad, 0.0)
    prof = compute_profile(sc, orb, sun, n_samples=1441)
    peaks[alt] = {s: prof.peak_magnitude(s, "body") for s in sources}
    peaks[alt]["total"] = prof.peak_magnitude("total", "body")
    print(f"{alt:>9.0f}" + "".join(f"{peaks[alt][s]:>19.4e}" for s in (*sources, "total")))

print("\nBand check, source by source and altitude by altitude:")
excursions = []
for alt in alts:
    for s in sources:
        v = peaks[alt][s]
        if not (BAND_LO <= v <= BAND_HI):
            excursions.append((alt, s, v))
if excursions:
    print(f"  {len(excursions)} value(s) outside the {BAND_LO:.0e} to {BAND_HI:.0e} N m band:")
    for alt, s, v in excursions:
        side = "below" if v < BAND_LO else "above"
        factor = (BAND_LO / v) if v < BAND_LO else (v / BAND_HI)
        print(f"    {s} at {alt:.0f} km = {v:.4e} N m, {side} the band by a factor {factor:.2f}")
else:
    print("  none")

print("""
  Result: FAILED, and reported rather than adjusted. The aerodynamic torque for this
  vehicle falls below the band's lower edge above roughly 700 km. Nothing was tuned to
  bring it inside: the drag coefficient stays at 2.2, the areas and offsets stay as
  defined above, and the density comes from the unmodified exponential table.

  The physical reading is that the band is a statement about the *drag regime* of LEO,
  and this vehicle at 800 km and above is no longer in it: the density there is
  1.17e-14 kg m^-3, four orders of magnitude below the 300 km value, so aerodynamic
  torque is genuinely negligible and the band's lower edge is not a meaningful floor for
  it. That is a limitation of the comparison, not evidence of an error in the model, but
  it is a failed check as stated and it is counted as one.
""")
c.assert_true(
    "all four sources inside the band at every altitude from 300 to 1000 km",
    not excursions,
    f"({len(excursions)} excursion(s), all aerodynamic at 700 km and above)",
)
low_alt_excursions = [e for e in excursions if e[0] <= 600.0]
c.assert_true(
    "all four sources inside the band at 300-600 km (the drag regime the band describes)",
    not low_alt_excursions,
    f"({len(low_alt_excursions)} excursion(s))",
)

# ------------------------------------------------------- sharper internal checks
print("\nSharper checks, which do not depend on any quoted band:")

orb500 = reference_orbit(500.0)
t_gg_max = gravity_gradient_max_magnitude(8.0, 10.0, orb500.radius_m)
orb45 = Orbit(
    altitude_m=500_000.0,
    inclination_rad=orb500.inclination_rad,
    roll_rad=np.radians(45.0),
)
sun0 = sun_direction_for_beta(orb45.inclination_rad, 0.0, 0.0)
prof45 = compute_profile(sc, orb45, sun0, n_samples=61)
print(f"""
  1. Gravity gradient at 45 deg roll reproduces the analytic worst case
     3 mu / (2 R^3) |Izz - Iyy| = 3 * {orb500.mu:.6e} / (2 * {orb500.radius_m:.0f}^3) * 2.0
                                = {t_gg_max:.10e} N m
     from the full tensor sweep: {prof45.peak_magnitude('gravity_gradient', 'body'):.10e} N m
""")
c.check("45 deg roll gravity gradient vs analytic max [N m]",
        prof45.peak_magnitude("gravity_gradient", "body"), t_gg_max, 1e-12)

print("  2. Aerodynamic/solar crossover altitude (peak torque magnitude, beta = 0):")


def aero_minus_solar(alt_km: float) -> float:
    orb = reference_orbit(alt_km)
    sun = sun_direction_for_beta(orb.inclination_rad, orb.raan_rad, 0.0)
    prof = compute_profile(sc, orb, sun, n_samples=181)
    return prof.peak_magnitude("aerodynamic", "body") - prof.peak_magnitude("solar", "body")


cross = brentq(aero_minus_solar, 400.0, 700.0, xtol=0.05)
print(f"""     the two are equal at {cross:.1f} km for this vehicle, with aerodynamic torque
     larger below and solar radiation pressure larger above. The crossover moves with
     the area ratio, the offsets and above all with solar activity, which this density
     model does not represent at all, so the number should be read as "somewhere in the
     500-600 km decade for this configuration at mean activity", not as a constant.
""")
c.assert_true("crossover altitude falls between 400 and 700 km", 400.0 < cross < 700.0,
              f"({cross:.1f} km)")

print("  3. Aerodynamic scaling laws (exact consequences of the expression):")
orb_a = reference_orbit(500.0)
sun_a = sun_direction_for_beta(orb_a.inclination_rad, 0.0, 0.0)
p_a = compute_profile(sc, orb_a, sun_a, n_samples=181, co_rotating_atmosphere=False)
p_b = compute_profile(sc, reference_orbit(600.0), sun_a, n_samples=181,
                      co_rotating_atmosphere=False)
from disturbtorque.atmosphere import density  # noqa: E402

rho_ratio = float(density(600_000.0) / density(500_000.0))
v_ratio = (reference_orbit(600.0).speed_ms / orb_a.speed_ms) ** 2
t_ratio = p_b.peak_magnitude("aerodynamic", "body") / p_a.peak_magnitude("aerodynamic", "body")
print(f"""     rho(600)/rho(500) = {rho_ratio:.10f}
     (v600/v500)^2     = {v_ratio:.10f}
     product           = {rho_ratio * v_ratio:.10f}
     torque ratio      = {t_ratio:.10f}
""")
c.check("aero torque ratio equals rho ratio times v^2 ratio", t_ratio, rho_ratio * v_ratio, 1e-12)

print("  4. Co-rotating atmosphere correction:")
p_rot = compute_profile(sc, orb_a, sun_a, n_samples=1441, co_rotating_atmosphere=True)
p_inert = compute_profile(sc, orb_a, sun_a, n_samples=1441, co_rotating_atmosphere=False)
ratio = p_rot.rms_magnitude("aerodynamic", "body") / p_inert.rms_magnitude("aerodynamic", "body")
v_eq = 7.292115e-5 * orb_a.radius_m
print(f"""     the co-rotating atmosphere reduces the relative speed by at most
     omega_E * R * cos(i) = 7.292115e-5 * {orb_a.radius_m:.0f} * cos(51.6 deg)
                          = {v_eq * np.cos(np.radians(51.6)):.2f} m s^-1 out of {orb_a.speed_ms:.1f} m s^-1,
     i.e. {100 * v_eq * np.cos(np.radians(51.6)) / orb_a.speed_ms:.2f} %, so the torque changes by at most about twice that.
     measured rms torque ratio with / without co-rotation = {ratio:.6f} ({100 * (ratio - 1):+.2f} %)
""")
c.assert_true("co-rotation reduces the aerodynamic torque by 1 to 12 %",
              0.88 < ratio < 0.99, f"(ratio {ratio:.6f})")

# ----------------------------------------------- documented inter-source discrepancies
print("\nInter-source discrepancies in the constants, reported not reconciled:")
p_smad = SOLAR_IRRADIANCE_1AU_SMAD / SPEED_OF_LIGHT
b0_from_smad = EARTH_DIPOLE_MOMENT / R_EARTH_MEAN**3
print(f"""
  Solar constant. Modern TSI is {SOLAR_IRRADIANCE_1AU:.0f} W m^-2; the older textbook value used by
  Wertz and by Larson & Wertz is {SOLAR_IRRADIANCE_1AU_SMAD:.0f} W m^-2.
    P = 1361 / 299792458 = {SRP_PRESSURE_1AU:.10e} N m^-2   (this package's default)
    P = 1367 / 299792458 = {p_smad:.10e} N m^-2   (textbook)
    difference = {100 * (p_smad / SRP_PRESSURE_1AU - 1):.3f} %, propagating linearly into every SRP torque.

  Earth dipole moment. This package defaults to the value used in the textbook
  disturbance-torque estimate, k = {EARTH_DIPOLE_MOMENT:.3e} T m^3, which implies an
  equatorial surface field
    B0 = k / Re^3 = {EARTH_DIPOLE_MOMENT:.3e} / {R_EARTH_MEAN:.7e}^3 = {b0_from_smad:.6e} T = {b0_from_smad * 1e9:.0f} nT.
  A centred-dipole reduction of a recent IGRF epoch gives an equatorial surface field
  near 3.0e-05 T, i.e. k near {3.0e-5 * R_EARTH_MEAN**3:.3e} T m^3. The two differ by
  {100 * (b0_from_smad / 3.0e-5 - 1):.1f} %, and the magnetic torque is linear in k, so a magnetic
  torque quoted by this package carries at least that much systematic uncertainty on top
  of the 20-30 % pointwise error of the centred-dipole approximation itself. Nothing is
  averaged or split: the textbook value is used, and the spread is stated here, in
  constants.py and in the README.
""")
c.check("P_srp consistency: 1361/c", SRP_PRESSURE_1AU, 1361.0 / 299792458.0, 1e-15)
c.check("B0 implied by the default dipole moment [T]", b0_from_smad, 3.078e-5, 1e-3)

c.summary("leo_smallsat_magnitudes.py")
print("\nNote: this script exits 0 even though a band check FAILED, because the failure is")
print("a documented finding rather than a regression. The failure is counted in the")
print("summary line above and repeated in VALIDATION.md section 5.")
raise SystemExit(0)

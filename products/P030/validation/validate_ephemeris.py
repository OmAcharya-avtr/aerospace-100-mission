"""The low-precision Sun and Moon series against published orbital constants.

These two routines are a convenience so the package can be exercised end to
end; they are not the product. This script measures what can be measured
without an ephemeris file, and states plainly what is *not* checked.

Checks
------
1. Sun: maximum and minimum declination against the mean obliquity of the
   ecliptic, 23.4393 deg at J2000 (IAU 2006 precession; Vallado 2013 Sec. 3.7).
2. Sun: ecliptic latitude, which the algorithm sets to zero by construction.
3. Sun: geocentric distance extremes against the Earth's orbital eccentricity,
   perihelion 0.98329 AU and aphelion 1.01671 AU, and the date of perihelion
   passage, which falls in the first week of January.
4. Sun: the tropical year recovered from successive northward equinox
   crossings, against 365.2422 days.
5. Moon: mean geocentric distance against the published 384 400 km, and the
   perigee/apogee spread.
6. Moon: sidereal period from successive returns in ecliptic longitude,
   against 27.321582 days.
7. Moon: maximum ecliptic latitude against the 5.145 deg inclination of the
   lunar orbit to the ecliptic.

NOT checked: agreement with a numerical ephemeris (DE440/JPL) at any epoch.
No ephemeris file is available in this environment. Any statement about the
absolute accuracy of these series would be unsupported, so none is made.

Run from products/P030/:  python validation/validate_ephemeris.py
"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from keepout import (  # noqa: E402
    ASTRONOMICAL_UNIT_M,
    moon_direction_mod,
    sun_direction_mod,
)

JD_2026_01_01 = 2461041.5
OBLIQUITY_J2000_DEG = 23.4393
PERIHELION_AU = 0.98329
APHELION_AU = 1.01671
TROPICAL_YEAR_D = 365.2422
MOON_MEAN_DISTANCE_M = 3.844e8
MOON_SIDEREAL_PERIOD_D = 27.321582
MOON_INCLINATION_DEG = 5.145


def ecliptic_latitude(direction: np.ndarray, jd: np.ndarray) -> np.ndarray:
    """Ecliptic latitude [deg] from an equator-of-date unit vector."""
    t = (jd - 2451545.0) / 36525.0
    eps = np.radians(23.439291 - 0.0130042 * t)
    z = -np.sin(eps) * direction[..., 1] + np.cos(eps) * direction[..., 2]
    return np.degrees(np.arcsin(np.clip(z, -1.0, 1.0)))


def zero_crossings(t: np.ndarray, y: np.ndarray, rising: bool = True) -> np.ndarray:
    """Linearly interpolated crossings of ``y = 0``."""
    s = np.sign(y)
    idx = np.nonzero((s[:-1] < 0) & (s[1:] >= 0))[0] if rising else (
        np.nonzero((s[:-1] > 0) & (s[1:] <= 0))[0]
    )
    return t[idx] - y[idx] * (t[idx + 1] - t[idx]) / (y[idx + 1] - y[idx])


def main() -> None:
    print("=" * 88)
    print("KeepOut validation: low-precision Sun and Moon series")
    print("=" * 88)
    print("Source: Vallado, Fundamentals of Astrodynamics and Applications, 4th ed.,")
    print("        Algorithm 29 (Sun) and Algorithm 31 (Moon).")
    print("NOT checked here: agreement with a numerical ephemeris (no ephemeris file")
    print("        is available in this environment). No accuracy claim is made.")
    print()

    results = []

    jd = JD_2026_01_01 + np.arange(0.0, 3 * 365.25, 0.01)
    d_sun, r_sun = sun_direction_mod(jd)
    dec = np.degrees(np.arcsin(d_sun[:, 2]))
    print("1. Sun declination extremes vs the obliquity of the ecliptic")
    print(f"   max declination = {dec.max():+.6f} deg")
    print(f"   min declination = {dec.min():+.6f} deg")
    print(f"   obliquity (IAU 2006 at J2000) = {OBLIQUITY_J2000_DEG:+.4f} deg")
    e_max = abs(dec.max() - OBLIQUITY_J2000_DEG)
    e_min = abs(dec.min() + OBLIQUITY_J2000_DEG)
    print(f"   |diff| = {e_max:.6f} deg (max), {e_min:.6f} deg (min)   tolerance 0.01 deg")
    ok = e_max < 0.01 and e_min < 0.01
    results.append(("Sun declination extremes", ok))
    print(f"   {'PASS' if ok else 'FAILED'}\n")

    lat = ecliptic_latitude(d_sun, jd)
    print("2. Sun ecliptic latitude (zero by construction in Algorithm 29)")
    print(f"   max |latitude| = {np.abs(lat).max():.3e} deg   tolerance 1e-9 deg")
    ok = np.abs(lat).max() < 1e-9
    results.append(("Sun ecliptic latitude is zero", ok))
    print(f"   {'PASS' if ok else 'FAILED'}\n")

    r_au = r_sun / ASTRONOMICAL_UNIT_M
    print("3. Sun geocentric distance vs perihelion and aphelion")
    print(f"   minimum = {r_au.min():.6f} AU   published perihelion = {PERIHELION_AU:.5f} AU")
    print(f"   maximum = {r_au.max():.6f} AU   published aphelion   = {APHELION_AU:.5f} AU")
    e_p = abs(r_au.min() - PERIHELION_AU)
    e_a = abs(r_au.max() - APHELION_AU)
    first_year = jd < JD_2026_01_01 + 365.25
    peri_jd = jd[first_year][np.argmin(r_au[first_year])]
    print(f"   |diff| = {e_p:.6f} AU (perihelion), {e_a:.6f} AU (aphelion)"
          f"   tolerance 0.0005 AU")
    print(f"   first perihelion after 2026-01-01 at JD {peri_jd:.4f} "
          f"= {peri_jd - JD_2026_01_01:.3f} days into the year")
    ok = e_p < 5e-4 and e_a < 5e-4 and 0.0 <= peri_jd - JD_2026_01_01 <= 7.0
    results.append(("Sun distance extremes and perihelion date", ok))
    print(f"   {'PASS' if ok else 'FAILED'}\n")

    crossings = zero_crossings(jd, dec, rising=True)
    years = np.diff(crossings)
    print("4. Tropical year from successive northward equinox crossings")
    print(f"   crossings at JD {np.array2string(crossings, precision=4)}")
    print(f"   intervals [d] = {np.array2string(years, precision=6)}")
    print(f"   mean = {years.mean():.6f} d   published tropical year = "
          f"{TROPICAL_YEAR_D:.4f} d")
    e_y = abs(years.mean() - TROPICAL_YEAR_D)
    print(f"   |diff| = {e_y:.6f} d   tolerance 0.01 d")
    ok = e_y < 0.01
    results.append(("Tropical year", ok))
    print(f"   {'PASS' if ok else 'FAILED'}\n")

    jd_m = JD_2026_01_01 + np.arange(0.0, 3 * 365.25, 0.005)
    d_moon, r_moon = moon_direction_mod(jd_m)
    print("5. Moon geocentric distance")
    print(f"   mean    = {r_moon.mean() / 1e3:.3f} km   published mean = "
          f"{MOON_MEAN_DISTANCE_M / 1e3:.1f} km")
    print(f"   minimum = {r_moon.min() / 1e3:.3f} km")
    print(f"   maximum = {r_moon.max() / 1e3:.3f} km")
    rel = abs(r_moon.mean() - MOON_MEAN_DISTANCE_M) / MOON_MEAN_DISTANCE_M
    print(f"   relative |diff| of the mean = {rel:.3e}   tolerance 1 %")
    ok = rel < 0.01
    results.append(("Moon mean distance", ok))
    print(f"   {'PASS' if ok else 'FAILED'}\n")

    t_m = (jd_m - 2451545.0) / 36525.0
    eps_m = np.radians(23.439291 - 0.0130042 * t_m)
    y_ecl = np.cos(eps_m) * d_moon[:, 1] + np.sin(eps_m) * d_moon[:, 2]
    lam = np.unwrap(np.arctan2(y_ecl, d_moon[:, 0]))
    n_rev = (lam[-1] - lam[0]) / (2 * np.pi)
    period = (jd_m[-1] - jd_m[0]) / n_rev
    print("6. Moon sidereal period from the ecliptic-longitude advance")
    print(f"   revolutions over {jd_m[-1] - jd_m[0]:.2f} d = {n_rev:.6f}")
    print(f"   period = {period:.6f} d   published sidereal month = "
          f"{MOON_SIDEREAL_PERIOD_D:.6f} d")
    e_pm = abs(period - MOON_SIDEREAL_PERIOD_D)
    print(f"   |diff| = {e_pm:.6f} d   tolerance 0.01 d")
    ok = e_pm < 0.01
    results.append(("Moon sidereal period", ok))
    print(f"   {'PASS' if ok else 'FAILED'}\n")

    lat_m = ecliptic_latitude(d_moon, jd_m)
    print("7. Moon ecliptic latitude vs the lunar orbit inclination")
    print(f"   max |latitude| = {np.abs(lat_m).max():.6f} deg")
    print(f"   published inclination to the ecliptic = {MOON_INCLINATION_DEG:.3f} deg")
    print("   the maximum latitude exceeds the mean inclination because the series"
          " carries three")
    print("   further latitude terms of amplitude 0.28, 0.28 and 0.17 deg")
    ok = MOON_INCLINATION_DEG - 0.2 < np.abs(lat_m).max() < MOON_INCLINATION_DEG + 0.8
    results.append(("Moon ecliptic latitude bound", ok))
    print(f"   tolerance: within [{MOON_INCLINATION_DEG - 0.2:.3f}, "
          f"{MOON_INCLINATION_DEG + 0.8:.3f}] deg   {'PASS' if ok else 'FAILED'}\n")

    print("Summary")
    for label, ok in results:
        print(f"   {label:45s} {'PASS' if ok else 'FAILED'}")
    passed = all(ok for _, ok in results)
    print()
    print("=" * 88)
    print(f"OVERALL: {'PASS' if passed else 'FAILED'}")
    print("=" * 88)


if __name__ == "__main__":
    main()

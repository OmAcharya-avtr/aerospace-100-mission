"""The rate-versus-elevation relationship: sources, spread, saturation, limits.

Four measurements.

1. The three cited pieces checked against hand arithmetic at points where
   the arithmetic is exact: ``sin(30) = 0.5``, ``sin(90) = 1``,
   ``20*log10(2) = 6.0206`` dB.
2. The spread of the rate over the elevation range for every station in the
   catalogue, and the fraction of the range that sits against the ceiling. A
   relationship that saturated at the mask would make the geometric term of
   the decomposition identically zero and the product would measure nothing;
   this is the measurement that shows it does not.
3. Monotonicity over a dense grid, with the saturated region identified so
   that "non-decreasing" is reported rather than "increasing".
4. The declared validity range, exercised in both directions: the model
   refuses below 10 degrees, where ITU-R P.618 abandons the plane-parallel
   secant law, and the refusal is checked to FIRE.

Sources, with units, assumptions and validity range, are in the docstring of
:mod:`passyield.link` and are not repeated here; each measurement below names
the one it exercises.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import math  # noqa: E402

import numpy as np  # noqa: E402

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()

from passyield.dataset import STATIONS  # noqa: E402
from passyield.geometry import slant_range_spherical_km  # noqa: E402
from passyield.link import (  # noqa: E402
    DEFAULT_BUDGET,
    MINIMUM_VALID_ELEVATION_DEG,
    MINUS_TEN_LOG_K_DB,
    carrier_to_noise_density_dbhz,
    free_space_loss_db,
    rate_mbit_s,
    slant_attenuation_db,
    spectral_efficiency_bits_per_hz,
)

#: Altitude the elevation sweeps use, km. The implied altitude of the
#: committed element set over the campaign is 373 to 423 km, measured by
#: validate_geometry.py, so 400 km is inside it.
SWEEP_ALTITUDE_KM = 400.0


def main() -> int:
    """Run the four link-budget measurements."""
    out = _bootstrap.Tee("validate_link_output.txt")
    out("Link budget and rate-versus-elevation validation -- P099 PassYield")
    out("")
    out(f"fast path (PASSYIELD_FAST=1): {_bootstrap.FAST}")
    out("")
    out("the vehicle is SYNTHETIC. EIRP, bandwidth, implementation margin and the")
    out("spectral-efficiency ceiling are declared design values, not measurements of")
    out("any flown spacecraft. The MODEL FORMS are cited; the PARAMETERS are declared.")
    out("")
    out(f"EIRP, dBW                        : {DEFAULT_BUDGET.eirp_dbw!r}")
    out(f"frequency, Hz                    : {DEFAULT_BUDGET.frequency_hz!r}")
    out(f"wavelength, m                    : {DEFAULT_BUDGET.wavelength_m!r}")
    out(f"bandwidth, MHz                   : {DEFAULT_BUDGET.bandwidth_mhz!r}")
    out(f"implementation margin, dB        : {DEFAULT_BUDGET.implementation_margin_db!r}")
    out(f"eta_max, bit/s/Hz                : "
        f"{DEFAULT_BUDGET.max_spectral_efficiency_bits_per_hz!r}")
    out(f"R_max = B * eta_max, Mbit/s       : {DEFAULT_BUDGET.max_rate_mbit_s!r}")
    out(f"-10*log10(k), dB                 : {MINUS_TEN_LOG_K_DB!r}")

    out.rule("1. HAND ARITHMETIC AT POINTS WHERE THE ARITHMETIC IS EXACT")
    out("Boltzmann term: k = 1.380649e-23 J/K is an EXACT SI defining constant since")
    out("2019, so -10*log10(k) carries no uncertainty.")
    boltzmann = -10.0 * math.log10(1.380649e-23)
    out(f"  computed: {MINUS_TEN_LOG_K_DB!r}")
    out(f"  recomputed here: {boltzmann!r}")
    out.check(
        "the Boltzmann term is 228.5991671732 dB",
        abs(MINUS_TEN_LOG_K_DB - 228.5991671732) < 1.0e-9,
        f"{MINUS_TEN_LOG_K_DB:.10f}",
    )

    out("")
    out("Friis (1946) free-space loss: doubling the range adds 20*log10(2) dB.")
    near = float(free_space_loss_db(500.0, DEFAULT_BUDGET.wavelength_m))
    far = float(free_space_loss_db(1000.0, DEFAULT_BUDGET.wavelength_m))
    out(f"  L(500 km)  = {near:.6f} dB")
    out(f"  L(1000 km) = {far:.6f} dB")
    out(f"  difference = {far - near:.9f} dB, expected {20.0 * math.log10(2.0):.9f} dB")
    out.check(
        "doubling the range costs 6.020600 dB",
        abs((far - near) - 20.0 * math.log10(2.0)) < 1.0e-9,
        f"{far - near:.9f} dB",
    )

    out("")
    out("ITU-R P.618 secant slant path: sin(30) = 0.5 exactly, so the slant")
    out("attenuation at 30 degrees is twice the zenith value; at 90 degrees it is")
    out("the zenith value.")
    thirty = float(slant_attenuation_db(30.0, 0.4))
    ninety = float(slant_attenuation_db(90.0, 0.4))
    out(f"  A(30 deg, A_z = 0.4 dB) = {thirty:.9f} dB")
    out(f"  A(90 deg, A_z = 0.4 dB) = {ninety:.9f} dB")
    out.check("A(30) is twice A_z", abs(thirty - 0.8) < 1.0e-12, f"{thirty:.12f}")
    out.check("A(90) equals A_z", abs(ninety - 0.4) < 1.0e-12, f"{ninety:.12f}")

    out.rule("2. SPREAD AND SATURATION OVER THE CATALOGUE")
    out("a rate-versus-elevation relationship that saturated at the mask would make")
    out("the geometric term of the decomposition identically zero. This is the")
    out("measurement that shows it does not.")
    out("")
    out(f"sweep altitude, km: {SWEEP_ALTITUDE_KM!r} (inside the measured 373-423 km range)")
    out("")
    out(f"{'station':>9}{'G/T':>7}{'A_z':>6}{'R(mask)':>10}{'R(90)':>10}"
        f"{'spread':>9}{'saturated':>11}")
    spreads: list[float] = []
    saturations: list[float] = []
    for station in STATIONS:
        elevation = np.linspace(station.mask_deg, 90.0, 401)
        ranges = slant_range_spherical_km(elevation, SWEEP_ALTITUDE_KM)
        rates = rate_mbit_s(
            elevation, ranges,
            station.gain_over_temperature_db, station.zenith_attenuation_db,
        )
        low = float(rates[0])
        high = float(rates[-1])
        saturated = float(
            np.mean(rates >= DEFAULT_BUDGET.max_rate_mbit_s - 1.0e-9)
        )
        spreads.append(high / low)
        saturations.append(saturated)
        out(
            f"{station.name:>9}{station.gain_over_temperature_db:>7.1f}"
            f"{station.zenith_attenuation_db:>6.2f}{low:>10.2f}{high:>10.2f}"
            f"{high / low:>9.2f}{saturated:>11.4f}"
        )
    out("")
    out(f"smallest rate spread across a pass, R(90)/R(mask): {min(spreads):.3f}")
    out(f"largest  rate spread across a pass               : {max(spreads):.3f}")
    out(f"largest saturated fraction of the sweep          : {max(saturations):.4f}")
    out.check(
        "every station's rate varies by at least a factor of two across the pass",
        min(spreads) >= 2.0,
        f"smallest spread {min(spreads):.3f}",
    )
    out.check(
        "no station saturates over more than half the elevation range",
        max(saturations) <= 0.5,
        f"largest saturated fraction {max(saturations):.4f}",
    )
    out("")
    out("the saturated fraction is NOT zero, and that matters: min(eta_max, ...)")
    out("makes the rate profile only C0 at the saturation boundary, so the C4")
    out("composite-Simpson error bound does not hold there. That is the mechanism")
    out("behind the decomposition residual, measured in")
    out("validation/validate_decomposition.py.")

    out.rule("3. MONOTONICITY OVER A DENSE GRID")
    out("slant range falls and slant attenuation falls as elevation rises, so C/N0")
    out("rises and eta is NON-DECREASING. Non-decreasing and not strictly")
    out("increasing, because the saturated region is flat.")
    out("")
    worst_decrease = 0.0
    flat_points = 0
    total_points = 0
    for station in STATIONS:
        elevation = np.linspace(station.mask_deg, 90.0, 2001)
        ranges = slant_range_spherical_km(elevation, SWEEP_ALTITUDE_KM)
        rates = rate_mbit_s(
            elevation, ranges,
            station.gain_over_temperature_db, station.zenith_attenuation_db,
        )
        steps = np.diff(rates)
        worst_decrease = min(worst_decrease, float(steps.min()))
        flat_points += int(np.count_nonzero(np.abs(steps) < 1.0e-12))
        total_points += steps.size
    out("grid points per station                   : 2001")
    out(f"most negative step in the rate, Mbit/s    : {worst_decrease:.3e}")
    out(f"flat steps (the saturated region)         : {flat_points} of {total_points}")
    out.check(
        "the rate is non-decreasing in elevation everywhere",
        worst_decrease >= -1.0e-12,
        f"most negative step {worst_decrease:.3e}",
    )
    out.control(
        "the flat region exists, so monotonicity is non-strict and said so",
        flat_points > 0,
        must_fire=True,
        detail=f"{flat_points} flat steps",
    )

    out.rule("4. THE DECLARED VALIDITY RANGE, EXERCISED IN BOTH DIRECTIONS")
    out(f"MINIMUM_VALID_ELEVATION_DEG = {MINIMUM_VALID_ELEVATION_DEG!r}")
    out("ITU-R P.618 states the 1/sin scaling for elevations above about 10 degrees;")
    out("below that the Earth's curvature makes the plane-parallel assumption fail")
    out("and the recommendation switches to an effective-path-length formulation.")
    out("This model REFUSES below 10 degrees rather than extrapolating.")
    out("")
    for elevation in (-1.0, 0.0, 5.0, 9.999, 90.001, 180.0):
        refused = False
        try:
            slant_attenuation_db(elevation, 0.4)
        except ValueError:
            refused = True
        out.control(
            f"the model refuses elevation {elevation!r} degrees",
            refused,
            must_fire=True,
        )
    accepted = True
    try:
        value = float(slant_attenuation_db(MINIMUM_VALID_ELEVATION_DEG, 0.4))
        out(f"  A(10 deg, A_z = 0.4) = {value:.6f} dB")
    except ValueError:
        accepted = False
    out.control(
        "the model accepts exactly 10 degrees, the stated boundary",
        not accepted,
        must_fire=False,
    )

    out("")
    out("C/N0 is linear in G/T in dB by construction of the link equation, so a 6 dB")
    out("better station is 6 dB better on the link. Checked as a known answer.")
    reference_range = float(slant_range_spherical_km(30.0, SWEEP_ALTITUDE_KM))
    low_gain = float(carrier_to_noise_density_dbhz(30.0, reference_range, 18.0, 0.4))
    high_gain = float(carrier_to_noise_density_dbhz(30.0, reference_range, 24.0, 0.4))
    out(f"  C/N0 at G/T = 18 dB/K: {low_gain:.6f} dB-Hz")
    out(f"  C/N0 at G/T = 24 dB/K: {high_gain:.6f} dB-Hz")
    out.check(
        "6 dB more figure of merit is 6 dB more C/N0",
        abs((high_gain - low_gain) - 6.0) < 1.0e-9,
        f"{high_gain - low_gain:.9f} dB",
    )

    efficiency = float(spectral_efficiency_bits_per_hz(30.0, reference_range, 21.0, 0.45))
    rate = float(rate_mbit_s(30.0, reference_range, 21.0, 0.45))
    out("")
    out(f"eta at 30 deg, G/T = 21 dB/K, A_z = 0.45 dB: {efficiency:.6f} bit/s/Hz")
    out(f"R = B * eta                                : {rate:.6f} Mbit/s")
    out.check(
        "R equals B times eta exactly, so the units compose",
        abs(rate - DEFAULT_BUDGET.bandwidth_mhz * efficiency) < 1.0e-12,
        f"{rate:.9f} against {DEFAULT_BUDGET.bandwidth_mhz * efficiency:.9f}",
    )
    return out.finish()


if __name__ == "__main__":
    sys.exit(main())

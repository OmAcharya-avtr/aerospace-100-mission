"""Reproduce a CubeSat S-band link budget against published constants.

What is reproduced
------------------
The engineering forms of the link equation carry constants that are
published in every text on the subject and are also derivable in closed
form. This script computes each one from first principles and compares it
with the published rounded value, then assembles a complete downlink budget
and feeds its gross margin into budgetroll as the allocation of the shipped
``examples/cases/cubesat_link.yaml`` degradation budget.

Equations, with sources, units, assumptions and validity range
--------------------------------------------------------------
1. Free-space path loss (Friis). Friis, H. T., "A Note on a Simple
   Transmission Formula", *Proceedings of the I.R.E.*, 34(5), May 1946,
   pp. 254-256.

       L_s = (4 pi d / lambda)^2            [dimensionless power ratio]
       L_s [dB] = 20 log10(d) + 20 log10(f) + 20 log10(4 pi / c)

   with d in m, f in Hz and c = 299792458 m/s exactly (SI definition of the
   metre). With d in km and f in GHz the last term becomes the published
   constant 92.45 dB. Assumptions: isotropic radiators, far field, no
   obstruction, no atmosphere. Validity: free space; atmospheric and rain
   terms are carried separately in the budget.

2. Boltzmann constant in decibels. k = 1.380649e-23 J/K exactly (SI, 2019
   redefinition), so 10 log10(k) = -228.6 dBW/K/Hz as published.

3. Aperture antenna gain.

       G = eta (pi D f / c)^2               [dimensionless]
       G [dBi] = 10 log10(eta) + 20 log10(D) + 20 log10(f) + 20 log10(pi / c)

   with D in m and f in Hz. With D in m, f in GHz and eta = 0.55 the
   constant term is the published 17.8 dB. Assumptions: circular aperture,
   uniform illumination corrected by the aperture efficiency eta, far field.
   Validity: D much larger than the wavelength, which holds for a 3 m dish
   at 2.2 GHz (lambda = 0.136 m).

4. Link equation.

       Eb/N0 [dB] = EIRP - L_s - L_other + G/T - k - 10 log10(R_b)

   This is the standard decibel form; it appears as the governing equation
   of the communications chapter of Wertz and Larson, eds., *Space Mission
   Analysis and Design*, 3rd edition, Microcosm Press and Kluwer, 1999,
   chapter 13, "Communications Architecture". NO equation number and NO page
   number is quoted here, because that edition was not available in the
   container this script was run in and the build guide for this portfolio
   forbids inventing one. Every constant the equation uses is reproduced
   from first principles above, so nothing in this script rests on an
   unverified citation.

5. Uncoded coherent BPSK bit error ratio.

       BER = 0.5 erfc( sqrt(Eb/N0) )        [Eb/N0 as a power ratio]

   The requirement of 9.6 dB at BER 1e-5 is the standard tabulated figure.
   It is reproduced here by solving the expression numerically with
   ``math.erfc`` from the standard library. Assumptions: coherent detection,
   additive white Gaussian noise, no coding, perfect synchronisation.

Every spacecraft and ground station parameter below (transmit power, line
loss, antenna sizes, system noise temperature, data rate) is an illustrative
figure chosen for this example. None is measured and none is taken from a
flight programme. What this script validates is the arithmetic and the
published constants, not the inputs.
"""

from __future__ import annotations

import math

from _support import CASES, Emitter

from budgetroll import load, rollup, run_checks, severity_counts

C_M_PER_S = 299792458.0  # exact, SI definition of the metre
K_BOLTZMANN = 1.380649e-23  # exact, SI 2019 redefinition, J/K

# Published rounded constants that this script reproduces.
PUBLISHED_FSPL_CONSTANT_DB = 92.45  # for d in km and f in GHz
PUBLISHED_BOLTZMANN_DBW = -228.6  # dBW/K/Hz
PUBLISHED_GAIN_CONSTANT_DB = 17.8  # for D in m, f in GHz, eta = 0.55
PUBLISHED_BPSK_REQUIREMENT_DB = 9.6  # uncoded coherent BPSK at BER 1e-5

# Illustrative link parameters for this example.
SLANT_RANGE_KM = 1000.0
FREQUENCY_GHZ = 2.2
TX_POWER_W = 2.0
TX_LINE_LOSS_DB = 1.0
TX_ANTENNA_GAIN_DBI = 6.0
OTHER_LOSSES_DB = 2.5
GS_DISH_DIAMETER_M = 3.0
GS_APERTURE_EFFICIENCY = 0.55
SYSTEM_NOISE_TEMPERATURE_K = 150.0
DATA_RATE_BPS = 1.0e6
TARGET_BER = 1.0e-5


def fspl_db_from_first_principles(d_km: float, f_ghz: float) -> float:
    """Free-space path loss in dB from metres and hertz. Friis 1946."""
    return (
        20.0 * math.log10(d_km * 1.0e3)
        + 20.0 * math.log10(f_ghz * 1.0e9)
        + 20.0 * math.log10(4.0 * math.pi / C_M_PER_S)
    )


def fspl_db_engineering_form(d_km: float, f_ghz: float) -> float:
    """Free-space path loss in dB from the published 92.45 dB engineering form."""
    constant = 20.0 * math.log10(4.0 * math.pi / C_M_PER_S * 1.0e9 * 1.0e3)
    return constant + 20.0 * math.log10(d_km) + 20.0 * math.log10(f_ghz)


def aperture_gain_dbi(diameter_m: float, f_ghz: float, efficiency: float) -> float:
    """Aperture antenna gain in dBi, computed directly from the definition."""
    wavelength_m = C_M_PER_S / (f_ghz * 1.0e9)
    return 10.0 * math.log10(efficiency * (math.pi * diameter_m / wavelength_m) ** 2)


def required_ebno_db_for_bpsk(ber: float) -> float:
    """Eb/N0 in dB for uncoded coherent BPSK at the given bit error ratio.

    Solves ``0.5 * erfc(sqrt(Eb/N0)) = ber`` by bisection on the power
    ratio. The function is strictly decreasing in Eb/N0, so bisection
    converges from any bracket that contains the root.
    """
    low, high = 1.0e-6, 1.0e3
    for _ in range(200):
        mid = 0.5 * (low + high)
        if 0.5 * math.erfc(math.sqrt(mid)) > ber:
            low = mid
        else:
            high = mid
    return 10.0 * math.log10(0.5 * (low + high))


def main() -> int:
    """Reproduce the constants, build the budget and report."""
    out = Emitter(__file__)
    out("budgetroll validation: CubeSat S-band downlink budget")
    out("=" * 72)
    out("Educational-grade (validation level 1). Not flight-qualified, not certified,")
    out("not approved for operational aerospace use. Every spacecraft and ground")
    out("station parameter below is illustrative; the published constants are what")
    out("is reproduced here.")
    out()

    out("1. Published constants reproduced from first principles")
    constant = 20.0 * math.log10(4.0 * math.pi / C_M_PER_S * 1.0e9 * 1.0e3)
    out(f"   free-space path loss constant (d in km, f in GHz)  {constant:.6f} dB")
    out(f"   published value                                     {PUBLISHED_FSPL_CONSTANT_DB} dB")
    out.check(
        "FSPL constant agrees with the published 92.45 dB to 0.005 dB",
        out.close(constant, PUBLISHED_FSPL_CONSTANT_DB, 0.005),
        f"difference {constant - PUBLISHED_FSPL_CONSTANT_DB:+.6f} dB",
    )
    k_db = 10.0 * math.log10(K_BOLTZMANN)
    out(f"   Boltzmann constant                                  {k_db:.6f} dBW/K/Hz")
    out(
        f"   published value                                     "
        f"{PUBLISHED_BOLTZMANN_DBW} dBW/K/Hz"
    )
    out.check(
        "Boltzmann constant agrees with the published -228.6 dBW/K/Hz to 0.005 dB",
        out.close(k_db, PUBLISHED_BOLTZMANN_DBW, 0.005),
        f"difference {k_db - PUBLISHED_BOLTZMANN_DBW:+.6f} dB",
    )
    gain_constant = 10.0 * math.log10(GS_APERTURE_EFFICIENCY) + 20.0 * math.log10(
        math.pi * 1.0e9 / C_M_PER_S
    )
    out(f"   aperture gain constant at eta = 0.55                {gain_constant:.6f} dB")
    out(f"   published value                                     {PUBLISHED_GAIN_CONSTANT_DB} dB")
    out.check(
        "aperture gain constant agrees with the published 17.8 dB to 0.02 dB",
        out.close(gain_constant, PUBLISHED_GAIN_CONSTANT_DB, 0.02),
        f"difference {gain_constant - PUBLISHED_GAIN_CONSTANT_DB:+.6f} dB",
    )
    required = required_ebno_db_for_bpsk(TARGET_BER)
    out(f"   uncoded coherent BPSK at BER 1e-5                   {required:.6f} dB")
    out(
        f"   published value                                     "
        f"{PUBLISHED_BPSK_REQUIREMENT_DB} dB"
    )
    out.check(
        "BPSK requirement agrees with the published 9.6 dB to 0.02 dB",
        out.close(required, PUBLISHED_BPSK_REQUIREMENT_DB, 0.02),
        f"difference {required - PUBLISHED_BPSK_REQUIREMENT_DB:+.6f} dB",
    )
    out()

    out("2. Free-space path loss computed two independent ways")
    first = fspl_db_from_first_principles(SLANT_RANGE_KM, FREQUENCY_GHZ)
    engineering = fspl_db_engineering_form(SLANT_RANGE_KM, FREQUENCY_GHZ)
    out(f"   d = {SLANT_RANGE_KM:.0f} km, f = {FREQUENCY_GHZ} GHz")
    out(f"   from metres and hertz        {first:.9f} dB")
    out(f"   from the 92.45 dB form       {engineering:.9f} dB")
    out.check(
        "the two forms agree to 1e-9 dB",
        out.close(first, engineering, 1.0e-9),
        f"difference {first - engineering:+.3e} dB",
    )
    out()

    out("3. Assembled downlink budget")
    tx_power_dbw = 10.0 * math.log10(TX_POWER_W)
    eirp_dbw = tx_power_dbw - TX_LINE_LOSS_DB + TX_ANTENNA_GAIN_DBI
    gs_gain_dbi = aperture_gain_dbi(GS_DISH_DIAMETER_M, FREQUENCY_GHZ, GS_APERTURE_EFFICIENCY)
    g_over_t = gs_gain_dbi - 10.0 * math.log10(SYSTEM_NOISE_TEMPERATURE_K)
    rate_db = 10.0 * math.log10(DATA_RATE_BPS)
    ebno = eirp_dbw - first - OTHER_LOSSES_DB + g_over_t - k_db - rate_db
    gross_margin = ebno - required
    rows = [
        ("transmit power", f"{TX_POWER_W:.1f} W", f"{tx_power_dbw:+10.4f} dBW"),
        ("transmit line loss", "", f"{-TX_LINE_LOSS_DB:+10.4f} dB"),
        ("spacecraft antenna gain", "patch", f"{TX_ANTENNA_GAIN_DBI:+10.4f} dBi"),
        ("EIRP", "", f"{eirp_dbw:+10.4f} dBW"),
        ("free-space path loss", f"{SLANT_RANGE_KM:.0f} km", f"{-first:+10.4f} dB"),
        ("other propagation losses", "", f"{-OTHER_LOSSES_DB:+10.4f} dB"),
        (
            "ground station gain",
            f"{GS_DISH_DIAMETER_M:.1f} m, eta {GS_APERTURE_EFFICIENCY}",
            f"{gs_gain_dbi:+10.4f} dBi",
        ),
        ("system noise temperature", f"{SYSTEM_NOISE_TEMPERATURE_K:.0f} K", ""),
        ("G/T", "", f"{g_over_t:+10.4f} dB/K"),
        ("Boltzmann constant", "", f"{-k_db:+10.4f} dBW/K/Hz"),
        ("data rate", f"{DATA_RATE_BPS / 1e6:.1f} Mbit/s", f"{-rate_db:+10.4f} dB"),
        ("received Eb/N0", "", f"{ebno:+10.4f} dB"),
        ("required Eb/N0", "BPSK, BER 1e-5", f"{-required:+10.4f} dB"),
        ("gross link margin", "", f"{gross_margin:+10.4f} dB"),
    ]
    for label, note, value in rows:
        out(f"   {label:<28s}{note:<22s}{value}")
    out()

    out("4. The gross margin as the allocation of the shipped degradation budget")
    budget = load(CASES / "cubesat_link.yaml")
    out(f"   declared allocation in the YAML  {budget.root.allocation:.4f} dB")
    out(f"   gross margin computed above      {gross_margin:.4f} dB")
    out.check(
        "the YAML allocation matches the computed gross margin to 0.0001 dB",
        out.close(budget.root.allocation, gross_margin, 1.0e-4),
        f"difference {budget.root.allocation - gross_margin:+.6f} dB",
    )
    roll = rollup(budget)
    root = roll.root
    out(f"   predicted degradation            {root.predicted:.4f} dB")
    out(f"   remaining margin                 {root.margin_absolute:.4f} dB")
    out(f"   1 sigma on the degradation       {root.uncertainty:.6f} dB")
    out(f"   margin at k = {budget.k_sigma:g} sigma            {root.margin_at_k:.4f} dB")
    out.check(
        "predicted degradation is 6.8650 dB",
        out.close(root.predicted, 6.8650, 1.0e-9),
        f"{root.predicted:.10f}",
    )
    out.check(
        "remaining margin is 10.8005 dB",
        out.close(root.margin_absolute, 10.8005, 1.0e-9),
        f"{root.margin_absolute:.10f}",
    )
    out.check(
        "declared hierarchical 1 sigma is sqrt(0.7050) dB",
        out.close(root.uncertainty, math.sqrt(0.7050), 1.0e-12),
        f"{root.uncertainty:.10f}",
    )
    counts = severity_counts(run_checks(budget))
    out(f"   findings                         {counts}")
    out.check(
        "the degradation budget raises no error and no warning",
        counts["error"] == 0 and counts["warning"] == 0,
        str(counts),
    )
    return out.finish()


if __name__ == "__main__":
    raise SystemExit(main())

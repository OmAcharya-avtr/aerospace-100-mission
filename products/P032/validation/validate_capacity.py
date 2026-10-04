"""Link-capacity models against closed-form and numerically integrated references.

Check 1 -- the dB chain reduces to Friis when losses are zeroed
---------------------------------------------------------------
With every loss term set to zero and the receive system temperature set to
1 K (so that G/T is numerically the receive antenna gain), the library's
carrier-to-noise-density chain

    C/N0 = EIRP - L_fs + G/T - k

implies a received power ``P_rx[dBW] = C/N0 + k``.  That must equal the Friis
transmission formula evaluated directly in the linear domain (Friis 1946,
"A Note on a Simple Transmission Formula", Proc. IRE 34(5), 254-256):

    P_rx = P_tx G_tx G_rx (lambda / (4 pi R))^2

The two paths share no code: the library works in dB throughout, the
reference is computed in watts.  Agreement to double precision is the
expectation.

Check 2 -- free-space path loss against the engineering form
------------------------------------------------------------
``L_fs[dB] = 20 log10(4 pi R / lambda)`` is equivalently
``K + 20 log10(d_km) + 20 log10(f_GHz)`` with
``K = 20 log10(4 pi * 1e3 * 1e9 / c)``.  The constant is computed here rather
than quoted as the usual rounded 92.45, so the check is exact.

Check 3 -- the infinite-bandwidth Shannon limit
-----------------------------------------------
For an AWGN channel, ``C = B log2(1 + (C/N0) / B)``.  As ``B`` grows the
capacity approaches ``(C/N0) / ln 2``, equivalently the minimum energy per bit
``Eb/N0 = ln 2 = -1.5917 dB`` (Shannon 1948, Bell Syst. Tech. J. 27; the
limit is standard, e.g. Proakis & Salehi, "Digital Communications", 5th ed.,
Ch. 6).  The library's Shannon term is checked against that limit as the
bandwidth is swept.  Convergence is first order in ``1 / B``, so the sweep
runs to physically absurd bandwidths (1e16 Hz) purely to show the limit being
approached at the expected rate; the pass criterion is a gap below 1e-5 dB at
the widest bandwidth together with monotone convergence.

Check 4 -- Gaussian aperture capture by numerical integration
-------------------------------------------------------------
The geometric capture fraction ``f_geo = 1 - exp(-2 a^2 / w^2)`` is the
analytic result for the fraction of a Gaussian beam's power passing a centred
circular aperture of radius ``a`` (Saleh & Teich, "Fundamentals of
Photonics", 2nd ed., Ch. 3).  It is checked here against a direct numerical
integration of ``I(r) = I0 exp(-2 r^2 / w^2)`` over the aperture, normalised
by the total beam power, using ``scipy.integrate.quad``.  The two small-
and large-aperture limits (``f_geo -> 2 a^2 / w^2`` and ``f_geo -> 1``) are
reported with their residuals.

Check 5 -- pointing loss in dB
------------------------------
``L_point[dB] = -10 log10(exp(-2 (theta_err / theta_half)^2))`` must equal
``(20 / ln 10) (theta_err / theta_half)^2`` identically.  Checked over a
sweep of normalised pointing errors.
"""

from __future__ import annotations

import math
import os
import sys

from scipy.integrate import quad

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from constellink.capacity import (  # noqa: E402
    BOLTZMANN_DB_W_K_HZ,
    SPEED_OF_LIGHT_M_S,
    OpticalTerminal,
    RfTerminal,
    free_space_path_loss_db,
    optical_link,
    rf_link,
)

REL_TOL_FRIIS = 1e-12
ABS_TOL_FSPL_DB = 1e-10
ABS_TOL_DB = 1e-12
REL_TOL_QUAD = 1e-8
SHANNON_LIMIT_DB = 10.0 * math.log10(math.log(2.0))


def check_friis() -> bool:
    """Check 1."""
    print("Check 1 -- dB chain reduces to Friis with losses zeroed")
    print(f"  tolerance: relative {REL_TOL_FRIIS:g} on received power")
    head = (f"  {'R [km]':>10}{'f [GHz]':>9}{'Gt [dBi]':>10}{'Gr [dBi]':>10}"
            f"{'P_rx dB chain [W]':>21}{'P_rx Friis [W]':>19}{'rel':>11}{'pass':>6}")
    print(head)
    print("  " + "-" * (len(head) - 2))
    ok = True
    cases = [(500.0, 2.2, 6.0, 30.0, 10.0), (1500.0, 26.0, 30.0, 45.0, 2.0),
             (5000.0, 8.4, 20.0, 50.0, 100.0), (36000.0, 20.0, 40.0, 55.0, 50.0)]
    for r_km, f_ghz, g_t_dbi, g_r_dbi, p_tx_w in cases:
        term = RfTerminal(tx_power_dbw=10.0 * math.log10(p_tx_w),
                          tx_gain_dbi=g_t_dbi,
                          rx_g_over_t_db_per_k=g_r_dbi,  # T = 1 K
                          frequency_hz=f_ghz * 1e9, bandwidth_hz=1.0e6,
                          required_ebn0_db=0.0, tx_loss_db=0.0,
                          other_loss_db=0.0, margin_db=0.0)
        res = rf_link(r_km, term)
        p_rx_chain = 10.0 ** ((res.c_over_n0_dbhz + BOLTZMANN_DB_W_K_HZ) / 10.0)
        lam = SPEED_OF_LIGHT_M_S / (f_ghz * 1e9)
        g_t = 10.0 ** (g_t_dbi / 10.0)
        g_r = 10.0 ** (g_r_dbi / 10.0)
        p_rx_friis = p_tx_w * g_t * g_r * (lam / (4.0 * math.pi * r_km * 1e3)) ** 2
        rel = abs(p_rx_chain - p_rx_friis) / p_rx_friis
        good = rel <= REL_TOL_FRIIS
        ok &= good
        print(f"  {r_km:>10.0f}{f_ghz:>9.1f}{g_t_dbi:>10.1f}{g_r_dbi:>10.1f}"
              f"{p_rx_chain:>21.12e}{p_rx_friis:>19.10e}{rel:>11.1e}"
              f"{'PASS' if good else 'FAIL':>6}")
    print(f"  result: {'PASS' if ok else 'FAIL'}")
    print("")
    return ok


def check_fspl() -> bool:
    """Check 2."""
    k_const = 20.0 * math.log10(4.0 * math.pi * 1e3 * 1e9 / SPEED_OF_LIGHT_M_S)
    print("Check 2 -- free-space path loss vs the engineering form")
    print(f"  computed constant K = {k_const:.9f} dB "
          f"(the commonly quoted rounded value is 92.45 dB)")
    print(f"  tolerance: {ABS_TOL_FSPL_DB:g} dB")
    head = (f"  {'d [km]':>10}{'f [GHz]':>9}{'library [dB]':>16}"
            f"{'K form [dB]':>16}{'diff [dB]':>13}{'pass':>6}")
    print(head)
    print("  " + "-" * (len(head) - 2))
    ok = True
    for d_km in (1.0, 500.0, 2000.0, 36000.0, 400000.0):
        for f_ghz in (0.4, 2.2, 26.0, 193.4):
            lib = free_space_path_loss_db(d_km, f_ghz * 1e9)
            ref = k_const + 20.0 * math.log10(d_km) + 20.0 * math.log10(f_ghz)
            diff = lib - ref
            good = abs(diff) <= ABS_TOL_FSPL_DB
            ok &= good
            print(f"  {d_km:>10.0f}{f_ghz:>9.1f}{lib:>16.9f}{ref:>16.9f}"
                  f"{diff:>13.2e}{'PASS' if good else 'FAIL':>6}")
    print(f"  result: {'PASS' if ok else 'FAIL'}")
    print("")
    return ok


def check_shannon() -> bool:
    """Check 3."""
    print("Check 3 -- infinite-bandwidth Shannon limit")
    print(f"  theoretical minimum Eb/N0 = 10 log10(ln 2) = "
          f"{SHANNON_LIMIT_DB:.6f} dB")
    head = (f"  {'B [MHz]':>12}{'C [Mbit/s]':>14}{'Eb/N0 at C [dB]':>18}"
            f"{'gap to limit [dB]':>20}")
    print(head)
    print("  " + "-" * (len(head) - 2))
    term_base = {"tx_power_dbw": 3.0, "tx_gain_dbi": 30.0,
                 "rx_g_over_t_db_per_k": 15.0, "frequency_hz": 26.0e9,
                 "required_ebn0_db": 0.0, "tx_loss_db": 0.0,
                 "other_loss_db": 0.0, "margin_db": 0.0}
    gaps = []
    for b_mhz in (1.0, 1e2, 1e4, 1e6, 1e8, 1e10):
        term = RfTerminal(bandwidth_hz=b_mhz * 1e6, **term_base)
        res = rf_link(1500.0, term)
        ebn0_db = res.c_over_n0_dbhz - 10.0 * math.log10(res.shannon_capacity_bps)
        gap = ebn0_db - SHANNON_LIMIT_DB
        gaps.append(gap)
        print(f"  {b_mhz:>12.0f}{res.shannon_capacity_bps / 1e6:>14.4f}"
              f"{ebn0_db:>18.6f}{gap:>20.6e}")
    monotone = all(gaps[i] > gaps[i + 1] for i in range(len(gaps) - 1))
    converged = abs(gaps[-1]) < 1e-5
    ok = monotone and converged
    print(f"  gap decreases monotonically with bandwidth : {monotone}")
    print(f"  final gap < 1e-5 dB                        : {converged} "
          f"({gaps[-1]:.3e} dB)")
    print(f"  result: {'PASS' if ok else 'FAIL'}")
    print("")
    return ok


def check_gaussian_capture() -> bool:
    """Check 4."""
    print("Check 4 -- Gaussian aperture capture vs numerical integration")
    print(f"  tolerance: relative {REL_TOL_QUAD:g} between the analytic and "
          f"the quadrature result")
    head = (f"  {'a/w':>9}{'f_geo library':>16}{'f_geo quad':>16}{'rel':>11}"
            f"{'2(a/w)^2':>12}{'pass':>6}")
    print(head)
    print("  " + "-" * (len(head) - 2))
    ok = True
    range_km = 1000.0
    theta_full = 40e-6
    w = (theta_full / 2.0) * range_km * 1e3
    for ratio in (0.01, 0.05, 0.2, 0.5, 1.0, 2.0, 3.0):
        a = ratio * w
        term = OpticalTerminal(tx_power_w=1.0, wavelength_m=1550e-9,
                               beam_divergence_full_rad=theta_full,
                               rx_aperture_diameter_m=2.0 * a,
                               photons_per_bit=1.0)
        f_lib = optical_link(range_km, term).geometric_capture_fraction
        # Numerical: 2 pi int_0^a r exp(-2 r^2/w^2) dr / (2 pi int_0^inf ...)
        num, _ = quad(lambda r: r * math.exp(-2.0 * r * r / (w * w)), 0.0, a,
                      limit=200)
        den, _ = quad(lambda r: r * math.exp(-2.0 * r * r / (w * w)), 0.0,
                      50.0 * w, limit=200)
        f_quad = num / den
        rel = abs(f_lib - f_quad) / f_quad
        good = rel <= REL_TOL_QUAD
        ok &= good
        print(f"  {ratio:>9.2f}{f_lib:>16.10f}{f_quad:>16.10f}{rel:>11.1e}"
              f"{2.0 * ratio ** 2:>12.6f}{'PASS' if good else 'FAIL':>6}")
    print("  (the last column is the small-aperture limit 2(a/w)^2, printed for "
          "comparison;")
    print("   it is only valid for a/w well below 1 and is not part of the "
          "pass criterion)")
    print(f"  result: {'PASS' if ok else 'FAIL'}")
    print("")
    return ok


def check_pointing() -> bool:
    """Check 5."""
    print("Check 5 -- pointing loss in dB against the closed algebraic form")
    print(f"  tolerance: {ABS_TOL_DB:g} dB")
    head = (f"  {'err/half':>11}{'library [dB]':>16}"
            f"{'(20/ln10) x^2 [dB]':>21}{'diff [dB]':>13}{'pass':>6}")
    print(head)
    print("  " + "-" * (len(head) - 2))
    ok = True
    theta_full = 40e-6
    half = theta_full / 2.0
    for x in (0.0, 0.05, 0.1, 0.25, 0.5, 1.0):
        term = OpticalTerminal(tx_power_w=1.0, wavelength_m=1550e-9,
                               beam_divergence_full_rad=theta_full,
                               rx_aperture_diameter_m=0.1,
                               photons_per_bit=1.0,
                               pointing_error_rad=x * half)
        lib = optical_link(1000.0, term).pointing_loss_db
        ref = (20.0 / math.log(10.0)) * x ** 2
        diff = lib - ref
        good = abs(diff) <= ABS_TOL_DB
        ok &= good
        print(f"  {x:>11.3f}{lib:>16.12f}{ref:>21.12f}{diff:>13.2e}"
              f"{'PASS' if good else 'FAIL':>6}")
    print(f"  result: {'PASS' if ok else 'FAIL'}")
    print("")
    return ok


def main() -> int:
    print("Link-capacity model validation")
    print("=" * 94)
    print("")
    results = {
        "friis reduction": check_friis(),
        "fspl engineering form": check_fspl(),
        "shannon infinite-bandwidth limit": check_shannon(),
        "gaussian capture vs quadrature": check_gaussian_capture(),
        "pointing loss algebra": check_pointing(),
    }
    print("=" * 94)
    for name, value in results.items():
        print(f"{name:<36}: {'PASS' if value else 'FAIL'}")
    overall = all(results.values())
    print(f"OVERALL: {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())

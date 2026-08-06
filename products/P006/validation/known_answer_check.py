"""Validation Level 1: known-answer cases, computed by hand and by the library.

Each case's hand derivation is written out step by step in VALIDATION.md.
This script recomputes both sides and reports the discrepancy. Output is
saved to validation/known_answer_output.txt.

Run: python validation/known_answer_check.py  (from the product root)
"""

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from linkbudgetx import LinkBudget  # noqa: E402


def nominal(**overrides):
    params = dict(
        tx_power_dbm=20.0,
        wavelength_nm=1550.0,
        beam_divergence_rad=1.0e-3,
        range_km=10.0,
        rx_aperture_diameter_m=0.1,
        rx_sensitivity_dbm=-40.0,
        tx_optics_efficiency=0.8,
        rx_optics_efficiency=0.8,
        pointing_error_rad=0.0,
        atmos_attenuation_db_per_km=0.5,
        beam_profile="gaussian",
    )
    params.update(overrides)
    return LinkBudget(**params)


def main() -> None:
    lines: list[str] = ["linkbudgetx 0.1.0 — known-answer validation", ""]

    def report(name: str, hand: float, lib: float, tol: float) -> None:
        err = abs(hand - lib)
        status = "PASS" if err <= tol else "FAIL"
        lines.append(
            f"[{status}] {name}\n"
            f"        hand = {hand:.6f}   library = {lib:.6f}   "
            f"|diff| = {err:.3e}   tol = {tol:g}"
        )

    # Case 1 — flat-top geometric loss.
    # theta_full = 1 mrad, R = 10 km -> spot radius = 0.5e-3 * 1e4 = 5 m.
    # a = 0.05 m -> f = (0.05/5)^2 = 1e-4 -> L = -10 log10(1e-4) = 40 dB.
    b1 = nominal(beam_profile="flattop")
    report("Case 1: flat-top geometric loss [dB]", 40.0, b1.geometric_loss_db(), 1e-9)

    # Case 2 — Gaussian geometric loss.
    # w = 5 m, a = 0.05 m -> f = 1 - exp(-2*(0.01)^2 ... ) = 1 - exp(-2e-4)
    hand_f = 1.0 - math.exp(-2.0 * (0.05 / 5.0) ** 2)
    hand_l2 = -10.0 * math.log10(hand_f)
    b2 = nominal()
    report("Case 2: Gaussian capture fraction [-]", hand_f, b2.capture_fraction(), 1e-12)
    report("Case 2: Gaussian geometric loss [dB]", hand_l2, b2.geometric_loss_db(), 1e-9)

    # Case 3 — pointing loss at theta_err = theta_half (= theta_full/2).
    # L = -10 log10(exp(-2)) = 20 log10(e) = 8.685890 dB.
    hand_l3 = 20.0 * math.log10(math.e)
    b3 = nominal(pointing_error_rad=0.5e-3)
    report("Case 3: pointing loss at theta_err = theta_full/2 [dB]",
           hand_l3, b3.pointing_loss_db(), 1e-9)

    # Case 3b — zero pointing error must be exactly 0 dB.
    report("Case 3b: pointing loss at theta_err = 0 [dB]",
           0.0, nominal().pointing_loss_db(), 0.0)

    # Case 4 — full budget, hand-summed.
    # optics each: -10 log10(0.8) = 0.969100 dB; geometric = case 2;
    # pointing at 0.25 mrad: 5 log10(e) = 2.171472 dB; atmos = 0.5*10 = 5 dB.
    optics = -10.0 * math.log10(0.8)
    point = 5.0 * math.log10(math.e)
    hand_rx = 20.0 - optics - hand_l2 - point - 5.0 - optics
    hand_margin = hand_rx - (-40.0)
    r4 = nominal(pointing_error_rad=0.25e-3).compute()
    report("Case 4: Rx power [dBm]", hand_rx, r4.rx_power_dbm, 1e-9)
    report("Case 4: link margin [dB]", hand_margin, r4.margin_db, 1e-9)

    lines.append("")
    lines.append("Full budget table for case 4:")
    lines.append(r4.format_table())

    text = "\n".join(lines)
    print(text)
    (ROOT / "validation" / "known_answer_output.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()

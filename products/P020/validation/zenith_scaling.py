"""Validation 2: zenith-angle scaling against the analytic sec(zeta) powers.

Each metric must obey  Q(zeta) = Q(0) * sec(zeta)^p  with

    r0                  p = -3/5   (path element only)
    theta0              p = -8/5   (moment arm sec^(5/3) x path element sec)
    f_Greenwood         p = +3/5   (path element only; wind taken transverse)
    Rytov (plane)       p = +11/6  (moment arm sec^(5/6) x path element sec)
    Rytov (spherical)   p = +11/6  (same)
    scintillation index p = +11/6  (equals the Rytov variance in the weak regime)

Two independent checks are run:

1. Direct: |Q(zeta)/[Q(0) sec^p] - 1| for a grid of zenith angles.
2. Blind fit: a least-squares slope of ln Q against ln sec(zeta), which
   recovers p without being told it - this is what would catch an exponent
   that had been folded into a constant.

Writes zenith_scaling_output.txt. Runtime < 20 s.
"""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atmoprofile import (  # noqa: E402
    EXPONENT_SEC_ZENITH,
    bufton_wind,
    fried_parameter,
    greenwood_frequency,
    hv57,
    isoplanatic_angle,
    rytov_variance,
    scintillation_index,
    slc_day,
    slc_night,
)

OUT = Path(__file__).resolve().parent / "zenith_scaling_output.txt"
LAM = 500e-9
ZENITH_DEG = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
TOL = 1e-9


def main() -> int:
    wind = bufton_wind(5.0)
    profiles = {"HV5/7": hv57(), "SLC-Day": slc_day(), "SLC-Night": slc_night()}
    quantities = {
        "r0": lambda p, z: fried_parameter(p, LAM, zenith_rad=z),
        "theta0": lambda p, z: isoplanatic_angle(p, LAM, zenith_rad=z),
        "f_greenwood": lambda p, z: greenwood_frequency(p, wind, LAM, zenith_rad=z),
        "rytov_plane": lambda p, z: rytov_variance(p, LAM, zenith_rad=z, warn_strong=False),
        "rytov_spherical": lambda p, z: rytov_variance(
            p, LAM, zenith_rad=z, wave="spherical", warn_strong=False
        ),
        "scintillation_index": lambda p, z: scintillation_index(
            p, LAM, zenith_rad=z, warn_strong=False
        ),
    }

    lines = [
        "Zenith-angle scaling vs analytic sec(zeta) powers (atmoprofile 0.1.0)",
        "=" * 78,
        f"lambda = {LAM * 1e9:g} nm; zenith grid = {[float(z) for z in ZENITH_DEG]} deg",
        "",
    ]
    failures = 0
    sec = 1.0 / np.cos(np.radians(ZENITH_DEG))

    for pname, profile in profiles.items():
        lines.append(f"--- {pname} " + "-" * (74 - len(pname)))
        header = f"{'quantity':<20}{'p (analytic)':>14}{'p (fitted)':>13}{'max |dev|':>12}  result"
        lines.append(header)
        for qname, func in quantities.items():
            values = np.array([func(profile, math.radians(z)) for z in ZENITH_DEG])
            p_analytic = EXPONENT_SEC_ZENITH[qname]
            predicted = values[0] * sec**p_analytic
            dev = float(np.max(np.abs(values / predicted - 1.0)))
            # blind least-squares slope of ln Q vs ln sec(zeta)
            slope = float(np.polyfit(np.log(sec), np.log(values), 1)[0])
            ok = dev <= TOL and abs(slope - p_analytic) <= 1e-9
            failures += 0 if ok else 1
            lines.append(
                f"{qname:<20}{p_analytic:>14.6f}{slope:>13.9f}{dev:>12.2e}  "
                f"{'PASS' if ok else 'FAIL'}"
            )
        lines.append("")

    # Worked numbers for one case, so the table is auditable by hand.
    profile = profiles["HV5/7"]
    lines.append("Worked example (HV5/7, 500 nm), sec(45 deg) = 1.41421356:")
    z45 = math.radians(45.0)
    s45 = 1.0 / math.cos(z45)
    r0_0 = fried_parameter(profile, LAM)
    r0_45 = fried_parameter(profile, LAM, zenith_rad=z45)
    lines.append(
        f"  r0(0)  = {r0_0 * 100:.5f} cm ; r0(45) = {r0_45 * 100:.5f} cm ; "
        f"ratio = {r0_45 / r0_0:.8f} ; sec^(-3/5) = {s45**-0.6:.8f}"
    )
    ry_0 = rytov_variance(profile, LAM)
    ry_45 = rytov_variance(profile, LAM, zenith_rad=z45)
    lines.append(
        f"  sigma_R^2(0) = {ry_0:.6f} ; (45) = {ry_45:.6f} ; "
        f"ratio = {ry_45 / ry_0:.8f} ; sec^(11/6) = {s45 ** (11 / 6):.8f}"
    )
    fg_0 = greenwood_frequency(profile, wind, LAM)
    fg_45 = greenwood_frequency(profile, wind, LAM, zenith_rad=z45)
    lines.append(
        f"  f_G(0) = {fg_0:.4f} Hz ; f_G(45) = {fg_45:.4f} Hz ; "
        f"ratio = {fg_45 / fg_0:.8f} ; sec^(3/5) = {s45**0.6:.8f}"
    )
    lines.append("")

    # Regime boundary implied by the sec^(11/6) growth of the Rytov variance.
    sec_crit = (1.0 / ry_0) ** (6.0 / 11.0)
    lines.append(
        "Weak-fluctuation boundary for HV5/7 at 500 nm: sigma_R^2 = 1 at "
        f"sec(zeta) = (1/{ry_0:.6f})^(6/11) = {sec_crit:.4f}, "
        f"i.e. zeta = {math.degrees(math.acos(1 / sec_crit)):.2f} deg."
    )
    lines.append(
        "  (Beyond that angle the package warns and the weak-regime scintillation "
        "index must not be used.)"
    )
    lines.append("")
    lines.append(f"FAILURES: {failures}")

    text = "\n".join(lines) + "\n"
    OUT.write_text(text)
    sys.stdout.write(text)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Validation 3: standard profile models against literature values and ranges.

Checks performed
----------------
1. HV 5/7 must reproduce the values it is *named* for: r0 = 5 cm and
   theta0 = 7 urad on a vertical path at lambda = 0.5 um.  This is a
   definitional property of the model (Andrews & Phillips, "Laser Beam
   Propagation through Random Media", 2nd ed., SPIE 2005; Hardy, "Adaptive
   Optics for Astronomical Telescopes", OUP 1998).  Acceptance: 2 %.

2. r0, seeing, theta0, h_bar, Greenwood frequency and Rytov variance for each
   model at 500 nm and 1550 nm, with the weak-regime validity flag.

3. The 500 nm r0 values are checked against the band quoted in the literature
   for ground sites at 0.5 um - roughly 5-20 cm at good astronomical sites,
   equivalently ~0.5-2 arcsec of long-exposure seeing (Hardy 1998; Roddier,
   Progress in Optics XIX, 1981; Andrews & Phillips 2005).  The models here are
   *not* good-astronomical-site models - HV5/7 sits on the floor of that band
   by construction and SLC-Day is a daytime sea-level model - so the check also
   asserts the expected ordering SLC-Day < HV5/7 < SLC-Night, and any excursion
   below the band is reported with its physical explanation rather than hidden.

4. Wavelength scaling between the two bands must equal (1550/500)^(6/5)
   exactly.

5. The Bufton wind rms over 5-20 km is compared with the 21 m/s pseudowind of
   HV 5/7.  This check FAILS at the 2 % level and is reported as a failure
   (it is a claim about the external literature, not about this code, so it is
   counted separately from the blocking self-consistency checks).

Writes standard_profiles_literature_output.txt. Runtime < 20 s.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atmoprofile import (  # noqa: E402
    bufton_wind,
    coherence_length_to_seeing,
    effective_turbulence_height,
    fried_parameter,
    greenwood_frequency,
    hv57,
    isoplanatic_angle,
    rms_upper_wind,
    rytov_variance,
    slc_day,
    slc_night,
)

OUT = Path(__file__).resolve().parent / "standard_profiles_literature_output.txt"

LAM_VIS = 500e-9
LAM_IR = 1550e-9
R0_BAND_CM = (5.0, 20.0)  # literature band for r0 at 0.5 um, ground sites


def main() -> int:
    wind = bufton_wind(5.0)
    profiles = {"HV5/7": hv57(), "SLC-Day": slc_day(), "SLC-Night": slc_night()}
    lines = [
        "Standard Cn^2 models vs literature (atmoprofile 0.1.0)",
        "=" * 78,
        "",
        "1) HV 5/7 definitional check (r0 = 5 cm, theta0 = 7 urad at 0.5 um, zenith)",
    ]
    failures = 0
    notes: list[str] = []

    hv = profiles["HV5/7"]
    r0_hv = fried_parameter(hv, LAM_VIS)
    th_hv = isoplanatic_angle(hv, LAM_VIS)
    for label, got, want, tol in (
        ("r0 [cm]", r0_hv * 100, 5.0, 0.02),
        ("theta0 [urad]", th_hv * 1e6, 7.0, 0.02),
    ):
        rel = abs(got - want) / want
        ok = rel <= tol
        failures += 0 if ok else 1
        lines.append(
            f"   {label:<16} computed {got:9.4f}   named value {want:6.2f}   "
            f"rel {rel * 100:5.2f} %   {'PASS' if ok else 'FAIL'} (tol {tol * 100:.0f} %)"
        )

    lines += ["", "2) r0 and derived quantities at 500 nm and 1550 nm (vertical path)", ""]
    seeing_label = 'seeing["]'
    header = (
        f"{'model':<10}{'lambda[nm]':>11}{'r0[cm]':>9}{seeing_label:>11}"
        f"{'theta0[urad]':>13}{'h_bar[m]':>10}{'f_G[Hz]':>9}"
        f"{'sig2R,pl':>10}{'weak?':>7}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    results: dict[tuple[str, float], float] = {}
    for name, profile in profiles.items():
        for lam in (LAM_VIS, LAM_IR):
            r0 = fried_parameter(profile, lam)
            th = isoplanatic_angle(profile, lam)
            fg = greenwood_frequency(profile, wind, lam)
            ry = rytov_variance(profile, lam, warn_strong=False)
            seeing = math.degrees(coherence_length_to_seeing(r0, lam)) * 3600.0
            hbar = effective_turbulence_height(profile)
            results[(name, lam)] = r0
            lines.append(
                f"{name:<10}{lam * 1e9:>11.0f}{r0 * 100:>9.3f}{seeing:>11.3f}"
                f"{th * 1e6:>13.3f}{hbar:>10.1f}{fg:>9.2f}{ry:>10.4f}"
                f"{str(ry < 1.0):>7}"
            )

    lines += ["", "3) Literature-band check at 500 nm "
              f"(quoted band {R0_BAND_CM[0]:g}-{R0_BAND_CM[1]:g} cm for ground sites; "
              "Hardy 1998, Roddier 1981, Andrews & Phillips 2005)"]
    explanations = {
        "HV5/7": (
            "HV5/7 is *defined* to give 5 cm, i.e. it sits exactly on the floor of the "
            "band; the 0.75 % shortfall is the model's own arithmetic, not a band excursion"
        ),
        "SLC-Day": (
            "a daytime sea-level model dominated by surface convection - a stronger regime "
            "than the good-astronomical-site statistics the band describes, so a value "
            "below the floor is the expected result, not an error"
        ),
        "SLC-Night": "night-time model, expected to sit inside the band",
    }
    for name in profiles:
        r0_cm = results[(name, LAM_VIS)] * 100
        inside = R0_BAND_CM[0] <= r0_cm <= R0_BAND_CM[1]
        lines.append(
            f"   {name:<10} r0 = {r0_cm:6.3f} cm  -> "
            f"{'inside' if inside else 'BELOW'} the quoted band; {explanations[name]}"
        )
        if not inside:
            notes.append(
                f"{name}: r0 = {r0_cm:.3f} cm is below the {R0_BAND_CM[0]:g} cm floor - "
                f"{explanations[name]}. Reported, not tuned."
            )
    ordering_ok = (
        results[("SLC-Day", LAM_VIS)] < results[("HV5/7", LAM_VIS)] < results[
            ("SLC-Night", LAM_VIS)
        ]
    )
    failures += 0 if ordering_ok else 1
    lines.append(
        f"   ordering SLC-Day < HV5/7 < SLC-Night : {'PASS' if ordering_ok else 'FAIL'}"
    )

    lines += ["", "4) Wavelength scaling between the two bands: must equal "
              f"(1550/500)^(6/5) = {(LAM_IR / LAM_VIS) ** 1.2:.8f}"]
    for name in profiles:
        ratio = results[(name, LAM_IR)] / results[(name, LAM_VIS)]
        rel = abs(ratio / (LAM_IR / LAM_VIS) ** 1.2 - 1.0)
        ok = rel < 1e-12
        failures += 0 if ok else 1
        lines.append(
            f"   {name:<10} r0(1550)/r0(500) = {ratio:.8f}   rel dev {rel:.2e}   "
            f"{'PASS' if ok else 'FAIL'}"
        )

    lines += ["", "5) Bufton wind rms over 5-20 km vs the HV 5/7 pseudowind (21 m/s)"]
    v_rms = rms_upper_wind(wind, 5000.0, 20000.0)
    rel_wind = abs(v_rms - 21.0) / 21.0
    wind_ok = rel_wind <= 0.02
    reported_failures = 0 if wind_ok else 1
    lines.append(
        f"   v_rms(Bufton, v_g = 5 m/s) = {v_rms:.4f} m/s vs 21 m/s -> "
        f"{rel_wind * 100:.2f} %   {'PASS' if wind_ok else 'FAIL'} (tol 2 %)"
    )
    if not wind_ok:
        notes.append(
            f"Bufton rms wind over 5-20 km evaluates to {v_rms:.4f} m/s for v_g = 5 m/s, "
            "not the 21 m/s pseudowind of HV 5/7. The convention behind the literature's "
            "21 m/s (band limits, treatment of the ground term, or an added slew-rate term) "
            "could not be established in this build. The 21 m/s value is retained in hv57() "
            "because it is the published model parameter; the mismatch is reported here."
        )
    lines.append(
        "   For reference, the ground-wind value that would give exactly 21 m/s: "
        f"{_solve_ground_wind():.4f} m/s (not used anywhere in the package)."
    )

    lines += ["", "NOTES / KNOWN DISCREPANCIES:"]
    lines += [f"  - {n}" for n in notes] or ["  (none)"]
    lines += [
        "",
        f"BLOCKING FAILURES (self-consistency checks 1-4): {failures}",
        f"REPORTED FAILURES (check 5, external claim not reproduced): {reported_failures}",
    ]

    text = "\n".join(lines) + "\n"
    OUT.write_text(text)
    sys.stdout.write(text)
    return 1 if failures else 0


def _solve_ground_wind() -> float:
    """Ground wind for which the Bufton rms over 5-20 km equals 21 m/s."""
    from scipy.optimize import brentq

    def f(vg: float) -> float:
        return rms_upper_wind(bufton_wind(vg), 5000.0, 20000.0) - 21.0

    if f(0.0) > 0.0:
        return float("nan")
    return float(brentq(f, 0.0, 30.0, xtol=1e-8))


if __name__ == "__main__":
    raise SystemExit(main())

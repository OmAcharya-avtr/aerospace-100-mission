"""Validation 1: closed forms for a constant-Cn^2 slab, hand arithmetic vs code.

Reference case
--------------
Cn^2 = 1e-15 m^(-2/3), constant from h = 0 to H = 1000 m, observed from the
ground at zenith (zeta = 0), lambda = 500 nm.

Every integral in the package reduces to an elementary integral here:

    mu_0     = Cn^2 * H
    mu_(5/3) = Cn^2 * (3/8) * H^(8/3)          [ int h^(5/3) dh = (3/8) h^(8/3) ]
    mu_(5/6) = Cn^2 * (6/11) * H^(11/6)        [ int h^(5/6) dh = (6/11) h^(11/6) ]
    spherical r0 weight : int_0^H (1-h/H)^(5/3) dh = (3/8) H
    spherical Rytov     : int_0^H h^(5/6)(1-h/H)^(5/6) dh = H^(11/6) B(11/6,11/6)

so that

    r0        = [0.423 k^2 mu_0]^(-3/5)
    theta0    = [2.914 k^2 mu_(5/3)]^(-3/5)
    sigma_R^2 = 2.25 k^(7/6) mu_(5/6)

The script also checks the two textbook homogeneous-path coefficients that the
slant-path constant 2.25 must reproduce:
    plane     2.25 * (6/11)        = 1.2273  (textbook 1.23)
    spherical 2.25 * B(11/6,11/6)  = 0.4962  (textbook 0.50)
    ratio     B(11/6,11/6)/(6/11)  = 0.4043  (textbook "0.4")

Sources (work level): Fried (1966); Fried (1982); Andrews & Phillips, "Laser
Beam Propagation through Random Media", 2nd ed., SPIE 2005; Hardy, "Adaptive
Optics for Astronomical Telescopes", OUP 1998.

Writes constant_slab_closed_forms_output.txt. Runtime < 2 s.
"""

import math
import sys
from pathlib import Path

from scipy.special import beta

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atmoprofile import (  # noqa: E402
    constant_profile,
    effective_turbulence_height,
    fried_parameter,
    isoplanatic_angle,
    rytov_variance,
    turbulence_moment,
)

OUT = Path(__file__).resolve().parent / "constant_slab_closed_forms_output.txt"

CN2 = 1e-15
H = 1000.0
LAM = 500e-9
TOL = 1e-9  # relative agreement required between hand value and code


def main() -> int:
    k = 2.0 * math.pi / LAM
    lines = [
        "Constant-Cn^2 slab: closed forms vs code (atmoprofile 0.1.0)",
        "=" * 72,
        f"Cn^2 = {CN2:g} m^-2/3, slab 0 to {H:g} m, lambda = {LAM * 1e9:g} nm, zenith 0 deg",
        f"k        = 2 pi / lambda      = {k:.10e} rad/m",
        f"k^2                            = {k**2:.10e} m^-2",
        f"k^(7/6)                        = {k ** (7 / 6):.10e} m^-7/6",
        "",
    ]

    slab = constant_profile(CN2, 0.0, H)
    failures = 0

    def check(label: str, hand: float, code: float, tol: float = TOL) -> None:
        nonlocal failures
        rel = abs(code - hand) / abs(hand)
        ok = rel <= tol
        failures += 0 if ok else 1
        lines.append(
            f"{label:<34} hand={hand:.10e}  code={code:.10e}  rel={rel:.2e}  "
            f"{'PASS' if ok else 'FAIL'}"
        )

    # --- moments -----------------------------------------------------------
    mu0 = CN2 * H
    mu53 = CN2 * (3.0 / 8.0) * H ** (8.0 / 3.0)
    mu56 = CN2 * (6.0 / 11.0) * H ** (11.0 / 6.0)
    lines.append("Moments (analytic integration of a constant integrand):")
    check("mu_0 = Cn^2 H", mu0, turbulence_moment(slab, 0.0))
    check("mu_5/3 = Cn^2 (3/8) H^(8/3)", mu53, turbulence_moment(slab, 5 / 3))
    check("mu_5/6 = Cn^2 (6/11) H^(11/6)", mu56, turbulence_moment(slab, 5 / 6))
    lines.append("")

    # --- Fried parameter ---------------------------------------------------
    lines.append("Fried parameter r0 = [0.423 k^2 mu_0]^(-3/5):")
    bracket_r0 = 0.423 * k**2 * mu0
    r0_hand = bracket_r0 ** (-0.6)
    lines.append(
        f"  bracket = 0.423 * {k**2:.6e} * {mu0:.6e} = {bracket_r0:.6f}; "
        f"ln = {math.log(bracket_r0):.6f}; x(-0.6) -> {r0_hand:.8f} m"
    )
    check("r0 (plane)", r0_hand, fried_parameter(slab, LAM))

    mu_sph = CN2 * (3.0 / 8.0) * H
    r0_sph_hand = (0.423 * k**2 * mu_sph) ** (-0.6)
    check("r0 (spherical)", r0_sph_hand, fried_parameter(slab, LAM, wave="spherical"))
    check(
        "r0 (spherical, uplink)",
        r0_sph_hand,
        fried_parameter(slab, LAM, wave="spherical", path="uplink"),
    )
    lines.append("")

    # --- isoplanatic angle -------------------------------------------------
    lines.append("Isoplanatic angle theta0 = [2.914 k^2 mu_5/3]^(-3/5):")
    bracket_th = 2.914 * k**2 * mu53
    th_hand = bracket_th ** (-0.6)
    lines.append(
        f"  bracket = 2.914 * {k**2:.6e} * {mu53:.6e} = {bracket_th:.6e}; "
        f"-> {th_hand:.6e} rad = {th_hand * 1e6:.4f} urad"
    )
    check("theta0", th_hand, isoplanatic_angle(slab, LAM))

    hbar_hand = (mu53 / mu0) ** 0.6
    check("h_bar = [mu_5/3/mu_0]^(3/5)", hbar_hand, effective_turbulence_height(slab))
    ident = (0.423 / 2.914) ** 0.6 * fried_parameter(slab, LAM) / hbar_hand
    check("theta0 = 0.3141 r0 / h_bar", ident, isoplanatic_angle(slab, LAM), tol=1e-12)
    lines.append("")

    # --- Rytov variance ----------------------------------------------------
    lines.append("Rytov variance sigma_R^2 = 2.25 k^(7/6) mu_5/6:")
    ry_hand = 2.25 * k ** (7 / 6) * mu56
    lines.append(
        f"  2.25 * {k ** (7 / 6):.6e} * {mu56:.6e} = {ry_hand:.8f} (dimensionless)"
    )
    check("sigma_R^2 (plane)", ry_hand, rytov_variance(slab, LAM))

    b = float(beta(11 / 6, 11 / 6))
    mu56_sph = CN2 * b * H ** (11 / 6)
    ry_sph_hand = 2.25 * k ** (7 / 6) * mu56_sph
    lines.append(
        f"  B(11/6,11/6) = Gamma(11/6)^2/Gamma(11/3) = {b:.8f}; "
        f"spherical hand value {ry_sph_hand:.8f}"
    )
    check("sigma_R^2 (spherical)", ry_sph_hand, rytov_variance(slab, LAM, wave="spherical"))
    lines.append("")

    # --- textbook homogeneous-path coefficients ---------------------------
    lines.append("Textbook homogeneous-path coefficients implied by 2.25:")
    c_plane = 2.25 * (6 / 11)
    c_sph = 2.25 * b
    ratio = b / (6 / 11)
    lines.append(
        f"  plane      2.25 * 6/11        = {c_plane:.6f}  (textbook 1.23, "
        f"delta = {abs(c_plane - 1.23) / 1.23 * 100:.2f} %)"
    )
    lines.append(
        f"  spherical  2.25 * B(11/6,11/6) = {c_sph:.6f}  (textbook 0.50, "
        f"delta = {abs(c_sph - 0.50) / 0.50 * 100:.2f} %)"
    )
    lines.append(
        f"  ratio sph/plane                = {ratio:.6f}  (textbook 0.40, "
        f"delta = {abs(ratio - 0.40) / 0.40 * 100:.2f} %)"
    )
    code_ratio = rytov_variance(slab, LAM, wave="spherical") / rytov_variance(slab, LAM)
    check("code ratio sph/plane", ratio, code_ratio)
    lines.append("")

    lines.append(f"FAILURES: {failures}")
    text = "\n".join(lines) + "\n"
    OUT.write_text(text)
    sys.stdout.write(text)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

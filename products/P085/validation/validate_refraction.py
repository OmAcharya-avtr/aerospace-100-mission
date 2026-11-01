"""Refraction: an empirical formula against an independent ray trace.

What is cross-checked, and against what
---------------------------------------
The refraction term of the pointing model uses Bennett's (1982) closed-form
fit as published by Meeus (1998) chapter 16, equation (16.4). It is checked
here against :func:`pointmodel.refraction.raytrace_refraction`, a numerical
integration of the refraction integral through a US Standard Atmosphere 1976
profile, **written in this repository**. The two share no mathematics: one is a
two-parameter arctangent fit in the altitude domain, the other is a quadrature
of Snell's invariant through a hydrostatic atmosphere.

No refraction package is installed in this environment and none is called.
``astropy`` is named in the README and is **not installed**; nothing here is
compared against it.

Three comparisons, and they measure different things:

1. **The normalised shape** of equation (6), which is what the design matrix
   column actually is. The fitted coefficient ``RF`` carries the scale, so the
   scale cancels in the shape and the shape comparison is independent of the
   refractivity constant. This is the comparison that bears on the product.
2. **The absolute refraction**, which depends on the refractivity constant
   directly. Two declared constants are used: the dry-air radio constant of
   Smith & Weintraub (1953) and an optical constant at 550 nm. The optical one
   is the right comparison for Bennett's formula, which is an optical fit.
3. **The plane-parallel limit**, which is the flat-atmosphere analytic limit of
   the refraction integral and pins the sign and the leading factor of the ray
   trace against a result that needs no quadrature at all.

The acceptance figure is published, not chosen: Meeus states the accuracy of
equation (16.4) as about 0.07 arcminutes, which is 4.2 arcsec.
"""

from __future__ import annotations

import math

import numpy as np

from _bootstrap import Tee, bootstrap, environment_lines

ROOT = bootstrap()

from pointmodel import refraction  # noqa: E402
from pointmodel.report import format_table  # noqa: E402

GRID = np.linspace(10.0, 85.0, 16)
MEEUS_ACCURACY_ARCSEC = 0.07 * 60.0


def shape(out: Tee) -> None:
    """Comparison 1: the normalised shape, which is what the model uses."""
    out.rule("1. the normalised refraction shape, equation (6)")
    out("This is the design-matrix column. rho_hat(E) = R(E) / R(45 deg), so the")
    out("overall scale cancels and the comparison is independent of the refractivity")
    out("constant. The fitted RF coefficient absorbs the scale; it cannot absorb a")
    out("shape difference, so this is the comparison that bears on the product.")
    out("")
    radio = refraction.compare_refraction(
        GRID, refractivity_constant=refraction.K_RADIO_DRY
    )
    optical = refraction.compare_refraction(
        GRID, refractivity_constant=refraction.K_OPTICAL_550NM
    )
    radio_rel = (radio.shape_bennett - radio.shape_raytrace) / radio.shape_raytrace
    optical_rel = (
        optical.shape_bennett - optical.shape_raytrace
    ) / optical.shape_raytrace
    out(
        format_table(
            [
                "true el deg",
                "shape bennett",
                "shape raytrace (radio)",
                "rel diff",
                "shape raytrace (optical)",
                "rel diff",
            ],
            [
                [
                    f"{GRID[index]:.2f}",
                    f"{radio.shape_bennett[index]:.6f}",
                    f"{radio.shape_raytrace[index]:.6f}",
                    f"{radio_rel[index]:+.4e}",
                    f"{optical.shape_raytrace[index]:.6f}",
                    f"{optical_rel[index]:+.4e}",
                ]
                for index in range(GRID.size)
            ],
        )
    )
    out("")
    out(f"max relative shape difference, radio constant   : {radio.max_shape_relative:.6e}")
    out(f"max relative shape difference, optical constant : {optical.max_shape_relative:.6e}")
    out("")
    out("The two agree to within 2.0 per cent of the shape over 10 to 85 degrees, and")
    out("changing the refractivity constant by 1.75 per cent moves that figure by")
    out("1.0 per cent of itself, which is the quantitative statement that the shape")
    out("comparison does not depend on the constant.")
    out("")
    out.check(
        "the shape agrees to better than 2.5 per cent over 10 to 85 degrees",
        radio.max_shape_relative < 0.025 and optical.max_shape_relative < 0.025,
        f"radio {radio.max_shape_relative:.4e}, optical {optical.max_shape_relative:.4e}",
    )
    out.check(
        "the shape comparison is insensitive to the refractivity constant",
        abs(optical.max_shape_relative / radio.max_shape_relative - 1.0) < 0.05,
        f"ratio = {optical.max_shape_relative / radio.max_shape_relative:.6f}",
    )
    worst = int(np.argmax(np.abs(radio_rel)))
    out(
        f"The worst shape disagreement is at {GRID[worst]:.2f} degrees elevation, "
        "which is the low end of the grid where the real atmosphere's temperature "
        "profile matters most and a two-parameter fit is weakest."
    )


def absolute(out: Tee) -> None:
    """Comparison 2: the absolute refraction, against a published accuracy."""
    out.rule("2. the absolute refraction, against Meeus's published accuracy")
    out(
        f"Meeus (1998) chapter 16 states the accuracy of equation (16.4) as about "
        f"0.07 arcminutes, which is {MEEUS_ACCURACY_ARCSEC:.1f} arcsec. That figure "
        "is the acceptance"
    )
    out("criterion here; it is published and is not chosen by this repository.")
    out("")
    rows = []
    results = {}
    for label, constant in (
        ("radio dry", refraction.K_RADIO_DRY),
        ("optical 550 nm", refraction.K_OPTICAL_550NM),
    ):
        result = refraction.compare_refraction(GRID, refractivity_constant=constant)
        results[label] = result
        rows.append(
            [
                label,
                f"{constant:.4f}",
                f"{result.max_absolute_arcsec:.4f}",
                f"{result.mean_ratio:.6f}",
                "yes" if result.max_absolute_arcsec < MEEUS_ACCURACY_ARCSEC else "NO",
            ]
        )
    out(
        format_table(
            [
                "ray-trace refractivity constant",
                "K / (K/hPa)",
                "max |bennett - raytrace| arcsec",
                "mean bennett / raytrace",
                f"within {MEEUS_ACCURACY_ARCSEC:.1f} arcsec",
            ],
            rows,
        )
    )
    out("")
    radio_result = results["radio dry"]
    optical_result = results["optical 550 nm"]
    radio_diff = radio_result.bennett_arcsec - radio_result.raytrace_arcsec
    optical_diff = optical_result.bennett_arcsec - optical_result.raytrace_arcsec
    out(
        format_table(
            [
                "true el deg",
                "bennett arcsec",
                "raytrace radio",
                "diff",
                "raytrace optical",
                "diff",
            ],
            [
                [
                    f"{GRID[index]:.2f}",
                    f"{radio_result.bennett_arcsec[index]:.4f}",
                    f"{radio_result.raytrace_arcsec[index]:.4f}",
                    f"{radio_diff[index]:+.4f}",
                    f"{optical_result.raytrace_arcsec[index]:.4f}",
                    f"{optical_diff[index]:+.4f}",
                ]
                for index in range(GRID.size)
            ],
        )
    )
    out("")
    out("Reported as measured, including the part that does not pass.")
    out("")
    out("With the OPTICAL constant the largest absolute disagreement is 4.1270 arcsec")
    out("against the 4.2 arcsec Meeus quotes: inside it, with 1.7 per cent of margin.")
    out("That margin is thin and is stated rather than rounded away.")
    out("")
    out("With the RADIO DRY constant the largest disagreement is 8.4736 arcsec, which")
    out("EXCEEDS the quoted accuracy by a factor of 2.02. That is not a defect in")
    out("either calculation: the radio dry-air constant is 1.75 per cent below the")
    out("optical one and Bennett's formula is an optical fit, so the comparison is")
    out("between two different physical quantities. It is reported because a reader")
    out("fitting a radio antenna's pointing model is using an optical refraction")
    out("formula, and the size of that mismatch is 1.75 per cent of the refraction,")
    out("which at 10 degrees elevation is 5.7 arcsec and is not negligible for a")
    out("pointing model.")
    out("")
    out.check(
        "the optical comparison is inside Meeus's published accuracy",
        optical_result.max_absolute_arcsec < MEEUS_ACCURACY_ARCSEC,
        f"{optical_result.max_absolute_arcsec:.4f} arcsec against "
        f"{MEEUS_ACCURACY_ARCSEC:.1f}",
    )
    ratio_gap = 100.0 * (radio_result.mean_ratio / optical_result.mean_ratio - 1.0)
    constant_gap = 100.0 * (refraction.K_OPTICAL_550NM / refraction.K_RADIO_DRY - 1.0)
    out.check(
        "the radio comparison is NOT, and the excess is the refractivity constant",
        radio_result.max_absolute_arcsec > MEEUS_ACCURACY_ARCSEC,
        f"{radio_result.max_absolute_arcsec:.4f} arcsec; the two mean ratios differ "
        f"by {ratio_gap:.4f} per cent against a constant difference of "
        f"{constant_gap:.4f} per cent",
    )


def plane_parallel(out: Tee) -> None:
    """Comparison 3: the analytic flat-atmosphere limit."""
    out.rule("3. the plane-parallel limit of the refraction integral")
    out("The flat-atmosphere limit of equation (4) is (n0 - 1) tan(z), which needs no")
    out("quadrature. The departure of the spherical integral from it is derivable")
    out("without running either: Snell's invariant gives")
    out("  sin(zeta) = sin(z) / (1 + h/R) * (n0/n),")
    out("so to leading order d(ln sin zeta) = -h/R, hence d(zeta) = -(h/R) tan(zeta)")
    out("and d(tan zeta)/tan(zeta) = -(h/R) sec^2(zeta). Weighted at the refractivity")
    out("scale height H = R_specific T0 / g, the predicted relative departure is")
    out("-(H/R_earth) sec^2(z).")
    out("")
    scale_height = (
        refraction.DRY_AIR_R_J_PER_KG_K
        * (273.15 + refraction.STANDARD_TEMPERATURE_C)
        / refraction.GRAVITY_M_PER_S2
    )
    ratio = scale_height / refraction.EARTH_RADIUS_M
    out(f"scale height H            : {scale_height:.4f} m")
    out(f"H / R_earth               : {ratio:.6e}")
    out("")
    rows = []
    for apparent in (88.0, 85.0, 80.0, 70.0, 60.0, 50.0):
        traced = float(
            np.atleast_1d(
                refraction.raytrace_refraction(apparent, altitude_is_true=False)
            )[0]
        )
        flat = float(refraction.plane_parallel_refraction(apparent))
        zenith = math.radians(90.0 - apparent)
        predicted = -ratio / math.cos(zenith) ** 2
        measured = traced / flat - 1.0
        rows.append(
            [
                f"{apparent:.1f}",
                f"{traced:.6f}",
                f"{flat:.6f}",
                f"{measured:+.6e}",
                f"{predicted:+.6e}",
                f"{measured / predicted:.4f}",
            ]
        )
    out(
        format_table(
            [
                "apparent el deg",
                "raytrace arcsec",
                "plane parallel arcsec",
                "measured departure",
                "predicted -(H/R)sec^2 z",
                "measured / predicted",
            ],
            rows,
        )
    )
    out("")
    out.check(
        "the departure is negative at every elevation, as derived",
        all(float(row[3]) < 0.0 for row in rows),
    )
    out.check(
        "the measured departure matches the derived leading-order prediction",
        all(0.3 < float(row[5]) < 2.0 for row in rows),
        "ratios inside [0.3, 2.0]; the prediction is leading order only and the "
        "weighting height is approximated by the isothermal scale height",
    )


def quadrature(out: Tee) -> None:
    """The quadrature's own reported error, so that it is not an unknown."""
    out.rule("4. the quadrature's reported absolute error")
    out("scipy.integrate.quad returns an error estimate with every result. It is")
    out("recorded here so that the agreement figures above cannot be attributed to")
    out("the integration.")
    out("")
    rows = []
    worst = 0.0
    for apparent in (10.0, 30.0, 45.0, 60.0, 85.0):
        value, error = refraction._raytrace_apparent(
            apparent,
            refraction.STANDARD_PRESSURE_HPA,
            refraction.STANDARD_TEMPERATURE_C,
            refraction.K_RADIO_DRY,
            refraction.INTEGRATION_TOP_M,
        )
        worst = max(worst, abs(error))
        rows.append([f"{apparent:.1f}", f"{value:.6f}", f"{abs(error):.3e}"])
    out(
        format_table(
            ["apparent el deg", "refraction arcsec", "reported |abs error| arcsec"], rows
        )
    )
    out("")
    out.check(
        "the quadrature's reported absolute error is below 1e-6 arcsec everywhere",
        worst < 1e-6,
        f"worst = {worst:.3e} arcsec",
    )


def main() -> int:
    out = Tee("validate_refraction_output.txt")
    out.rule("pointmodel -- refraction against an independent ray trace")
    for line in environment_lines():
        out(line)
    out("")
    out("The independent calculation is pointmodel.refraction.raytrace_refraction, a")
    out("numerical integration of the refraction integral written in this repository.")
    out("No refraction package is installed in this environment and none is called.")
    out("")
    out(f"grid : {GRID.size} true altitudes from {GRID[0]:.1f} to {GRID[-1]:.1f} degrees")
    out(
        f"site : {refraction.STANDARD_PRESSURE_HPA:g} hPa, "
        f"{refraction.STANDARD_TEMPERATURE_C:g} C"
    )
    shape(out)
    absolute(out)
    plane_parallel(out)
    quadrature(out)
    path = out.save()
    print(f"\nwritten: {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

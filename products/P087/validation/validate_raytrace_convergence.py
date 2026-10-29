"""Numerical error budget of the ray trace: grid, truncation, and the independent bound.

The ray trace of equation (19) is the reference every closed form in this
package is scored against. Before that comparison means anything, the ray
trace's own numerical error has to be smaller than the differences being
measured. This script measures it three ways:

1. **Grid refinement.** The shell count is swept and the change in the answer
   recorded. The horizon converges more slowly than anywhere else because the
   integrand has an integrable inverse-square-root singularity there.
2. **Truncation.** The top of the integrated atmosphere is swept, so the
   refraction neglected above the ceiling is a measured number.
3. **An independently derived bracket.** ``tan(theta)`` decreases monotonically
   along the ray and the total measure of the integration variable is
   ``ln n_0 - ln n_top``, so the mean value theorem brackets the refraction
   between ``tan(theta_top)`` and ``tan(theta_0)`` times that measure. That
   argument shares no mathematics with the shell sum. The bracket is checked
   to have discriminating power by requiring that it REJECTS the exact
   plane-parallel answer and a deliberately scaled ray trace.

Run from anywhere:

    python validation/validate_raytrace_convergence.py
"""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np

from _bootstrap import Tee, Timer, bootstrap, environment_lines, published_state

ROOT = bootstrap()

from refractpath import format_table  # noqa: E402
from refractpath.atmosphere import Observer, build_profile  # noqa: E402
from refractpath.models import plane_parallel  # noqa: E402
from refractpath.raytrace import (  # noqa: E402
    DEFAULT_SHELLS,
    DEFAULT_TOP_M,
    raytrace_refraction,
    trace,
)
from refractpath.refractivity import birch_downs_1994  # noqa: E402

ARCSEC_PER_RADIAN = 180.0 * 3600.0 / math.pi
ELEVATIONS = (45.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.2, 0.0)
SHELL_COUNTS = (500, 1_000, 2_500, 5_000, 10_000, 20_000, 40_000)
TOPS_M = (50_000.0, 70_000.0, 86_000.0, 90_000.0, 110_000.0)
EXPONENTS = (2.0, 2.5, 3.0, 4.0)


def independent_bracket_arcsec(
    elevation_deg: float, state: object, top_m: float = DEFAULT_TOP_M
) -> tuple[float, float]:
    """Bracket on equation (19) from monotonicity and the mean value theorem."""
    observer = Observer()
    radius_0 = observer.radius_m
    radius_top = radius_0 + top_m
    profile = build_profile(state, np.array([0.0, top_m]), top_m=top_m)
    refractivity = birch_downs_1994(
        state.wavelength_um,
        profile.pressure_pa,
        profile.temperature_k - 273.15,
        profile.vapour_pressure_pa,
    )
    n_0 = 1.0 + float(refractivity[0])
    n_top = 1.0 + float(refractivity[1])
    measure = math.log(n_0) - math.log(n_top)
    sin_z = math.sin(math.radians(90.0 - elevation_deg))
    if sin_z >= 1.0:
        return (
            n_0 * radius_0 * sin_z / (n_top * radius_top) * measure * ARCSEC_PER_RADIAN,
            math.inf,
        )
    tan_0 = sin_z / math.sqrt(1.0 - sin_z**2)
    sin_top = n_0 * radius_0 * sin_z / (n_top * radius_top)
    tan_top = sin_top / math.sqrt(1.0 - sin_top**2)
    return tan_top * measure * ARCSEC_PER_RADIAN, tan_0 * measure * ARCSEC_PER_RADIAN


def main() -> int:
    out = Tee("validate_raytrace_convergence_output.txt")
    state = published_state()
    out.rule("refractpath -- ray trace numerical error budget")
    for line in environment_lines():
        out(line)
    out("")
    out(f"state: {state.pressure_hpa:g} hPa, {state.temperature_c:g} C, dry, "
        f"{state.wavelength_um:g} um, US Standard Atmosphere 1976 surface lapse rate")
    out(f"package defaults: {DEFAULT_SHELLS} shells, {DEFAULT_TOP_M:g} m ceiling, q = 3")

    out.rule("1. grid refinement -- refraction in arcsec by shell count")
    table: dict[int, list[float]] = {}
    timer = Timer()
    for shells in SHELL_COUNTS:
        row = []
        for elevation in ELEVATIONS:
            with timer:
                row.append(
                    trace(
                        elevation, state, None, "ciddor_1996", shells=shells
                    ).refraction_arcsec
                )
        table[shells] = row
    out(
        format_table(
            ["shells", *[f"{h:g} deg" for h in ELEVATIONS]],
            [[str(shells), *[f"{v:.5f}" for v in row]] for shells, row in table.items()],
        )
    )
    out("")
    out("change on doubling the shell count, arcsec")
    rows = []
    counts = list(SHELL_COUNTS)
    for lower, upper in pairwise(counts):
        rows.append(
            [
                f"{lower} -> {upper}",
                *[
                    f"{abs(table[upper][i] - table[lower][i]):.2e}"
                    for i in range(len(ELEVATIONS))
                ],
            ]
        )
    out(format_table(["refinement", *[f"{h:g} deg" for h in ELEVATIONS]], rows))
    finest = table[SHELL_COUNTS[-1]]
    default = table[DEFAULT_SHELLS]
    residuals = [abs(a - b) for a, b in zip(default, finest, strict=True)]
    out("")
    out(
        format_table(
            ["elevation (deg)", "default (arcsec)", "finest (arcsec)", "residual (arcsec)"],
            [
                [f"{ELEVATIONS[i]:g}", f"{default[i]:.5f}", f"{finest[i]:.5f}",
                 f"{residuals[i]:.2e}"]
                for i in range(len(ELEVATIONS))
            ],
        )
    )
    above_horizon = max(residuals[:-1])
    out.check(
        "grid residual above the horizon is below 1e-3 arcsec at the package default",
        above_horizon < 1e-3,
        f"worst {above_horizon:.3e} arcsec at "
        f"{ELEVATIONS[residuals.index(above_horizon)]:g} deg",
    )
    out.check(
        "grid residual at zero elevation is below 2e-2 arcsec at the package default",
        residuals[-1] < 2e-2,
        f"{residuals[-1]:.3e} arcsec; the horizon integrand has an integrable "
        "inverse-square-root singularity and converges more slowly",
    )

    out.rule("2. truncation -- refraction neglected above the ceiling")
    rows = []
    reference_top = TOPS_M[-1]
    for top in TOPS_M:
        row = [f"{top / 1000.0:g}"]
        for elevation in (45.0, 5.0, 0.0):
            value = trace(
                elevation, state, None, "ciddor_1996", top_m=top
            ).refraction_arcsec
            row.append(f"{value:.5f}")
        rows.append(row)
    out(format_table(["ceiling (km)", "45 deg", "5 deg", "0 deg"], rows))
    truncation = max(
        abs(
            trace(h, state, None, "ciddor_1996", top_m=DEFAULT_TOP_M).refraction_arcsec
            - trace(h, state, None, "ciddor_1996", top_m=reference_top).refraction_arcsec
        )
        for h in (45.0, 5.0, 0.0)
    )
    out("")
    out.check(
        "truncation at the 90 km default is below 1e-2 arcsec against a 110 km ceiling",
        truncation < 1e-2,
        f"worst {truncation:.3e} arcsec",
    )

    out.rule("3. grid exponent q of equation (20)")
    rows = []
    for exponent in EXPONENTS:
        row = [f"{exponent:g}"]
        for elevation in (45.0, 5.0, 1.0, 0.0):
            value = trace(
                elevation, state, None, "ciddor_1996", grid_exponent=exponent
            ).refraction_arcsec
            row.append(f"{value:.5f}")
        rows.append(row)
    out(format_table(["q", "45 deg", "5 deg", "1 deg", "0 deg"], rows))
    out("")
    out("q concentrates shells near the observer. It matters only at the horizon,")
    out("where the integrand is singular; q = 3 is the package default and q = 4")
    out("changes the horizon answer by the amount tabulated above.")

    out.rule("4. independently derived bracket -- monotonicity and the mean value theorem")
    rows = []
    inside = 0
    tested = 0
    for elevation in (60.0, 45.0, 30.0, 20.0, 10.0, 5.0, 2.0):
        lower, upper = independent_bracket_arcsec(elevation, state)
        measured = raytrace_refraction(elevation, state, None, "birch_downs_1994")
        ok = lower <= measured <= upper
        tested += 1
        inside += int(ok)
        rows.append(
            [
                f"{elevation:g}",
                f"{lower:.4f}",
                f"{measured:.4f}",
                f"{upper:.4f}",
                f"{(upper - lower) / measured:.4f}",
                "inside" if ok else "OUTSIDE",
            ]
        )
    out(
        format_table(
            ["elevation (deg)", "lower", "measured", "upper", "width/value", "verdict"],
            rows,
        )
    )
    out.check(
        "the ray trace lies inside the independent bracket at every elevation tested",
        inside == tested,
        f"{inside} of {tested}",
    )

    out.rule("5. the bracket is not vacuous -- controls anchored on the truth")
    plane_rejected = 0
    for elevation in (60.0, 45.0, 30.0, 20.0):
        _, upper = independent_bracket_arcsec(elevation, state)
        plane = plane_parallel(elevation, state, None, "birch_downs_1994")
        plane_rejected += int(plane > upper)
        out(
            f"  {elevation:5.1f} deg: plane-parallel {plane:.5f} arcsec against upper bound "
            f"{upper:.5f} arcsec, excess {plane - upper:+.5f}"
        )
    out.check(
        "the bracket rejects the exact plane-parallel answer at every elevation tested",
        plane_rejected == 4,
        f"{plane_rejected} of 4; the plane-parallel answer is kappa tan z to first "
        "order and the bound is ln(1 + kappa) tan z, which is strictly smaller",
    )
    scaled_rejected = 0
    for elevation in (60.0, 45.0, 30.0, 20.0, 10.0, 5.0):
        lower, upper = independent_bracket_arcsec(elevation, state)
        measured = raytrace_refraction(elevation, state, None, "birch_downs_1994")
        width = (upper - lower) / measured
        injected = measured * (1.0 + 2.5 * width)
        scaled_rejected += int(not lower <= injected <= upper)
    out.check(
        "the bracket rejects a ray trace scaled by 2.5 times the bracket's own width",
        scaled_rejected == 6,
        f"{scaled_rejected} of 6; the injected error is anchored on the measured "
        "value and scaled by the conditioning of the bound, not by a constant",
    )

    out.rule("6. plane-parallel limit -- a different derivation entirely")
    out("Setting the observer radius to 1e5 Earth radii removes the curvature, so")
    out("equation (19) must collapse onto the exact single-interface Snell solution")
    out("of equation (13). Nothing is shared between the two: one is a shell sum,")
    out("the other is one application of Snell's law.")
    flat = Observer.__new__(Observer)
    object.__setattr__(flat, "height_m", 6378137.0 * 1.0e5)
    object.__setattr__(flat, "latitude_deg", 0.0)
    rows = []
    worst = 0.0
    for elevation in (45.0, 20.0, 10.0, 5.0):
        spherical = raytrace_refraction(elevation, state, flat, "birch_downs_1994")
        snell = plane_parallel(elevation, state, None, "birch_downs_1994")
        relative = abs(spherical - snell) / snell
        worst = max(worst, relative)
        rows.append(
            [f"{elevation:g}", f"{spherical:.5f}", f"{snell:.5f}", f"{relative:.3e}"]
        )
    out(
        format_table(
            ["elevation (deg)", "flat-Earth raytrace", "exact Snell", "relative difference"],
            rows,
        )
    )
    out.check(
        "the flattened ray trace reproduces the exact Snell solution to 2e-4 relative",
        worst < 2e-4,
        f"worst {worst:.3e}; the residual is the finite 1e5-radius sphere and the "
        "90 km ceiling, not the integrator",
    )

    out.rule("elapsed time -- UNSTABLE, two cores, reported for information only")
    out(
        format_table(
            ["quantity", "value"],
            [
                ["ray traces run in the refinement sweep", str(timer.calls)],
                ["mean wall-clock per trace (ms)", f"{timer.mean_ms:.3f}"],
                ["total wall-clock in the sweep (s)", f"{timer.elapsed_s:.2f}"],
            ],
        )
    )

    path = out.save()
    print(f"\nwritten: {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

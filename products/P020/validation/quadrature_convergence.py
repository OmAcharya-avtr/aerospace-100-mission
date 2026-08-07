"""Validation 4: quadrature convergence under grid refinement.

The package integrates with adaptive Gauss-Kronrod (QUADPACK) by default,
splitting the path at the profile's declared breakpoints.  This script shows
that the answer is converged by

1. refining a composite-Simpson grid (201 -> 64001 nodes) and reporting both
   the change between successive grids and the difference from the adaptive
   result, for r0 on all three standard models;
2. comparing the adaptive result for the *piecewise* SLC-Day model against the
   integral assembled by hand from the analytic value of each band, which is
   the only exact reference available for a discontinuous profile;
3. repeating (1) for the mu_(5/3) and mu_(5/6) moments, which weight high
   altitudes much more strongly than mu_0 and therefore converge differently.

Expected behaviour: fast convergence on the smooth Hufnagel-Valley model, slow
convergence on the discontinuous SLC models (Simpson's error estimate assumes a
smooth fourth derivative that does not exist at a jump).  That is a property of
the rule, not a defect, and it is the reason the default method is adaptive.

Writes quadrature_convergence_output.txt. Runtime ~60 s.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atmoprofile import (  # noqa: E402
    fried_parameter,
    grid_convergence,
    hv57,
    slc_day,
    slc_night,
    turbulence_moment,
)

OUT = Path(__file__).resolve().parent / "quadrature_convergence_output.txt"
LAM = 500e-9
NODES = (201, 801, 3201, 12801, 64001)


def slc_day_moment0_exact() -> float:
    """Analytic integral of the SLC-Day model, band by band (h in m)."""
    seg1 = 1.7e-14 * 18.5  # constant band
    seg2 = 3.13e-13 * (110.0**-0.05 - 18.5**-0.05) / -0.05  # h^-1.05
    seg3 = 1.3e-15 * (1500.0 - 110.0)  # constant band
    seg4 = 8.87e-7 * 0.5 * (1500.0**-2 - 7200.0**-2)  # h^-3
    seg5 = 2.0e-16 * 2.0 * (20000.0**0.5 - 7200.0**0.5)  # h^-0.5
    return seg1 + seg2 + seg3 + seg4 + seg5


def main() -> int:
    t0 = time.perf_counter()
    profiles = {"HV5/7": hv57(), "SLC-Day": slc_day(), "SLC-Night": slc_night()}
    lines = [
        "Quadrature convergence under grid refinement (atmoprofile 0.1.0)",
        "=" * 78,
        f"lambda = {LAM * 1e9:g} nm, vertical path 0-20 km, Simpson node counts {NODES}",
        "Reference for the 'rel err' column is the default adaptive (QUADPACK) result.",
        "",
        "1) Fried parameter r0",
    ]
    failures = 0

    for name, profile in profiles.items():
        reference = fried_parameter(profile, LAM)
        records = grid_convergence(
            lambda n, p=profile: fried_parameter(p, LAM, method="simpson", n_nodes=n),
            NODES,
            reference,
        )
        lines.append(f"   {name}: adaptive r0 = {reference * 100:.6f} cm")
        lines.append(
            f"   {'nodes':>8}{'r0 [cm]':>14}{'rel change':>14}{'rel err vs adaptive':>22}"
        )
        for r in records:
            change = "     -" if r.n_nodes == NODES[0] else f"{r.rel_change:.3e}"
            lines.append(
                f"   {r.n_nodes:>8}{r.value * 100:>14.8f}{change:>14}"
                f"{r.rel_error_vs_reference:>22.3e}"
            )
        finest = records[-1].rel_error_vs_reference
        ok = finest < 1e-3
        failures += 0 if ok else 1
        lines.append(
            f"   -> finest grid agrees with the adaptive result to {finest:.2e} "
            f"({'PASS' if ok else 'FAIL'}, tol 1e-3)"
        )
        lines.append("")

    lines.append("2) SLC-Day mu_0 against the hand-assembled analytic band sum")
    exact = slc_day_moment0_exact()
    adaptive = turbulence_moment(profiles["SLC-Day"], 0.0)
    rel = abs(adaptive - exact) / exact
    ok = rel < 1e-9
    failures += 0 if ok else 1
    lines.append(f"   analytic band sum   = {exact:.12e} m^(1/3)")
    lines.append(f"   adaptive quadrature = {adaptive:.12e} m^(1/3)")
    lines.append(f"   relative difference = {rel:.3e}  ({'PASS' if ok else 'FAIL'}, tol 1e-9)")
    simpson_64k = turbulence_moment(profiles["SLC-Day"], 0.0, method="simpson", n_nodes=64001)
    lines.append(
        f"   Simpson (64001 nodes) = {simpson_64k:.12e}, "
        f"relative difference {abs(simpson_64k - exact) / exact:.3e} "
        "- one to two orders worse than the adaptive rule, as expected for a "
        "discontinuous integrand"
    )
    lines.append("")

    lines.append("3) Higher moments (HV5/7): mu_(5/3) and mu_(5/6)")
    for power, label in ((5 / 3, "mu_5/3"), (5 / 6, "mu_5/6")):
        profile = profiles["HV5/7"]
        reference = turbulence_moment(profile, power)
        records = grid_convergence(
            lambda n, p=power: turbulence_moment(
                profiles["HV5/7"], p, method="simpson", n_nodes=n
            ),
            NODES,
            reference,
        )
        lines.append(f"   {label}: adaptive = {reference:.10e}")
        for r in records:
            change = "     -" if r.n_nodes == NODES[0] else f"{r.rel_change:.3e}"
            lines.append(
                f"   {r.n_nodes:>8}{r.value:>18.10e}{change:>14}"
                f"{r.rel_error_vs_reference:>16.3e}"
            )
        finest = records[-1].rel_error_vs_reference
        ok = finest < 1e-6
        failures += 0 if ok else 1
        lines.append(f"   -> {'PASS' if ok else 'FAIL'} (tol 1e-6 on the finest grid)")
        lines.append("")

    lines.append(f"Wall time: {time.perf_counter() - t0:.1f} s")
    lines.append(f"FAILURES: {failures}")
    text = "\n".join(lines) + "\n"
    OUT.write_text(text)
    sys.stdout.write(text)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

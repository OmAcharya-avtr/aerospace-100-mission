"""Measured validity regimes: boundary curves, not assertions.

The quadrature reference of :mod:`conjprob.quadrature` shares no algebra with
the analytic paths beyond the definition of the integral. Each analytic method
is evaluated against it over a grid in the three dimensionless parameters that
describe the whole family of encounters, and the result is reduced to the value
of ``R / sqrt(sigma_x sigma_y)`` at which the method's relative error first
exceeds a stated tolerance. Entries reported as nan are cases where the
tolerance was never exceeded over the swept range; they are printed rather than
clipped, because "the method held everywhere we looked" and "the method failed
at the end of the sweep" are different results.
"""

from __future__ import annotations

import numpy as np

from _bootstrap import Tee, bootstrap, environment_lines

bootstrap()

from conjprob.regimes import METHOD_NAMES, regime_sweep  # noqa: E402

RHO = np.logspace(-3.0, 0.8, 60)
ASPECTS = np.array([1.0, 2.0, 5.0, 10.0, 20.0, 30.0])
MAHALANOBIS = np.array([0.25, 0.5, 1.0, 2.0, 4.0])
PSI_VALUES = (0.0, 0.25 * np.pi, 0.5 * np.pi)
TOLERANCES = (1e-2, 1e-3)


def main() -> int:
    out = Tee("validate_regimes_output.txt")
    out.rule("conjprob validation: measured validity regimes")
    for line in environment_lines():
        out(line)
    out("")
    out(f"grid: R/sigma_eff over {RHO.size} logarithmic points from {RHO[0]:.3g} to {RHO[-1]:.3g}")
    out(f"      aspect ratios {list(ASPECTS)}")
    out(f"      Mahalanobis distances {list(MAHALANOBIS)}")
    out("      miss-vector direction psi = pi/4 unless stated")
    out("reference: conjprob.quadrature.pc_quadrature, adaptive 2-D polar quadrature")
    out("chan order: conjprob.analytic.chan_recommended_order(M), so the measured Chan")
    out("            error is its equivalent-area error and not a truncation artefact")
    failures = 0

    sweep = regime_sweep(RHO, ASPECTS, MAHALANOBIS)

    out.rule("1. worst error anywhere on the grid, by method")
    out(f"{'method':<12s} {'worst |rel. error|':>19s} {'verdict':>34s}")
    for name in METHOD_NAMES:
        worst = sweep.worst_error(name)
        if name == "foster":
            verdict = "must be <= 1e-8 (PASS)" if worst <= 1e-8 else "FAIL"
            ok = worst <= 1e-8
        elif name == "alfano":
            verdict = "Simpson floor, measured" if worst <= 1e-2 else "FAIL"
            ok = worst <= 1e-2
        else:
            verdict = "approximation error, measured"
            ok = True
        out(f"{name:<12s} {worst:>19.6e} {verdict:>34s}")
        failures += 0 if ok else 1
    out("")
    out("quadrature error estimate over the grid:")
    out("  largest reported abserr relative to the value: not stored; see check 4")

    for tolerance in TOLERANCES:
        out.rule(
            f"2. measured R/sigma_eff at which the relative error first exceeds "
            f"{tolerance:.0e}"
        )
        for name in METHOD_NAMES:
            boundary = sweep.boundary(name, tolerance)
            finite = boundary[np.isfinite(boundary)]
            out("")
            out(f"{name}")
            header = "  aspect  " + "".join(f"M={m:<9.3g}" for m in MAHALANOBIS)
            out(header)
            for i, aspect in enumerate(ASPECTS):
                row = "".join(f"{boundary[i, j]:<11.4g}" for j in range(boundary.shape[1]))
                out(f"  {aspect:<8.3g}{row}")
            if finite.size:
                out(
                    f"  boundary spans {finite.min():.4g} to {finite.max():.4g}; "
                    f"{int(np.count_nonzero(~np.isfinite(boundary)))} of {boundary.size} "
                    "cells never exceeded the tolerance"
                )
            else:
                out(
                    f"  the tolerance was never exceeded in any of {boundary.size} cells "
                    "over the swept range"
                )

    out.rule("3. dependence on the miss-vector direction in the principal frame")
    out("psi = 0 puts the miss on the major axis, psi = pi/2 on the minor axis")
    out("boundary at a 1 per cent tolerance, aspect 10, Mahalanobis 1")
    out(f"{'psi [rad]':>10s} {'chan':>12s} {'small_body':>12s}")
    for psi in PSI_VALUES:
        single = regime_sweep(RHO, np.array([10.0]), np.array([1.0]), psi_rad=psi)
        out(
            f"{psi:>10.5f} {single.boundary('chan', 1e-2)[0, 0]:>12.4g} "
            f"{single.boundary('small_body', 1e-2)[0, 0]:>12.4g}"
        )

    out.rule("4. quadrature reference self-report")
    out("the reference's own absolute error estimate, relative to its value")
    out(f"{'R/sigma_eff':>12s} {'M':>6s} {'aspect':>7s} {'pc':>15s} {'abserr/pc':>12s}")
    from conjprob.regimes import evaluate_methods, geometry_from_dimensionless

    worst_ratio = 0.0
    for ratio in (0.001, 0.01, 0.1, 1.0, 5.0):
        for mahalanobis in (0.5, 2.0, 6.0):
            for aspect in (1.0, 10.0):
                geometry = geometry_from_dimensionless(ratio, aspect, mahalanobis)
                values = evaluate_methods(*geometry)
                reference = values["reference"]
                if reference <= 0.0:
                    continue
                relative = values["reference_abserr"] / reference
                worst_ratio = max(worst_ratio, relative)
                out(
                    f"{ratio:>12.4g} {mahalanobis:>6.2f} {aspect:>7.3g} "
                    f"{reference:>15.6e} {relative:>12.3e}"
                )
    out("")
    out(f"worst reported abserr / pc over this sample: {worst_ratio:.6e}")
    ok = worst_ratio <= 1e-8
    out(f"tolerance: <= 1e-8; result: {'PASS' if ok else 'FAIL'}")
    failures += 0 if ok else 1

    out.rule("summary")
    out(f"checks failed: {failures}")
    out(f"written to {out.save()}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

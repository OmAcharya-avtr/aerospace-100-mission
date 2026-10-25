"""How much of a divergence curve is the B* inversion assumption.

The independent propagator needs ``C_D A / m``, and the only thing a two-line
element set carries is ``B*``. The inversion ``C_D A / m = 2 B* / rho_0`` with
``rho_0 = 0.157 kg/(m^2 ER)`` needs no assumed drag coefficient, because
``C_D`` cancels, but it does assume that ``rho_0`` and that an operational
``B*`` is a physical ballistic parameter. It is not: orbit determination lets
``B*`` absorb whatever else the fit could not explain.

So this script answers the question a reader should ask: if the drag parameter
is wrong by a factor of two in either direction, how much does the reported
divergence move? The answer bounds the part of every drag-driven number in this
repository that is the assumption rather than the measurement.
"""

from __future__ import annotations

import sys

from _bootstrap import Tee, Timer, bootstrap, environment_lines, is_fast

bootstrap()

import numpy as np  # noqa: E402

from orbitdrift.constants import BSTAR_REFERENCE_DENSITY_KG_M2_ER  # noqa: E402
from orbitdrift.dataset import load_population  # noqa: E402
from orbitdrift.features import features_from_satrec  # noqa: E402
from orbitdrift.forces import ForceModel, cd_area_over_mass_m2_kg  # noqa: E402

FACTORS = (0.5, 1.0, 2.0)
STEPS_PER_REV = 120
"""Coarser than the 360 the committed dataset uses, and deliberately so.

This study reports a RATIO of two divergences computed with the same step, so
the truncation error largely cancels out of it, and the saving brings the
script from 378 s to inside a third of the 600 s per-script budget. The
absolute divergences printed below are therefore not the committed ones; the
committed ones are in fixtures/divergence_dataset.npz at 360 steps per
revolution.
"""


def main() -> int:
    out = Tee("validate_bstar_sensitivity_output.txt")
    out.rule("orbitdrift -- sensitivity of the divergence to the B* inversion")
    for line in environment_lines():
        out(line)

    out.rule("the inversion")
    out(f"  reference density rho_0       : {BSTAR_REFERENCE_DENSITY_KG_M2_ER} kg/(m^2 ER)")
    out("  relation                      : B* = (1/2) C_D (A/m) rho_0")
    out("  what drag needs               : C_D A/m = 2 B* / rho_0")
    ratio = 2.0 / BSTAR_REFERENCE_DENSITY_KG_M2_ER
    out(f"  numerically                   : C_D A/m = {ratio:.4f} B*")
    out("  C_D cancels, so no drag coefficient is assumed. rho_0 is nominal and an")
    out("  operational B* is not a physical ballistic parameter, so the product is")
    out("  an assumption and this script bounds its effect.")
    out("")
    out("  B*          C_D A/m m^2/kg")
    for bstar in (1e-6, 1e-5, 1e-4, 1e-3, 5e-3, -1e-4):
        out(f"  {bstar:+.1e}    {float(cd_area_over_mass_m2_kg(bstar)):+.6e}")

    population = load_population((bootstrap() / "fixtures" / "population.tle").read_text())
    stride = max(1, len(population) // (2 if is_fast() else 8))
    sample = population[::stride]
    span_days = 0.1 if is_fast() else 2.0
    timer = Timer()

    out.rule(f"divergence at {span_days} days under a factor of two in C_D A/m")
    out(
        "  object  perigee km   B*          log10 dragpar   0.5x km     1.0x km"
        "     2.0x km   span/1.0x"
    )
    spans = []
    drag_parameters = []
    for satrec in sample:
        features = features_from_satrec(satrec)
        base = float(cd_area_over_mass_m2_kg(float(satrec.bstar)))
        # ``Satrec`` is immutable in the accelerated build, so the drag parameter
        # cannot be changed by editing the element set. _curve_with_coefficient
        # drives the propagation with an explicit coefficient instead, which is
        # the same code path model_divergence uses with the value the element set
        # implies.
        values = []
        for factor in FACTORS:
            with timer:
                values.append(_curve_with_coefficient(satrec, base * factor, span_days))
        spread = (max(values) - min(values)) / values[1]
        spans.append(spread)
        drag_parameters.append(features.log10_drag_parameter)
        out(
            f"  {int(satrec.satnum) - 70000:6d}  {features.perigee_altitude_km:10.1f}   "
            f"{satrec.bstar:+.2e}  {features.log10_drag_parameter:13.4f}  "
            f"{values[0]:10.4f}  {values[1]:10.4f}  {values[2]:10.4f}  {spread:9.4f}"
        )

    out("")
    out(f"  median span / nominal         : {float(np.median(spans)):.4f}")
    out(f"  worst span / nominal          : {float(np.max(spans)):.4f}")
    out("")
    out("  Read this as: a factor of two error in the drag parameter moves the")
    out("  reported divergence by this fraction of itself. Where the fraction is")
    out("  small the curve is dominated by the difference in dynamical structure")
    out("  between SGP4 and the numerical model, which is the signal the product")
    out("  is about. Where it is large the curve is dominated by the drag")
    out("  assumption, and the number should be read as conditional on it.")

    out.rule("checks")
    out.check(
        "the drag parameter inversion is exactly 2 B* / rho_0",
        float(cd_area_over_mass_m2_kg(1e-4)) == 2.0e-4 / BSTAR_REFERENCE_DENSITY_KG_M2_ER,
        f"{float(cd_area_over_mass_m2_kg(1e-4)):.12e}",
    )
    out.check(
        "a negative B* passes through to a negative drag parameter, not zero",
        float(cd_area_over_mass_m2_kg(-1e-4)) < 0.0,
        f"{float(cd_area_over_mass_m2_kg(-1e-4)):.6e}",
    )
    if not is_fast():
        out.check(
            "a factor of two in the drag parameter moves the median divergence by "
            "less than half of itself",
            float(np.median(spans)) < 0.5,
            f"median {float(np.median(spans)):.4f}, worst {float(np.max(spans)):.4f}",
        )
    out("")
    out("  The sensitivity is not a function of altitude alone. The population draws")
    out("  B* independently of altitude, so a 500 km object with B* = 1e-6 has")
    out("  almost no drag and almost no sensitivity. The quantity that orders the")
    out("  table is log10(C_D A/m * rho_perigee), the combination that actually")
    out("  appears in the drag acceleration, and the correlation below is the")
    out("  physical statement.")
    order = np.argsort(drag_parameters)
    log_spans = np.log10(np.maximum(np.array(spans)[order], 1e-12))
    correlation = float(np.corrcoef(np.array(drag_parameters)[order], log_spans)[0, 1])
    out("")
    out(f"  correlation of log10(sensitivity) with log10(drag parameter): {correlation:.4f}")
    heaviest = int(np.argmax(drag_parameters))
    lightest = int(np.argmin(drag_parameters))
    out(
        f"  heaviest drag parameter {drag_parameters[heaviest]:.3f}: "
        f"sensitivity {spans[heaviest]:.6f}"
    )
    out(
        f"  lightest drag parameter {drag_parameters[lightest]:.3f}: "
        f"sensitivity {spans[lightest]:.6f}"
    )
    if is_fast():
        out("")
        out("  probe mode: the sample is too small and the span too short for the")
        out("  correlation and magnitude checks below, which are skipped. Only the")
        out("  exit status is being probed.")
        out.check(
            "every sensitivity is finite and non-negative",
            bool(np.all(np.isfinite(spans)) and np.all(np.array(spans) >= 0.0)),
            f"{len(spans)} objects",
        )
    else:
        out.check(
            "the sensitivity rises with the drag parameter (correlation above 0.7)",
            correlation > 0.7,
            f"{correlation:.4f} over {len(spans)} objects",
        )
        out.check(
            "the most drag-affected object in the sample is materially sensitive, so "
            "the drag term is reaching the integrator",
            spans[heaviest] > 1e-2,
            f"{spans[heaviest]:.6f} at log10 drag parameter {drag_parameters[heaviest]:.3f}",
        )
        out.check(
            "the least drag-affected object is insensitive, which is the expected "
            "result and not a defect",
            spans[lightest] < 1e-3,
            f"{spans[lightest]:.6e} at log10 drag parameter {drag_parameters[lightest]:.3f}",
        )

    out.rule("timings (UNSTABLE: wall clock on a shared two-core container)")
    out(f"  propagations                  : {timer.calls}, {timer.elapsed_s:.1f} s total")

    out("")
    out(f"wrote {out.save()}")
    return 1 if out.failures else 0


def _curve_with_coefficient(satrec, coefficient: float, span_days: float) -> float:
    """Divergence at ``span_days`` with the drag parameter forced to ``coefficient``."""
    import numpy as np

    from orbitdrift.divergence import sgp4_states, window_rms
    from orbitdrift.features import epoch_jd, period_s_from_satrec
    from orbitdrift.integrate import propagate

    period = period_s_from_satrec(satrec)
    step = period / STEPS_PER_REV
    n_steps = int(round(span_days * 86400.0 / step))
    n_steps -= n_steps % 6
    code, r0, v0 = satrec.sgp4(float(satrec.jdsatepoch), float(satrec.jdsatepochF))
    assert code == 0
    trajectory = propagate(
        np.asarray(r0)[None, :],
        np.asarray(v0)[None, :],
        epoch_jd(satrec),
        np.array([coefficient]),
        n_steps=n_steps,
        step_s=step,
        force=ForceModel(),
        sample_every=6,
    )
    reference, _ = sgp4_states(satrec, trajectory.times_s)
    magnitude = np.linalg.norm(trajectory.positions_km[0] - reference, axis=-1)
    age = trajectory.times_s[-1] - 0.5 * period
    return float(window_rms(trajectory.times_s, magnitude, period, np.array([age]))[0])


if __name__ == "__main__":
    sys.exit(main())

"""Validate the identifiability analysis against exactly known answers.

Run from the repository root: ``python validation/validate_identifiability.py``
Writes ``validation/validate_identifiability_output.txt``.

The two known answers:

1. ``LinearNodeModel``: the solution depends on ``(C, k, q)`` only through
   ``q/k`` and ``k/C``, so a common scaling leaves every prediction unchanged
   and the null direction in log-parameter coordinates is exactly
   ``(1, 1, 1)/sqrt(3)``. Rank 2 of 3, for every design.
2. ``ThermalTwinModel`` on the ``full-sun`` design: ``q_sun`` and ``q_par``
   enter only through ``q_sun s0 + q_par``, so with ``s0 = 1`` the null
   direction is proportional to ``(q_par, -q_sun) = (18, -55)`` on those two
   coordinates and zero on the other four. Rank 5 of 6.

References: Jacquez and Greif (1985), *Mathematical Biosciences* 77, 201-227
(rank of the sensitivity matrix as the local identifiability test); Golub and
Van Loan, *Matrix Computations*, 4th ed., 2013, sections 2.4 and 5.4 (SVD,
numerical rank, condition number); Brun, Reichert and Kuensch (2001), *Water
Resources Research* 37(4), 1015-1030 (collinearity index); Rao (1945) and
Cramer (1946) (the information inequality).
"""

from __future__ import annotations

import time

import numpy as np

from _bootstrap import Tee, add_src_to_path

ROOT = add_src_to_path()

from twincalib import (  # noqa: E402
    DESIGN_NAMES,
    analyse,
    build_inputs,
    calibration_model,
    design,
    linear_design,
    linear_model,
    log_sensitivity,
    render_correlation,
    worst_subsets,
)
from twincalib.scenarios import LINEAR_THETA_TRUE  # noqa: E402


def main() -> None:
    started = time.perf_counter()
    out = Tee(ROOT / "validation" / "validate_identifiability_output.txt")
    out("validate_identifiability.py -- rank, null directions, collinearity")

    out.rule("1. Known answer: the single-node model's exact null direction")
    model = linear_model()
    inputs = linear_design(121, 25.0)
    sens = log_sensitivity(
        model, LINEAR_THETA_TRUE, inputs, [300.0], [0], 0.3
    )
    report = analyse(sens)
    expected = np.full(3, 1.0 / np.sqrt(3.0))
    out(f"parameters            : {report.parameter_names}")
    out(f"numerical rank        : {report.rank} of 3   expected 2")
    ratios = report.singular_values / report.singular_values[0]
    out(f"singular value ratios : {', '.join(f'{r:.4e}' for r in ratios)}")
    out(f"condition number      : {report.condition_number:.6e}")
    assert len(report.null_combinations) == 1
    vector = report.null_combinations[0].vector
    aligned = vector if float(vector @ expected) > 0.0 else -vector
    worst = float(np.max(np.abs(aligned - expected)))
    out("")
    out(f"expected null direction : {', '.join(f'{v:+.9f}' for v in expected)}")
    out(f"measured null direction : {', '.join(f'{v:+.9f}' for v in aligned)}")
    out(f"worst component error   : {worst:.3e}   tolerance 1e-6")
    out(f"verdict: {'PASS' if report.rank == 2 and worst < 1e-6 else 'FAIL'}")
    out("")
    out("Direct consequence, checked independently of the SVD: scaling all three")
    out("parameters by a common factor must not move the trajectory.")
    base = model.observe(LINEAR_THETA_TRUE, inputs, [300.0], [0])
    out(f"  {'factor':>8}{'worst |T(scaled) - T(base)| [K]':>34}")
    worst_scale = 0.0
    for factor in (0.25, 0.5, 2.0, 4.0, 10.0):
        scaled = model.observe(
            np.asarray(LINEAR_THETA_TRUE, dtype=float) * factor, inputs, [300.0], [0]
        )
        diff = float(np.max(np.abs(scaled - base)))
        worst_scale = max(worst_scale, diff)
        out(f"  {factor:>8.2f}{diff:>34.6e}")
    out("")
    out(f"worst over all factors = {worst_scale:.3e} K   tolerance 1e-5 K")
    out(f"verdict: {'PASS' if worst_scale < 1e-5 else 'FAIL'}")

    out.rule("2. Known answer: the full-sun design's exact null direction")
    twin = calibration_model()
    d = design("full-sun")
    sens = log_sensitivity(
        twin,
        d.theta_true,
        build_inputs(d),
        d.initial_state,
        d.channels,
        d.sigma_k,
        d.sample_stride,
    )
    report = analyse(sens)
    expected = np.zeros(6)
    expected[4] = d.theta_true[5]
    expected[5] = -d.theta_true[4]
    expected /= np.linalg.norm(expected)
    vector = report.null_combinations[0].vector
    aligned = vector if float(vector @ expected) > 0.0 else -vector
    worst = float(np.max(np.abs(aligned - expected)))
    out(f"numerical rank        : {report.rank} of 6   expected 5")
    out(f"condition number      : {report.condition_number:.6e}")
    out(f"collinearity index    : {report.collinearity:.6e}")
    out("")
    out(f"  {'parameter':<10}{'expected':>14}{'measured':>14}{'difference':>14}")
    for name, e, m in zip(report.parameter_names, expected, aligned, strict=True):
        out(f"  {name:<10}{e:>+14.9f}{m:>+14.9f}{m - e:>+14.3e}")
    out("")
    out(f"worst component error = {worst:.3e}   tolerance 1e-5")
    out(f"verdict: {'PASS' if report.rank == 5 and worst < 1e-5 else 'FAIL'}")
    out("")
    out("Direct consequence: moving heat from the solar load to the parasitic load")
    out("must not move the trajectory, because only the sum enters.")
    base = twin.observe(
        d.theta_true, build_inputs(d), d.initial_state, d.channels, d.sample_stride
    )
    out(f"  {'shift [W]':>10}{'worst |T(shifted) - T(base)| [K]':>36}")
    worst_shift = 0.0
    for shift in (1.0, 5.0, 10.0, 17.0):
        moved = list(d.theta_true)
        moved[4] -= shift
        moved[5] += shift
        other = twin.observe(
            moved, build_inputs(d), d.initial_state, d.channels, d.sample_stride
        )
        diff = float(np.max(np.abs(other - base)))
        worst_shift = max(worst_shift, diff)
        out(f"  {shift:>10.1f}{diff:>36.6e}")
    out("")
    out(f"worst over all shifts = {worst_shift:.3e} K   tolerance 1e-9 K")
    out(f"verdict: {'PASS' if worst_shift < 1e-9 else 'FAIL'}")

    out.rule("3. Rank verdicts do not balance on the tolerance")
    out("The rank cut must not be the thing that decides the verdict.")
    out("")
    tols = (1e-5, 1e-6, 1e-8, 1e-10, 1e-12)
    out(f"  {'design':<18}" + "".join(f"{f'tol {t:.0e}':>12}" for t in tols))
    smallest = {}
    for name in DESIGN_NAMES:
        dd = design(name)
        ss = log_sensitivity(
            twin,
            dd.theta_true,
            build_inputs(dd),
            dd.initial_state,
            dd.channels,
            dd.sigma_k,
            dd.sample_stride,
        )
        ranks = [analyse(ss, rank_tol=t).rank for t in tols]
        sv = ss.weighted
        ratios = np.linalg.svd(sv, compute_uv=False)
        smallest[name] = float(ratios[-1] / ratios[0])
        out(f"  {name:<18}" + "".join(f"{r:>12d}" for r in ranks))
    out("")
    out(f"  {'design':<18}{'smallest s/s_max':>20}{'decades from 1e-8':>20}")
    for name, value in smallest.items():
        out(f"  {name:<18}{value:>20.4e}{np.log10(value / 1e-8):>20.2f}")
    out("")
    out("Read this table honestly. Within two decades of the default cut -- that")
    out("is, for tol in [1e-10, 1e-6] -- every verdict is stable: full-sun is rank")
    out("5 and the other four are rank 6. Outside that band two verdicts do move,")
    out("and both moves are expected from the spectrum above:")
    out("  tol = 1e-12 puts the cut BELOW full-sun's structural null at 1.06e-11,")
    out("    so that design is then called full rank. The cut has gone past the")
    out("    accuracy of the differentiation itself (validate_sensitivity.py check")
    out("    2 measures that floor at about 5e-08 relative), so 1e-12 is not a")
    out("    tolerance this machinery can support.")
    out("  tol = 1e-05 puts the cut ABOVE weak-excitation's smallest ratio of")
    out("    4.22e-06, so that design is then called rank deficient. That is a")
    out("    defensible reading of a design whose relative standard deviations run")
    out("    to several hundred per cent; it is a change of reporting threshold,")
    out("    not a disagreement about the numbers.")
    out("The default 1e-08 is two decades above the measured differentiation floor")
    out("and three decades above the structural null it has to detect.")

    out.rule("4. Every shipped design, in full")
    for name in DESIGN_NAMES:
        dd = design(name)
        ss = log_sensitivity(
            twin,
            dd.theta_true,
            build_inputs(dd),
            dd.initial_state,
            dd.channels,
            dd.sigma_k,
            dd.sample_stride,
        )
        rr = analyse(ss)
        out("")
        out(
            f"--- {name} ({dd.n_observations} observations, "
            f"channels {dd.channel_labels}) ---"
        )
        out(f"  {rr.verdict()}")
        out(f"  condition number   : {rr.condition_number:.6g}")
        out(f"  collinearity index : {rr.collinearity:.6g}")
        out(
            "  singular ratios    : "
            + ", ".join(f"{v:.4e}" for v in rr.singular_values / rr.singular_values[0])
        )
        out(
            "  relative sd        : "
            + ", ".join(
                f"{n}={v:.4g}" for n, v in zip(rr.parameter_names, rr.relative_sd, strict=True)
            )
        )
        for combo in rr.weak_combinations:
            tag = "UNCONSTRAINED" if combo.ratio < rr.rank_tol else "near-null"
            out(f"  [{tag}] s/s_max = {combo.ratio:.4e}")
            out(f"    {combo.as_text()}")
        out("  most collinear pairs:")
        for pair, value in worst_subsets(ss, 2, 3):
            out(f"    {' + '.join(pair):<20} {value:.4g}")

    out.rule("5. Correlation matrices, and the pairs that are nearly degenerate")
    for name in ("baseline-eclipse", "weak-excitation"):
        dd = design(name)
        ss = log_sensitivity(
            twin,
            dd.theta_true,
            build_inputs(dd),
            dd.initial_state,
            dd.channels,
            dd.sigma_k,
            dd.sample_stride,
        )
        rr = analyse(ss)
        out("")
        out(f"--- {name} ---")
        out(render_correlation(rr.parameter_names, rr.correlation))

    out.rule("6. Negative controls")
    dd = design("baseline-eclipse")
    ss = log_sensitivity(
        twin,
        dd.theta_true,
        build_inputs(dd),
        dd.initial_state,
        dd.channels,
        dd.sigma_k,
        dd.sample_stride,
    )
    checks = [
        ("rank_tol = 0", lambda: analyse(ss, rank_tol=0.0)),
        ("rank_tol = 1.5", lambda: analyse(ss, rank_tol=1.5)),
        ("weak_ratio = -1", lambda: analyse(ss, weak_ratio=-1.0)),
        ("subset size 1", lambda: worst_subsets(ss, size=1)),
        ("subset size 7", lambda: worst_subsets(ss, size=7)),
        ("top = 0", lambda: worst_subsets(ss, size=2, top=0)),
    ]
    failures = 0
    for label, call in checks:
        try:
            call()
            out(f"  {label:<20} NOT REJECTED  <- defect")
            failures += 1
        except ValueError as error:
            out(f"  {label:<20} rejected: {str(error)[:58]}")
    out("")
    out(f"verdict: {'PASS' if failures == 0 else 'FAIL'}  {len(checks)} negative controls")

    out("")
    out(f"elapsed {time.perf_counter() - started:.1f} s")
    out.save()


if __name__ == "__main__":
    main()

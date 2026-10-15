"""Validate the sensitivity machinery: closed forms, step study, Fisher matrix.

Run from the repository root: ``python validation/validate_sensitivity.py``
Writes ``validation/validate_sensitivity_output.txt``.

References:

* the closed-form sensitivities of the single-node model, derived in
  ``LinearNodeModel.analytic_sensitivity``'s docstring from the
  lumped-capacitance solution (Incropera and DeWitt, Chapter 5);
* Fornberg, B. (1988), *Mathematics of Computation* 51(184), 699-706 -- the
  central difference and its ``O(h^2)`` truncation error;
* Bard, Y., *Nonlinear Parameter Estimation*, Academic Press, 1974, eq. 7-7-5 --
  the Fisher information of a weighted least-squares problem is ``J^T Sigma^-1 J``;
* Brun, R., Reichert, P. and Kuensch, H. R. (2001), *Water Resources Research*
  37(4), 1015-1030 -- relative sensitivity scaling and ``delta_msqr``.
"""

from __future__ import annotations

import time

import numpy as np

from _bootstrap import Tee, add_src_to_path

ROOT = add_src_to_path()

from twincalib import (  # noqa: E402
    DEFAULT_LOG_STEP,
    build_inputs,
    calibration_model,
    design,
    fisher_information,
    linear_design,
    linear_model,
    log_sensitivity,
)
from twincalib.scenarios import LINEAR_THETA_TRUE  # noqa: E402

STEPS = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8)


def main() -> None:
    started = time.perf_counter()
    out = Tee(ROOT / "validation" / "validate_sensitivity_output.txt")
    out("validate_sensitivity.py -- sensitivity matrix, step study, Fisher matrix")

    out.rule("1. Central differences against the closed-form sensitivity")
    out("Reference: LinearNodeModel.analytic_sensitivity, differentiated by hand")
    out("from the closed-form lumped-capacitance solution. Scaled to log")
    out("coordinates by multiplying column j by theta_j.")
    model = linear_model()
    inputs = linear_design(121, 25.0)
    exact = model.analytic_sensitivity(
        LINEAR_THETA_TRUE, inputs.times, 300.0
    ) * np.asarray(LINEAR_THETA_TRUE, dtype=float)
    scale = float(np.max(np.abs(exact)))
    out("")
    out(f"largest |exact log-sensitivity| = {scale:.6f} K per log unit")
    out(f"  {'log step h':>12}{'worst abs error [K]':>22}{'relative to scale':>20}")
    best = None
    for h in STEPS:
        sens = log_sensitivity(
            model, LINEAR_THETA_TRUE, inputs, [300.0], [0], 0.3, log_step=h
        )
        err = float(np.max(np.abs(sens.jacobian_log - exact)))
        out(f"  {h:>12.0e}{err:>22.6e}{err / scale:>20.6e}")
        if best is None or err < best[1]:
            best = (h, err)
    out("")
    out(f"best step {best[0]:.0e} with worst error {best[1]:.3e} K")
    out(
        f"the default step is {DEFAULT_LOG_STEP:.0e}; the table shows a flat "
        "region from 1e-3 down,"
    )
    out("which is the error floor of the integrator rather than of the difference")
    out("formula, so the default sits inside the flat region by two decades.")
    default_err = float(
        np.max(
            np.abs(
                log_sensitivity(
                    model, LINEAR_THETA_TRUE, inputs, [300.0], [0], 0.3
                ).jacobian_log
                - exact
            )
        )
    )
    out(f"error at the default step = {default_err:.6e} K  ({default_err / scale:.3e} relative)")
    out(f"verdict: {'PASS' if default_err / scale < 1e-6 else 'FAIL'}  tolerance 1e-6 relative")

    out.rule("2. Step insensitivity on the two-node model")
    out("No closed form exists here, so the check is self-consistency: a hundred-")
    out("fold change in the step must not move the Jacobian.")
    d = design("baseline-eclipse")
    twin = calibration_model()
    reference = log_sensitivity(
        twin,
        d.theta_true,
        build_inputs(d),
        d.initial_state,
        d.channels,
        d.sigma_k,
        d.sample_stride,
        log_step=1e-4,
    )
    ref_scale = float(np.max(np.abs(reference.jacobian_log)))
    out("")
    out(f"largest |log-sensitivity| at the default step = {ref_scale:.4f} K")
    out(f"  {'log step h':>12}{'worst diff vs 1e-4 [K]':>25}{'relative':>14}")
    worst_in_band = 0.0
    for h in STEPS:
        sens = log_sensitivity(
            twin,
            d.theta_true,
            build_inputs(d),
            d.initial_state,
            d.channels,
            d.sigma_k,
            d.sample_stride,
            log_step=h,
        )
        diff = float(np.max(np.abs(sens.jacobian_log - reference.jacobian_log)))
        out(f"  {h:>12.0e}{diff:>25.6e}{diff / ref_scale:>14.3e}")
        if 1e-6 <= h <= 1e-2:
            worst_in_band = max(worst_in_band, diff / ref_scale)
    out("")
    out(f"worst relative movement over h in [1e-6, 1e-2] = {worst_in_band:.3e}")
    out(f"verdict: {'PASS' if worst_in_band < 1e-4 else 'FAIL'}  tolerance 1e-4")

    out.rule("3. Weighting identity")
    out("The weighted matrix must be the log Jacobian divided by sigma, exactly.")
    worst = float(
        np.max(np.abs(reference.weighted * reference.sigma[:, None] - reference.jacobian_log))
    )
    out(f"worst |J_w * sigma - J| = {worst:.3e} K   tolerance 1e-12")
    out(f"verdict: {'PASS' if worst < 1e-12 else 'FAIL'}")

    out.rule("4. Fisher information: symmetry, positive semidefiniteness, Gram identity")
    fisher = fisher_information(reference)
    asym = float(np.max(np.abs(fisher - fisher.T)))
    eig = np.linalg.eigvalsh(fisher)
    gram = reference.weighted.T @ reference.weighted
    gram_diff = float(np.max(np.abs(fisher - gram)) / np.max(np.abs(gram)))
    out(f"worst asymmetry                 = {asym:.3e}   tolerance 0 (symmetrised)")
    out(f"smallest eigenvalue             = {float(np.min(eig)):.6e}")
    out(f"largest eigenvalue              = {float(np.max(eig)):.6e}")
    out(f"smallest / largest              = {float(np.min(eig) / np.max(eig)):.6e}")
    out(f"relative difference from J_w^T J_w = {gram_diff:.3e}   tolerance 1e-9")
    ok = asym == 0.0 and float(np.min(eig)) > 0.0 and gram_diff < 1e-9
    out(f"verdict: {'PASS' if ok else 'FAIL'}")

    out.rule("5. Structural proportionality on the full-sun design")
    out("With s(t) = 1 the two heat loads enter only through q_sun + q_par, so")
    out("dy/d(ln q_sun) = (q_sun/q_par) dy/d(ln q_par), exactly.")
    ds = design("full-sun")
    sens = log_sensitivity(
        twin,
        ds.theta_true,
        build_inputs(ds),
        ds.initial_state,
        ds.channels,
        ds.sigma_k,
        ds.sample_stride,
    )
    factor = ds.theta_true[4] / ds.theta_true[5]
    col_sun = sens.jacobian_log[:, 4]
    col_par = sens.jacobian_log[:, 5]
    rel = float(np.max(np.abs(col_sun - factor * col_par)) / np.max(np.abs(col_sun)))
    out(f"predicted factor q_sun/q_par = {factor:.6f}")
    out(f"worst relative residual      = {rel:.3e}   tolerance 1e-7")
    out(f"verdict: {'PASS' if rel < 1e-7 else 'FAIL'}")
    out("")
    out("The same comparison on baseline-eclipse, where the eclipse separates them:")
    sens_b = reference
    factor_b = d.theta_true[4] / d.theta_true[5]
    rel_b = float(
        np.max(np.abs(sens_b.jacobian_log[:, 4] - factor_b * sens_b.jacobian_log[:, 5]))
        / np.max(np.abs(sens_b.jacobian_log[:, 4]))
    )
    out(f"worst relative residual      = {rel_b:.4f}   must exceed 0.1")
    out(f"verdict: {'PASS' if rel_b > 0.1 else 'FAIL'}")

    out.rule("6. Per-parameter sensitivity measure on every design")
    out("delta_msqr = ||J_w[:, j]|| / sqrt(n_obs): the root-mean-square change in")
    out("the data, in units of the measurement noise, for one natural log unit of")
    out("the parameter (Brun et al. 2001).")
    out("")
    names = twin.parameter_names
    header = f"  {'design':<18}" + "".join(f"{n:>11}" for n in names)
    out(header)
    for name in ("baseline-eclipse", "full-sun", "lumped-nodes", "coarse-sampled",
                 "weak-excitation"):
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
        out(f"  {name:<18}" + "".join(f"{v:>11.3f}" for v in ss.delta_msqr()))
    out("")
    out("Every parameter moves the data by many noise standard deviations per log")
    out("unit on every design, which is why the MAGNITUDE of the sensitivity is")
    out("not what goes wrong here. What goes wrong is collinearity, and that is")
    out("what validate_identifiability.py measures.")

    out.rule("7. Negative controls")
    checks = [
        ("log_step = 0", lambda: log_sensitivity(
            twin, d.theta_true, build_inputs(d), d.initial_state, d.channels,
            d.sigma_k, log_step=0.0)),
        ("log_step = 1", lambda: log_sensitivity(
            twin, d.theta_true, build_inputs(d), d.initial_state, d.channels,
            d.sigma_k, log_step=1.0)),
        ("sigma = 0", lambda: log_sensitivity(
            twin, d.theta_true, build_inputs(d), d.initial_state, d.channels, 0.0)),
        ("sigma wrong length", lambda: log_sensitivity(
            twin, d.theta_true, build_inputs(d), d.initial_state, d.channels,
            np.full(3, 0.3))),
    ]
    failures = 0
    for label, call in checks:
        try:
            call()
            out(f"  {label:<22} NOT REJECTED  <- defect")
            failures += 1
        except ValueError as error:
            out(f"  {label:<22} rejected: {str(error)[:62]}")
    out("")
    out(f"verdict: {'PASS' if failures == 0 else 'FAIL'}  {len(checks)} negative controls")

    out("")
    out(f"elapsed {time.perf_counter() - started:.1f} s")
    out.save()


if __name__ == "__main__":
    main()

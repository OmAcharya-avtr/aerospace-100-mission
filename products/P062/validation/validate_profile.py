"""Validate the profile-likelihood intervals: agreement, coverage, openness.

Run from the repository root: ``python validation/validate_profile.py``
Writes ``validation/validate_profile_output.txt``.

What is checked:

1. On a sound design the profile interval agrees with the quadratic (Hessian)
   interval, because the likelihood is near quadratic there. The ratio of
   widths is the number.
2. The deliberately non-identifiable design is REPORTED as such: the two heat
   loads get intervals that are open below, while their sum is determined.
3. The practically non-identifiable design, where the rank test passes.

Interval coverage over seeded repetitions is measured by its own script,
``validate_coverage.py``, so that each stays inside the three-minute budget.

References: Wilks (1938), *Annals of Mathematical Statistics* 9(1), 60-62 (the
chi-squared threshold); Raue et al. (2009), *Bioinformatics* 25(15), 1923-1929
(structural against practical non-identifiability by profile likelihood);
Clopper and Pearson (1934), *Biometrika* 26(4), 404-413 (the exact binomial
interval used by ``validate_coverage.py``).
"""

from __future__ import annotations

import time

import numpy as np

from _bootstrap import Tee, add_src_to_path

ROOT = add_src_to_path()

from twincalib import (  # noqa: E402
    CalibrationProblem,
    calibration_model,
    chi2_threshold,
    design,
    fit,
    make_dataset,
    profile_all,
    profile_likelihood,
    render_profiles,
)

BASE_SEED = 8800
N_POINTS = 5
REFINE = 4


def _problem(d, seed, generator):
    model = calibration_model()
    data = make_dataset(d, seed=seed, model=generator)
    return CalibrationProblem(
        model=model,
        inputs=data.inputs,
        initial_state=d.initial_state,
        channels=d.channels,
        sample_stride=d.sample_stride,
        y=data.y,
        sigma=data.sigma,
    )


def main() -> None:
    started = time.perf_counter()
    out = Tee(ROOT / "validation" / "validate_profile_output.txt")
    out("validate_profile.py -- profile-likelihood intervals")
    out(f"Wilks threshold at 0.95 with one degree of freedom = {chi2_threshold(0.95):.10f}")
    out(f"grid: {N_POINTS} points each side plus {REFINE} bisections per crossing")

    out.rule("1. Profile against the quadratic interval on a sound design")
    d = design("baseline-eclipse")
    problem = _problem(d, 8001, calibration_model())
    result = fit(problem, np.asarray(d.theta_true, dtype=float) * 1.2, max_nfev=2000)
    curves = profile_all(problem, result, n_points=N_POINTS, refine_steps=REFINE)
    out(render_profiles(curves))
    out("")
    ratios = [c.width_ratio for c in curves if np.isfinite(c.width_ratio)]
    out(f"width ratios: {', '.join(f'{r:.4f}' for r in ratios)}")
    out(f"worst deviation from 1 = {max(abs(r - 1.0) for r in ratios):.4f}")
    out(
        "verdict: "
        + ("PASS" if all(0.85 < r < 1.2 for r in ratios) else "FAIL")
        + "   criterion every ratio in (0.85, 1.20)"
    )
    out("")
    out("truth inside the interval: " + ", ".join(
        f"{c.name}={c.contains(d.theta_true[c.index])}" for c in curves
    ))

    out.rule("2. The deliberately non-identifiable design, reported as such")
    d = design("full-sun")
    problem = _problem(d, 8301, calibration_model())
    result = fit(problem, np.asarray(d.theta_true, dtype=float), max_nfev=2000)
    out("Quadratic standard deviations the fit reports, after the infinities have")
    out("been put back by twincalib.identifiability.null_direction_loading:")
    for name, sd in zip(result.parameter_names, result.relative_sd, strict=True):
        out(f"  {name:<8}{sd:>14.6g}")
    raw = np.sqrt(np.diag(np.linalg.pinv(result.sensitivity.fisher(), rcond=1e-10)))
    out("")
    out("What the RAW pseudo-inverse diagonal would have printed instead, which is")
    out("the trap this package exists to close:")
    for name, sd in zip(result.parameter_names, raw, strict=True):
        out(f"  {name:<8}{float(sd):>14.6g}")
    out("")
    curves = []
    for index in range(6):
        points = 2 if index >= 4 else N_POINTS
        refine = 8 if index >= 4 else REFINE
        curves.append(
            profile_likelihood(
                problem, result, index, n_points=points, refine_steps=refine
            )
        )
    out(render_profiles(curves))
    out("")
    total_load = float(result.theta[4] + result.theta[5])
    out(f"identified total heat load q_sun + q_par = {total_load:.6f} W")
    out(f"true total                               = {d.theta_true[4] + d.theta_true[5]:.6f} W")
    out(
        f"relative error on the total              = "
        f"{total_load / (d.theta_true[4] + d.theta_true[5]) - 1.0:+.6f}"
    )
    out("")
    out("So the design determines the SUM to better than one per cent and neither")
    out("part at all. Each part is open below -- it can go to zero with the other")
    out("taking up the slack -- and bounded above only by the positivity of its")
    out("partner, at or a little below the identified total:")
    for c in curves[4:]:
        out(
            f"  {c.name:<7} upper limit {c.upper:.4f} W = "
            f"{c.upper / total_load:.4f} of the identified total"
        )
    out("The partner is confined to the nuisance box of plus or minus two natural")
    out("log units, which is why q_par's limit is the lower of the two: q_sun")
    out("cannot be pushed below exp(-2) of its estimate to make room.")
    bad = [c for c in curves if not c.bounded]
    out(f"open-sided parameters: {[c.name for c in bad]}")
    out(
        "verdict: "
        + ("PASS" if {c.name for c in bad} == {"q_sun", "q_par"} else "FAIL")
        + "   criterion exactly q_sun and q_par are open"
    )
    out("")
    out("Total identifiability over 8 seeds, to show this is not one draw:")
    out(f"  {'seed':>6}{'q_sun + q_par [W]':>20}{'relative error':>17}")
    errors = []
    for seed in range(8401, 8409):
        pr = _problem(d, seed, calibration_model())
        res = fit(pr, np.asarray(d.theta_true, dtype=float), max_nfev=2000)
        value = float(res.theta[4] + res.theta[5])
        rel = value / (d.theta_true[4] + d.theta_true[5]) - 1.0
        errors.append(rel)
        out(f"  {seed:>6d}{value:>20.6f}{rel:>+17.6f}")
    out("")
    out(f"worst |relative error| on the total = {max(abs(e) for e in errors):.6f}")
    out(
        "verdict: "
        + ("PASS" if max(abs(e) for e in errors) < 0.05 else "FAIL")
        + "   criterion below 0.05"
    )

    out.rule("3. The practically non-identifiable design")
    d = design("weak-excitation")
    problem = _problem(d, 8501, calibration_model())
    result = fit(problem, np.asarray(d.theta_true, dtype=float), max_nfev=400)
    out(f"fit converged: {result.success}  ({result.message})")
    out(f"residual evaluations: {result.n_function_evaluations}")
    out(f"reduced chi-squared : {result.reduced_chi2:.6f}")
    out("")
    out("The rank test passes on this design (validate_identifiability.py check 4)")
    out("and the parameters are still worthless:")
    out(f"  {'parameter':<10}{'estimate':>14}{'truth':>14}{'rel. error':>13}{'quad rel sd':>14}")
    for j, name in enumerate(result.parameter_names):
        rel = result.theta[j] / d.theta_true[j] - 1.0
        out(
            f"  {name:<10}{result.theta[j]:>14.6g}{d.theta_true[j]:>14.6g}"
            f"{rel:>+13.4f}{result.relative_sd[j]:>14.4g}"
        )
    out("")
    curves = [
        profile_likelihood(problem, result, index, n_points=3, refine_steps=3)
        for index in range(6)
    ]
    out(render_profiles(curves))
    out("")
    out("Caveats this fit carries:")
    for c in result.caveats():
        out(f"  - {c}")
    out("")
    out("Reported, not bounded. The nuisance box of plus or minus two natural log")
    out("units is active on this design, so these profiles are box-constrained")
    out("profiles and the intervals are narrower than true profiles would be. The")
    out("verdict a user needs is the one the fit already prints: do not report")
    out("these parameters.")

    out("")
    out(f"elapsed {time.perf_counter() - started:.1f} s")
    out.save()


if __name__ == "__main__":
    main()

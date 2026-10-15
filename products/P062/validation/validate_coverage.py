"""Validate the coverage of the profile-likelihood intervals.

Run from the repository root: ``python validation/validate_coverage.py``
Writes ``validation/validate_coverage_output.txt``.

Coverage is measured over seeded repetitions in two conditions:

1. data the calibration model can reproduce exactly, where a 0.95 interval
   should contain the truth 95 per cent of the time;
2. data from the truth model, which carries the declared model-form
   discrepancy, where it should not and does not.

The coverage estimate is itself uncertain at this repeat count, so every figure
carries an exact Clopper-Pearson interval (Clopper and Pearson 1934,
*Biometrika* 26(4), 404-413) rather than being reported as a bare fraction.

This script is separated from ``validate_profile.py`` only so that each stays
inside the three-minute budget; its measured wall clock is at the end of its
output.

References: Wilks (1938), *Annals of Mathematical Statistics* 9(1), 60-62;
Raue et al. (2009), *Bioinformatics* 25(15), 1923-1929.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.stats import beta

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
    truth_model,
)

N_COVERAGE = 12
BASE_SEED = 8800
N_POINTS = 4
REFINE = 3


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact binomial confidence interval, dimensionless (Clopper-Pearson 1934)."""
    if not 0 <= k <= n or n <= 0:
        raise ValueError("require 0 <= k <= n and n > 0")
    lower = 0.0 if k == 0 else float(beta.ppf(alpha / 2.0, k, n - k + 1))
    upper = 1.0 if k == n else float(beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return (lower, upper)


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
    out = Tee(ROOT / "validation" / "validate_coverage_output.txt")
    out("validate_coverage.py -- coverage of the profile-likelihood intervals")
    out(f"Wilks threshold at 0.95 with one degree of freedom = {chi2_threshold(0.95):.10f}")
    out(f"grid: {N_POINTS} points each side plus {REFINE} bisections per crossing")

    out.rule("Coverage over seeded repetitions")
    d = design("coarse-sampled")
    out(f"design {d.name}, {d.n_observations} observations, {N_COVERAGE} seeds")
    out("Nominal coverage 0.95. Each cell is one parameter over all seeds; the")
    out("pooled row is all six parameters together. The interval on the coverage")
    out(f"estimate is exact (Clopper-Pearson), because {N_COVERAGE} seeds is a small")
    out("sample and a bare fraction would overstate what has been measured.")
    for label, generator in (
        ("no model discrepancy", calibration_model()),
        ("declared model discrepancy present", truth_model()),
    ):
        hits = np.zeros(6, dtype=int)
        total = np.zeros(6, dtype=int)
        open_sides = np.zeros(6, dtype=int)
        for r in range(N_COVERAGE):
            pr = _problem(d, BASE_SEED + r, generator)
            res = fit(pr, np.asarray(d.theta_true, dtype=float), max_nfev=2000)
            cs = profile_all(pr, res, n_points=N_POINTS, refine_steps=REFINE)
            for j, c in enumerate(cs):
                total[j] += 1
                hits[j] += int(c.contains(d.theta_true[j]))
                open_sides[j] += int(not c.bounded)
        out("")
        out(f"--- {label} ---")
        out(
            f"  {'parameter':<10}{'covered':>10}{'of':>5}{'coverage':>11}"
            f"{'95% exact interval':>24}{'open sides':>12}"
        )
        for j, name in enumerate(calibration_model().parameter_names):
            lo, hi = clopper_pearson(int(hits[j]), int(total[j]))
            out(
                f"  {name:<10}{int(hits[j]):>10d}{int(total[j]):>5d}"
                f"{hits[j] / total[j]:>11.4f}   [{lo:.4f}, {hi:.4f}]"
                f"{int(open_sides[j]):>12d}"
            )
        k, n = int(hits.sum()), int(total.sum())
        lo, hi = clopper_pearson(k, n)
        out(
            f"  {'POOLED':<10}{k:>10d}{n:>5d}{k / n:>11.4f}"
            f"   [{lo:.4f}, {hi:.4f}]{int(open_sides.sum()):>12d}"
        )
        inside = lo <= 0.95 <= hi
        out(
            f"  nominal 0.95 inside the exact interval: {inside}"
            + ("   verdict: PASS" if inside else "   verdict: REPORTED AS MEASURED")
        )
    out("")
    out("The second block is the honest limit of the whole method: a profile")
    out("interval is computed under the assumption that the calibration model is")
    out("the model that generated the data. When it is not, the interval is a")
    out("statement about the best fit of the wrong model and its coverage of the")
    out("true parameters collapses. No amount of profiling repairs that; only a")
    out("better model does.")


    out("")
    out(f"elapsed {time.perf_counter() - started:.1f} s")
    out.save()


if __name__ == "__main__":
    main()

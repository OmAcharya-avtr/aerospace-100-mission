"""Validation 3: the documented failure regime -- saturation of scintillation.

Demonstrates and quantifies three distinct failures of the analytic scintillation
inversion at strong turbulence:

  (a) the weak-regime linear inversion becomes strongly biased low;
  (b) the exact inversion becomes MULTI-VALUED (two turbulence strengths give the
      same reading), and above the peak it has NO solution at all;
  (c) the propagated uncertainty diverges as the sensitivity goes to zero.

Run:  python validation/validate_saturation.py > validation/validate_saturation_output.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from turbscope import (  # noqa: E402
    PathGeometry,
    SensorSuite,
    invert_scintillation,
    saturation_peak,
    saturation_report,
    scintillation_branches,
    scintillation_index,
    simulate_measurement,
)
from turbscope.inversion import scintillation_index_relative_sigma  # noqa: E402
from turbscope.scintillation import (  # noqa: E402
    aperture_parameter_sq,
    rytov_variance_from_average,
    uniform_cn2_from_beta0_sq,
)

SEED = 424242


def rule(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def section_1_bias_of_the_weak_inversion() -> None:
    rule("1. Bias of the weak-regime linear inversion vs true beta_0^2")
    print("The textbook inversion assumes sigma_I^2 = beta_0^2.  Below is the Cn2 it")
    print("returns, relative to the truth, using the exact forward model to generate")
    print("the (noise-free) reading.\n")
    print(f"{'beta_0^2':>10} {'sigma_I^2':>11} {'Cn2_weak/Cn2_true':>20} {'error %':>10}")
    for beta in (0.01, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 7.2966, 10.0, 20.0, 50.0, 100.0):
        sigma = float(scintillation_index(beta, 0.0))
        ratio = sigma / beta
        print(f"{beta:>10.4g} {sigma:>11.5f} {ratio:>20.5f} {100 * (ratio - 1):>10.2f}")
    print("\nInside the weak regime the linear inversion is good to ~1 % (its largest")
    print("deviation there is +1.06 % near beta_0^2 = 0.1, and it is -0.10 % at the")
    print("beta_0^2 = 0.3 limit).  Beyond it the bias is one-sided and severe: -13.9 %")
    print("at beta_0^2 = 1, -66.7 % at 5, -92.1 % at 20 and -98.7 % at 100, i.e. nearly")
    print("two orders of magnitude of under-estimate.  This is why WEAK_REGIME_BETA0_SQ")
    print("is set to 0.3 and why the package warns above it.")


def section_2_multivalued() -> None:
    rule("2. The inversion is multi-valued: quantifying the ambiguity")
    rep = saturation_report(0.0)
    print(f"peak of sigma_I^2(beta_0^2):  beta_0^2 = {rep.beta0_sq_peak:.6f}, "
          f"sigma_I^2 = {rep.sigma_i2_peak:.6f}")
    print(f"large-beta asymptote:         sigma_I^2 -> {rep.sigma_i2_asymptote:.6f}")
    lo, hi = rep.ambiguous_sigma_i2_range
    print(f"multi-valued reading range:   {lo:.6f} < sigma_I^2 < {hi:.6f}")
    print(f"unexplainable readings:       sigma_I^2 > {rep.sigma_i2_peak:.6f}\n")
    print("For a 2 km path at 850 nm, the two solutions and the Cn2 ambiguity they imply:")
    path = PathGeometry(2000.0, 850e-9)
    print(f"{'sigma_I^2':>10} {'beta_0^2 (low)':>15} {'beta_0^2 (high)':>16} "
          f"{'Cn2 low':>13} {'Cn2 high':>13} {'ratio':>8}")
    ambiguous = 0
    for sigma in (1.06, 1.10, 1.20, 1.30, 1.40, 1.50, 1.60, 1.65, 1.69):
        branches = scintillation_branches(sigma, 0.0)
        if len(branches) < 2:
            print(f"{sigma:>10.3f} {'single branch':>15}")
            continue
        ambiguous += 1
        lo_b, hi_b = branches[0], branches[-1]
        c_lo = uniform_cn2_from_beta0_sq(lo_b, path)
        c_hi = uniform_cn2_from_beta0_sq(hi_b, path)
        print(f"{sigma:>10.3f} {lo_b:>15.4f} {hi_b:>16.4f} {c_lo:>13.4e} {c_hi:>13.4e} "
              f"{c_hi / c_lo:>8.1f}x")
    print(f"\n{ambiguous} of 9 test readings admit two solutions.  At sigma_I^2 = 1.06 the")
    print("two branches differ by a factor of 5153 in Cn2; at sigma_I^2 = 1.50 by a factor")
    print("of 10.9; and even at sigma_I^2 = 1.69, one part in a thousand below the peak,")
    print("by a factor of 1.2.  No estimator, and no amount of averaging, resolves this")
    print("from a single point-receiver scintillometer reading -- the information is not")
    print("in the measurement.  (The upper branch above beta_0^2 ~ 1e3 is a root of the")
    print("Andrews-Phillips fit far outside the range over which it was validated; it is")
    print("reported because it is a genuine solution of the model the package uses, not")
    print("because such turbulence is expected on a real path.)")


def section_3_sensitivity_collapse() -> None:
    rule("3. Sensitivity collapse: the propagated uncertainty diverges at the peak")
    path = PathGeometry(1000.0, 1550e-9)
    b_peak = saturation_peak(0.0)[0]
    print("Relative standard deviation of the recovered Cn2 from N = 1000 irradiance")
    print("samples, as the true turbulence approaches the peak.\n")
    print(f"{'beta_0^2':>10} {'beta/beta_pk':>13} {'d sigma/d beta':>16} "
          f"{'rel sigma (Cn2)':>17} {'beta recovered':>15} {'recovery err':>13}")
    for frac in (0.01, 0.05, 0.1, 0.3, 0.5, 0.8, 0.95, 0.99, 0.999):
        beta = b_peak * frac
        eps = 1e-5 * beta
        deriv = float(
            (scintillation_index(beta + eps, 0.0) - scintillation_index(beta - eps, 0.0))
            / (2 * eps)
        )
        sigma = float(scintillation_index(beta, 0.0))
        rel_s = scintillation_index_relative_sigma(beta, 0.0, 1000)
        rel_cn2 = rel_s * sigma / (deriv * beta)
        branches = scintillation_branches(sigma, 0.0)
        beta_hat = branches[0] if branches else float("nan")
        print(f"{beta:>10.4f} {frac:>13.3f} {deriv:>16.3e} {rel_cn2:>17.4f} "
              f"{beta_hat:>15.4f} {beta_hat / beta - 1:>13.2e}")
    print("\nThe sensitivity d(sigma_I^2)/d(beta_0^2) falls by five orders of magnitude")
    print("between 1 % and 99.9 % of the peak, so the delta-method uncertainty on Cn2")
    print("grows without bound: 4.8 % at beta_0^2 = 0.073 becomes 222 % at 5.84 and")
    print("55880 % at 7.29.  Away from the peak the root-finder is exact (recovery error")
    print("<= 2e-13 in every row up to 99 % of the peak), but within 0.1 % of the peak it")
    print("fails completely: sigma_I^2 is flat there to within the scan resolution, no")
    print("bracketing sign change exists, and the inversion returns no branch at all (NaN")
    print("in the last row).  The package reports the widening interval, and then the")
    print("absence of a solution, rather than returning a confident number.")


def section_4_end_to_end_failure_rate() -> None:
    rule("4. End-to-end failure rate on simulated strong-turbulence observations")
    rng = np.random.default_rng(SEED)
    path = PathGeometry(2000.0, 850e-9)
    suite = SensorSuite(
        receiver_diameter_m=0.25, n_irradiance_samples=2000, n_dimm_frames=1000
    )
    d_sq = aperture_parameter_sq(suite.receiver_diameter_m, path)
    print(f"path 2 km at 850 nm, receiver D = 0.25 m -> aperture d^2 = {d_sq:.2f}")
    print(f"point-channel peak sigma_I^2 = {saturation_peak(0.0)[1]:.4f}; "
          f"aperture-channel peak = {saturation_peak(d_sq)[1]:.4f} at "
          f"beta_0^2 = {saturation_peak(d_sq)[0]:.1f}\n")
    print(f"{'Cn2 true':>11} {'beta_0^2':>10} {'invalid %':>10} {'ambiguous %':>12} "
          f"{'median Cn2_hat/Cn2':>20} {'p95 ratio':>11}")
    n_trials = 200
    for level in (1e-15, 3e-15, 1e-14, 3e-14, 1e-13):
        z = path.uniform_grid(201)
        cn2 = np.full_like(z, level)
        beta_true = rytov_variance_from_average(level, path)
        ratios, invalid, ambiguous = [], 0, 0
        for _ in range(n_trials):
            meas = simulate_measurement(z, cn2, path, suite, rng)
            est = invert_scintillation(
                meas.sigma_i2_point, path, n_samples=suite.n_irradiance_samples
            )
            if not est.valid:
                invalid += 1
                continue
            if est.ambiguous:
                ambiguous += 1
            ratios.append(est.cn2 / level)
        med = np.median(ratios) if ratios else float("nan")
        p95 = np.percentile(ratios, 95) if ratios else float("nan")
        print(f"{level:>11.1e} {beta_true:>10.2f} {100 * invalid / n_trials:>10.1f} "
              f"{100 * ambiguous / n_trials:>12.1f} {med:>20.4f} {p95:>11.4f}")
    print("\nRESULT: the analytic point-receiver inversion degrades sharply above")
    print("beta_0^2 ~ 1.7 and is unusable at beta_0^2 ~ 5.8, where 39.0 % of readings")
    print("exceed the attainable maximum and are refused outright, 61.0 % are flagged")
    print("ambiguous, and the surviving estimates have a median Cn2_hat/Cn2 of 0.658.")
    print("At beta_0^2 = 1.73 essentially every reading (99.5 %) is already formally")
    print("ambiguous even though the low branch is still the right one (median ratio")
    print("0.976) -- the ambiguity flag fires well before the point estimate fails.")
    print("Note that aperture averaging only postpones the problem: the aperture channel")
    print("here peaks at sigma_I^2 = 0.1925 at beta_0^2 = 26.2 and is multi-valued above")
    print("that.  TurbScope documents all of this as a limit of the measurement, not a")
    print("defect of the software.  validation/benchmark_ml.py reports how much of the")
    print("gap the learned multi-sensor model closes, and how much it does not.")


def main() -> int:
    print("TurbScope validation 3 - saturation of scintillation (documented failure regime)")
    section_1_bias_of_the_weak_inversion()
    section_2_multivalued()
    section_3_sensitivity_collapse()
    section_4_end_to_end_failure_rate()
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

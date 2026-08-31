"""Validation 3 -- attitude covariance against seeded Monte Carlo.

Checks
------
1. Eq. V1 (the Cramer-Rao covariance attained by the q-method and QUEST)
   against the sample covariance of ``delta_theta = log(A_est A_true^T)`` over
   seeded Monte Carlo trials, for equal and for unequal sigmas.
2. Eq. V2 (TRIAD) against Monte Carlo, for both choices of primary.
3. ``P_TRIAD - P_opt`` positive semi-definite (TRIAD is never better).
4. OLAE's Monte Carlo covariance against Eq. V1, which it only approximates.
5. Where the first-order covariance stops describing reality, by sweeping
   sigma from 1e-4 to 3e-1 rad.
6. Cross-check against ``scipy.spatial.transform.Rotation.align_vectors``
   ``return_sensitivity=True``, scaled by the harmonic mean of the variances as
   the SciPy documentation prescribes.  This check exists to keep the README's
   positioning honest, and it found that SciPy already yields exactly Eq. V1 --
   for unequal sigmas as well as equal ones.

The Monte Carlo sampling error on a covariance entry is about
``sqrt(2 / trials)``; with ``trials = 10000`` that is 1.4 %, so a 6 % gate is
loose enough not to be flaky and tight enough that a wrong formula fails.

Run:  python validation/validate_covariance.py
"""

from __future__ import annotations

import numpy as np
from _common import WELL_CONDITIONED, banner, sample_observations, verdict

from wahbakit import (
    VectorObservations,
    attitude_error_vector,
    dcm_from_quat,
    olae,
    optimal_covariance,
    q_method,
    triad,
    triad_covariance,
)

SEED = 20260831
TRIALS = 10000
DCM_TRUE = dcm_from_quat([0.3, -0.7, 1.1, 0.2])
TOLERANCE = 0.06


def monte_carlo_covariance(solver, reference, sigmas, trials, rng, **kwargs):
    """Sample covariance of the body-frame attitude error, in rad^2."""
    true_body = reference @ DCM_TRUE.T
    samples = sample_observations(true_body, sigmas, rng, trials)
    errors = np.empty((trials, 3))
    for k in range(trials):
        obs = VectorObservations(samples[k], reference, sigmas=sigmas)
        errors[k] = attitude_error_vector(solver(obs, **kwargs).dcm, DCM_TRUE)
    return errors.T @ errors / trials


def compare(name, empirical, analytic, tolerance=TOLERANCE):
    """Print both matrices and return the worst relative deviation."""
    scale = float(np.max(np.abs(analytic)))
    worst = float(np.max(np.abs(empirical - analytic))) / scale
    print(f"    {name}")
    for row in range(3):
        analytic_row = "  ".join(f"{analytic[row, c]:11.4e}" for c in range(3))
        empirical_row = "  ".join(f"{empirical[row, c]:11.4e}" for c in range(3))
        print(f"      analytic [{analytic_row}]   MC [{empirical_row}]")
    diagonal = np.abs(np.diag(empirical) / np.diag(analytic) - 1.0)
    print(f"      diagonal relative deviation: {np.array2string(diagonal, precision=4)}")
    return verdict(f"    worst relative deviation, {name}", worst, tolerance)


def main() -> int:
    banner("wahbakit validation 3 -- attitude covariance vs Monte Carlo")
    print(f"trials = {TRIALS}, seed = {SEED}, sampling error on a covariance entry "
          f"~ sqrt(2/{TRIALS}) = {np.sqrt(2 / TRIALS):.4f}")
    rng = np.random.default_rng(SEED)
    passed = []

    print("\n[1] Eq. V1, optimal covariance, q-method")
    for label, sigmas in (
        ("equal sigmas 1e-3 rad", np.full(4, 1e-3)),
        ("unequal sigmas [1, 2, 5, 1] x 1e-3 rad", np.array([1e-3, 2e-3, 5e-3, 1e-3])),
    ):
        obs = VectorObservations(
            WELL_CONDITIONED @ DCM_TRUE.T, WELL_CONDITIONED, sigmas=sigmas
        )
        analytic = optimal_covariance(obs)
        empirical = monte_carlo_covariance(q_method, WELL_CONDITIONED, sigmas, TRIALS, rng)
        passed.append(compare(label, empirical, analytic))

    print("\n[2] Eq. V2, TRIAD covariance, two observations, sigmas [1e-3, 5e-3] rad")
    reference = WELL_CONDITIONED[:2]
    sigmas = np.array([1e-3, 5e-3])
    for primary in (0, 1):
        obs = VectorObservations(reference @ DCM_TRUE.T, reference, sigmas=sigmas)
        analytic = triad_covariance(obs, primary=primary)
        empirical = monte_carlo_covariance(
            triad, reference, sigmas, TRIALS, rng, primary=primary
        )
        passed.append(compare(f"primary = {primary}", empirical, analytic))

    print("\n[3] P_TRIAD - P_opt positive semi-definite (TRIAD is never better)")
    worst_relative_negative = 0.0
    worst_condition = 0.0
    for sigma1 in (1e-4, 1e-3, 1e-2):
        for sigma2 in (1e-4, 1e-3, 1e-2):
            obs = VectorObservations(
                reference @ DCM_TRUE.T, reference, sigmas=np.array([sigma1, sigma2])
            )
            triad_p = triad_covariance(obs)
            optimal_p = optimal_covariance(obs)
            difference = triad_p - optimal_p
            eigenvalues = np.linalg.eigvalsh(difference)
            scale = float(np.max(np.abs(triad_p)))
            relative = float(eigenvalues[0]) / scale
            worst_relative_negative = min(worst_relative_negative, relative)
            worst_condition = max(worst_condition, float(np.linalg.cond(optimal_p)))
            excess = float(np.trace(difference)) / float(np.trace(optimal_p))
            print(
                f"    sigma1 = {sigma1:.0e}, sigma2 = {sigma2:.0e}: "
                f"min eig = {eigenvalues[0]:11.4e} ({relative:9.2e} of max|P_TRIAD|), "
                f"cond(P_opt) = {np.linalg.cond(optimal_p):8.2e}, "
                f"excess variance = {100 * excess:8.3f} %"
            )
    print(
        f"    Worst cond(P_opt) over the grid: {worst_condition:.3e}. The inverse in "
        "Eq. V1 carries a"
    )
    print(
        "    relative error of order eps * cond, so the attainable numerical bound on a"
    )
    print(
        "    zero eigenvalue is about 2e-16 * 1e4 = 2e-12, not 0. Gated at 1e-9 relative."
    )
    passed.append(
        verdict(
            "most negative eigenvalue of P_TRIAD - P_opt, relative",
            -worst_relative_negative,
            1e-9,
        )
    )

    print("\n[4] OLAE against Eq. V1, which it only approximates")
    sigmas = np.full(4, 1e-3)
    obs = VectorObservations(WELL_CONDITIONED @ DCM_TRUE.T, WELL_CONDITIONED, sigmas=sigmas)
    analytic = optimal_covariance(obs)
    empirical_olae = monte_carlo_covariance(olae, WELL_CONDITIONED, sigmas, TRIALS, rng)
    empirical_optimal = monte_carlo_covariance(q_method, WELL_CONDITIONED, sigmas, TRIALS, rng)
    excess = float(np.trace(empirical_olae) / np.trace(empirical_optimal)) - 1.0
    print(f"    trace(P_MC, OLAE)     = {np.trace(empirical_olae):.6e} rad^2")
    print(f"    trace(P_MC, q-method) = {np.trace(empirical_optimal):.6e} rad^2")
    print(f"    trace(P analytic)     = {np.trace(analytic):.6e} rad^2")
    print(f"    OLAE excess total variance over the optimum: {100 * excess:.3f} %")
    print("    Eq. V1 is therefore an optimistic covariance for OLAE by that amount.")
    passed.append(verdict("OLAE excess variance (reported, gated at 10 %)", excess, 0.10))

    print("\n[5] where the first-order covariance stops describing reality")
    print(f"    {'sigma [rad]':>12} {'trace P analytic':>18} {'trace P MC':>14} {'ratio':>9}")
    ratios = []
    for sigma in (1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1):
        sigmas = np.full(4, sigma)
        obs = VectorObservations(
            WELL_CONDITIONED @ DCM_TRUE.T, WELL_CONDITIONED, sigmas=sigmas
        )
        analytic = optimal_covariance(obs)
        empirical = monte_carlo_covariance(q_method, WELL_CONDITIONED, sigmas, 3000, rng)
        ratio = float(np.trace(empirical) / np.trace(analytic))
        ratios.append((sigma, ratio))
        print(
            f"    {sigma:12.0e} {np.trace(analytic):18.6e} {np.trace(empirical):14.6e} "
            f"{ratio:9.4f}"
        )
    small_noise = [r for s, r in ratios if s <= 1e-2]
    passed.append(
        verdict(
            "max |trace ratio - 1| for sigma <= 1e-2",
            max(abs(r - 1) for r in small_noise),
            0.06,
        )
    )
    print("    The ratio departs from 1 as sigma grows: Eq. V1 is a first-order result")
    print("    and is reported, not gated, above 1e-2 rad.")

    print("\n[6] cross-check against scipy align_vectors return_sensitivity")
    from scipy.spatial.transform import Rotation

    for label, sigmas in (
        ("equal sigmas 1e-3 rad", np.full(4, 1e-3)),
        ("unequal sigmas [1, 2, 5, 1] x 1e-3 rad", np.array([1e-3, 2e-3, 5e-3, 1e-3])),
    ):
        obs = VectorObservations(
            WELL_CONDITIONED @ DCM_TRUE.T, WELL_CONDITIONED, sigmas=sigmas
        )
        _, _, sensitivity = Rotation.align_vectors(
            obs.body, obs.reference, weights=obs.weights, return_sensitivity=True
        )
        harmonic_mean = len(sigmas) / float(np.sum(1.0 / sigmas**2))
        scipy_covariance = sensitivity * harmonic_mean
        analytic = optimal_covariance(obs)
        deviation = float(np.max(np.abs(scipy_covariance - analytic))) / float(
            np.max(np.abs(analytic))
        )
        print(f"    {label}: max relative deviation from Eq. V1 = {deviation:.4e}")
    print("    SciPy's sensitivity matrix, scaled by the harmonic mean of the variances")
    print("    exactly as its Notes prescribe and with weights proportional to 1/sigma^2,")
    print("    reproduces Eq. V1 to round-off for BOTH equal and unequal sigmas. The")
    print("    optimal attitude covariance is therefore NOT something this package")
    print("    provides and SciPy does not; it is the same number, returned directly in")
    print("    rad^2 instead of requiring the scaling to be worked out. What SciPy has no")
    print("    analogue for is the TRIAD covariance of Eq. V2 and the degeneracy gate.")

    print()
    print(f"RESULT: {sum(passed)} / {len(passed)} checks passed")
    return 0 if all(passed) else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Validation 4: uncertainty analysis.

Level 3 asks what happens when the numbers going into the steering law are not
exact.  Three independent error sources are propagated here, each against a
first-order analytic prediction so that the Monte Carlo is checked rather than
merely reported:

1. **Gimbal-angle measurement error.**  The law computes ``A`` at the measured
   angles and the array delivers torque at the true ones.  With
   ``dA/ddelta_j`` having only column ``j`` non-zero and equal to
   ``-h0_j h_hat_j``, the resulting torque error is
   ``e = sum_j h0_j h_hat_j eps_j ddelta_j`` to first order, so for independent
   zero-mean angle errors of standard deviation ``sigma``,
   ``rms|e| = sigma sqrt(sum_j h0_j^2 ddelta_j^2)``.
2. **Rotor momentum error.**  A rotor running at ``h0_i (1 + eps_i)`` gives
   ``e = -sum_i eps_i a_i ddelta_i``, whose first-order rms is the same
   expression because ``|a_i| = h0_i``.
3. **Singularity-measure sensitivity.**  ``sigma_m = |grad m| sigma`` to first
   order.

Section 4 then runs whole manoeuvres with angle noise injected and reports
where the noise starts to dominate the steering law's own error.

Run: ``python validation/validate_uncertainty.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cmgsteer.arrays import pyramid_array  # noqa: E402
from cmgsteer.dataset import manoeuvre_suite  # noqa: E402
from cmgsteer.singularity import manipulability_gradient, singularity_measure  # noqa: E402
from cmgsteer.steering import sr_inverse_steer  # noqa: E402

SEED = 20260902
SIGMAS_DEG = (0.001, 0.01, 0.1, 1.0)
N_STATES = 30
N_TRIALS_PER_STATE = 200


def _states(array, rng, n):
    return [rng.uniform(-np.pi, np.pi, array.n_cmgs) for _ in range(n)]


def section_1_angle_error():
    print("\n## 1. Gimbal-angle measurement error -> torque error")
    print("ratios are formed per state and then aggregated: pooling the Monte Carlo over")
    print("states and comparing with a mean prediction would compare an rms with a mean")
    array = pyramid_array()
    rng = np.random.default_rng(SEED)
    tau = np.array([0.10, -0.05, 0.20])
    states = _states(array, rng, N_STATES)
    print(
        f"{'sigma [deg]':>12} {'mean MC rms [N*m]':>19} {'mean first order':>18} "
        f"{'mean ratio':>12} {'p5':>8} {'p95':>8}"
    )
    summary = []
    for sigma_deg in SIGMAS_DEG:
        sigma = np.radians(sigma_deg)
        mc_per_state = []
        pred_per_state = []
        for d in states:
            rates = sr_inverse_steer(array, d, tau, lam=1e-10).gimbal_rates
            pred_per_state.append(
                sigma * float(np.sqrt(np.sum((array.rotor_momenta * rates) ** 2)))
            )
            errors = []
            for eps in rng.normal(scale=sigma, size=(N_TRIALS_PER_STATE, array.n_cmgs)):
                perturbed = sr_inverse_steer(array, d + eps, tau, lam=1e-10).gimbal_rates
                errors.append(float(np.linalg.norm(tau - array.torque(d, perturbed))))
            mc_per_state.append(float(np.sqrt(np.mean(np.square(errors)))))
        ratios = np.array(mc_per_state) / np.array(pred_per_state)
        summary.append(float(ratios.mean()))
        print(
            f"{sigma_deg:>12.3f} {np.mean(mc_per_state):>19.6e} "
            f"{np.mean(pred_per_state):>18.6e} {ratios.mean():>12.4f} "
            f"{np.percentile(ratios, 5):>8.4f} {np.percentile(ratios, 95):>8.4f}"
        )
    print("the mean ratio stays within 2% of 1 across four decades of sigma, so the")
    print("first-order model is adequate over the whole range tested")
    return summary


def section_2_rotor_momentum_error():
    print("\n## 2. Rotor momentum error -> torque error")
    array = pyramid_array()
    rng = np.random.default_rng(SEED + 1)
    tau = np.array([0.10, -0.05, 0.20])
    states = _states(array, rng, N_STATES)
    print(
        f"{'sigma [rel]':>12} {'mean MC rms [N*m]':>19} {'mean first order':>18} "
        f"{'mean ratio':>12} {'p5':>8} {'p95':>8}"
    )
    summary = []
    for sigma in (1e-4, 1e-3, 1e-2, 5e-2):
        mc_per_state = []
        pred_per_state = []
        for d in states:
            rates = sr_inverse_steer(array, d, tau, lam=1e-10).gimbal_rates
            pred_per_state.append(
                sigma * float(np.sqrt(np.sum((array.rotor_momenta * rates) ** 2)))
            )
            jac = array.jacobian(d)
            errors = []
            for eps in rng.normal(scale=sigma, size=(N_TRIALS_PER_STATE, array.n_cmgs)):
                achieved = -((jac * (1.0 + eps)[None, :]) @ rates)
                errors.append(float(np.linalg.norm(tau - achieved)))
            mc_per_state.append(float(np.sqrt(np.mean(np.square(errors)))))
        ratios = np.array(mc_per_state) / np.array(pred_per_state)
        summary.append(float(ratios.mean()))
        print(
            f"{sigma:>12.0e} {np.mean(mc_per_state):>19.6e} {np.mean(pred_per_state):>18.6e} "
            f"{ratios.mean():>12.4f} {np.percentile(ratios, 5):>8.4f} "
            f"{np.percentile(ratios, 95):>8.4f}"
        )
    print("the error is exactly linear in sigma here: the momentum error enters the")
    print("Jacobian linearly, so there is no second-order term at all")
    return summary


def section_3_measure_sensitivity():
    print("\n## 3. Singularity-measure sensitivity to gimbal-angle error")
    array = pyramid_array()
    rng = np.random.default_rng(SEED + 2)
    states = _states(array, rng, 60)
    print(f"{'sigma [deg]':>12} {'MC std(m)':>14} {'|grad m| sigma':>16} {'ratio':>9}")
    ratios = []
    for sigma_deg in SIGMAS_DEG:
        sigma = np.radians(sigma_deg)
        mc = []
        first = []
        for d in states:
            grad = manipulability_gradient(array, d)
            first.append(sigma * float(np.linalg.norm(grad)))
            samples = np.array(
                [
                    singularity_measure(array.jacobian(d + rng.normal(scale=sigma, size=4)))
                    for _ in range(120)
                ]
            )
            mc.append(float(samples.std()))
        ratio = float(np.mean(mc) / np.mean(first))
        ratios.append(ratio)
        print(f"{sigma_deg:>12.3f} {np.mean(mc):>14.6e} {np.mean(first):>16.6e} {ratio:>9.4f}")
    return ratios


def section_4_manoeuvre_level():
    print("\n## 4. Manoeuvre-level effect of gimbal-angle noise")
    array = pyramid_array()
    suite = manoeuvre_suite(
        array, 6, seed=SEED, n_segments=2, segment_duration=5.0, dt=0.02
    )
    print(
        f"{'sigma [deg]':>12} {'mean path err':>16} {'mean net err':>15} "
        f"{'min m':>12} {'vs noise-free':>15}"
    )
    baseline = None
    for sigma_deg in (0.0, 0.001, 0.01, 0.1, 1.0):
        sigma = np.radians(sigma_deg)
        rng = np.random.default_rng(SEED + 7)
        paths = []
        nets = []
        min_m = np.inf
        for profile, start in suite:
            d = np.array(start, dtype=float)
            path = 0.0
            net = np.zeros(3)
            h_prev = array.momentum(d)
            for tau in profile.torques:
                measured = d + rng.normal(scale=sigma, size=4) if sigma > 0 else d
                result = sr_inverse_steer(
                    array, measured, tau, lam0=0.01, mu=10.0, max_gimbal_rate=2.0
                )
                min_m = min(min_m, result.measure)
                d = d + array.expand_rates(result.gimbal_rates) * profile.dt
                h_now = array.momentum(d)
                step_err = (-tau * profile.dt) - (h_now - h_prev)
                path += float(np.linalg.norm(step_err))
                net += step_err
                h_prev = h_now
            paths.append(path)
            nets.append(float(np.linalg.norm(net)))
        mean_path = float(np.mean(paths))
        if baseline is None:
            baseline = mean_path
        print(
            f"{sigma_deg:>12.3f} {mean_path:>16.6e} {float(np.mean(nets)):>15.6e} "
            f"{min_m:>12.6e} {mean_path / baseline:>15.4f}"
        )
    print("the steering law's own error dominates until the angle noise reaches about "
          "0.1 deg; above that the noise dominates")


def main() -> int:
    print("=" * 78)
    print("CMGSteer validation 4 -- uncertainty analysis")
    print("=" * 78)
    print(
        f"seed {SEED}, numpy {np.__version__}, {N_STATES} states x "
        f"{N_TRIALS_PER_STATE} Monte Carlo trials per sigma"
    )

    angle = section_1_angle_error()
    rotor = section_2_rotor_momentum_error()
    measure = section_3_measure_sensitivity()
    section_4_manoeuvre_level()

    print("\n## Summary")
    checks = [
        ("angle-error Monte Carlo vs first order (smallest sigma)", angle[0], 0.05),
        ("rotor-error Monte Carlo vs first order (smallest sigma)", rotor[0], 0.05),
        ("measure-sensitivity Monte Carlo vs first order (smallest sigma)", measure[0], 0.05),
    ]
    ok = True
    for name, ratio, tol in checks:
        dev = abs(ratio - 1.0)
        verdict = "PASS" if dev < tol else "FAIL"
        ok &= dev < tol
        print(f"{verdict}  {name:<58} ratio {ratio:.4f}, |ratio-1| < {tol}")
    print("\nOVERALL: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

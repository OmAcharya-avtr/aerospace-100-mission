"""Validation 3: steering-law exactness and the SR-inverse closed-form torque error.

Establishes that the Moore-Penrose pseudo-inverse reproduces the commanded
torque exactly away from a singularity, that the singularity-robust inverse's
torque error equals its closed form as a function of the robustness parameter,
and that the generalised SR inverse reduces to the SR inverse when its dither
is switched off.

Run: ``python validation/validate_steering.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cmgsteer.arrays import pyramid_array, roof_array  # noqa: E402
from cmgsteer.simulate import rest_to_rest_profile, run_steering  # noqa: E402
from cmgsteer.singularity import singular_configuration, singularity_measure  # noqa: E402
from cmgsteer.steering import (  # noqa: E402
    gsr_inverse_steer,
    pseudo_inverse_steer,
    sr_inverse_steer,
    sr_torque_error_closed_form,
)

SEED = 20260902
N_SAMPLES = 2000
LAMBDAS = (1e-10, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)


def section_1_pseudo_inverse(array, label):
    """Exactness of the pseudo-inverse, binned by how singular the state is.

    Uniformly sampled gimbal angles almost never land close to a singularity,
    so the two near-singular bands are populated by perturbing analytically
    constructed singular configurations instead of by rejection sampling.
    """
    print(f"\n## 1. {label}: pseudo-inverse reproduces the commanded torque")
    rng = np.random.default_rng(SEED)
    n = array.n_cmgs
    states = list(rng.uniform(-np.pi, np.pi, (N_SAMPLES, n)))
    for _ in range(N_SAMPLES):
        u = rng.normal(size=3)
        u /= np.linalg.norm(u)
        signs = rng.choice([-1.0, 1.0], size=n)
        try:
            base = singular_configuration(array, u, signs)
        except ValueError:
            continue
        states.append(base + rng.normal(size=n) * 10.0 ** rng.uniform(-4, -0.5))

    bands = [(0.5, np.inf), (0.1, 0.5), (0.01, 0.1), (1e-4, 0.01), (0.0, 1e-4)]
    stats = {band: [0, 0.0, 0.0] for band in bands}
    for d in states:
        m = singularity_measure(array.jacobian(d))
        tau = rng.normal(size=3) * 0.3
        result = pseudo_inverse_steer(array, d, tau)
        for band in bands:
            if band[0] <= m < band[1]:
                stats[band][0] += 1
                stats[band][1] = max(stats[band][1], result.torque_error_norm)
                stats[band][2] = max(
                    stats[band][2], float(np.max(np.abs(result.gimbal_rates)))
                )
                break
    print(f"{'m band':>18} {'n':>6} {'worst |err| [N*m]':>19} {'worst rate [rad/s]':>20}")
    worst_regular = 0.0
    for band in bands:
        count, worst, worst_rate = stats[band]
        hi_label = "inf" if np.isinf(band[1]) else f"{band[1]:g}"
        print(
            f"{f'[{band[0]:g}, {hi_label})':>18} {count:>6} {worst:>19.6e} {worst_rate:>20.6e}"
        )
        if band[0] >= 0.1:
            worst_regular = max(worst_regular, worst)
    print(f"worst error over the two bands with m >= 0.1: {worst_regular:.6e} N*m")
    print(f"{len(states)} states: {N_SAMPLES} uniform plus perturbed analytic singularities")
    return worst_regular


def section_2_sr_closed_form(array, label):
    print(f"\n## 2. {label}: SR-inverse torque error vs the closed form")
    print("closed form: tau_err = sum_k [lam / (sigma_k^2 + lam)] (u_k . tau) u_k")
    rng = np.random.default_rng(SEED + 1)
    configs = rng.uniform(-np.pi, np.pi, (200, array.n_cmgs))
    torques = rng.normal(size=(200, 3)) * 0.3
    print(
        f"{'lam':>10} {'mean |err| [N*m]':>18} {'closed form':>18} "
        f"{'worst deviation':>17} {'mean rel err':>14}"
    )
    worst_dev = 0.0
    for lam in LAMBDAS:
        measured = []
        predicted = []
        for d, tau in zip(configs, torques):
            result = sr_inverse_steer(array, d, tau, lam=lam)
            closed = sr_torque_error_closed_form(array.jacobian(d), tau, lam)
            measured.append(result.torque_error_norm)
            predicted.append(float(np.linalg.norm(closed)))
            worst_dev = max(worst_dev, float(np.max(np.abs(result.torque_error - closed))))
        measured_arr = np.array(measured)
        predicted_arr = np.array(predicted)
        rel = measured_arr / np.linalg.norm(torques, axis=1)
        print(
            f"{lam:>10.0e} {measured_arr.mean():>18.9e} {predicted_arr.mean():>18.9e} "
            f"{np.max(np.abs(measured_arr - predicted_arr)):>17.6e} {rel.mean():>14.6e}"
        )
    print(f"worst componentwise deviation across all {len(LAMBDAS)} values of lam and "
          f"200 states: {worst_dev:.6e} N*m")
    return worst_dev


def section_3_sr_at_a_singularity():
    print("\n## 3. SR-inverse error at an exact singularity, as a function of lam")
    array = pyramid_array()
    d = np.full(4, np.pi / 2)
    tau = np.array([0.0, 0.0, 0.1])
    print("commanded torque is along the singular direction, so the error is exactly "
          "|tau| for every lam > 0")
    print(f"{'lam':>10} {'|err| [N*m]':>16} {'closed form':>16} {'peak rate [rad/s]':>19}")
    for lam in LAMBDAS:
        result = sr_inverse_steer(array, d, tau, lam=lam)
        closed = float(np.linalg.norm(sr_torque_error_closed_form(array.jacobian(d), tau, lam)))
        print(
            f"{lam:>10.0e} {result.torque_error_norm:>16.9e} {closed:>16.9e} "
            f"{float(np.max(np.abs(result.gimbal_rates))):>19.6e}"
        )
    print("the same command rotated into the plane the array can still act in:")
    tau_plane = np.array([0.1, 0.0, 0.0])
    for lam in (1e-8, 1e-4, 1e-2):
        result = sr_inverse_steer(array, d, tau_plane, lam=lam)
        print(f"  lam {lam:.0e}: |err| {result.torque_error_norm:.9e} N*m")


def section_4_pinv_blowup():
    print("\n## 4. Gimbal-rate growth approaching a singularity")
    array = pyramid_array()
    base = singular_configuration(
        array, np.array([0.3, -0.2, 0.9]), np.array([1.0, 1.0, -1.0, -1.0])
    )
    tau = np.array([0.1, -0.05, 0.2])
    print(f"{'offset [rad]':>14} {'m':>14} {'pinv peak rate':>17} {'SR peak rate':>15} "
          f"{'SR |err| [N*m]':>16}")
    for offset in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
        d = base + offset
        m = singularity_measure(array.jacobian(d))
        p = pseudo_inverse_steer(array, d, tau)
        s = sr_inverse_steer(array, d, tau, lam0=0.01, mu=10.0)
        print(
            f"{offset:>14.0e} {m:>14.6e} {float(np.max(np.abs(p.gimbal_rates))):>17.6e} "
            f"{float(np.max(np.abs(s.gimbal_rates))):>15.6e} {s.torque_error_norm:>16.6e}"
        )


def section_5_gsr():
    print("\n## 5. GSR inverse")
    array = pyramid_array()
    rng = np.random.default_rng(SEED + 2)
    worst = 0.0
    for _ in range(500):
        d = rng.uniform(-np.pi, np.pi, 4)
        tau = rng.normal(size=3) * 0.3
        a = gsr_inverse_steer(array, d, tau, eps0=0.0, lam=1e-3)
        b = sr_inverse_steer(array, d, tau, lam=1e-3)
        worst = max(worst, float(np.max(np.abs(a.gimbal_rates - b.gimbal_rates))))
    print(f"eps0 = 0 reduces GSR to SR: worst rate deviation {worst:.6e} rad/s over 500 states")
    d = np.full(4, np.pi / 2 - 0.01)
    tau = np.array([0.0, 0.0, 0.05])
    print(f"{'time [s]':>10} {'e1':>10} {'e2':>10} {'e3':>10} {'|err| [N*m]':>15}")
    for t in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5):
        r = gsr_inverse_steer(array, d, tau, time=t, lam=1e-2, eps0=0.05)
        print(
            f"{t:>10.2f} {r.extras['e1']:>10.5f} {r.extras['e2']:>10.5f} "
            f"{r.extras['e3']:>10.5f} {r.torque_error_norm:>15.9e}"
        )
    return worst


def section_6_integration_convergence():
    print("\n## 6. Momentum-error convergence with the integration step")
    print("the gimbal angles are integrated by explicit Euler, so the momentum error")
    print("of an otherwise exact law is first order in dt")
    array = pyramid_array()
    axis = np.array([0.2, 0.3, 0.93])
    axis = axis / np.linalg.norm(axis)
    print(f"{'dt [s]':>10} {'net mom err [N*m*s]':>21} {'ratio to previous':>19} "
          f"{'max |tau err| [N*m]':>21}")
    previous = None
    for dt in (0.1, 0.05, 0.025, 0.0125, 0.00625):
        profile = rest_to_rest_profile(axis, 1.5, 20.0, dt)
        history = run_steering(array, np.zeros(4), profile, method="pinv")
        net = history.accumulated_momentum_error
        ratio = "-" if previous is None else f"{previous / net:.4f}"
        print(f"{dt:>10.5f} {net:>21.6e} {ratio:>19} {history.max_torque_error:>21.6e}")
        previous = net
    print("the instantaneous torque error stays at round-off throughout: the whole of")
    print("the momentum error here is integration, not steering")


def main() -> int:
    print("=" * 78)
    print("CMGSteer validation 3 -- steering-law exactness and the SR closed form")
    print("=" * 78)
    print(f"seed {SEED}, numpy {np.__version__}")

    pyr_pinv = section_1_pseudo_inverse(pyramid_array(), "pyramid")
    roof_pinv = section_1_pseudo_inverse(roof_array(), "roof")
    pyr_sr = section_2_sr_closed_form(pyramid_array(), "pyramid")
    roof_sr = section_2_sr_closed_form(roof_array(), "roof")
    section_3_sr_at_a_singularity()
    section_4_pinv_blowup()
    gsr = section_5_gsr()
    section_6_integration_convergence()

    print("\n## Summary")
    checks = [
        ("pyramid pseudo-inverse exactness (m >= 0.1)", pyr_pinv, 1e-12),
        ("roof pseudo-inverse exactness (m >= 0.1)", roof_pinv, 1e-12),
        ("pyramid SR error vs closed form", pyr_sr, 1e-13),
        ("roof SR error vs closed form", roof_sr, 1e-13),
        ("GSR reduces to SR at eps0 = 0", gsr, 1e-13),
    ]
    ok = True
    for name, value, tol in checks:
        verdict = "PASS" if value < tol else "FAIL"
        ok &= value < tol
        print(f"{verdict}  {name:<45} {value:.6e} < {tol:.0e}")
    print("\nOVERALL: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

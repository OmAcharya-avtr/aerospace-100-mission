"""Validation 1: the Jacobian against numerical differentiation of the momentum map.

Establishes that ``A(delta) = dh/ddelta`` as implemented is the derivative of
``h(delta)`` as implemented, that the analytic manipulability gradient is the
derivative of the singularity measure, and that the pyramid momentum map agrees
with the closed form quoted throughout the SGCMG literature.

Run: ``python validation/validate_jacobian.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cmgsteer.arrays import (  # noqa: E402
    STANDARD_PYRAMID_SKEW_DEG,
    pyramid_array,
    roof_array,
)
from cmgsteer.singularity import (  # noqa: E402
    manipulability_gradient,
    singularity_measure,
)

SEED = 20260902
N_CONFIGS = 400


def central_jacobian(array, deltas, step):
    """Central-difference Jacobian of the momentum map, shape (3, n)."""
    out = np.empty((3, array.n_cmgs))
    for j in range(array.n_cmgs):
        plus, minus = deltas.copy(), deltas.copy()
        plus[j] += step
        minus[j] -= step
        out[:, j] = (array.momentum(plus) - array.momentum(minus)) / (2.0 * step)
    return out


def section_1_step_study(array, label):
    print(f"\n## 1. {label}: Jacobian vs central differences, step study")
    rng = np.random.default_rng(SEED)
    configs = rng.uniform(-np.pi, np.pi, (N_CONFIGS, array.n_cmgs))
    print(f"{'step [rad]':>12} {'worst abs dev':>16} {'worst rel dev':>16}")
    best = (np.inf, None)
    for step in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8):
        worst_abs = 0.0
        worst_rel = 0.0
        for d in configs:
            analytic = array.jacobian(d, free_only=False)
            numeric = central_jacobian(array, d, step)
            dev = np.abs(analytic - numeric)
            worst_abs = max(worst_abs, float(dev.max()))
            scale = max(float(np.abs(analytic).max()), 1e-30)
            worst_rel = max(worst_rel, float(dev.max() / scale))
        print(f"{step:>12.0e} {worst_abs:>16.6e} {worst_rel:>16.6e}")
        if worst_abs < best[0]:
            best = (worst_abs, step)
    print(f"best step {best[1]:.0e}, worst absolute deviation {best[0]:.6e} N*m*s/rad")
    print(f"over {N_CONFIGS} configurations drawn uniformly from [-pi, pi]^{array.n_cmgs}")
    return best


def section_2_gradient(array, label):
    print(f"\n## 2. {label}: manipulability gradient vs central differences")
    rng = np.random.default_rng(SEED + 1)
    worst_abs = 0.0
    worst_rel = 0.0
    step = 1e-6
    for _ in range(N_CONFIGS):
        d = rng.uniform(-np.pi, np.pi, array.n_cmgs)
        analytic = manipulability_gradient(array, d)
        numeric = np.empty(array.n_free)
        for j, col in enumerate(array.free_indices):
            plus, minus = d.copy(), d.copy()
            plus[col] += step
            minus[col] -= step
            numeric[j] = (
                singularity_measure(array.jacobian(plus))
                - singularity_measure(array.jacobian(minus))
            ) / (2.0 * step)
        dev = float(np.max(np.abs(analytic - numeric)))
        worst_abs = max(worst_abs, dev)
        worst_rel = max(worst_rel, dev / max(float(np.abs(analytic).max()), 1e-30))
    print(f"step {step:.0e}, worst absolute deviation {worst_abs:.6e}")
    print(f"worst relative deviation {worst_rel:.6e} over {N_CONFIGS} configurations")
    return worst_abs


def section_3_closed_form():
    print("\n## 3. Pyramid momentum map vs the published closed form")
    array = pyramid_array()
    beta = np.radians(STANDARD_PYRAMID_SKEW_DEG)
    cb, sb = np.cos(beta), np.sin(beta)
    rng = np.random.default_rng(SEED + 2)
    worst = 0.0
    for _ in range(2000):
        d = rng.uniform(-np.pi, np.pi, 4)
        expected = np.array(
            [
                -cb * np.sin(d[0]) - np.cos(d[1]) + cb * np.sin(d[2]) + np.cos(d[3]),
                np.cos(d[0]) - cb * np.sin(d[1]) - np.cos(d[2]) + cb * np.sin(d[3]),
                sb * np.sum(np.sin(d)),
            ]
        )
        worst = max(worst, float(np.max(np.abs(array.momentum(d) - expected))))
    print(f"skew angle {STANDARD_PYRAMID_SKEW_DEG:.9f} deg, sin = {sb:.15f}, cos = {cb:.15f}")
    print(f"worst deviation over 2000 configurations: {worst:.6e} N*m*s")
    return worst


def section_4_torque_convention():
    print("\n## 4. Torque convention: tau = -A ddelta round trip")
    array = pyramid_array()
    rng = np.random.default_rng(SEED + 3)
    worst = 0.0
    for _ in range(2000):
        d = rng.uniform(-np.pi, np.pi, 4)
        rates = rng.normal(size=4)
        direct = array.torque(d, rates)
        expected = -(array.jacobian(d) @ rates)
        worst = max(worst, float(np.max(np.abs(direct - expected))))
    print(f"worst deviation over 2000 (configuration, rate) pairs: {worst:.6e} N*m")
    return worst


def main() -> int:
    print("=" * 78)
    print("CMGSteer validation 1 -- Jacobian vs numerical differentiation")
    print("=" * 78)
    print(f"seed {SEED}, numpy {np.__version__}")

    results = {}
    for array, label in ((pyramid_array(), "pyramid"), (roof_array(), "roof")):
        results[label] = section_1_step_study(array, label)
        results[label + "_grad"] = section_2_gradient(array, label)
    closed = section_3_closed_form()
    torque = section_4_torque_convention()

    print("\n## Summary")
    tol_jac = 1e-8
    tol_grad = 1e-7
    tol_exact = 1e-13
    checks = [
        ("pyramid Jacobian vs central differences", results["pyramid"][0], tol_jac),
        ("roof Jacobian vs central differences", results["roof"][0], tol_jac),
        ("pyramid gradient vs central differences", results["pyramid_grad"], tol_grad),
        ("roof gradient vs central differences", results["roof_grad"], tol_grad),
        ("pyramid momentum map vs closed form", closed, tol_exact),
        ("torque convention round trip", torque, tol_exact),
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

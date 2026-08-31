"""Validation 1: allocation reproduces a commanded torque inside the AMS.

Checks, for three effector configurations:

* every allocator's torque residual on commands sampled strictly inside the
  attainable moment set;
* the bound violation each allocator leaves behind;
* the QP's 1/gamma residual trade-off, swept over gamma;
* the HiGHS-versus-CBC agreement of the linear programme.

Run: ``python validation/validate_exact_allocation.py``
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from alloclab.allocation import (  # noqa: E402
    METHODS,
    allocate,
    lp_allocate,
    qp_allocate,
)
from alloclab.ams import attainable_moment_set  # noqa: E402
from alloclab.dataset import reference_thruster_cluster  # noqa: E402
from alloclab.effectors import orthogonal_effectors, pyramid_reaction_wheels  # noqa: E402

SEED = 20260831
N_SAMPLES = 400
FILL = 0.8  # fraction of the AMS boundary radius each command is placed at

CONFIGS = {
    "orthogonal triad (m=3, +/-1)": orthogonal_effectors(1.0),
    "pyramid wheels (m=4, +/-0.1 N*m)": pyramid_reaction_wheels(0.1),
    "thruster cluster (m=8, [0,1] N, arm 0.5 m)": reference_thruster_cluster(1.0, 0.5),
}


def sample_interior(eset, n, rng, fill):
    """Torques at ``fill`` of the AMS boundary along uniformly random directions."""
    ams = attainable_moment_set(eset)
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return np.array([fill * ams.boundary_scale(v) * v for v in d])


def main() -> None:
    rng = np.random.default_rng(SEED)
    print("=" * 78)
    print("VALIDATION 1: exact allocation inside the attainable moment set")
    print(f"seed={SEED}  n_samples={N_SAMPLES}  commands at {FILL:.0%} of the AMS boundary")
    print("=" * 78)

    for label, eset in CONFIGS.items():
        taus = sample_interior(eset, N_SAMPLES, rng, FILL)
        print(f"\n--- {label} ---")
        print(
            f"{'method':<8}{'max |residual| [N*m]':>22}{'max bound viol':>18}"
            f"{'n infeasible':>14}{'mean solve [us]':>18}"
        )
        for method in METHODS:
            res = [allocate(eset, tau, method=method) for tau in taus]
            resid = np.array([r.residual_norm for r in res])
            viol = np.array([r.bound_violation for r in res])
            bad = int(np.sum([not r.feasible for r in res]))
            t = np.array([r.solve_time_s for r in res]).mean() * 1e6
            print(f"{method:<8}{resid.max():>22.6e}{viol.max():>18.6e}{bad:>14d}{t:>18.1f}")

    print("\n" + "=" * 78)
    print("QP residual versus gamma (pyramid wheels, 200 interior commands)")
    print("=" * 78)
    eset = CONFIGS["pyramid wheels (m=4, +/-0.1 N*m)"]
    taus = sample_interior(eset, 200, np.random.default_rng(SEED + 1), FILL)
    print(f"{'gamma':>10}{'max |residual| [N*m]':>24}{'max bound violation':>22}")
    for gamma in (1e4, 1e6, 1e8, 1e10, 1e12, 1e14, 1e16):
        res = [qp_allocate(eset, tau, gamma=gamma) for tau in taus]
        resid = max(r.residual_norm for r in res)
        viol = max(r.bound_violation for r in res)
        print(f"{gamma:>10.0e}{resid:>24.6e}{viol:>22.6e}")

    print("\n" + "=" * 78)
    print("LP cross-check: scipy/HiGHS versus PuLP/CBC, min_error objective")
    print("=" * 78)
    for label, eset in CONFIGS.items():
        taus = sample_interior(eset, 40, np.random.default_rng(SEED + 2), FILL)
        t0 = time.perf_counter()
        highs = [lp_allocate(eset, tau, solver="highs") for tau in taus]
        t_highs = (time.perf_counter() - t0) / len(taus)
        t0 = time.perf_counter()
        cbc = [lp_allocate(eset, tau, solver="pulp") for tau in taus]
        t_cbc = (time.perf_counter() - t0) / len(taus)
        gap = max(
            float(np.linalg.norm(a.achieved_torque - b.achieved_torque))
            for a, b in zip(highs, cbc, strict=True)
        )
        print(f"{label}")
        print(f"  max |achieved torque difference|   : {gap:.6e} N*m")
        print(f"  max HiGHS residual                 : {max(r.residual_norm for r in highs):.6e}")
        print(f"  max CBC residual                   : {max(r.residual_norm for r in cbc):.6e}")
        print(f"  mean solve time HiGHS / CBC        : {t_highs * 1e3:.3f} / {t_cbc * 1e3:.3f} ms")

    print("\n" + "=" * 78)
    print("Commands OUTSIDE the AMS: bounds must still hold and the shortfall be reported")
    print("=" * 78)
    eset = CONFIGS["thruster cluster (m=8, [0,1] N, arm 0.5 m)"]
    taus = sample_interior(eset, 200, np.random.default_rng(SEED + 3), 1.5)
    print(f"{'method':<8}{'min |residual| [N*m]':>22}{'max bound viol':>18}{'n reported bad':>16}")
    for method in METHODS:
        res = [allocate(eset, tau, method=method) for tau in taus]
        resid = np.array([r.residual_norm for r in res])
        viol = np.array([r.bound_violation for r in res])
        bad = int(np.sum([not r.feasible for r in res]))
        print(f"{method:<8}{resid.min():>22.6e}{viol.max():>18.6e}{bad:>16d}")
    print("\nAll 200 commands here are at 150% of the boundary, so the correct")
    print("count in the last column is 200 for every method.")


if __name__ == "__main__":
    main()

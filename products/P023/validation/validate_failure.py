"""Validation 4: failure reallocation meets what it can and reports what it cannot.

The claim under test has two halves and both are checked separately:

* when the degraded effector set **can** produce the command, reallocation
  must produce it exactly and inside the remaining actuator bounds;
* when it **cannot**, the result must carry ``status="infeasible"`` and the
  size of the shortfall -- never a silently clipped command reported as a
  success.

Feasibility is decided independently of the allocator, by the exact LP
certificate ``alloclab.allocation.is_attainable``.

Run: ``python validation/validate_failure.py``
"""

import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from alloclab.allocation import InfeasibleAllocationError, is_attainable  # noqa: E402
from alloclab.ams import attainable_moment_set  # noqa: E402
from alloclab.dataset import reference_thruster_cluster  # noqa: E402
from alloclab.effectors import pyramid_reaction_wheels  # noqa: E402
from alloclab.failure import failure_margin, reallocate_after_failure  # noqa: E402

SEED = 90210
N_DIRECTIONS = 60
TORQUE_TOL = 1e-8
BOUND_TOL = 1e-9

CONFIGS = {
    "thruster cluster (m=8)": reference_thruster_cluster(1.0, 0.5),
    "pyramid wheels (m=4)": pyramid_reaction_wheels(0.1),
}


def sweep(eset, failed, rng, method):
    """Sweep command magnitudes across the degraded AMS boundary in each direction."""
    degraded = eset.with_failures(failed)
    ams = attainable_moment_set(degraded)
    n_ok = n_missed = n_infeasible_reported = n_infeasible_true = 0
    n_wrong_verdict = 0
    worst_resid_inside = 0.0
    worst_viol = 0.0
    if ams.degenerate:
        return None

    dirs = rng.normal(size=(N_DIRECTIONS, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    for d in dirs:
        scale = ams.boundary_scale(d)
        if not np.isfinite(scale) or scale <= 0.0:
            continue
        for rho in (0.25, 0.6, 0.9, 1.1, 1.5, 2.5):
            tau = rho * scale * d
            report = reallocate_after_failure(
                eset, tau, failed, method=method, compute_volume=False
            )
            res = report.degraded
            truly = is_attainable(degraded, tau, tol=TORQUE_TOL)
            worst_viol = max(worst_viol, res.bound_violation)
            if truly:
                n_infeasible_true += 0
                if res.feasible and res.residual_norm <= TORQUE_TOL:
                    n_ok += 1
                else:
                    n_missed += 1
                worst_resid_inside = max(worst_resid_inside, res.residual_norm)
            else:
                n_infeasible_true += 1
                if res.status == "infeasible" and not res.feasible:
                    n_infeasible_reported += 1
            if report.attainable != truly:
                n_wrong_verdict += 1
    return {
        "n_ok": n_ok,
        "n_missed": n_missed,
        "n_infeasible_true": n_infeasible_true,
        "n_infeasible_reported": n_infeasible_reported,
        "n_wrong_verdict": n_wrong_verdict,
        "worst_resid_inside": worst_resid_inside,
        "worst_viol": worst_viol,
    }


def main() -> None:
    print("=" * 78)
    print("VALIDATION 4a: failed-effector reallocation, systematic sweep")
    print(f"seed={SEED}  {N_DIRECTIONS} directions x 6 magnitudes per failure case")
    print("=" * 78)

    total_missed = 0
    total_unreported = 0
    total_wrong = 0
    for label, eset in CONFIGS.items():
        m = eset.n_effectors
        cases = [(i,) for i in range(m)] + list(combinations(range(m), 2))[:8]
        print(f"\n--- {label}, method='qp' ---")
        print(
            f"{'failed':<12}{'rank':>5}{'met/attainable':>16}{'missed':>8}"
            f"{'infeas reported/true':>22}{'max resid':>13}{'max viol':>11}"
        )
        for failed in cases:
            rng = np.random.default_rng(SEED + sum(failed) * 17 + len(failed))
            out = sweep(eset, list(failed), rng, "qp")
            if out is None:
                print(f"{str(list(failed)):<12}{eset.with_failures(list(failed)).rank:>5}"
                      f"{'AMS degenerate (rank < 3): swept skipped':>60}")
                continue
            total_missed += out["n_missed"]
            total_unreported += out["n_infeasible_true"] - out["n_infeasible_reported"]
            total_wrong += out["n_wrong_verdict"]
            print(
                f"{str(list(failed)):<12}"
                f"{eset.with_failures(list(failed)).rank:>5}"
                f"{out['n_ok']:>10}/{out['n_ok'] + out['n_missed']:<5}"
                f"{out['n_missed']:>8}"
                f"{out['n_infeasible_reported']:>14}/{out['n_infeasible_true']:<7}"
                f"{out['worst_resid_inside']:>13.3e}{out['worst_viol']:>11.3e}"
            )

    print("\nPASS criteria:")
    print(f"  attainable commands missed by the QP        : {total_missed}   "
          f"({'PASS' if total_missed == 0 else 'FAILED'}, must be 0)")
    print(f"  unattainable commands not reported infeasible: {total_unreported}   "
          f"({'PASS' if total_unreported == 0 else 'FAILED'}, must be 0)")
    print(f"  feasibility verdicts disagreeing with the LP : {total_wrong}   "
          f"({'PASS' if total_wrong == 0 else 'FAILED'}, must be 0)")

    print("\n" + "=" * 78)
    print("VALIDATION 4b: the same sweep for the redistributed pseudo-inverse")
    print("=" * 78)
    eset = CONFIGS["thruster cluster (m=8)"]
    print(
        f"{'failed':<12}{'met/attainable':>16}{'missed':>8}"
        f"{'infeas reported/true':>22}{'max resid':>13}{'max viol':>11}"
    )
    rpi_missed = 0
    for failed in [(i,) for i in range(8)]:
        rng = np.random.default_rng(SEED + sum(failed) * 17 + len(failed))
        out = sweep(eset, list(failed), rng, "rpi")
        rpi_missed += out["n_missed"]
        print(
            f"{str(list(failed)):<12}"
            f"{out['n_ok']:>10}/{out['n_ok'] + out['n_missed']:<5}"
            f"{out['n_missed']:>8}"
            f"{out['n_infeasible_reported']:>14}/{out['n_infeasible_true']:<7}"
            f"{out['worst_resid_inside']:>13.3e}{out['worst_viol']:>11.3e}"
        )
    print(f"\nredistributed-pseudo-inverse attainable commands missed: {rpi_missed}")
    print("This is a property of the heuristic, not a defect in this code; the")
    print("report still says the command WAS attainable, so the caller is told.")

    print("\n" + "=" * 78)
    print("VALIDATION 4c: worked cases, printed in full")
    print("=" * 78)
    cluster = CONFIGS["thruster cluster (m=8)"]

    print("\n[1] +x command of 0.05 N*m after t1 fails off -- expected: still met")
    rep = reallocate_after_failure(cluster, [0.05, 0.0, 0.0], [0], method="qp")
    print(f"    attainable        : {rep.attainable}")
    print(f"    status            : {rep.degraded.status}")
    print(f"    commands [N]      : {np.array2string(rep.degraded.commands, precision=6)}")
    print(f"    residual [N*m]    : {rep.degraded.residual_norm:.6e}")
    print(f"    bound violation   : {rep.degraded.bound_violation:.6e}")
    print(f"    AMS volume ratio  : {rep.volume_ratio:.6f}")
    print(f"    failure margin    : {failure_margin(cluster, [0.05, 0.0, 0.0], [0]):.4f}")

    print("\n[2] +x command of 0.6 N*m after t1 fails off -- expected: NOT met")
    rep = reallocate_after_failure(cluster, [0.6, 0.0, 0.0], [0], method="qp")
    print(f"    attainable        : {rep.attainable}")
    print(f"    status            : {rep.degraded.status}")
    print(f"    commands [N]      : {np.array2string(rep.degraded.commands, precision=6)}")
    print(f"    achieved [N*m]    : {np.array2string(rep.degraded.achieved_torque, precision=6)}")
    print(f"    residual [N*m]    : {rep.degraded.residual_norm:.6e}")
    print(f"    bound violation   : {rep.degraded.bound_violation:.6e}")
    print(f"    failure margin    : {failure_margin(cluster, [0.6, 0.0, 0.0], [0]):.4f}")
    print(f"    message           : {rep.degraded.message}")

    print("\n[3] t1 stuck OPEN at 1 N, zero torque commanded -- must be cancelled")
    rep = reallocate_after_failure(cluster, np.zeros(3), [0], stuck_at=1.0, method="qp")
    print(f"    attainable        : {rep.attainable}")
    print(f"    status            : {rep.degraded.status}")
    print(f"    commands [N]      : {np.array2string(rep.degraded.commands, precision=6)}")
    print(f"    achieved [N*m]    : {np.array2string(rep.degraded.achieved_torque, precision=6)}")
    print(f"    residual [N*m]    : {rep.degraded.residual_norm:.6e}")

    print("\n[4] all four z-authority thrusters lost -- rank collapse")
    rep = reallocate_after_failure(cluster, [0.0, 0.0, 0.2], [4, 5, 6, 7], method="qp")
    print(f"    remaining rank    : {rep.remaining_rank}")
    print(f"    attainable        : {rep.attainable}")
    print(f"    status            : {rep.degraded.status}")
    print(f"    residual [N*m]    : {rep.degraded.residual_norm:.6e} (command was 0.2)")
    print(f"    AMS volume ratio  : {rep.volume_ratio:.6f}")

    print("\n[5] require_feasible=True must raise, not return a clipped command")
    try:
        reallocate_after_failure(
            cluster, [5.0, 0.0, 0.0], [0, 1], method="qp", require_feasible=True
        )
        print("    FAILED: no exception raised")
    except InfeasibleAllocationError as exc:
        print(f"    InfeasibleAllocationError raised: {exc}")


if __name__ == "__main__":
    main()

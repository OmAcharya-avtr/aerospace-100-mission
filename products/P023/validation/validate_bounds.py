"""Validation 3: every allocation respects the actuator bounds.

Two parts.

**Monte Carlo** over randomly generated effector configurations and randomly
generated commands -- including commands well outside the attainable moment set
-- measuring the worst bound violation each method leaves behind.

**Property test**: the same claim is stated as a Hypothesis property in
``tests/test_properties.py`` and is executed here so its outcome is recorded
alongside the Monte-Carlo numbers rather than only in the test log.

Run: ``python validation/validate_bounds.py``
"""

import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alloclab.allocation import METHODS, allocate  # noqa: E402
from alloclab.ams import attainable_moment_set  # noqa: E402
from alloclab.effectors import general_effector_set  # noqa: E402

SEED = 4242
N_CONFIGS = 300
N_COMMANDS = 8

PROPERTY_TESTS = [
    "tests/test_properties.py::test_qp_allocation_always_respects_actuator_bounds",
    "tests/test_properties.py::test_lp_allocation_always_respects_actuator_bounds",
    "tests/test_properties.py::test_redistributed_pseudo_inverse_always_respects_actuator_bounds",
    "tests/test_properties.py::test_allocation_reproduces_any_attainable_command_exactly",
]


def random_config(rng):
    """A random full-rank effector set with random, possibly one-sided, bounds."""
    while True:
        m = int(rng.integers(3, 9))
        b = rng.normal(size=(3, m))
        if np.linalg.matrix_rank(b) < 3:
            continue
        if np.min(np.linalg.svd(b, compute_uv=False)) < 1e-2:
            continue
        lower = rng.uniform(-1.0, 0.5, size=m)
        upper = lower + rng.uniform(0.1, 2.0, size=m)
        return general_effector_set(b, lower, upper)


def main() -> None:
    rng = np.random.default_rng(SEED)
    print("=" * 78)
    print("VALIDATION 3a: Monte-Carlo bound compliance")
    print(f"seed={SEED}  {N_CONFIGS} random configurations x {N_COMMANDS} random commands")
    print("commands drawn at rho x the AMS boundary radius, rho ~ U(0, 2);")
    print("membership is then decided by the hull itself, not by rho")
    print("=" * 78)

    worst = {m: 0.0 for m in METHODS}
    n_over = {m: 0 for m in METHODS}
    n_total = 0
    worst_inside_residual = {m: 0.0 for m in METHODS}
    n_missed_inside = {m: 0 for m in METHODS}
    n_inside = 0

    for _ in range(N_CONFIGS):
        eset = random_config(rng)
        ams = attainable_moment_set(eset)
        if ams.degenerate:  # pragma: no cover - full-rank by construction
            continue
        for _ in range(N_COMMANDS):
            d = rng.normal(size=3)
            d /= np.linalg.norm(d)
            scale = ams.boundary_scale(d)
            if not np.isfinite(scale) or scale <= 0.0:
                continue
            rho = rng.uniform(0.0, 2.0)
            tau = rho * scale * d
            n_total += 1
            # Decide membership with the hull itself, not from rho: these
            # random command boxes need not contain a zero-torque point, so a
            # ray from the origin is not a reliable parameterisation of the
            # set.
            inside = bool(ams.contains(tau, tol=-1e-9))
            if inside:
                n_inside += 1
            for method in METHODS:
                res = allocate(eset, tau, method=method)
                worst[method] = max(worst[method], res.bound_violation)
                if res.bound_violation > 1e-9:
                    n_over[method] += 1
                if inside:
                    worst_inside_residual[method] = max(
                        worst_inside_residual[method], res.residual_norm
                    )
                    if res.residual_norm > 1e-8:
                        n_missed_inside[method] += 1

    print(f"\n{n_total} commands evaluated, {n_inside} of them inside the AMS\n")
    print(
        f"{'method':<8}{'max bound violation':>22}{'n violating':>13}{'% viol':>9}"
        f"{'max resid inside':>19}{'n missed inside':>17}"
    )
    for method in METHODS:
        pct = 100.0 * n_over[method] / n_total
        print(
            f"{method:<8}{worst[method]:>22.6e}{n_over[method]:>13d}{pct:>9.2f}"
            f"{worst_inside_residual[method]:>19.6e}{n_missed_inside[method]:>17d}"
        )
    print("\nPASS criterion: max bound violation <= 1e-9 for lp, qp and rpi.")
    for method in ("lp", "qp", "rpi"):
        verdict = "PASS" if worst[method] <= 1e-9 else "FAILED"
        print(f"  {method:<6}: {verdict} (max violation {worst[method]:.6e})")
    print("pinv and wpinv are unconstrained by construction and are expected to")
    print("violate; their numbers above quantify by how much.")
    print()
    print("'n missed inside' counts attainable commands the method failed to meet")
    print("to 1e-8 N*m while staying inside the box -- the interesting failure,")
    print("because a solution existed. lp and qp must be 0 here. rpi is a")
    print("heuristic and is expected to be non-zero: Bodson (2002) sec. V.A and")
    print("Haerkegaard (2002) sec. 2.2.1 both say the redistributed pseudoinverse")
    print("is not guaranteed to find a feasible command when one exists.")

    print("\n" + "=" * 78)
    print("VALIDATION 3b: the same claim as a Hypothesis property test")
    print("=" * 78)
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:randomly", *PROPERTY_TESTS]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)
    print("$ " + " ".join(["python", "-m", "pytest", "-q", *PROPERTY_TESTS]))
    print(proc.stdout.strip()[-2000:])
    if proc.stderr.strip():
        print(proc.stderr.strip()[-1000:])
    print(f"\npytest exit code: {proc.returncode}  ({'PASS' if proc.returncode == 0 else 'FAILED'})")


if __name__ == "__main__":
    main()

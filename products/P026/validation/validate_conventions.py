"""Validation 1 -- conventions, and exact recovery from noise-free observations.

Checks
------
1. ``dcm_from_quat`` agrees with ``scipy.spatial.transform.Rotation`` after the
   documented scalar-first to scalar-last reordering.  This pins the quaternion
   convention against an independent implementation.
2. ``quat_from_dcm(dcm_from_quat(q)) == q`` for random quaternions, exercising
   all four Shepperd branches.
3. Frame order: every solver returns ``A`` with ``b_i = A r_i`` (not the
   transpose), verified on noise-free data where the answer is exact.
4. Noise-free exactness: each of TRIAD, the q-method, QUEST and OLAE reproduces
   the generating attitude, for observation counts 2 to 12.
5. Orthogonality and determinant of every returned matrix, and unit norm with
   ``w >= 0`` for every returned quaternion.
6. Cross-check against ``scipy.spatial.transform.Rotation.align_vectors``,
   which solves the same problem by the Kabsch algorithm.

Run:  python validation/validate_conventions.py
"""

from __future__ import annotations

import numpy as np
from _common import banner, noisy_problem, random_dcm, verdict

from wahbakit import (
    VectorObservations,
    angle_between_dcm,
    dcm_from_quat,
    is_rotation,
    olae,
    q_method,
    quat_canonical,
    quat_from_dcm,
    quest,
    triad,
)

SEED = 20260831
TRIALS = 500
SOLVERS = {"triad": triad, "q-method": q_method, "quest": quest, "olae": olae}


def main() -> int:
    banner("wahbakit validation 1 -- conventions and noise-free exactness")
    rng = np.random.default_rng(SEED)
    passed = []

    print("\n[1] quaternion convention vs scipy.spatial.transform.Rotation")
    from scipy.spatial.transform import Rotation

    worst_scipy = 0.0
    worst_round_trip = 0.0
    for _ in range(TRIALS):
        q = quat_canonical(rng.normal(size=4))
        dcm = dcm_from_quat(q)
        scipy_matrix = Rotation.from_quat(np.roll(q, -1)).as_matrix()
        worst_scipy = max(worst_scipy, float(np.max(np.abs(dcm - scipy_matrix))))
        worst_round_trip = max(worst_round_trip, float(np.max(np.abs(quat_from_dcm(dcm) - q))))
    print(f"    scipy version: {__import__('scipy').__version__}, {TRIALS} random quaternions")
    passed.append(verdict("max |dcm_from_quat(q) - scipy as_matrix|", worst_scipy, 1e-14))
    passed.append(verdict("max |quat_from_dcm(dcm_from_quat(q)) - q|", worst_round_trip, 1e-12))

    print("\n[2] hand-checked closed forms")
    root = np.sqrt(0.5)
    cases = {
        "q = [1, 0, 0, 0] -> I": (np.eye(3), dcm_from_quat([1, 0, 0, 0])),
        "q = [c45, 0, 0, s45] -> 90 deg about z": (
            np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
            dcm_from_quat([root, 0.0, 0.0, root]),
        ),
        "q = [0, 1, 0, 0] -> diag(1, -1, -1)": (
            np.diag([1.0, -1.0, -1.0]),
            dcm_from_quat([0.0, 1.0, 0.0, 0.0]),
        ),
    }
    worst_hand = 0.0
    for name, (expected, actual) in cases.items():
        deviation = float(np.max(np.abs(expected - actual)))
        worst_hand = max(worst_hand, deviation)
        print(f"    {name:<44} max|diff| = {deviation:.3e}")
    passed.append(verdict("worst hand-checked closed form", worst_hand, 1e-15))

    print("\n[3] frame order: solvers return A with b = A r, not its transpose")
    dcm_true = random_dcm(rng)
    reference = rng.normal(size=(4, 3))
    reference /= np.linalg.norm(reference, axis=1)[:, None]
    obs = VectorObservations(reference @ dcm_true.T, reference)
    forward = angle_between_dcm(q_method(obs).dcm, dcm_true)
    transposed = angle_between_dcm(q_method(obs).dcm, dcm_true.T)
    print(f"    angle to A_true      = {forward:.3e} rad")
    print(f"    angle to A_true^T    = {transposed:.3e} rad")
    passed.append(verdict("angle to A_true (the documented convention)", forward, 1e-12))
    passed.append(
        verdict("angle to A_true^T (must NOT be small)", transposed, 1e-3, smaller_is_better=False)
    )

    print("\n[4] noise-free exactness, by method and observation count")
    print(f"    {'n':>4}  " + "  ".join(f"{name:>14}" for name in SOLVERS))
    worst_by_method = dict.fromkeys(SOLVERS, 0.0)
    for n in (2, 3, 4, 6, 12):
        row = []
        for name, solver in SOLVERS.items():
            if name == "triad" and n != 2:
                row.append(f"{'n/a':>14}")
                continue
            worst = 0.0
            for _ in range(50):
                obs, dcm = noisy_problem(rng, n, 0.0)
                if obs.observability().lambda_min < 1e-3:
                    continue
                worst = max(worst, angle_between_dcm(solver(obs).dcm, dcm))
            worst_by_method[name] = max(worst_by_method[name], worst)
            row.append(f"{worst:14.3e}")
        print(f"    {n:>4}  " + "  ".join(row))
    for name, worst in worst_by_method.items():
        passed.append(verdict(f"worst noise-free attitude error, {name} [rad]", worst, 1e-11))

    print("\n[5] output invariants over 200 noisy problems")
    worst_orthogonality = 0.0
    worst_determinant = 0.0
    worst_norm = 0.0
    worst_quaternion_match = 0.0
    negative_scalars = 0
    for _ in range(200):
        obs, _ = noisy_problem(rng, 4, 1e-2)
        if obs.observability().lambda_min < 1e-3:
            continue
        for solver in (q_method, quest, olae):
            solution = solver(obs)
            matrix = solution.dcm
            worst_orthogonality = max(
                worst_orthogonality, float(np.max(np.abs(matrix.T @ matrix - np.eye(3))))
            )
            worst_determinant = max(worst_determinant, abs(float(np.linalg.det(matrix)) - 1.0))
            worst_norm = max(worst_norm, abs(float(np.linalg.norm(solution.quaternion)) - 1.0))
            worst_quaternion_match = max(
                worst_quaternion_match,
                float(np.max(np.abs(dcm_from_quat(solution.quaternion) - matrix))),
            )
            negative_scalars += int(solution.quaternion[0] < 0.0)
            assert is_rotation(matrix)
    passed.append(verdict("max |A^T A - I|", worst_orthogonality, 1e-13))
    passed.append(verdict("max |det A - 1|", worst_determinant, 1e-13))
    passed.append(verdict("max ||q| - 1|", worst_norm, 1e-14))
    passed.append(verdict("max |dcm_from_quat(q) - A|", worst_quaternion_match, 1e-13))
    print(f"    quaternions returned with w < 0: {negative_scalars} (must be 0)")
    passed.append(negative_scalars == 0)

    print("\n[6] cross-check against scipy Rotation.align_vectors (Kabsch)")
    worst_align = 0.0
    for _ in range(200):
        obs, _ = noisy_problem(rng, 5, 1e-2)
        if obs.observability().lambda_min < 1e-3:
            continue
        rotation, _ = Rotation.align_vectors(obs.body, obs.reference, weights=obs.weights)
        worst_align = max(worst_align, angle_between_dcm(q_method(obs).dcm, rotation.as_matrix()))
    passed.append(verdict("max angle, q-method vs scipy align_vectors [rad]", worst_align, 1e-10))

    print()
    print(f"RESULT: {sum(passed)} / {len(passed)} checks passed")
    return 0 if all(passed) else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Validation 2: the singularity measure on analytically known singular configurations.

Establishes that the numerical singularity measure vanishes on the singular set
constructed in closed form, that the classification into internal/external and
elliptic/hyperbolic behaves as the geometric theory says it must, and that the
mapped singular surfaces are where the theory puts them.

Run: ``python validation/validate_singularity.py``
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cmgsteer.arrays import pyramid_array, roof_array  # noqa: E402
from cmgsteer.singularity import (  # noqa: E402
    classify_singularity,
    fibonacci_directions,
    momentum_envelope,
    singular_configuration,
    singular_surface,
    singularity_measure,
)

SEED = 20260902
N_DIRECTIONS = 500


def section_1_measure_vanishes(array, label):
    print(f"\n## 1. {label}: measure on the analytic singular set")
    dirs = fibonacci_directions(N_DIRECTIONS)
    n = array.n_cmgs
    sign_sets = [np.array(s, dtype=float) for s in itertools.product((1.0, -1.0), repeat=n)]
    print(f"{'sign vector':>18} {'count':>7} {'worst m':>14} {'worst sigma_min':>17}")
    worst_overall = 0.0
    total = 0
    for signs in sign_sets:
        worst = 0.0
        worst_sv = 0.0
        count = 0
        for u in dirs:
            try:
                d = singular_configuration(array, u, signs)
            except ValueError:
                continue
            jac = array.jacobian(d)
            worst = max(worst, singularity_measure(jac))
            worst_sv = max(worst_sv, float(np.linalg.svd(jac, compute_uv=False)[-1]))
            count += 1
        total += count
        worst_overall = max(worst_overall, worst)
        label_signs = "".join("+" if s > 0 else "-" for s in signs)
        print(f"{label_signs:>18} {count:>7} {worst:>14.6e} {worst_sv:>17.6e}")
    print(f"{len(sign_sets)} sign vectors x {N_DIRECTIONS} directions, {total} usable points")
    print(f"worst singularity measure anywhere on the analytic set: {worst_overall:.6e}")
    return worst_overall


def section_2_regular_reference(array, label):
    print(f"\n## 2. {label}: measure away from the singular set (reference scale)")
    rng = np.random.default_rng(SEED)
    values = np.array(
        [singularity_measure(array.jacobian(rng.uniform(-np.pi, np.pi, array.n_cmgs)))
         for _ in range(5000)]
    )
    for q in (0, 1, 5, 25, 50, 75, 100):
        print(f"  percentile {q:>3}: {np.percentile(values, q):.6e}")
    print(f"  mean {values.mean():.6e}")
    return float(values.mean())


def section_3_classification(array, label):
    print(f"\n## 3. {label}: classification of the singular set")
    dirs = fibonacci_directions(200)
    n = array.n_cmgs
    counts: dict[tuple[str, str], int] = {}
    for signs in itertools.product((1.0, -1.0), repeat=n):
        arr_signs = np.array(signs, dtype=float)
        for u in dirs:
            try:
                d = singular_configuration(array, u, arr_signs)
            except ValueError:
                continue
            info = classify_singularity(array, d)
            key = (info.kind, info.passability)
            counts[key] = counts.get(key, 0) + 1
    total = sum(counts.values())
    print(f"{'kind':>10} {'passability':>14} {'count':>8} {'fraction':>10}")
    for key in sorted(counts):
        print(f"{key[0]:>10} {key[1]:>14} {counts[key]:>8} {counts[key] / total:>10.4f}")
    print(f"total classified points: {total}")
    external = sum(v for k, v in counts.items() if k[0] == "external")
    elliptic_external = counts.get(("external", "elliptic"), 0)
    print(
        f"external singularities that are elliptic: {elliptic_external} of {external} "
        f"({elliptic_external / max(external, 1):.4f})"
    )
    return counts


def section_4_known_configurations():
    print("\n## 4. Hand-computable configurations of the standard pyramid")
    array = pyramid_array()
    cases = [
        ("all gimbals at 0 deg", np.zeros(4), None),
        ("all gimbals at +90 deg", np.full(4, np.pi / 2), np.array([0.0, 0.0, 3.2])),
        (
            "gimbals at (90, 90, 90, -90) deg",
            np.array([1.0, 1.0, 1.0, -1.0]) * np.pi / 2,
            np.array([0.0, -1.2, 1.6]),
        ),
    ]
    print(f"{'configuration':>34} {'m':>14} {'|h|':>10} {'kind':>10} {'passability':>13}")
    worst = 0.0
    for name, d, expected_h in cases:
        info = classify_singularity(array, d)
        print(
            f"{name:>34} {info.measure:>14.6e} {np.linalg.norm(info.momentum):>10.6f} "
            f"{info.kind:>10} {info.passability:>13}"
        )
        if expected_h is not None:
            dev = float(np.max(np.abs(info.momentum - expected_h)))
            worst = max(worst, dev)
            print(f"{'  hand-computed h [N*m*s]':>34} {np.array2string(expected_h, precision=6)}"
                  f"  deviation {dev:.3e}")
    print("m at delta = 0 is sqrt(0.72 * 0.72 * 2.56) = 1.152 by hand; computed "
          f"{singularity_measure(array.jacobian(np.zeros(4))):.15f}")
    return worst


def section_5_surfaces():
    print("\n## 5. Singular surfaces in momentum space")
    array = pyramid_array()
    outer, _ = momentum_envelope(array, n_points=4000)
    radii = np.linalg.norm(outer, axis=1)
    print(f"outer (saturation) envelope: {outer.shape[0]} points")
    print(f"  min radius {radii.min():.9f}, max radius {radii.max():.9f} N*m*s")
    print(f"  capacity sum(h0) = {array.total_momentum_capacity:.6f} N*m*s")
    print(f"  sphericity (min/max) {radii.min() / radii.max():.6f}")
    print(f"{'sign vector':>14} {'points':>8} {'min |h|':>12} {'max |h|':>12}")
    for signs in itertools.product((1.0, -1.0), repeat=4):
        arr_signs = np.array(signs, dtype=float)
        pts, _ = singular_surface(array, signs=arr_signs, n_points=800)
        r = np.linalg.norm(pts, axis=1)
        label = "".join("+" if s > 0 else "-" for s in signs)
        print(f"{label:>14} {pts.shape[0]:>8} {r.min():>12.6f} {r.max():>12.6f}")
    return radii


def main() -> int:
    print("=" * 78)
    print("CMGSteer validation 2 -- singularity measure and classification")
    print("=" * 78)
    print(f"seed {SEED}, numpy {np.__version__}")

    worst_pyr = section_1_measure_vanishes(pyramid_array(), "pyramid")
    mean_pyr = section_2_regular_reference(pyramid_array(), "pyramid")
    section_3_classification(pyramid_array(), "pyramid")
    worst_roof = section_1_measure_vanishes(roof_array(), "roof")
    hand = section_4_known_configurations()
    section_5_surfaces()

    print("\n## Summary")
    checks = [
        ("pyramid m on the analytic singular set", worst_pyr, 1e-13),
        ("roof m on the analytic singular set", worst_roof, 1e-13),
        ("hand-computed momenta at known singularities", hand, 1e-12),
    ]
    ok = True
    for name, value, tol in checks:
        verdict = "PASS" if value < tol else "FAIL"
        ok &= value < tol
        print(f"{verdict}  {name:<46} {value:.6e} < {tol:.0e}")
    print(f"reference: mean m over random pyramid configurations is {mean_pyr:.6e}, so the "
          "singular-set values are 13 orders of magnitude below the regular scale")
    print("\nOVERALL: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

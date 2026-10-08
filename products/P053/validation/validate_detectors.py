"""Known-answer validation of the four detector statistics.

Every expected value below is a hand calculation written out in full, so a
reader can check the arithmetic without running anything. The same cases are
asserted in ``tests/test_detectors.py``; this script exists so the numbers
appear in the validation record with their derivations.
"""

from __future__ import annotations

import numpy as np
from _bootstrap import add_src_to_path

add_src_to_path()

from twininvalidate import (  # noqa: E402
    cusum_statistic,
    ewma_statistic,
    first_alarm,
    glr_statistic,
    variance_cusum_statistic,
)

TOL = 1e-12


def report(label: str, got: np.ndarray, expected: list[float], derivation: str) -> bool:
    err = float(np.max(np.abs(np.asarray(got, dtype=float).ravel() - np.array(expected))))
    ok = err < TOL
    print(f"{label}")
    print(f"  derivation  {derivation}")
    print(f"  expected    {[round(v, 10) for v in expected]}")
    print(f"  computed    {[round(float(v), 10) for v in np.asarray(got).ravel()]}")
    print(f"  max error   {err:.3e}   tolerance {TOL:.0e}   {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    print("=== CUSUM (Page 1954), two-sided tabular form ===")
    results = [
        report(
            "z = [1, 1, 1], reference k = 0.25",
            cusum_statistic(np.array([[1.0, 1.0, 1.0]]), 0.25),
            [0.75, 1.5, 2.25],
            "S+ = max(0, S+ + z - k): 0.75, 1.50, 2.25; S- stays 0 since -1-0.25 < 0",
        ),
        report(
            "z = [-1, -1], reference k = 0.25",
            cusum_statistic(np.array([[-1.0, -1.0]]), 0.25),
            [0.75, 1.5],
            "lower arm S- = max(0, S- - z - k): 0.75, 1.50",
        ),
        report(
            "z = [2, -5, 0], reference k = 0.5",
            cusum_statistic(np.array([[2.0, -5.0, 0.0]]), 0.5),
            [1.5, 4.5, 4.0],
            "S+: 1.5, 0, 0; S-: 0, 4.5, 4.0; statistic is the max of the arms",
        ),
    ]

    print()
    print("=== EWMA (Roberts 1959), standardised by the asymptotic sd ===")
    results += [
        report(
            "z = [2, -1, 0.5], lambda = 1",
            ewma_statistic(np.array([[2.0, -1.0, 0.5]]), 1.0),
            [2.0, 1.0, 0.5],
            "lambda=1 gives A_k = z_k and sigma = sqrt(1/1) = 1, so g = |z|",
        ),
        report(
            "z = [1], lambda = 0.1",
            ewma_statistic(np.array([[1.0]]), 0.1),
            [0.4358898943540674],
            "A_0 = 0.1; sigma = sqrt(0.1/1.9) = 0.2294157339; 0.1/0.2294157339",
        ),
        report(
            "z = [2, 0], lambda = 0.5",
            ewma_statistic(np.array([[2.0, 0.0]]), 0.5),
            [1.7320508075688772, 0.8660254037844386],
            "A = 1.0, 0.5; sigma = sqrt(0.5/1.5) = 0.5773502692; g = sqrt(3), sqrt(3)/2",
        ),
    ]

    print()
    print("=== windowed GLR (Willsky & Jones 1976) ===")
    results += [
        report(
            "z = [2, 0, -3], window = 1",
            glr_statistic(np.array([[2.0, 0.0, -3.0]]), 1),
            [2.0, 0.0, 4.5],
            "window 1 gives g = z^2/2: 4/2, 0/2, 9/2",
        ),
        report(
            "z = [2, 0, 0], window = 3",
            glr_statistic(np.array([[2.0, 0.0, 0.0]]), 3),
            [2.0, 1.0, 2.0 / 3.0],
            "max over n of S_n^2/(2n): 4/2; max(0, 4/4); max(0, 0, 4/6)",
        ),
        report(
            "z = [1, 1, 1, 1], window = 4",
            glr_statistic(np.array([[1.0, 1.0, 1.0, 1.0]]), 4),
            [0.5, 1.0, 1.5, 2.0],
            "S_n = n so g = n^2/(2n) = n/2: 0.5, 1.0, 1.5, 2.0",
        ),
    ]

    print()
    print("=== variance-CUSUM oracle (not one of the three declared baselines) ===")
    results += [
        report(
            "z = [2, 0, 2], reference c = 0.5",
            variance_cusum_statistic(np.array([[2.0, 0.0, 2.0]]), 0.5),
            [2.5, 1.0, 3.5],
            "V = max(0, V + z^2 - 1 - c): 4-1.5 = 2.5; 2.5-1.5 = 1.0; 1.0+4-1.5 = 3.5",
        ),
    ]

    print()
    print("=== first-alarm indexing ===")
    stat = np.array([[0.0, 1.0, 5.0, 9.0], [0.0, 0.0, 0.0, 0.0]])
    idx = first_alarm(stat, 2.0)
    ok = idx.tolist() == [2, -1]
    print("statistic [[0,1,5,9],[0,0,0,0]], threshold 2")
    print("  expected    [2, -1]   (-1 marks a right-censored run)")
    print(f"  computed    {idx.tolist()}")
    print(f"  {'PASS' if ok else 'FAIL'}")
    results.append(ok)
    equal = first_alarm(np.array([[1.0, 2.0]]), 2.0)
    ok2 = equal.tolist() == [-1]
    print("statistic [[1,2]], threshold 2: the test is strict (> not >=)")
    print(f"  expected    [-1]   computed {equal.tolist()}   {'PASS' if ok2 else 'FAIL'}")
    results.append(ok2)

    print()
    print(f"SUMMARY  {sum(results)}/{len(results)} known-answer checks PASS")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

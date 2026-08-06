"""Level-1 validation: hand-checkable 90° principal-axis rotation test vectors.

Reference: right-hand-rule action of 90° rotations about x, y, z (any linear
algebra text, e.g. Markley & Crassidis 2014, Sec. 2.6). Every expected value
below is hand-derivable: an active 90° rotation about +z maps x̂ -> ŷ,
ŷ -> -x̂, ẑ -> ẑ, etc.

Run from products/P007/:  python validation/check_known_rotations.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quatkit import Quaternion, quat_to_dcm  # noqa: E402

S2 = np.sqrt(0.5)

CASES = [
    # (name, quaternion [w,x,y,z], vector in, expected vector out)
    ("z90: x->y", [S2, 0, 0, S2], [1, 0, 0], [0, 1, 0]),
    ("z90: y->-x", [S2, 0, 0, S2], [0, 1, 0], [-1, 0, 0]),
    ("z90: z->z", [S2, 0, 0, S2], [0, 0, 1], [0, 0, 1]),
    ("x90: y->z", [S2, S2, 0, 0], [0, 1, 0], [0, 0, 1]),
    ("x90: z->-y", [S2, S2, 0, 0], [0, 0, 1], [0, -1, 0]),
    ("y90: z->x", [S2, 0, S2, 0], [0, 0, 1], [1, 0, 0]),
    ("y90: x->-z", [S2, 0, S2, 0], [1, 0, 0], [0, 0, -1]),
    ("x180: y->-y", [0, 1, 0, 0], [0, 1, 0], [0, -1, 0]),
    ("z180: x->-x", [0, 0, 0, 1], [1, 0, 0], [-1, 0, 0]),
]

DCM_CASES = [
    ("Rz(90)", [S2, 0, 0, S2], np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)),
    ("Ry(90)", [S2, 0, S2, 0], np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], float)),
    ("Rx(90)", [S2, S2, 0, 0], np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)),
]


def main() -> int:
    print("Known-rotation test vectors (hand-checkable, scalar-first [w,x,y,z])")
    print("=" * 72)
    worst = 0.0
    n_fail = 0
    for name, q, v, expected in CASES:
        out = Quaternion.from_array(np.array(q, float)).rotate(np.array(v, float))
        err = float(np.max(np.abs(out - np.array(expected, float))))
        worst = max(worst, err)
        status = "PASS" if err < 1e-15 else "FAIL"
        n_fail += status == "FAIL"
        q_str = "[" + ", ".join(f"{float(c):.6f}" for c in q) + "]"
        print(f"  {status}  {name:14s} q={q_str} v={v} -> {np.round(out, 12)}  max|err|={err:.2e}")
    print()
    print("DCM known answers (active rotation matrices, hand-written):")
    for name, q, r_exp in DCM_CASES:
        r = quat_to_dcm(np.array(q, float))
        err = float(np.max(np.abs(r - r_exp)))
        worst = max(worst, err)
        status = "PASS" if err < 1e-15 else "FAIL"
        n_fail += status == "FAIL"
        print(f"  {status}  {name}: max|R - R_expected| = {err:.2e}")
    print()
    print(f"Worst-case absolute error over all cases: {worst:.3e} (tolerance 1e-15)")
    print(f"Failures: {n_fail}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Level-1 validation: cross-check against scipy.spatial.transform.Rotation.

For N = 1000 uniformly random unit quaternions (seeded), compare:
  1. quat_to_dcm(q)          vs scipy Rotation.as_matrix()
  2. dcm_to_quat(R)          vs the original quaternion (rotation angle error)
  3. quat_rotate(q, v)       vs scipy Rotation.apply(v)
  4. quat_to_euler_zyx(q)    vs scipy Rotation.as_euler('ZYX') (away from lock)

Note scipy stores quaternions scalar-LAST [x, y, z, w]; quatkit is
scalar-FIRST [w, x, y, z] — converted with np.roll.

Run from products/P007/:  python validation/check_scipy_cross.py
"""

import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quatkit import (  # noqa: E402
    angle_between,
    dcm_to_quat,
    quat_rotate,
    quat_to_dcm,
    quat_to_euler_zyx,
)

N = 1000
SEED = 20260801


def main() -> int:
    rng = np.random.default_rng(SEED)
    q = rng.standard_normal((N, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)  # uniform on S3 (Marsaglia 1972)
    q_scipy = np.roll(q, -1, axis=1)  # [w,x,y,z] -> [x,y,z,w]
    rot = Rotation.from_quat(q_scipy)

    print(f"scipy cross-check, N = {N} random unit quaternions, seed = {SEED}")
    print(f"numpy {np.__version__}, scipy Rotation reference")
    print("=" * 72)

    # 1. DCM forward
    r_ours = quat_to_dcm(q)
    r_ref = rot.as_matrix()
    dev_dcm = float(np.max(np.abs(r_ours - r_ref)))
    print(f"1. quat_to_dcm vs Rotation.as_matrix : max|dR|      = {dev_dcm:.3e}")

    # 2. DCM inverse (compare as rotation angle, double-cover safe)
    q_back = dcm_to_quat(r_ref)
    ang = angle_between(q_back, q)
    dev_inv = float(np.max(ang))
    print(f"2. dcm_to_quat(R_scipy) vs original  : max angle err = {dev_inv:.3e} rad")

    # 3. Vector rotation
    v = rng.standard_normal((N, 3))
    dev_rot = float(np.max(np.abs(quat_rotate(q, v) - rot.apply(v))))
    print(f"3. quat_rotate vs Rotation.apply     : max|dv|      = {dev_rot:.3e}")

    # 4. Euler ZYX (mask samples within 0.01 of the sin(pitch) singular margin)
    sin_pitch = 2.0 * (q[:, 0] * q[:, 2] - q[:, 1] * q[:, 3])
    mask = np.abs(sin_pitch) < 0.99
    eul_ours = quat_to_euler_zyx(q[mask])
    eul_ref = rot[mask].as_euler("ZYX")
    dev_eul = float(np.max(np.abs(eul_ours - eul_ref)))
    print(
        f"4. quat_to_euler_zyx vs as_euler('ZYX'): max|dangle|  = {dev_eul:.3e} rad "
        f"({int(mask.sum())}/{N} samples outside lock margin)"
    )

    tol = 1e-12
    ok = max(dev_dcm, dev_inv, dev_rot, dev_eul) < tol
    print()
    print(f"Tolerance: {tol:.0e} on every check -> {'ALL PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Validation 1: frames, projection and the attitude solve.

Nothing here is about star identification; it is the layer everything else
stands on, and a silent transpose in it would make every later number wrong in
a way no identification test would catch (the identification would still be
"correct" and the attitude would be the inverse rotation).

References
----------
* Wahba, G. (1965). A least squares estimate of satellite attitude.
  *SIAM Review* 7(3), 409.
* Davenport, P. B. (1968). A vector approach to the algebra of rotations with
  applications. NASA TN D-4696.
* Shuster, M. D. & Oh, S. D. (1981). Three-axis attitude determination from
  vector observations. *Journal of Guidance and Control* 4(1), 70-77.

Run: ``python validation/validate_geometry.py``
"""

from __future__ import annotations

import numpy as np

from _common import SEED, banner, finish, report, unit_sphere, verdict  # noqa: E402

from skymatch.camera import CameraModel  # noqa: E402
from skymatch.geometry import (  # noqa: E402
    ARCSEC,
    angle_between_dcm,
    angular_separation,
    davenport_attitude,
    dcm_from_quat,
    quat_from_dcm,
    radec_from_unit_vectors,
    random_rotation,
    unit_vectors_from_radec,
)


def main() -> int:
    passed: list[bool] = []
    rng = np.random.default_rng(SEED)

    banner("VALIDATION 1a: quaternion and DCM conventions")
    print("    dcm_from_quat against scipy Rotation.from_quat([x, y, z, w]).as_matrix()")
    try:
        from scipy.spatial.transform import Rotation

        worst = 0.0
        for _ in range(500):
            q = rng.normal(size=4)
            q /= np.linalg.norm(q)
            ref = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
            worst = max(worst, float(np.max(np.abs(dcm_from_quat(q) - ref))))
        passed.append(verdict("max |dcm_from_quat - scipy as_matrix|", worst, 1e-14))
    except ImportError:  # pragma: no cover - scipy is a hard dependency here
        print("    scipy unavailable, cross-check skipped")

    worst = 0.0
    for _ in range(500):
        a = random_rotation(rng)
        worst = max(worst, float(np.max(np.abs(dcm_from_quat(quat_from_dcm(a)) - a))))
    passed.append(verdict("max |dcm_from_quat(quat_from_dcm(A)) - A|", worst, 1e-13))

    worst = 0.0
    for _ in range(300):
        a = random_rotation(rng)
        worst = max(worst, float(np.max(np.abs(a @ a.T - np.eye(3)))))
    passed.append(verdict("max |A A^T - I| over random rotations", worst, 1e-14))

    print()
    banner("VALIDATION 1b: ra/dec round trip (Eq. G1)")
    ra = rng.random(2000) * 2.0 * np.pi
    dec = np.arcsin(2.0 * rng.random(2000) - 1.0)
    v = unit_vectors_from_radec(ra, dec)
    ra2, dec2 = radec_from_unit_vectors(v)
    ddec = float(np.max(np.abs(dec - dec2)))
    passed.append(verdict("max |dec - dec_roundtrip| [rad]", ddec, 1e-14))
    dra = np.abs(np.mod(ra - ra2 + np.pi, 2.0 * np.pi) - np.pi)
    passed.append(verdict("max |ra - ra_roundtrip| [rad]", float(np.max(dra)), 1e-13))
    passed.append(
        verdict("max |‖v‖ - 1|", float(np.max(np.abs(np.linalg.norm(v, axis=1) - 1.0))), 1e-15)
    )

    print()
    banner("VALIDATION 1c: angular separation (Eq. G2) against arccos")
    print("    both forms are exact analytically; atan2 keeps its digits where arccos does not")
    print(f"{'true angle [rad]':>18} {'atan2 err':>12} {'arccos err':>12}   ratio")
    worst_abs = 0.0
    for target in (1e-8, 1e-6, 1e-4, 1e-2, 1.0, 3.0):
        errs_a, errs_c = [], []
        for _ in range(200):
            axis = unit_sphere(rng, 1)[0]
            base = unit_sphere(rng, 1)[0]
            base = base - (base @ axis) * axis
            base /= np.linalg.norm(base)
            perp = np.cross(axis, base)
            second = np.cos(target) * axis + np.sin(target) * perp
            got = float(angular_separation(axis, second)[0])
            errs_a.append(abs(got - target))
            errs_c.append(abs(float(np.arccos(np.clip(axis @ second, -1.0, 1.0))) - target))
        ea, ec = max(errs_a), max(errs_c)
        worst_abs = max(worst_abs, ea)
        ratio = ec / ea if ea > 0 else float("inf")
        print(f"{target:18.1e} {ea:12.3e} {ec:12.3e}   {ratio:8.1f}x")
    print("    The gate is on the ABSOLUTE error. The relative error at 1e-8 rad is 9e-9,")
    print("    which is the precision of the test vectors themselves: two unit vectors")
    print("    1e-8 rad apart differ only in their 8th significant digit, so no separation")
    print("    function can beat that relatively. Eq. G2 is absolutely accurate to ~1e-16 at")
    print("    every angle, and that is the property the matcher depends on.")
    passed.append(verdict("worst absolute error of Eq. G2 over 6 decades [rad]", worst_abs, 1e-15))

    print()
    banner("VALIDATION 1d: the attitude solve (Eq. G3)")
    print("    noise-free observations: the q-method must reproduce the attitude exactly")
    worst = 0.0
    for n in (2, 3, 4, 6, 10):
        for _ in range(100):
            a_true = random_rotation(rng)
            ref = unit_sphere(rng, n)
            worst = max(worst, angle_between_dcm(davenport_attitude(ref @ a_true.T, ref), a_true))
    passed.append(verdict("max attitude error, noise free, n = 2..10 [rad]", worst, 1e-12))

    print()
    print("    the transpose regression: the solve must return A, not A^T")
    a_true = random_rotation(rng)
    ref = unit_sphere(rng, 5)
    est = davenport_attitude(ref @ a_true.T, ref)
    to_a = angle_between_dcm(est, a_true)
    to_at = angle_between_dcm(est, a_true.T)
    report("angle to A [rad]", to_a)
    report("angle to A^T [rad]", to_at)
    passed.append(verdict("angle to A [rad]", to_a, 1e-12))
    passed.append(verdict("angle to A^T [rad] (must be far)", to_at, 1e-3, mode=">="))

    print()
    print("    attitude error against measurement noise, 4 stars, 400 trials per level")
    print(f"{'sigma [arcsec]':>16} {'RMS error [arcsec]':>20} {'error / sigma':>15}")
    ratios = []
    for sigma_as in (1.0, 5.0, 20.0, 60.0):
        sigma = sigma_as * ARCSEC
        errs = []
        for _ in range(400):
            a_true = random_rotation(rng)
            ref = unit_sphere(rng, 4)
            body = ref @ a_true.T
            body = body + sigma * rng.normal(size=body.shape)
            body /= np.linalg.norm(body, axis=1)[:, None]
            errs.append(angle_between_dcm(davenport_attitude(body, ref), a_true) / ARCSEC)
        rms = float(np.sqrt(np.mean(np.square(errs))))
        ratios.append(rms / sigma_as)
        print(f"{sigma_as:16.1f} {rms:20.3f} {rms / sigma_as:15.3f}")
    spread = float(np.max(ratios) - np.min(ratios))
    print("    the ratio is constant because the estimator is linear in the noise to first order")
    passed.append(verdict("spread of error/sigma over 60x in sigma", spread, 0.05))

    print()
    print("    collinear observations must raise, not invent the missing rotation")
    raised = 0
    for _ in range(50):
        d = unit_sphere(rng, 1)[0]
        ref = np.vstack([d, d + 1e-12 * unit_sphere(rng, 1)[0]])
        ref /= np.linalg.norm(ref, axis=1)[:, None]
        a_true = random_rotation(rng)
        try:
            davenport_attitude(ref @ a_true.T, ref)
        except ValueError:
            raised += 1
    passed.append(verdict("collinear cases raising ValueError", raised / 50.0, 1.0, mode=">="))

    print()
    banner("VALIDATION 1e: the camera projection (Eq. K1-K2)")
    cam = CameraModel()
    report("focal length [pixels]", cam.focal_length_px)
    report("plate scale [arcsec/pixel]", cam.arcsec_per_pixel)
    report("field solid angle [sq.deg]", cam.solid_angle_sqdeg)
    half = cam.pixels / 2.0
    px = rng.uniform(-half, half, size=(5000, 2))
    back = cam.project(cam.unproject(px))
    round_trip = float(np.max(np.abs(back - px)))
    passed.append(verdict("max |project(unproject(p)) - p| [pixels]", round_trip, 1e-9))

    edge = np.array([[half, half], [-half, half], [0.0, 0.0]])
    ang = angular_separation(cam.unproject(edge), np.array([0.0, 0.0, 1.0]))
    print(f"    corner angle from boresight {np.degrees(ang[0]):.6f} deg "
          f"(half-diagonal {np.degrees(cam.half_diagonal_rad):.6f} deg)")
    corner_err = abs(float(ang[0]) - cam.half_diagonal_rad)
    passed.append(verdict("corner angle vs half_diagonal_rad [rad]", corner_err, 1e-14))
    passed.append(verdict("boresight pixel maps to zero angle [rad]", float(ang[2]), 1e-15))

    print()
    print("    the on-axis plate scale understates the off-axis scale by 1/cos^2(r)")
    for frac, label in ((0.0, "centre"), (0.5, "half width"), (1.0, "corner")):
        p0 = np.array([[frac * half, frac * half]])
        p1 = p0 + np.array([[1.0, 0.0]])
        step = float(angular_separation(cam.unproject(p0), cam.unproject(p1))[0]) / ARCSEC
        print(f"      {label:<12s} local scale {step:8.4f} arcsec/pixel "
              f"({step / cam.arcsec_per_pixel:6.4f} x nominal)")

    return finish(passed)


if __name__ == "__main__":
    raise SystemExit(main())

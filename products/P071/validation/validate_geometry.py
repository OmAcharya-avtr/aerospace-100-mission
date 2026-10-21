"""Close-approach geometry: closed forms, invariances, and the shipped cases.

Checks:
1. time of closest approach against a hand-calculated head-on geometry;
2. the miss vector perpendicular to the relative velocity, as a residual;
3. the projected covariance trace preserved under the plane projection;
4. Pc invariance under a rigid rotation of the entire problem, measured over
   random rotations;
5. Pc invariance under a common rescaling of every length;
6. the Mahalanobis distance from the principal frame against the explicit
   quadratic form with the full covariance.
"""

from __future__ import annotations

import math

import numpy as np

from _bootstrap import Tee, bootstrap, environment_lines

bootstrap()

from conjprob.analytic import pc_foster  # noqa: E402
from conjprob.cases import load_case  # noqa: E402
from conjprob.geometry import (  # noqa: E402
    ObjectState,
    encounter_geometry,
    encounter_plane_basis,
    mahalanobis_distance,
    miss_vector,
    project_covariance,
    time_of_closest_approach,
)

CASES = (
    "examples/cases/leo_crossing.yaml",
    "examples/cases/large_body_tight_covariance.yaml",
    "examples/cases/far_miss_tight_covariance.yaml",
)


def rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Right-handed rotation about a unit axis, Rodrigues' formula."""
    axis = axis / np.linalg.norm(axis)
    cross = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return (
        np.eye(3)
        + math.sin(angle) * cross
        + (1.0 - math.cos(angle)) * (cross @ cross)
    )


def main() -> int:
    out = Tee("validate_geometry_output.txt")
    out.rule("conjprob validation: close-approach geometry")
    for line in environment_lines():
        out(line)

    failures = 0

    out.rule("1. time of closest approach against a hand calculation")
    out("geometry: r = (100, 0, 0) m, v = (-10, 0, 0) m/s; closing head on at 10 m/s")
    out("expected: tca = +10 s exactly in exact arithmetic, miss distance 0 m")
    tca = time_of_closest_approach([100.0, 0.0, 0.0], [-10.0, 0.0, 0.0])
    miss = float(np.linalg.norm(miss_vector([100.0, 0.0, 0.0], [-10.0, 0.0, 0.0])))
    out(f"measured: tca = {tca:.15f} s, miss = {miss:.6e} m")
    out("tolerance: |tca - 10| <= 1e-12 s, miss <= 1e-9 m")
    ok = abs(tca - 10.0) <= 1e-12 and miss <= 1e-9
    out(f"result: {'PASS' if ok else 'FAIL'}")
    failures += 0 if ok else 1

    out.rule("2. miss vector perpendicular to the relative velocity")
    out("residual: |m . v| / (|m| |v|), over 200 random geometries")
    rng = np.random.default_rng(20261021)
    worst = 0.0
    for _ in range(200):
        r = rng.normal(scale=1.0e4, size=3)
        v = rng.normal(scale=5.0e3, size=3)
        m = miss_vector(r, v)
        denominator = float(np.linalg.norm(m) * np.linalg.norm(v))
        worst = max(worst, abs(float(m @ v)) / max(denominator, 1e-30))
    out(f"measured: worst normalised residual {worst:.6e}")
    out("tolerance: <= 1e-12")
    ok = worst <= 1e-12
    out(f"result: {'PASS' if ok else 'FAIL'}")
    failures += 0 if ok else 1

    out.rule("3. projection of an isotropic covariance preserves the in-plane trace")
    out("for C = sigma^2 I the projected 2x2 trace must be 2 sigma^2")
    worst = 0.0
    for _ in range(200):
        sigma = float(rng.uniform(1.0, 1.0e3))
        v = rng.normal(scale=5.0e3, size=3)
        projected = project_covariance(np.eye(3) * sigma**2, encounter_plane_basis(v))
        worst = max(worst, abs(float(np.trace(projected)) / (2.0 * sigma**2) - 1.0))
    out(f"measured: worst relative trace error {worst:.6e}")
    out("tolerance: <= 1e-12")
    ok = worst <= 1e-12
    out(f"result: {'PASS' if ok else 'FAIL'}")
    failures += 0 if ok else 1

    out.rule("4. Pc invariance under a rigid rotation of the whole problem")
    out("both states, both covariances and the hard body rotated together")
    out(f"{'case':<34s} {'rotations':>9s} {'worst rel. change':>18s}")
    for path in CASES:
        case = load_case(path)
        reference = None
        worst = 0.0
        for index in range(32):
            axis = rng.normal(size=3)
            angle = float(rng.uniform(-math.pi, math.pi))
            matrix = rotation_matrix(axis, angle) if index else np.eye(3)
            primary = ObjectState(
                position_m=matrix @ case.primary.position_m,
                velocity_mps=matrix @ case.primary.velocity_mps,
                covariance_m2=matrix @ case.primary.covariance_m2 @ matrix.T,
            )
            secondary = ObjectState(
                position_m=matrix @ case.secondary.position_m,
                velocity_mps=matrix @ case.secondary.velocity_mps,
                covariance_m2=matrix @ case.secondary.covariance_m2 @ matrix.T,
            )
            frame = encounter_geometry(primary, secondary).frame
            value = pc_foster(
                frame.sigma_x_m, frame.sigma_y_m, frame.x_bar_m, frame.y_bar_m, case.radius_m
            )
            if reference is None:
                reference = value
            else:
                worst = max(worst, abs(value / reference - 1.0))
        out(f"{case.name:<34s} {32:>9d} {worst:>18.6e}")
        ok = worst <= 1e-6
        failures += 0 if ok else 1
    out("tolerance: <= 1e-6 relative; the rotation changes the in-plane basis and the")
    out("           eigendecomposition, so this is a rounding bound and not an exact one")

    out.rule("5. Pc invariance under a common rescaling of every length")
    out(f"{'case':<34s} {'factor':>10s} {'rel. change':>14s}")
    for path in CASES:
        case = load_case(path)
        frame = encounter_geometry(case.primary, case.secondary).frame
        base = pc_foster(
            frame.sigma_x_m, frame.sigma_y_m, frame.x_bar_m, frame.y_bar_m, case.radius_m
        )
        for factor in (1e-4, 1e-2, 1e2, 1e4):
            scaled = pc_foster(
                frame.sigma_x_m * factor,
                frame.sigma_y_m * factor,
                frame.x_bar_m * factor,
                frame.y_bar_m * factor,
                case.radius_m * factor,
            )
            change = abs(scaled / base - 1.0) if base > 0.0 else 0.0
            out(f"{case.name:<34s} {factor:>10.0e} {change:>14.6e}")
            ok = change <= 1e-9
            failures += 0 if ok else 1
    out("tolerance: <= 1e-9 relative")

    out.rule("6. Mahalanobis distance against the explicit quadratic form")
    out("principal-frame form sqrt((x/sx)^2 + (y/sy)^2) against sqrt(m^T C^-1 m)")
    worst = 0.0
    for _ in range(200):
        a = rng.uniform(1.0, 1.0e3)
        b = rng.uniform(1.0, 1.0e3)
        angle = float(rng.uniform(0.0, math.pi))
        rot = np.array(
            [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
        )
        covariance = rot @ np.diag([a**2, b**2]) @ rot.T
        m = rng.normal(scale=1.0e3, size=2)
        explicit = float(np.sqrt(m @ np.linalg.inv(covariance) @ m))
        worst = max(worst, abs(mahalanobis_distance(covariance, m) / explicit - 1.0))
    out(f"measured: worst relative difference {worst:.6e}")
    out("tolerance: <= 1e-9")
    ok = worst <= 1e-9
    out(f"result: {'PASS' if ok else 'FAIL'}")
    failures += 0 if ok else 1

    out.rule("7. shipped case geometry, for the record")
    out(
        f"{'case':<34s} {'tca [s]':>12s} {'miss [m]':>12s} {'|v| [m/s]':>12s} "
        f"{'sigma_x [m]':>12s} {'sigma_y [m]':>12s} {'M':>9s}"
    )
    for path in CASES:
        case = load_case(path)
        geometry = encounter_geometry(case.primary, case.secondary)
        frame = geometry.frame
        out(
            f"{case.name:<34s} {geometry.tca_s:>12.6f} {geometry.miss_distance_m:>12.4f} "
            f"{geometry.relative_speed_mps:>12.3f} {frame.sigma_x_m:>12.4f} "
            f"{frame.sigma_y_m:>12.4f} {geometry.mahalanobis:>9.5f}"
        )
    out("")
    out("principal frame convention: sigma_x is the larger principal standard deviation")

    out.rule("summary")
    out(f"checks failed: {failures}")
    out(f"written to {out.save()}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

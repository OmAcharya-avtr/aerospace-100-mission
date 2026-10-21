"""Where the 2-D rectilinear formulation fails, measured on both of its assumptions.

Left panel: restricting the encounter to a finite window loses probability. The
2-D formulation integrates over all time, so it overstates the probability of a
conjunction inside a short screening window.

Right panel: giving the relative motion a constant acceleration breaks the
straight-line assumption. The error of the 2-D value against the curved 3-D
Monte Carlo is plotted against the dimensionless curvature
``kappa = |a| T^2 / (2 sigma_eff)``, and the measured crossing of the 1 and 10
per cent levels is printed. A reference value of kappa for a real low Earth
conjunction, from the gravity-gradient difference, is marked: the assumption
is excellent there, and the point of the sweep is to say how slow an encounter
would have to be before it is not.
"""

from __future__ import annotations

import warnings

import numpy as np

from _common import FOSTER, MONTE_CARLO, REFERENCE, bootstrap, save, style

bootstrap()

from conjprob.cases import load_case  # noqa: E402
from conjprob.dilution import DilutionWarning  # noqa: E402
from conjprob.nonlinear import (  # noqa: E402
    curvature_sweep,
    first_crossing,
    gravity_gradient_acceleration_scale,
    window_sweep,
)
from conjprob.report import assess_conjunction  # noqa: E402

CASE = "examples/cases/large_body_tight_covariance.yaml"
WINDOW_FACTORS = np.array([0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0])
WINDOW_WIDTH_FACTOR = 8.0
GEOCENTRIC_RADIUS_M = 7078137.0
SEED = 4242


def main() -> None:
    style()
    import matplotlib.pyplot as plt

    case = load_case(CASE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DilutionWarning)
        report = assess_conjunction(case.primary, case.secondary, case.radius_m)
    geometry = report.geometry
    sigma_eff = geometry.sigma_effective_m
    crossing = geometry.crossing_time_s
    reference = report.analytic["foster"]

    unbounded, window_points = window_sweep(
        geometry.combined_covariance_m2,
        geometry.miss_vector_m,
        geometry.relative_velocity_mps,
        case.radius_m,
        WINDOW_FACTORS * crossing,
        sigma_effective_m=sigma_eff,
        n_samples=200_000,
        seed=SEED,
    )

    window_s = WINDOW_WIDTH_FACTOR * crossing
    accelerations = 2.0 * sigma_eff * np.logspace(-3.0, 4.0, 15) / window_s**2
    curvature_points = curvature_sweep(
        geometry.combined_covariance_m2,
        geometry.miss_vector_m,
        geometry.relative_velocity_mps,
        case.radius_m,
        accelerations,
        window_s=window_s,
        pc_linear_2d=reference,
        sigma_effective_m=sigma_eff,
        n_samples=60_000,
        seed=SEED,
        time_nodes=129,
        refinements=4,
    )

    figure, axes = plt.subplots(1, 2, figsize=(11.4, 4.3))
    ratios = np.array([point.ratio_to_unbounded for point in window_points])
    axes[0].semilogx(
        WINDOW_FACTORS, ratios, marker="o", color=MONTE_CARLO, label="windowed / unbounded"
    )
    axes[0].axhline(1.0, color=REFERENCE, linewidth=0.9, linestyle="--")
    axes[0].set_xlabel("window half-width / encounter crossing time")
    axes[0].set_ylabel("Pc inside the window / Pc over all time")
    axes[0].set_title(
        f"{case.name}: finite-window shortfall\n"
        f"crossing time {crossing * 1e3:.4f} ms, "
        f"unbounded Pc {unbounded.pc:.5f}"
    )
    axes[0].legend(loc="lower right")

    kappa = np.array([point.curvature_parameter for point in curvature_points])
    errors = np.array([abs(point.relative_error) for point in curvature_points])
    axes[1].loglog(kappa, errors, marker="o", color=FOSTER, label="|2-D error| vs curved 3-D")
    for level, dash in ((0.01, "--"), (0.10, ":")):
        axes[1].axhline(level, color=REFERENCE, linewidth=0.9, linestyle=dash)
    gradient = gravity_gradient_acceleration_scale(
        GEOCENTRIC_RADIUS_M, geometry.miss_distance_m
    )
    kappa_real = gradient * window_s**2 / (2.0 * sigma_eff)
    axes[1].axvline(
        kappa_real,
        color=MONTE_CARLO,
        linewidth=1.1,
        label=f"gravity gradient here, kappa = {kappa_real:.2e}",
    )
    axes[1].set_xlabel("curvature parameter kappa = |a| T^2 / (2 sigma_eff)")
    axes[1].set_ylabel("|relative error| of the rectilinear 2-D Pc")
    axes[1].set_title("curvature breakdown; dashed 1 per cent, dotted 10 per cent")
    axes[1].legend(loc="upper left")
    print(f"wrote {save(figure, 'nonlinear_breakdown.png')}")

    print(f"encounter crossing time            : {crossing:.6e} s")
    print(f"curvature window half-width        : {window_s:.6e} s")
    print(f"gravity-gradient acceleration      : {gradient:.6e} m/s^2")
    print(f"curvature parameter at that gradient: {kappa_real:.6e}")
    for point in window_points:
        print(
            f"window {point.window_over_crossing_time:7.3f} t_cross  "
            f"ratio {point.ratio_to_unbounded:.6f}  "
            f"CI [{point.ratio_ci_low:.6f}, {point.ratio_ci_high:.6f}]"
        )
    for point in curvature_points:
        print(
            f"kappa {point.curvature_parameter:10.4e}  "
            f"mc {point.monte_carlo.pc:.6f}  "
            f"rel err {point.relative_error:+.6f}  "
            f"CI [{point.relative_error_ci_low:+.6f}, {point.relative_error_ci_high:+.6f}]"
        )
    crossings = {}
    for level in (0.01, 0.10):
        crossings[level] = first_crossing(kappa, errors, level)
        print(
            f"measured kappa at which the 2-D error reaches {level * 100:.0f} per cent: "
            f"{crossings[level]:.6e}"
        )

    # kappa = |a| T^2 / (2 sigma_eff) with T = w sigma_eff / v gives
    # kappa = |a| w^2 sigma_eff / (2 v^2), so the relative speed at which the
    # measured kappa is reached is v = w sqrt(|a| sigma_eff / (2 kappa)).
    for label, sigma, separation in (
        ("this case", sigma_eff, geometry.miss_distance_m),
        ("a declared 200 m / 300 m low Earth encounter", 200.0, 300.0),
    ):
        acceleration = gravity_gradient_acceleration_scale(GEOCENTRIC_RADIUS_M, separation)
        for level, value in crossings.items():
            if not np.isfinite(value):
                continue
            speed = WINDOW_WIDTH_FACTOR * np.sqrt(
                acceleration * sigma / (2.0 * value)
            )
            print(
                f"relative speed at which curvature costs {level * 100:.0f} per cent for "
                f"{label}: {speed:.6f} m/s"
            )


if __name__ == "__main__":
    main()

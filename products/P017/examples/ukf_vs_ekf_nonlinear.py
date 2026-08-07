"""Example 2 — UKF versus EKF where the EKF linearisation degrades.

Problem: long-range radar tracking of a 2-D constant-velocity target. The
state is Cartesian, ``[x, vx, y, vy]`` in metres and metres per second; the
sensor at the origin reports polar measurements

    h(x) = [ sqrt(x^2 + y^2),  atan2(y, x) ]        [m, rad]

with independent range and bearing noise. This is the standard
polar-measurement / Cartesian-state mismatch: the EKF replaces the arc
swept by the bearing uncertainty with its tangent at the current estimate,
and the neglected curvature grows with the *cross-range* uncertainty
``r * sigma_theta``. At 50 km with a 5 deg bearing sigma that arc spans
about 4.4 km, so the linearisation error is no longer small compared with
the state uncertainty, the EKF becomes optimistic (mean NIS above its
chi-squared expectation) and its position error grows. The unscented
transform samples the arc instead of tangent-approximating it.

Background: Lerro, D. and Bar-Shalom, Y., "Tracking with debiased
consistent converted measurements versus EKF", IEEE Transactions on
Aerospace and Electronic Systems, Vol. 29, No. 3, 1993 (the polar-to-
Cartesian bias); Julier, S. J. and Uhlmann, J. K., "Unscented filtering and
nonlinear estimation", Proceedings of the IEEE, Vol. 92, No. 3, 2004.

Run from the product root::

    PYTHONPATH=src python examples/ukf_vs_ekf_nonlinear.py

Writes ``screenshots/ukf_vs_ekf_nonlinear.png``. Runtime: about 25 s on
the 2-core build environment (well inside the 3-minute budget).
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")  # non-interactive backend; never calls show()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from estimkit import (  # noqa: E402
    ExtendedKalmanFilter,
    UnscentedKalmanFilter,
    constant_velocity_cwna,
)

DT = 1.0  # s
Q_PSD = 0.5  # m^2/s^3 per axis
SIGMA_R = 50.0  # m
STEPS = 60
N_SEEDS = 50
BEARING_SIGMAS_DEG = (1.0, 2.0, 3.0, 5.0, 7.0, 10.0)
X0_TRUE = np.array([50_000.0, -300.0, 20_000.0, 0.0])  # [x, vx, y, vy]
SHOWCASE_SIGMA_DEG = 10.0
SHOWCASE_SEED = 3

OUT = pathlib.Path(__file__).resolve().parent.parent / "screenshots"

_FB, _QB = constant_velocity_cwna(DT, Q_PSD)
F = np.zeros((4, 4))
Q = np.zeros((4, 4))
F[:2, :2] = _FB
F[2:, 2:] = _FB
Q[:2, :2] = _QB
Q[2:, 2:] = _QB


def f_state(x: np.ndarray) -> np.ndarray:
    """Constant-velocity transition."""
    return F @ x


def f_jac(x: np.ndarray) -> np.ndarray:
    """Transition Jacobian (exact: the transition is linear)."""
    return F


def h_meas(x: np.ndarray) -> np.ndarray:
    """Polar measurement [range m, bearing rad]."""
    return np.array([np.hypot(x[0], x[2]), np.arctan2(x[2], x[0])])


def h_jac(x: np.ndarray) -> np.ndarray:
    """Analytic measurement Jacobian.

    d(range)/d[x, y]   = [x/r, y/r]
    d(bearing)/d[x, y] = [-y/r^2, x/r^2]
    """
    px, py = x[0], x[2]
    r2 = px * px + py * py
    r = np.sqrt(r2)
    return np.array([[px / r, 0.0, py / r, 0.0], [-py / r2, 0.0, px / r2, 0.0]])


def make_scenario(sigma_theta: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Seeded truth trajectory and polar measurements."""
    rng = np.random.default_rng(seed)
    chol = np.linalg.cholesky(Q + 1e-12 * np.eye(4))
    x = X0_TRUE.copy()
    truth = np.empty((STEPS, 4))
    zs = np.empty((STEPS, 2))
    for k in range(STEPS):
        x = F @ x + chol @ rng.standard_normal(4)
        truth[k] = x
        zs[k] = h_meas(x) + np.array(
            [SIGMA_R * rng.standard_normal(), sigma_theta * rng.standard_normal()]
        )
    return truth, zs


def initialise(zs: np.ndarray, sigma_theta: float) -> tuple[np.ndarray, np.ndarray]:
    """Single-point initialisation from the first polar measurement."""
    r0, th0 = zs[0]
    x0 = np.array([r0 * np.cos(th0), 0.0, r0 * np.sin(th0), 0.0])
    pos_var = SIGMA_R**2 + (r0 * sigma_theta) ** 2
    p0 = np.diag([pos_var, 1.0e4, pos_var, 1.0e4])
    return x0, p0


def run_pair(sigma_theta: float, seed: int) -> tuple[float, float, float, float]:
    """Return (EKF RMSE, UKF RMSE, EKF mean NIS, UKF mean NIS) for one seed."""
    truth, zs = make_scenario(sigma_theta, seed)
    r = np.diag([SIGMA_R**2, sigma_theta**2])
    x0, p0 = initialise(zs, sigma_theta)

    ekf = ExtendedKalmanFilter(
        f=f_state, h=h_meas, process_noise=Q, measurement_noise=r,
        f_jac=f_jac, h_jac=h_jac,
    )
    ukf = UnscentedKalmanFilter(
        f=f_state, h=h_meas, process_noise=Q, measurement_noise=r,
        alpha=1.0, beta=2.0, kappa=0.0,
    )
    re = ekf.filter(x0, p0, zs)
    ru = ukf.filter(x0, p0, zs)

    def rmse(est: np.ndarray) -> float:
        err = est[:, [0, 2]] - truth[:, [0, 2]]
        return float(np.sqrt(np.mean(np.sum(err**2, axis=1))))

    return rmse(re.x_post), rmse(ru.x_post), float(np.mean(re.nis)), float(np.mean(ru.nis))


def showcase() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One run at the harshest bearing noise, for the trajectory panel."""
    sigma = np.deg2rad(SHOWCASE_SIGMA_DEG)
    truth, zs = make_scenario(sigma, SHOWCASE_SEED)
    r = np.diag([SIGMA_R**2, sigma**2])
    x0, p0 = initialise(zs, sigma)
    ekf = ExtendedKalmanFilter(
        f=f_state, h=h_meas, process_noise=Q, measurement_noise=r,
        f_jac=f_jac, h_jac=h_jac,
    )
    ukf = UnscentedKalmanFilter(
        f=f_state, h=h_meas, process_noise=Q, measurement_noise=r,
        alpha=1.0, beta=2.0, kappa=0.0,
    )
    meas_xy = np.column_stack(
        [zs[:, 0] * np.cos(zs[:, 1]), zs[:, 0] * np.sin(zs[:, 1])]
    )
    return truth, meas_xy, ekf.filter(x0, p0, zs).x_post, ukf.filter(x0, p0, zs).x_post


def main() -> None:
    ekf_rmse = np.empty(len(BEARING_SIGMAS_DEG))
    ukf_rmse = np.empty(len(BEARING_SIGMAS_DEG))
    ekf_nis = np.empty(len(BEARING_SIGMAS_DEG))
    ukf_nis = np.empty(len(BEARING_SIGMAS_DEG))
    ukf_wins = np.empty(len(BEARING_SIGMAS_DEG))

    for i, deg in enumerate(BEARING_SIGMAS_DEG):
        sigma = np.deg2rad(deg)
        e = np.empty(N_SEEDS)
        u = np.empty(N_SEEDS)
        ne = np.empty(N_SEEDS)
        nu = np.empty(N_SEEDS)
        for s in range(N_SEEDS):
            e[s], u[s], ne[s], nu[s] = run_pair(sigma, s)
        ekf_rmse[i], ukf_rmse[i] = e.mean(), u.mean()
        ekf_nis[i], ukf_nis[i] = ne.mean(), nu.mean()
        ukf_wins[i] = float(np.mean(u < e))

    truth, meas_xy, ekf_track, ukf_track = showcase()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    ax.plot(meas_xy[:, 0] / 1e3, meas_xy[:, 1] / 1e3, ".", ms=4, color="0.7",
            label="measurements (polar -> Cartesian)")
    ax.plot(truth[:, 0] / 1e3, truth[:, 2] / 1e3, "k-", lw=1.6, label="truth")
    ax.plot(ekf_track[:, 0] / 1e3, ekf_track[:, 2] / 1e3, "-", lw=1.2,
            color="tab:orange", label="EKF")
    ax.plot(ukf_track[:, 0] / 1e3, ukf_track[:, 2] / 1e3, "-", lw=1.2,
            color="tab:green", label="UKF")
    ax.plot([0.0], [0.0], "r*", ms=12, label="radar")
    ax.set_xlabel("x [km]")
    ax.set_ylabel("y [km]")
    ax.set_title(f"Single run, bearing sigma = {SHOWCASE_SIGMA_DEG:.0f} deg "
                 f"(seed {SHOWCASE_SEED})")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")

    ax = axes[1]
    ax.plot(BEARING_SIGMAS_DEG, ekf_rmse, "o-", color="tab:orange", label="EKF")
    ax.plot(BEARING_SIGMAS_DEG, ukf_rmse, "s-", color="tab:green", label="UKF")
    for x, y, w in zip(BEARING_SIGMAS_DEG, ukf_rmse, ukf_wins, strict=True):
        ax.annotate(f"{w * 100:.0f}%", (x, y), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=7, color="tab:green")
    ax.set_xlabel("bearing noise sigma [deg]")
    ax.set_ylabel("mean position RMSE [m]")
    ax.set_title(f"Position RMSE, {N_SEEDS} seeds\n(labels: fraction of seeds where UKF wins)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(BEARING_SIGMAS_DEG, ekf_nis, "o-", color="tab:orange", label="EKF")
    ax.plot(BEARING_SIGMAS_DEG, ukf_nis, "s-", color="tab:green", label="UKF")
    ax.axhline(2.0, color="k", ls="--", lw=1.0, label="consistent value (m = 2 dof)")
    ax.set_xlabel("bearing noise sigma [deg]")
    ax.set_ylabel("mean NIS")
    ax.set_title("Filter consistency\n(NIS above 2 = over-confident filter)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle(
        "Long-range polar radar tracking: the EKF tangent approximation of the "
        "bearing arc degrades with cross-range uncertainty",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    OUT.mkdir(exist_ok=True)
    path = OUT / "ukf_vs_ekf_nonlinear.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)

    print(f"target start [x, y] = [{X0_TRUE[0]:.0f}, {X0_TRUE[2]:.0f}] m, "
          f"range sigma = {SIGMA_R:.0f} m, {STEPS} steps, {N_SEEDS} seeds per point")
    print(f"{'sigma_th[deg]':>13} {'EKF RMSE[m]':>12} {'UKF RMSE[m]':>12} "
          f"{'UKF win frac':>13} {'EKF NIS':>9} {'UKF NIS':>9}")
    for i, deg in enumerate(BEARING_SIGMAS_DEG):
        print(f"{deg:13.1f} {ekf_rmse[i]:12.2f} {ukf_rmse[i]:12.2f} "
              f"{ukf_wins[i]:13.2f} {ekf_nis[i]:9.3f} {ukf_nis[i]:9.3f}")
    print(f"figure: {path}")


if __name__ == "__main__":
    main()

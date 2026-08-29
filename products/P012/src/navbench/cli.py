"""Command-line interface: ``python -m navbench <subcommand>``.

Subcommands
-----------
``riccati``      steady-state Riccati solution for a named linear model
``bench``        KF / EKF / UKF on the radar tracking scenario, scored
``attitude``     MEKF on a seeded rigid-body attitude scenario, scored
``consistency``  Monte Carlo NEES/NIS with chi-squared bounds
``adaptive``     train and benchmark the adaptive-Q tuners

Every subcommand accepts ``--json`` for machine-readable output.  Invalid
input produces a one-line ``error: ...`` on stderr and exit code 2 — never a
traceback; a traceback would indicate a defect in the package.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import numpy as np

from . import __version__
from .adaptive import LearnedAdaptiveQ, generate_adaptive_dataset, run_adaptive_kf
from .attitude import quat_from_euler_zyx, quat_from_small_angle, quat_multiply
from .bench import compare_scores, score_run
from .consistency import chi2_bounds, consistency_test, ensemble_consistency, nees, nis
from .ekf import ExtendedKalmanFilter
from .kf import KalmanFilter, steady_state_riccati
from .mekf import MultiplicativeEKF
from .models import (
    constant_velocity_2d,
    constant_velocity_cwna,
    constant_velocity_dwna,
    radar_jacobian,
    radar_measurement,
    random_walk,
    simulate_linear_system,
    simulate_radar_scenario,
)
from .sensors import (
    GyroModel,
    StarTrackerModel,
    arw_deg_per_sqrt_hour_to_si,
    rrw_deg_per_hour_1p5_to_si,
)
from .truth import attitude_trajectory
from .ukf import UnscentedKalmanFilter


def _emit(payload: dict[str, Any], text: list[str], as_json: bool) -> int:
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))
    else:
        print("\n".join(text))
    return 0


def _json_default(obj: object) -> object:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"object of type {type(obj).__name__} is not JSON serialisable")


def _cmd_riccati(args: argparse.Namespace) -> int:
    if args.model == "random-walk":
        f, h, q, r = random_walk(args.q, args.r)
        label = f"scalar random walk q = {args.q}, r = {args.r}"
    elif args.model == "cv-cwna":
        f, q = constant_velocity_cwna(args.dt, args.q)
        h = np.array([[1.0, 0.0]])
        r = np.array([[args.r]])
        label = f"CWNA constant velocity dt = {args.dt} s, q_psd = {args.q}, R = {args.r}"
    else:
        f, q = constant_velocity_dwna(args.dt, np.sqrt(args.q))
        h = np.array([[1.0, 0.0]])
        r = np.array([[args.r]])
        label = f"DWNA constant velocity dt = {args.dt} s, sigma_a^2 = {args.q}, R = {args.r}"
    p_prior, p_post, gain, iters = steady_state_riccati(f, h, q, r)
    payload = {
        "model": args.model,
        "label": label,
        "iterations": iters,
        "p_prior": p_prior,
        "p_post": p_post,
        "gain": gain,
    }
    text = [
        f"steady-state Riccati solution — {label}",
        f"  converged in {iters} iterations",
        f"  P^-_inf =\n{np.array2string(p_prior, precision=12)}",
        f"  P^+_inf =\n{np.array2string(p_post, precision=12)}",
        f"  K_inf   =\n{np.array2string(gain, precision=12)}",
    ]
    return _emit(payload, text, args.json)


def _radar_setup(dt: float, q_psd: float, sigma_range: float, sigma_bearing: float):
    f, q = constant_velocity_2d(dt, q_psd)
    r = np.diag([sigma_range**2, sigma_bearing**2])
    return f, q, r


def _cmd_bench(args: argparse.Namespace) -> int:
    rng = np.random.default_rng(args.seed)
    x0 = np.array([args.x0, args.vx0, args.y0, args.vy0])
    truth, meas = simulate_radar_scenario(
        dt=args.dt,
        n_steps=args.steps,
        q_psd=args.q,
        sigma_range=args.sigma_range,
        sigma_bearing=args.sigma_bearing,
        x0=x0,
        rng=rng,
    )
    f, q, r = _radar_setup(args.dt, args.q, args.sigma_range, args.sigma_bearing)
    p0 = np.diag([args.sigma_range**2, 100.0, args.sigma_range**2, 100.0])
    x_init = np.array([truth[0, 0], 0.0, truth[0, 2], 0.0])

    scores = []
    ekf = ExtendedKalmanFilter(
        lambda x: f @ x,
        radar_measurement,
        q,
        r,
        x_init,
        p0,
        f_jac=lambda x: f,
        h_jac=radar_jacobian,
    )
    res_e = ekf.run(meas)
    scores.append(
        score_run(
            "EKF", truth, res_e.x_post, res_e.p_post, res_e.innovation, res_e.innovation_cov,
            burn_in=args.burn_in,
        )
    )
    ukf = UnscentedKalmanFilter(
        lambda x: f @ x, radar_measurement, q, r, x_init, p0,
        alpha=args.alpha, beta=2.0, kappa=0.0,
    )
    res_u = ukf.run(meas)
    scores.append(
        score_run(
            "UKF", truth, res_u.x_post, res_u.p_post, res_u.innovation, res_u.innovation_cov,
            burn_in=args.burn_in,
        )
    )
    # Linear KF on the same trajectory with the measurement converted to
    # Cartesian: the "wrong but common" baseline, included so the bench shows
    # what converting a nonlinear measurement costs.
    z_cart = np.column_stack(
        [meas[:, 0] * np.cos(meas[:, 1]), meas[:, 0] * np.sin(meas[:, 1])]
    )
    h_lin = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    r_lin = np.diag([args.sigma_range**2, args.sigma_range**2])
    kf = KalmanFilter(f, h_lin, q, r_lin, x_init, p0)
    res_k = kf.run(z_cart)
    scores.append(
        score_run(
            "KF (converted)", truth, res_k.x_post, res_k.p_post,
            res_k.innovation, res_k.innovation_cov, burn_in=args.burn_in,
        )
    )
    payload = {
        "seed": args.seed,
        "steps": args.steps,
        "scores": [
            {
                "name": s.name,
                "rmse_total": s.rmse_total,
                "rmse": s.rmse,
                "mean_nees": s.mean_nees,
                "mean_nis": s.mean_nis,
                "nees_lower": s.nees_result.lower,
                "nees_upper": s.nees_result.upper,
                "verdict": s.nees_result.verdict,
                "diverged": s.diverged,
            }
            for s in scores
        ],
    }
    text = [
        f"navbench bench — radar tracking, seed {args.seed}, {args.steps} steps, "
        f"burn-in {args.burn_in}",
        "",
        compare_scores(scores),
        "",
        "NEES/NIS bounds are time-averaged and therefore indicative "
        "(successive steps are not independent).",
        "Use `navbench consistency` for the Monte Carlo form over independent runs.",
    ]
    return _emit(payload, text, args.json)


def _cmd_attitude(args: argparse.Namespace) -> int:
    rng = np.random.default_rng(args.seed)
    inertia = np.diag([args.ixx, args.iyy, args.izz])
    truth = attitude_trajectory(
        inertia=inertia,
        quat0=quat_from_euler_zyx(0.2, -0.1, 0.3),
        omega0=np.array([0.01, -0.02, 0.015]),
        dt=args.dt,
        n_steps=args.steps,
        torque_fn=lambda t, q, w: np.array([1e-5 * np.sin(0.01 * t), 0.0, 0.0]),
    )
    sigma_v = arw_deg_per_sqrt_hour_to_si(args.arw)
    sigma_u = rrw_deg_per_hour_1p5_to_si(args.rrw)
    # Initial errors are drawn FROM p0 so that NEES is a meaningful test; an
    # arbitrary fixed initial error makes the filter look pessimistic for
    # reasons that have nothing to do with the filter.
    sig_a0, sig_b0 = args.sigma_a0, args.sigma_b0
    p0 = np.diag([sig_a0**2] * 3 + [sig_b0**2] * 3)
    bias_true0 = sig_b0 * rng.standard_normal(3)
    gyro = GyroModel(sigma_v=sigma_v, sigma_u=sigma_u, dt=args.dt, bias0=bias_true0)
    rates, biases = gyro.sample_series(truth.interval_rate(), rng)
    tracker = StarTrackerModel(sigma_rad=args.sigma_st, reference_vectors=np.eye(3))
    q_meas = np.array([tracker.sample_quaternion(q, rng) for q in truth.quat[1:]])
    quat_init = quat_multiply(
        truth.quat[0], quat_from_small_angle(sig_a0 * rng.standard_normal(3))
    )

    mekf = MultiplicativeEKF(
        sigma_v=sigma_v,
        sigma_u=sigma_u,
        dt=args.dt,
        quat0=quat_init,
        bias0=np.zeros(3),
        p0=p0,
    )
    res = mekf.run(
        rates,
        quat_meas=q_meas,
        sigma_rad=args.sigma_st,
        measurement_every=args.meas_every,
    )
    err = res.error_state(truth.quat[1:], biases)
    b = args.burn_in
    nees_att = consistency_test(
        nees(err[b:, :3], res.covariance[b:, :3, :3]), 3, statistic="NEES", independent=False
    )
    nees_full = consistency_test(
        nees(err[b:], res.covariance[b:]), 6, statistic="NEES", independent=False
    )
    nis_res = consistency_test(
        nis(res.innovation[b:], res.innovation_cov[b:]), 3, statistic="NIS", independent=False
    )
    att_rms = float(np.sqrt(np.mean(np.sum(err[b:, :3] ** 2, axis=1))))
    bias_rms = float(np.sqrt(np.mean(np.sum(err[b:, 3:] ** 2, axis=1))))
    payload = {
        "seed": args.seed,
        "steps": args.steps,
        "attitude_rms_rad": att_rms,
        "attitude_rms_arcsec": att_rms * 180.0 / np.pi * 3600.0,
        "bias_rms_rad_s": bias_rms,
        "mean_nees_attitude": nees_att.mean,
        "mean_nees_full": nees_full.mean,
        "mean_nis": nis_res.mean,
        "nees_attitude_bounds": [nees_att.lower, nees_att.upper],
        "nis_bounds": [nis_res.lower, nis_res.upper],
        "max_quat_norm_error": float(
            np.max(np.abs(np.linalg.norm(res.quat, axis=1) - 1.0))
        ),
        "max_reset_angle_rad": float(np.max(res.reset_angle)),
    }
    text = [
        f"navbench attitude — MEKF, seed {args.seed}, {args.steps} steps of {args.dt} s",
        f"  attitude RMS error : {att_rms:.6e} rad "
        f"({att_rms * 180.0 / np.pi * 3600.0:.3f} arcsec)",
        f"  gyro bias RMS error: {bias_rms:.6e} rad/s",
        f"  attitude block {nees_att.summary()}",
        f"  full 6-state  {nees_full.summary()}",
        f"  {nis_res.summary()}",
        f"  max |q| - 1 over the run: {payload['max_quat_norm_error']:.3e}",
        f"  max reset angle folded into the reference: "
        f"{payload['max_reset_angle_rad']:.3e} rad",
        "",
        "CAVEAT: these are TIME averages over one run.  The gyro-bias error is",
        "nearly constant once converged, so its time average is effectively one",
        "sample and its chi-squared bound does not apply.  The defensible test is",
        "the Monte Carlo form in validation/v4_mekf_quaternion.py.",
    ]
    return _emit(payload, text, args.json)


def _cmd_consistency(args: argparse.Namespace) -> int:
    dt = args.dt
    f, q_true = constant_velocity_cwna(dt, args.q)
    h = np.array([[1.0, 0.0]])
    r = np.array([[args.sigma_z**2]])
    q_filter = constant_velocity_cwna(dt, args.q * args.q_mismatch)[1]
    p0 = np.diag([100.0, 10.0])
    nees_runs = np.zeros((args.runs, args.steps))
    nis_runs = np.zeros((args.runs, args.steps))
    for i in range(args.runs):
        rng = np.random.default_rng(args.seed + i)
        truth, meas = simulate_linear_system(
            f, h, q_true, r, np.array([0.0, 1.0]), args.steps, rng
        )
        kf = KalmanFilter(f, h, q_filter, r, np.array([0.0, 0.0]), p0)
        res = kf.run(meas)
        nees_runs[i] = nees(truth - res.x_post, res.p_post)
        nis_runs[i] = nis(res.innovation, res.innovation_cov)
    b = args.burn_in
    avg_nees, lo_e, hi_e = ensemble_consistency(nees_runs[:, b:], 2, args.alpha_level)
    avg_nis, lo_i, hi_i = ensemble_consistency(nis_runs[:, b:], 1, args.alpha_level)
    frac_e = float(np.mean((avg_nees >= lo_e) & (avg_nees <= hi_e)))
    frac_i = float(np.mean((avg_nis >= lo_i) & (avg_nis <= hi_i)))
    payload = {
        "runs": args.runs,
        "steps": args.steps,
        "q_mismatch": args.q_mismatch,
        "nees_bounds": [lo_e, hi_e],
        "nis_bounds": [lo_i, hi_i],
        "mean_anees": float(np.mean(avg_nees)),
        "mean_anis": float(np.mean(avg_nis)),
        "fraction_nees_inside": frac_e,
        "fraction_nis_inside": frac_i,
    }
    text = [
        f"navbench consistency — {args.runs} independent runs x {args.steps} steps, "
        f"Q mismatch factor {args.q_mismatch}",
        f"  ANEES (dof 2): mean {np.mean(avg_nees):.4f}, "
        f"bounds [{lo_e:.4f}, {hi_e:.4f}], {100.0 * frac_e:.1f} % of steps inside",
        f"  ANIS  (dof 1): mean {np.mean(avg_nis):.4f}, "
        f"bounds [{lo_i:.4f}, {hi_i:.4f}], {100.0 * frac_i:.1f} % of steps inside",
        f"  single-sample NEES bounds: {chi2_bounds(2, 1, args.alpha_level)}",
    ]
    return _emit(payload, text, args.json)


def _cmd_adaptive(args: argparse.Namespace) -> int:
    x_train, y_train, _ = generate_adaptive_dataset(
        n_runs=args.train_runs, n_steps=args.steps, seed=args.seed
    )
    model = LearnedAdaptiveQ(n_members=args.members, random_state=args.seed).fit(
        x_train, y_train
    )
    f, q_nom = constant_velocity_cwna(1.0, 0.05)
    h = np.array([[1.0, 0.0]])
    r = np.array([[9.0]])
    p0 = np.diag([100.0, 10.0])
    results: dict[str, dict[str, float]] = {}
    for tuner in ("fixed", "mehra", "learned"):
        rmses, nees_means = [], []
        for i in range(args.test_runs):
            rng = np.random.default_rng(args.seed + 100000 + i)
            u = float(rng.uniform(-1.5, 1.5))
            _, q_true = constant_velocity_cwna(1.0, 0.05 * 10.0**u)
            truth, meas = simulate_linear_system(
                f, h, q_true, r, np.array([0.0, 1.0]), args.steps, rng
            )
            res = run_adaptive_kf(
                f=f, h=h, q_nominal=q_nom, r=r, x0=np.zeros(2), p0=p0,
                measurements=meas, tuner=tuner,
                model=model if tuner == "learned" else None,
            )
            err = truth[args.burn_in :] - res.states[args.burn_in :]
            rmses.append(float(np.sqrt(np.mean(err[:, 0] ** 2))))
            nees_means.append(float(np.mean(nees(err, res.covariances[args.burn_in :]))))
        results[tuner] = {
            "position_rmse": float(np.mean(rmses)),
            "mean_nees": float(np.mean(nees_means)),
        }
    lo, hi = chi2_bounds(2, args.test_runs, 0.05)
    payload = {"train_samples": int(x_train.shape[0]), "results": results,
               "nees_bounds": [lo, hi]}
    text = [
        f"navbench adaptive — {args.train_runs} training runs "
        f"({x_train.shape[0]} windows), {args.test_runs} held-out runs",
        f"{'tuner':<10}{'pos RMSE [m]':>16}{'mean NEES (dof 2)':>20}",
        "-" * 46,
    ]
    for name, vals in results.items():
        text.append(
            f"{name:<10}{vals['position_rmse']:>16.5f}{vals['mean_nees']:>20.4f}"
        )
    text.append(f"NEES acceptance band over {args.test_runs} runs: [{lo:.4f}, {hi:.4f}]")
    return _emit(payload, text, args.json)


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser (exposed for testing)."""
    p = argparse.ArgumentParser(
        prog="navbench",
        description="Attitude and navigation filter bench with first-class "
        "NEES/NIS consistency diagnostics.",
    )
    p.add_argument("--version", action="version", version=f"navbench {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    pr = sub.add_parser("riccati", help="steady-state Riccati solution")
    pr.add_argument(
        "--model", choices=["random-walk", "cv-cwna", "cv-dwna"], default="random-walk"
    )
    pr.add_argument("--q", type=float, default=1.0, help="process noise intensity")
    pr.add_argument("--r", type=float, default=1.0, help="measurement noise variance")
    pr.add_argument("--dt", type=float, default=1.0, help="sample interval [s]")
    pr.add_argument("--json", action="store_true")
    pr.set_defaults(func=_cmd_riccati)

    pb = sub.add_parser("bench", help="KF/EKF/UKF on the radar scenario")
    pb.add_argument("--seed", type=int, default=2026)
    pb.add_argument("--steps", type=int, default=200)
    pb.add_argument("--dt", type=float, default=1.0)
    pb.add_argument("--q", type=float, default=0.05, help="acceleration PSD [m^2/s^3]")
    pb.add_argument("--sigma-range", type=float, default=20.0, help="[m]")
    pb.add_argument("--sigma-bearing", type=float, default=0.01, help="[rad]")
    pb.add_argument("--x0", type=float, default=3000.0)
    pb.add_argument("--vx0", type=float, default=-5.0)
    pb.add_argument("--y0", type=float, default=3000.0)
    pb.add_argument("--vy0", type=float, default=3.0)
    pb.add_argument("--alpha", type=float, default=1.0, help="UKF sigma-point alpha")
    pb.add_argument("--burn-in", type=int, default=20)
    pb.add_argument("--json", action="store_true")
    pb.set_defaults(func=_cmd_bench)

    pa = sub.add_parser("attitude", help="MEKF on a rigid-body attitude scenario")
    pa.add_argument("--seed", type=int, default=7)
    pa.add_argument("--steps", type=int, default=600)
    pa.add_argument("--dt", type=float, default=0.5)
    pa.add_argument("--arw", type=float, default=0.05, help="gyro ARW [deg/sqrt(hr)]")
    pa.add_argument("--rrw", type=float, default=0.005, help="gyro RRW [deg/hr^1.5]")
    pa.add_argument("--sigma-st", type=float, default=3e-5, help="star tracker sigma [rad]")
    pa.add_argument("--meas-every", type=int, default=4)
    pa.add_argument("--ixx", type=float, default=10.0)
    pa.add_argument("--iyy", type=float, default=15.0)
    pa.add_argument("--izz", type=float, default=20.0)
    pa.add_argument("--sigma-a0", type=float, default=0.05,
                    help="initial attitude uncertainty per axis [rad]")
    pa.add_argument("--sigma-b0", type=float, default=2e-6,
                    help="initial gyro bias uncertainty per axis [rad/s]")
    pa.add_argument("--burn-in", type=int, default=100)
    pa.add_argument("--json", action="store_true")
    pa.set_defaults(func=_cmd_attitude)

    pc = sub.add_parser("consistency", help="Monte Carlo NEES/NIS with chi-squared bounds")
    pc.add_argument("--seed", type=int, default=11)
    pc.add_argument("--runs", type=int, default=50)
    pc.add_argument("--steps", type=int, default=150)
    pc.add_argument("--dt", type=float, default=1.0)
    pc.add_argument("--q", type=float, default=0.05)
    pc.add_argument("--sigma-z", type=float, default=3.0)
    pc.add_argument(
        "--q-mismatch", type=float, default=1.0,
        help="factor applied to the filter's Q relative to the truth (1.0 = correct)",
    )
    pc.add_argument("--alpha-level", type=float, default=0.05)
    pc.add_argument("--burn-in", type=int, default=20)
    pc.add_argument("--json", action="store_true")
    pc.set_defaults(func=_cmd_consistency)

    pd = sub.add_parser("adaptive", help="train and benchmark the adaptive-Q tuners")
    pd.add_argument("--seed", type=int, default=20260812)
    pd.add_argument("--train-runs", type=int, default=60)
    pd.add_argument("--test-runs", type=int, default=30)
    pd.add_argument("--steps", type=int, default=400)
    pd.add_argument("--members", type=int, default=5)
    pd.add_argument("--burn-in", type=int, default=50)
    pd.add_argument("--json", action="store_true")
    pd.set_defaults(func=_cmd_adaptive)
    return p


def main(argv: list[str] | None = None) -> int:
    """Entry point.  Returns 0 on success, 2 on invalid input."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (ValueError, TypeError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

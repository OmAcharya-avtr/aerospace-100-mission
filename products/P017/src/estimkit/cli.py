"""Command-line entry point: ``python -m estimkit``.

Two subcommands, both deterministic:

``steady-state``
    Iterate the Riccati recursion for a scalar random-walk model or a
    2-state constant-velocity model and print the converged gain and
    covariances.

``track``
    Run a seeded constant-velocity tracking scenario through the linear
    filter and the RTS smoother and print position/velocity RMS errors.

Printing happens only here; the library itself never prints.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np

from .linear import KalmanFilter, steady_state
from .models import constant_velocity_cwna, random_walk
from .smoother import rts_smooth


def _steady_state(args: argparse.Namespace) -> dict[str, Any]:
    if args.model == "random-walk":
        f, h, q, r = random_walk(args.q, args.r)
    else:
        f, q = constant_velocity_cwna(args.dt, args.q)
        h = np.array([[1.0, 0.0]])
        r = np.array([[args.r]])
    p_prior, p_post, gain, iters = steady_state(f, h, q, r)
    return {
        "model": args.model,
        "iterations": iters,
        "P_prior": p_prior.tolist(),
        "P_post": p_post.tolist(),
        "K": gain.tolist(),
    }


def _track(args: argparse.Namespace) -> dict[str, Any]:
    rng = np.random.default_rng(args.seed)
    dt = args.dt
    f, q = constant_velocity_cwna(dt, args.q)
    h = np.array([[1.0, 0.0]])
    r = np.array([[args.r]])

    n = args.steps
    truth = np.zeros((n, 2))
    x = np.array([0.0, 10.0])
    chol_q = np.linalg.cholesky(q + 1e-15 * np.eye(2))
    for k in range(n):
        x = f @ x + chol_q @ rng.standard_normal(2)
        truth[k] = x
    zs = truth[:, 0:1] + np.sqrt(args.r) * rng.standard_normal((n, 1))

    kf = KalmanFilter(f, h, q, r)
    res = kf.filter(np.array([0.0, 0.0]), np.diag([100.0, 100.0]), zs)
    sm = rts_smooth(res)

    def rms(est: np.ndarray, idx: int) -> float:
        return float(np.sqrt(np.mean((est[:, idx] - truth[:, idx]) ** 2)))

    return {
        "seed": args.seed,
        "steps": n,
        "rms_position_filter_m": rms(res.x_post, 0),
        "rms_position_smoother_m": rms(sm.x, 0),
        "rms_velocity_filter_mps": rms(res.x_post, 1),
        "rms_velocity_smoother_mps": rms(sm.x, 1),
        "mean_nis": float(np.mean(res.nis)),
        "nis_dof": kf.m,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for ``python -m estimkit``."""
    parser = argparse.ArgumentParser(
        prog="estimkit",
        description="Compact Kalman filter family (educational). SI units throughout.",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    sub = parser.add_subparsers(dest="command", required=True)

    ss = sub.add_parser("steady-state", help="converged Riccati gain and covariance")
    ss.add_argument(
        "--model",
        choices=("random-walk", "constant-velocity"),
        default="random-walk",
        help="model family (default: random-walk)",
    )
    ss.add_argument("--q", type=float, default=1.0, help="process noise (variance or PSD)")
    ss.add_argument("--r", type=float, default=1.0, help="measurement noise variance")
    ss.add_argument("--dt", type=float, default=1.0, help="sample interval [s]")
    ss.set_defaults(func=_steady_state)

    tr = sub.add_parser("track", help="seeded constant-velocity filter vs smoother RMS")
    tr.add_argument("--steps", type=int, default=200, help="number of time steps")
    tr.add_argument("--dt", type=float, default=1.0, help="sample interval [s]")
    tr.add_argument("--q", type=float, default=0.01, help="acceleration PSD [m^2/s^3]")
    tr.add_argument("--r", type=float, default=4.0, help="range-measurement variance [m^2]")
    tr.add_argument("--seed", type=int, default=2026, help="RNG seed")
    tr.set_defaults(func=_track)
    return parser


def _format_table(payload: dict[str, Any]) -> str:
    lines = []
    width = max(len(k) for k in payload)
    for key, value in payload.items():
        if isinstance(value, float):
            lines.append(f"{key:<{width}} : {value:.6g}")
        elif isinstance(value, list):
            flat = np.array2string(np.asarray(value), precision=6, suppress_small=False)
            indent = " " * (width + 3)
            lines.append(f"{key:<{width}} : " + flat.replace("\n", "\n" + indent))
        else:
            lines.append(f"{key:<{width}} : {value}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run the CLI. Returns a process exit code (0 ok, 2 on bad input)."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        payload = args.func(args)
    except (ValueError, RuntimeError) as exc:
        parser.exit(2, f"estimkit: error: {exc}\n")
    print(json.dumps(payload, indent=2) if args.json else _format_table(payload))
    return 0

"""Command-line interface: ``python -m navbench``.

Subcommands
-----------
``consistency``
    Monte Carlo a linear KF on the constant-velocity model and report NEES/NIS
    against their chi-squared acceptance regions.
``riccati``
    Compare the converged filter covariance with the analytic steady-state
    Riccati solution for the same LTI model.
``adapt``
    Benchmark fixed, covariance-matching, IAE and learned process-noise
    adaptation on the same mis-specified trajectories.

Exit codes: 0 success, 2 user error (bad arguments or invalid physics values;
no traceback is shown).
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

import numpy as np

from .adaptive import CovarianceMatching, FixedQ, IaeScaleAdapter
from .bench import run_linear_mc, tune_fixed_scale
from .kf import KalmanFilter, steady_state_riccati
from .models import ConstantVelocity

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    p = argparse.ArgumentParser(
        prog="python -m navbench",
        description="Navigation-filter bench with consistency diagnostics.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dt", type=float, default=1.0, help="sample interval [s]")
    common.add_argument("--q-psd", type=float, default=0.1,
                        help="nominal acceleration PSD [m^2/s^3]")
    common.add_argument("--sigma-pos", type=float, default=5.0,
                        help="position measurement 1-sigma [m]")
    common.add_argument("--steps", type=int, default=160, help="steps per run")
    common.add_argument("--seed", type=int, default=0, help="base seed")

    c = sub.add_parser("consistency", parents=[common], help="NEES/NIS Monte Carlo")
    c.add_argument("--runs", type=int, default=100, help="Monte Carlo runs")
    c.add_argument("--true-scale", type=float, default=1.0,
                   help="multiplier on the TRUE process noise (1.0 = correctly specified)")

    sub.add_parser("riccati", parents=[common], help="steady-state Riccati comparison")

    a = sub.add_parser("adapt", parents=[common], help="process-noise adaptation benchmark")
    a.add_argument("--runs", type=int, default=40, help="Monte Carlo runs per scheme")
    a.add_argument("--train-episodes", type=int, default=300,
                   help="training episodes for the learned adapter")
    a.add_argument("--skip-learned", action="store_true",
                   help="skip the learned adapter (classical baselines only)")
    return p


def _model(args: argparse.Namespace) -> ConstantVelocity:
    return ConstantVelocity(
        dt=args.dt, q_psd=args.q_psd, sigma_pos=args.sigma_pos, dim=2
    )


def _cmd_consistency(args: argparse.Namespace, out) -> int:
    model = _model(args)
    res = run_linear_mc(
        model, n_runs=args.runs, n_steps=args.steps, seed=args.seed,
        q_true_scale=args.true_scale, label=f"kf(true_scale={args.true_scale:g})",
    )
    print(res.summary(), file=out)
    print(res.nees_report.summary(), file=out)
    print(res.nis_report.summary(), file=out)
    return 0


def _cmd_riccati(args: argparse.Namespace, out) -> int:
    model = _model(args)
    f, q, h, r = model.f(), model.q(1.0), model.h(), model.r()
    p_prior, p_post, gain = steady_state_riccati(f, q, h, r)
    kf = KalmanFilter(f=f, q=q, h=h, r=r, x=np.zeros(model.n), p=np.eye(model.n) * 1e4)
    for _ in range(args.steps):
        kf.predict()
        kf.update(h @ np.zeros(model.n))
    err = float(np.max(np.abs(kf.p - p_post)))
    rel = err / float(np.max(np.abs(p_post)))
    print(f"analytic steady-state posterior covariance P+:\n{p_post}", file=out)
    print(f"filter covariance after {args.steps} steps:\n{kf.p}", file=out)
    print(f"steady-state gain K:\n{gain}", file=out)
    print(f"max |P_filter - P_analytic| = {err:.6e}  (relative {rel:.3e})", file=out)
    print(f"prior covariance trace = {float(np.trace(p_prior)):.6f}", file=out)
    return 0


def _cmd_adapt(args: argparse.Namespace, out) -> int:
    model = _model(args)
    mixed = [0.05, 0.2, 1.0, 5.0, 20.0]
    best, table = tune_fixed_scale(
        model, candidate_scales=[0.1, 0.3, 1.0, 3.0, 10.0],
        train_true_scales=mixed, n_runs=12, n_steps=args.steps, seed=args.seed + 555,
    )
    print(f"hand-tuned fixed scale (grid search on training seeds): {best:g}", file=out)
    print(f"  grid: {{{', '.join(f'{k:g}: {v:.3f}' for k, v in sorted(table.items()))}}}", file=out)

    factories = [
        ("fixed[1.0]", lambda: FixedQ(1.0)),
        (f"fixed[tuned={best:g}]", lambda: FixedQ(best)),
        ("covariance-matching", lambda: CovarianceMatching(window=30)),
        ("iae", lambda: IaeScaleAdapter(window=30)),
    ]
    if not args.skip_learned:
        from .ai import LearnedQAdapter, QScaleEnsemble, generate_training_data

        x, y, _ = generate_training_data(n_episodes=args.train_episodes, seed=args.seed + 991)
        ens = QScaleEnsemble(seed=0).fit(x, y)
        print(
            f"learned ensemble: {x.shape[0]} training windows, calibrated z="
            f"{ens.calibration_z:.3f}, raw/calibrated coverage "
            f"{ens.raw_coverage:.3f}/{ens.calibrated_coverage:.3f}",
            file=out,
        )
        factories.append(("learned", lambda: LearnedQAdapter(ensemble=ens, window=30)))

    for name, fac in factories:
        res = run_linear_mc(
            model, adapter_factory=fac, n_runs=args.runs, n_steps=args.steps,
            seed=args.seed, q_true_scale=mixed, label=name,
        )
        print(res.summary(), file=out)
    return 0


def main(argv: Sequence[str] | None = None, out=None) -> int:
    """Entry point. Returns a process exit code."""
    stream = sys.stdout if out is None else out
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        if args.command == "consistency":
            return _cmd_consistency(args, stream)
        if args.command == "riccati":
            return _cmd_riccati(args, stream)
        if args.command == "adapt":
            return _cmd_adapt(args, stream)
        parser.error(f"unknown command {args.command!r}")
        return 2
    except (ValueError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

"""Command-line interface: ``python -m trackforge run|benchmark|reacq``.

Subcommands
-----------
``run SCENARIO.yaml``   run one end-to-end episode and print its metrics
``benchmark``           controller benchmark + simulation-throughput figures
``reacq``               train the Q-learning policy and compare it against
                        both scripted baselines on identical Monte Carlo
                        episodes
"""

from __future__ import annotations

import argparse
import json
import math
import sys

import numpy as np

from trackforge import __version__
from trackforge.control import (
    LQRController,
    PIDController,
    benchmark_controllers,
    lqr_weights_from_bandwidth,
    pid_gains_from_bandwidth,
)
from trackforge.dynamics import GimbalAxis, JitterPSD, synthesize_jitter
from trackforge.reacq import (
    AlwaysFullPolicy,
    AlwaysLocalPolicy,
    ReacqConfig,
    evaluate_policy,
    train_q_learning,
)
from trackforge.sim import DEFAULT_SCENARIO, load_scenario, run_episode, sim_steps_per_second


def _fmt_table(rows: list[dict], keys: list[str]) -> str:
    widths = {k: max(len(k), *(len(_cell(r.get(k))) for r in rows)) for k in keys}
    head = " | ".join(k.ljust(widths[k]) for k in keys)
    sep = "-+-".join("-" * widths[k] for k in keys)
    body = "\n".join(" | ".join(_cell(r.get(k)).ljust(widths[k]) for k in keys) for r in rows)
    return f"{head}\n{sep}\n{body}"


def _cell(v: object) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        if v == 0 or (1e-3 <= abs(v) < 1e5):
            return f"{v:.4f}"
        return f"{v:.3e}"
    return str(v)


def cmd_run(args: argparse.Namespace) -> int:
    """Run a single episode from a YAML scenario."""
    sc = load_scenario(args.scenario) if args.scenario else DEFAULT_SCENARIO
    res = run_episode(sc, seed=args.seed, keep_series=False)
    summary = res.summary()
    if args.json:
        sys.stdout.write(json.dumps(summary, indent=2, default=float) + "\n")
    else:
        sys.stdout.write(f"scenario: {sc.name}  seed: {args.seed or sc.seed}\n")
        for k, v in summary.items():
            sys.stdout.write(f"  {k:<24} {_cell(v)}\n")
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Controller comparison table and simulation throughput."""
    sc = load_scenario(args.scenario) if args.scenario else DEFAULT_SCENARIO
    dt = sc.dt

    def axis_factory() -> GimbalAxis:
        return GimbalAxis(sc.inertia, sc.damping, sc.torque_max, sc.rate_max, sc.accel_max)

    kp, ki, kd = pid_gains_from_bandwidth(
        sc.inertia, 2 * math.pi * sc.bandwidth_hz, sc.damping_ratio, sc.integral_alpha
    )
    rng = np.random.default_rng(7)
    n = int(round(2.0 / dt))
    dist = synthesize_jitter(
        JitterPSD(sc.jitter_s0, sc.jitter_f_corner, sc.jitter_order), n, 1.0 / dt, rng
    )
    lqr_q, lqr_qr, lqr_r = lqr_weights_from_bandwidth(
        sc.inertia, 2 * math.pi * sc.bandwidth_hz, sc.lqr_q_angle
    )
    if sc.lqr_r_torque is not None:
        lqr_qr, lqr_r = sc.lqr_q_rate, sc.lqr_r_torque
    ctrls = {
        "PID": lambda ax: PIDController(kp, ki, kd, ax.torque_max),
        "PD (Ki=0)": lambda ax: PIDController(kp, 0.0, kd, ax.torque_max),
        "LQR": lambda ax: LQRController(
            ax, q_angle=lqr_q, q_rate=lqr_qr, r_torque=lqr_r
        ),
    }
    rows = benchmark_controllers(axis_factory, ctrls, dt=dt, disturbance=dist)
    keys = [
        "name",
        "rise_time_s",
        "overshoot",
        "settling_time_s",
        "dist_rms_rad",
        "rejection_factor",
        "bandwidth_hz",
    ]
    sys.stdout.write("Controller benchmark (open-loop disturbance RMS = "
                     f"{np.std(dist):.3e} rad)\n")
    sys.stdout.write(_fmt_table(rows, keys) + "\n\n")
    perf = sim_steps_per_second(sc)
    sys.stdout.write(
        f"Simulation throughput: {perf['steps_per_second']:.0f} steps/s "
        f"({perf['realtime_factor']:.1f}x realtime, {perf['steps']} steps)\n"
    )
    return 0


def cmd_reacq(args: argparse.Namespace) -> int:
    """Train the Q-learning policy and compare with both baselines."""
    cfg = ReacqConfig()
    sys.stdout.write(f"training tabular Q-learning ({args.episodes} episodes, "
                     f"seed {args.seed})...\n")
    learned = train_q_learning(cfg, episodes=args.episodes, seed=args.seed)
    rows = [
        evaluate_policy(p, cfg, n_episodes=args.eval_episodes, seed=args.eval_seed)
        for p in (AlwaysFullPolicy(), AlwaysLocalPolicy(), learned)
    ]
    keys = ["policy", "mean_time_s", "ci_low_s", "ci_high_s", "median_time_s",
            "p90_time_s", "success_rate", "mean_attempts"]
    sys.stdout.write(_fmt_table(rows, keys) + "\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse CLI parser."""
    p = argparse.ArgumentParser(
        prog="python -m trackforge",
        description="TrackForge: PAT simulation suite for optical links",
    )
    p.add_argument("--version", action="version", version=f"trackforge {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="run one end-to-end episode from a YAML scenario")
    r.add_argument("scenario", nargs="?", default=None, help="path to scenario YAML")
    r.add_argument("--seed", type=int, default=None)
    r.add_argument("--json", action="store_true", help="emit JSON instead of text")
    r.set_defaults(func=cmd_run)

    b = sub.add_parser("benchmark", help="controller and throughput benchmarks")
    b.add_argument("scenario", nargs="?", default=None, help="path to scenario YAML")
    b.set_defaults(func=cmd_benchmark)

    q = sub.add_parser("reacq", help="train and benchmark the reacquisition policy")
    q.add_argument("--episodes", type=int, default=20000)
    q.add_argument("--seed", type=int, default=12345)
    q.add_argument("--eval-episodes", type=int, default=2000)
    q.add_argument("--eval-seed", type=int, default=999)
    q.set_defaults(func=cmd_reacq)
    return p


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns a process exit code."""
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

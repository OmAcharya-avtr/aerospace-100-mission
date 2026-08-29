"""Command-line interface: ``python -m linkswitch <command> ...``.

Examples
--------
python -m linkswitch threshold
python -m linkswitch compare --n-steps 2000 --n-reps 100 --horizon 5
python -m linkswitch simulate --policy hysteresis --tau-low 0.20 --tau-high 0.30
"""

from __future__ import annotations

import argparse
import sys

from .analytic import optimal_threshold_analytic, optimal_threshold_grid
from .learn import train_outage_predictor
from .metrics import compare_policies
from .optical import OpticalParams
from .policies import FixedThresholdPolicy, HysteresisPolicy, LearnedPolicy
from .rf import RFParams
from .scenario import ScenarioConfig, SwitchCost, generate_telemetry
from .simulate import simulate_policy


def _build_config(args: argparse.Namespace) -> ScenarioConfig:
    optical = OpticalParams(
        sigma_i2=args.sigma_i2,
        coherence_steps=args.coherence_steps,
        margin_db=args.margin_db,
        rate_mbps=args.opt_rate,
    )
    rf = RFParams(rate_mbps=args.rf_rate)
    switch_cost = SwitchCost(downtime_steps=args.downtime_steps)
    return ScenarioConfig(optical=optical, rf=rf, switch_cost=switch_cost)


def _cmd_threshold(args: argparse.Namespace) -> int:
    config = _build_config(args)
    ana = optimal_threshold_analytic(config.optical, config.rf, config.switch_cost.downtime_steps)
    grid = optimal_threshold_grid(config.optical, config.rf, config.switch_cost.downtime_steps)
    print("Optimal fixed switching threshold (closed-form channel-statistics model)")
    print(f"  z_phys (physical outage, standardised)  = {ana.z_phys:.6f}")
    print(f"  rho (AR(1) lag-1 correlation)            = {ana.rho:.6f}")
    print(f"  bounded-optimizer  z_th* = {ana.z_th:.6f}  tau* = {ana.tau:.6f}  "
          f"J* = {ana.objective:.6f}")
    print(f"  grid-search        z_th* = {grid.z_th:.6f}  tau* = {grid.tau:.6f}  "
          f"J* = {grid.objective:.6f}")
    print(f"  |z_th optimizer - z_th grid| = {abs(ana.z_th - grid.z_th):.6e}")
    return 0


def _cmd_simulate(args: argparse.Namespace) -> int:
    config = _build_config(args)
    telemetry = generate_telemetry(config, n_steps=args.n_steps, seed=args.seed)
    if args.policy == "fixed":
        policy = FixedThresholdPolicy(tau=args.tau)
    elif args.policy == "hysteresis":
        policy = HysteresisPolicy(tau_low=args.tau_low, tau_high=args.tau_high)
    else:
        raise SystemExit("error: --policy learned requires the 'compare' subcommand "
                          "(needs a trained model)")
    select = policy.select_channels(telemetry)
    metrics = simulate_policy(telemetry, select, config)
    print(f"policy={policy.name} n_steps={metrics.n_steps} seed={args.seed}")
    print(f"  throughput_mbps = {metrics.throughput_mbps:.4f}")
    print(f"  outage_fraction = {metrics.outage_fraction:.4f} "
          f"({metrics.outage_steps} steps)")
    print(f"  switch_count    = {metrics.switch_count}")
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    config = _build_config(args)
    tau_opt = config.optical.tau_phys

    train_telemetries = [
        generate_telemetry(config, n_steps=args.n_steps, seed=90_000 + i)
        for i in range(args.n_train_episodes)
    ]
    model = train_outage_predictor(
        train_telemetries, tau_phys=tau_opt, horizon=args.horizon, window=args.window,
        random_state=args.seed,
    )

    factories = {
        "fixed_threshold": lambda: FixedThresholdPolicy(tau=tau_opt),
        "hysteresis": lambda: HysteresisPolicy(tau_low=args.tau_low, tau_high=args.tau_high),
        "learned": lambda: LearnedPolicy(
            model, tau_phys=tau_opt, confidence_threshold=args.confidence, window=args.window
        ),
    }
    results = compare_policies(config, factories, n_steps=args.n_steps, n_reps=args.n_reps,
                                seed0=args.seed)

    print(f"Policy comparison: n_steps={args.n_steps} n_reps={args.n_reps} "
          f"(paired Monte Carlo, seed0={args.seed})")
    header = f"{'policy':<16}{'throughput Mb/s':>20}{'outage frac':>18}{'switches':>14}"
    print(header)
    for name, agg in results.items():
        t = agg["throughput_mbps"]
        o = agg["outage_fraction"]
        s = agg["switch_count"]
        print(
            f"{name:<16}"
            f"{t.mean:>10.3f} [{t.ci_low:.3f},{t.ci_high:.3f}]"
            f"{o.mean:>10.4f} [{o.ci_low:.4f},{o.ci_high:.4f}]"
            f"{s.mean:>8.2f}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m linkswitch",
        description="Hybrid RF/FSO switching policy simulation and comparison.",
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--sigma-i2", type=float, default=0.25, dest="sigma_i2",
                         help="optical scintillation index (weak-fluctuation lognormal)")
    common.add_argument("--coherence-steps", type=float, default=5.0, dest="coherence_steps",
                         help="AR(1) fade coherence time in steps")
    common.add_argument("--margin-db", type=float, default=6.0, dest="margin_db",
                         help="optical link margin (dB) defining the outage threshold")
    common.add_argument("--opt-rate", type=float, default=1000.0, dest="opt_rate",
                         help="optical rate when available, Mb/s")
    common.add_argument("--rf-rate", type=float, default=150.0, dest="rf_rate",
                         help="RF rate when available, Mb/s")
    common.add_argument("--downtime-steps", type=int, default=1, dest="downtime_steps",
                         help="switch downtime, in steps")
    common.add_argument("--seed", type=int, default=0, help="base RNG seed")

    sub = p.add_subparsers(dest="cmd", required=True)

    sp_t = sub.add_parser("threshold", parents=[common],
                           help="analytic vs grid-search optimal fixed threshold")
    sp_t.set_defaults(func=_cmd_threshold)

    sp_s = sub.add_parser("simulate", parents=[common], help="run one policy on one episode")
    sp_s.add_argument("--policy", choices=("fixed", "hysteresis"), default="fixed")
    sp_s.add_argument("--n-steps", type=int, default=2000, dest="n_steps")
    sp_s.add_argument("--tau", type=float, default=None,
                       help="fixed-threshold tau (default: physical outage threshold)")
    sp_s.add_argument("--tau-low", type=float, default=None, dest="tau_low")
    sp_s.add_argument("--tau-high", type=float, default=None, dest="tau_high")
    sp_s.set_defaults(func=_cmd_simulate)

    sp_c = sub.add_parser("compare", parents=[common],
                           help="Monte Carlo comparison of all three policies")
    sp_c.add_argument("--n-steps", type=int, default=2000, dest="n_steps")
    sp_c.add_argument("--n-reps", type=int, default=100, dest="n_reps")
    sp_c.add_argument("--n-train-episodes", type=int, default=20, dest="n_train_episodes")
    sp_c.add_argument("--horizon", type=int, default=5)
    sp_c.add_argument("--window", type=int, default=8)
    sp_c.add_argument("--confidence", type=float, default=0.5)
    sp_c.add_argument("--tau-low", type=float, default=None, dest="tau_low")
    sp_c.add_argument("--tau-high", type=float, default=None, dest="tau_high")
    sp_c.set_defaults(func=_cmd_compare)

    args = p.parse_args(argv)

    try:
        # Fill in scenario-dependent defaults that need the built config.
        if args.cmd == "simulate":
            config = _build_config(args)
            tau_phys = config.optical.tau_phys
            if args.tau is None:
                args.tau = tau_phys
            if args.tau_low is None:
                args.tau_low = tau_phys * 0.8
            if args.tau_high is None:
                args.tau_high = tau_phys * 1.2
        elif args.cmd == "compare":
            config = _build_config(args)
            tau_phys = config.optical.tau_phys
            if args.tau_low is None:
                args.tau_low = tau_phys * 0.8
            if args.tau_high is None:
                args.tau_high = tau_phys * 1.2

        return args.func(args)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from None


if __name__ == "__main__":
    sys.exit(main())

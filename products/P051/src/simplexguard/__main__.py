"""Command-line interface: ``python -m simplexguard``.

Subcommands
-----------
``plant``       print the declared plant, disturbance bound and constraint sets
``invariant``   compute the robust invariant set and verify it independently
``guard``       print the switching condition's precomputed data
``run``         simulate one guarded episode with its unguarded and baseline pairs
``accounting``  the six assurance quantities for one scenario
``bound-sweep`` the deliberate bound-violation experiment
``predict``     train the learned switch predictor and benchmark it against exact

Exit statuses: 0 success, 2 bad input, 3 the robust invariant set is empty or the
recursion did not converge -- which is a modelling outcome, not a crash, and gets
its own status so a script can tell the two apart.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from .accounting import account
from .boundviolation import bound_violation_sweep
from .controllers import reference_controllers
from .guard import SimplexGuard
from .invariant import (
    EmptyInvariantSet,
    RecursionDidNotConverge,
    robust_invariant_set,
    verify_robust_invariance,
)
from .plant import reference_plant
from .predictor import (
    build_dataset,
    exact_predictor_scores,
    fit_switch_predictor,
    guard_condition_scores,
    lead_times,
    measure_decision_cost,
    score_binary,
)
from .simulate import (
    disturbance_sequence,
    simulate_baseline,
    simulate_guarded,
    simulate_unguarded,
    square_wave_reference,
)

_DISCLAIMER = (
    "simplexguard is research-grade. It is not flight-qualified, not certified and "
    "not approved for operational aerospace use. It is not a verification tool and "
    "proves nothing about a plant it was not given."
)


def _plant_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dt", type=float, default=0.05, help="sample interval, s")
    parser.add_argument(
        "--angle-limit", type=float, default=0.30, help="declared |theta| bound, rad"
    )
    parser.add_argument(
        "--rate-limit", type=float, default=0.50, help="declared |theta_dot| bound, rad/s"
    )
    parser.add_argument(
        "--accel-limit", type=float, default=3.0, help="declared |u| bound, rad/s^2"
    )
    parser.add_argument(
        "--disturbance-accel",
        type=float,
        default=0.12,
        help="declared unmodelled angular acceleration bound, rad/s^2",
    )
    parser.add_argument(
        "--baseline-r",
        type=float,
        default=1.0,
        help="LQR input weight of the baseline; larger is more conservative",
    )
    parser.add_argument(
        "--min-baseline-dwell",
        type=int,
        default=1,
        help="minimum consecutive steps of baseline authority once it takes over",
    )


def _episode_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--steps", type=int, default=2000, help="steps per episode")
    parser.add_argument("--seed", type=int, default=5101)
    parser.add_argument(
        "--reference-amplitude", type=float, default=0.18, help="square-wave amplitude, rad"
    )
    parser.add_argument(
        "--reference-period", type=int, default=80, help="steps per half period"
    )
    parser.add_argument(
        "--disturbance-mode",
        choices=("uniform", "vertex", "zero"),
        default="uniform",
        help="how the realised disturbance is drawn from the declared box",
    )


def _build(args: argparse.Namespace):
    plant = reference_plant(
        dt=args.dt,
        angle_limit_rad=args.angle_limit,
        rate_limit_rad_s=args.rate_limit,
        accel_limit_rad_s2=args.accel_limit,
        disturbance_accel_rad_s2=args.disturbance_accel,
    )
    baseline, performance = reference_controllers(plant, baseline_r=args.baseline_r)
    result = robust_invariant_set(
        plant, baseline, max_iterations=getattr(args, "max_iterations", 200)
    )
    guard = SimplexGuard(
        plant,
        baseline,
        result.polytope,
        min_baseline_dwell=getattr(args, "min_baseline_dwell", 1),
    )
    return plant, baseline, performance, result, guard


def _episodes(args: argparse.Namespace, plant, baseline, performance, guard):
    rng = np.random.default_rng(args.seed)
    w = disturbance_sequence(plant, args.steps, rng, args.disturbance_mode)
    reference = square_wave_reference(
        args.reference_amplitude, args.reference_period, plant.n_states
    )
    guarded = simulate_guarded(plant, guard, performance, args.steps, reference, w)
    unguarded = simulate_unguarded(plant, performance, args.steps, reference, w)
    base_only = simulate_baseline(plant, baseline, args.steps, reference, w)
    return guarded, unguarded, base_only


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns the process exit status."""
    parser = argparse.ArgumentParser(
        prog="simplexguard",
        description=(
            "Runtime-assurance (Simplex) architecture benchmark: exact switching "
            "condition, assurance accounting, bound-violation experiment and a "
            "learned switch predictor benchmarked against the exact condition."
        ),
        epilog=_DISCLAIMER,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_plant = sub.add_parser("plant", help="print the declared plant and sets")
    _plant_args(p_plant)

    p_inv = sub.add_parser("invariant", help="compute and verify the robust invariant set")
    _plant_args(p_inv)
    p_inv.add_argument("--max-iterations", type=int, default=200)

    p_guard = sub.add_parser("guard", help="print the switching condition data")
    _plant_args(p_guard)

    p_run = sub.add_parser("run", help="simulate one guarded episode and its pairs")
    _plant_args(p_run)
    _episode_args(p_run)

    p_acc = sub.add_parser("accounting", help="the six assurance quantities")
    _plant_args(p_acc)
    _episode_args(p_acc)

    p_bs = sub.add_parser("bound-sweep", help="the deliberate bound-violation experiment")
    _plant_args(p_bs)
    p_bs.add_argument("--steps", type=int, default=1500)
    p_bs.add_argument("--seed", type=int, default=5102)
    p_bs.add_argument("--episodes", type=int, default=6)
    p_bs.add_argument(
        "--disturbance-mode", choices=("uniform", "vertex"), default="uniform"
    )
    p_bs.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=[1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0],
    )

    p_pred = sub.add_parser(
        "predict", help="train the learned switch predictor and benchmark it"
    )
    _plant_args(p_pred)
    p_pred.add_argument("--lead", type=int, default=5, help="prediction lead, steps")
    p_pred.add_argument("--episodes", type=int, default=40)
    p_pred.add_argument("--steps", type=int, default=1200)
    p_pred.add_argument("--seed", type=int, default=5101)
    p_pred.add_argument("--trees", type=int, default=150)

    args = parser.parse_args(argv)
    try:
        if args.command == "plant":
            plant = reference_plant(
                dt=args.dt,
                angle_limit_rad=args.angle_limit,
                rate_limit_rad_s=args.rate_limit,
                accel_limit_rad_s2=args.accel_limit,
                disturbance_accel_rad_s2=args.disturbance_accel,
            )
            print(plant.describe())
            return 0
        plant, baseline, performance, result, guard = _build(args)
        if args.command == "invariant":
            print(result.describe())
            print()
            report = verify_robust_invariance(plant, baseline, result.polytope)
            for key, value in report.items():
                print(f"{key:<28s}{value}")
            return 0
        if args.command == "guard":
            print(guard.describe())
            return 0
        if args.command in ("run", "accounting"):
            guarded, unguarded, base_only = _episodes(
                args, plant, baseline, performance, guard
            )
            if args.command == "run":
                for ep in (guarded, unguarded, base_only):
                    print(
                        f"{ep.architecture:<11s} cost={ep.cost():12.6f} "
                        f"baseline_fraction={ep.baseline_fraction:8.6f} "
                        f"X violations={ep.constraint_violations().size:<6d} "
                        f"S exits={ep.invariant_exits(result.polytope).size:<6d} "
                        f"worst X residual={ep.worst_constraint_residual():12.5e}"
                    )
                return 0
            print(account(guarded, unguarded, base_only, result.polytope).describe())
            return 0
        if args.command == "bound-sweep":
            sweep = bound_violation_sweep(
                plant,
                guard,
                performance,
                result.polytope,
                scales=args.scales,
                n_episodes=args.episodes,
                n_steps=args.steps,
                seed=args.seed,
                disturbance_mode=args.disturbance_mode,
            )
            print(sweep.describe())
            return 0
        if args.command == "predict":
            data = build_dataset(
                plant, guard, performance, args.episodes, args.steps, args.seed, args.lead
            )
            n = args.episodes
            train = data.select_episodes(np.arange(0, int(0.65 * n)))
            calib = data.select_episodes(np.arange(int(0.65 * n), int(0.8 * n)))
            test = data.select_episodes(np.arange(int(0.8 * n), n))
            print(
                f"lead={data.lead} rows={len(data)} base rate={data.base_rate:.6f} "
                f"test rows={len(test)} test base rate={test.base_rate:.6f} "
                f"performance-input saturation rate={data.saturation_rate:.6f}"
            )
            print(guard_condition_scores(test, guard, performance).row())
            for mode in ("worst_case", "nominal"):
                print(
                    exact_predictor_scores(
                        test, guard, performance, max(args.lead, 1), mode
                    ).row()
                )
            model = fit_switch_predictor(train, calib, "forest", n_estimators=args.trees)
            prob = model.predict_proba(test.features)[:, 1]
            row0 = test.features[0:1]
            cost = measure_decision_cost(lambda: model.predict_proba(row0), n_calls=200)
            print(
                score_binary(
                    f"learned forest ({args.trees} trees)",
                    test.labels,
                    prob >= 0.5,
                    probability=prob,
                    microseconds_per_decision=cost,
                ).row()
            )
            print(f"learned lead times  {lead_times(prob >= 0.5, test.fires_now)}")
            return 0
    except (EmptyInvariantSet, RecursionDidNotConverge) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    except (ValueError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 2  # pragma: no cover - argparse requires a subcommand


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

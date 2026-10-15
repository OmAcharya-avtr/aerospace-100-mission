"""Command-line interface. The only module in the package that writes to a stream.

``python -m activityplan --help`` lists the subcommands. Every one of them
renders through :mod:`activityplan.report` and prints the result; none of them
computes anything the library API does not expose.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Sequence

import numpy as np

from . import __version__
from .exact import solve_exact
from .feasibility import evaluate
from .features import feature_matrix, realise, sample_requests
from .greedy import plan_greedy
from .instances import INSTANCE_ORDER, RECOMMENDED_STEP_S, named_instance, sample_instance_spec
from .predictor import (
    DurationEnergyPredictor,
    analytic_predictions,
    coverage,
    mean_interval_width,
)
from .report import (
    format_gap,
    render_declaration,
    render_feasibility,
    render_plan,
    render_rejections,
    render_table,
)
from .robust import PLANNING_MODES, compare_modes, estimate_for_mode
from .timeline import RasterisedProblem
from .warmstart import compare_warm_start, warm_start_from_plan

_EPILOGUE = """
activityplan is research-grade software. It is not flight-qualified, not
certified, and not approved for operational aerospace use. Every constant in
every shipped instance is a declared input of a synthetic benchmark and none was
identified from any spacecraft.

A plan this tool calls feasible is feasible with respect to the declaration you
supplied, on the discretised timeline you chose, under a resource model whose
assumptions are listed in the module docstrings. Change the step and you change
the answer.
"""


def _instance_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--instance",
        default="leo-imaging",
        choices=list(INSTANCE_ORDER),
        help="shipped hand-declared instance to use (default: leo-imaging)",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=None,
        help="grid step in seconds; defaults to the step the instance is documented at",
    )


def _resolve(args: argparse.Namespace) -> RasterisedProblem:
    problem = named_instance(args.instance)
    step = args.step if args.step is not None else RECOMMENDED_STEP_S[args.instance]
    return RasterisedProblem.build(problem, step)


def _cmd_instances(_: argparse.Namespace) -> int:
    rows = []
    for name in INSTANCE_ORDER:
        problem = named_instance(name)
        generated, fixed = problem.baseline_energy_balance_wh()
        rows.append(
            (
                name,
                str(problem.n_activities),
                f"{problem.horizon_s:.0f}",
                f"{RECOMMENDED_STEP_S[name]:.0f}",
                f"{problem.total_priority:.1f}",
                f"{problem.total_activity_energy_wh:.2f}",
                f"{generated - fixed:.2f}",
                f"{problem.resources.usable_wh:.2f}",
                f"{problem.total_data_mbit:.0f}",
                f"{problem.resources.recorder_capacity_mbit:.0f}",
                f"{problem.resources.total_contact_capacity_mbit:.0f}",
            )
        )
    print(
        render_table(
            (
                "instance",
                "acts",
                "horizon/s",
                "step/s",
                "priority",
                "demand/Wh",
                "surplus/Wh",
                "usable/Wh",
                "data/Mbit",
                "recorder",
                "contacts",
            ),
            tuple(rows),
        )
    )
    print()
    print("demand/Wh is the activity energy if every activity is placed.")
    print("surplus/Wh is generated minus fixed load, both closed form over the horizon.")
    return 0


def _cmd_declare(args: argparse.Namespace) -> int:
    rasterised = _resolve(args)
    print(render_declaration(rasterised.problem, rasterised.grid.step_s))
    tightening = rasterised.window_tightening_s()
    rows = tuple(
        (
            activity.name,
            str(int(rasterised.n_occupied[position])),
            str(rasterised.allowed_starts[position].size),
            f"{tightening[position]:.1f}",
        )
        for position, activity in enumerate(rasterised.problem.activities)
    )
    print()
    print("discretisation, per activity")
    print(render_table(("activity", "steps", "admissible starts", "window float lost/s"), rows))
    return 0


def _cmd_plan(args: argparse.Namespace) -> int:
    rasterised = _resolve(args)
    if args.planner == "greedy":
        result = plan_greedy(rasterised)
        plan = result.plan
        print(f"planner             : greedy, {result.simulations} full-horizon simulations")
        print(f"activity order      : {', '.join(result.order)}")
    else:
        result_exact = solve_exact(
            rasterised, time_limit_s=args.time_limit, presolve=not args.no_presolve
        )
        plan = result_exact.plan
        print(
            f"planner             : exact MILP, {result_exact.n_binaries} binaries, "
            f"{result_exact.n_variables} columns, {result_exact.n_constraints} rows"
        )
        print(
            f"solver              : status {result_exact.status}, gap "
            f"{result_exact.gap:.3e}, {result_exact.node_count} nodes, "
            f"{result_exact.solve_s:.3f} s, proven optimal "
            f"{result_exact.proven_optimal}"
        )
    print()
    print(render_plan(plan, rasterised))
    print()
    print(render_feasibility(evaluate(rasterised, plan), rasterised))
    if args.planner == "greedy":
        print()
        print(render_rejections(result))
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    rasterised = _resolve(args)
    greedy = plan_greedy(rasterised)
    exact = solve_exact(rasterised, time_limit_s=args.time_limit)
    greedy_report = evaluate(rasterised, greedy.plan)
    exact_report = evaluate(rasterised, exact.plan)
    rows = (
        (
            "greedy",
            f"{greedy.value:.1f}",
            str(greedy.plan.n_placed),
            f"{greedy.simulations} simulations",
            greedy_report.binding.name,
            f"{greedy_report.binding.normalised:+.6f}",
        ),
        (
            "exact",
            f"{exact.value:.1f}",
            str(exact.plan.n_placed),
            f"{exact.node_count} nodes, {exact.solve_s:.3f} s",
            exact_report.binding.name,
            f"{exact_report.binding.normalised:+.6f}",
        ),
    )
    print(
        render_table(
            ("planner", "objective", "placed", "cost", "binding constraint", "normalised slack"),
            rows,
        )
    )
    print()
    print(f"instance            : {rasterised.problem.name}")
    print(f"total priority      : {rasterised.problem.total_priority:.1f}")
    print(f"exact dual bound    : {exact.dual_bound:.6f}, gap {exact.gap:.3e}")
    print(f"greedy shortfall    : {format_gap(greedy.value, exact.value)}")
    print()
    print("the exact optimum is the optimum of the DISCRETISED problem at this step,")
    print("not of the continuous-time problem. A finer step can only help it.")
    return 0


def _cmd_predict(args: argparse.Namespace) -> int:
    rng = np.random.default_rng(args.seed)
    requests = sample_requests(args.n_train + args.n_test, rng)
    realised = [realise(request, rng) for request in requests]
    features = feature_matrix(list(requests))
    durations = np.array([item.duration_s for item in realised], dtype=float)
    energies = np.array([item.energy_wh for item in realised], dtype=float)
    split = args.n_train
    predictor = DurationEnergyPredictor(
        alpha=args.alpha, mode=args.mode, random_state=args.seed
    ).fit(features[:split], durations[:split], energies[:split])
    duration_interval = predictor.duration.predict(features[split:])
    energy_interval = predictor.energy.predict(features[split:])
    analytic_duration, analytic_energy = analytic_predictions(requests[split:])

    def mae(truth: np.ndarray, prediction: np.ndarray) -> float:
        return float(np.mean(np.abs(truth - prediction)))

    rows = (
        (
            "duration [s]",
            f"{mae(durations[split:], analytic_duration):.4f}",
            f"{mae(durations[split:], duration_interval.point):.4f}",
            f"{coverage(durations[split:], duration_interval):.4f}",
            f"{mean_interval_width(duration_interval):.4f}",
        ),
        (
            "energy [W*h]",
            f"{mae(energies[split:], analytic_energy):.4f}",
            f"{mae(energies[split:], energy_interval.point):.4f}",
            f"{coverage(energies[split:], energy_interval):.4f}",
            f"{mean_interval_width(energy_interval):.4f}",
        ),
    )
    print(
        render_table(
            ("target", "analytic MAE", "learned MAE", "interval coverage", "mean width"), rows
        )
    )
    print()
    print(f"mode                : {args.mode} split conformal, alpha {args.alpha}")
    print(
        f"split               : {predictor.n_train_} proper-train, "
        f"{predictor.n_calibration_} calibration, {args.n_test} test"
    )
    print(f"nominal coverage    : {1.0 - args.alpha:.4f}")
    print()
    print("the learned model is given the analytic prediction as a feature, so it fits the")
    print("correction rather than relearning the model. It also beats the analytic model")
    print("because the analytic model omits two terms the generator contains. See")
    print("MODEL_CARD.md; this is not evidence about a real spacecraft.")
    return 0


def _cmd_robust(args: argparse.Namespace) -> int:
    rng = np.random.default_rng(args.seed)
    requests = sample_requests(args.n_train, rng)
    realised = [realise(request, rng) for request in requests]
    predictor = DurationEnergyPredictor(
        alpha=args.alpha, mode="normalised", random_state=args.seed
    ).fit(
        feature_matrix(list(requests)),
        np.array([item.duration_s for item in realised], dtype=float),
        np.array([item.energy_wh for item in realised], dtype=float),
    )
    tally: dict[str, list[float]] = {mode: [0.0, 0.0, 0.0] for mode in PLANNING_MODES}
    for offset in range(args.n_instances):
        spec = sample_instance_spec(args.instance_seed + offset)
        outcomes = compare_modes(
            spec, np.random.default_rng(args.instance_seed + 10_000 + offset), predictor=predictor
        )
        for outcome in outcomes:
            tally[outcome.mode][0] += 1.0 if outcome.feasible else 0.0
            tally[outcome.mode][1] += outcome.realised_value
            tally[outcome.mode][2] += (
                outcome.planned_value if math.isfinite(outcome.planned_value) else 0.0
            )
    rows = tuple(
        (
            mode,
            f"{values[0]:.0f}/{args.n_instances}",
            f"{values[0] / args.n_instances:.4f}",
            f"{values[2]:.1f}",
            f"{values[1]:.1f}",
        )
        for mode, values in tally.items()
    )
    print(
        render_table(
            ("mode", "feasible", "feasible rate", "planned priority", "realised priority"),
            tuple(rows),
        )
    )
    print()
    print(f"instances           : {args.n_instances} from seed {args.instance_seed}")
    print("a realised plan that violates any declared constraint scores zero, not partial")
    print("credit; see activityplan.robust for why.")
    return 0


def _cmd_warmstart(args: argparse.Namespace) -> int:
    rng = np.random.default_rng(args.seed)
    requests = sample_requests(args.n_train, rng)
    realised = [realise(request, rng) for request in requests]
    predictor = DurationEnergyPredictor(
        alpha=0.1, mode="normalised", random_state=args.seed
    ).fit(
        feature_matrix(list(requests)),
        np.array([item.duration_s for item in realised], dtype=float),
        np.array([item.energy_wh for item in realised], dtype=float),
    )
    rows = []
    for offset in range(args.n_instances):
        spec = sample_instance_spec(args.instance_seed + offset)
        estimate = estimate_for_mode(spec, "ml-point", predictor=predictor)
        target = RasterisedProblem.build(
            spec.problem_from(estimate.durations_s, estimate.energies_wh, name_suffix="/ml-point"),
            spec.step_s,
        )
        candidate = plan_greedy(target).plan
        warm_start = warm_start_from_plan(target, candidate, source="greedy-ml")
        comparison = compare_warm_start(
            target, warm_start, presolve=not args.no_presolve, time_limit_s=args.time_limit
        )
        rows.append(
            (
                spec.name,
                f"{warm_start.candidate_value:.1f}",
                str(warm_start.verified),
                f"{comparison.cold.value:.1f}",
                f"{comparison.cold.node_count}",
                f"{comparison.cold.solve_s:.3f}",
                "-" if comparison.warm is None else f"{comparison.warm.node_count}",
                "-" if comparison.warm is None else f"{comparison.warm.solve_s:.3f}",
                str(comparison.optimum_preserved),
            )
        )
    print(
        render_table(
            (
                "instance",
                "warm value",
                "verified",
                "optimum",
                "cold nodes",
                "cold s",
                "warm nodes",
                "warm s",
                "optimum kept",
            ),
            tuple(rows),
        )
    )
    print()
    print(f"presolve            : {not args.no_presolve}")
    print("scipy.optimize.milp exposes no MIP start, so the warm start is a primal-bound")
    print("cut. With presolve on these instances are solved before branching and the node")
    print("counts measure presolve, not search. See activityplan.warmstart.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """The argument parser, exposed so that tests can exercise it without a shell."""
    parser = argparse.ArgumentParser(
        prog="python -m activityplan",
        description=(
            "Onboard activity planning with an auditable feasibility report: a timeline, "
            "the binding constraint, and the slack on every other constraint."
        ),
        epilog=_EPILOGUE,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"activityplan {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    instances = subparsers.add_parser(
        "instances", help="list the shipped hand-declared instances"
    )
    instances.set_defaults(handler=_cmd_instances)

    declare = subparsers.add_parser(
        "declare", help="print one instance's declaration and its discretisation"
    )
    _instance_argument(declare)
    declare.set_defaults(handler=_cmd_declare)

    plan = subparsers.add_parser(
        "plan", help="plan one instance and print the timeline and feasibility report"
    )
    _instance_argument(plan)
    plan.add_argument("--planner", choices=("greedy", "exact"), default="greedy")
    plan.add_argument("--time-limit", type=float, default=60.0, help="MILP time limit in seconds")
    plan.add_argument("--no-presolve", action="store_true", help="disable HiGHS presolve")
    plan.set_defaults(handler=_cmd_plan)

    compare = subparsers.add_parser("compare", help="greedy against the exact optimum")
    _instance_argument(compare)
    compare.add_argument("--time-limit", type=float, default=60.0)
    compare.set_defaults(handler=_cmd_compare)

    predict = subparsers.add_parser(
        "predict", help="learned duration and energy predictor against the analytic model"
    )
    predict.add_argument("--seed", type=int, default=61)
    predict.add_argument("--n-train", type=int, default=1000)
    predict.add_argument("--n-test", type=int, default=400)
    predict.add_argument("--alpha", type=float, default=0.1)
    predict.add_argument("--mode", choices=("absolute", "normalised"), default="normalised")
    predict.set_defaults(handler=_cmd_predict)

    robust = subparsers.add_parser(
        "robust", help="the four planning modes under realised durations"
    )
    robust.add_argument("--seed", type=int, default=61)
    robust.add_argument("--n-train", type=int, default=1000)
    robust.add_argument("--alpha", type=float, default=0.1)
    robust.add_argument("--n-instances", type=int, default=8)
    robust.add_argument("--instance-seed", type=int, default=9000)
    robust.set_defaults(handler=_cmd_robust)

    warmstart = subparsers.add_parser(
        "warmstart", help="does a learned warm start help the exact solver"
    )
    warmstart.add_argument("--seed", type=int, default=61)
    warmstart.add_argument("--n-train", type=int, default=1000)
    warmstart.add_argument("--n-instances", type=int, default=6)
    warmstart.add_argument("--instance-seed", type=int, default=9000)
    warmstart.add_argument("--time-limit", type=float, default=60.0)
    warmstart.add_argument("--no-presolve", action="store_true")
    warmstart.set_defaults(handler=_cmd_warmstart)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns the process exit status."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    sys.exit(main())

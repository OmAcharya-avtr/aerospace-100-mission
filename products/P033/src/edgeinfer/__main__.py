"""Command-line interface: ``python -m edgeinfer``.

Five subcommands, each printing numbers with their units, their measurement
method and their repeat count:

``analyse``
    Analytic estimate for one of the built-in graphs, or for an ONNX file.
``profile``
    Measure a built-in or supplied graph with ``onnxruntime`` and print the
    latency profile, tail first.
``check``
    Analytic estimate plus measured profile checked against a declared budget,
    printing the pass/fail table with uncertainties.
``feasibility``
    Check a declared budget for self-consistency without touching a model.
``env``
    Print the environment and the clock's measured properties.

Every path that prints a latency prints the environment with it. There is no
flag that suppresses that.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from edgeinfer import __version__
from edgeinfer.analytic import analytic_estimate
from edgeinfer.backends import OnnxRuntimeBackend
from edgeinfer.budget import Budget, build_report
from edgeinfer.dataset import hand_counted_cnn, hand_counted_mlp, random_cnn, random_mlp
from edgeinfer.environment import environment_record
from edgeinfer.graph import GraphError
from edgeinfer.harness import benchmark
from edgeinfer.onnx_io import OnnxModel, parse_model
from edgeinfer.report import CLOUD_ENVIRONMENT_NOTE
from edgeinfer.roofline import DeviceModel
from edgeinfer.uncertainty import timer_overhead_s

BUILTIN_GRAPHS = ("hand_mlp", "hand_cnn", "mlp", "cnn")


def _builtin(name: str) -> OnnxModel:
    rng = np.random.default_rng(20260401)
    if name == "hand_mlp":
        return hand_counted_mlp()
    if name == "hand_cnn":
        return hand_counted_cnn()
    if name == "mlp":
        return random_mlp(rng, "cli_mlp", n_in=64, widths=(128, 64), n_out=8)
    if name == "cnn":
        return random_cnn(rng, "cli_cnn", spatial=24, channels=(8, 16), kernel=3, n_out=8)
    raise ValueError(f"unknown built-in graph {name!r}; choose from {BUILTIN_GRAPHS}")


def _load(args: argparse.Namespace) -> OnnxModel:
    if args.onnx:
        path = Path(args.onnx)
        if not path.is_file():
            raise FileNotFoundError(f"no such ONNX file: {path}")
        graph, arrays = parse_model(path.read_bytes())
        graph.infer_shapes()
        return OnnxModel(graph=graph, model_bytes=path.read_bytes(), initializer_arrays=arrays)
    return _builtin(args.graph)


def _device(args: argparse.Namespace) -> DeviceModel:
    return DeviceModel(
        name=args.device_name,
        peak_flops=args.peak_gflops * 1e9,
        peak_bandwidth_bytes_s=args.peak_bandwidth_gbs * 1e9,
        overhead_per_node_s=args.overhead_per_node_us * 1e-6,
        fixed_overhead_s=args.fixed_overhead_us * 1e-6,
        source=(
            "declared on the command line; not measured and not a datasheet figure"
        ),
    )


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--graph",
        choices=BUILTIN_GRAPHS,
        default="mlp",
        help="built-in graph to use (default: mlp)",
    )
    group.add_argument("--onnx", help="path to an ONNX file with static shapes")


def _add_device_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device-name", default="declared-target", help="device model name")
    parser.add_argument(
        "--peak-gflops", type=float, default=2.0, help="declared peak compute [GFLOP/s]"
    )
    parser.add_argument(
        "--peak-bandwidth-gbs",
        type=float,
        default=8.0,
        help="declared peak memory bandwidth [GB/s]",
    )
    parser.add_argument(
        "--overhead-per-node-us",
        type=float,
        default=0.0,
        help="declared per-node dispatch overhead [us]",
    )
    parser.add_argument(
        "--fixed-overhead-us",
        type=float,
        default=0.0,
        help="declared constant per-inference overhead [us]",
    )


def _add_budget_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--budget-latency-ms", type=float, default=1.0, help="worst-case latency ceiling [ms]"
    )
    parser.add_argument(
        "--budget-median-ms", type=float, default=None, help="median latency ceiling [ms]"
    )
    parser.add_argument(
        "--budget-memory-kb", type=float, default=512.0, help="peak memory ceiling [kB]"
    )
    parser.add_argument(
        "--budget-power-w", type=float, default=None, help="power ceiling [W], never measured"
    )
    parser.add_argument(
        "--budget-duty-cycle", type=float, default=None, help="duty cycle [dimensionless]"
    )
    parser.add_argument(
        "--budget-period-ms", type=float, default=None, help="control period [ms]"
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.99,
        help="latency quantile the worst-case ceiling applies to (default 0.99)",
    )


def _budget(args: argparse.Namespace) -> Budget:
    return Budget(
        name=getattr(args, "budget_name", "cli budget"),
        latency_s=args.budget_latency_ms * 1e-3,
        peak_memory_bytes=int(args.budget_memory_kb * 1024),
        median_latency_s=(
            None if args.budget_median_ms is None else args.budget_median_ms * 1e-3
        ),
        power_w=args.budget_power_w,
        duty_cycle=args.budget_duty_cycle,
        period_s=None if args.budget_period_ms is None else args.budget_period_ms * 1e-3,
        worst_case_quantile=args.quantile,
    )


def _print_env(note: str = CLOUD_ENVIRONMENT_NOTE) -> None:
    print(environment_record(shared_host=True, note="").one_line())
    print(note)


def cmd_analyse(args: argparse.Namespace) -> int:
    """Print the analytic estimate for a graph."""
    model = _load(args)
    estimate = analytic_estimate(model.graph, _device(args))
    print("\n".join(estimate.summary_lines()))
    print("\nper-node roofline breakdown (time [us], bound only, no overhead):")
    print(f"{'node':<16}{'op':<22}{'flops':>12}{'bytes':>10}{'I [F/B]':>10}{'t [us]':>10}  bound")
    for node in estimate.nodes:
        bound = "memory" if node.memory_bound else "compute"
        print(
            f"{node.name:<16}{node.op_type:<22}{node.cost.flops:>12}"
            f"{node.cost.bytes_total:>10}{node.cost.arithmetic_intensity:>10.3g}"
            f"{node.roofline_s * 1e6:>10.4f}  {bound}"
        )
    print("\nmethod: analytic, no execution. Roofline bound from Williams, Waterman &")
    print("Patterson 2009 (Comm. ACM 52(4), 65-76); device peaks are declared, not measured.")
    return 0


def cmd_profile(args: argparse.Namespace) -> int:
    """Measure a graph with onnxruntime and print the latency profile."""
    model = _load(args)
    backend = OnnxRuntimeBackend(model.model_bytes, model.input_feed(args.seed))
    backend.prepare()
    try:
        profile = benchmark(
            backend.infer,
            label=f"{model.graph.name} via onnxruntime",
            repeats=args.repeats,
            warmup=args.warmup,
            environment_note=CLOUD_ENVIRONMENT_NOTE,
            extra={"backend": backend.kind},
        )
    finally:
        backend.close()
    print("\n".join(profile.summary_lines()))
    for statistic in ("p50", "p99", "mean"):
        print()
        print("\n".join(profile.uncertainty(statistic).summary_lines()))
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    """Check a graph against a declared budget, analytic and measured."""
    model = _load(args)
    budget = _budget(args)
    device = _device(args)
    estimate = analytic_estimate(model.graph, device)

    feasibility = budget.feasibility()
    print("\n".join(budget.summary_lines()))
    print()
    if not feasibility.feasible:
        print("budget is INFEASIBLE BY CONSTRUCTION:")
        for reason in feasibility.reasons:
            print(f"  - {reason}")
        print("\nno model can satisfy it; not measuring.")
        return 2

    backend = OnnxRuntimeBackend(model.model_bytes, model.input_feed(args.seed))
    backend.prepare()
    try:
        profile = benchmark(
            backend.infer,
            label=model.graph.name,
            repeats=args.repeats,
            warmup=args.warmup,
            environment_note=CLOUD_ENVIRONMENT_NOTE,
        )
    finally:
        backend.close()

    tail = profile.uncertainty(f"p{budget.worst_case_quantile * 100:g}")
    median = profile.uncertainty("p50")
    report = build_report(
        budget,
        candidate=model.graph.name,
        environment=profile.environment.one_line(),
        worst_case_latency_s=tail.value,
        worst_case_uncertainty_s=tail.combined_s,
        median_latency_s=median.value,
        median_uncertainty_s=median.combined_s,
        peak_memory_bytes=float(estimate.peak_memory_bytes),
        peak_memory_uncertainty_bytes=None,
        declared_power_w=args.budget_power_w,
        latency_method=profile.method_line(),
        memory_method=(
            "analytic liveness analysis over the graph (Aho et al. 2006 section 8.4); "
            "a lower bound, no runtime arena or scratch buffer included"
        ),
        notes=(
            f"analytic latency estimate {estimate.latency_s * 1e6:.3f} us "
            f"on device {device.name} ({device.source})",
            f"measured p99/p50 tail ratio {profile.tail_ratio:.3f}",
            CLOUD_ENVIRONMENT_NOTE,
        ),
    )
    print("\n".join(report.summary_lines()))
    return 0 if report.overall.is_pass else 1


def cmd_feasibility(args: argparse.Namespace) -> int:
    """Check a declared budget for self-consistency."""
    budget = _budget(args)
    print("\n".join(budget.summary_lines()))
    result = budget.feasibility()
    print()
    if result.feasible:
        print("feasibility: the declared limits are self-consistent.")
        print("(this says nothing about whether any model meets them)")
        return 0
    print("feasibility: INFEASIBLE BY CONSTRUCTION")
    for reason in result.reasons:
        print(f"  - {reason}")
    return 2


def cmd_env(args: argparse.Namespace) -> int:
    """Print the environment and the measured timer properties."""
    record = environment_record(shared_host=True, note="")
    for key, value in record.as_dict().items():
        print(f"{key:<24}: {value}")
    bias = timer_overhead_s(args.timer_samples)
    print(f"{'timer pair cost (median)':<24}: {bias * 1e9:.1f} ns "
          f"over {args.timer_samples} pairs")
    print()
    print(CLOUD_ENVIRONMENT_NOTE)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m edgeinfer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Declared latency/memory/power budgets for edge inference, with an\n"
            "analytic roofline estimate and a measured profile side by side.\n"
            "\n"
            "Validation level: 3, hardware-pending. Every number this tool prints\n"
            "comes from the host it runs on, not from an edge target. Research-grade\n"
            "software; it is not flight-qualified and carries no certification."
        ),
        epilog=(
            "Numbers printed by this tool carry their measurement method and\n"
            "repeat count."
        ),
    )
    parser.add_argument("--version", action="version", version=f"edgeinfer {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyse = subparsers.add_parser("analyse", help="analytic estimate, nothing executed")
    _add_model_args(analyse)
    _add_device_args(analyse)
    analyse.set_defaults(func=cmd_analyse)

    profile = subparsers.add_parser("profile", help="measure with onnxruntime")
    _add_model_args(profile)
    profile.add_argument("--repeats", type=int, default=200, help="retained repeats")
    profile.add_argument("--warmup", type=int, default=10, help="discarded warm-up calls")
    profile.add_argument("--seed", type=int, default=0, help="input-feed seed")
    profile.set_defaults(func=cmd_profile)

    check = subparsers.add_parser("check", help="check against a declared budget")
    _add_model_args(check)
    _add_device_args(check)
    _add_budget_args(check)
    check.add_argument("--budget-name", default="cli budget", help="budget name")
    check.add_argument("--repeats", type=int, default=200, help="retained repeats")
    check.add_argument("--warmup", type=int, default=10, help="discarded warm-up calls")
    check.add_argument("--seed", type=int, default=0, help="input-feed seed")
    check.set_defaults(func=cmd_check)

    feasibility = subparsers.add_parser(
        "feasibility", help="check a budget for self-consistency"
    )
    _add_budget_args(feasibility)
    feasibility.add_argument("--budget-name", default="cli budget", help="budget name")
    feasibility.set_defaults(func=cmd_feasibility)

    env = subparsers.add_parser("env", help="print the environment and timer properties")
    env.add_argument("--timer-samples", type=int, default=2000, help="timer-pair repeats")
    env.set_defaults(func=cmd_env)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit status.

    ``0`` success or budget pass, ``1`` budget fail, ``2`` infeasible budget or
    an invalid input, ``3`` an unsupported operator.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except GraphError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    except (ValueError, FileNotFoundError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess in tests
    raise SystemExit(main())

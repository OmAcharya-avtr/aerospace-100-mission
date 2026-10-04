"""Command-line interface: ``python -m rtclock <subcommand>``.

Subcommands
    ``resolution``  measure and report the host clock's resolution
    ``bound``       rate-monotonic utilization bound for n tasks
    ``analyse``     schedulability of a task set given on the command line
    ``percentile``  exact percentiles of a list of samples
    ``drift``       simulated loop drift for an injected clock skew

Every subcommand writes to stdout and exits 0 on success, 1 on a usage or
value error with the message on stderr. No subcommand writes a file.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Sequence

from . import __version__
from .histogram import percentile
from .loop import FixedRateLoop, absolute_mode_drift, relative_mode_drift
from .schedulability import (
    edf_test,
    hyperbolic_bound_test,
    priority_ceiling_blocking,
    response_time_analysis,
    rm_utilization_bound,
    rm_utilization_test,
)
from .taskset import PeriodicTask, TaskSet
from .timebase import SkewedSimulatedTimebase, measure_clock_resolution

__all__ = ["build_parser", "main"]


def _parse_task(spec: str) -> PeriodicTask:
    """Parse ``name:period_s:wcet_s[:deadline_s]`` into a PeriodicTask."""
    parts = spec.split(":")
    if len(parts) not in (3, 4):
        raise ValueError(
            f"task spec must be name:period_s:wcet_s[:deadline_s], got {spec!r}"
        )
    name = parts[0]
    try:
        period = float(parts[1])
        wcet = float(parts[2])
        deadline = float(parts[3]) if len(parts) == 4 else None
    except ValueError as exc:
        raise ValueError(f"non-numeric field in task spec {spec!r}: {exc}") from exc
    return PeriodicTask(name=name, period_s=period, wcet_s=wcet, deadline_s=deadline)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for ``python -m rtclock``."""
    parser = argparse.ArgumentParser(
        prog="rtclock",
        description=(
            "Real-time timing arithmetic and schedulability analysis. Analyses timing; "
            "does not deliver it. Research-grade, not flight-qualified."
        ),
        epilog="Durations are in seconds unless a flag says otherwise.",
    )
    parser.add_argument("--version", action="version", version=f"rtclock {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="SUBCOMMAND")

    p_res = sub.add_parser("resolution", help="measure the host clock resolution")
    p_res.add_argument("--samples", type=int, default=20_000, help="clock reads (default 20000)")
    p_res.add_argument(
        "--clock",
        choices=("monotonic", "perf_counter"),
        default="monotonic",
        help="clock to measure (default monotonic)",
    )

    p_bound = sub.add_parser("bound", help="rate-monotonic utilization bound n(2^(1/n)-1)")
    p_bound.add_argument("n", type=int, nargs="+", help="task counts, each >= 1")

    p_an = sub.add_parser("analyse", help="schedulability of a task set")
    p_an.add_argument(
        "task",
        nargs="+",
        help="task as name:period_s:wcet_s[:deadline_s]",
    )
    p_an.add_argument(
        "--priority",
        choices=("rm", "dm"),
        default="rm",
        help="priority assignment: rate- or deadline-monotonic (default rm)",
    )
    p_an.add_argument(
        "--critical-section",
        action="append",
        default=[],
        metavar="TASK:SEM:DURATION_S",
        help="a critical section for priority-ceiling blocking; repeatable",
    )

    p_pct = sub.add_parser("percentile", help="exact percentiles of a sample list")
    p_pct.add_argument("samples", type=float, nargs="+", help="sample values")
    p_pct.add_argument(
        "--p",
        type=float,
        action="append",
        default=None,
        help="percentile in [0,100]; repeatable (default 50 90 99 100)",
    )
    p_pct.add_argument(
        "--method",
        choices=("nearest_rank", "linear"),
        default="nearest_rank",
        help="percentile definition (default nearest_rank)",
    )

    p_dr = sub.add_parser("drift", help="simulated loop drift for an injected clock skew")
    p_dr.add_argument("--period", type=float, required=True, help="loop period, s")
    p_dr.add_argument("--iterations", type=int, default=1000, help="iterations (default 1000)")
    p_dr.add_argument("--skew-ppm", type=float, default=0.0, help="sleep rate error, ppm")
    p_dr.add_argument("--wake-delay", type=float, default=0.0, help="per-wait latency, s")
    p_dr.add_argument(
        "--mode",
        choices=("absolute", "relative"),
        default="absolute",
        help="scheduling mode (default absolute)",
    )
    return parser


def _cmd_resolution(args: argparse.Namespace) -> int:
    res = measure_clock_resolution(samples=args.samples, clock_name=args.clock)
    print(f"clock                            {res.clock_name}")
    print(f"method                           {res.method}")
    print(f"samples                          {res.samples}")
    print(f"advertised resolution       [s]  {res.advertised_s:.12e}")
    print(f"measured observable tick    [s]  {res.measured_tick_s:.12e}")
    print(f"median per-read delta       [s]  {res.median_call_delta_s:.12e}")
    print(f"identical-reading fraction  [-]  {res.zero_delta_fraction:.6f}")
    print(f"worst-case duration error   [s]  {res.worst_case_duration_error_s:.12e}")
    print(f"standard duration uncert.   [s]  {res.standard_duration_uncertainty_s:.12e}")
    print("note: the measured tick is an upper bound on the hardware granularity;")
    print("      an identical-reading fraction of 0 means the read loop, not the")
    print("      clock, was the limiting factor.")
    return 0


def _cmd_bound(args: argparse.Namespace) -> int:
    print(f"{'n':>5}  {'n(2^(1/n)-1)':>18}  {'excess over ln2':>18}")
    for n in args.n:
        bound = rm_utilization_bound(n)
        print(f"{n:>5}  {bound:>18.12f}  {bound - math.log(2.0):>18.12f}")
    print(f"{'inf':>5}  {math.log(2.0):>18.12f}  {0.0:>18.12f}   (ln 2, Liu & Layland 1973)")
    return 0


def _cmd_analyse(args: argparse.Namespace) -> int:
    tasks = [_parse_task(spec) for spec in args.task]
    ts = TaskSet(tasks)
    ts = ts.rate_monotonic() if args.priority == "rm" else ts.deadline_monotonic()

    sections: dict[str, dict[str, float]] = {}
    for spec in args.critical_section:
        parts = spec.split(":")
        if len(parts) != 3:
            raise ValueError(f"--critical-section must be TASK:SEM:DURATION_S, got {spec!r}")
        sections.setdefault(parts[0], {})[parts[1]] = float(parts[2])
    blocking = priority_ceiling_blocking(ts, sections) if sections else {}

    assignment = "rate-monotonic" if args.priority == "rm" else "deadline-monotonic"
    print(f"tasks                 {len(ts)}")
    print(f"priority assignment   {assignment}")
    print(f"total utilization     {ts.total_utilization:.12f}")
    print(f"total density         {ts.total_density:.12f}")
    print("")
    if ts.all_implicit_deadlines:
        for result in (rm_utilization_test(ts), hyperbolic_bound_test(ts)):
            print(f"[{result.strength:>10}] {result.test}: {result.detail}")
    else:
        print("[  skipped] utilization bounds: set has D < T, bounds not valid")
    edf = edf_test(ts)
    print(f"[{edf.strength:>10}] {edf.test}: {edf.detail}")
    print("")
    print("response-time analysis (exact for the model; see module docstring)")
    header = f"{'task':>10} {'prio':>5} {'C [s]':>12} {'B [s]':>12} {'R [s]':>12} {'D [s]':>12}"
    print(f"{header} {'slack [s]':>12}  iters  ok")
    rts = response_time_analysis(ts, blocking_s=blocking or None)
    by_name = {t.name: t for t in ts}
    worst = True
    for rt in rts:
        t = by_name[rt.name]
        print(
            f"{rt.name:>10} {t.priority:>5} {t.wcet_s:>12.9f} {rt.blocking_s:>12.9f} "
            f"{rt.response_s:>12.9f} {rt.deadline_s:>12.9f} {rt.slack_s:>12.9f} "
            f"{rt.iterations:>6}  {'yes' if rt.meets_deadline else 'NO'}"
        )
        worst = worst and rt.meets_deadline
    print("")
    print(f"verdict: {'all deadlines met' if worst else 'at least one deadline missed'}")
    return 0


def _cmd_percentile(args: argparse.Namespace) -> int:
    ps = args.p if args.p else [50.0, 90.0, 99.0, 100.0]
    print(f"samples   {len(args.samples)}")
    print(f"method    {args.method}")
    for p in ps:
        print(f"p{p:<8g} {percentile(args.samples, p, args.method):.12g}")
    return 0


def _cmd_drift(args: argparse.Namespace) -> int:
    tb = SkewedSimulatedTimebase(skew_ppm=args.skew_ppm, wake_delay_s=args.wake_delay)
    loop = FixedRateLoop(period_s=args.period, timebase=tb, mode=args.mode)
    report = loop.run(iterations=args.iterations)
    closed = relative_mode_drift if args.mode == "relative" else absolute_mode_drift
    last = args.iterations - 1
    print(f"mode                       {args.mode}")
    print(f"period                [s]  {args.period:.12g}")
    print(f"iterations            [-]  {args.iterations}")
    print(f"injected skew       [ppm]  {args.skew_ppm:.12g}")
    print(f"injected wake delay   [s]  {args.wake_delay:.12g}")
    print(f"final drift           [s]  {report.final_drift_s:.12e}")
    if args.iterations > 0:
        expected = closed(last, args.period, args.skew_ppm, args.wake_delay)
        print(f"closed-form drift     [s]  {expected:.12e}")
        print(f"difference            [s]  {report.final_drift_s - expected:.12e}")
    print(f"max |drift|           [s]  {report.max_abs_drift_s:.12e}")
    if args.iterations >= 2:
        print(f"drift slope    [s/iter]  {report.drift_slope_s_per_iteration():.12e}")
    print("note: simulated timebase, not a measurement of this machine.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns the process exit status."""
    parser = build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "resolution": _cmd_resolution,
        "bound": _cmd_bound,
        "analyse": _cmd_analyse,
        "percentile": _cmd_percentile,
        "drift": _cmd_drift,
    }
    try:
        return handlers[args.command](args)
    except (ValueError, TypeError) as exc:
        print(f"rtclock: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

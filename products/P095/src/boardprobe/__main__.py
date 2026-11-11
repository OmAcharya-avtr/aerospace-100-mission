"""``python -m boardprobe`` -- the command line, which goes through the same choke point.

Six subcommands. Every one of them that prints a figure prints the certificate
headline above it, because the renderers in ``boardprobe.report`` take
certificates and not bare values.

    channels     the declared channel table, with the verified-on-hardware flag
    sample       one sample from one backend across the channels it can serve
    benchmark    the harness, with the overhead characterisation beside it
    replay       replay a trace file and summarise it
    provenance   the lattice, the label rules, and what this build can label
    dry-run      the dry-run walk and the evidence that it stayed dry

Exit statuses: 0 on success, 2 on a bad argument (argparse's own), and 1 on a
run that could not produce a result -- a missing trace, an unreachable device
where one was required. A refusal to label a figure is NOT an error: the
refusal is the product working, and it is printed and exits 0.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .certify import LabelViolation, certify, label_for
from .dryrun import dry_run
from .harness import HarnessError, benchmark, characterise_overhead, reference_payload
from .orin_nano import ORIN_NANO
from .platform_attest import attest_platform
from .provenance import ORDER
from .readings import attribute
from .report import (
    format_table,
    render_benchmark,
    render_certificate,
    render_channels,
    render_label_rules,
    render_lattice,
    render_overhead,
    render_readings,
    render_statistics,
)
from .sources import LiveSource, backend_names, make_source, read_available
from .statistics import summarise
from .sysio import DeviceUnreachableError
from .trace import Trace, TraceError

#: Fixtures shipped with the repository, looked up relative to this file.
#: Named rather than globbed so a missing fixture is an actionable error.
IDLE_TRACE = "orin_nano_idle_simulated_trace.json"
REPLAY_TRACE = "replay_path_recorded_trace.json"


def fixture_path(name: str) -> Path:
    """Locate a committed fixture relative to this package, never to the cwd."""
    candidate = Path(__file__).resolve().parents[2] / "fixtures" / name
    if not candidate.is_file():
        raise TraceError(
            f"the fixture {name!r} is not present beside this package; expected it at "
            f"fixtures/{name} relative to the repository root. Pass --trace with a path "
            "to a trace document instead."
        )
    return candidate


def _load_trace(argument: str | None, default: str) -> Trace:
    return Trace.from_file(Path(argument) if argument else fixture_path(default))


def cmd_channels(args: argparse.Namespace) -> int:
    """Print the declared channel table and the counts that qualify it."""
    print(f"board: {ORIN_NANO.model}")
    print(f"channels declared: {len(ORIN_NANO.channels)}")
    print(
        f"channels verified on hardware: {ORIN_NANO.verified_count} of "
        f"{len(ORIN_NANO.channels)}"
    )
    print("")
    print(render_channels(ORIN_NANO))
    print("")
    print("EVERY LOCATOR, RAW UNIT AND SCALE ABOVE IS DECLARED FROM DOCUMENTATION AND")
    print("IS UNVERIFIED ON HARDWARE. No NVIDIA Jetson Orin Nano was reachable from the")
    print("environment this package was built in; see src/boardprobe/orin_nano.py for")
    print("the reference behind each line.")
    if args.references:
        print("")
        print(
            format_table(
                ["channel", "reference"],
                [[c.name, c.reference] for c in ORIN_NANO.channels],
            )
        )
    return 0


def cmd_sample(args: argparse.Namespace) -> int:
    """Take one sample from one backend and summarise one channel of it."""
    trace = _load_trace(args.trace, IDLE_TRACE) if args.backend == "recorded" else None
    source = make_source(args.backend, ORIN_NANO, seed=args.seed, trace=trace)
    requested = tuple(args.channel) if args.channel else None
    readings, refusals = read_available(source, requested)
    print(f"backend: {args.backend}  source_id: {source.source_id}")
    print(f"backend provenance: {source.provenance}")
    if trace is not None:
        print(f"trace origin: {trace.origin}")
    print("")
    if readings:
        print(render_readings(readings))
    else:
        print("no channel could be served by this backend.")
    if refusals:
        print("")
        print(f"channels refused ({len(refusals)} of {len(refusals) + len(readings)}):")
        print(
            format_table(
                ["channel", "reason"], [[name, reason[:100]] for name, reason in refusals]
            )
        )
    if not readings:
        return 1
    attribution = attribute(readings)
    certificate = certify(
        f"sample from the {args.backend} backend",
        attribution,
        label_for(attribution.provenance),
        attest_platform(),
    )
    print("")
    print(render_certificate(certificate))
    by_channel = [r for r in readings if r.channel == readings[0].channel]
    if len(by_channel) > 1:
        print("")
        print(render_statistics(summarise(by_channel)))
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Run the harness and print its result and its own overhead."""
    clock = LiveSource(ORIN_NANO)
    environment = None
    trace = None
    if args.environment != "none":
        trace = _load_trace(args.trace, IDLE_TRACE) if args.environment == "recorded" else None
        environment = make_source(args.environment, ORIN_NANO, seed=args.seed, trace=trace)
    payload = reference_payload(args.work)
    try:
        result = benchmark(
            payload,
            f"reference_payload({args.work})",
            clock,
            repeats=args.repeats,
            warmup=args.warmup,
            environment=environment,
        )
    except (DeviceUnreachableError, HarnessError) as error:
        print(f"the benchmark could not run: {error}", file=sys.stderr)
        return 1
    print(render_benchmark(result))
    print("")
    print("harness overhead, characterised on this host")
    print(render_overhead(characterise_overhead(clock, repeats=args.overhead_repeats)))
    print("")
    print("Every figure above is a property of the host this ran on and NOT of an")
    print("NVIDIA Jetson Orin Nano. A MEASURED label means a device responded; the")
    print("platform line above says which device, and it is not the declared target.")
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    """Replay a trace and summarise one of its channels."""
    trace = _load_trace(args.trace, REPLAY_TRACE)
    source = make_source("recorded", ORIN_NANO, trace=trace)
    print(f"trace channels      : {len(trace.channels)}")
    print(f"trace samples       : {trace.sample_count}")
    print(f"content provenance  : {trace.content_provenance}")
    print(f"replay provenance   : {source.provenance}  "
          f"(meet of recorded and the document's content provenance)")
    print(f"trace origin        : {trace.origin}")
    print("")
    channel = args.channel or trace.channels[0]
    readings = [source.read(channel) for _ in range(args.samples)]
    print(render_statistics(summarise(readings)))
    attribution = attribute(readings)
    certificate = certify(
        f"replay of {channel!r}",
        attribution,
        label_for(attribution.provenance),
        attest_platform(),
    )
    print("")
    print(render_certificate(certificate))
    return 0


def cmd_provenance(args: argparse.Namespace) -> int:
    """Print the lattice, the label rules, and what this build can and cannot label."""
    platform = attest_platform()
    print("the provenance lattice: meet is the greatest lower bound")
    print("")
    print(render_lattice())
    print("")
    print("the label rules, enforced at one choke point (boardprobe.certify.certify)")
    print("")
    print(render_label_rules())
    print("")
    print(f"identity of meet   : {ORDER[-1].value} (meet(identity, x) == x)")
    print(f"absorbing element  : {ORDER[0].value} (meet(absorbing, x) == absorbing)")
    print("meet over an empty collection is REFUSED, not taken as the identity")
    print("")
    print("platform attestation")
    print(f"  declared target  : {platform.target_model}")
    print(f"  observed         : {platform.describe()}")
    print(f"  matches target   : {'yes' if platform.matches_target else 'NO'}")
    print("")
    rows = []
    for provenance in reversed(ORDER):
        label = label_for(provenance)
        reachable = "yes" if provenance is not ORDER[-1] else "clock channel only"
        rows.append([provenance.value, label.value, reachable])
    print(
        format_table(
            ["provenance", "label", "reachable on this host"],
            rows,
        )
    )
    print("")
    print("A MEASURED label on this host would read MEASURED OFF-TARGET and name both")
    print("models, because the platform attestation does not match the declared target.")
    print("No figure in this repository was measured on an NVIDIA Jetson Orin Nano.")
    print(f"validation level: {args.level}")
    return 0


def cmd_dry_run(args: argparse.Namespace) -> int:
    """Walk every code path, touch no board, and report the evidence."""
    trace = _load_trace(args.trace, IDLE_TRACE)
    watch = Path(args.watch) if args.watch else Path(__file__).resolve().parents[2]
    report = dry_run(ORIN_NANO, trace, seed=args.seed, watch=watch)
    print(f"code paths exercised                  : {report.path_count}")
    print(f"device accesses against the real root : {len(report.device_accesses)}")
    print(
        f"attempts against a root that does not exist : "
        f"{len(report.absent_root_accesses)} (the live backend's refusal path; the "
        f"root {report.absent_root!r} was asserted absent first)"
    )
    print(f"files under {report.watched!r} before      : {len(report.files_before)}")
    print(f"files under {report.watched!r} after       : {len(report.files_after)}")
    print(f"files added                           : {list(report.files_added) or 'none'}")
    print(f"side-effect free                      : "
          f"{'yes' if report.side_effect_free else 'NO'}")
    print("")
    print(format_table(["#", "code path exercised"],
                       [[str(i + 1), name] for i, name in enumerate(report.paths)]))
    print("")
    print(f"labels granted ({len(report.labels_granted)}): "
          f"{'; '.join(report.labels_granted)}")
    print(f"labels refused ({len(report.labels_refused)}): "
          f"{'; '.join(report.labels_refused)}")
    print(f"channel refusals recorded ({len(report.refusals)}): "
          f"{', '.join(report.refusals[:4])}, ...")
    return 0 if report.side_effect_free else 1


def cmd_attest(args: argparse.Namespace) -> int:
    """Print the platform attestation and nothing else."""
    platform = attest_platform(root=Path(args.root))
    print(f"declared target : {platform.target_model}")
    print(f"observed model  : {platform.observed_model}")
    print(f"machine         : {platform.machine}")
    print(f"system          : {platform.system}")
    print(f"source          : {platform.source}")
    print(f"matches target  : {'yes' if platform.matches_target else 'NO'}")
    if not platform.matches_target:
        print("")
        print("A figure measured here is not a figure measured on the declared target.")
        print("Every certificate this package issues on this host says so in its own")
        print("headline: MEASURED OFF-TARGET, with both model strings.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """The argument parser. Built in one place so ``--help`` is testable."""
    parser = argparse.ArgumentParser(
        prog="python -m boardprobe",
        description=(
            "A hardware abstraction layer for an NVIDIA Jetson Orin Nano whose readings "
            "carry their provenance. Validation level 3, hardware-pending: the board was "
            "not reachable when this was built, every declared path is UNVERIFIED, and no "
            "figure here was measured on the board. Research-grade; not flight-qualified, "
            "not certified, not approved for operational aerospace use."
        ),
        epilog=(
            "A refused label is the product working, not an error: it is printed and the "
            "command exits 0."
        ),
    )
    parser.add_argument("--version", action="version", version=f"boardprobe {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    channels = subparsers.add_parser("channels", help="the declared channel table")
    channels.add_argument(
        "--references", action="store_true",
        help="also print the documentation reference behind every channel",
    )
    channels.set_defaults(handler=cmd_channels)

    sample = subparsers.add_parser("sample", help="one sample from one backend")
    sample.add_argument("--backend", choices=backend_names(), default="simulated")
    sample.add_argument("--seed", type=int, default=0, help="seed for the simulated backend")
    sample.add_argument("--channel", action="append", help="channel to read; repeatable")
    sample.add_argument("--trace", help="trace file for the recorded backend")
    sample.set_defaults(handler=cmd_sample)

    bench = subparsers.add_parser("benchmark", help="run the harness")
    bench.add_argument("--repeats", type=int, default=200)
    bench.add_argument("--warmup", type=int, default=5)
    bench.add_argument("--work", type=int, default=2000,
                       help="iterations of the declared reference payload")
    bench.add_argument("--overhead-repeats", type=int, default=300)
    bench.add_argument("--environment", choices=(*backend_names(), "none"),
                       default="simulated",
                       help="backend for the resource channels, or none for timing only")
    bench.add_argument("--seed", type=int, default=0)
    bench.add_argument("--trace", help="trace file for a recorded environment backend")
    bench.set_defaults(handler=cmd_benchmark)

    replay = subparsers.add_parser("replay", help="replay a trace file")
    replay.add_argument("--trace", help="trace file; defaults to the committed fixture")
    replay.add_argument("--channel", help="channel to summarise")
    replay.add_argument("--samples", type=int, default=12)
    replay.set_defaults(handler=cmd_replay)

    provenance = subparsers.add_parser("provenance", help="the lattice and the label rules")
    provenance.add_argument("--level", default="Level 3, hardware-pending",
                            help=argparse.SUPPRESS)
    provenance.set_defaults(handler=cmd_provenance)

    dry = subparsers.add_parser("dry-run", help="walk every path without touching a board")
    dry.add_argument("--seed", type=int, default=0)
    dry.add_argument("--trace", help="trace file for the replay leg of the walk")
    dry.add_argument("--watch", help="directory to snapshot before and after")
    dry.set_defaults(handler=cmd_dry_run)

    attest = subparsers.add_parser("attest", help="the platform attestation")
    attest.add_argument("--root", default="/",
                        help="filesystem root to read the device-tree model under")
    attest.set_defaults(handler=cmd_attest)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse ``argv`` and run one subcommand. Returns the process exit status."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (
        TraceError,
        LabelViolation,
        DeviceUnreachableError,
        HarnessError,
        ValueError,
    ) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

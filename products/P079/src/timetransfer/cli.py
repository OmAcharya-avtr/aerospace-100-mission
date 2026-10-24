"""Command-line interface: ``python -m timetransfer <command>``.

Every command is a thin wrapper over the library and prints only measured
values. No command writes a number it did not compute in that process.

Commands
--------
``version``    package version and validation label
``info``       environment, including what is **not** installed
``simulate``   seeded acquisition, offset, skew and the uncertainty budget
``allan``      Allan deviation over tau with a confidence interval at each tau
``record``     write a recorded exchange set for dry-run use
``dryrun``     replay a recording through the full path, touching no device
``preflight``  run the pre-flight checks against the simulated backend
``parity``     simulation/device parity digests
``bench``      the benchmark harness, with the hardware rows empty
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from . import __version__
from .acquire import acquire
from .allan import allan_table
from .backends import SimulatedBackend, SimulationSpec, make_backend_pair
from .bench import run_benchmark
from .clockmodel import ClockSpec
from .deploy import preflight
from .dryrun import dry_run, dry_run_negative_control, record_exchanges
from .errors import TimeTransferError
from .linkmodel import LinkSpec
from .skew import KalmanConfig, fit_least_squares, kalman_skew
from .twoway import two_way_estimate
from .uncertainty import AsymmetrySource, offset_uncertainty

VALIDATION_LABEL = "Level 3, hardware-pending"

_Q1 = 1.0e-20
_Q2 = 1.0e-26


def _spec(args: argparse.Namespace) -> SimulationSpec:
    """Build a simulation spec from the common command-line options."""
    return SimulationSpec(
        clock=ClockSpec(
            offset0_s=float(args.offset),
            skew_s_per_s=float(args.skew),
            drift_per_s=float(args.drift),
            q1_s2_per_s=_Q1,
            q2_per_s=_Q2,
        ),
        link=LinkSpec(
            floor_ab_s=float(args.floor) + 0.5 * float(args.asymmetry),
            floor_ba_s=float(args.floor) - 0.5 * float(args.asymmetry),
            jitter_scale_ab_s=float(args.jitter),
            jitter_scale_ba_s=float(args.jitter),
            loss_probability=float(args.loss),
        ),
        period_s=float(args.period),
        timestamp_noise_s=float(args.timestamp_noise),
        resolution_ns=int(args.resolution),
        seed=int(args.seed),
    )


def _add_sim_options(parser: argparse.ArgumentParser) -> None:
    """Options shared by every command that drives the simulated backend."""
    parser.add_argument("--n", type=int, default=300, help="exchanges to attempt")
    parser.add_argument("--seed", type=int, default=20261024, help="simulation seed")
    parser.add_argument("--offset", type=float, default=1.0e-3, help="true initial offset [s]")
    parser.add_argument("--skew", type=float, default=2.0e-6, help="true skew [s/s]")
    parser.add_argument("--drift", type=float, default=0.0, help="true drift [1/s]")
    parser.add_argument("--floor", type=float, default=5.0e-3, help="mean one-way delay floor [s]")
    parser.add_argument(
        "--asymmetry", type=float, default=0.0, help="static asymmetry d_ab - d_ba [s]"
    )
    parser.add_argument("--jitter", type=float, default=2.0e-4, help="queueing gamma scale [s]")
    parser.add_argument("--loss", type=float, default=0.0, help="per-exchange loss probability")
    parser.add_argument(
        "--timestamp-noise", type=float, default=2.0e-8, help="white-PM timestamping noise [s]"
    )
    parser.add_argument("--resolution", type=int, default=1, help="timestamp resolution [ns]")
    parser.add_argument("--period", type=float, default=1.0, help="exchange interval [s]")


def _cmd_version(_: argparse.Namespace) -> int:
    print(f"timetransfer {__version__}")
    print(f"validation: {VALIDATION_LABEL}")
    return 0


def _cmd_info(_: argparse.Namespace) -> int:
    import importlib.metadata as md
    import platform

    def version(name: str) -> str:
        try:
            return md.version(name)
        except md.PackageNotFoundError:
            return "not installed"

    print(f"timetransfer       : {__version__}")
    print(f"validation         : {VALIDATION_LABEL}")
    print(f"python             : {platform.python_version()}")
    for name in ("numpy", "scipy", "scikit-learn", "matplotlib"):
        print(f"{name:<19}: {version(name)}")
    print("")
    print("not used by this package, and deliberately named here:")
    for name in ("allantools", "ntplib"):
        print(f"{name:<19}: {version(name)}")
    print("  the Allan deviation is implemented in timetransfer.allan and verified against")
    print("  closed-form noise processes; no number here comes from either package.")
    print("")
    print("this is a measurement and characterisation tool, not a time-sync daemon.")
    return 0


def _cmd_simulate(args: argparse.Namespace) -> int:
    spec = _spec(args)
    backend = SimulatedBackend(spec)
    backend.open()
    report = acquire(backend, int(args.n), min_exchanges=8)
    backend.close()
    estimate = two_way_estimate(report.exchanges, assumed_skew_s_per_s=float(args.skew))
    fit = fit_least_squares(estimate.midpoint_s, estimate.offset_s, order=int(args.order))
    source = (
        AsymmetrySource.DECLARED_BOUND
        if float(args.asymmetry_bound) > 0.0
        else AsymmetrySource.UNACCOUNTED
    )
    unc = offset_uncertainty(
        fit,
        resolution_ns=int(args.resolution),
        asymmetry_source=source,
        asymmetry_bound_s=float(args.asymmetry_bound),
    )
    kal = kalman_skew(
        estimate.midpoint_s,
        estimate.offset_s,
        config=KalmanConfig(q1_s2_per_s=_Q1, q2_per_s=_Q2, n_states=2),
    )
    print(report.as_text())
    print("")
    print(estimate.as_text())
    print("")
    print("least squares (baseline 2a)")
    print(fit.as_text())
    print("")
    print("kalman (baseline 2b)")
    print(kal.as_text())
    print("")
    print("offset uncertainty budget")
    print(unc.as_text())
    truth = report.exchanges.true_offset_s
    if truth is not None:
        true_at_epoch = float(np.interp(unc.epoch_s, estimate.midpoint_s, truth))
        print("")
        print(f"simulation truth at epoch : {true_at_epoch:.9e} s")
        print(f"estimate minus truth      : {unc.offset_s - true_at_epoch:+.6e} s")
        print(f"declared static bias      : {spec.link.static_bias_s:+.6e} s")
    return 0


def _cmd_allan(args: argparse.Namespace) -> int:
    spec = _spec(args)
    backend = SimulatedBackend(spec)
    backend.open()
    report = acquire(backend, int(args.n), min_exchanges=8)
    backend.close()
    estimate = two_way_estimate(report.exchanges)
    fit = fit_least_squares(estimate.midpoint_s, estimate.offset_s, order=int(args.order))
    residual = fit.detrended(estimate.midpoint_s, estimate.offset_s)
    table = allan_table(residual, float(args.period), noise_kind=args.noise_kind)
    print("Allan deviation of the detrended offset residual")
    print(table.as_text())
    return 0


def _cmd_record(args: argparse.Namespace) -> int:
    spec = _spec(args)
    backend = SimulatedBackend(spec)
    backend.open()
    path, report = record_exchanges(backend, int(args.n), Path(args.path), min_exchanges=8)
    backend.close()
    print(f"wrote {path.name} ({report.exchanges.n} exchanges, {report.n_lost} lost)")
    print(f"backend {report.backend_kind}/{report.backend_driver or 'n/a'} "
          f"is_hardware={report.is_hardware}")
    return 0


def _cmd_dryrun(args: argparse.Namespace) -> int:
    result = dry_run(Path(args.path), order=int(args.order))
    print(result.as_text())
    control = dry_run_negative_control(Path(args.path))
    print("")
    print("negative control (device operations forbidden)")
    print(f"  completed  : {control.completed} (must be 0)")
    print(f"  refused    : {control.refused} (must be > 0)")
    print(f"  raised     : {control.raised_insufficient} (must be True)")
    print(f"  passed     : {control.passed}")
    return 0 if control.passed else 1


def _cmd_preflight(args: argparse.Namespace) -> int:
    spec = _spec(args)
    backend = SimulatedBackend(spec)
    report = preflight(backend, expect_hardware=False)
    print(report.as_text())
    return 0 if report.passed else 1


def _cmd_parity(args: argparse.Namespace) -> int:
    spec = _spec(args)
    sim, dev = make_backend_pair(spec)
    sim.open()
    dev.open()
    a = acquire(sim, int(args.n), min_exchanges=8).exchanges
    b = acquire(dev, int(args.n), min_exchanges=8).exchanges
    sim.close()
    dev.close()
    est_a = two_way_estimate(a)
    est_b = two_way_estimate(b)
    max_abs = float(np.max(np.abs(est_a.offset_s - est_b.offset_s)))
    print(f"simulated backend : {a.backend_kind}/{a.backend_driver or 'n/a'} "
          f"is_hardware={a.is_hardware}")
    print(f"device backend    : {b.backend_kind}/{b.backend_driver} "
          f"is_hardware={b.is_hardware}")
    print(f"exchanges         : {a.n} / {b.n}")
    bytes_equal = a.timestamp_bytes() == b.timestamp_bytes()
    print(f"timestamp bytes   : {'equal' if bytes_equal else 'DIFFER'}")
    print(f"digest            : {a.digest() == b.digest()} (identity fields differ by design)")
    print(f"offset max |diff| : {max_abs:.3e} s")
    return 0 if bytes_equal and max_abs == 0.0 else 1


def _cmd_bench(args: argparse.Namespace) -> int:
    spec = _spec(args)

    def acquire_stage() -> object:
        backend = SimulatedBackend(spec)
        backend.open()
        out = acquire(backend, int(args.n), min_exchanges=8)
        backend.close()
        return out

    backend = SimulatedBackend(spec)
    backend.open()
    exchanges = acquire(backend, int(args.n), min_exchanges=8).exchanges
    backend.close()
    estimate = two_way_estimate(exchanges)

    record = run_benchmark(
        {
            "acquire": acquire_stage,
            "two_way_estimate": lambda: two_way_estimate(exchanges),
            "fit_least_squares": lambda: fit_least_squares(
                estimate.midpoint_s, estimate.offset_s, order=2
            ),
            "kalman_skew": lambda: kalman_skew(estimate.midpoint_s, estimate.offset_s),
            "allan_table": lambda: allan_table(
                estimate.offset_s - float(np.mean(estimate.offset_s)),
                float(args.period),
                noise_kind="white_fm",
            ),
        },
        label=str(args.label),
        environment_note=str(args.note),
        backend_kind="simulated",
        backend_driver="",
        n_repeats=int(args.repeats),
    )
    print(record.as_text())
    if args.json:
        path = record.write(Path(args.json))
        print("")
        print(f"wrote {path.name}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """The argument parser for ``python -m timetransfer``."""
    parser = argparse.ArgumentParser(
        prog="timetransfer",
        description=(
            "Clock offset and skew across a link, with the uncertainty measured rather "
            "than quoted. A measurement and characterisation tool, not a time-sync daemon."
        ),
    )
    subs = parser.add_subparsers(dest="command", required=True)

    subs.add_parser("version", help="package version and validation label").set_defaults(
        func=_cmd_version
    )
    subs.add_parser("info", help="environment, including what is not installed").set_defaults(
        func=_cmd_info
    )

    sim = subs.add_parser("simulate", help="seeded acquisition and the uncertainty budget")
    _add_sim_options(sim)
    sim.add_argument("--order", type=int, default=2, choices=(1, 2), help="polynomial order")
    sim.add_argument(
        "--asymmetry-bound",
        type=float,
        default=0.0,
        help="declared bound on |d_ab - d_ba| [s]; 0 means no asymmetry term is reported",
    )
    sim.set_defaults(func=_cmd_simulate)

    al = subs.add_parser("allan", help="Allan deviation over tau with intervals")
    _add_sim_options(al)
    al.add_argument("--order", type=int, default=2, choices=(1, 2), help="detrending order")
    al.add_argument(
        "--noise-kind",
        default="white_fm",
        choices=("white_pm", "white_fm", "flicker_fm", "random_walk_fm"),
        help="noise type used for the degrees of freedom",
    )
    al.set_defaults(func=_cmd_allan)

    rec = subs.add_parser("record", help="write a recorded exchange set")
    _add_sim_options(rec)
    rec.add_argument("path", help="output JSON path")
    rec.set_defaults(func=_cmd_record)

    dry = subs.add_parser("dryrun", help="replay a recording, touching no device")
    dry.add_argument("path", help="recorded exchange set")
    dry.add_argument("--order", type=int, default=1, choices=(1, 2), help="polynomial order")
    dry.set_defaults(func=_cmd_dryrun)

    pre = subs.add_parser("preflight", help="pre-flight checks against the simulated backend")
    _add_sim_options(pre)
    pre.set_defaults(func=_cmd_preflight)

    par = subs.add_parser("parity", help="simulation/device parity digests")
    _add_sim_options(par)
    par.set_defaults(func=_cmd_parity)

    ben = subs.add_parser("bench", help="benchmark harness; hardware rows stay empty")
    _add_sim_options(ben)
    ben.add_argument("--repeats", type=int, default=5, help="repetitions per stage")
    ben.add_argument("--label", default="cli", help="label for the record")
    ben.add_argument(
        "--note",
        default="interactive run; no attempt made to quiesce the machine",
        help="mandatory environment note",
    )
    ben.add_argument("--json", default=None, help="also write the record as JSON here")
    ben.set_defaults(func=_cmd_bench)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit status."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except TimeTransferError as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        # A missing or unreadable recording is a user error, not a crash.
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

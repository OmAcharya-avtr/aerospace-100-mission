"""Command-line interface: ``python -m hilforge <command>``.

Commands
--------
``info``       package, clock and backend identity
``preflight``  the pre-run checks, printed as a GO/NO-GO report
``run``        one loop run, with the deadline accounting
``dryrun``     a dry run, printing the write counts that prove no write
``parity``     simulated vs loopback-device digests
``replay``     record a run, replay it, compare digests
``bench``      the benchmark harness, optionally writing a record
``overruns``   overrun accounting for a synthetic or stored duration trace
``predict``    train and score the two baselines and the learned model

Every command prints plain, parseable lines and returns 0 on success. A
failed check or a mismatched digest returns a non-zero code, so the CLI is
usable in a script.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import __version__
from .backends import AbsentDriver, DeviceBackend, make_backend_pair
from .bench import run_benchmark, write_record
from .deploy import preflight
from .loop import HilLoop, LoopConfig
from .replay import replay_run
from .timebase import clock_report
from .timing import PeriodSpec, overrun_report

_EPILOG = (
    "HilForge is research-grade and validated at Level 3, hardware-pending: "
    "nothing in it has been run on a board, and no output of this CLI is a "
    "hardware measurement."
)


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--period", type=float, default=0.010, help="loop period in seconds (default 0.010)"
    )
    parser.add_argument(
        "--iterations", type=int, default=1000, help="number of iterations (default 1000)"
    )
    parser.add_argument("--seed", type=int, default=20261004, help="plant seed (default 20261004)")
    parser.add_argument(
        "--backend",
        choices=("simulated", "device", "absent"),
        default="simulated",
        help="simulated plant, loopback device stub, or an absent device",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="hilforge",
        description=(
            "Hardware-in-the-loop harness: HAL with simulation/device parity, "
            "deadline accounting, dry-run rehearsal and a benchmark record."
        ),
        epilog=_EPILOG,
    )
    parser.add_argument("--version", action="version", version=f"hilforge {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_info = sub.add_parser("info", help="package, clock and backend identity")
    p_info.add_argument(
        "--clock-samples", type=int, default=2000, help="samples for the clock measurement"
    )

    p_pre = sub.add_parser("preflight", help="pre-run checks, GO/NO-GO")
    _add_common(p_pre)

    p_run = sub.add_parser("run", help="one loop run with deadline accounting")
    _add_common(p_run)
    p_run.add_argument(
        "--inject",
        choices=("none", "ramp", "bursty"),
        default="none",
        help="inject deterministic per-iteration durations instead of measuring",
    )
    p_run.add_argument(
        "--cascade-limit",
        type=int,
        default=0,
        help="consecutive cascade overruns that abort the run (0 disables)",
    )

    p_dry = sub.add_parser("dryrun", help="dry run; prints the write counts")
    _add_common(p_dry)

    p_par = sub.add_parser("parity", help="simulated vs loopback-device digests")
    _add_common(p_par)

    p_rep = sub.add_parser("replay", help="run, replay, compare digests")
    _add_common(p_rep)

    p_ben = sub.add_parser("bench", help="benchmark harness")
    _add_common(p_ben)
    p_ben.add_argument(
        "--note",
        default="unspecified host; environment not recorded by the caller",
        help="required environment note written into the record",
    )
    p_ben.add_argument("--out", default="", help="path stem for the .txt and .json record")
    p_ben.add_argument("--dry-run", action="store_true", help="benchmark the dry-run path")

    p_ovr = sub.add_parser("overruns", help="overrun accounting for a duration trace")
    p_ovr.add_argument("--period", type=float, default=0.010, help="period in seconds")
    p_ovr.add_argument(
        "--trace",
        default="",
        help="path to a .npy or .json file of per-iteration durations in seconds",
    )
    p_ovr.add_argument(
        "--preset",
        choices=("jittery", "bursty"),
        default="jittery",
        help="synthetic trace preset when --trace is not given",
    )
    p_ovr.add_argument("--iterations", type=int, default=1000, help="synthetic trace length")
    p_ovr.add_argument("--seed", type=int, default=20261004, help="synthetic trace seed")
    p_ovr.add_argument("--json", action="store_true", help="print the account as JSON")

    p_prd = sub.add_parser("predict", help="baselines and learned overrun predictor")
    p_prd.add_argument(
        "--preset", choices=("jittery", "bursty"), default="bursty", help="trace family"
    )
    p_prd.add_argument("--iterations", type=int, default=30000, help="trace length")
    p_prd.add_argument("--seed", type=int, default=20261004, help="trace seed")
    p_prd.add_argument("--horizon", type=int, default=3, help="label look-ahead in iterations")
    p_prd.add_argument(
        "--flag-rate",
        type=float,
        default=0.15,
        help="matched flag-rate operating point (0 disables, using F1 points)",
    )
    return parser


def _make_backend(args) -> object:
    if args.backend == "simulated":
        sim, _ = make_backend_pair(seed=args.seed, sample_dt_s=args.period)
        return sim
    if args.backend == "device":
        _, dev = make_backend_pair(seed=args.seed, sample_dt_s=args.period)
        return dev
    return DeviceBackend(AbsentDriver(), sample_dt_s=args.period)


def _injected(kind: str, n: int, period: float, seed: int) -> tuple[float, ...] | None:
    if kind == "none":
        return None
    rng = np.random.Generator(np.random.PCG64(seed))
    if kind == "ramp":
        return tuple(np.linspace(0.3 * period, 1.4 * period, n))
    base = rng.gamma(6.0, 0.45 * period / 6.0, size=n)
    burst = rng.random(n) < 0.08
    return tuple(base + burst * rng.exponential(0.9 * period, size=n))


def _cmd_info(args) -> int:
    report = clock_report(args.clock_samples)
    sim, dev = make_backend_pair()
    print(f"hilforge {__version__}")
    print("validation level    : 3, hardware-pending")
    print("licence             : AGPL-3.0-or-later, (c) 2026 OPTIMA Organisation")
    print()
    print(report.as_text())
    print()
    for backend in (sim, dev):
        info = backend.info
        print(
            f"backend {info.kind:<10} name={info.name:<16} driver={info.driver or 'n/a':<10} "
            f"is_hardware={info.is_hardware}"
        )
    print()
    print("no hardware backend is configured; no number here is a hardware measurement")
    return 0


def _cmd_preflight(args) -> int:
    backend = _make_backend(args)
    report = preflight(backend, period_s=args.period)
    print(report.as_text())
    return 0 if report.passed else 2


def _cmd_run(args) -> int:
    backend = _make_backend(args)
    inj = _injected(args.inject, args.iterations, args.period, args.seed)
    cfg = LoopConfig(
        period=PeriodSpec(period_s=args.period, cascade_limit=args.cascade_limit),
        n_iterations=args.iterations,
        injected_durations_s=inj,
        on_cascade="abort" if args.cascade_limit else "record",
    )
    record = HilLoop(backend, cfg).run()
    for key, value in record.summary().items():
        print(f"{key:<26}: {value}")
    print(f"{'data_digest':<26}: {record.data_digest()}")
    if record.deterministic_timing:
        print(f"{'full_digest':<26}: {record.full_digest()}")
    return 0


def _cmd_dryrun(args) -> int:
    backend = _make_backend(args)
    backend.open()
    raw = backend.actuators["torque"]
    before = raw.write_count
    cfg = LoopConfig(
        period=PeriodSpec(period_s=args.period),
        n_iterations=args.iterations,
        dry_run=True,
        injected_durations_s=tuple(np.full(args.iterations, 0.4 * args.period)),
    )
    record = HilLoop(backend, cfg).run()
    after = raw.write_count
    print(f"iterations                : {record.n_completed}")
    print(f"rehearsed writes          : {record.rehearsed_writes}")
    print(f"writes issued             : {record.writes_issued}")
    print(f"actuator write_count      : {before} -> {after}")
    print(f"no write reached hardware : {after == before}")
    backend.close()
    return 0 if after == before and record.rehearsed_writes == record.n_completed else 3


def _cmd_parity(args) -> int:
    sim, dev = make_backend_pair(seed=args.seed, sample_dt_s=args.period)
    inj = tuple(np.full(args.iterations, 0.4 * args.period))
    cfg = LoopConfig(
        period=PeriodSpec(period_s=args.period),
        n_iterations=args.iterations,
        injected_durations_s=inj,
    )
    a = HilLoop(sim, cfg).run()
    b = HilLoop(dev, cfg).run()
    sig_a = a.signal_matrix()
    sig_b = b.signal_matrix()
    identical = sig_a.tobytes() == sig_b.tobytes()
    print(f"simulated data_digest : {a.data_digest()}")
    print(f"device    data_digest : {b.data_digest()}")
    print(f"simulated full_digest : {a.full_digest()}")
    print(f"device    full_digest : {b.full_digest()}")
    print(f"signal bytes identical: {identical}")
    print(f"max |difference|      : {float(np.max(np.abs(sig_a - sig_b))):.3e}")
    print("device backend driver : loopback stub, not hardware")
    return 0 if identical and a.full_digest() == b.full_digest() else 4


def _cmd_replay(args) -> int:
    sim = _make_backend(args)
    cfg = LoopConfig(
        period=PeriodSpec(period_s=args.period),
        n_iterations=args.iterations,
        injected_durations_s=tuple(np.full(args.iterations, 0.4 * args.period)),
    )
    original = HilLoop(sim, cfg).run()
    again = replay_run(original)
    same = original.data_digest() == again.data_digest()
    print(f"original data_digest : {original.data_digest()}")
    print(f"replay   data_digest : {again.data_digest()}")
    print(f"identical            : {same}")
    return 0 if same else 5


def _cmd_bench(args) -> int:
    backend = _make_backend(args)
    record, _ = run_benchmark(
        backend,
        period_s=args.period,
        n_iterations=args.iterations,
        environment_note=args.note,
        dry_run=args.dry_run,
        label=f"{args.backend}-backend",
    )
    print(record.as_text())
    if args.out:
        txt, js = write_record(record, args.out)
        print()
        print(f"written: {txt}")
        print(f"written: {js}")
    return 0


def _load_trace(path: str) -> np.ndarray:
    p = Path(path)
    if p.suffix == ".npy":
        return np.asarray(np.load(p), dtype=np.float64).ravel()
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("trace", [])
    return np.asarray(data, dtype=np.float64).ravel()


def _cmd_overruns(args) -> int:
    from .predict.data import TraceConfig, generate_trace

    if args.trace:
        durations = _load_trace(args.trace)
    else:
        cfg = TraceConfig.preset(args.preset, n_iterations=max(args.iterations, 50))
        durations = generate_trace(cfg, seed=args.seed).totals_s
    account = overrun_report(durations, args.period)
    if args.json:
        print(json.dumps(account.as_dict(), indent=2))
        return 0
    print(f"iterations               : {account.n_iterations}")
    print(f"period                   : {account.period_s:.6e} s")
    print(f"deadline                 : {account.deadline_s:.6e} s")
    print(f"direct overruns          : {account.direct_count}")
    print(f"direct overrun rate      : {account.direct_rate:.6f}")
    print(f"cascade overruns         : {account.cascade_count}")
    print(f"cascade overrun rate     : {account.cascade_rate:.6f}")
    print(f"max consecutive cascade  : {account.max_consecutive_cascade}")
    print(f"mean duration            : {float(np.mean(durations)):.6e} s")
    print(f"utilisation (mean / T)   : {float(np.mean(durations)) / args.period:.6f}")
    return 0


def _cmd_predict(args) -> int:
    from .predict import (
        FixedThresholdPredictor,
        LearnedOverrunPredictor,
        OverrunMetrics,
        QueueingOverrunPredictor,
        TraceConfig,
        build_dataset,
        evaluate_predictor,
        generate_trace,
        split_trace,
    )

    cfg = TraceConfig.preset(args.preset, n_iterations=args.iterations)
    trace = generate_trace(cfg, seed=args.seed)
    train, test = split_trace(trace, train_fraction=0.6)
    deadline = cfg.period_s
    x_tr, y_tr, _ = build_dataset(train.stage_s, deadline, horizon=args.horizon)
    x_te, y_te, idx = build_dataset(test.stage_s, deadline, horizon=args.horizon)
    over = test.overruns()
    window = np.column_stack([over[idx + h] for h in range(1, args.horizon + 1)])
    predictors = [
        FixedThresholdPredictor(deadline_s=deadline).fit(x_tr, y_tr),
        QueueingOverrunPredictor(deadline_s=deadline, horizon=args.horizon).fit(x_tr, y_tr),
        LearnedOverrunPredictor().fit(x_tr, y_tr),
    ]
    print(f"trace preset        : {args.preset}  seed {args.seed}  n {trace.n}")
    print(f"period / deadline   : {deadline:.6e} s")
    print(f"utilisation         : {trace.utilisation:.6f}")
    print(f"per-iteration overrun rate : {float(trace.overruns().mean()):.6f}")
    print(f"horizon             : {args.horizon} iterations")
    print(f"train rows / test rows     : {y_tr.size} / {y_te.size}")
    print(f"test positive rate  : {float(y_te.mean()):.6f}")
    print()
    if args.flag_rate > 0.0:
        for p in predictors:
            p.calibrate_flag_rate(x_tr, args.flag_rate)
        print(f"operating point: matched flag rate, target {args.flag_rate:.3f}")
    else:
        print("operating point: F1-maximising on the training set")
    print(OverrunMetrics.header())
    for p in predictors:
        print(evaluate_predictor(p, x_te, y_te, window, horizon=args.horizon).as_row())
    return 0


_COMMANDS = {
    "info": _cmd_info,
    "preflight": _cmd_preflight,
    "run": _cmd_run,
    "dryrun": _cmd_dryrun,
    "parity": _cmd_parity,
    "replay": _cmd_replay,
    "bench": _cmd_bench,
    "overruns": _cmd_overruns,
    "predict": _cmd_predict,
}


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns the process exit code."""
    args = build_parser().parse_args(argv)
    return _COMMANDS[args.command](args)


if __name__ == "__main__":  # pragma: no cover - exercised via __main__
    sys.exit(main())

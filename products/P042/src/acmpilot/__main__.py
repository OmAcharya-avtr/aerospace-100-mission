"""Command line interface: ``python -m acmpilot <subcommand>``.

Subcommands
-----------
``thresholds``
    Measure the MODCOD ladder thresholds and print the table; optionally write
    it as JSON.
``simulate``
    Run the three non-learned policies at one feedback delay and print the full
    accounting.
``sweep``
    Sweep the feedback delay and print one CSV row per (delay, policy).
``predict``
    Train the learned quantile predictor and the analytic AR(1) predictor at one
    delay, print their calibration, and score both as policies against the three
    baselines.

Every number printed comes from a run in the current process. No output contains
a filesystem path other than one the caller supplied.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from . import __version__
from .channel import ChannelConfig
from .modcod import DEFAULT_TARGET_BER, ModcodTable, measure_thresholds
from .policy import baseline_policies
from .predictor import (
    GaussMarkovPredictor,
    PredictivePolicy,
    QuantilePredictor,
    analytic_calibration_report,
    calibration_report,
    make_lag_features,
)
from .simulate import run_policy, sweep_delay


def _channel_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--slot-ms", type=float, default=1.0, help="slot interval, ms")
    parser.add_argument(
        "--tau-c-ms", type=float, default=10.0, help="channel correlation time, ms"
    )
    parser.add_argument(
        "--sigma-i2", type=float, default=0.5, help="scintillation index, dimensionless"
    )
    parser.add_argument(
        "--mean-snr-db", type=float, default=14.0, help="SNR at mean irradiance, dB"
    )
    parser.add_argument(
        "--marginal",
        choices=("lognormal", "gamma-gamma"),
        default="lognormal",
        help="amplitude fading marginal",
    )
    parser.add_argument("--n-slots", type=int, default=20000, help="slots per episode")
    parser.add_argument("--seed", type=int, default=1, help="episode seed")


def _config(args: argparse.Namespace) -> ChannelConfig:
    return ChannelConfig(
        slot_s=args.slot_ms * 1e-3,
        tau_c_s=args.tau_c_ms * 1e-3,
        sigma_i2=args.sigma_i2,
        mean_snr_db=args.mean_snr_db,
        marginal=args.marginal,
    )


def _table(args: argparse.Namespace) -> ModcodTable:
    if getattr(args, "table", None):
        return ModcodTable.load_json(args.table)
    table, _ = measure_thresholds(target_ber=args.target_ber, seed=args.threshold_seed)
    return table


def _print_table(table: ModcodTable) -> None:
    print(f"target post-decoding BER: {table.target_ber:g}")
    print(f"monotone ladder: {table.is_monotone}")
    print(
        f"{'idx':>3} {'MODCOD':22s} {'bit/sym':>8} {'eta':>7} "
        f"{'thr_dB':>8} {'sigma_dB':>9}"
    )
    for i, (mode, thr, sig) in enumerate(
        zip(table.modcods, table.thresholds_db, table.threshold_sigma_db, strict=True)
    ):
        print(
            f"{i:3d} {mode.name:22s} {mode.bits_per_symbol:8d} "
            f"{mode.spectral_efficiency:7.3f} {thr:8.3f} {sig:9.3f}"
        )


def _cmd_thresholds(args: argparse.Namespace) -> int:
    table, _ = measure_thresholds(target_ber=args.target_ber, seed=args.threshold_seed)
    _print_table(table)
    if args.out:
        table.save_json(args.out)
        print(f"written: {args.out}")
    return 0


def _cmd_simulate(args: argparse.Namespace) -> int:
    from .channel import snr_db_path

    table = _table(args)
    config = _config(args)
    snr = snr_db_path(config, args.n_slots, args.seed)
    d = config.delay_slots(args.tau_ms * 1e-3)
    print(f"tau = {args.tau_ms:g} ms = {d} slots; tau_c = {args.tau_c_ms:g} ms")
    print(f"mean SNR of path: {snr.mean():.3f} dB (std {snr.std():.3f} dB)")
    for policy in baseline_policies(
        margin_db=args.margin_db,
        up_margin_db=args.up_margin_db,
        down_margin_db=args.down_margin_db,
    ):
        acc = run_policy(table, snr, policy, delay_slots=d).accounting
        print(f"\n-- {policy.name}")
        for key, value in acc.as_dict().items():
            print(f"   {key:36s} {value:12.6f}")
    return 0


def _cmd_sweep(args: argparse.Namespace) -> int:
    table = _table(args)
    config = _config(args)
    taus = [t * 1e-3 for t in args.tau_ms_list]
    rows = sweep_delay(
        table,
        config,
        baseline_policies(
            margin_db=args.margin_db,
            up_margin_db=args.up_margin_db,
            down_margin_db=args.down_margin_db,
        ),
        tau_list_s=taus,
        n_slots=args.n_slots,
        seeds=tuple(range(1, args.n_seeds + 1)),
    )
    keys = list(rows[0].keys())
    print(",".join(keys))
    for row in rows:
        print(",".join(f"{row[k]:.6g}" if isinstance(row[k], float) else str(row[k]) for k in keys))
    return 0


def _cmd_predict(args: argparse.Namespace) -> int:
    from .channel import snr_db_path

    table = _table(args)
    config = _config(args)
    d = config.delay_slots(args.tau_ms * 1e-3)
    train = np.concatenate(
        [snr_db_path(config, args.train_slots, 10_000 + i) for i in range(args.train_episodes)]
    )
    feat, target, _ = make_lag_features(train, delay_slots=d, n_lags=args.n_lags)
    learned = QuantilePredictor(delay_slots=d, n_estimators=args.n_estimators).fit(
        feat, target
    )
    analytic = GaussMarkovPredictor(delay_slots=d).fit(train)
    test = snr_db_path(config, args.n_slots, args.seed)
    feat_t, target_t, _ = make_lag_features(test, delay_slots=d, n_lags=args.n_lags)
    print(f"tau = {args.tau_ms:g} ms = {d} slots; training rows = {feat.shape[0]}")
    print("\n-- calibration, learned quantile GBR")
    for key, value in calibration_report(learned, feat_t, target_t).items():
        print(f"   {key:22s} {value:12.6f}")
    print("\n-- calibration, analytic AR(1) MMSE (not learned)")
    for key, value in analytic_calibration_report(analytic, feat_t, target_t).items():
        print(f"   {key:22s} {value:12.6f}")
    policies = [
        *baseline_policies(
            margin_db=args.margin_db,
            up_margin_db=args.up_margin_db,
            down_margin_db=args.down_margin_db,
        ),
        PredictivePolicy(analytic, delay_slots=d, n_lags=args.n_lags, gate_k=args.gate_k),
        PredictivePolicy(learned, delay_slots=d, n_lags=args.n_lags, gate_k=args.gate_k),
    ]
    print(f"\n-- goodput, bit/symbol (gate k = {args.gate_k:g})")
    for policy in policies:
        acc = run_policy(table, test, policy, delay_slots=d).accounting
        print(
            f"   {policy.name:46s} goodput={acc.goodput_bit_per_symbol:7.4f} "
            f"efficiency={acc.efficiency:6.4f} outage={acc.outage_fraction:7.4f} "
            f"conservative={acc.conservative_fraction:6.4f}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for ``python -m acmpilot``."""
    parser = argparse.ArgumentParser(
        prog="acmpilot",
        description=(
            "Adaptive coding and modulation on one optical link under feedback "
            "delay. Research-grade; not flight-qualified, not certified, not "
            "approved for operational aerospace use."
        ),
    )
    parser.add_argument("--version", action="version", version=f"acmpilot {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    common_threshold = argparse.ArgumentParser(add_help=False)
    common_threshold.add_argument(
        "--target-ber", type=float, default=DEFAULT_TARGET_BER,
        help="post-decoding BER defining a MODCOD threshold",
    )
    common_threshold.add_argument(
        "--threshold-seed", type=int, default=20261006, help="seed for the BER Monte Carlo"
    )

    common_policy = argparse.ArgumentParser(add_help=False)
    common_policy.add_argument("--margin-db", type=float, default=3.0)
    common_policy.add_argument("--up-margin-db", type=float, default=2.0)
    common_policy.add_argument("--down-margin-db", type=float, default=0.5)
    common_policy.add_argument(
        "--table", default=None, help="path to a thresholds JSON file to load instead of measuring"
    )

    p_thr = sub.add_parser(
        "thresholds", parents=[common_threshold], help="measure the MODCOD thresholds"
    )
    p_thr.add_argument("--out", default=None, help="write the table as JSON to this path")
    p_thr.set_defaults(func=_cmd_thresholds)

    p_sim = sub.add_parser(
        "simulate",
        parents=[common_threshold, common_policy],
        help="run the three non-learned policies at one feedback delay",
    )
    _channel_args(p_sim)
    p_sim.add_argument("--tau-ms", type=float, default=10.0, help="feedback delay, ms")
    p_sim.set_defaults(func=_cmd_simulate)

    p_sw = sub.add_parser(
        "sweep",
        parents=[common_threshold, common_policy],
        help="sweep the feedback delay, CSV to stdout",
    )
    _channel_args(p_sw)
    p_sw.add_argument(
        "--tau-ms-list", type=float, nargs="+", default=[0.0, 2.0, 5.0, 10.0, 20.0, 40.0]
    )
    p_sw.add_argument("--n-seeds", type=int, default=5)
    p_sw.set_defaults(func=_cmd_sweep)

    p_pr = sub.add_parser(
        "predict",
        parents=[common_threshold, common_policy],
        help="train and score the predictors at one feedback delay",
    )
    _channel_args(p_pr)
    p_pr.add_argument("--tau-ms", type=float, default=10.0, help="feedback delay, ms")
    p_pr.add_argument("--n-lags", type=int, default=8)
    p_pr.add_argument("--gate-k", type=float, default=0.5)
    p_pr.add_argument("--train-slots", type=int, default=5200)
    p_pr.add_argument("--train-episodes", type=int, default=3)
    p_pr.add_argument("--n-estimators", type=int, default=100)
    p_pr.set_defaults(func=_cmd_predict)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit status."""
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())

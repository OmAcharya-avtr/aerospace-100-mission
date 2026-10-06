"""Command-line interface: ``python -m linkoutage``.

Four subcommands, all of which print numbers and the definitions behind them:

``stats``
    Fade statistics of a supplied or generated amplitude series.
``fit``
    Fade-duration distribution comparison and Markov / semi-Markov fitting
    with goodness of fit.
``crosscheck``
    The X1 cross-check configuration: the exact seeded series, the mean
    irradiance check, the level-crossing rate and the mean fade duration.
``predict``
    Train the three baselines and the forest on a generated record and print
    the calibration table.

Input files may be ``.npy`` (one-dimensional float array) or a text file with
one amplitude per line (``.csv`` and ``.txt`` both read this way, first column
only). Paths are echoed back exactly as given, never absolutised.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from . import __version__
from .calibration import CalibrationReport, evaluate_forecast
from .channel import (
    analytic_level_crossing_rate,
    analytic_mean_fade_duration,
    lognormal_amplitude_series,
)
from .distributions import compare_fade_duration_models
from .fade import FadeDefinitions, fade_durations, fade_statistics
from .features import build_outage_dataset
from .markov import (
    dwell_time_goodness_of_fit,
    fit_markov,
    fit_semi_markov,
    markov_order_test,
    state_sequence,
)
from .predictors import (
    AnalyticLcrPredictor,
    ConstantRatePredictor,
    LogisticBaseline,
    PlattCalibrated,
    RandomForestOutageClassifier,
)

X1_N_SAMPLES = 2_000_000
X1_SEED = 41
X1_FS_HZ = 1.0e6
X1_TAU_S = 2.0e-4
X1_SI = 0.6
X1_THRESHOLD = 0.6
X1_REFERENCE_MEAN_IRRADIANCE = 0.998907972681559


def _load_amplitude(path: str) -> NDArray[np.float64]:
    if path.endswith(".npy"):
        arr = np.load(path)
    else:
        arr = np.loadtxt(path, delimiter=",", ndmin=2)
        arr = arr[:, 0]
    out = np.asarray(arr, dtype=np.float64).ravel()
    if out.size < 2:
        raise SystemExit(f"{path}: need at least 2 samples, read {out.size}")
    return out


def _definitions(args: argparse.Namespace) -> FadeDefinitions:
    return FadeDefinitions(
        strict_below=not args.non_strict,
        duration_convention=args.duration_convention,
        censoring=args.censoring,
        count_single_sample_fades=not args.drop_single_sample_fades,
        record_duration_convention=args.record_duration,
    )


def _series(args: argparse.Namespace) -> tuple[NDArray[np.float64], float, str]:
    if args.input is not None:
        return _load_amplitude(args.input), float(args.fs), f"supplied file {args.input}"
    series = lognormal_amplitude_series(
        int(args.n_samples),
        fs_hz=float(args.fs),
        tau_s=float(args.tau),
        si=float(args.si),
        seed=int(args.seed),
    )
    return series.amplitude, float(args.fs), series.describe()


def _add_series_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--input", help="amplitude series file (.npy, or text, first column)")
    p.add_argument("--n-samples", type=int, default=200_000, help="samples to generate")
    p.add_argument("--fs", type=float, default=1.0e6, help="sample rate, Hz")
    p.add_argument("--tau", type=float, default=2.0e-4, help="correlation time, s")
    p.add_argument("--si", type=float, default=0.6, help="scintillation index")
    p.add_argument("--seed", type=int, default=41, help="generator seed")
    p.add_argument("--threshold", type=float, default=0.6, help="amplitude fade threshold")


def _add_definition_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--non-strict", action="store_true", help="treat a[n] == T as in fade"
    )
    p.add_argument(
        "--duration-convention",
        choices=("sample_count", "interval_count", "interpolated"),
        default="sample_count",
    )
    p.add_argument("--censoring", choices=("exclude", "include_as_complete"), default="exclude")
    p.add_argument(
        "--drop-single-sample-fades",
        action="store_true",
        help="do not count single-sample excursions as fades",
    )
    p.add_argument("--record-duration", choices=("intervals", "samples"), default="intervals")


def _cmd_stats(args: argparse.Namespace) -> int:
    amplitude, fs, origin = _series(args)
    defs = _definitions(args)
    stats = fade_statistics(amplitude, float(args.threshold), fs, definitions=defs)
    print(origin)
    print()
    print(stats.report())
    return 0


def _cmd_fit(args: argparse.Namespace) -> int:
    amplitude, fs, origin = _series(args)
    defs = _definitions(args)
    print(origin)
    print()
    complete, censored = fade_durations(
        amplitude, float(args.threshold), fs, definitions=defs
    )
    comparison = compare_fade_duration_models(complete, fs_hz=fs, censored_s=censored)
    print(comparison.report())
    print()
    states = state_sequence(amplitude, [float(args.threshold)])
    fit = fit_markov(states)
    print(fit.report())
    print()
    for gof in dwell_time_goodness_of_fit(states, fit):
        print(gof.line())
    print()
    print(markov_order_test(states).report())
    print()
    print(fit_semi_markov(states).report())
    return 0


def _cmd_crosscheck(args: argparse.Namespace) -> int:
    del args
    series = lognormal_amplitude_series(
        X1_N_SAMPLES, fs_hz=X1_FS_HZ, tau_s=X1_TAU_S, si=X1_SI, seed=X1_SEED
    )
    mean_i = float(series.irradiance.mean())
    print(series.describe())
    print()
    print("Input check (the series itself, not the compared statistics):")
    print(f"  computed mean irradiance : {mean_i!r}")
    print(f"  reference mean irradiance: {X1_REFERENCE_MEAN_IRRADIANCE!r}")
    print(f"  absolute difference      : {abs(mean_i - X1_REFERENCE_MEAN_IRRADIANCE)!r}")
    print()
    stats = fade_statistics(series.amplitude, X1_THRESHOLD, X1_FS_HZ)
    print(stats.report())
    print()
    print("Context only, NOT the compared quantity:")
    print(
        "  analytic discrete LCR    : "
        f"{analytic_level_crossing_rate(X1_THRESHOLD, si=X1_SI, tau_s=X1_TAU_S, fs_hz=X1_FS_HZ)!r}"
        " Hz"
    )
    print(
        "  analytic mean fade dur.  : "
        f"{analytic_mean_fade_duration(X1_THRESHOLD, si=X1_SI, tau_s=X1_TAU_S, fs_hz=X1_FS_HZ)!r}"
        " s"
    )
    return 0


def _cmd_predict(args: argparse.Namespace) -> int:
    series = lognormal_amplitude_series(
        int(args.n_samples),
        fs_hz=float(args.fs),
        tau_s=float(args.tau),
        si=float(args.si),
        seed=int(args.seed),
    )
    data = build_outage_dataset(
        series.amplitude,
        threshold=float(args.threshold),
        window_samples=int(args.window),
        horizon_samples=int(args.horizon),
        stride_samples=int(args.stride),
        gaussian=series.gaussian,
    )
    print(series.describe())
    print()
    print(data.report())
    print()
    train, cal, test = data.split.train, data.split.calibration, data.split.test
    constant = ConstantRatePredictor().fit(data.y[train])
    last_train = int(data.index[train[-1]])
    analytic = AnalyticLcrPredictor(
        threshold=float(args.threshold), horizon_samples=int(args.horizon)
    ).fit(series.amplitude[: last_train + 1])
    logistic = LogisticBaseline().fit(data.x[train], data.y[train])
    forest = RandomForestOutageClassifier(
        n_estimators=int(args.trees), min_samples_leaf=int(args.leaf), seed=int(args.seed)
    ).fit(data.x[train], data.y[train])
    rows = [
        evaluate_forecast(
            data.y[test],
            m.predict_proba_onset(data.x[test]),
            name=m.name,
            reference_rate=constant.rate,
        )
        for m in (constant, analytic, logistic, forest)
    ]
    for base in (analytic, forest):
        platt = PlattCalibrated(base).fit(data.x[cal], data.y[cal])
        rows.append(
            evaluate_forecast(
                data.y[test],
                platt.predict_proba_onset(data.x[test]),
                name=platt.name,
                reference_rate=constant.rate,
            )
        )
    print("Held-out test split, baselines first:")
    print(CalibrationReport.header())
    for row in rows:
        print(row.line())
    print()
    best = min(rows, key=lambda r: r.brier)
    print(f"Lowest Brier score on the held-out split: {best.name}")
    print(
        "Accuracy is not reported as a headline: an all-negative forecaster scores "
        f"{rows[0].accuracy_all_negative!r} on this split."
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m linkoutage",
        description=(
            "Outage and availability statistics for an optical link: fade durations, "
            "level-crossing rates, Markov channel-state fitting and a short-horizon "
            "outage classifier. Research-grade; not flight-qualified, not certified, "
            "not approved for operational aerospace use."
        ),
    )
    parser.add_argument("--version", action="version", version=f"linkoutage {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_stats = sub.add_parser("stats", help="fade statistics of a series")
    _add_series_args(p_stats)
    _add_definition_args(p_stats)
    p_stats.set_defaults(func=_cmd_stats)

    p_fit = sub.add_parser("fit", help="fade-duration and Markov fitting with goodness of fit")
    _add_series_args(p_fit)
    _add_definition_args(p_fit)
    p_fit.set_defaults(func=_cmd_fit)

    p_x1 = sub.add_parser(
        "crosscheck", help="the X1 cross-check series, LCR and mean fade duration"
    )
    p_x1.set_defaults(func=_cmd_crosscheck)

    p_pred = sub.add_parser("predict", help="train and score the outage forecasters")
    _add_series_args(p_pred)
    p_pred.set_defaults(threshold=0.35)
    p_pred.add_argument("--window", type=int, default=400, help="observation window, samples")
    p_pred.add_argument("--horizon", type=int, default=200, help="prediction horizon, samples")
    p_pred.add_argument("--stride", type=int, default=50, help="decision spacing, samples")
    p_pred.add_argument("--trees", type=int, default=200, help="forest size")
    p_pred.add_argument("--leaf", type=int, default=5, help="forest min_samples_leaf")
    p_pred.set_defaults(func=_cmd_predict)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns the process exit code."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        return int(args.func(args))
    except (ValueError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

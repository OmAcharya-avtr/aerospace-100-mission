"""Command-line interface: ``python -m deadlinerisk``.

Subcommands, each taking ``--traces`` pointing at a campaign directory:

``manifest``   board identity, period, deadline, payload, probe table, runs.
``account``    release and miss reconstruction for one run or all runs.
``headline``   the understatement factor with its moving-block interval.
``evt``        the GPD threshold-stability diagnostic and the chosen fit.
``learned``    the learned predictor against both baselines on both splits.

Exit status is 0 on success, 2 on a usage or input error, and 3 when the
requested traces are absent. The absent case is distinguished because it is not
a defect: the campaign may simply not have written them yet.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import numpy as np

from .accounting import account, miss_indicator
from .baselines import choose_threshold, mean_plus_margin, threshold_stability
from .bootstrap import counted_rate_interval, understatement_factor
from .predictor import ConstantRatePredictor, ContentionMissPredictor, score
from .provenance import Origin
from .report import (
    account_lines,
    campaign_lines,
    evt_lines,
    headline_lines,
    interval_lines,
    margin_lines,
    score_table,
)
from .speed_scaling import BOARD_CORES, predicted_miss_probability
from .trace import TraceDataAbsentError, load_campaign
from .windows import extract_windows, grouped_splits, time_split


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m deadlinerisk",
        description=(
            "Deadline-miss risk from measured periodic-task traces. The headline is "
            "the factor by which the mean-plus-margin rule understates the counted "
            "miss rate, with a moving-block bootstrap interval."
        ),
    )
    parser.add_argument(
        "--traces",
        default="data/traces",
        help="campaign directory holding manifest.json and jobs_<run>.csv "
        "(default: data/traces)",
    )
    parser.add_argument(
        "--origin",
        choices=[item.value for item in Origin],
        default=Origin.MEASURED.value,
        help="origin to record for the loaded traces (default: measured)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("manifest", help="campaign metadata and probe table")

    account_parser = subparsers.add_parser("account", help="releases, misses, drops")
    account_parser.add_argument("--run", default=None, help="one run, or all runs")

    for name, help_text in (
        ("headline", "understatement factor with its interval"),
        ("evt", "GPD threshold stability and the chosen fit"),
    ):
        sub = subparsers.add_parser(name, help=help_text)
        sub.add_argument("--run", default="gt_long", help="run to analyse")
        if name == "headline":
            sub.add_argument("--resamples", type=int, default=2000)
            sub.add_argument("--seed", type=int, default=920092)

    learned = subparsers.add_parser("learned", help="learned predictor vs baselines")
    learned.add_argument("--idle-run", default="mix_c0", help="idle run for speed scaling")
    learned.add_argument("--holdout-fraction", type=float, default=0.3)
    return parser


def _run_or_fail(campaign, run: str):
    if run not in campaign.jobs:
        raise TraceDataAbsentError(
            f"run {run!r} is not in {campaign.provenance.directory}/; "
            f"runs present: {', '.join(campaign.provenance.runs) or 'none'}"
        )
    return campaign.jobs[run]


def _learned(campaign, arguments, emit) -> int:
    windows = extract_windows(campaign)
    emit(f"windows extracted       : {len(windows)}")
    emit(f"window length           : {windows.window_s:.1f} s")
    emit(f"windows discarded       : {len(windows.dropped_windows)}")
    emit(f"descriptors             : {', '.join(windows.feature_names)}")
    idle = campaign.jobs.get(arguments.idle_run)
    if idle is None:
        emit(
            f"idle run {arguments.idle_run!r} absent: the speed-scaling baseline cannot "
            "be scored and is omitted"
        )
    emit("")
    emit("held-out workload mixes (leave-one-run-out)")
    rows = []
    truth_all: list[float] = []
    learned_all: list[float] = []
    learned_sigma: list[float] = []
    constant_all: list[float] = []
    classical_all: list[float] = []
    for label, train, test in grouped_splits(windows):
        model = ContentionMissPredictor().fit(
            windows.features[train], windows.targets[train], windows.feature_names
        )
        prediction = model.predict(windows.features[test], windows.releases[test])
        constant = ConstantRatePredictor().fit(
            windows.features[train], windows.targets[train]
        )
        truth_all.extend(windows.targets[test].tolist())
        learned_all.extend(prediction.rate.tolist())
        learned_sigma.extend(prediction.sigma.tolist())
        constant_all.extend(constant.predict(windows.features[test]).rate.tolist())
        if idle is not None:
            workers = int(windows.features[test, 2][0])
            classical_all.extend(
                [
                    predicted_miss_probability(
                        idle.exec_s, campaign.deadline_s, BOARD_CORES, workers
                    )
                ]
                * int(test.size)
            )
        rows.append(label)
    emit(f"  held-out runs scored  : {len(rows)} ({', '.join(rows)})")
    scores = [
        score("learned forest (grouped)", np.asarray(truth_all),
              np.asarray(learned_all), np.asarray(learned_sigma)),
        score("constant rate (grouped)", np.asarray(truth_all), np.asarray(constant_all)),
    ]
    if classical_all:
        scores.append(
            score("speed scaling (grouped)", np.asarray(truth_all),
                  np.asarray(classical_all))
        )
    for line in score_table(scores):
        emit("  " + line)

    emit("")
    emit("held-out time (later portion of every run)")
    train, test = time_split(windows, arguments.holdout_fraction)
    model = ContentionMissPredictor().fit(
        windows.features[train], windows.targets[train], windows.feature_names
    )
    prediction = model.predict(windows.features[test], windows.releases[test])
    constant = ConstantRatePredictor().fit(windows.features[train], windows.targets[train])
    time_scores = [
        score("learned forest (time)", windows.targets[test], prediction),
        score(
            "constant rate (time)",
            windows.targets[test],
            constant.predict(windows.features[test]).rate,
        ),
    ]
    if idle is not None:
        classical = np.asarray(
            [
                predicted_miss_probability(
                    idle.exec_s, campaign.deadline_s, BOARD_CORES, int(workers)
                )
                for workers in windows.features[test, 2]
            ]
        )
        time_scores.append(score("speed scaling (time)", windows.targets[test], classical))
    emit(f"  train / test windows  : {train.size} / {test.size}")
    for line in score_table(time_scores):
        emit("  " + line)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns the process exit status."""
    parser = _parser()
    arguments = parser.parse_args(argv)
    lines: list[str] = []

    def emit(text: str = "") -> None:
        lines.append(text)

    try:
        campaign = load_campaign(arguments.traces, origin=Origin(arguments.origin))
    except TraceDataAbsentError as error:
        print(f"TRACES ABSENT: {error}", file=sys.stderr)
        return 3
    except (ValueError, OSError) as error:
        print(f"INPUT ERROR: {error}", file=sys.stderr)
        return 2

    try:
        if arguments.command == "manifest":
            lines.extend(campaign_lines(campaign))
            probes = campaign.manifest.get("probes", [])
            if probes:
                emit("")
                emit("probe table (declared rule chose the ground-truth level from it)")
                emit(f"  {'n_cpu':>6}{'releases':>10}{'misses':>9}{'miss_rate':>14}")
                for probe in probes:
                    emit(
                        f"  {probe.get('n_cpu_stress', -1):>6}"
                        f"{probe.get('releases', -1):>10}"
                        f"{probe.get('misses', -1):>9}"
                        f"{probe.get('miss_rate', float('nan')):>14.6f}"
                    )
        elif arguments.command == "account":
            runs = [arguments.run] if arguments.run else list(campaign.provenance.runs)
            for index, run in enumerate(runs):
                if index:
                    emit("")
                lines.extend(account_lines(account(_run_or_fail(campaign, run))))
        elif arguments.command == "headline":
            jobs = _run_or_fail(campaign, arguments.run)
            lines.extend(margin_lines(mean_plus_margin(jobs.response_s, jobs.deadline_s)))
            emit("")
            lines.extend(
                interval_lines(
                    counted_rate_interval(jobs, arguments.resamples, arguments.seed)
                )
            )
            emit("")
            lines.extend(
                headline_lines(
                    understatement_factor(jobs, arguments.resamples, arguments.seed)
                )
            )
        elif arguments.command == "evt":
            jobs = _run_or_fail(campaign, arguments.run)
            fits = threshold_stability(jobs.response_s, jobs.deadline_s)
            emit(f"admissible candidate thresholds: {len(fits)}")
            emit(
                f"  {'quantile':>9}{'u_ms':>12}{'n_exc':>8}{'xi':>12}"
                f"{'beta_ms':>12}{'mod_beta_ms':>14}{'P(resp>D)':>14}"
            )
            for fit in fits:
                emit(
                    f"  {fit.quantile:>9.4f}{fit.threshold_s * 1e3:>12.6f}"
                    f"{fit.n_exceedances:>8}{fit.shape:>12.6f}"
                    f"{fit.scale * 1e3:>12.6f}{fit.modified_scale * 1e3:>14.6f}"
                    f"{fit.probability:>14.6e}"
                )
            emit("")
            lines.extend(evt_lines(choose_threshold(fits)))
            emit("")
            pair = miss_indicator(jobs)
            emit(f"releases in indicator   : {pair.size}")
        elif arguments.command == "learned":
            status = _learned(campaign, arguments, emit)
            if status:
                return status
    except TraceDataAbsentError as error:
        print(f"TRACES ABSENT: {error}", file=sys.stderr)
        return 3
    except (ValueError, KeyError, RuntimeError) as error:
        print(f"INPUT ERROR: {error}", file=sys.stderr)
        return 2

    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

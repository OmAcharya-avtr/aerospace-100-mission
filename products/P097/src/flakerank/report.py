"""Plain-text rendering of a :class:`~flakerank.ranking.FlakeRankReport`.

No ``print`` anywhere: every function here returns a string and the caller
decides where it goes. The CLI in :mod:`flakerank.__main__` is the only thing
in the package that writes to a stream.
"""

from __future__ import annotations

import math

from .disagreement import RankDisagreement
from .ranking import FlakeRankReport

__all__ = ["render_disagreement", "render_report", "render_top_table"]

_RULE = "=" * 78


def _number(value: float, digits: int = 4) -> str:
    """Format a float, rendering NaN as ``n/a`` rather than as a number."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:.{digits}f}"


def render_disagreement(disagreement: RankDisagreement, *, movers: int = 10) -> str:
    """Render one pairwise disagreement, with its intervals and movers.

    Args:
        disagreement: the measured disagreement.
        movers: how many movers to list.

    Returns:
        A multi-line string.
    """
    tau = disagreement.tau_b_interval
    rho = disagreement.rho_interval
    lines = [
        f"{disagreement.label_x}",
        f"    versus {disagreement.label_y}",
        f"  tests ranked: {disagreement.n_items}",
        (
            f"  Kendall tau-b : {_number(disagreement.tau_b, 6)}  "
            f"[{_number(tau.low, 6)}, {_number(tau.high, 6)}] "
            f"at {tau.confidence:.2f}, {tau.resamples} resamples of "
            f"{tau.n_units} {tau.resampling_unit}s"
        ),
        (
            f"  Spearman rho  : {_number(disagreement.rho, 6)}  "
            f"[{_number(rho.low, 6)}, {_number(rho.high, 6)}] "
            f"at {rho.confidence:.2f}, {rho.resamples} resamples of "
            f"{rho.n_units} {rho.resampling_unit}s"
        ),
        (
            f"  ties: pairs {disagreement.ties.n_pairs}, tied in x "
            f"{disagreement.ties.tied_x_pairs}, tied in y "
            f"{disagreement.ties.tied_y_pairs}, concordant "
            f"{disagreement.ties.concordant}, discordant "
            f"{disagreement.ties.discordant}"
        ),
        (
            f"  rank displacement: max {disagreement.max_displacement}, mean absolute "
            f"{_number(disagreement.mean_absolute_displacement, 3)}"
        ),
        (
            f"  top-{disagreement.top_k} overlap: {disagreement.top_k_overlap} "
            f"of {disagreement.top_k}"
        ),
    ]
    if movers and disagreement.movers:
        lines.append("  tests that move most:")
        lines.append(
            f"    {'test':<44} {'rank x':>7} {'rank y':>7} {'move':>6} "
            f"{'score x':>14} {'score y':>12}"
        )
        for mover in disagreement.movers[:movers]:
            lines.append(
                f"    {mover.name:<44} {mover.rank_x:>7} {mover.rank_y:>7} "
                f"{mover.displacement:>+6} {mover.score_x:>14.4f} {mover.score_y:>12.6f}"
            )
    return "\n".join(lines)


def render_top_table(report: FlakeRankReport, count: int = 10) -> str:
    """Render the top ``count`` tests by ground-truth cost, with all three scores.

    Args:
        report: the measured report.
        count: how many rows.

    Returns:
        A multi-line string. Each row carries the ground-truth interval, whose
        resampling unit is named in the header, because a cost without an
        interval is the thing this product exists to stop people shipping.
    """
    truth = report.truth
    names = truth.ranking.names
    index_of = {name: index for index, name in enumerate(names)}
    lines = [
        f"top {count} by accumulated wasted time, with the 95 per cent interval from "
        f"resampling {truth.resampling_unit}s",
        f"  {'test':<44} {'wasted s':>11} {'interval s':>24} {'analytic s':>12} {'rate':>8}",
    ]
    for name, score in truth.ranking.top(count):
        index = index_of[name]
        interval = truth.per_test_intervals[index]
        analytic = float(report.analytic.scores[index])
        rate = float(report.rate.scores[index])
        lines.append(
            f"  {name:<44} {score:>11.3f} "
            f"[{interval.low:>10.3f}, {interval.high:>10.3f}] {analytic:>12.3f} {rate:>8.5f}"
        )
    return "\n".join(lines)


def render_report(report: FlakeRankReport, *, count: int = 10, movers: int = 10) -> str:
    """Render a whole report as plain text.

    Args:
        report: the measured report.
        count: rows in the top table.
        movers: movers listed per pairwise comparison.

    Returns:
        A multi-line string.
    """
    bracket = report.bracket
    total = report.truth.total_interval
    lines = [
        _RULE,
        "flakerank -- tests ranked by the cost of their intermittency",
        _RULE,
        f"history               : {report.source}",
        f"tests ranked          : {report.n_tests}",
        f"sessions              : {report.n_sessions}",
        f"attempts recorded     : {report.n_attempts}",
        f"retry budget          : {report.retry_budget} attempts per invocation",
        "",
        "THE HEADLINE: how far the cost ranking and the flake-rate ranking disagree",
        "",
        render_disagreement(report.truth_vs_rate, movers=movers),
        "",
        (
            f"cost-weighted capture at top {report.top_k}: of the "
            f"{report.truth.total_seconds:.3f} s actually wasted, the flake-rate top "
            f"{report.top_k} holds {_number(report.capture_by_rate)}, the analytic cost "
            f"top {report.top_k} holds {_number(report.capture_by_analytic)}, and the "
            f"best possible top {report.top_k} holds {_number(report.capture_oracle)}"
        ),
        "",
        _RULE,
        "THE ANALYTIC COST MODEL AGAINST THE GROUND TRUTH",
        _RULE,
        render_disagreement(report.truth_vs_analytic, movers=movers),
        "",
        _RULE,
        "WHAT A USER CAN COMPUTE FROM ONE WINDOW: analytic against rate",
        _RULE,
        render_disagreement(report.analytic_vs_rate, movers=movers),
        "",
        _RULE,
        "GROUND TRUTH AND THE INDEPENDENTLY DERIVED BRACKET",
        _RULE,
        (
            f"accumulated wasted time: {report.truth.total_seconds:.3f} s, interval "
            f"[{total.low:.3f}, {total.high:.3f}] s at {total.confidence:.2f} from "
            f"{total.resamples} resamples of {total.n_units} {total.resampling_unit}s"
        ),
        f"failed attempts        : {report.truth.failed_attempts}",
        (
            f"order-statistic bracket: [{bracket.order_statistic_low:.3f}, "
            f"{bracket.order_statistic_high:.3f}] s from the failure count and the "
            f"extreme attempt durations"
        ),
        f"Cauchy-Schwarz upper   : {bracket.cauchy_schwarz_high:.3f} s",
        f"tightest upper bound   : {bracket.tightest_high_seconds:.3f} s",
        f"summation slack        : {bracket.slack_seconds:.3e} s",
        f"brackets the total     : {bracket.brackets}",
        (
            f"margins                : low {bracket.margin_low_seconds:.3f} s, high "
            f"{bracket.margin_high_seconds:.3f} s"
        ),
        "",
        _RULE,
        render_top_table(report, count=count),
        "",
        _RULE,
        "this software is research-grade. It is not flight-qualified, not certified",
        "and not approved for operational aerospace use.",
        _RULE,
    ]
    return "\n".join(lines)

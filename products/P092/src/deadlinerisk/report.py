"""Text report assembly for the CLI. No printing happens in this module.

Library code in this product does not print. Every function here returns a list
of lines and ``deadlinerisk.__main__`` is the only place that writes to a
stream.
"""

from __future__ import annotations

from .accounting import DeadlineAccount
from .baselines import MeanMarginResult, ThresholdChoice
from .bootstrap import RateInterval, UnderstatementFactor
from .predictor import Score
from .trace import Campaign


def campaign_lines(campaign: Campaign) -> list[str]:
    """Header lines: origin, board, period, deadline, payload, runs."""
    env = campaign.manifest.get("env", {})
    lines = [
        campaign.provenance.banner(),
        f"board            : {env.get('model', 'unrecorded')}",
        f"kernel/uname     : {env.get('uname', 'unrecorded')}",
        f"cores            : {env.get('nproc', 'unrecorded')}",
        f"governor         : {env.get('governor', 'unrecorded')}",
        f"cpuinfo_max_freq : {env.get('clk_max_khz', 'unrecorded')} kHz",
        f"sudo -n true     : exit {env.get('sudo_n_true', 'unrecorded')}",
        f"period T         : {campaign.period_s * 1e3:.3f} ms",
        f"deadline D       : {campaign.deadline_s * 1e3:.3f} ms (implicit, D = T)",
        f"payload_w        : {campaign.manifest.get('payload_w', 'unrecorded')}",
        f"gt_level         : {campaign.manifest.get('gt_level', 'unrecorded')}",
        f"gt_reason        : {campaign.manifest.get('gt_reason', 'unrecorded')}",
        f"manifest complete: {campaign.complete}",
        f"runs loaded      : {', '.join(campaign.provenance.runs)}",
    ]
    return lines


def account_lines(totals: DeadlineAccount) -> list[str]:
    """Release and miss reconstruction for one run."""
    return [
        f"run                     : {totals.run}",
        f"executed rows           : {totals.executed}",
        f"dropped releases        : {totals.dropped}",
        f"releases = sum(1+drop)  : {totals.releases}",
        f"misses   = sum(m+drop)  : {totals.misses}",
        f"counted miss rate       : {totals.miss_rate:.6e}",
        f"executed-only miss rate : {totals.executed_miss_rate:.6e}"
        "   (what a tool that ignored dropped releases would report)",
    ]


def margin_lines(result: MeanMarginResult) -> list[str]:
    """Baseline 1's own output, in the form a practitioner states it."""
    return [
        f"mean response           : {result.mean_s * 1e3:.6f} ms",
        f"sd response             : {result.sd_s * 1e3:.6f} ms",
        f"k with mean + k*sd = D  : {result.k_to_deadline:.6f}",
        f"Gaussian P(resp > D)    : {result.probability:.6e}",
        f"samples (executed jobs) : {result.n}",
    ]


def evt_lines(choice: ThresholdChoice) -> list[str]:
    """Baseline 2's chosen fit and the rule that chose it."""
    fit = choice.chosen
    return [
        f"declared rule           : {choice.rule}",
        f"chosen threshold u      : {fit.threshold_s * 1e3:.6f} ms "
        f"(sample quantile {fit.quantile:.4f})",
        f"exceedances at u        : {fit.n_exceedances} of {fit.n_total}",
        f"exceedance rate zeta    : {fit.zeta:.6e}",
        f"GPD shape xi            : {fit.shape:.6f}",
        f"GPD scale beta          : {fit.scale * 1e3:.6f} ms",
        f"modified scale          : {fit.modified_scale * 1e3:.6f} ms",
        f"GPD P(resp > D)         : {fit.probability:.6e}",
        f"endpoint below D        : {fit.endpoint_below_deadline}",
    ]


def interval_lines(interval: RateInterval) -> list[str]:
    """Baseline 3's counted rate with its moving-block interval."""
    block = interval.block
    return [
        f"counted miss rate       : {interval.rate:.6e}",
        f"{int(interval.confidence * 100)}% moving-block interval: "
        f"[{interval.low:.6e}, {interval.high:.6e}]",
        f"releases in series      : {block.n}",
        f"white-noise band 2/sqrt(n): {block.band:.6e}",
        "autocorrelation lags 0..5: "
        + ", ".join(f"{value:.6f}" for value in block.rho_head),
        f"cut-off lag L           : {block.lag_cut}",
        f"integrated tau          : {block.tau_integrated:.6f}",
        f"block length (releases) : {block.block_length}"
        + ("  [capped at n//20]" if block.capped else ""),
        f"resamples / seed        : {interval.resamples} / {interval.seed}",
    ]


def headline_lines(headline: UnderstatementFactor) -> list[str]:
    """The headline factor, and the sentence that follows from its interval."""
    verdict = (
        "the interval CONTAINS 1: on this board under this payload the "
        "mean-plus-margin rule is indistinguishable from adequate"
        if headline.interval_contains_one
        else (
            "the interval EXCLUDES 1: the rule's error is resolved by this campaign "
            + ("and it UNDERSTATES the counted rate" if headline.low > 1.0
               else "and it OVERSTATES the counted rate")
        )
    )
    return [
        f"counted miss rate       : {headline.counted_rate:.6e}",
        f"mean-plus-margin P      : {headline.margin_probability:.6e}",
        f"understatement factor   : {headline.factor:.6e}",
        f"{int(headline.confidence * 100)}% moving-block interval: "
        f"[{headline.low:.6e}, {headline.high:.6e}]",
        f"block length (releases) : {headline.block.block_length}",
        f"block length (rows)     : {headline.block_rows}",
        f"non-finite replicates   : {headline.n_degenerate} of {headline.resamples}",
        f"verdict                 : {verdict}",
    ]


def score_table(scores: list[Score]) -> list[str]:
    """One row per predictor per split, widest column first."""
    header = f"{'predictor':<34}{'n':>6}{'MAE':>12}{'RMSE':>12}{'bias':>12}{'cover':>8}"
    lines = [header, "-" * len(header)]
    for item in scores:
        cover = "n/a" if item.coverage != item.coverage else f"{item.coverage:.3f}"
        lines.append(
            f"{item.label:<34}{item.n_test:>6}{item.mae:>12.6f}"
            f"{item.rmse:>12.6f}{item.bias:>+12.6f}{cover:>8}"
        )
    return lines

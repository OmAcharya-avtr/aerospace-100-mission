"""The top-level API: rank the tests three ways and measure the disagreement.

The headline output of this product is the disagreement, not the ranking. A
team that ranks by flake rate and a team that ranks by expected wasted time
agree on what to fix exactly when the rank correlation between the two is 1;
everything below 1 is the part of the problem a flake-rate dashboard cannot
see, and the operational size of that part is the **cost-weighted capture**:
of the wasted time that actually happened, what fraction sits in the tests the
ranking put in its top ten.

:func:`rank_tests` returns all of it in one object: the three baselines, the
three pairwise disagreements, the movers, and the independently derived bracket
on the accumulated total.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .baselines import (
    GroundTruthCost,
    Ranking,
    accumulated_wasted_time,
    analytic_expected_cost,
    flake_rate_ranking,
)
from .bounds import BoundBracket, wasted_time_bracket
from .disagreement import RankDisagreement, rank_disagreement
from .history import RunHistory

__all__ = ["FlakeRankReport", "cost_capture", "rank_tests"]


def cost_capture(
    scores: np.ndarray,
    truth_seconds: np.ndarray,
    names: tuple[str, ...],
    top_k: int,
) -> float:
    """Fraction of the recorded wasted time held by the top ``top_k`` by score.

    The operational reading of a ranking: if a team works the top ``top_k``
    entries and fixes them, this is the fraction of the measured waste that
    goes away. Unlike a rank correlation it is weighted by cost rather than by
    test, which is the weighting the question actually has.

    Args:
        scores: one score per test; higher is ranked nearer position 1.
        truth_seconds: recorded wasted seconds per test, same order.
        names: test identifiers, same order, used to break score ties so the
            result is a deterministic function of the inputs.
        top_k: prefix length.

    Returns:
        A fraction in ``[0, 1]``, or ``nan`` when no time was wasted at all.

    Raises:
        ValueError: for mismatched lengths or a non-positive ``top_k``.
    """
    scores = np.asarray(scores, dtype=np.float64)
    truth = np.asarray(truth_seconds, dtype=np.float64)
    if scores.shape != truth.shape or scores.size != len(names):
        raise ValueError(
            f"scores, truth_seconds and names must be the same length, got "
            f"{scores.size}, {truth.size} and {len(names)}"
        )
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")
    total = float(truth.sum())
    if total <= 0.0:
        return float("nan")
    count = min(top_k, scores.size)
    order = sorted(range(scores.size), key=lambda index: (-scores[index], names[index]))[:count]
    return float(truth[order].sum()) / total


@dataclass(frozen=True)
class FlakeRankReport:
    """Everything :func:`rank_tests` measured over one history window.

    Attributes:
        source: the history file the window came from.
        n_tests: tests with at least ``minimum_attempts`` attempts.
        n_sessions: sessions in the window.
        n_attempts: attempts in the window.
        retry_budget: the history's recorded retry budget.
        rate: baseline 1, the flake-rate ranking.
        analytic: baseline 2, the analytic expected re-run cost.
        truth: baseline 3, accumulated wasted time with session bootstrap.
        truth_vs_rate: THE HEADLINE -- how far the ground-truth cost ranking
            and the flake-rate ranking disagree.
        truth_vs_analytic: how well the analytic cost model reproduces the
            ground-truth ranking.
        analytic_vs_rate: the disagreement a user can compute without a long
            history, since both sides need only one window.
        bracket: the independently derived bracket on the accumulated total.
        top_k: prefix length used for every capture figure.
        capture_by_rate: cost-weighted capture of the flake-rate top ``k``.
        capture_by_analytic: the same for the analytic ranking.
        capture_oracle: the ceiling, from ranking by the truth itself.
    """

    source: str
    n_tests: int
    n_sessions: int
    n_attempts: int
    retry_budget: int
    rate: Ranking
    analytic: Ranking
    truth: GroundTruthCost
    truth_vs_rate: RankDisagreement
    truth_vs_analytic: RankDisagreement
    analytic_vs_rate: RankDisagreement
    bracket: BoundBracket
    top_k: int
    capture_by_rate: float
    capture_by_analytic: float
    capture_oracle: float

    @property
    def capture_shortfall(self) -> float:
        """``capture_oracle - capture_by_rate``, the waste the rate ranking misses."""
        return self.capture_oracle - self.capture_by_rate


def rank_tests(
    history: RunHistory,
    *,
    minimum_attempts: int = 1,
    top_k: int = 10,
    top_movers: int = 10,
    resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 97044,
) -> FlakeRankReport:
    """Rank a history three ways and measure how far the rankings disagree.

    Args:
        history: the recorded history, or a window of it.
        minimum_attempts: tests with fewer attempts in the window are omitted
            from every ranking, so all three rank the same test set -- a rank
            correlation over two different test sets is not defined.
        top_k: prefix length for the capture figures and the top-k overlaps.
        top_movers: how many movers to report per pairwise comparison.
        resamples: bootstrap resamples for every reported interval.
        confidence: nominal coverage of every reported interval.
        seed: seed for ``numpy.random.default_rng``. Each interval is given a
            distinct offset from it, recorded on the interval, so that two
            intervals reported side by side are not the same draws.

    Returns:
        A :class:`FlakeRankReport`.

    Raises:
        ValueError: for an empty window or an invalid argument, propagated from
            the baseline and disagreement functions with their own messages.
    """
    summaries = history.summaries(minimum_attempts=minimum_attempts)
    rate = flake_rate_ranking(summaries)
    analytic = analytic_expected_cost(summaries, history.retry_budget)
    truth = accumulated_wasted_time(
        history,
        minimum_attempts=minimum_attempts,
        resamples=resamples,
        confidence=confidence,
        seed=seed,
    ).require_resampling_unit("session")

    if truth.ranking.names != rate.names:
        raise ValueError(
            "the three baselines were built over different test sets, which makes a "
            "rank correlation between them undefined; this is a defect in the "
            "minimum_attempts handling and not a user error"
        )

    names = rate.names
    common = {
        "top_movers": top_movers,
        "top_k": top_k,
        "resamples": resamples,
        "confidence": confidence,
    }
    truth_vs_rate = rank_disagreement(
        names, truth.ranking.scores, rate.scores,
        label_x="accumulated wasted time (ground truth)", label_y="flake rate",
        seed=seed + 10, **common,
    )
    truth_vs_analytic = rank_disagreement(
        names, truth.ranking.scores, analytic.scores,
        label_x="accumulated wasted time (ground truth)",
        label_y="analytic expected re-run cost",
        seed=seed + 20, **common,
    )
    analytic_vs_rate = rank_disagreement(
        names, analytic.scores, rate.scores,
        label_x="analytic expected re-run cost", label_y="flake rate",
        seed=seed + 30, **common,
    )

    truth_seconds = truth.ranking.scores
    return FlakeRankReport(
        source=history.source,
        n_tests=len(names),
        n_sessions=history.n_sessions,
        n_attempts=history.n_attempts,
        retry_budget=history.retry_budget,
        rate=rate,
        analytic=analytic,
        truth=truth,
        truth_vs_rate=truth_vs_rate,
        truth_vs_analytic=truth_vs_analytic,
        analytic_vs_rate=analytic_vs_rate,
        bracket=wasted_time_bracket(history, truth.total_seconds),
        top_k=top_k,
        capture_by_rate=cost_capture(rate.scores, truth_seconds, names, top_k),
        capture_by_analytic=cost_capture(analytic.scores, truth_seconds, names, top_k),
        capture_oracle=cost_capture(truth_seconds, truth_seconds, names, top_k),
    )

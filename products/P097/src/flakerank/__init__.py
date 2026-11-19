"""flakerank -- rank intermittently failing tests by the cost of their intermittency.

A flaky-test dashboard reports a rate and a first-seen date. A rate cannot say
which flake is worth fixing first, because the cost of a flake is the machine
time its re-runs waste, and that is the rate multiplied by the runtime. A
1-in-44 failure in a 600-second test wastes about 14 seconds per invocation and
scores 0.023 on the dashboard; a 1-in-2 failure in a 0.1-second test wastes
0.1 seconds and scores 0.5. The dashboard puts the second one 22 times higher
and the true ordering is the other way round by a factor of 140.

So this package ranks by expected wasted re-run time, and its headline output
is **how far that ranking disagrees with the rate ranking** -- Kendall tau-b
and Spearman rho with intervals, the identity of the tests that move most, and
the fraction of the measured waste each ranking's top ten actually holds.

The pieces:

* :mod:`flakerank.geometric` -- the closed-form geometric re-run count and
  expected wasted runtime, derived, with its assumptions and its divergence at
  ``p -> 1`` stated. The mathematical core.
* :mod:`flakerank.history` -- the committed recorded run history, its schema
  and its validation. Nothing in this package reads a clock.
* :mod:`flakerank.baselines` -- the three baselines: rank by flake rate (the
  thing being refuted), the analytic expected re-run cost, and the accumulated
  ground truth with session-bootstrap intervals.
* :mod:`flakerank.disagreement` -- Kendall tau-b with the tie correction of
  Kendall (1945), Spearman rho from midranks, bootstrap intervals and movers.
* :mod:`flakerank.bounds` -- a bracket on total wasted time derived from a
  failure count and order statistics, mathematics the accumulator does not use.
* :mod:`flakerank.resample` -- percentile bootstrap intervals that declare
  their resampling unit on the returned object.
* :mod:`flakerank.predictor` -- the learned intermittency-cost predictor with a
  confidence output, scored on held-out tests and held-out time separately.
* :mod:`flakerank.ranking` -- the top-level :func:`~flakerank.ranking.rank_tests`.
* :mod:`flakerank.report` -- plain-text rendering.

This software is research-grade. It is not flight-qualified, not certified and
not approved for operational aerospace use.
"""

from __future__ import annotations

from .baselines import (
    GroundTruthCost,
    Ranking,
    accumulated_wasted_time,
    analytic_expected_cost,
    flake_rate_ranking,
)
from .bounds import BoundBracket, wasted_time_bracket
from .disagreement import (
    RankDisagreement,
    kendall_tau_b,
    midranks,
    rank_disagreement,
    spearman_rho,
)
from .geometric import (
    expected_attempts_to_first_pass,
    expected_failed_attempts,
    expected_wasted_seconds,
)
from .history import RunHistory, TestSummary, history_sha256, load_history
from .ranking import FlakeRankReport, cost_capture, rank_tests
from .report import render_report
from .resample import BootstrapInterval

__version__ = "0.1.0"

__all__ = [
    "BootstrapInterval",
    "BoundBracket",
    "FlakeRankReport",
    "GroundTruthCost",
    "RankDisagreement",
    "Ranking",
    "RunHistory",
    "TestSummary",
    "__version__",
    "accumulated_wasted_time",
    "analytic_expected_cost",
    "cost_capture",
    "expected_attempts_to_first_pass",
    "expected_failed_attempts",
    "expected_wasted_seconds",
    "flake_rate_ranking",
    "history_sha256",
    "kendall_tau_b",
    "load_history",
    "midranks",
    "rank_disagreement",
    "rank_tests",
    "render_report",
    "spearman_rho",
    "wasted_time_bracket",
]

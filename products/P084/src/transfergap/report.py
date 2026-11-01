"""Plain-text rendering. No colour, no unicode box drawing, no dependencies.

Every renderer here puts the ASSUMPTION before the number. A bound printed
without the assumption it rests on is the thing this package exists to stop,
so the ``distribution-free`` column is never optional and the exchangeability
verdict is printed above anything that needs it.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .bounds import Bound
from .exchangeability import ExchangeabilityTest
from .plan import SamplePlan
from .scoring import CoverageScore

__all__ = [
    "format_table",
    "render_bounds",
    "render_coverage",
    "render_exchangeability",
    "render_plan",
]


def format_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    """Fixed-width text table. Columns are sized to their widest cell.

    A row whose length differs from ``headers`` raises :class:`ValueError`
    rather than being padded, because a silently padded row is a wrong table.
    """
    head = [str(header) for header in headers]
    body: list[list[str]] = []
    for index, row in enumerate(rows):
        cells = [str(cell) for cell in row]
        if len(cells) != len(head):
            raise ValueError(
                f"row {index} has {len(cells)} cells but there are {len(head)} headers"
            )
        body.append(cells)
    widths = [len(header) for header in head]
    for cells in body:
        for column, cell in enumerate(cells):
            widths[column] = max(widths[column], len(cell))
    lines = ["  ".join(header.ljust(widths[c]) for c, header in enumerate(head))]
    lines.append("  ".join("-" * width for width in widths))
    for cells in body:
        lines.append("  ".join(cell.ljust(widths[c]) for c, cell in enumerate(cells)))
    return "\n".join(lines)


def _number(value: float | None, digits: int = 4) -> str:
    """Render a float compactly, with ``None``, ``nan`` and ``inf`` spelled out."""
    if value is None:
        return "-"
    number = float(value)
    if np.isnan(number):
        return "nan"
    if np.isinf(number):
        return "inf" if number > 0 else "-inf"
    return f"{number:.{digits}f}"


def render_bounds(bounds: Sequence[Bound], unit: str = "") -> str:
    """A table of bounds, with target and standing in their own columns."""
    rows = []
    for bound in bounds:
        rows.append(
            [
                bound.method,
                bound.target,
                str(bound.n),
                _number(bound.confidence, 3),
                _number(bound.proportion, 3) if bound.proportion is not None else "-",
                _number(bound.estimate),
                _number(bound.upper),
                _number(bound.slack),
                "yes" if bound.distribution_free else "NO",
            ]
        )
    label = f" [{unit}]" if unit else ""
    return "\n".join(
        [
            format_table(
                [
                    "method",
                    "target",
                    "n",
                    "conf",
                    "prop",
                    f"estimate{label}",
                    f"upper{label}",
                    f"slack{label}",
                    "dist-free",
                ],
                rows,
            ),
            "",
            "target 'mean' bounds E[d]; target 'quantile' covers a future measurement.",
            "The two answer different questions and are not comparable row to row.",
            "dist-free = NO means the number rests on a distributional or asymptotic",
            "assumption and carries no finite-sample guarantee.",
        ]
    )


def render_exchangeability(tests: Sequence[ExchangeabilityTest]) -> str:
    """The permutation tests, with the verdict sentence under the table."""
    rows = [
        [
            test.statistic,
            str(test.n),
            str(test.permutations),
            _number(test.observed, 4),
            _number(test.null_mean, 4),
            _number(test.null_sd, 4),
            _number(test.p_value, 4),
            "REJECTED" if test.rejected else "not rejected",
        ]
        for test in tests
    ]
    lines = [
        format_table(
            ["statistic", "n", "perms", "observed", "null mean", "null sd", "p", "verdict"],
            rows,
        ),
        "",
        "A rejection is evidence AGAINST exchangeability: the bootstrap and split-",
        "conformal limits have no guarantee on that sample. A non-rejection is not",
        "evidence FOR exchangeability; these two statistics have power against drift",
        "and against a step change and little against anything else.",
    ]
    return "\n".join(lines)


def _verdict(score: CoverageScore) -> str:
    """The coverage verdict, or ``no limit`` where the method could not run.

    A method that refuses to produce a limit at a sample size equation (7) or
    (12) does not support has not under-covered; it has correctly declined,
    and printing ``UNDER-COVERS`` against a ``nan`` coverage would be reading
    a refusal as a failure.
    """
    if np.isnan(score.coverage):
        return "no limit"
    return "covers" if score.covers else "UNDER-COVERS"


def render_coverage(scores: Sequence[CoverageScore]) -> str:
    """Achieved coverage first, width second, verdict last."""
    rows = []
    for score in scores:
        rows.append(
            [
                score.method,
                str(score.n),
                _number(score.declared, 3),
                _number(score.coverage, 4),
                _number(score.coverage_se, 4),
                _number(score.tolerance_rate, 4) if score.tolerance_rate is not None else "-",
                _number(score.mean_upper, 3),
                str(score.failures),
                "yes" if score.distribution_free else "NO",
                _verdict(score),
            ]
        )
    unit = next((score.unit for score in scores if score.unit), "")
    label = f" [{unit}]" if unit else ""
    return "\n".join(
        [
            format_table(
                [
                    "method",
                    "n",
                    "declared",
                    "coverage",
                    "+/- se",
                    "tol rate",
                    f"mean upper{label}",
                    "fails",
                    "dist-free",
                    "verdict",
                ],
                rows,
            ),
            "",
            "coverage is the achieved rate; mean upper is the width. The verdict is on",
            "COVERAGE ONLY: a tighter limit that under-covers is worse than a loose one",
            "that holds. 'covers' means not measurably below the declared level, with a",
            "two-standard-error Monte Carlo allowance, not that coverage is proved.",
            "tol rate is the fraction of repetitions whose own coverage reached the",
            "declared proportion -- the training-conditional statement a tolerance",
            "interval makes and a marginal conformal interval does not.",
        ]
    )


def render_plan(plans: Sequence[SamplePlan]) -> str:
    """How many more measurements each method needs, with the basis in a column."""
    rows = [
        [
            plan.method,
            plan.basis,
            str(plan.current_n),
            str(plan.required_n),
            str(plan.additional_n),
            _number(plan.target_slack, 3),
            _number(plan.achieved_slack, 3),
            _number(plan.confidence, 3),
        ]
        for plan in plans
    ]
    unit = next((plan.unit for plan in plans if plan.unit), "")
    label = f" [{unit}]" if unit else ""
    lines = [
        format_table(
            [
                "method",
                "basis",
                "have",
                "need",
                "MORE",
                f"target slack{label}",
                f"slack at need{label}",
                "conf",
            ],
            rows,
        ),
        "",
    ]
    for plan in plans:
        lines.append(f"{plan.method}: {plan.note}")
    return "\n".join(lines)

"""Plain-text rendering. No colour, no unicode box drawing, no dependencies.

Every renderer here puts the identifiability verdict before the coefficients,
because that is the order a reader of a pointing-model fit needs: what the
geometry cannot separate, before what the fit appears to say about it.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .budget import PointingBudget
from .identifiability import IdentifiabilityReport
from .robust import Disagreement, RobustFit
from .terms import TERM_NAMES, term_table
from .wls import WlsFit

__all__ = [
    "format_table",
    "render_budget",
    "render_disagreement",
    "render_fit",
    "render_identifiability",
    "render_robust",
    "render_terms",
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


def _number(value: float, digits: int = 6) -> str:
    """Render a float compactly, with ``nan`` and ``inf`` spelled out."""
    if value is None:
        return "-"
    value = float(value)
    if np.isnan(value):
        return "nan"
    if np.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.{digits}g}"


def render_terms() -> str:
    """The declared term set, its units and its design-matrix columns."""
    return "\n".join(
        [
            "DECLARED POINTING MODEL TERMS",
            "",
            "Angles: azimuth A from north toward east, elevation E from the horizon.",
            "dX is cross-elevation (= dA cos E). Every coefficient is in arcseconds.",
            "rho_hat(E) is the refraction shape normalised to 1.0 at 45 deg elevation.",
            "",
            format_table(
                ["term", "unit", "meaning", "dX column", "dE column"], term_table()
            ),
        ]
    )


def render_identifiability(report: IdentifiabilityReport) -> str:
    """The rank verdict: spectrum, null directions, identifiable combinations."""
    lines = [
        "WHAT THIS GEOMETRY CANNOT DETERMINE",
        "",
        f"verdict             : {report.verdict}",
        f"numerical rank      : {report.numerical_rank} of {len(TERM_NAMES)}",
        f"condition number    : {_number(report.condition_number, 6)}",
        f"declared thresholds : null <= {report.null_threshold:g}, "
        f"marginal <= {report.marginal_threshold:g}  (ratios of s_k / s_1)",
        "",
        format_table(
            ["k", "singular value", "s_k / s_1", "class"],
            [
                [
                    str(index),
                    _number(float(value), 6),
                    _number(float(ratio), 4),
                    (
                        "not identifiable"
                        if ratio <= report.null_threshold
                        else "weakly determined"
                        if ratio <= report.marginal_threshold
                        else "determined"
                    ),
                ]
                for index, (value, ratio) in enumerate(
                    zip(report.singular_values, report.ratios, strict=True)
                )
            ],
        ),
        "",
        format_table(
            ["term", "variance inflation", "column scale"],
            [
                [name, _number(float(report.vif[i]), 6), _number(float(report.column_scales[i]), 6)]
                for i, name in enumerate(TERM_NAMES)
            ],
        ),
    ]
    if report.null_directions:
        lines.extend(
            [
                "",
                "NOT IDENTIFIABLE. Adding any multiple of a direction below to the",
                "coefficient vector leaves every predicted residual unchanged to within",
                "s_k / s_1 of the residual scale. The terms in it are not separable and",
                "their individual values are not reported as measurements.",
                "",
            ]
        )
        for direction in report.null_directions:
            lines.append(f"  [{direction.index}] {direction.statement()}")
        lines.append("")
        lines.append(f"  terms refused: {', '.join(report.unidentifiable_terms)}")
    else:
        lines.extend(["", "No direction falls below the declared null threshold."])
    if report.marginal_directions:
        lines.extend(["", "WEAKLY DETERMINED. Reported, and flagged.", ""])
        for direction in report.marginal_directions:
            lines.append(f"  [{direction.index}] {direction.statement()}")
    lines.extend(
        [
            "",
            "IDENTIFIABLE LINEAR COMBINATIONS, descending in determination. Weights are",
            "on the physical coefficients, normalised to unit length for reading; the",
            "precision column is |g| / se and is what says whether a combination was",
            "measured.",
            "",
        ]
    )
    for combination in report.combinations:
        mark = "    " if combination.determined else "  * "
        lines.append(
            f"{mark}[{combination.index}] s/s1={combination.ratio:.3e}  "
            f"{combination.statement()}"
        )
    lines.append("")
    lines.append("  * = at or below the marginal threshold; not a measurement.")
    return "\n".join(lines)


def render_fit(result: WlsFit) -> str:
    """The least-squares fit: coefficients, intervals, inflation, fit quality."""
    rows = []
    for index, name in enumerate(TERM_NAMES):
        rows.append(
            [
                name,
                _number(float(result.coefficients[index]), 6),
                _number(float(result.standard_errors[index]), 4),
                _number(float(result.confidence_intervals[index, 0]), 6),
                _number(float(result.confidence_intervals[index, 1]), 6),
                _number(float(result.vif[index]), 4),
            ]
        )
    return "\n".join(
        [
            f"WEIGHTED LEAST SQUARES ({result.method}, error model {result.error_model})",
            "",
            f"rows                : {result.residuals.size}",
            f"rank                : {result.rank} of {len(TERM_NAMES)}",
            f"chi square          : {_number(result.chi_square, 6)}  "
            f"dof {result.degrees_of_freedom}  reduced "
            f"{_number(result.reduced_chi_square, 6)}",
            f"residual rms        : {_number(result.rms_arcsec, 6)} arcsec",
            f"condition number    : {_number(result.condition_number, 6)}",
            f"interval multiplier : {_number(result.interval_multiplier, 5)} at "
            f"{100 * result.confidence_level:g} per cent",
            "",
            "The VIF column uses the full singular-value spectrum and is a statement",
            "about the DESIGN: an infinite or enormous value says the geometry carries",
            "no information about that term. The se column is a statement about the",
            "ESTIMATOR, which sets directions below the rank cut to zero and therefore",
            "has a smaller variance than the design supports, at the cost of a bias no",
            "standard error can show. The two disagreeing is the signal, not an error.",
            "",
            format_table(
                ["term", "arcsec", "se", "lower", "upper", "VIF"], rows
            ),
        ]
    )


def render_robust(result: RobustFit) -> str:
    """One robust fit: coefficients, asymptotic errors, downweighting."""
    detail = "  ".join(f"{key}={_number(value, 6)}" for key, value in sorted(result.detail.items()))
    rows = [
        [
            name,
            _number(float(result.coefficients[index]), 6),
            _number(float(result.standard_errors[index]), 4),
        ]
        for index, name in enumerate(TERM_NAMES)
    ]
    return "\n".join(
        [
            f"ROBUST FIT ({result.method})",
            "",
            f"detail              : {detail}",
            f"iterations          : {result.iterations}",
            f"downweighted rows   : {result.downweighted} of {result.residuals.size}",
            f"robust scale        : {_number(result.scale_arcsec, 6)} arcsec",
            "",
            "Standard errors are asymptotic. For RANSAC they are conditional on the",
            "consensus set, which was chosen using the data, and are optimistic.",
            "",
            format_table(["term", "arcsec", "se"], rows),
        ]
    )


def render_disagreement(delta: Disagreement, label: str) -> str:
    """The diagnostic. Neither fit is preferred here."""
    rows = []
    for index, name in enumerate(TERM_NAMES):
        meaningful = name not in set(delta.unidentifiable)
        rows.append(
            [
                name,
                _number(float(delta.difference[index]), 6),
                _number(float(delta.z_score[index]), 4) if meaningful else "refused",
            ]
        )
    return "\n".join(
        [
            f"DISAGREEMENT: least squares against {label}",
            "",
            f"predicted sky rms   : {_number(delta.prediction_rms_arcsec, 6)} arcsec",
            f"predicted sky max   : {_number(delta.prediction_max_arcsec, 6)} arcsec",
            f"threshold           : {_number(delta.threshold_sigma, 4)} x "
            f"{_number(delta.sigma_reference, 4)} arcsec",
            f"flagged             : {'YES' if delta.flagged else 'no'}",
            "",
            "A flag means the two fits describe measurably different skies. It does not",
            "say which one is right; that depends on what the downweighted observations",
            "are, which this package cannot determine.",
            "",
            format_table(["term", "difference arcsec", "z"], rows),
        ]
    )


def render_budget(budget: PointingBudget) -> str:
    """The whole decomposition, identifiability verdict first."""
    rows = []
    for row in budget.terms:
        rows.append(
            [
                row.name,
                _number(row.coefficient, 6),
                _number(row.standard_error, 4),
                f"[{_number(row.interval[0], 5)}, {_number(row.interval[1], 5)}]",
                _number(row.vif, 4),
                _number(row.contribution_rms_arcsec, 5),
                "yes" if row.separable else "REFUSED",
            ]
        )
    parts = [
        render_identifiability(budget.identifiability),
        "",
        "=" * 78,
        "",
        "POINTING ERROR BUDGET",
        "",
        f"declared sigma      : {_number(budget.sigma_arcsec, 4)} arcsec per row",
        f"observed rms        : {_number(budget.total_rms_arcsec, 6)} arcsec",
        f"post-fit rms        : {_number(budget.fit_rms_arcsec, 6)} arcsec",
        f"explained fraction  : {_number(budget.explained_fraction, 6)}",
        f"quadrature closure  : {_number(budget.quadrature_closure, 6)}  "
        "(1.0 only for an orthogonal design; the departure is the term correlation)",
        "",
        "A contribution is this column's own rms over the geometry that was observed,",
        "and nowhere else. For a REFUSED term the split of the contribution between",
        "the terms involved is as arbitrary as the split of the coefficients.",
        "",
        format_table(
            ["term", "arcsec", "se", f"{100 * budget.least_squares.confidence_level:g}% interval",
             "VIF", "contrib", "separable"],
            rows,
        ),
    ]
    if budget.refused_terms:
        parts.extend(
            [
                "",
                f"REFUSED per-term split for: {', '.join(budget.refused_terms)}. Read the",
                "identifiable combinations above instead.",
            ]
        )
    if budget.huber is not None and budget.huber_disagreement is not None:
        parts.extend(["", "=" * 78, "", render_robust(budget.huber), "",
                      render_disagreement(budget.huber_disagreement, "Huber")])
    if budget.ransac is not None and budget.ransac_disagreement is not None:
        parts.extend(["", "=" * 78, "", render_robust(budget.ransac), "",
                      render_disagreement(budget.ransac_disagreement, "RANSAC")])
    return "\n".join(parts)

"""Plain-text rendering. No colour, no box drawing, nothing beyond numpy.

Every table here prints the STANDING of a number beside it. "This parameter
moved by 6.1 per cent" and "this parameter cannot be determined from this
telemetry at all" are not the same kind of statement, and a table that showed
only a number would hide the difference this whole product is about. So the
identifiability table carries a ``verdict`` column, the attribution table
prints ``REFUSED`` where a number would otherwise go, and the profile table
prints whether the profile overrode the covariance.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from .attribution import Attribution
from .identifiability import IdentifiabilityReport
from .plant import PARAMETER_NAMES, PARAMETER_UNITS
from .profile import ProfileResult
from .telemetry import TelemetrySet

__all__ = [
    "format_table",
    "number",
    "render_attribution",
    "render_identifiability",
    "render_matrix",
    "render_profiles",
    "render_telemetry_sets",
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
    lines.extend(
        "  ".join(cell.ljust(widths[c]) for c, cell in enumerate(cells)) for cells in body
    )
    return "\n".join(lines)


def number(value: float | None, digits: int = 6) -> str:
    """Render a float compactly, with ``None``, ``nan`` and ``inf`` spelled out."""
    if value is None:
        return "-"
    quantity = float(value)
    if math.isnan(quantity):
        return "nan"
    if math.isinf(quantity):
        return "inf" if quantity > 0.0 else "-inf"
    if quantity != 0.0 and (abs(quantity) < 1.0e-4 or abs(quantity) >= 1.0e6):
        return f"{quantity:.{digits}e}"
    return f"{quantity:.{digits}f}"


def render_telemetry_sets(sets: dict[str, TelemetrySet]) -> str:
    """The declared telemetry sets: campaign, channels, rows, noise levels."""
    rows = []
    for name, telemetry in sets.items():
        rows.append(
            [
                name,
                telemetry.campaign.name,
                f"{telemetry.campaign.duration_s:.1f}",
                f"{telemetry.campaign.drive_duration_s:.1f}",
                f"{telemetry.campaign.drive_current_a:+.2f}",
                "+".join(telemetry.channels),
                str(telemetry.rows),
            ]
        )
    return format_table(
        ["set", "campaign", "length [s]", "driven [s]", "current [A]", "channels", "rows"],
        rows,
    )


def render_identifiability(report: IdentifiabilityReport) -> str:
    """Baseline 1: singular values, verdicts, bounds, and the combinations."""
    ratios = report.singular_values / report.singular_values[0]
    lines = [
        f"telemetry set            : {report.telemetry_name}",
        (
            f"numerical rank           : {report.rank} of 5 "
            f"(relative cutoff {report.rank_rtol:.1e})"
        ),
        (
            f"rank margin              : {report.rank_margin_decades:.4f} decades "
            "from the cutoff to the nearest singular-value ratio"
        ),
        f"condition number (rank)  : {report.condition_number:.6e}",
        (
            f"Fisher condition number  : {report.fisher_condition_number:.6e}  "
            "(the square; nothing here is computed from it)"
        ),
        (
            f"Fisher least eigenvalue  : {report.fisher_min_eigenvalue:+.6e}  "
            "(a Fisher matrix is positive semi-definite; a negative value here is "
            "rounding from forming A^T A)"
        ),
        "",
        format_table(
            ["i", "singular value", "ratio to largest", "in null space"],
            [
                [
                    str(index + 1),
                    number(float(value)),
                    f"{float(ratio):.6e}",
                    "yes" if index >= report.rank else "no",
                ]
                for index, (value, ratio) in enumerate(
                    zip(report.singular_values, ratios, strict=True)
                )
            ],
        ),
        "",
        format_table(
            ["parameter", "unit", "verdict", "null component", "CRB (relative)",
             "pinv diagonal (NOT a bound)"],
            [
                [
                    name,
                    unit,
                    verdict,
                    f"{float(component):.6e}",
                    number(float(bound)),
                    number(float(pinv)),
                ]
                for name, unit, verdict, component, bound, pinv in zip(
                    PARAMETER_NAMES,
                    PARAMETER_UNITS,
                    report.verdict,
                    report.null_component,
                    report.crb_relative,
                    report.pinv_diagonal_not_a_bound,
                    strict=True,
                )
            ],
        ),
        "",
        "identifiable combinations, equation (11), best determined first:",
        "",
        format_table(
            ["combination", "CRB on its relative standard error"],
            [[label, number(bound)] for label, bound in report.combinations()],
        ),
    ]
    if report.null_basis.size:
        lines += [
            "",
            "null space (orthonormal rows, logarithmic parameters):",
            "",
            format_table(
                ["direction", *PARAMETER_NAMES],
                [
                    [f"n{index + 1}", *[f"{value:+.6f}" for value in row]]
                    for index, row in enumerate(report.null_basis)
                ],
            ),
        ]
    return "\n".join(lines)


def render_attribution(attribution: Attribution) -> str:
    """Baseline 2: detection, the combinations, and the two single-parameter routes."""
    lines = [
        f"telemetry set            : {attribution.telemetry_name}",
        (
            f"residual chi-square      : {attribution.residual_chi2:.6f} on "
            f"{attribution.residual_rows} rows"
        ),
        f"detection threshold      : {attribution.detection_threshold:.6f}",
        f"detected                 : {'yes' if attribution.detected else 'no'}",
        f"chi-square after the fit : {attribution.fit_chi2:.6f}",
        "",
        "identifiable directions, equation (13):",
        "",
        format_table(
            ["combination", "estimate", "sd", "z", "95 per cent interval", "significant"],
            [
                [
                    direction.label,
                    f"{direction.estimate:+.6f}",
                    number(direction.standard_deviation),
                    f"{direction.z_score:+.4f}",
                    f"[{direction.interval[0]:+.6f}, {direction.interval[1]:+.6f}]",
                    "yes" if direction.significant else "no",
                ]
                for direction in attribution.directions
            ],
        ),
        "",
        "per-parameter attribution. REFUSED means baseline 1 found this parameter",
        "structurally unidentifiable from this telemetry set, and no residual can",
        "change that:",
        "",
        format_table(
            ["parameter", "estimate", "95 per cent half-width", "standing"],
            [
                [
                    name,
                    "REFUSED" if name in attribution.refused else f"{estimate:+.6f}",
                    "REFUSED" if name in attribution.refused else number(width),
                    "structurally unidentifiable"
                    if name in attribution.refused
                    else "estimated",
                ]
                for name, estimate, width in zip(
                    PARAMETER_NAMES,
                    attribution.parameter_estimate,
                    attribution.parameter_half_width,
                    strict=True,
                )
            ],
        ),
        "",
        "single-parameter routes. Both ASSUME exactly one parameter moved, which",
        "this telemetry set cannot verify:",
        "",
        format_table(
            ["parameter", "share, eq (14)", "chi2, eq (15)", "amplitude", "weight, eq (16)"],
            [
                [
                    name,
                    f"{share:.6f}",
                    f"{chi2:.6f}",
                    f"{amplitude:+.6f}",
                    f"{weight:.6f}",
                ]
                for name, share, chi2, amplitude, weight in zip(
                    PARAMETER_NAMES,
                    attribution.sensitivity_share,
                    attribution.constrained_chi2,
                    attribution.constrained_estimate,
                    attribution.constrained_weight,
                    strict=True,
                )
            ],
        ),
        "",
        f"first-order sensitivity attribution, eq (14) : {attribution.sensitivity_choice}",
        (
            "constrained maximum likelihood, eq (15)      : "
            f"{attribution.constrained_choice} "
            f"(relative weight {attribution.constrained_confidence:.6f})"
        ),
    ]
    family = attribution.equivalence_family()
    if family.shape[0] > 1:
        lines += [
            "",
            "THE ATTRIBUTION ABOVE IS NOT UNIQUE. Every row below produces identical",
            "noiseless telemetry on this set; the telemetry cannot distinguish them:",
            "",
            format_table(
                ["alternative", *PARAMETER_NAMES],
                [
                    [
                        "chosen" if index == 0 else f"equivalent {index}",
                        *[f"{value:+.6f}" for value in row],
                    ]
                    for index, row in enumerate(family)
                ],
            ),
        ]
    else:
        lines += [
            "",
            "This telemetry set has no structural degeneracy, so the single-parameter",
            "attribution above is unique given its own assumption.",
        ]
    return "\n".join(lines)


def render_profiles(results: Sequence[ProfileResult]) -> str:
    """Baseline 3: the profile table, including the override column."""
    rows = []
    for result in results:
        rows.append(
            [
                result.label,
                f"{result.grid_half_width:.6f}",
                f"{result.max_delta_chi2:.6e}",
                "FLAT" if result.flat else "curved",
                str(result.sides_crossed),
                "-" if result.lower_limit is None else f"{result.lower_limit:+.6e}",
                "-" if result.upper_limit is None else f"{result.upper_limit:+.6e}",
                number(result.wald_half_width),
                number(result.one_sided_ratio),
                "OVERRIDES" if result.overrides_covariance else "-",
                "truncated" if result.interval_truncated else "-",
            ]
        )
    return "\n".join(
        [
            format_table(
                [
                    "direction",
                    "grid half-width",
                    "max delta chi2",
                    "shape",
                    "sides",
                    "lower limit",
                    "upper limit",
                    "Wald half-width",
                    "profile/Wald",
                    "verdict",
                    "box",
                ],
                rows,
            ),
            "",
            "'sides' is how many of the two threshold crossings fall inside the grid.",
            "A profile with one side is the ordinary case on an ill-conditioned",
            "problem and its one-sided half-width is the only one the data support;",
            "'profile/Wald' scores the linearisation against it. OVERRIDES means the",
            "profile is flat while the linearised covariance claims a finite interval:",
            "the covariance number is wrong and the profile is the reference.",
            "'truncated' means the inner optimisation at a grid point bracketing a",
            "crossing hit the declared box, so the profile there is an upper bound on",
            "the true one and no limit is quoted.",
        ]
    )


def render_matrix(matrix: np.ndarray, labels: Sequence[str], corner: str = "true/pred") -> str:
    """A square integer matrix with labelled rows and columns."""
    array = np.asarray(matrix)
    names = [str(label) for label in labels]
    if array.shape != (len(names), len(names)):
        raise ValueError(f"matrix shape {array.shape} against {len(names)} labels")
    return format_table(
        [corner, *names],
        [[names[index], *[str(int(value)) for value in row]]
         for index, row in enumerate(array)],
    )

"""Plain-text rendering. No colour, no box drawing, no dependencies beyond numpy.

Every renderer here prints the **standing** of a number beside it. A dismissal
that rests on a proof and a dismissal that rests on a classifier's probability
are not the same kind of statement, and a table that showed only the verdict
would hide the difference that this whole product is about. So the certificate
table carries a ``proof`` column, the score table carries a ``sound`` column,
and the clock verdict is printed above anything that depends on a duration.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from .certificate import MissCertificate, RunCertificate
from .clock import ClockEstimate
from .resimulate import Resimulation
from .scoring import DecisionScore
from .verdict import RunVerdict

__all__ = [
    "format_table",
    "render_certificates",
    "render_clock",
    "render_resimulations",
    "render_run_certificate",
    "render_scores",
    "render_verdict",
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
    if math.isnan(number):
        return "nan"
    if math.isinf(number):
        return "inf" if number > 0 else "-inf"
    if number != 0.0 and (abs(number) < 1e-4 or abs(number) >= 1e6):
        return f"{number:.{digits}e}"
    return f"{number:.{digits}f}"


def render_certificates(
    certificates: Sequence[MissCertificate], limit: int | None = None
) -> str:
    """Table of per-miss certificates, with the proof standing in its own column."""
    shown = list(certificates) if limit is None else list(certificates)[:limit]
    rows = []
    for certificate in shown:
        event = certificate.event
        rows.append(
            [
                str(event.index),
                _number(event.overrun_s * 1e3, 3),
                _number(event.overrun_periods, 3),
                str(event.hold_periods),
                _number(math.degrees(event.input_step_rad), 4),
                str(event.steps_remaining),
                _number(certificate.bound),
                _number(certificate.margin),
                certificate.verdict,
                "yes" if certificate.dismissed else "-",
            ]
        )
    table = format_table(
        [
            "iter",
            "tau [ms]",
            "tau/T",
            "p",
            "du [deg]",
            "horizon",
            "bound",
            "margin",
            "verdict",
            "proof",
        ],
        rows,
    )
    trailer = [
        "",
        "bound is equation (13) in the declared metric; the tolerance is 1 by",
        "construction, so a bound below 1 is a dismissal. proof = yes means the",
        "dismissal is a theorem about the declared linearisation, not a measurement.",
        "UNDECIDED is not a claim that the miss mattered: it is the certificate",
        "declining, and re-simulation settles it.",
    ]
    if limit is not None and len(certificates) > limit:
        trailer.insert(1, f"showing {limit} of {len(certificates)} misses")
    return "\n".join([table, *trailer])


def render_run_certificate(certificate: RunCertificate) -> str:
    """Summary of a run certificate, refusals first."""
    lines: list[str] = []
    if certificate.refusals:
        lines.append("REFUSED -- the declared derivation does not apply to this run:")
        for reason in certificate.refusals:
            lines.append(f"  - {reason}")
        lines.append("")
    rows = [
        ["run", certificate.label],
        ["misses recorded", str(len(certificate.certificates))],
        ["dismissed in closed form", str(len(certificate.dismissed))],
        ["left undecided", str(len(certificate.undecided))],
        ["dismissal rate", _number(certificate.dismissal_rate)],
        ["aggregate bound, eq (15)", _number(certificate.aggregate_bound)],
        ["tolerance", _number(certificate.tolerance)],
        [
            "certified overrun, eq (14) [ms]",
            _number(certificate.certified_overrun_s * 1e3, 4),
        ],
    ]
    lines.append(format_table(["quantity", "value"], rows))
    lines.append("")
    lines.append(
        "the certified overrun is evaluated at this run's LARGEST input step and "
        "full horizon,"
    )
    lines.append(
        "so it is the conservative single threshold: any miss shorter than it is "
        "dismissed"
    )
    lines.append("whatever else the run was doing at the time.")
    return "\n".join(lines)


def render_resimulations(results: Sequence[Resimulation]) -> str:
    """Table of re-simulated answers, with the cost in plant evaluations."""
    rows = []
    for result in results:
        rows.append(
            [
                str(result.event.index),
                _number(result.event.overrun_s * 1e3, 3),
                _number(result.peak_divergence),
                str(result.peak_index),
                str(result.steps_simulated),
                str(result.plant_calls),
                result.verdict,
            ]
        )
    return "\n".join(
        [
            format_table(
                [
                    "iter",
                    "tau [ms]",
                    "peak divergence",
                    "at iter",
                    "steps",
                    "plant calls",
                    "verdict",
                ],
                rows,
            ),
            "",
            "peak divergence is the largest weighted-norm difference between the",
            "with-miss and the miss-free trajectory, in the declared metric. This is",
            "the reference every other number in this product is scored against.",
        ]
    )


def render_clock(estimate: ClockEstimate) -> str:
    """The clock verdict above the per-segment fits, never below them."""
    lines = [
        f"CLOCK VERDICT: {estimate.verdict}",
        f"  pooled drift      : {_number(estimate.drift_ppm, 4)} ppm "
        f"+- {_number(estimate.drift_ppm_standard_error, 4)}",
        f"  declared allowance: {_number(estimate.allowance_ppm, 4)} ppm",
        f"  resets detected   : {list(estimate.resets) or 'none'}",
        "",
    ]
    rows = [
        [
            str(segment.start),
            str(segment.stop),
            str(segment.samples),
            _number(segment.drift_ppm, 4),
            _number(segment.drift_ppm_standard_error, 4),
            _number(segment.drift_ppm_theil_sen, 4),
        ]
        for segment in estimate.segments
    ]
    lines.append(
        format_table(
            ["start", "stop", "samples", "drift [ppm]", "se [ppm]", "theil-sen [ppm]"],
            rows,
        )
    )
    lines.append("")
    lines.append(
        "a reset makes every duration spanning it meaningless, so a detected reset"
    )
    lines.append(
        "makes the run undecidable whatever the fitted rates look like. A drift is"
    )
    lines.append("correctable and the correction changes which iterations appear to miss.")
    return "\n".join(lines)


def render_scores(scores: Sequence[DecisionScore]) -> str:
    """Comparison table, recall first and the proof standing in its own column."""
    rows = []
    for score in scores:
        counts = score.counts
        rows.append(
            [
                score.method,
                str(counts.total),
                str(counts.positives),
                _number(counts.recall),
                f"[{_number(score.recall_interval[0])}, "
                f"{_number(score.recall_interval[1])}]",
                str(score.missed_consequential),
                _number(counts.flag_rate),
                _number(counts.dismissal_rate),
                "yes" if score.sound else "NO",
            ]
        )
    return "\n".join(
        [
            format_table(
                [
                    "method",
                    "misses",
                    "consequential",
                    "recall",
                    "recall 95% interval",
                    "missed",
                    "flag rate",
                    "dismissal rate",
                    "proof",
                ],
                rows,
            ),
            "",
            "recall is of CONSEQUENTIAL misses and is the quantity that decides.",
            "'missed' counts consequential misses the method dismissed: the",
            "asymmetric error. 'proof' = yes means the dismissals are a theorem",
            "about the declared linearisation; NO means they rest on a measurement",
            "or a prediction and carry the sampling error of the interval beside",
            "them.",
        ]
    )


def render_verdict(verdict: RunVerdict) -> str:
    """The run verdict, reasons first, then the numbers behind it."""
    lines = [f"RUN VERDICT: {verdict.verdict}  ({verdict.label})"]
    for reason in verdict.reasons:
        lines.append(f"  - {reason}")
    if not verdict.reasons:
        lines.append(
            "  - every recorded miss was dismissed in closed form or shown "
            "inconsequential"
        )
    lines.append("")
    rows = [
        ["misses recorded", str(verdict.misses)],
        ["dismissed by the certificate", str(verdict.dismissed)],
        ["re-simulated", str(verdict.resimulated)],
        ["consequential", str(len(verdict.consequential))],
        ["re-simulations saved", _number(verdict.resimulation_saving)],
        ["aggregate bound, eq (15)", _number(verdict.aggregate_bound)],
        ["aggregate below tolerance", "yes" if verdict.aggregate_dismissed else "NO"],
        ["mean utilisation", _number(verdict.utilisation)],
        ["cascade / direct overruns", _number(verdict.cascade_ratio, 3)],
        ["clock", verdict.clock.verdict],
    ]
    lines.append(format_table(["quantity", "value"], rows))
    lines.append("")
    lines.append(
        "TIMING_FAITHFUL is an evidence statement about the declared plant,"
    )
    lines.append(
        "controller, metric and tolerance. It is not a certification, not an"
    )
    lines.append(
        "airworthiness statement, and not a claim about any real rig. A cascade"
    )
    lines.append(
        "ratio well above 1 means the run is further from the declared"
    )
    lines.append("non-cascading miss model than the single word above suggests.")
    return "\n".join(lines)


def render_array(name: str, values: np.ndarray, digits: int = 4) -> str:
    """One line of numbers, for a short array in a transcript."""
    array = np.asarray(values, dtype=float).ravel()
    return f"{name}: " + " ".join(_number(value, digits) for value in array)

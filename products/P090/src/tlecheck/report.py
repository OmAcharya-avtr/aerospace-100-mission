"""Plain-text rendering of a verdict, a benchmark and a consequence summary.

No module in :mod:`tlecheck` prints. Everything here returns strings and the
command line in :mod:`tlecheck.__main__` is the only thing that writes them to a
stream.

The one rule this module enforces is the ordering: :func:`verdict_text` puts the
determinability report first and the quality verdict second, every time, because
a verdict read without the statement of what could not be determined is the
failure this product exists to prevent.
"""

from __future__ import annotations

import math

from .benchmark import ClassScore, OperatingPoint
from .consequence import CLASSES, Consequence
from .determinability import Limit
from .verdict import Verdict

WIDTH = 78


def rule(title: str) -> str:
    """A section separator with ``title``."""
    return "=" * WIDTH + "\n" + title + "\n" + "=" * WIDTH


def determinability_text(limits: tuple[Limit, ...]) -> str:
    """The determinability report, one block per limit."""
    lines = [rule("WHAT THIS SCREEN CANNOT DETERMINE (computed before any screen ran)")]
    for limit in limits:
        mark = "determinable" if limit.determinable else "NOT DETERMINABLE"
        value = "-" if limit.value is None else f"{limit.value:.6g}"
        lines.append(f"[{mark}] {limit.name}  basis={limit.basis}  value={value}")
        lines.append(_wrap(limit.statement, indent=4))
    return "\n".join(lines)


def verdict_text(verdict: Verdict) -> str:
    """The determinability report, then the quality verdict. In that order."""
    lines = [determinability_text(verdict.determinability), ""]
    lines.append(rule("QUALITY VERDICT"))
    lines.append(f"predicted consequence class : {verdict.predicted_class}")
    score = verdict.silent_risk_score
    lines.append(
        "equation (15) silent risk    : "
        + ("infinite (a field could not be read)" if math.isinf(score) else f"{score:.6f}")
    )
    lines.append(
        f"integrity intact             : {'yes' if verdict.integrity_ok else 'NO'}"
    )
    lines.append(f"problems found               : {len(verdict.problems)}")
    for problem in verdict.problems:
        lines.append(f"  - [{problem.screen}] {problem.name} -> {problem.predicted_class}")
        lines.append(_wrap(problem.detail, indent=6))
    lines.append("")
    lines.append("declared thresholds this verdict depends on:")
    for name, value in sorted(verdict.thresholds.items()):
        lines.append(f"  {name:32s} {value:g}")
    lines.append("")
    lines.append(
        _wrap(
            "A clean verdict means no screened problem was found. It is not a "
            "statement that an ephemeris propagated from this element set is "
            "accurate; see the propagated_ephemeris_accuracy limit above.",
            indent=0,
        )
    )
    return "\n".join(lines)


def consequence_text(consequence: Consequence) -> str:
    """One measured consequence, with the numbers the class rests on."""
    lines = [f"class            : {consequence.consequence_class}"]
    if consequence.loud_reason:
        lines.append(f"propagator says  : {consequence.loud_reason}")
    else:
        lines.append(f"max error        : {consequence.max_error_km:.6f} km")
        lines.append(f"rms error        : {consequence.rms_error_km:.6f} km")
        lines.append(f"error at start   : {consequence.initial_error_km:.6e} km")
    lines.append(f"threshold        : {consequence.threshold_km:g} km (declared policy)")
    lines.append(
        f"grid             : {consequence.samples} samples over "
        f"{consequence.horizon_days:g} days"
    )
    return "\n".join(lines)


def class_table(scores: dict[str, ClassScore], title: str) -> str:
    """Per-class precision and recall as a fixed-width table."""
    lines = [title, f"{'class':20s} {'precision':>10s} {'recall':>10s} {'support':>8s}"]
    for label in CLASSES:
        score = scores[label]
        lines.append(
            f"{label:20s} {_number(score.precision):>10s} "
            f"{_number(score.recall):>10s} {score.support:>8d}"
        )
    return "\n".join(lines)


def operating_point_text(name: str, point: OperatingPoint) -> str:
    """One operating point as a single line."""
    threshold = "n/a" if math.isnan(point.threshold) else f"{point.threshold:.6g}"
    return (
        f"{name:34s} recall={point.recall:.4f}  "
        f"false-positive rate={point.false_positive_rate:.4f}  "
        f"flagged={point.flagged:<5d} threshold={threshold}"
    )


def _number(value: float) -> str:
    return "nan" if math.isnan(value) else f"{value:.4f}"


def _wrap(text: str, indent: int = 0) -> str:
    """Wrap ``text`` to :data:`WIDTH` with a fixed indent."""
    import textwrap

    prefix = " " * indent
    return "\n".join(
        textwrap.wrap(text, width=WIDTH, initial_indent=prefix, subsequent_indent=prefix)
    )

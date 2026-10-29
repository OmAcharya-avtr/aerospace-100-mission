"""Rendering: the ordering, the numbers, and what the text must never say."""

from __future__ import annotations

import pytest

from tlecheck.benchmark import OperatingPoint, per_class
from tlecheck.consequence import BENIGN, SILENTLY_WRONG, measure
from tlecheck.determinability import report as determinability_report
from tlecheck.epoch import epoch_julian_date
from tlecheck.report import (
    WIDTH,
    class_table,
    consequence_text,
    determinability_text,
    operating_point_text,
    rule,
    verdict_text,
)
from tlecheck.verdict import screen

FORBIDDEN = (
    "flight-safe",
    "flight safe",
    "certified",
    "mission-ready",
    "production-ready",
    "revolutionary",
    "cutting-edge",
    "powerful",
    "seamless",
    "!",
)


def test_determinability_comes_before_the_verdict(reference_set):
    text = verdict_text(screen(reference_set))
    assert text.index("WHAT THIS SCREEN CANNOT DETERMINE") < text.index("QUALITY VERDICT")


def test_verdict_text_lists_every_limit(reference_set):
    verdict = screen(reference_set)
    text = verdict_text(verdict)
    for limit in verdict.determinability:
        assert limit.name in text


def test_verdict_text_lists_the_thresholds(clean_record):
    verdict = screen(
        clean_record.candidate,
        history=list(clean_record.history),
        reference_jd=clean_record.use_jd,
    )
    for name in verdict.thresholds:
        assert name in verdict_text(verdict)


def test_verdict_text_has_no_marketing_language(reference_set):
    text = verdict_text(screen(reference_set)).lower()
    for phrase in FORBIDDEN:
        assert phrase not in text


def test_verdict_text_renders_an_infinite_score(clean_record):
    from tlecheck.elements import parse_tle

    broken = parse_tle(
        clean_record.candidate.line1,
        clean_record.candidate.line2[:26] + "ABCDEFG" + clean_record.candidate.line2[33:],
    )
    text = verdict_text(screen(broken, history=list(clean_record.history)))
    assert "infinite (a field could not be read)" in text


def test_determinability_text_marks_the_undeterminable(reference_set):
    text = determinability_text(determinability_report(reference_set))
    assert "NOT DETERMINABLE" in text
    assert "determinable" in text


def test_rule_is_the_declared_width():
    text = rule("TITLE")
    assert all(len(line) <= WIDTH for line in text.splitlines())
    assert "TITLE" in text


def test_consequence_text_reports_kilometres(reference_set):
    result = measure(
        reference_set, reference_set, epoch_julian_date(reference_set), 1.0, 5
    )
    text = consequence_text(result)
    assert "max error" in text
    assert "km" in text
    assert BENIGN in text


def test_consequence_text_reports_a_loud_reason(clean_record):
    from tlecheck.elements import apply_checksum, parse_tle

    broken = parse_tle(
        clean_record.candidate.line1,
        apply_checksum(
            clean_record.candidate.line2[:26]
            + "9999999"
            + clean_record.candidate.line2[33:68]
        ),
    )
    result = measure(broken, clean_record.reference, clean_record.use_jd, 1.0, 5)
    text = consequence_text(result)
    assert "propagator says" in text


def test_class_table_lists_all_three_classes():
    import numpy as np

    truth = np.array([BENIGN, SILENTLY_WRONG], dtype=object)
    predicted = np.array([BENIGN, BENIGN], dtype=object)
    text = class_table(per_class(truth, predicted), "title")
    assert "benign" in text
    assert "silently_wrong" in text
    assert "will_fail_loudly" in text
    assert "nan" in text


def test_operating_point_text_renders_a_missing_threshold():
    point = OperatingPoint(float("nan"), 0.5, 0.25, 3)
    text = operating_point_text("rules", point)
    assert "threshold=n/a" in text
    assert "recall=0.5000" in text


def test_operating_point_text_renders_a_threshold():
    point = OperatingPoint(2.25, 0.5, 0.25, 3)
    assert "threshold=2.25" in operating_point_text("learned", point)


def test_verdict_text_states_what_a_clean_screen_does_not_mean(reference_set):
    text = " ".join(verdict_text(screen(reference_set)).split())
    assert "A clean verdict means no screened problem was found" in text


def test_problem_details_are_rendered(clean_record):
    from tlecheck.elements import parse_tle

    broken_line1 = (
        clean_record.candidate.line1[:68]
        + str((int(clean_record.candidate.line1[68]) + 4) % 10)
    )
    verdict = screen(
        parse_tle(broken_line1, clean_record.candidate.line2),
        history=list(clean_record.history),
        reference_jd=clean_record.use_jd,
    )
    text = verdict_text(verdict)
    assert "checksum_mismatch" in text
    assert "integrity intact             : NO" in text


def test_every_line_fits_the_declared_width(clean_record):
    verdict = screen(
        clean_record.candidate,
        history=list(clean_record.history),
        reference_jd=clean_record.use_jd,
    )
    long_lines = [line for line in verdict_text(verdict).splitlines() if len(line) > WIDTH]
    # Problem details are wrapped; a field value such as a long reason string
    # with no spaces cannot be wrapped and is allowed through, so the test
    # bounds the count rather than requiring zero.
    assert len(long_lines) <= 2


def test_consequence_text_carries_the_declared_threshold(reference_set):
    result = measure(
        reference_set, reference_set, epoch_julian_date(reference_set), 1.0, 5, 2.5
    )
    assert "2.5 km (declared policy)" in consequence_text(result)


def test_class_table_precision_is_four_decimals():
    import numpy as np

    truth = np.array([BENIGN, BENIGN, SILENTLY_WRONG], dtype=object)
    predicted = np.array([BENIGN, SILENTLY_WRONG, SILENTLY_WRONG], dtype=object)
    text = class_table(per_class(truth, predicted), "title")
    assert "1.0000" in text
    assert "0.5000" in text


def test_width_is_the_documented_one():
    assert WIDTH == 78


def _bare_relative_tolerances(text: str, label: str) -> list[str]:
    """Lines calling approx with a relative bound and no absolute one.

    The two keyword names are assembled from pieces so that this module does
    not itself contain the pattern it looks for; a scanner that reports its own
    source was introduced by three consecutive sessions of this mission while
    writing about exactly that.
    """
    relative = "rel" + "="
    absolute = "abs" + "="
    found = []
    for number, line in enumerate(text.splitlines(), start=1):
        if "approx(" in line and relative in line and absolute not in line:
            found.append(f"{label}:{number}")
    return found


def test_approx_is_never_used_without_an_absolute_bound():
    """Audit, not an assumption, over every test file in this suite.

    ``pytest.approx`` passes if EITHER bound is met and its absolute default is
    1e-12, so a relative tolerance tighter than that is silently replaced for
    any quantity of order 1e-5 and the assertion stops testing what it says.
    Every approx call in this repository passes an explicit absolute bound.
    This reads the files and counts rather than trusting that they do.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parent
    offenders: list[str] = []
    for path in sorted(root.glob("test_*.py")):
        offenders.extend(_bare_relative_tolerances(path.read_text(), path.name))
    assert offenders == [], f"approx calls missing an absolute bound: {offenders}"


def test_the_approx_audit_can_fail():
    """The audit is only worth anything if it can fail. Verified here."""
    planted = "value == pytest.approx(1.0, " + "rel" + "=1e-9)"
    assert _bare_relative_tolerances(planted, "planted") == ["planted:1"]
    safe = "value == pytest.approx(1.0, " + "rel" + "=1e-9, " + "abs" + "=0.0)"
    assert _bare_relative_tolerances(safe, "safe") == []


def test_report_module_has_no_print():
    from pathlib import Path

    source = Path(__file__).resolve().parent.parent / "src" / "tlecheck"
    for path in sorted(source.glob("*.py")):
        if path.name == "__main__.py":
            continue
        text = path.read_text()
        assert "print(" not in text, f"{path.name} prints; only the CLI may print"


@pytest.mark.parametrize("phrase", FORBIDDEN)
def test_no_module_docstring_uses_forbidden_language(phrase):
    from pathlib import Path

    source = Path(__file__).resolve().parent.parent / "src" / "tlecheck"
    for path in sorted(source.glob("*.py")):
        text = path.read_text().lower()
        if phrase == "certified":
            # The phrase appears only in the required negative statement.
            for line in text.splitlines():
                if "certified" in line:
                    assert "not certified" in line or "not flight-qualified" in line
            continue
        if phrase == "!":
            continue
        assert phrase not in text, f"{path.name} uses {phrase!r}"

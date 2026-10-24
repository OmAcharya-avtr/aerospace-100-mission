"""Report rendering tests. The verdict must be legible and must say FAILED when it did."""

from __future__ import annotations

import numpy as np

from boundcompress.certificate import certify
from boundcompress.codec import encode
from boundcompress.report import format_certificate, format_table


def test_certificate_report_says_holds(signals: dict[str, np.ndarray]) -> None:
    result = encode(signals["battery_voltage"], "abs", 1e-3)
    text = format_certificate(result.certificate)
    assert text.splitlines()[0].endswith("HOLDS")
    assert "reference, not a floor" in text
    assert result.certificate.digest in text


def test_certificate_report_says_failed_and_lists_the_index() -> None:
    values = np.linspace(1.0, 2.0, 50)
    corrupted = values.copy()
    corrupted[7] += 1.0
    text = format_certificate(certify(values, corrupted, "abs", 0.01))
    assert text.splitlines()[0].endswith("FAILED")
    assert "first violating indices: [7]" in text


def test_relative_mode_omits_the_effective_absolute_bound() -> None:
    values = np.array([2.0, 4.0, 8.0])
    text = format_certificate(certify(values, values.copy(), "rel", 0.01))
    assert "effective abs bound" not in text


def test_notes_are_rendered() -> None:
    values = np.array([1.0, 2.0])
    certificate = certify(values, values.copy(), "abs", 0.1, notes=("a deliberate note",))
    assert "note                   : a deliberate note" in format_certificate(certificate)


def test_table_columns_align() -> None:
    text = format_table(["a", "long_header"], [["xx", "1"], ["y", "22"]])
    lines = text.splitlines()
    assert len(lines) == 4
    assert lines[1].startswith("--")
    assert all(len(line) == len(lines[0]) for line in lines)

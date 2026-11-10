"""Table and line rendering. No library module prints; this one formats."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from sitediverse.report import (
    format_table,
    interval,
    number,
    plus_minus,
    scientific,
    verdict,
)


def test_a_table_has_a_header_a_rule_and_one_line_per_row():
    """Structural."""
    text = format_table(["a", "bb"], [["1", "2"], ["33", "4"]])
    lines = text.splitlines()
    assert len(lines) == 4
    assert lines[0].startswith("a")
    assert set(lines[1]) <= {"-", " "}


def test_a_table_column_is_at_least_as_wide_as_its_widest_cell():
    """So that nothing is truncated."""
    text = format_table(["a"], [["longvalue"]])
    assert "longvalue" in text


def test_a_table_with_no_rows_still_renders_its_header():
    """An empty result is a result."""
    text = format_table(["alpha", "beta"], [])
    assert text.splitlines()[0].startswith("alpha")
    assert len(text.splitlines()) == 2


def test_a_table_rejects_a_row_with_the_wrong_number_of_cells():
    """A silently padded row in a validation output is a wrong number.

    The message names the offending row index, which is what makes the error
    actionable in a script that builds a hundred rows.
    """
    with pytest.raises(ValueError, match="row 1 has 1 cells"):
        format_table(["a", "b"], [["1", "2"], ["3"]])


def test_a_table_rejects_no_columns():
    """Input validation."""
    with pytest.raises(ValueError, match="at least one column"):
        format_table([], [])


def test_a_table_has_no_trailing_whitespace():
    """So that a committed output file does not carry invisible differences."""
    text = format_table(["a", "bbbbb"], [["1", "2"]])
    for line in text.splitlines():
        assert line == line.rstrip()


def test_number_renders_the_declared_number_of_decimals():
    """Six by default."""
    assert number(0.5) == "0.500000"
    assert number(0.5, 2) == "0.50"


def test_number_spells_out_the_non_finite_values():
    """A rendered ``nan`` must be readable as ``nan``, not as a blank or a zero."""
    assert number(float("nan")) == "nan"
    assert number(float("inf")) == "inf"
    assert number(float("-inf")) == "-inf"


def test_scientific_spells_out_the_non_finite_values():
    """The same for the scientific form."""
    assert scientific(float("nan")) == "nan"
    assert scientific(float("inf")) == "inf"
    assert scientific(float("-inf")) == "-inf"


def test_scientific_renders_a_small_number_without_losing_it():
    """A value of 1e-21 must not render as 0.000000."""
    assert "e-21" in scientific(1.234e-21)


def test_an_interval_renders_both_ends():
    """Structural."""
    assert interval(0.1, 0.2) == "[0.100000, 0.200000]"


def test_plus_minus_renders_the_form_every_availability_figure_takes():
    """The specification requires the uncertainty beside every figure."""
    assert plus_minus(0.02, 0.001) == "0.020000 +/- 0.001000"


def test_verdict_is_pass_or_fail_and_nothing_else():
    """So that a validation output can be grepped."""
    assert verdict(True) == "PASS"
    assert verdict(False) == "FAIL"


@given(st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False))
def test_number_round_trips_through_float(value):
    """The rendering is a faithful fixed-point form to its stated precision."""
    rendered = number(value, 9)
    assert float(rendered) == pytest.approx(value, rel=1e-8, abs=1e-8)


@given(st.floats(1e-30, 1e30, allow_nan=False, allow_infinity=False))
def test_scientific_round_trips_through_float(value):
    """The same for the scientific form, over thirty decades each way."""
    rendered = scientific(value, 12)
    assert float(rendered) == pytest.approx(value, rel=1e-11, abs=0.0)


def test_a_numpy_scalar_renders_like_a_python_float():
    """Everything in this package produces numpy scalars."""
    assert number(np.float64(0.25)) == "0.250000"
    assert scientific(np.float64(2.5e-7)) == "2.500000e-07"

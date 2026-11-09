"""Rendering: the table, and that every renderer produces the standing of its numbers."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from twindiverge.attribution import CONFIDENCE_LEVEL, attribute
from twindiverge.plant import NOMINAL, perturb
from twindiverge.profile import grid_half_width_for, profile_parameters
from twindiverge.report import (
    format_table,
    number,
    render_attribution,
    render_identifiability,
    render_matrix,
    render_profiles,
    render_telemetry_sets,
)
from twindiverge.telemetry import TELEMETRY_SETS, measure, residual_vector


def test_format_table_sizes_columns_to_the_widest_cell():
    text = format_table(["a", "bbbb"], [["xxxxx", "y"]])
    lines = text.splitlines()
    assert len(lines) == 3
    assert lines[0].startswith("a      bbbb")
    assert set(lines[1]) <= {"-", " "}


def test_format_table_refuses_a_row_of_the_wrong_width():
    with pytest.raises(ValueError, match="row 0 has 1 cells"):
        format_table(["a", "b"], [["x"]])


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, "-"),
        (float("nan"), "nan"),
        (float("inf"), "inf"),
        (float("-inf"), "-inf"),
        (0.0, "0.000000"),
        (1.0, "1.000000"),
    ],
)
def test_number_spells_out_the_special_cases(value, expected: str):
    assert number(value) == expected


def test_number_switches_to_exponential_for_extreme_magnitudes():
    assert "e" in number(1.0e-9)
    assert "e" in number(1.0e9)


def test_render_telemetry_sets_names_every_declared_set():
    text = render_telemetry_sets(TELEMETRY_SETS)
    for name in TELEMETRY_SETS:
        assert name in text


def test_render_identifiability_prints_the_verdict_and_the_warning(short_report):
    text = render_identifiability(short_report)
    assert "REFUSED" in text
    assert "pinv diagonal (NOT a bound)" in text
    assert "null space" in text
    assert "identifiable combinations" in text


def test_render_identifiability_omits_the_null_block_when_there_is_none(torque_report):
    text = render_identifiability(torque_report)
    assert "null space (orthonormal rows" not in text


def test_render_attribution_prints_refused_instead_of_a_number(short_set, short_report):
    offset = np.zeros(5)
    offset[0] = 0.06
    observed, _ = measure(short_set, perturb(NOMINAL, offset), seed=3)
    from twindiverge.telemetry import design_matrix

    _, twin = design_matrix(short_set, NOMINAL)
    result = attribute(
        short_set, short_report, residual_vector(short_set, observed, twin), twin=twin
    )
    text = render_attribution(result)
    assert "REFUSED" in text
    assert "NOT UNIQUE" in text
    assert "eq (14)" in text and "eq (15)" in text


def test_render_attribution_says_so_when_the_attribution_is_unique(
    torque_set, torque_report
):
    offset = np.zeros(5)
    offset[0] = 0.06
    observed, _ = measure(torque_set, perturb(NOMINAL, offset), seed=3)
    from twindiverge.telemetry import design_matrix

    _, twin = design_matrix(torque_set, NOMINAL)
    text = render_attribution(
        attribute(
            torque_set,
            torque_report,
            residual_vector(torque_set, observed, twin),
            twin=twin,
        )
    )
    assert "no structural degeneracy" in text
    assert "NOT UNIQUE" not in text


def test_render_profiles_prints_the_override_verdict(short_set, short_report):
    critical = float(stats.norm.ppf(0.5 + CONFIDENCE_LEVEL / 2.0))
    observed, _ = measure(short_set, NOMINAL, seed=3)
    results = profile_parameters(
        short_set,
        observed,
        critical * short_report.pinv_diagonal_not_a_bound,
        points=3,
    )
    text = render_profiles(results)
    assert "OVERRIDES" in text
    assert "FLAT" in text
    assert grid_half_width_for(1.0) > 0.0


def test_render_matrix_labels_rows_and_columns():
    text = render_matrix(np.array([[1, 2], [3, 4]]), ["a", "b"])
    assert "true/pred" in text
    assert "a" in text and "b" in text


def test_render_matrix_refuses_a_mismatched_label_count():
    with pytest.raises(ValueError, match="matrix shape"):
        render_matrix(np.zeros((2, 2), dtype=int), ["only one"])

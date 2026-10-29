"""The determinability report, including an independent combinatorial cross-check.

:func:`test_double_substitution_matches_closed_form` is the bound-check class of
test the mission's build guide requires: the enumeration in
:func:`tlecheck.determinability.checksum_detection_fractions` loops over every
pair of digit columns and every pair of replacement digits, while the closed
form below counts the same set by a congruence argument that performs no
enumeration at all. They agree exactly, and the closed form also proves the
answer is independent of the digits of the particular line, which the
enumeration cannot show.
"""

from __future__ import annotations

import math
from fractions import Fraction

import pytest

from tlecheck.determinability import (
    COMPUTED,
    DATA_DEPENDENT,
    STRUCTURAL,
    bound_slack,
    checksum_detection_fractions,
    report,
)
from tlecheck.elements import format_tle, parse_tle

from conftest import REFERENCE_FIELDS, REFERENCE_LINE1, REFERENCE_LINE2

EXPECTED_LIMIT_NAMES = (
    "propagated_ephemeris_accuracy",
    "which_source_is_correct",
    "element_covariance",
    "checksum_blindness_to_transposition",
    "checksum_blindness_to_double_substitution",
    "bound_slack_in_history_scale",
    "continuity_resolution_km",
    "history_consistency",
    "cross_source_divergence",
    "epoch_staleness_requirement",
    "epoch_year_window",
)


def test_single_substitution_is_always_detected():
    fractions = checksum_detection_fractions(REFERENCE_LINE1)
    assert fractions["single_substitution"] == pytest.approx(1.0, rel=0.0, abs=0.0)


def test_transposition_is_never_detected():
    fractions = checksum_detection_fractions(REFERENCE_LINE1)
    assert fractions["transposition"] == pytest.approx(0.0, rel=0.0, abs=0.0)


def test_double_substitution_matches_closed_form():
    """Independent derivation, by congruence rather than by enumeration.

    For one digit column holding ``d``, the nine replacements ``d' != d`` give
    the nine distinct non-zero residues ``d' - d`` modulo 10. For a pair of
    columns the checksum is unchanged exactly when the two residues sum to zero
    modulo 10, and for each of the nine choices at the first column there is
    exactly one at the second. So 9 of every 81 pairs are undetected and the
    detected fraction is 72/81 = 8/9 for every line, whatever its digits.
    """
    closed_form = Fraction(72, 81)
    for line in (REFERENCE_LINE1, REFERENCE_LINE2):
        measured = checksum_detection_fractions(line)["double_substitution"]
        assert measured == pytest.approx(float(closed_form), rel=0.0, abs=1.0e-12)
    assert float(closed_form) == pytest.approx(0.8888888888888888, rel=0.0, abs=1.0e-15)


def test_double_substitution_is_independent_of_the_digits():
    """The closed form says the answer does not depend on the line. Checked."""
    first = checksum_detection_fractions(REFERENCE_LINE1)["double_substitution"]
    altered = REFERENCE_LINE1[:20] + "999999999999" + REFERENCE_LINE1[32:]
    second = checksum_detection_fractions(altered)["double_substitution"]
    assert first == pytest.approx(second, rel=0.0, abs=0.0)


def test_family_sizes_are_reported():
    fractions = checksum_detection_fractions(REFERENCE_LINE1)
    digits = fractions["digit_columns"]
    assert fractions["single_substitution_count"] == pytest.approx(
        9.0 * digits, rel=0.0, abs=0.0
    )
    # Two-column substitutions: C(n, 2) pairs times 81 replacement pairs.
    pairs = digits * (digits - 1) / 2
    assert fractions["double_substitution_count"] == pytest.approx(
        81.0 * pairs, rel=0.0, abs=0.0
    )


def test_checksum_fractions_reject_a_short_line():
    with pytest.raises(ValueError, match="columns"):
        checksum_detection_fractions(REFERENCE_LINE1[:40])


def test_bound_slack_is_infinite_without_history(reference_set):
    slacks = bound_slack(reference_set, [])
    assert all(value == math.inf for value in slacks.values())


def test_bound_slack_is_finite_with_history(clean_record):
    slacks = bound_slack(clean_record.candidate, list(clean_record.history))
    assert any(math.isfinite(value) for value in slacks.values())


def test_bound_slack_covers_the_history_tested_elements(clean_record):
    slacks = bound_slack(clean_record.candidate, list(clean_record.history))
    assert set(slacks) == {
        "inclination_deg",
        "eccentricity",
        "raan_deg",
        "mean_motion_rev_per_day",
        "bstar",
    }


def test_report_order_is_fixed(reference_set):
    limits = report(reference_set)
    assert tuple(limit.name for limit in limits) == EXPECTED_LIMIT_NAMES


def test_report_has_eleven_limits(reference_set):
    assert len(report(reference_set)) == len(EXPECTED_LIMIT_NAMES)


def test_structural_limits_are_never_determinable(reference_set):
    limits = {limit.name: limit for limit in report(reference_set)}
    for name in (
        "propagated_ephemeris_accuracy",
        "which_source_is_correct",
        "element_covariance",
    ):
        assert limits[name].basis == STRUCTURAL
        assert limits[name].determinable is False
        assert limits[name].value is None


def test_ephemeris_accuracy_limit_says_what_a_clean_screen_does_not_mean(reference_set):
    limits = {limit.name: limit for limit in report(reference_set)}
    statement = limits["propagated_ephemeris_accuracy"].statement
    assert "not a statement about ephemeris accuracy" in statement


def test_history_limit_is_data_dependent(clean_record):
    limits = {
        limit.name: limit
        for limit in report(clean_record.candidate, list(clean_record.history))
    }
    assert limits["history_consistency"].basis == DATA_DEPENDENT
    assert limits["history_consistency"].determinable is True
    assert limits["history_consistency"].value == float(len(clean_record.history))


def test_history_limit_not_determinable_without_history(reference_set):
    limits = {limit.name: limit for limit in report(reference_set, [])}
    assert limits["history_consistency"].determinable is False


def test_cross_source_limit_tracks_the_source_count(reference_set):
    one = {limit.name: limit for limit in report(reference_set, sources=1)}
    two = {limit.name: limit for limit in report(reference_set, sources=2)}
    assert one["cross_source_divergence"].determinable is False
    assert two["cross_source_divergence"].determinable is True


def test_epoch_requirement_limit_tracks_the_declaration(reference_set):
    without = {limit.name: limit for limit in report(reference_set)}
    with_requirement = {
        limit.name: limit for limit in report(reference_set, epoch_requirement_days=3.0)
    }
    assert without["epoch_staleness_requirement"].determinable is False
    assert with_requirement["epoch_staleness_requirement"].determinable is True
    assert with_requirement["epoch_staleness_requirement"].value == 3.0


def test_epoch_year_window_limit_states_the_cliff(reference_set):
    limits = {limit.name: limit for limit in report(reference_set)}
    limit = limits["epoch_year_window"]
    assert limit.basis == COMPUTED
    assert limit.value == 2056.0
    assert "2057" in limit.statement


def test_continuity_resolution_not_determinable_without_history(reference_set):
    limits = {limit.name: limit for limit in report(reference_set, [])}
    limit = limits["continuity_resolution_km"]
    assert limit.determinable is False
    assert math.isnan(limit.value)


def test_continuity_resolution_determinable_with_history(clean_record):
    limits = {
        limit.name: limit
        for limit in report(clean_record.candidate, list(clean_record.history))
    }
    limit = limits["continuity_resolution_km"]
    assert limit.determinable is True
    assert limit.value > 0.0


def test_resolution_statement_matches_the_declared_threshold(clean_record):
    limits = {
        limit.name: limit
        for limit in report(
            clean_record.candidate, list(clean_record.history), silent_threshold_km=1.0e-9
        )
    }
    assert "does NOT resolve it" in limits["continuity_resolution_km"].statement


def test_report_tolerates_a_short_line():
    element_set = parse_tle("1 25544U", "2 25544")
    limits = report(element_set)
    assert len(limits) == len(EXPECTED_LIMIT_NAMES)


def test_bound_slack_nan_for_unparsable_element():
    broken_line2 = REFERENCE_LINE2[:26] + "ABCDEFG" + REFERENCE_LINE2[33:]
    element_set = parse_tle(REFERENCE_LINE1, broken_line2)
    slacks = bound_slack(element_set, [])
    assert math.isnan(slacks["eccentricity"])


def test_report_on_a_formatted_element_set():
    element_set = parse_tle(*format_tle(**REFERENCE_FIELDS))
    limits = report(element_set)
    assert limits[0].name == "propagated_ephemeris_accuracy"

"""Features: the declared configurations, finiteness, and what is kept out."""

from __future__ import annotations

import numpy as np
import pytest

from tlecheck import features
from tlecheck.features import (
    BASE_NAMES,
    CONFIGURATIONS,
    HISTORY_NAMES,
    RULE_NAMES,
    Z_CAP,
    build,
    feature_names,
    row,
)
from tlecheck.screens import run


def test_three_declared_configurations():
    assert set(CONFIGURATIONS) == {
        "elements_and_history",
        "elements_only",
        "with_rule_score",
    }


def test_elements_only_is_a_prefix_of_the_others():
    assert feature_names("elements_only") == BASE_NAMES
    assert feature_names("elements_and_history")[: len(BASE_NAMES)] == BASE_NAMES


def test_with_rule_score_adds_exactly_the_rule_score():
    full = feature_names("with_rule_score")
    assert full == BASE_NAMES + HISTORY_NAMES + RULE_NAMES
    assert RULE_NAMES == ("rule_silent_risk_score",)


def test_feature_names_rejects_an_unknown_configuration():
    with pytest.raises(ValueError, match="unknown configuration"):
        feature_names("everything")


def test_names_are_unique():
    for name in CONFIGURATIONS:
        columns = feature_names(name)
        assert len(set(columns)) == len(columns)


def test_no_feature_names_the_reference_element_set():
    """The uncorrupted reference is not available operationally and is not a feature."""
    for name in CONFIGURATIONS:
        for column in feature_names(name):
            assert "reference" not in column
            assert "truth" not in column
            assert "corruption" not in column


def test_row_has_every_column_of_every_configuration(clean_record):
    evidence = run(
        clean_record.candidate,
        history=list(clean_record.history),
        reference_jd=clean_record.use_jd,
    )
    values = row(clean_record, evidence)
    for name in CONFIGURATIONS:
        for column in feature_names(name):
            assert column in values


def test_matrix_shape(prepared_test):
    for name, columns in CONFIGURATIONS.items():
        assert prepared_test.matrices[name].shape == (
            len(prepared_test.records),
            len(columns),
        )


def test_matrix_is_finite(prepared_test):
    for matrix in prepared_test.matrices.values():
        assert np.isfinite(matrix).all()


def test_infinite_z_is_capped_and_flagged(clean_record):
    evidence = run(clean_record.candidate, history=list(clean_record.history))
    values = row(clean_record, evidence)
    for column, value in values.items():
        if column.startswith("z_") and not column.endswith("_infinite"):
            assert value <= Z_CAP


def test_checksum_flags_are_zero_or_one(prepared_test):
    columns = feature_names("elements_and_history")
    matrix = prepared_test.matrices["elements_and_history"]
    for name in ("checksum_line1_ok", "checksum_line2_ok"):
        values = matrix[:, columns.index(name)]
        assert set(np.unique(values)) <= {0.0, 1.0}


def test_clean_records_have_valid_checksums_as_features(prepared_test):
    columns = feature_names("elements_and_history")
    matrix = prepared_test.matrices["elements_and_history"]
    clean = np.array([r.corruption == "none" for r in prepared_test.records])
    assert np.all(matrix[clean][:, columns.index("checksum_line1_ok")] == 1.0)
    assert np.all(matrix[clean][:, columns.index("checksum_line2_ok")] == 1.0)


def test_build_rejects_mismatched_lengths(prepared_test):
    with pytest.raises(ValueError, match="evidence objects supplied"):
        build(prepared_test.records, prepared_test.evidences[:-1])


def test_build_rejects_an_unknown_configuration(prepared_test):
    with pytest.raises(ValueError, match="unknown configuration"):
        build(prepared_test.records, prepared_test.evidences, "nonsense")


def test_bstar_sign_feature_is_a_sign(prepared_test):
    columns = feature_names("elements_and_history")
    values = prepared_test.matrices["elements_and_history"][:, columns.index("bstar_sign")]
    assert set(np.unique(values)) <= {-1.0, 0.0, 1.0}


def test_rule_score_feature_is_capped(prepared_test):
    columns = feature_names("with_rule_score")
    values = prepared_test.matrices["with_rule_score"][
        :, columns.index("rule_silent_risk_score")
    ]
    assert np.isfinite(values).all()
    assert values.max() <= Z_CAP


def test_continuity_features_are_present_so_the_model_sees_what_the_rules_see():
    columns = feature_names("elements_and_history")
    for name in (
        "continuity_offset_km",
        "continuity_median_km",
        "continuity_scale_km",
        "continuity_z",
        "continuity_excess_km",
        "continuity_resolution_km",
        "continuity_resolves_threshold",
    ):
        assert name in columns


def test_elements_only_has_no_history_or_continuity_features():
    columns = feature_names("elements_only")
    assert not any(column.startswith("z_") for column in columns)
    assert not any("continuity" in column for column in columns)
    assert not any("cross_source" in column for column in columns)


def test_z_cap_is_the_documented_value():
    assert Z_CAP == 1000.0


def test_feature_module_exposes_tested_elements():
    assert features.TESTED_ELEMENTS == (
        "inclination_deg",
        "eccentricity",
        "raan_deg",
        "mean_motion_rev_per_day",
        "bstar",
    )

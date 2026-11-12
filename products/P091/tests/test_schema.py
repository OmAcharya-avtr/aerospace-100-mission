"""The binding schema: header validation and the inferred temperature scale."""

from __future__ import annotations

import math

import pytest

from thermalbudget import schema


def test_expected_columns_is_the_schema_of_the_spec():
    columns = schema.expected_columns()
    assert columns[:7] == schema.FIXED_COLUMNS
    assert columns[-1] == "gpu_hz"
    assert schema.DIE_COLUMN in columns
    # 7 fixed + 6 zones + 6 cores + gpu_hz = 20, which is the width of the
    # measured cold-burst header quoted in docs/SPEC.md.
    assert len(columns) == 20


def test_the_three_unreadable_zones_are_not_in_the_header():
    columns = schema.expected_columns()
    for zone in schema.UNREADABLE_ZONES:
        assert f"temp_{zone}" not in columns


def test_a_present_unreadable_zone_is_rejected_by_name():
    columns = (*schema.expected_columns(), "temp_cv0-thermal")
    with pytest.raises(schema.TraceSchemaError, match="cv0-thermal"):
        schema.validate_header(columns)


def test_header_validation_lists_every_violation_not_just_the_first():
    with pytest.raises(schema.TraceSchemaError) as error:
        schema.validate_header(("t_wall", "t_rel", "nonsense"))
    message = str(error.value)
    assert "fixed prefix" in message
    assert "die-temperature column" in message
    assert "no cpu<i>_khz column" in message


def test_duplicate_column_is_rejected():
    columns = (*schema.expected_columns(), "gpu_hz")
    with pytest.raises(schema.TraceSchemaError, match="duplicated"):
        schema.validate_header(columns)


def test_a_header_with_a_different_zone_count_is_still_valid():
    # The campaign discovers readable zones at run time, so a board exposing
    # four readable zones writes a valid four-zone trace.
    columns = schema.expected_columns(zones=("cpu-thermal", "tj-thermal"), n_cores=4)
    schema.validate_header(columns)
    assert len(columns) == 7 + 2 + 4 + 1


def test_milli_degree_scale_is_inferred_from_the_real_measured_rows():
    # The five real rows quoted in docs/SPEC.md, temp_tj-thermal column.
    raw = [45875.0, 46343.0, 46906.0, 47781.0, 48687.0]
    assert schema.infer_temperature_divisor(raw) == 1000.0
    assert math.isclose(raw[0] / 1000.0, 45.875)


def test_degree_scale_is_inferred_when_a_file_is_already_converted():
    assert schema.infer_temperature_divisor([45.875, 46.343, 48.687]) == 1.0


def test_the_missing_sentinel_is_excluded_before_inferring_the_scale():
    # A single -1 among degree-scale samples must not drag the median out of band.
    assert schema.infer_temperature_divisor([45.0, 46.0, -1.0, 47.0]) == 1.0


def test_all_missing_column_refuses_rather_than_guessing():
    with pytest.raises(schema.TraceSchemaError, match="no finite non-sentinel"):
        schema.infer_temperature_divisor([-1.0, -1.0])


def test_an_unrecognisable_scale_refuses_rather_than_guessing():
    # 500 is neither a plausible degree reading nor a milli-degree one. Guessing
    # here would scale every thermal figure by a thousand.
    with pytest.raises(schema.TraceSchemaError, match="neither the degree band"):
        schema.infer_temperature_divisor([500.0, 505.0, 510.0])


def test_sentinel_columns_are_exactly_the_sysfs_read_columns():
    columns = schema.expected_columns()
    sentinel = set(schema.sentinel_columns(columns))
    assert "temp_tj-thermal" in sentinel
    assert "cpu3_khz" in sentinel
    assert "gpu_hz" in sentinel
    assert "foreign_cpu_pct" in sentinel
    # Campaign-computed columns are NOT sentinel-coded: a negative there is a
    # damaged file, not a missing reading.
    for name in ("t_wall", "t_rel", "iters_cum", "iters_per_s", "achieved_duty", "loadavg1"):
        assert name not in sentinel

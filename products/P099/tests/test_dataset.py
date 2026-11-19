"""The committed record: schema, invariants, round trip, the two splits."""

from __future__ import annotations

import json
import os

import pytest

from passyield.dataset import (
    COLUMNS,
    DEFAULT_HELDOUT_STATIONS,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_RECORD_PATH,
    STATIONS,
    PassRow,
    Split,
    events_of,
    feature_matrix,
    file_digest,
    format_intervals,
    load_passes,
    parse_intervals,
    station_by_name,
    station_split,
    target_vector,
    time_split,
    window_of,
    write_passes,
)
from passyield.predictor import FEATURE_NAMES


def _home_prefix() -> str:
    """``<sep>home<sep>``, assembled rather than written as a literal.

    An absolute home-directory path in any committed file fails release-gate
    check 8. The scanner that enforces that lives in
    ``tests/test_determinism.py`` and assembles its needle from ``os.sep`` so
    that its own source does not match; this helper does the same, so a test
    that CHECKS for an absolute path does not become one.
    """
    return os.sep + "home" + os.sep




def test_record_and_manifest_agree_on_the_digest():
    """The committed record's SHA-256 matches the generator's manifest.

    The record is the product's evidence, so a record that has drifted from
    its manifest is a defect whether or not any test fails. The digest and
    the pass count are both checked.
    """
    manifest = json.loads(DEFAULT_MANIFEST_PATH.read_text())
    assert file_digest(DEFAULT_RECORD_PATH) == manifest["record_sha256"]
    assert manifest["record_bytes"] == DEFAULT_RECORD_PATH.stat().st_size
    assert manifest["record_relative_path"] == "data/passes.csv"


def test_record_loads_and_every_row_satisfies_its_invariants(record):
    """Loading validates; a record that loads has already been checked."""
    assert len(record) == 1990
    assert {row.station for row in record} == {station.name for station in STATIONS}
    for row in record:
        assert row.duration_s > 0.0
        assert row.capacity_mbit > 0.0
        assert 0.0 <= row.realised_mbit <= row.capacity_mbit * (1.0 + 1.0e-9)
        assert 0.0 <= row.realised_fraction <= 1.0 + 1.0e-9


def test_record_pass_count_matches_the_manifest(record):
    """The manifest's count is the number of rows."""
    manifest = json.loads(DEFAULT_MANIFEST_PATH.read_text())
    assert manifest["passes"] == len(record)


def test_columns_are_derived_from_the_dataclass():
    """Schema and dataclass cannot drift: one is computed from the other."""
    assert COLUMNS[0] == "pass_id"
    assert "residual_mbit" in COLUMNS
    assert len(COLUMNS) == len(set(COLUMNS))


def test_interval_field_round_trip():
    """``format_intervals`` and ``parse_intervals`` are inverses to 9 decimals."""
    intervals = ((1.5, 2.25), (100.0, 0.125))
    assert parse_intervals(format_intervals(intervals)) == intervals
    assert parse_intervals("") == ()
    assert format_intervals(((1.5, 2.25),)) == "1.500000000:2.250000000"


def test_interval_field_rejects_a_malformed_value():
    """A malformed field is refused with the expected format named."""
    with pytest.raises(ValueError, match="start:duration"):
        parse_intervals("1.0")


def test_window_round_trips_to_full_double_precision(record):
    """The rise instant survives the CSV exactly.

    The record writes ``rise_jd`` and ``rise_fr`` at seventeen significant
    digits for this reason: at nine digits the mask crossing resolved to about
    9 ms, which moved the endpoint elevation by 2e-4 degrees and tripped the
    mask-clamp guard on reconstruction.
    """
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as directory:
        target = Path(directory) / "round_trip.csv"
        write_passes(record[:20], target)
        rebuilt = load_passes(target)
    for original, copy in zip(record[:20], rebuilt, strict=True):
        assert copy.rise_jd == original.rise_jd
        assert copy.rise_fr == original.rise_fr
        assert window_of(copy) == window_of(original)


def test_events_round_trip_from_the_record(record):
    """The event timeline is reconstructed exactly, not just its totals.

    Storing only the totals would make the committed decomposition
    unfalsifiable from the committed record.
    """
    row = next(r for r in record if r.outage_count >= 2 and r.idle_intervals)
    events = events_of(row)
    assert len(events.resync_intervals_s) == row.outage_count
    assert events.acquisition_s == row.acquisition_s
    assert events.resync_total_s == pytest.approx(row.resync_total_s, rel=1.0e-12, abs=0.0)
    assert events.idle_total_s == pytest.approx(row.idle_total_s, rel=1.0e-12, abs=0.0)


def test_missing_record_raises_with_a_relative_path():
    """The error message names a relative path, never an absolute one.

    An absolute path in an error message reaches a committed transcript and
    then publishes the build machine's filesystem layout; release-gate
    check 8 fails the whole batch on one.
    """
    with pytest.raises(FileNotFoundError) as caught:
        load_passes("does-not-exist.csv")
    message = str(caught.value)
    assert "does-not-exist.csv" in message
    assert not message.startswith(os.sep)
    assert _home_prefix() not in message


def test_record_with_a_wrong_header_is_refused(tmp_path):
    """Schema validation: the header must match exactly."""
    target = tmp_path / "bad.csv"
    target.write_text("pass_id,station\n0,GS-01\n")
    with pytest.raises(ValueError, match="header does not match"):
        load_passes(target)


def test_empty_record_is_refused(tmp_path):
    """A file with no header is not a record."""
    target = tmp_path / "empty.csv"
    target.write_text("")
    with pytest.raises(ValueError, match="no header"):
        load_passes(target)


def test_header_only_record_is_refused(tmp_path):
    """A header with no rows is not a record."""
    target = tmp_path / "headeronly.csv"
    target.write_text(",".join(COLUMNS) + "\n")
    with pytest.raises(ValueError, match="no rows"):
        load_passes(target)


def _row(**overrides) -> PassRow:
    """A valid row, with fields overridden for the validation tests."""
    base = dict(
        pass_id=0, station="GS-01", rise_jd=2453911.0, rise_fr=0.5, campaign_hour=1.0,
        local_hour=12.0, duration_s=300.0, max_elevation_deg=40.0, min_slant_range_km=700.0,
        rise_azimuth_deg=10.0, slew_angle_deg=20.0, backlog_mbit=1000.0,
        hours_since_last_contact=2.0, prior_pass_outage_count=1, acquisition_s=30.0,
        outage_count=1, resync_intervals="50.000000:10.000000", idle_intervals="",
        nominal_mbit=20000.0, capacity_mbit=18000.0, realised_mbit=15000.0,
        shortfall_mbit=5000.0, geometry_mbit=2000.0, acquisition_mbit=1000.0,
        resync_mbit=1000.0, idle_mbit=1000.0, residual_mbit=0.0,
    )
    base.update(overrides)
    return PassRow(**base)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("overrides", "pattern"),
    [
        ({"station": "GS-99"}, "not in the catalogue"),
        ({"duration_s": 0.0}, "duration_s"),
        ({"capacity_mbit": 0.0}, "capacity_mbit"),
        ({"acquisition_s": -1.0}, "acquisition_s"),
        ({"realised_mbit": -1.0}, "negative realised yield"),
        ({"realised_mbit": 1.0e9}, "exceeds the window capacity"),
    ],
)
def test_row_invariants_are_enforced_on_load(tmp_path, overrides, pattern):
    """Every declared row invariant is checked, and the message names the line."""
    target = tmp_path / "invalid.csv"
    write_passes([_row(**overrides)], target)
    with pytest.raises(ValueError, match=pattern):
        load_passes(target)


def test_station_by_name_rejects_an_unknown_station():
    """Input validation, with the known names listed in the message."""
    with pytest.raises(ValueError, match="no station named"):
        station_by_name("GS-99")


def test_feature_matrix_shape_and_order(record):
    """One column per declared feature name, in that order."""
    matrix = feature_matrix(record[:50])
    assert matrix.shape == (50, len(FEATURE_NAMES))
    assert matrix.dtype.kind == "f"


def test_feature_matrix_carries_no_station_identity(record):
    """Two rows at the same station with different geometry differ in features.

    The point of the check is the converse: the station enters only through
    its physical figures, so there is no column a model could use as an
    identity. Verified structurally by the feature names.
    """
    assert not any("station_name" in name or name == "station" for name in FEATURE_NAMES)
    assert "station_gain_over_temperature_db" in FEATURE_NAMES


def test_feature_matrix_rejects_an_empty_record():
    """Input validation."""
    with pytest.raises(ValueError, match="at least one row"):
        feature_matrix([])


def test_target_vector_is_the_realised_fraction(record):
    """The target is ``realised / capacity``, dimensionless and in ``[0, 1]``."""
    target = target_vector(record[:30])
    assert target.shape == (30,)
    assert float(target.min()) >= 0.0
    assert float(target.max()) <= 1.0 + 1.0e-9


def test_station_split_holds_out_whole_stations(record):
    """No station appears on both sides of the station split."""
    split = station_split(record)
    train_stations = {record[index].station for index in split.train}
    test_stations = {record[index].station for index in split.test}
    assert train_stations.isdisjoint(test_stations)
    assert test_stations == set(DEFAULT_HELDOUT_STATIONS)
    assert len(split.train) + len(split.test) == len(record)


def test_station_split_rejects_an_unknown_holdout(record):
    """Input validation."""
    with pytest.raises(ValueError, match="not in the station catalogue"):
        station_split(record, ("GS-99",))


def test_time_split_is_chronological_and_not_random(record):
    """Every held-out pass is later than every training pass.

    A random subset would let the model interpolate between passes that
    straddle the one it is forecasting, which a schedule cannot do.
    """
    split = time_split(record)
    latest_train = max(record[index].campaign_hour for index in split.train)
    earliest_test = min(record[index].campaign_hour for index in split.test)
    assert earliest_test >= latest_train
    assert len(split.train) + len(split.test) == len(record)
    assert 0.2 < len(split.test) / len(record) < 0.3


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.5])
def test_time_split_rejects_bad_fractions(record, fraction):
    """Input validation."""
    with pytest.raises(ValueError, match="holdout_fraction"):
        time_split(record, fraction)


def test_split_refuses_to_leak_indices():
    """A split with an index on both sides is refused at construction."""
    with pytest.raises(ValueError, match="leaks"):
        Split("bad", "overlapping", (1, 2), (2, 3))


def test_split_refuses_an_empty_side():
    """A split with nothing held out is not a split."""
    with pytest.raises(ValueError, match="empty side"):
        Split("bad", "empty", (1, 2), ())


def test_station_catalogue_is_eight_distinct_synthetic_sites():
    """Eight stations, distinct names, masks inside the model's validity range."""
    assert len(STATIONS) == 8
    assert len({station.name for station in STATIONS}) == 8
    for station in STATIONS:
        assert station.mask_deg >= 10.0
        assert station.name.startswith("GS-")

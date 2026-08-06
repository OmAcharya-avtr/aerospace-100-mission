"""Availability model and expected-data tests."""

from datetime import datetime, timedelta, timezone

import pytest

from passplanner import (
    ClimatologyAvailability,
    ForecastAvailability,
    ForecastInterval,
    Station,
    expected_data,
    load_stations,
)
from passplanner.passes import Pass

T0 = datetime(2026, 3, 1, 0, 0, 0, tzinfo=timezone.utc)
PRIORS = tuple(round(0.05 * (m + 1), 2) for m in range(12))  # Jan 0.05 ... Dec 0.60
ST = Station(name="S1", lat_deg=0.0, lon_deg=0.0, data_rate_gbps=2.0,
             monthly_clear_prob=PRIORS)


def mk(start_s: float, end_s: float, station: Station = ST) -> Pass:
    return Pass("SAT", station,
                T0 + timedelta(seconds=start_s), T0 + timedelta(seconds=end_s),
                T0 + timedelta(seconds=0.5 * (start_s + end_s)), 45.0)


def test_climatology_indexes_month_correctly():
    clim = ClimatologyAvailability({"S1": PRIORS})
    assert clim.p_clear("S1", datetime(2026, 1, 15, tzinfo=timezone.utc)) == pytest.approx(0.05)
    assert clim.p_clear("S1", datetime(2026, 3, 15, tzinfo=timezone.utc)) == pytest.approx(0.15)
    assert clim.p_clear("S1", datetime(2026, 12, 1, tzinfo=timezone.utc)) == pytest.approx(0.60)


def test_climatology_from_stations_and_unknown_station():
    clim = ClimatologyAvailability.from_stations([ST])
    assert clim.p_clear("S1", T0) == pytest.approx(0.15)
    with pytest.raises(KeyError):
        clim.p_clear("nope", T0)


def test_expected_data_known_answer():
    # Hand check: rate 2 Gbit/s * 300 s * p_clear(March) 0.15 = 90 Gbit.
    clim = ClimatologyAvailability({"S1": PRIORS})
    assert expected_data(mk(0, 300), clim) == pytest.approx(90.0)


def test_expected_data_unweighted():
    # availability=None -> p = 1: 2 Gbit/s * 300 s = 600 Gbit.
    assert expected_data(mk(0, 300), None) == pytest.approx(600.0)


def test_forecast_override_inside_interval_only():
    clim = ClimatologyAvailability({"S1": PRIORS})
    fc = ForecastAvailability(clim, [ForecastInterval("S1", T0, T0 + timedelta(hours=1), 0.05)])
    assert fc.p_clear("S1", T0 + timedelta(minutes=30)) == pytest.approx(0.05)
    assert fc.p_clear("S1", T0 + timedelta(hours=2)) == pytest.approx(0.15)
    # Interval is half-open [start, end).
    assert fc.p_clear("S1", T0) == pytest.approx(0.05)
    assert fc.p_clear("S1", T0 + timedelta(hours=1)) == pytest.approx(0.15)


def test_forecast_last_matching_interval_wins():
    clim = ClimatologyAvailability({"S1": PRIORS})
    fc = ForecastAvailability(clim, [
        ForecastInterval("S1", T0, T0 + timedelta(hours=4), 0.2),
        ForecastInterval("S1", T0 + timedelta(hours=1), T0 + timedelta(hours=2), 0.9)])
    assert fc.p_clear("S1", T0 + timedelta(minutes=90)) == pytest.approx(0.9)
    assert fc.p_clear("S1", T0 + timedelta(minutes=30)) == pytest.approx(0.2)


def test_forecast_other_station_not_affected():
    clim = ClimatologyAvailability({"S1": PRIORS, "S2": PRIORS})
    fc = ForecastAvailability(clim, [ForecastInterval("S2", T0, T0 + timedelta(hours=1), 0.0)])
    assert fc.p_clear("S1", T0) == pytest.approx(0.15)
    assert fc.p_clear("S2", T0) == pytest.approx(0.0)


def test_load_stations_from_example_yaml(tmp_path):
    yaml_text = """
stations:
  - name: A
    lat_deg: 10.0
    lon_deg: 20.0
    alt_km: 1.0
    min_elevation_deg: 25.0
    data_rate_gbps: 5.0
    monthly_clear_prob: [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
  - name: B
    lat_deg: -30.0
    lon_deg: 15.0
"""
    path = tmp_path / "st.yaml"
    path.write_text(yaml_text)
    stations = load_stations(path)
    assert [s.name for s in stations] == ["A", "B"]
    assert stations[0].min_elevation_deg == 25.0
    assert stations[1].alt_km == 0.0 and stations[1].data_rate_gbps == 1.0
    assert stations[1].monthly_clear_prob is None


def test_shipped_example_stations_load():
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "examples" / "stations_example.yaml"
    stations = load_stations(path)
    assert len(stations) == 3
    for s in stations:
        assert s.monthly_clear_prob is not None
        assert all(0.0 <= p <= 1.0 for p in s.monthly_clear_prob)


# --- input validation -------------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    {"lat_deg": 91.0}, {"lat_deg": -91.0}, {"lon_deg": 181.0},
    {"min_elevation_deg": 90.0}, {"min_elevation_deg": -1.0},
    {"data_rate_gbps": 0.0}, {"data_rate_gbps": -1.0}, {"name": ""},
])
def test_station_rejects_bad_values(kwargs):
    base = {"name": "S", "lat_deg": 0.0, "lon_deg": 0.0}
    base.update(kwargs)
    with pytest.raises(ValueError):
        Station(**base)


def test_station_rejects_wrong_prior_length():
    with pytest.raises(ValueError, match="12 values"):
        Station(name="S", lat_deg=0.0, lon_deg=0.0, monthly_clear_prob=(0.5,) * 11)


def test_station_rejects_out_of_range_prior():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        Station(name="S", lat_deg=0.0, lon_deg=0.0, monthly_clear_prob=(1.5,) * 12)


def test_climatology_rejects_bad_priors():
    with pytest.raises(ValueError, match="12 monthly"):
        ClimatologyAvailability({"S": (0.5,) * 6})
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        ClimatologyAvailability({"S": (-0.1,) * 12})


def test_forecast_interval_validation():
    with pytest.raises(ValueError, match="after start"):
        ForecastInterval("S", T0, T0, 0.5)
    with pytest.raises(ValueError, match="p_clear"):
        ForecastInterval("S", T0, T0 + timedelta(hours=1), 1.2)


def test_expected_data_rejects_bad_availability():
    class Broken:
        def p_clear(self, station, when):
            return 2.0
    with pytest.raises(ValueError, match="outside"):
        expected_data(mk(0, 100), Broken())


def test_load_stations_rejects_duplicate_names(tmp_path):
    path = tmp_path / "dup.yaml"
    path.write_text("stations:\n"
                    "  - {name: A, lat_deg: 0.0, lon_deg: 0.0}\n"
                    "  - {name: A, lat_deg: 1.0, lon_deg: 1.0}\n")
    with pytest.raises(ValueError, match="unique"):
        load_stations(path)


def test_load_stations_rejects_missing_key(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("not_stations: []\n")
    with pytest.raises(ValueError, match="stations"):
        load_stations(path)

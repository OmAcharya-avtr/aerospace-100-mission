"""Medium-class integration and benchmark tests.

Integration: TLE -> passes -> availability weighting -> greedy and ILP
schedules, plus the CLI end-to-end on the shipped example scenario.
Benchmark: runtime bounds on a 2-core machine (generous factor over the
measured times so the test is a regression guard, not a flaky timing check).
"""

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from passplanner import (
    ClimatologyAvailability,
    Station,
    expected_data,
    find_passes,
    load_stations,
    schedule_greedy,
    schedule_ilp,
)
from passplanner.cli import build_availability, load_scenario, main
from passplanner.fixtures import ISS_2008
from passplanner.scheduler import passes_conflict

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
T0 = datetime(2008, 9, 20, 12, 0, tzinfo=timezone.utc)
T1 = T0 + timedelta(days=1)


def _candidate_passes(stations, setup_ok=True):
    passes = []
    for st in stations:
        passes.extend(find_passes(ISS_2008, st, T0, T1))
    passes.sort(key=lambda p: p.t_rise)
    assert setup_ok
    return passes


def test_end_to_end_tle_to_schedule():
    stations = load_stations(EXAMPLES / "stations_example.yaml")
    passes = _candidate_passes(stations)
    assert len(passes) >= 5
    availability = ClimatologyAvailability.from_stations(stations)

    greedy = schedule_greedy(passes, availability, setup_time_s=1800.0)
    ilp = schedule_ilp(passes, availability, setup_time_s=1800.0)

    # ILP is exact, so it can never be worse than greedy.
    assert ilp.total_value >= greedy.total_value - 1e-9
    # Both schedules must be conflict-free.
    for res in (greedy, ilp):
        sel = res.selected
        for i in range(len(sel)):
            for j in range(i + 1, len(sel)):
                assert not passes_conflict(sel[i], sel[j], 1800.0)
        # Reported totals equal the sum of per-pass expected data.
        assert res.total_value == pytest.approx(
            sum(expected_data(p, availability) for p in sel))
        assert res.n_candidates == len(passes)


def test_availability_weighting_changes_the_schedule_value():
    stations = load_stations(EXAMPLES / "stations_example.yaml")
    passes = _candidate_passes(stations)
    availability = ClimatologyAvailability.from_stations(stations)
    weighted = schedule_ilp(passes, availability, setup_time_s=1800.0)
    unweighted = schedule_ilp(passes, None, setup_time_s=1800.0)
    # Unweighted values are p_clear=1, so they are strictly larger in raw Gbit.
    assert unweighted.total_value > weighted.total_value
    # And the weighted plan must be at least as good as the unweighted plan
    # when both are scored with the availability model.
    unweighted_scored = sum(expected_data(p, availability) for p in unweighted.selected)
    assert weighted.total_value >= unweighted_scored - 1e-9


def test_scenario_loading_and_forecast_override():
    scn = load_scenario(EXAMPLES / "scenario_example.yaml")
    assert len(scn["tles"]) == 1
    assert len(scn["stations"]) == 3
    assert scn["method"] == "both"
    availability = build_availability(scn["stations"], scn["forecast"])
    # Inside the forecast window the alpine site is overridden to 0.10.
    inside = datetime(2008, 9, 20, 20, 0, tzinfo=timezone.utc)
    outside = datetime(2008, 9, 20, 14, 0, tzinfo=timezone.utc)
    assert availability.p_clear("Alpengipfel OGS", inside) == pytest.approx(0.10)
    assert availability.p_clear("Alpengipfel OGS", outside) == pytest.approx(0.52)  # Sep prior


def test_cli_plan_runs_and_prints_table(capsys):
    rc = main(["plan", "--config", str(EXAMPLES / "scenario_example.yaml")])
    out = capsys.readouterr().out
    assert rc == 0
    assert "schedule (greedy)" in out and "schedule (ilp)" in out
    assert "ISS (ZARYA)" in out
    assert "% of ILP optimum" in out


def test_cli_reports_error_on_missing_file(capsys):
    rc = main(["plan", "--config", "/nonexistent/scenario.yaml"])
    assert rc == 2
    assert "error:" in capsys.readouterr().err


def test_cli_rejects_bad_scenario(tmp_path, capsys):
    bad = tmp_path / "bad.yaml"
    bad.write_text("window: {start: '2020-01-02T00:00:00Z', end: '2020-01-01T00:00:00Z'}\n"
                   "satellites: []\n")
    assert main(["plan", "--config", str(bad)]) == 2
    assert "error:" in capsys.readouterr().err


# --- benchmark / regression -------------------------------------------------

def test_benchmark_pass_finding_runtime():
    """24 h of ISS passes over 3 stations must be found in well under 10 s."""
    stations = load_stations(EXAMPLES / "stations_example.yaml")
    t_start = time.perf_counter()
    passes = _candidate_passes(stations)
    elapsed = time.perf_counter() - t_start
    assert len(passes) >= 5
    assert elapsed < 10.0, f"pass finding took {elapsed:.2f} s (bound 10 s)"


def test_benchmark_ilp_runtime_on_medium_instance():
    """ILP on a ~60-pass instance must solve in well under 30 s."""
    stations = [
        Station(name=f"GS{i}", lat_deg=-60.0 + 15.0 * i, lon_deg=-180.0 + 40.0 * i,
                alt_km=1.0, min_elevation_deg=10.0, data_rate_gbps=10.0,
                monthly_clear_prob=tuple([0.5 + 0.02 * i] * 12))
        for i in range(8)]
    passes = []
    for st in stations:
        passes.extend(find_passes(ISS_2008, st, T0, T1))
    availability = ClimatologyAvailability.from_stations(stations)
    assert len(passes) >= 20
    t_start = time.perf_counter()
    res = schedule_ilp(passes, availability, setup_time_s=600.0)
    elapsed = time.perf_counter() - t_start
    assert res.total_value > 0.0
    assert elapsed < 30.0, f"ILP solve took {elapsed:.2f} s (bound 30 s)"

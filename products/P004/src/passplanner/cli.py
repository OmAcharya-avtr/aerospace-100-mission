"""Command-line interface: ``python -m passplanner plan --config scenario.yaml``.

Scenario YAML layout::

    window:
      start: "2008-09-20T12:00:00Z"
      end:   "2008-09-21T12:00:00Z"
    satellites:
      - name: ISS (ZARYA)
        tle:
          - "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
          - "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
    stations_file: stations.yaml        # relative to the scenario file
    scheduler: ilp                      # ilp | greedy | both
    setup_time_s: 60.0                  # optional, default 0
    forecast:                           # optional overrides
      - {station: Cerro Ficticio OGS, start: "2008-09-20T18:00:00Z",
         end: "2008-09-21T06:00:00Z", p_clear: 0.15}
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .availability import (
    Availability,
    ClimatologyAvailability,
    ForecastAvailability,
    ForecastInterval,
)
from .passes import TLE, Pass, find_passes
from .scheduler import ScheduleResult, schedule_greedy, schedule_ilp
from .stations import Station, load_stations


def _parse_time(value: str) -> datetime:
    t = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return t.astimezone(timezone.utc)


def load_scenario(path: str | Path) -> dict:
    """Parse and validate a scenario YAML into a plain dict of typed objects."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    if not isinstance(doc, dict):
        raise ValueError(f"{path}: scenario must be a YAML mapping")
    for key in ("window", "satellites"):
        if key not in doc:
            raise ValueError(f"{path}: missing required key '{key}'")
    t0 = _parse_time(doc["window"]["start"])
    t1 = _parse_time(doc["window"]["end"])
    if t1 <= t0:
        raise ValueError(f"{path}: window end must be after start")

    tles = []
    for sat in doc["satellites"]:
        lines = sat.get("tle", [])
        if len(lines) != 2:
            raise ValueError(f"{path}: satellite '{sat.get('name')}' needs 2 TLE lines")
        tles.append(TLE(str(sat.get("name", "SAT")), lines[0], lines[1]))

    if "stations_file" in doc:
        stations = load_stations(path.parent / doc["stations_file"])
    elif "stations" in doc:
        stations = [Station(
            name=str(e["name"]), lat_deg=float(e["lat_deg"]), lon_deg=float(e["lon_deg"]),
            alt_km=float(e.get("alt_km", 0.0)),
            min_elevation_deg=float(e.get("min_elevation_deg", 20.0)),
            data_rate_gbps=float(e.get("data_rate_gbps", 1.0)),
            monthly_clear_prob=tuple(e["monthly_clear_prob"])
            if e.get("monthly_clear_prob") else None,
        ) for e in doc["stations"]]
    else:
        raise ValueError(f"{path}: need 'stations_file' or inline 'stations'")

    forecast = [ForecastInterval(station=str(f["station"]),
                                 start=_parse_time(f["start"]),
                                 end=_parse_time(f["end"]),
                                 p_clear=float(f["p_clear"]))
                for f in doc.get("forecast", [])]
    method = str(doc.get("scheduler", "ilp")).lower()
    if method not in ("ilp", "greedy", "both"):
        raise ValueError(f"{path}: scheduler must be ilp|greedy|both, got '{method}'")
    return {
        "t0": t0, "t1": t1, "tles": tles, "stations": stations,
        "forecast": forecast, "method": method,
        "setup_time_s": float(doc.get("setup_time_s", 0.0)),
    }


def build_availability(stations: list[Station],
                       forecast: list[ForecastInterval]) -> Availability | None:
    """Climatology from station priors, wrapped with forecast overrides."""
    with_priors = [s for s in stations if s.monthly_clear_prob is not None]
    if not with_priors:
        return None
    base = ClimatologyAvailability.from_stations(with_priors)
    return ForecastAvailability(base, forecast) if forecast else base


def _fmt_schedule(result: ScheduleResult, availability: Availability | None) -> str:
    hdr = (f"{'satellite':<14} {'station':<22} {'rise (UTC)':<20} {'set (UTC)':<20} "
           f"{'dur[s]':>7} {'maxEl':>6} {'p_clr':>6} {'E[Gbit]':>9}")
    lines = [(f"schedule ({result.method}): {len(result.selected)}/{result.n_candidates} "
              f"passes selected, total expected data {result.total_value:,.1f} Gbit")]
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for p, v in zip(result.selected, result.values):
        p_clr = 1.0 if availability is None else availability.p_clear(
            p.station.name, p.t_culminate)
        lines.append(
            f"{p.satellite:<14.14} {p.station.name:<22.22} "
            f"{p.t_rise.strftime('%Y-%m-%d %H:%M:%S'):<20} "
            f"{p.t_set.strftime('%Y-%m-%d %H:%M:%S'):<20} "
            f"{p.duration_s:>7.0f} {p.max_elevation_deg:>6.1f} {p_clr:>6.2f} {v:>9.1f}")
    return "\n".join(lines)


def cmd_plan(args: argparse.Namespace) -> int:
    scn = load_scenario(args.config)
    availability = build_availability(scn["stations"], scn["forecast"])
    passes: list[Pass] = []
    for tle in scn["tles"]:
        for station in scn["stations"]:
            passes.extend(find_passes(tle, station, scn["t0"], scn["t1"]))
    passes.sort(key=lambda p: p.t_rise)
    print(f"window : {scn['t0'].isoformat()} .. {scn['t1'].isoformat()}")
    print(f"inputs : {len(scn['tles'])} satellite(s), {len(scn['stations'])} station(s), "
          f"{len(passes)} candidate pass(es), setup_time_s={scn['setup_time_s']:.0f}")
    if availability is None:
        print("weight : none (no station priors given; p_clear = 1)")
    else:
        print(f"weight : climatological priors"
              f"{' + forecast overrides' if scn['forecast'] else ''}")
    print()
    methods = ("greedy", "ilp") if scn["method"] == "both" else (scn["method"],)
    results = []
    for m in methods:
        fn = schedule_greedy if m == "greedy" else schedule_ilp
        res = fn(passes, availability=availability, setup_time_s=scn["setup_time_s"])
        results.append(res)
        print(_fmt_schedule(res, availability))
        print()
    if len(results) == 2 and results[1].total_value > 0:
        gap = 100.0 * (1.0 - results[0].total_value / results[1].total_value)
        print(f"summary: greedy achieves {100.0 - gap:.2f}% of ILP optimum "
              f"(gap {gap:.2f}%)")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m passplanner``."""
    parser = argparse.ArgumentParser(
        prog="passplanner",
        description="Optical ground-station contact planner (research-grade; "
                    "not certified for operational use).")
    sub = parser.add_subparsers(dest="command", required=True)
    p_plan = sub.add_parser("plan", help="find passes and build a schedule")
    p_plan.add_argument("--config", required=True, help="scenario YAML file")
    p_plan.set_defaults(func=cmd_plan)
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

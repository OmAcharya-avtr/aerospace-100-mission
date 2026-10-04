"""Command-line interface: ``python -m constellink``.

Subcommands
-----------
``contacts``    contact-window report for a Walker shell and the built-in stations
``route``       minimum-latency route over the time-expanded graph
``flow``        integer maximum-flow schedule
``capacity``    RF and optical link evaluation at one range
``predict``     train and compare the three link-availability predictors
``benchmark``   the performance harness

Library code never prints; all output is produced here.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime, timedelta

import numpy as np

from . import __version__
from .availability import (
    ClimatologyBaseline,
    LinkAvailabilityModel,
    LogisticBaseline,
    PredictorScores,
    grouped_split,
    score_predictor,
)
from .benchmark import BenchmarkResult, benchmark, environment
from .capacity import optical_link, rf_link
from .constellation import walker_delta
from .contacts import all_contact_windows
from .flow import ilp_max_flow
from .graph import ContactGraph, TimeExpandedGraph
from .routing import shortest_route
from .synthdata import (
    DatasetConfig,
    default_optical_terminal,
    default_rf_terminal,
    default_stations,
    generate_dataset,
)

_EPOCH = datetime(2026, 4, 1, tzinfo=UTC)


def _walker_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--n-total", type=int, default=24, help="satellites T (default 24)")
    p.add_argument("--n-planes", type=int, default=4, help="planes P (default 4)")
    p.add_argument("--phasing", type=int, default=1, help="phasing F (default 1)")
    p.add_argument("--inclination-deg", type=float, default=53.0,
                   help="inclination [deg] (default 53)")
    p.add_argument("--altitude-km", type=float, default=550.0,
                   help="circular altitude [km] (default 550)")
    p.add_argument("--hours", type=float, default=3.0,
                   help="horizon [h] (default 3; keep small, see README budget)")
    p.add_argument("--step-s", type=float, default=60.0,
                   help="contact-scan grid step [s] (default 60)")
    p.add_argument("--isl-max-range-km", type=float, default=5000.0,
                   help="ISL range limit [km] (default 5000)")
    p.add_argument("--no-stations", action="store_true",
                   help="omit the built-in ground stations")


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(
        prog="constellink",
        description=("Time-varying constellation contact graphs, routing over a "
                     "time-expanded graph, and per-link RF/optical capacity. "
                     "Research-grade; not flight-qualified or certified."))
    parser.add_argument("--version", action="version",
                        version=f"constellink {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_c = sub.add_parser("contacts", help="contact-window report")
    _walker_args(p_c)
    p_c.add_argument("--limit", type=int, default=15,
                     help="windows to list (default 15)")

    p_r = sub.add_parser("route", help="minimum-latency route")
    _walker_args(p_r)
    p_r.add_argument("--source", required=True, help="source node name")
    p_r.add_argument("--destination", required=True, help="destination node name")
    p_r.add_argument("--slot-s", type=float, default=60.0,
                     help="time-expanded slot [s] (default 60)")
    p_r.add_argument("--message-mbit", type=float, default=0.0,
                     help="message size [Mbit] (default 0)")
    p_r.add_argument("--rate-mbps", type=float, default=100.0,
                     help="uniform link rate [Mbit/s] (default 100)")

    p_f = sub.add_parser("flow", help="integer maximum-flow schedule")
    _walker_args(p_f)
    p_f.add_argument("--source", required=True)
    p_f.add_argument("--destination", required=True)
    p_f.add_argument("--slot-s", type=float, default=60.0)
    p_f.add_argument("--rate-mbps", type=float, default=100.0)
    p_f.add_argument("--unit-mbit", type=float, default=1000.0,
                     help="flow quantum [Mbit] (default 1000)")
    p_f.add_argument("--backend", choices=("auto", "scipy", "pulp"), default="auto")

    p_k = sub.add_parser("capacity", help="RF and optical link evaluation")
    p_k.add_argument("--range-km", type=float, required=True, help="slant range [km]")
    p_k.add_argument("--extra-rf-loss-db", type=float, default=0.0)
    p_k.add_argument("--optical-atm-loss-db", type=float, default=0.0)

    p_p = sub.add_parser("predict", help="train and compare the three predictors")
    p_p.add_argument("--hours", type=float, default=18.0,
                     help="dataset horizon [h] (default 18)")
    p_p.add_argument("--seed", type=int, default=20260401, help="dataset seed")
    p_p.add_argument("--split-seed", type=int, default=0, help="grouped-split seed")
    p_p.add_argument("--test-fraction", type=float, default=0.3)
    p_p.add_argument("--method", choices=("sigmoid", "isotonic"), default="sigmoid")
    p_p.add_argument("--n-bins", type=int, default=10,
                     help="reliability bins (default 10)")

    p_b = sub.add_parser("benchmark", help="performance harness")
    _walker_args(p_b)
    p_b.add_argument("--repeats", type=int, default=3)

    return parser


def _build_graph(args: argparse.Namespace):
    const = walker_delta(args.n_total, args.n_planes, args.phasing,
                         args.inclination_deg, args.altitude_km, _EPOCH,
                         max_epoch_age_days=max(1.0, args.hours / 24.0 + 1.0))
    stations = [] if args.no_stations else default_stations()
    const.stations = stations
    t1 = _EPOCH + timedelta(hours=args.hours)
    eph = const.ephemeris(_EPOCH, t1, args.step_s)
    windows = all_contact_windows(eph, const.satellites, stations,
                                  isl_max_range_km=args.isl_max_range_km)
    # Declare every satellite and station as a node, whether or not it has a
    # contact in this horizon. A node with no contact must be reported as
    # unreachable, not as unknown: "no route" is an answer, "unknown node" is
    # a different claim.
    nodes = [s.name for s in const.satellites] + [g.name for g in stations]
    return const, ContactGraph(windows=windows, t0=_EPOCH, t1=t1, nodes=nodes)


def _cmd_contacts(args: argparse.Namespace, out) -> int:
    const, cg = _build_graph(args)
    isl = [w for w in cg.windows if w.kind == "isl"]
    gnd = [w for w in cg.windows if w.kind == "ground"]
    print(f"constellation: {const.name}", file=out)
    print(f"horizon:       {cg.t0.isoformat()} .. {cg.t1.isoformat()} "
          f"({cg.horizon_s / 3600.0:.2f} h), scan step {args.step_s:g} s", file=out)
    print(f"nodes:         {len(cg.nodes)}", file=out)
    print(f"windows:       {len(cg.windows)} total "
          f"({len(isl)} ISL, {len(gnd)} ground)", file=out)
    comps = cg.components()
    print(f"union graph:   {len(comps)} component(s), largest {len(comps[0])} nodes"
          if comps else "union graph:   empty", file=out)
    if isl:
        d = np.array([w.duration_s for w in isl])
        print(f"ISL duration:  median {np.median(d):.1f} s, "
              f"min {d.min():.1f} s, max {d.max():.1f} s", file=out)
    if gnd:
        d = np.array([w.duration_s for w in gnd])
        e = np.array([w.max_elevation_deg for w in gnd])
        print(f"pass duration: median {np.median(d):.1f} s, max {d.max():.1f} s; "
              f"max elevation median {np.median(e):.1f} deg", file=out)
    print("", file=out)
    head = (f"{'node A':<12}{'node B':<12}{'kind':<8}{'open (UTC)':<28}"
            f"{'dur [s]':>9}{'minR [km]':>11}")
    print(head, file=out)
    print("-" * len(head), file=out)
    for w in cg.windows[:max(0, args.limit)]:
        print(f"{w.node_a:<12}{w.node_b:<12}{w.kind:<8}"
              f"{w.t_open.isoformat():<28}{w.duration_s:>9.1f}"
              f"{w.min_range_km:>11.1f}", file=out)
    if len(cg.windows) > max(0, args.limit):
        print(f"... {len(cg.windows) - max(0, args.limit)} more", file=out)
    return 0


def _cmd_route(args: argparse.Namespace, out) -> int:
    _, cg = _build_graph(args)
    teg = TimeExpandedGraph.from_contact_graph(
        cg, args.slot_s, default_rate_bps=args.rate_mbps * 1e6)
    msg_bits = args.message_mbit * 1e6
    route = shortest_route(teg, args.source, args.destination,
                           release_slot=0, message_bits=msg_bits)
    print(f"time-expanded graph: {teg.n_te_nodes} nodes, {len(teg.edges)} edges, "
          f"{teg.n_slots} slots of {teg.slot_s:g} s", file=out)
    if route is None:
        print(f"no route from {args.source} to {args.destination} within the horizon "
              f"for a {args.message_mbit:g} Mbit message", file=out)
        return 0
    print(f"route:    {' -> '.join(route.node_sequence)}", file=out)
    print(f"latency:  {route.total_latency_s:.6f} s", file=out)
    print(f"hops:     {len(route.hops)} edges, {route.n_transmissions} transmissions",
          file=out)
    print(f"arrival:  slot {route.arrival_slot} "
          f"({teg.slot_start(route.arrival_slot).isoformat()})", file=out)
    return 0


def _cmd_flow(args: argparse.Namespace, out) -> int:
    _, cg = _build_graph(args)
    teg = TimeExpandedGraph.from_contact_graph(
        cg, args.slot_s, default_rate_bps=args.rate_mbps * 1e6,
        store_capacity_bits=args.unit_mbit * 1e6 * 10)
    res = ilp_max_flow(teg, args.source, args.destination, release_slot=0,
                       flow_unit_bits=args.unit_mbit * 1e6, backend=args.backend)
    print(f"backend:   {res.backend} ({res.status})", file=out)
    print(f"quantum:   {args.unit_mbit:g} Mbit", file=out)
    print(f"delivered: {res.flow_units} units = "
          f"{res.delivered_bits / 1e9:.3f} Gbit", file=out)
    return 0


def _cmd_capacity(args: argparse.Namespace, out) -> int:
    rf = default_rf_terminal()
    opt = default_optical_terminal()
    r = rf_link(args.range_km, rf, extra_loss_db=args.extra_rf_loss_db)
    o = optical_link(args.range_km, opt,
                     atmospheric_loss_db=args.optical_atm_loss_db)
    print(f"range: {args.range_km:g} km", file=out)
    print("", file=out)
    print("RF (Ka-band terminal assumptions, see synthdata.default_rf_terminal)",
          file=out)
    print(f"  EIRP            {r.eirp_dbw:+10.2f} dBW", file=out)
    print(f"  free-space loss {r.fspl_db:10.2f} dB", file=out)
    print(f"  other loss      {r.other_loss_db:10.2f} dB", file=out)
    print(f"  C/N0            {r.c_over_n0_dbhz:10.2f} dB-Hz", file=out)
    print(f"  Shannon bound   {r.shannon_capacity_bps / 1e6:10.2f} Mbit/s", file=out)
    print(f"  achievable rate {r.achievable_rate_bps / 1e6:10.2f} Mbit/s", file=out)
    print("", file=out)
    print("Optical (1550 nm terminal assumptions, see "
          "synthdata.default_optical_terminal)", file=out)
    print(f"  beam radius     {o.beam_radius_m:10.2f} m", file=out)
    print(f"  geometric loss  {o.geometric_loss_db:10.2f} dB", file=out)
    print(f"  pointing loss   {o.pointing_loss_db:10.2f} dB", file=out)
    print(f"  atmospheric     {o.atmospheric_loss_db:10.2f} dB", file=out)
    print(f"  Rx power        {o.rx_power_w * 1e6:10.4f} uW", file=out)
    print(f"  achievable rate {o.achievable_rate_bps / 1e6:10.2f} Mbit/s", file=out)
    return 0


def _cmd_predict(args: argparse.Namespace, out) -> int:
    cfg = DatasetConfig(horizon_hours=args.hours, seed=args.seed)
    data = generate_dataset(cfg)
    tr, te = grouped_split(data, test_fraction=args.test_fraction,
                           seed=args.split_seed)
    print(f"dataset: {len(data)} contacts, base rate {data.base_rate:.4f}", file=out)
    print(f"grouped split (by link): {tr.size} train / {te.size} test", file=out)
    print("", file=out)
    clim = ClimatologyBaseline().fit(data.x[tr], data.y[tr], data.stratum[tr])
    p_clim = clim.predict_proba(data.x[te], data.stratum[te])
    logi = LogisticBaseline().fit(data.x[tr], data.y[tr])
    p_logi = logi.predict_proba(data.x[te])
    model = LinkAvailabilityModel(seed=1, method=args.method).fit(
        data.x[tr], data.y[tr])
    p_model, s_model = model.predict_with_uncertainty(data.x[te])
    print(PredictorScores.header(), file=out)
    print("-" * len(PredictorScores.header()), file=out)
    scores = []
    for name, p in (("climatology (base)", p_clim), ("logistic (base)", p_logi),
                    (f"learned ({args.method})", p_model)):
        s = score_predictor(name, p, data.y[te], n_bins=args.n_bins)
        scores.append(s)
        print(s.format_row(), file=out)
    print("", file=out)
    print(f"reliability bins: {args.n_bins}; lower Brier/REL/ECE is better, "
          f"higher RES is better", file=out)
    print(f"learned-model ensemble std: mean {s_model.mean():.4f}, "
          f"max {s_model.max():.4f}", file=out)
    best = min(scores, key=lambda s: s.reliability)
    print(f"best calibrated (lowest reliability term): {best.name}", file=out)
    return 0


def _cmd_benchmark(args: argparse.Namespace, out) -> int:
    env = environment()
    print(env.format_block(), file=out)
    print("", file=out)
    const = walker_delta(args.n_total, args.n_planes, args.phasing,
                         args.inclination_deg, args.altitude_km, _EPOCH,
                         max_epoch_age_days=max(1.0, args.hours / 24.0 + 1.0))
    stations = [] if args.no_stations else default_stations()
    t1 = _EPOCH + timedelta(hours=args.hours)
    results = [benchmark("ephemeris",
                         lambda: const.ephemeris(_EPOCH, t1, args.step_s),
                         repeats=args.repeats)]
    eph = const.ephemeris(_EPOCH, t1, args.step_s)
    results.append(benchmark(
        "contact scan (all pairs)",
        lambda: all_contact_windows(eph, const.satellites, stations,
                                    isl_max_range_km=args.isl_max_range_km),
        repeats=args.repeats))
    windows = all_contact_windows(eph, const.satellites, stations,
                                  isl_max_range_km=args.isl_max_range_km)
    cg = ContactGraph(windows=windows, t0=_EPOCH, t1=t1)
    results.append(benchmark("time-expanded unroll",
                             lambda: TimeExpandedGraph.from_contact_graph(cg, 60.0),
                             repeats=args.repeats))
    teg = TimeExpandedGraph.from_contact_graph(cg, 60.0)
    src, dst = cg.nodes[0], cg.nodes[-1]
    results.append(benchmark("dijkstra (one pair)",
                             lambda: shortest_route(teg, src, dst),
                             repeats=args.repeats))
    print(BenchmarkResult.header(), file=out)
    print("-" * len(BenchmarkResult.header()), file=out)
    for r in results:
        print(r.format_row(), file=out)
    print("", file=out)
    print("peakPy is a tracemalloc lower bound: C-extension memory is not counted.",
          file=out)
    return 0


_COMMANDS = {
    "contacts": _cmd_contacts,
    "route": _cmd_route,
    "flow": _cmd_flow,
    "capacity": _cmd_capacity,
    "predict": _cmd_predict,
    "benchmark": _cmd_benchmark,
}


def main(argv: list[str] | None = None, out=None) -> int:
    """Entry point.  Returns a process exit code."""
    out = out or sys.stdout
    args = build_parser().parse_args(argv)
    try:
        return _COMMANDS[args.command](args, out)
    except (ValueError, KeyError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

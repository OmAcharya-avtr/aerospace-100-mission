"""Regression suite with pinned seeded outputs (Level 3 requirement).

Every number in this file was produced by running this repository's own code
in the session that wrote the file, and is pinned so that an unintended change
in geometry, graph construction, routing, capacity or model training shows up
as a test failure rather than as a silently different answer.

Pinned values are NOT physical claims.  They are a change detector.  When a
change here is deliberate, the right response is to re-measure, update the
constant and say so in CHANGELOG.md -- never to loosen the tolerance.

Scenario, fixed in ``_scenario`` below:
    8/2/1 Walker shell, 53 deg, 550 km, epoch 2026-04-01T00:00:00Z,
    2 h horizon, 60 s contact-scan step, the five built-in ground stations,
    60 s time-expanded slots at 100 Mbit/s uniform rate.

Model scenario:
    the synthetic dataset at ``horizon_hours=6, seed=7``, grouped split at
    ``test_fraction=0.3, seed=0``, and a 3-member learned model with
    ``seed=1, max_iter=40``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from constellink.availability import (
    ClimatologyBaseline,
    LinkAvailabilityModel,
    LogisticBaseline,
    grouped_split,
    score_predictor,
)
from constellink.capacity import optical_link, rf_link
from constellink.constellation import walker_delta
from constellink.contacts import all_contact_windows
from constellink.flow import ilp_max_flow
from constellink.graph import ContactGraph, TimeExpandedGraph
from constellink.routing import shortest_route
from constellink.synthdata import (
    DatasetConfig,
    default_optical_terminal,
    default_rf_terminal,
    default_stations,
    generate_dataset,
)

EPOCH = datetime(2026, 4, 1, tzinfo=UTC)
HORIZON_H = 2.0
STEP_S = 60.0
SLOT_S = 60.0
RATE_BPS = 1.0e8

# Pinned geometry and graph outputs.
PIN_N_WINDOWS = 27
PIN_N_ISL = 20
PIN_N_GROUND = 7
PIN_TOTAL_ISL_DURATION_S = 12576.708985
PIN_TOTAL_GROUND_DURATION_S = 2164.980468
PIN_FIRST_WINDOW = ("W00-02", "W01-03", "isl", 629.941406)
PIN_N_NODES = 10
PIN_N_COMPONENTS = 1
PIN_N_TE_EDGES = 1638

# Pinned routing and flow outputs.
PIN_ROUTE_LATENCY_S = 5760.016016254
PIN_ROUTE_NODES = ("W00-00", "W01-02", "AWARUA")
PIN_FLOW_UNITS = 4
PIN_FLOW_UNIT_BITS = 6.0e9

# Pinned capacity outputs at 1500 km with the default terminals.
PIN_RF_RATE_MBPS = 152.408497
PIN_OPT_RATE_MBPS = 11071.122553

# Pinned dataset and model outputs.
PIN_DATASET_N = 673
PIN_DATASET_BASE_RATE = 0.757800892
PIN_DATASET_X_SUM = 2258980.6921
PIN_SPLIT = (495, 178)
PIN_SCORES = {
    "climatology": (0.155990487, 0.003488179),
    "logistic": (0.082848357, 0.004595313),
    "learned": (0.089018047, 0.012436083),
}

# Tolerances.  Durations are pinned to the microsecond because the bisection
# tolerance is 0.05 s and the refinement is deterministic; rates to 1e-6
# relative because they are closed-form evaluations; Brier terms to 1e-7
# absolute because sklearn's float accumulation order is not guaranteed
# bit-for-bit across BLAS builds.
TOL_DURATION_S = 1e-5
TOL_REL = 1e-9
TOL_SCORE = 1e-7


def _scenario():
    const = walker_delta(8, 2, 1, 53.0, 550.0, EPOCH, max_epoch_age_days=2.0)
    t1 = EPOCH + timedelta(hours=HORIZON_H)
    eph = const.ephemeris(EPOCH, t1, STEP_S)
    windows = all_contact_windows(eph, const.satellites, default_stations())
    cg = ContactGraph(windows=windows, t0=EPOCH, t1=t1)
    teg = TimeExpandedGraph.from_contact_graph(cg, SLOT_S,
                                               default_rate_bps=RATE_BPS)
    return const, cg, teg


@pytest.fixture(scope="module")
def scenario():
    """The pinned scenario, built once."""
    return _scenario()


def test_window_counts_are_pinned(scenario):
    _, cg, _ = scenario
    assert len(cg.windows) == PIN_N_WINDOWS
    assert sum(1 for w in cg.windows if w.kind == "isl") == PIN_N_ISL
    assert sum(1 for w in cg.windows if w.kind == "ground") == PIN_N_GROUND


def test_total_durations_are_pinned(scenario):
    _, cg, _ = scenario
    isl = sum(w.duration_s for w in cg.windows if w.kind == "isl")
    gnd = sum(w.duration_s for w in cg.windows if w.kind == "ground")
    assert isl == pytest.approx(PIN_TOTAL_ISL_DURATION_S, abs=TOL_DURATION_S)
    assert gnd == pytest.approx(PIN_TOTAL_GROUND_DURATION_S, abs=TOL_DURATION_S)


def test_first_window_is_pinned(scenario):
    _, cg, _ = scenario
    w = cg.windows[0]
    assert (w.node_a, w.node_b, w.kind) == PIN_FIRST_WINDOW[:3]
    assert w.duration_s == pytest.approx(PIN_FIRST_WINDOW[3], abs=TOL_DURATION_S)


def test_graph_shape_is_pinned(scenario):
    _, cg, teg = scenario
    assert len(cg.nodes) == PIN_N_NODES
    assert len(cg.components()) == PIN_N_COMPONENTS
    assert len(teg.edges) == PIN_N_TE_EDGES


def test_route_is_pinned(scenario):
    _, _, teg = scenario
    route = shortest_route(teg, "W00-00", "AWARUA")
    assert route is not None
    assert route.node_sequence == PIN_ROUTE_NODES
    assert route.total_latency_s == pytest.approx(PIN_ROUTE_LATENCY_S,
                                                   rel=TOL_REL)


def test_max_flow_is_pinned(scenario):
    _, _, teg = scenario
    res = ilp_max_flow(teg, "W00-00", "AWARUA",
                       flow_unit_bits=PIN_FLOW_UNIT_BITS)
    assert res.flow_units == PIN_FLOW_UNITS


def test_capacity_values_are_pinned():
    rf = rf_link(1500.0, default_rf_terminal()).achievable_rate_bps / 1e6
    opt = optical_link(1500.0,
                       default_optical_terminal()).achievable_rate_bps / 1e6
    assert rf == pytest.approx(PIN_RF_RATE_MBPS, rel=1e-6)
    assert opt == pytest.approx(PIN_OPT_RATE_MBPS, rel=1e-6)


@pytest.fixture(scope="module")
def pinned_dataset():
    """The pinned dataset scenario."""
    return generate_dataset(DatasetConfig(horizon_hours=6.0, seed=7))


def test_dataset_is_pinned(pinned_dataset):
    d = pinned_dataset
    assert len(d) == PIN_DATASET_N
    assert d.base_rate == pytest.approx(PIN_DATASET_BASE_RATE, abs=1e-9)
    assert float(d.x.sum()) == pytest.approx(PIN_DATASET_X_SUM, rel=1e-9)


def test_grouped_split_is_pinned(pinned_dataset):
    train, test = grouped_split(pinned_dataset, test_fraction=0.3, seed=0)
    assert (train.size, test.size) == PIN_SPLIT


def test_predictor_scores_are_pinned(pinned_dataset):
    d = pinned_dataset
    train, test = grouped_split(d, test_fraction=0.3, seed=0)
    clim = ClimatologyBaseline().fit(d.x[train], d.y[train], d.stratum[train])
    logi = LogisticBaseline().fit(d.x[train], d.y[train])
    learned = LinkAvailabilityModel(n_members=3, seed=1, max_iter=40).fit(
        d.x[train], d.y[train])
    got = {
        "climatology": clim.predict_proba(d.x[test], d.stratum[test]),
        "logistic": logi.predict_proba(d.x[test]),
        "learned": learned.predict_proba(d.x[test]),
    }
    for name, p in got.items():
        s = score_predictor(name, p, d.y[test])
        pin_brier, pin_rel = PIN_SCORES[name]
        assert s.brier == pytest.approx(pin_brier, abs=TOL_SCORE), name
        assert s.reliability == pytest.approx(pin_rel, abs=TOL_SCORE), name


def test_the_pinned_ranking_is_the_measured_one():
    # The pinned result on this scenario: the logistic baseline has the best
    # Brier score, and the CLIMATOLOGY baseline has the lowest reliability
    # term. The learned model is best on neither. This assertion exists so the
    # README's honesty claim cannot drift away from the code.
    briers = {k: v[0] for k, v in PIN_SCORES.items()}
    rels = {k: v[1] for k, v in PIN_SCORES.items()}
    assert min(briers, key=briers.get) == "logistic"
    assert min(rels, key=rels.get) == "climatology"
    assert briers["learned"] > briers["logistic"]
    assert rels["learned"] > rels["logistic"] > rels["climatology"]


def test_reruns_are_bit_identical(scenario):
    # Determinism of the whole pipeline, not just of the seeds: build the
    # scenario a second time and compare.
    _, cg_b, teg_b = _scenario()
    _, cg_a, teg_a = scenario
    assert len(cg_a.windows) == len(cg_b.windows)
    a = np.array([w.duration_s for w in cg_a.windows])
    b = np.array([w.duration_s for w in cg_b.windows])
    assert np.array_equal(a, b)
    assert len(teg_a.edges) == len(teg_b.edges)

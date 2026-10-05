"""Campaign: case identity, serialisation, pools, execution, replay."""

from __future__ import annotations

import json

import numpy as np
import pytest

from faultinject.campaign import (
    FaultCase,
    build_pool,
    case_in_cell,
    evaluate_pool,
    execute_case,
    replay_case,
    run_campaign,
)
from faultinject.coverage import all_cells
from faultinject.faults import Injection
from faultinject.taxonomy import FaultKind, kinds

ANCHOR = FaultCase(
    Injection.create(FaultKind.SENSOR_BIAS, "pos", {"offset": 3.0}, 40, 100), 7, 150
)


def test_case_id_regression():
    assert ANCHOR.case_id == "bd32605725b734a5"


def test_case_id_is_sensitive_to_every_field():
    base = ANCHOR.case_id
    variants = [
        FaultCase(
            Injection.create(FaultKind.SENSOR_BIAS, "vel", {"offset": 3.0}, 40, 100), 7, 150
        ),
        FaultCase(
            Injection.create(FaultKind.SENSOR_BIAS, "pos", {"offset": 3.5}, 40, 100), 7, 150
        ),
        FaultCase(
            Injection.create(FaultKind.SENSOR_BIAS, "pos", {"offset": 3.0}, 41, 100), 7, 150
        ),
        FaultCase(
            Injection.create(FaultKind.SENSOR_BIAS, "pos", {"offset": 3.0}, 40, 101), 7, 150
        ),
        FaultCase(ANCHOR.injection, 8, 150),
        FaultCase(ANCHOR.injection, 7, 151),
    ]
    assert all(v.case_id != base for v in variants)
    assert len({v.case_id for v in variants}) == len(variants)


def test_case_json_round_trip():
    restored = FaultCase.from_json(ANCHOR.to_json())
    assert restored == ANCHOR
    assert restored.case_id == ANCHOR.case_id


def test_case_json_is_canonical():
    d = json.loads(ANCHOR.to_json())
    assert list(d) == sorted(d)


def test_tampered_case_rejected():
    d = ANCHOR.to_dict()
    d["seed"] = 8
    with pytest.raises(ValueError, match="case_id mismatch"):
        FaultCase.from_dict(d)


def test_case_without_id_is_accepted():
    d = ANCHOR.to_dict()
    d.pop("case_id")
    assert FaultCase.from_dict(d) == ANCHOR


def test_case_validation():
    with pytest.raises(ValueError, match="n_steps"):
        FaultCase(ANCHOR.injection, 1, 0)
    with pytest.raises(ValueError, match="seed"):
        FaultCase(ANCHOR.injection, -1, 150)


def test_case_cell():
    assert ANCHOR.cell().label() == "sensor_bias/pos/s0/d1/p[2]"


def test_execute_case_populates_result():
    res = execute_case(ANCHOR, keep_trace=True)
    assert res.trace is not None
    assert 0.0 <= res.severity.severity <= 1.0
    assert res.updates + res.skipped == 150
    d = res.to_dict()
    assert d["case"]["case_id"] == ANCHOR.case_id
    assert "severity" in d


def test_execute_case_without_trace():
    assert execute_case(ANCHOR).trace is None


def test_replay_is_bit_identical():
    a, b = replay_case(ANCHOR)
    assert a.float_bytes() == b.float_bytes()


def test_case_in_cell_round_trips_every_cell():
    for cell in all_cells():
        assert case_in_cell(cell, seed=1).cell() == cell


def test_case_in_cell_with_rng_round_trips():
    rng = np.random.default_rng(4)
    for cell in all_cells():
        assert case_in_cell(cell, seed=2, rng=rng).cell() == cell


def test_build_pool_is_deterministic_and_covers_every_cell():
    a = build_pool(pool_seed=1, replicates=1)
    b = build_pool(pool_seed=1, replicates=1)
    assert [c.case_id for c in a] == [c.case_id for c in b]
    assert len(a) == 248
    assert {c.cell() for c in a} == set(all_cells())


def test_build_pool_different_seed_differs():
    a = build_pool(pool_seed=1, replicates=1)
    b = build_pool(pool_seed=2, replicates=1)
    assert [c.case_id for c in a] != [c.case_id for c in b]


def test_build_pool_replicates():
    assert len(build_pool(pool_seed=1, replicates=3)) == 744


def test_build_pool_rejects_zero_replicates():
    with pytest.raises(ValueError, match="replicates"):
        build_pool(pool_seed=1, replicates=0)


def test_build_pool_subset():
    pool = build_pool(pool_seed=1, replicates=1, subset=(FaultKind.SENSOR_STUCK,))
    assert len(pool) == 8
    assert {c.injection.kind for c in pool} == {FaultKind.SENSOR_STUCK}


def test_evaluate_pool_in_range():
    pool = build_pool(pool_seed=1, replicates=1, subset=(FaultKind.SENSOR_BIAS,))
    sev = evaluate_pool(pool)
    assert len(sev) == len(pool)
    assert all(0.0 <= s <= 1.0 for s in sev)


def test_run_campaign_coverage_curve():
    pool = build_pool(pool_seed=1, replicates=1, subset=(FaultKind.SENSOR_STUCK,))
    res = run_campaign(pool, subset=(FaultKind.SENSOR_STUCK,))
    curve = res.coverage_curve()
    assert len(curve) == 8
    assert curve == sorted(curve)
    assert curve[-1] == 1.0
    assert 0 <= res.n_severe <= 8
    assert len(res.severities) == 8


def test_campaign_tracks_every_kind_when_unrestricted():
    pool = build_pool(pool_seed=1, replicates=1, subset=(FaultKind.SENSOR_STUCK,))
    res = run_campaign(pool)
    assert res.tracker.total == 248
    assert set(res.tracker.subset) == set(kinds())

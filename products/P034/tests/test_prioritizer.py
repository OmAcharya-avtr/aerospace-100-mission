"""Prioritiser: feature encoding, fitting, uncertainty output, validation."""

from __future__ import annotations

import numpy as np
import pytest

from faultinject.campaign import FaultCase, build_pool, evaluate_pool
from faultinject.faults import Injection
from faultinject.prioritizer import (
    CHANNELS,
    MAX_PARAMS,
    N_FEATURES,
    CampaignPrioritizer,
    encode,
    encode_many,
    feature_names,
)
from faultinject.taxonomy import FaultKind, kinds


def case(kind, channel, params, start=0, duration=150, seed=1):
    return FaultCase(Injection.create(kind, channel, params, start, duration), seed, 150)


def test_feature_layout():
    names = feature_names()
    assert len(names) == N_FEATURES == 16 + len(CHANNELS) + 2 + MAX_PARAMS
    assert names[0] == "kind:sensor_bias"
    assert names[16] == "channel:pos"
    assert names[-1] == f"param{MAX_PARAMS - 1}"


def test_encode_one_hot_and_normalisation():
    c = case(FaultKind.SENSOR_BIAS, "pos", {"offset": 10.0}, 0, 150)
    x = encode(c)
    assert x.shape == (N_FEATURES,)
    assert x[kinds().index(FaultKind.SENSOR_BIAS)] == 1.0
    assert x.sum() >= 2.0  # two one-hot entries at minimum
    assert x[16] == 1.0  # channel pos
    assert x[20] == 0.0  # start_frac 0
    assert x[21] == 1.0  # duration_frac 1 -> normalised 1
    assert x[22] == pytest.approx(1.0, abs=1e-12)  # offset at the top of its log range


def test_encode_unused_param_slots_are_zero():
    x = encode(case(FaultKind.SENSOR_STUCK, "pos", {}))
    assert list(x[22:]) == [0.0, 0.0, 0.0]


def test_encode_rejects_unknown_channel():
    c = case(FaultKind.SENSOR_BIAS, "pos", {"offset": 1.0})
    object.__setattr__(c.injection, "channel", "wing")
    with pytest.raises(ValueError, match="unknown channel"):
        encode(c)


def test_encode_many_shapes():
    assert encode_many([]).shape == (0, N_FEATURES)
    cs = [case(FaultKind.SENSOR_BIAS, "pos", {"offset": o}) for o in (0.2, 1.0, 5.0)]
    assert encode_many(cs).shape == (3, N_FEATURES)


def test_fit_and_predict_shapes():
    pool = build_pool(pool_seed=1, replicates=1, subset=(FaultKind.SENSOR_BIAS,))
    sev = evaluate_pool(pool)
    model = CampaignPrioritizer(n_estimators=20, random_state=1).fit(list(pool), list(sev))
    pred = model.predict(list(pool))
    assert len(pred) == len(pool)
    assert pred.mean.shape == pred.std.shape == (len(pool),)
    assert np.all(pred.std >= 0.0)
    assert np.all(pred.lower <= pred.upper)
    assert np.all((pred.lower >= 0.0) & (pred.upper <= 1.0))


def test_predict_on_empty_input():
    pool = build_pool(pool_seed=1, replicates=1, subset=(FaultKind.SENSOR_BIAS,))
    sev = evaluate_pool(pool)
    model = CampaignPrioritizer(n_estimators=10, random_state=1).fit(list(pool), list(sev))
    assert len(model.predict([])) == 0


def test_acquisition_uses_the_uncertainty():
    pool = build_pool(pool_seed=1, replicates=1, subset=(FaultKind.SENSOR_BIAS,))
    sev = evaluate_pool(pool)
    model = CampaignPrioritizer(n_estimators=20, random_state=1).fit(list(pool), list(sev))
    pred = model.predict(list(pool))
    assert np.allclose(pred.acquisition(0.0), pred.mean)
    assert np.all(pred.acquisition(2.0) >= pred.mean)


def test_interval_coverage_in_unit_range():
    pool = build_pool(pool_seed=1, replicates=1, subset=(FaultKind.SENSOR_BIAS,))
    sev = evaluate_pool(pool)
    model = CampaignPrioritizer(n_estimators=20, random_state=1).fit(list(pool), list(sev))
    cov, width = model.interval_coverage(list(pool), list(sev))
    assert 0.0 <= cov <= 1.0
    assert 0.0 <= width <= 1.0


def test_feature_importances_sum_to_one():
    pool = build_pool(pool_seed=1, replicates=1, subset=(FaultKind.SENSOR_BIAS,))
    sev = evaluate_pool(pool)
    model = CampaignPrioritizer(n_estimators=20, random_state=1).fit(list(pool), list(sev))
    imp = model.feature_importances()
    assert set(imp) == set(feature_names())
    assert sum(imp.values()) == pytest.approx(1.0, abs=1e-9)


def test_predict_before_fit_raises():
    model = CampaignPrioritizer()
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict([case(FaultKind.SENSOR_BIAS, "pos", {"offset": 1.0})])
    with pytest.raises(RuntimeError, match="not fitted"):
        model.feature_importances()


def test_fit_validation():
    c = case(FaultKind.SENSOR_BIAS, "pos", {"offset": 1.0})
    model = CampaignPrioritizer()
    with pytest.raises(ValueError, match="must match"):
        model.fit([c, c], [0.1])
    with pytest.raises(ValueError, match="at least 2"):
        model.fit([c], [0.1])
    with pytest.raises(ValueError, match="finite"):
        model.fit([c, c], [0.1, float("nan")])


def test_fit_is_deterministic_for_a_fixed_random_state():
    pool = build_pool(pool_seed=1, replicates=1, subset=(FaultKind.SENSOR_BIAS,))
    sev = list(evaluate_pool(pool))
    a = CampaignPrioritizer(n_estimators=15, random_state=3).fit(list(pool), sev)
    b = CampaignPrioritizer(n_estimators=15, random_state=3).fit(list(pool), sev)
    assert np.allclose(a.predict(list(pool)).mean, b.predict(list(pool)).mean)
    assert np.allclose(a.predict(list(pool)).std, b.predict(list(pool)).std)

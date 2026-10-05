"""Property-based tests for the algebraic identities in the package.

These cover the places where an identity actually exists: the binning
arithmetic, the additive and multiplicative fault models, serialisation round
trips, and the bounds the severity score and the feature encoder claim.
"""

from __future__ import annotations

import math

import numpy as np
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from faultinject.campaign import FaultCase
from faultinject.coverage import CoverageTracker, cell_of
from faultinject.faults import Injection, make_handler
from faultinject.harness import Trace, fault_rng
from faultinject.prioritizer import N_FEATURES, encode
from faultinject.severity import W_VIOL, label_for, score
from faultinject.taxonomy import DURATION_FRAC, START_FRAC, FaultKind, kinds, spec
from faultinject.wrapper import classify_value

SETTINGS = settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

ALL_PARAMS = [
    (kind, p) for kind in kinds() for p in spec(kind).params
] + [(None, START_FRAC), (None, DURATION_FRAC)]

param_index = st.integers(min_value=0, max_value=len(ALL_PARAMS) - 1)
unit = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)


@SETTINGS
@given(param_index, unit)
def test_sample_is_inside_the_declared_range(i, u):
    _, p = ALL_PARAMS[i]
    v = p.sample(u)
    assert p.lo - 1e-9 <= v <= p.hi + 1e-9
    assert math.isfinite(v)


@SETTINGS
@given(param_index, unit, unit)
def test_sample_is_monotone_in_u(i, u1, u2):
    _, p = ALL_PARAMS[i]
    lo, hi = sorted((u1, u2))
    assert p.sample(lo) <= p.sample(hi)


@SETTINGS
@given(param_index, unit)
def test_bin_of_sample_is_a_valid_bin(i, u):
    _, p = ALL_PARAMS[i]
    b = p.bin_of(p.sample(u))
    assert 0 <= b < p.n_bins


@SETTINGS
@given(param_index)
def test_edges_are_strictly_increasing(i):
    _, p = ALL_PARAMS[i]
    e = p.edges()
    assert len(e) == p.n_bins + 1
    assert all(a < b for a, b in zip(e[:-1], e[1:], strict=True))


@SETTINGS
@given(param_index)
def test_representative_round_trips_through_bin_of(i):
    _, p = ALL_PARAMS[i]
    for b in range(p.n_bins):
        assert p.bin_of(p.representative(b)) == b


@SETTINGS
@given(
    param_index,
    st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
)
def test_normalise_is_bounded(i, value):
    _, p = ALL_PARAMS[i]
    n = p.normalise(value)
    assert 0.0 <= n <= 1.0


@SETTINGS
@given(
    st.integers(min_value=0, max_value=len(kinds()) - 1),
    unit,
    st.integers(min_value=0, max_value=149),
    st.integers(min_value=1, max_value=150),
    st.integers(min_value=0, max_value=10_000),
)
def test_case_serialisation_round_trip(kind_index, u, start, duration, seed):
    kind = kinds()[kind_index]
    sp = spec(kind)
    params = {p.name: p.sample(u) for p in sp.params}
    inj = Injection.create(kind, sp.channels[0], params, start, duration)
    case = FaultCase(inj, seed, 150)
    restored = FaultCase.from_json(case.to_json())
    assert restored == case
    assert restored.case_id == case.case_id
    assert restored.cell() == case.cell()


@SETTINGS
@given(
    st.floats(min_value=0.1, max_value=10.0, allow_nan=False),
    st.floats(min_value=-1e3, max_value=1e3, allow_nan=False),
)
def test_bias_is_exactly_additive(offset, value):
    inj = Injection.create(FaultKind.SENSOR_BIAS, "pos", {"offset": offset}, 0, 10)
    handler = make_handler(inj)
    handler.reset()
    out = handler.apply_signal(0, value, fault_rng(1))
    assert out == value + offset


@SETTINGS
@given(
    st.floats(min_value=0.0, max_value=0.9, allow_nan=False),
    st.floats(min_value=-1e3, max_value=1e3, allow_nan=False),
)
def test_loss_of_effectiveness_is_exactly_multiplicative(retained, value):
    inj = Injection.create(
        FaultKind.ACTUATOR_LOSS_EFFECTIVENESS, "u", {"retained": retained}, 0, 10
    )
    handler = make_handler(inj)
    handler.reset()
    assert handler.apply_signal(0, value, fault_rng(1)) == value * retained


@SETTINGS
@given(
    st.floats(min_value=0.05, max_value=20.0, allow_nan=False),
    st.floats(min_value=-1e3, max_value=1e3, allow_nan=False),
)
def test_quantisation_error_is_at_most_half_an_lsb(lsb, value):
    inj = Injection.create(
        FaultKind.SENSOR_QUANT_COLLAPSE, "pos", {"lsb": lsb}, 0, 10
    )
    handler = make_handler(inj)
    handler.reset()
    out = handler.apply_signal(0, value, fault_rng(1))
    assert abs(out - value) <= 0.5 * lsb + 1e-9 * max(1.0, abs(value))
    quotient = out / lsb
    assert abs(quotient - round(quotient)) < 1e-6


@SETTINGS
@given(st.floats(allow_nan=True, allow_infinity=True))
def test_classify_value_is_total(value):
    assert classify_value(value) in (None, "nan", "inf", "subnormal", "large")


@SETTINGS
@given(
    st.lists(
        st.floats(min_value=-10.0, max_value=10.0, allow_nan=False),
        min_size=2,
        max_size=30,
    ),
    st.lists(
        st.floats(min_value=-10.0, max_value=10.0, allow_nan=False),
        min_size=2,
        max_size=30,
    ),
)
def test_severity_is_bounded_and_labelled(a, b):
    n = min(len(a), len(b))
    assume(n >= 2)

    def trace(p):
        tr = Trace(seed=1, n_steps=n)
        tr.p_true = list(p[:n])
        tr.v_true = [0.0] * n
        tr.p_hat = [0.0] * n
        tr.v_hat = [0.0] * n
        tr.u_applied = [0.0] * n
        tr.ref = [0.0] * n
        return tr

    rep = score(trace(a), trace(b))
    assert 0.0 <= rep.severity <= 1.0
    assert rep.label == label_for(rep.severity)
    assert rep.severe == (rep.severity >= 0.6)


@SETTINGS
@given(
    st.lists(
        st.floats(min_value=-10.0, max_value=10.0, allow_nan=False),
        min_size=2,
        max_size=30,
    )
)
def test_identical_traces_have_zero_severity(p):
    n = len(p)

    def trace():
        tr = Trace(seed=1, n_steps=n)
        tr.p_true = list(p)
        tr.v_true = [0.0] * n
        tr.p_hat = [0.0] * n
        tr.v_hat = [0.0] * n
        tr.u_applied = [0.0] * n
        tr.ref = [0.0] * n
        return tr

    rep = score(trace(), trace())
    # The deviation and RMSE terms vanish for identical traces. The violation
    # term is absolute by design (a position beyond POS_LIMIT is a violation
    # whether or not a fault caused it), so the residual severity is exactly
    # W_VIOL times the violation fraction and nothing else.
    assert rep.max_pos_deviation == 0.0
    assert rep.components["s_dev"] == 0.0
    assert rep.components["s_rmse"] == 0.0
    assert rep.severity == W_VIOL * rep.violation_fraction


@SETTINGS
@given(
    st.lists(
        st.tuples(
            st.integers(min_value=0, max_value=len(kinds()) - 1),
            unit,
            st.integers(min_value=0, max_value=149),
            st.integers(min_value=1, max_value=150),
        ),
        min_size=1,
        max_size=12,
    )
)
def test_coverage_is_monotone_and_bounded(items):
    tracker = CoverageTracker.full()
    previous = 0.0
    for kind_index, u, start, duration in items:
        kind = kinds()[kind_index]
        sp = spec(kind)
        params = {p.name: p.sample(u) for p in sp.params}
        inj = Injection.create(kind, sp.channels[0], params, start, duration)
        tracker.add(inj, 150)
        assert previous <= tracker.fraction <= 1.0
        previous = tracker.fraction


@SETTINGS
@given(
    st.integers(min_value=0, max_value=len(kinds()) - 1),
    unit,
    st.integers(min_value=0, max_value=149),
    st.integers(min_value=1, max_value=150),
)
def test_feature_vector_is_bounded_with_two_one_hot_entries(kind_index, u, start, duration):
    kind = kinds()[kind_index]
    sp = spec(kind)
    params = {p.name: p.sample(u) for p in sp.params}
    inj = Injection.create(kind, sp.channels[0], params, start, duration)
    x = encode(FaultCase(inj, 1, 150))
    assert x.shape == (N_FEATURES,)
    assert np.all((x >= 0.0) & (x <= 1.0))
    assert x[:16].sum() == 1.0
    assert x[16:20].sum() == 1.0


@SETTINGS
@given(
    st.integers(min_value=0, max_value=len(kinds()) - 1),
    unit,
    st.integers(min_value=0, max_value=149),
    st.integers(min_value=1, max_value=150),
)
def test_cell_of_agrees_with_the_parameter_binning(kind_index, u, start, duration):
    kind = kinds()[kind_index]
    sp = spec(kind)
    params = {p.name: p.sample(u) for p in sp.params}
    inj = Injection.create(kind, sp.channels[0], params, start, duration)
    cell = cell_of(inj, 150)
    assert cell.start_bin == START_FRAC.bin_of(start / 150)
    assert cell.duration_bin == DURATION_FRAC.bin_of(duration / 150)
    for p, b in zip(sp.params, cell.param_bins, strict=True):
        assert b == p.bin_of(params[p.name])

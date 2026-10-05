"""Severity scoring: hand-computed scores, overrides, labels, validation."""

from __future__ import annotations

import pytest

from faultinject.faults import Injection
from faultinject.harness import Trace, nominal_trace, run_case
from faultinject.severity import (
    DEV_REF,
    POS_LIMIT,
    RMSE_SAT,
    SEVERE_THRESHOLD,
    W_DEV,
    W_RMSE,
    W_VIOL,
    label_for,
    score,
)
from faultinject.taxonomy import FaultKind


def make_trace(seed: int, p, ref=None) -> Trace:
    n = len(p)
    ref = ref if ref is not None else [0.0] * n
    tr = Trace(seed=seed, n_steps=n)
    tr.p_true = list(p)
    tr.v_true = [0.0] * n
    tr.p_hat = [0.0] * n
    tr.v_hat = [0.0] * n
    tr.u_applied = [0.0] * n
    tr.ref = list(ref)
    return tr


def test_weights_sum_to_one():
    assert W_DEV + W_RMSE + W_VIOL == pytest.approx(1.0, abs=1e-15)


def test_identical_traces_score_zero():
    nom = make_trace(1, [0.1, 0.2, 0.3])
    rep = score(make_trace(1, [0.1, 0.2, 0.3]), nom)
    assert rep.severity == 0.0
    assert rep.label == "negligible"
    assert rep.max_pos_deviation == 0.0
    assert rep.rmse_ratio == pytest.approx(1.0, abs=1e-15)


def test_hand_computed_score():
    # nominal p = [1, 1, 1], ref = [1, 1, 1] -> rms nominal error = 0, so the
    # rmse ratio is defined as 1.0 and s_rmse = 0.
    # faulted p = [1, 1.25, 1] -> d_p = 0.25, s_dev = 0.25/0.5 = 0.5
    # no |p| > 2 so s_viol = 0 -> severity = 0.5 * 0.5 = 0.25
    nom = make_trace(2, [1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
    f = make_trace(2, [1.0, 1.25, 1.0], [1.0, 1.0, 1.0])
    rep = score(f, nom)
    assert rep.max_pos_deviation == pytest.approx(0.25, abs=1e-15)
    assert rep.components["s_dev"] == pytest.approx(0.5, abs=1e-15)
    assert rep.components["s_rmse"] == 0.0
    assert rep.components["s_viol"] == 0.0
    assert rep.severity == pytest.approx(0.25, abs=1e-15)
    assert rep.label == "minor"


def test_hand_computed_violation_fraction():
    # two of four steps beyond POS_LIMIT = 2 -> s_viol = 0.5
    nom = make_trace(3, [0.0] * 4)
    f = make_trace(3, [0.0, 3.0, 3.0, 0.0])
    rep = score(f, nom)
    assert rep.violation_fraction == pytest.approx(0.5, abs=1e-15)
    assert rep.components["s_viol"] == pytest.approx(0.5, abs=1e-15)


def test_rmse_saturation():
    # nominal tracking error rms 1.0 ; faulted rms 5.0 -> ratio 5 = RMSE_SAT
    nom = make_trace(4, [1.0, 1.0], [0.0, 0.0])
    f = make_trace(4, [5.0, 5.0], [0.0, 0.0])
    rep = score(f, nom)
    assert rep.rmse_ratio == pytest.approx(RMSE_SAT, rel=1e-15)
    assert rep.components["s_rmse"] == pytest.approx(1.0, abs=1e-15)


def test_nonfinite_override():
    nom = make_trace(5, [0.0, 0.0])
    f = make_trace(5, [0.0, float("nan")])
    rep = score(f, nom)
    assert rep.nonfinite is True
    assert rep.severity == 1.0
    assert rep.label == "severe"


def test_divergence_override_floor():
    nom = make_trace(6, [0.0] * 3)
    f = make_trace(6, [0.0, 1e6, 0.0])
    rep = score(f, nom)
    assert rep.divergent is True
    assert rep.severity >= 0.9


def test_severity_is_clipped_to_one():
    nom = make_trace(7, [0.0] * 4)
    f = make_trace(7, [100.0] * 4)
    rep = score(f, nom)
    assert rep.severity <= 1.0


def test_label_boundaries():
    assert label_for(0.0) == "negligible"
    assert label_for(0.0999) == "negligible"
    assert label_for(0.1) == "minor"
    assert label_for(0.2999) == "minor"
    assert label_for(0.3) == "moderate"
    assert label_for(0.5999) == "moderate"
    assert label_for(SEVERE_THRESHOLD) == "severe"
    assert label_for(1.0) == "severe"


def test_label_rejects_out_of_range():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        label_for(1.5)


def test_score_requires_matching_seed_and_length():
    a = make_trace(1, [0.0, 0.0])
    b = make_trace(2, [0.0, 0.0])
    with pytest.raises(ValueError, match="matching seeds"):
        score(a, b)
    c = make_trace(1, [0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="matching lengths"):
        score(c, a)


def test_severe_property_matches_threshold():
    nom = make_trace(9, [0.0] * 3)
    # d_p = 10 -> s_dev 1 ; all three steps beyond POS_LIMIT -> s_viol 1
    f = make_trace(9, [10.0, 10.0, 10.0])
    assert score(f, nom).severe is True
    assert score(make_trace(9, [0.0] * 3), nom).severe is False


def test_report_to_dict_round_trips_keys():
    nom = make_trace(10, [0.0, 0.0])
    rep = score(make_trace(10, [0.0, 0.2]), nom)
    d = rep.to_dict()
    assert set(d) == {
        "severity",
        "label",
        "max_pos_deviation",
        "max_vel_deviation",
        "rmse_ratio",
        "violation_fraction",
        "nonfinite",
        "divergent",
        "components",
    }


def test_real_nan_case_scores_one():
    inj = Injection.create(FaultKind.NUMERICAL_NAN, "pos", {}, 50, 10)
    rep = score(run_case([inj], 3, 150), nominal_trace(3, 150))
    assert rep.severity == 1.0


def test_dev_ref_and_pos_limit_are_documented_constants():
    assert DEV_REF == 0.5
    assert POS_LIMIT == 2.0

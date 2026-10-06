"""Markov and semi-Markov fitting and goodness-of-fit tests."""

from __future__ import annotations

import numpy as np
import pytest

from linkoutage.markov import (
    dwell_lengths,
    dwell_time_goodness_of_fit,
    fit_markov,
    fit_semi_markov,
    markov_order_test,
    state_sequence,
)


def _simulate_markov(p: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    k = p.shape[0]
    out = np.empty(n, dtype=np.int64)
    s = 0
    for i in range(n):
        out[i] = s
        s = int(rng.choice(k, p=p[s]))
    return out


def test_state_sequence_two_states():
    a = np.array([1.0, 0.5, 0.5, 1.0, 0.4])
    assert state_sequence(a, [0.6]).tolist() == [1, 0, 0, 1, 0]


def test_state_sequence_three_states():
    a = np.array([0.1, 0.5, 0.9, 1.4])
    assert state_sequence(a, [0.3, 1.0]).tolist() == [0, 1, 1, 2]


def test_state_sequence_boundary_strict():
    a = np.array([0.6, 0.6, 1.0])
    assert state_sequence(a, [0.6], strict_below=True).tolist() == [1, 1, 1]
    assert state_sequence(a, [0.6], strict_below=False).tolist() == [0, 0, 1]


def test_state_sequence_rejects_unsorted_thresholds():
    with pytest.raises(ValueError, match="strictly increasing"):
        state_sequence(np.array([1.0, 2.0]), [1.0, 0.5])


def test_state_sequence_rejects_empty_thresholds():
    with pytest.raises(ValueError, match="at least one"):
        state_sequence(np.array([1.0, 2.0]), [])


def test_fit_markov_alternating_sequence():
    s = np.array([0, 1] * 50)
    fit = fit_markov(s)
    assert fit.transition_matrix[0, 1] == pytest.approx(1.0)
    assert fit.transition_matrix[1, 0] == pytest.approx(1.0)
    assert fit.transition_matrix[0, 0] == pytest.approx(0.0)


def test_fit_markov_hand_counted_matrix():
    # 0 0 1 0 1 1 : transitions 0->0, 0->1, 1->0, 0->1, 1->1
    # row 0: [1, 2] -> [1/3, 2/3];  row 1: [1, 1] -> [1/2, 1/2]
    s = np.array([0, 0, 1, 0, 1, 1])
    fit = fit_markov(s)
    assert fit.transition_counts.tolist() == [[1, 2], [1, 1]]
    assert fit.transition_matrix[0].tolist() == pytest.approx([1 / 3, 2 / 3])
    assert fit.transition_matrix[1].tolist() == pytest.approx([0.5, 0.5])


def test_fit_markov_recovers_a_known_chain():
    p = np.array([[0.90, 0.10], [0.02, 0.98]])
    s = _simulate_markov(p, 200_000, seed=21)
    fit = fit_markov(s)
    assert np.max(np.abs(fit.transition_matrix - p)) < 0.01


def test_markov_stationary_matches_occupancy():
    p = np.array([[0.90, 0.10], [0.02, 0.98]])
    s = _simulate_markov(p, 200_000, seed=22)
    fit = fit_markov(s)
    assert np.max(np.abs(fit.stationary - fit.empirical_occupancy)) < 0.01


def test_markov_mean_dwell_matches_geometric_prediction():
    p = np.array([[0.90, 0.10], [0.02, 0.98]])
    s = _simulate_markov(p, 200_000, seed=23)
    fit = fit_markov(s)
    assert fit.empirical_mean_dwell_samples[0] == pytest.approx(10.0, rel=0.05)
    assert fit.empirical_mean_dwell_samples[1] == pytest.approx(50.0, rel=0.05)


def test_markov_report_contains_the_matrix():
    s = _simulate_markov(np.array([[0.8, 0.2], [0.1, 0.9]]), 5000, seed=24)
    text = fit_markov(s).report()
    assert "transition matrix" in text
    assert "AIC" in text


def test_markov_free_parameter_count():
    s = _simulate_markov(np.array([[0.8, 0.2], [0.1, 0.9]]), 1000, seed=25)
    assert fit_markov(s).n_parameters == 2
    s3 = np.array([0, 1, 2] * 100)
    assert fit_markov(s3).n_parameters == 6


def test_dwell_lengths_hand_counted():
    s = np.array([0, 0, 1, 1, 1, 0, 1])
    per_state, n_cens = dwell_lengths(s, n_states=2)
    # runs: [0,2) state 0 censored (touches 0); [2,5) state 1 complete len 3;
    #       [5,6) state 0 complete len 1; [6,7) state 1 censored (touches end)
    assert per_state[0].tolist() == [1]
    assert per_state[1].tolist() == [3]
    assert n_cens == [1, 1]


def test_dwell_lengths_including_censored():
    s = np.array([0, 0, 1, 1, 1, 0, 1])
    per_state, _ = dwell_lengths(s, n_states=2, exclude_censored=False)
    assert sorted(per_state[0].tolist()) == [1, 2]
    assert sorted(per_state[1].tolist()) == [1, 3]


def test_dwell_gof_does_not_reject_a_true_markov_chain():
    p = np.array([[0.90, 0.10], [0.02, 0.98]])
    s = _simulate_markov(p, 200_000, seed=26)
    fit = fit_markov(s)
    gof = dwell_time_goodness_of_fit(s, fit)
    assert all(g.valid for g in gof)
    assert all(g.p_value > 1e-3 for g in gof), [(g.state, g.p_value) for g in gof]


def test_dwell_gof_geometric_cv_bound_holds_for_a_true_chain():
    p = np.array([[0.90, 0.10], [0.02, 0.98]])
    s = _simulate_markov(p, 200_000, seed=27)
    gof = dwell_time_goodness_of_fit(s, fit_markov(s))
    for g in gof:
        assert g.cv_observed == pytest.approx(g.cv_geometric, abs=0.05)


def test_dwell_gof_rejects_the_lognormal_channel(short_channel):
    s = state_sequence(short_channel.amplitude, [0.6])
    fit = fit_markov(s)
    gof = dwell_time_goodness_of_fit(s, fit)
    assert all(g.p_value < 1e-10 for g in gof)
    # A geometric dwell has cv = sqrt(1-p) <= 1; the channel's is far above.
    assert gof[0].cv_observed > 1.5
    assert gof[0].cv_geometric < 1.0


def test_dwell_gof_line_renders(short_channel):
    s = state_sequence(short_channel.amplitude, [0.6])
    for g in dwell_time_goodness_of_fit(s, fit_markov(s)):
        assert "state" in g.line()


def test_markov_order_test_accepts_a_true_first_order_chain():
    p = np.array([[0.90, 0.10], [0.02, 0.98]])
    s = _simulate_markov(p, 200_000, seed=28)
    result = markov_order_test(s)
    assert result.p_value > 1e-3
    assert result.cramers_v < 0.02


def test_markov_order_test_detects_second_order_structure():
    # Next state depends on the previous TWO states, not just one.
    rng = np.random.default_rng(29)
    n = 200_000
    s = np.zeros(n, dtype=np.int64)
    for i in range(2, n):
        if s[i - 1] == s[i - 2]:
            s[i] = s[i - 1] if rng.random() < 0.95 else 1 - s[i - 1]
        else:
            s[i] = s[i - 1] if rng.random() < 0.40 else 1 - s[i - 1]
    result = markov_order_test(s)
    assert result.p_value < 1e-12
    assert result.cramers_v > 0.1


def test_markov_order_test_rejects_the_lognormal_channel(short_channel):
    s = state_sequence(short_channel.amplitude, [0.6])
    result = markov_order_test(s)
    assert result.p_value < 1e-12
    assert result.cramers_v > 0.1
    assert "chi2" in result.report()


def test_markov_order_test_sparse_case_reports_a_note():
    s = np.array([0] * 20 + [1] * 20 + [0] * 20)
    result = markov_order_test(s)
    assert result.note


def test_semi_markov_two_state_jump_chain_is_deterministic(short_channel):
    s = state_sequence(short_channel.amplitude, [0.6])
    sm = fit_semi_markov(s)
    assert sm.jump_matrix[0, 1] == pytest.approx(1.0)
    assert sm.jump_matrix[1, 0] == pytest.approx(1.0)
    assert sm.jump_matrix[0, 0] == pytest.approx(0.0)


def test_semi_markov_mean_dwell_matches_the_data(short_channel):
    s = state_sequence(short_channel.amplitude, [0.6])
    sm = fit_semi_markov(s)
    per_state, _ = dwell_lengths(s, n_states=2)
    assert sm.mean_dwell_samples[0] == pytest.approx(float(per_state[0].mean()))
    assert sm.mean_dwell_samples[1] == pytest.approx(float(per_state[1].mean()))


def test_semi_markov_occupancy_is_close_to_measured(short_channel):
    s = state_sequence(short_channel.amplitude, [0.6])
    sm = fit_semi_markov(s)
    measured = float(np.mean(s == 0))
    assert sm.occupancy()[0] == pytest.approx(measured, abs=0.01)


def test_semi_markov_simulation_length_and_states(short_channel):
    s = state_sequence(short_channel.amplitude, [0.6])
    sm = fit_semi_markov(s)
    sim = sm.simulate(5000, np.random.default_rng(30))
    assert sim.size == 5000
    assert set(np.unique(sim).tolist()) <= {0, 1}


def test_semi_markov_simulation_reproduces_occupancy(short_channel):
    s = state_sequence(short_channel.amplitude, [0.6])
    sm = fit_semi_markov(s)
    sim = sm.simulate(400_000, np.random.default_rng(31))
    assert float(np.mean(sim == 0)) == pytest.approx(float(np.mean(s == 0)), abs=0.02)


def test_semi_markov_has_more_parameters_than_markov(short_channel):
    s = state_sequence(short_channel.amplitude, [0.6])
    assert fit_semi_markov(s).n_parameters > fit_markov(s).n_parameters


def test_semi_markov_report_states_the_parameter_cost(short_channel):
    s = state_sequence(short_channel.amplitude, [0.6])
    assert "free parameters" in fit_semi_markov(s).report()


def test_three_state_markov_fit(short_channel):
    s = state_sequence(short_channel.amplitude, [0.5, 0.8])
    fit = fit_markov(s)
    assert fit.n_states == 3
    assert np.allclose(fit.transition_matrix.sum(axis=1), 1.0)


def test_three_state_order_test_runs(short_channel):
    s = state_sequence(short_channel.amplitude, [0.5, 0.8])
    result = markov_order_test(s)
    assert result.n_states == 3
    assert result.dof >= 1


def test_states_too_short_raises():
    with pytest.raises(ValueError, match="at least 3"):
        fit_markov(np.array([0, 1]))


def test_negative_states_raise():
    with pytest.raises(ValueError, match="non-negative"):
        fit_markov(np.array([0, -1, 1]))


def test_state_out_of_range_raises():
    with pytest.raises(ValueError, match="n_states"):
        fit_markov(np.array([0, 1, 2, 1]), n_states=2)


def test_single_state_raises():
    with pytest.raises(ValueError, match="at least 2"):
        fit_markov(np.array([0, 0, 0]), n_states=1)

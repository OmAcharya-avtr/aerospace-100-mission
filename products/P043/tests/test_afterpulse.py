"""Afterpulsing: closed-form cluster moments, Fano factors, dead-time interaction."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from photoncount import afterpulse as ap


def test_cascading_moments_known_answers():
    # p = 0.1, geometric cluster: E[C] = 1/0.9 = 1.111..., Var = 0.1/0.81 = 0.12345679
    mom = ap.cluster_size_moments(0.1, "cascading")
    assert mom["mean"] == pytest.approx(1.1111111111, rel=1e-9)
    assert mom["variance"] == pytest.approx(0.1234567901, rel=1e-8)
    assert mom["second"] == pytest.approx(1.1 / 0.81, rel=1e-9)


def test_first_order_moments_known_answers():
    # p = 0.1: E[C] = 1.1, E[C^2] = 1.3, Var = 1.3 - 1.21 = 0.09
    mom = ap.cluster_size_moments(0.1, "first_order")
    assert mom["mean"] == pytest.approx(1.1)
    assert mom["second"] == pytest.approx(1.3)
    assert mom["variance"] == pytest.approx(0.09, abs=1e-12)


def test_fano_factor_known_answers():
    # cascading: (1+p)/(1-p) = 1.1/0.9 = 1.2222222
    assert ap.fano_factor_prediction(0.1, "cascading") == pytest.approx(1.2222222, rel=1e-6)
    # first order: (1+3p)/(1+p) = 1.3/1.1 = 1.1818182
    assert ap.fano_factor_prediction(0.1, "first_order") == pytest.approx(1.1818182, rel=1e-6)


def test_fano_factor_is_one_without_afterpulsing():
    for model in ("cascading", "first_order"):
        assert ap.fano_factor_prediction(0.0, model) == pytest.approx(1.0)


def test_fano_factor_always_above_one():
    for p in (0.001, 0.01, 0.1, 0.5, 0.9):
        assert ap.fano_factor_prediction(p, "cascading") > 1.0
        assert ap.fano_factor_prediction(min(p, 1.0), "first_order") > 1.0


def test_cascading_inflates_more_than_first_order():
    for p in (0.01, 0.05, 0.2, 0.5):
        assert (
            ap.cluster_size_moments(p, "cascading")["mean"]
            > ap.cluster_size_moments(p, "first_order")["mean"]
        )


def test_effective_probability_known_answer():
    # p = 0.03, tau = 50 ns, t_ap = 100 ns: p_eff = 0.03 * exp(-0.5) = 0.018195920
    assert ap.effective_afterpulse_probability(0.03, 50e-9, 100e-9) == pytest.approx(
        0.0181959198, rel=1e-8
    )


def test_effective_probability_limits():
    # No dead time suppresses nothing.
    assert ap.effective_afterpulse_probability(0.1, 0.0, 1e-7) == pytest.approx(0.1)
    # A dead time far longer than the release time suppresses almost everything.
    assert ap.effective_afterpulse_probability(0.1, 1e-4, 1e-8) < 1e-12


def test_forward_and_inverse_are_exact_inverses():
    for model in ("cascading", "first_order"):
        m = ap.observed_from_primary(1e5, 0.07, model)
        assert float(ap.primary_from_observed(m, 0.07, model)[0]) == pytest.approx(1e5, rel=1e-12)


def test_cascading_rejects_unit_probability():
    with pytest.raises(ValueError, match="cascading"):
        ap.cluster_size_moments(1.0, "cascading")


def test_first_order_accepts_unit_probability():
    assert ap.cluster_size_moments(1.0, "first_order")["mean"] == pytest.approx(2.0)


@pytest.mark.parametrize("bad", [-0.1, float("nan"), float("inf"), 1.5])
def test_bad_probability_raises(bad):
    with pytest.raises(ValueError):
        ap.cluster_size_moments(bad, "first_order")


def test_unknown_model_raises():
    with pytest.raises(ValueError, match="model must be"):
        ap.cluster_size_moments(0.1, "branching")  # type: ignore[arg-type]


def test_bad_delay_raises():
    with pytest.raises(ValueError):
        ap.effective_afterpulse_probability(0.1, 1e-7, 0.0)
    with pytest.raises(ValueError):
        ap.effective_afterpulse_probability(0.1, -1e-7, 1e-7)


def test_negative_rate_raises():
    with pytest.raises(ValueError):
        ap.observed_from_primary(-1.0, 0.1)
    with pytest.raises(ValueError):
        ap.primary_from_observed(-1.0, 0.1)


@given(st.floats(min_value=0.0, max_value=0.95))
@settings(max_examples=60, deadline=None)
def test_compound_poisson_identity(p):
    """Fano = E[C^2]/E[C] must hold for both models by construction."""
    for model in ("cascading", "first_order"):
        mom = ap.cluster_size_moments(p, model)
        assert ap.fano_factor_prediction(p, model) == pytest.approx(
            mom["second"] / mom["mean"], rel=1e-12
        )


@given(st.floats(min_value=0.0, max_value=0.9))
@settings(max_examples=40, deadline=None)
def test_observed_rate_never_below_primary(p):
    assert float(ap.observed_from_primary(1000.0, p)[0]) >= 1000.0 - 1e-9

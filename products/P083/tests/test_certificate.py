"""Baseline 1: the closed forms, the inversion, the verdicts and the refusals."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.linalg import expm

from hilverdict.certificate import (
    DISMISSED,
    UNDECIDED,
    CertificateModel,
    certified_overrun_s,
    certify_miss,
    certify_run,
    divergence_bound,
    exponential_integral,
    gronwall_horizon_factor,
    horizon_gains,
    log_norm,
    require_admissible,
)
from hilverdict.plant import SHORT_PERIOD, DivergenceMetric, LoopModel, Plant
from hilverdict.synthetic import generate_run
from hilverdict.timeline import MissEvent, RunLog, miss_events


def _event(overrun=0.004, step=0.01, remaining=40, index=5):
    return MissEvent(
        index=index,
        overrun_s=overrun,
        hold_periods=int(overrun // 0.010),
        remainder_s=overrun - int(overrun // 0.010) * 0.010,
        period_s=0.010,
        stale_command_rad=0.0,
        input_step_rad=step,
        state_at_miss=np.zeros(2),
        steps_remaining=remaining,
    )


def test_log_norm_of_a_diagonal_matrix_is_its_largest_entry():
    assert log_norm(np.diag([-3.0, -1.0])) == pytest.approx(-1.0, rel=1e-15, abs=0.0)


def test_log_norm_of_a_symmetric_matrix_is_its_largest_eigenvalue():
    matrix = np.array([[1.0, 2.0], [2.0, 1.0]])
    assert log_norm(matrix) == pytest.approx(3.0, rel=1e-14, abs=1e-15)


def test_log_norm_ignores_the_antisymmetric_part():
    base = np.diag([-2.0, -4.0])
    skew = np.array([[0.0, 7.0], [-7.0, 0.0]])
    assert log_norm(base + skew) == pytest.approx(log_norm(base), rel=1e-15, abs=1e-15)


def test_log_norm_is_translation_equivariant():
    matrix = SHORT_PERIOD.jacobian(0.0)
    shifted = matrix + 3.0 * np.eye(2)
    assert log_norm(shifted) == pytest.approx(log_norm(matrix) + 3.0, rel=1e-13, abs=0.0)


def test_log_norm_rejects_a_non_square_matrix():
    with pytest.raises(ValueError, match="must be square"):
        log_norm(np.ones((2, 3)))


@pytest.mark.parametrize("duration", [1e-6, 1e-3, 1e-2, 0.1])
def test_dahlquist_inequality_holds_on_the_declared_jacobian(duration):
    matrix = SHORT_PERIOD.jacobian(0.0)
    left = float(np.linalg.norm(expm(matrix * duration), 2))
    right = math.exp(log_norm(matrix) * duration)
    assert left <= right * (1.0 + 1e-13)


def test_exponential_integral_at_zero_mu_is_exactly_the_duration():
    assert exponential_integral(0.0, 0.01) == 0.01


def test_exponential_integral_at_zero_duration_is_zero():
    assert exponential_integral(5.0, 0.0) == pytest.approx(0.0, rel=0.0, abs=0.0)


@pytest.mark.parametrize("mu", [-50.0, -1.0, 1.0, 50.0])
def test_exponential_integral_matches_the_closed_form(mu):
    expected = math.expm1(mu * 0.01) / mu
    assert exponential_integral(mu, 0.01) == pytest.approx(
        expected, rel=1e-15, abs=0.0
    )


def test_exponential_integral_is_continuous_at_zero_mu():
    assert exponential_integral(1e-14, 0.01) == pytest.approx(
        0.01, rel=1e-13, abs=1e-18
    )


def test_exponential_integral_is_increasing_in_duration():
    values = [exponential_integral(20.0, t) for t in np.linspace(0.0, 0.05, 50)]
    assert all(second > first for first, second in zip(values, values[1:], strict=False))


def test_exponential_integral_rejects_a_negative_duration():
    with pytest.raises(ValueError, match="non-negative"):
        exponential_integral(1.0, -0.001)


def test_exponential_integral_is_below_the_duration_for_negative_mu():
    assert exponential_integral(-10.0, 0.01) < 0.01


def test_exponential_integral_is_above_the_duration_for_positive_mu():
    assert exponential_integral(10.0, 0.01) > 0.01


def test_horizon_gains_start_at_one(model):
    gains = horizon_gains(model.scaled_monodromy(), 5)
    assert gains[0] == pytest.approx(1.0, rel=0.0, abs=0.0)


def test_horizon_gains_length_is_steps_plus_one(model):
    assert horizon_gains(model.scaled_monodromy(), 7).shape == (8,)


def test_horizon_gain_one_is_the_matrix_norm(model):
    gains = horizon_gains(model.scaled_monodromy(), 3)
    assert gains[1] == pytest.approx(
        float(np.linalg.norm(model.scaled_monodromy(), 2)), rel=1e-14, abs=0.0
    )


def test_horizon_gains_are_submultiplicative(model):
    gains = horizon_gains(model.scaled_monodromy(), 12)
    for index in range(1, 12):
        assert gains[index + 1] <= gains[index] * gains[1] * (1.0 + 1e-12)


def test_horizon_gains_of_the_identity_are_all_one():
    gains = horizon_gains(np.eye(2), 6)
    assert gains == pytest.approx(np.ones(7), rel=0.0, abs=1e-15)


def test_horizon_gains_rejects_negative_steps(model):
    with pytest.raises(ValueError, match="non-negative"):
        horizon_gains(model.scaled_monodromy(), -1)


def test_horizon_gains_rejects_a_non_square_matrix():
    with pytest.raises(ValueError, match="must be square"):
        horizon_gains(np.ones((2, 3)), 4)


def test_gronwall_dominates_the_discrete_horizon_factor(model):
    gains = horizon_gains(model.scaled_monodromy(), 20)
    for horizon in range(0, 21):
        assert float(np.max(gains[: horizon + 1])) <= gronwall_horizon_factor(
            model, horizon
        )


def test_gronwall_at_zero_horizon_is_one(model):
    assert gronwall_horizon_factor(model, 0) == pytest.approx(1.0, rel=1e-15, abs=0.0)


def test_gronwall_rejects_negative_steps(model):
    with pytest.raises(ValueError, match="non-negative"):
        gronwall_horizon_factor(model, -1)


def test_certificate_model_constants_are_positive(constants):
    assert constants.mu_s > 0.0
    assert constants.input_gain > 0.0
    assert constants.free_factor >= 1.0


def test_certificate_model_horizon_steps(constants):
    assert constants.horizon_steps == constants.gains.shape[0] - 1


def test_gamma_sup_is_monotone_in_the_horizon(constants):
    values = [constants.gamma_sup(n) for n in range(constants.horizon_steps + 1)]
    assert all(second >= first for first, second in zip(values, values[1:], strict=False))


def test_gamma_sup_rejects_a_horizon_beyond_the_build(constants):
    with pytest.raises(ValueError, match="rebuild"):
        constants.gamma_sup(constants.horizon_steps + 1)


def test_gamma_sup_rejects_a_negative_horizon(constants):
    with pytest.raises(ValueError, match="non-negative"):
        constants.gamma_sup(-1)


def test_certificate_model_build_rejects_a_negative_horizon(model):
    with pytest.raises(ValueError, match="non-negative"):
        CertificateModel.build(model, -1)


def test_prefactor_is_the_product_of_the_three_constants(constants):
    expected = (
        constants.free_factor * constants.gamma_sup(10) * constants.input_gain
    )
    assert constants.prefactor(10) == pytest.approx(expected, rel=1e-15, abs=0.0)


def test_divergence_bound_is_zero_for_a_zero_overrun(constants):
    assert divergence_bound(constants, 0.0, 0.1, 10) == pytest.approx(
        0.0, rel=0.0, abs=0.0
    )


def test_divergence_bound_is_zero_for_a_zero_input_step(constants):
    assert divergence_bound(constants, 0.01, 0.0, 10) == pytest.approx(
        0.0, rel=0.0, abs=0.0
    )


def test_divergence_bound_is_proportional_to_the_input_step(constants):
    single = divergence_bound(constants, 0.004, 0.01, 20)
    double = divergence_bound(constants, 0.004, 0.02, 20)
    assert double == pytest.approx(2.0 * single, rel=1e-13, abs=0.0)


def test_divergence_bound_increases_with_the_overrun(constants):
    values = [
        divergence_bound(constants, tau, 0.01, 20)
        for tau in np.linspace(0.0, 0.03, 40)
    ]
    assert all(second > first for first, second in zip(values, values[1:], strict=False))


def test_divergence_bound_does_not_decrease_with_the_horizon(constants):
    values = [divergence_bound(constants, 0.004, 0.01, n) for n in range(0, 40)]
    assert all(second >= first for first, second in zip(values, values[1:], strict=False))


def test_divergence_bound_rejects_a_negative_input_step(constants):
    with pytest.raises(ValueError, match="non-negative"):
        divergence_bound(constants, 0.004, -0.01, 10)


def test_divergence_bound_matches_its_factors_by_hand(constants):
    overrun = 0.003
    step = 0.02
    expected = (
        constants.free_factor
        * constants.gamma_sup(15)
        * constants.input_gain
        * exponential_integral(constants.mu_s, overrun)
        * step
    )
    assert divergence_bound(constants, overrun, step, 15) == pytest.approx(
        expected, rel=1e-15, abs=0.0
    )


@pytest.mark.parametrize("degrees", [0.01, 0.1, 1.0, 10.0])
@pytest.mark.parametrize("horizon", [0, 5, 40])
def test_certified_overrun_inverts_the_bound(constants, degrees, horizon):
    step = math.radians(degrees)
    threshold = certified_overrun_s(constants, step, horizon, 1.0)
    assert divergence_bound(constants, threshold, step, horizon) == pytest.approx(
        1.0, rel=1e-12, abs=0.0
    )


def test_certified_overrun_is_infinite_for_a_zero_input_step(constants):
    assert math.isinf(certified_overrun_s(constants, 0.0, 10, 1.0))


def test_certified_overrun_decreases_with_the_input_step(constants):
    values = [
        certified_overrun_s(constants, math.radians(d), 20, 1.0)
        for d in (0.01, 0.1, 1.0, 5.0, 20.0)
    ]
    assert all(second < first for first, second in zip(values, values[1:], strict=False))


def test_certified_overrun_increases_with_the_tolerance(constants):
    small = certified_overrun_s(constants, 0.01, 20, 0.5)
    large = certified_overrun_s(constants, 0.01, 20, 2.0)
    assert large > small


def test_certified_overrun_rejects_a_non_positive_tolerance(constants):
    with pytest.raises(ValueError, match="tolerance must be positive"):
        certified_overrun_s(constants, 0.01, 10, 0.0)


def test_certified_overrun_rejects_a_negative_input_step(constants):
    with pytest.raises(ValueError, match="non-negative"):
        certified_overrun_s(constants, -0.01, 10, 1.0)


def test_certified_overrun_is_infinite_for_a_stable_log_norm_beyond_the_asymptote():
    # With mu_S < 0 the bound saturates at -1/mu_S times the prefactor, so a
    # tolerance above that asymptote can never be reached and the threshold is
    # correctly infinite rather than a large finite number.
    # An isotropic metric and a plant whose symmetric part is diagonal: with
    # m_alpha = -1 the Jacobian's symmetric off-diagonal is exactly zero, so
    # mu_2 is the larger of the two diagonal entries and is negative.
    stable = Plant(
        z_alpha=-5.0, z_delta=-0.1, m_alpha=-1.0, m_alpha3=0.0, m_q=-5.0,
        m_delta=-0.1, u_max_rad=1.0,
    )
    loop = LoopModel(
        plant=stable,
        metric=DivergenceMetric(
            alpha_tolerance_rad=0.01, rate_tolerance_rad_s=0.01
        ),
    )
    constants = CertificateModel.build(loop, 10)
    assert constants.mu_s < 0.0
    assert math.isinf(certified_overrun_s(constants, 1e-9, 10, 1.0))


def test_certify_miss_dismisses_a_tiny_overrun(constants):
    certificate = certify_miss(constants, _event(overrun=1e-7, step=1e-4), 1.0)
    assert certificate.verdict == DISMISSED
    assert certificate.dismissed is True
    assert certificate.margin > 0.0


def test_certify_miss_leaves_a_large_overrun_undecided(constants):
    certificate = certify_miss(constants, _event(overrun=0.03, step=0.3), 1.0)
    assert certificate.verdict == UNDECIDED
    assert certificate.dismissed is False
    assert certificate.margin < 0.0


def test_certify_miss_margin_is_tolerance_minus_bound(constants):
    certificate = certify_miss(constants, _event(), 1.0)
    assert certificate.margin == pytest.approx(
        1.0 - certificate.bound, rel=1e-15, abs=0.0
    )


def test_certify_miss_rejects_a_non_positive_tolerance(constants):
    with pytest.raises(ValueError, match="tolerance must be positive"):
        certify_miss(constants, _event(), -1.0)


def test_certify_run_covers_every_miss(long_run):
    certificate = certify_run(long_run, 1.0)
    assert len(certificate.certificates) == len(miss_events(long_run))


def test_certify_run_partitions_into_dismissed_and_undecided(long_run):
    certificate = certify_run(long_run, 1.0)
    assert len(certificate.dismissed) + len(certificate.undecided) == len(
        certificate.certificates
    )


def test_certify_run_aggregate_is_the_sum_of_the_bounds(long_run):
    certificate = certify_run(long_run, 1.0)
    assert certificate.aggregate_bound == pytest.approx(
        sum(item.bound for item in certificate.certificates), rel=1e-12, abs=1e-18
    )


def test_certify_run_dismissal_rate_is_one_with_no_misses():
    log = generate_run(seed=7, profile="nominal", amplitude_deg=3.0, steps=20)
    clean = RunLog(
        model=log.model,
        command_rad=log.command_rad,
        durations_s=np.full(log.steps, 0.004),
        states=log.states,
        computed_rad=log.computed_rad,
        saturated=log.saturated,
        host_time_s=log.host_time_s,
        device_time_s=log.device_time_s,
    )
    certificate = certify_run(clean, 1.0)
    assert certificate.certificates == ()
    assert certificate.dismissal_rate == pytest.approx(1.0, rel=0.0, abs=0.0)


def test_certify_run_is_admissible_on_a_declared_campaign(long_run):
    assert certify_run(long_run, 1.0).admissible is True


def test_certify_run_refuses_a_saturating_run():
    log = generate_run(seed=7, profile="loaded", amplitude_deg=14.0, steps=200)
    certificate = certify_run(log, 1.0)
    assert certificate.refusals
    assert any("saturat" in reason for reason in certificate.refusals)
    assert certificate.admissible is False


def test_certify_run_refuses_an_overloaded_run(long_run):
    overloaded = RunLog(
        model=long_run.model,
        command_rad=long_run.command_rad,
        durations_s=np.asarray(long_run.durations_s) * 2.0,
        states=long_run.states,
        computed_rad=long_run.computed_rad,
        saturated=long_run.saturated,
        host_time_s=long_run.host_time_s,
        device_time_s=long_run.device_time_s,
    )
    certificate = certify_run(overloaded, 1.0)
    assert any("utilisation" in reason for reason in certificate.refusals)


def test_require_admissible_passes_an_admissible_certificate(long_run):
    require_admissible(certify_run(long_run, 1.0))


def test_require_admissible_raises_on_a_refusal():
    log = generate_run(seed=7, profile="loaded", amplitude_deg=14.0, steps=200)
    with pytest.raises(ValueError, match="not admissible"):
        require_admissible(certify_run(log, 1.0))


def test_certified_overrun_on_a_run_is_finite_when_a_miss_has_an_input_step(long_run):
    certificate = certify_run(long_run, 1.0)
    assert math.isfinite(certificate.certified_overrun_s)
    assert certificate.certified_overrun_s > 0.0


def test_a_tighter_tolerance_dismisses_no_more_misses(long_run):
    loose = certify_run(long_run, 2.0)
    tight = certify_run(long_run, 0.5)
    assert len(tight.dismissed) <= len(loose.dismissed)

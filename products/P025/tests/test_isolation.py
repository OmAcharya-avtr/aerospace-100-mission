"""GLR isolation bank: hand-checked statistics, signatures and structure."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.stats import chi2

from fdiscope.analytic import normalised_bias_signature
from fdiscope.faults import FaultSpec, FaultType
from fdiscope.isolation import (
    SignatureBank,
    build_signature_bank,
    fault_signature,
    glr_statistics,
    isolate_window,
)
from fdiscope.plant import PlantConfig, loop_matrices
from fdiscope.simulate import LoopConfig, build_filter

TINY_CONFIG = LoopConfig(n_steps=900, noise=False)


def toy_bank() -> SignatureBank:
    # Two orthonormal signatures over a 2-sample window of a 2-channel
    # residual: e0 (angle channel at the first sample) and e3 (rate channel at
    # the second sample).
    matrix = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    return SignatureBank(
        window=2,
        dim=2,
        faults=(FaultType.SENSOR_BIAS, FaultType.SENSOR_DRIFT),
        matrix=matrix,
    )


class TestGlrStatistics:
    def test_known_answer(self):
        # r flattened = (3, 0, 0, 4).  phi_0 = e0 -> (3)^2 = 9;
        # phi_1 = e3 -> (4)^2 = 16.
        stats = glr_statistics([[3.0, 0.0], [0.0, 4.0]], toy_bank())
        assert np.allclose(stats, [9.0, 16.0])

    def test_statistics_are_non_negative(self):
        r = np.random.default_rng(0).standard_normal((2, 2))
        assert np.all(glr_statistics(r, toy_bank()) >= 0.0)

    def test_statistic_is_bounded_by_the_squared_residual_norm(self):
        # (phi . r)^2 <= |phi|^2 |r|^2 = |r|^2 for a unit signature.
        rng = np.random.default_rng(1)
        bank = toy_bank()
        for _ in range(20):
            r = rng.standard_normal((2, 2))
            assert np.all(glr_statistics(r, bank) <= np.sum(r * r) + 1e-12)

    def test_accepts_a_flat_window(self):
        assert np.allclose(
            glr_statistics([3.0, 0.0, 0.0, 4.0], toy_bank()),
            glr_statistics([[3.0, 0.0], [0.0, 4.0]], toy_bank()),
        )

    def test_rejects_wrong_window_size(self):
        with pytest.raises(ValueError, match="expected 4"):
            glr_statistics(np.zeros((3, 2)), toy_bank())

    def test_rejects_non_finite_window(self):
        with pytest.raises(ValueError, match="finite"):
            glr_statistics([np.nan, 0.0, 0.0, 0.0], toy_bank())

    def test_sign_invariance(self):
        # The GLR squares the projection, so flipping the residual sign
        # changes nothing.  This is why an anti-aligned signature pair is as
        # inseparable as an aligned one.
        r = np.random.default_rng(2).standard_normal((2, 2))
        assert np.allclose(glr_statistics(r, toy_bank()), glr_statistics(-r, toy_bank()))


class TestIsolateWindow:
    def test_picks_the_larger_statistic(self):
        result = isolate_window([[0.0, 0.0], [0.0, 40.0]], toy_bank(), alpha=1e-3)
        assert result.fault is FaultType.SENSOR_DRIFT
        assert np.isclose(result.statistic, 1600.0)

    def test_threshold_is_bonferroni_corrected(self):
        # Two hypotheses at alpha = 1e-3 -> chi2.isf(5e-4, 1)
        result = isolate_window(np.zeros((2, 2)), toy_bank(), alpha=1e-3)
        assert np.isclose(result.threshold, float(chi2.isf(5e-4, 1)))

    def test_declares_no_fault_below_the_threshold(self):
        result = isolate_window([[0.1, 0.0], [0.0, 0.1]], toy_bank(), alpha=1e-3)
        assert result.fault is FaultType.NONE
        assert np.isnan(result.confidence)

    def test_posterior_sums_to_one_and_peaks_on_the_winner(self):
        result = isolate_window([[0.0, 0.0], [0.0, 40.0]], toy_bank(), alpha=1e-3)
        assert np.isclose(result.posterior.sum(), 1.0)
        assert np.argmax(result.posterior) == 1
        assert np.isclose(result.confidence, result.posterior[1])

    def test_posterior_is_numerically_stable_for_huge_statistics(self):
        result = isolate_window([[0.0, 0.0], [0.0, 1e4]], toy_bank(), alpha=1e-3)
        assert np.all(np.isfinite(result.posterior))
        assert np.isclose(result.posterior.sum(), 1.0)

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.1])
    def test_rejects_bad_alpha(self, bad):
        with pytest.raises(ValueError, match="alpha"):
            isolate_window(np.zeros((2, 2)), toy_bank(), alpha=bad)


class TestSignatureBank:
    def test_rejects_a_shape_mismatch(self):
        with pytest.raises(ValueError, match="does not match"):
            SignatureBank(window=2, dim=2, faults=(FaultType.SENSOR_BIAS,), matrix=np.zeros((1, 3)))

    def test_gram_of_orthonormal_signatures_is_the_identity(self):
        assert np.allclose(toy_bank().gram(), np.eye(2))

    def test_gram_diagonal_is_one_for_unit_signatures(self):
        bank = build_signature_bank(
            TINY_CONFIG,
            {
                FaultType.SENSOR_BIAS: FaultSpec(FaultType.SENSOR_BIAS, 0, 1e-3, 1),
                FaultType.SENSOR_STUCK: FaultSpec(FaultType.SENSOR_STUCK, 0, 0.0, 0),
            },
            window=40,
            onset_steps=[400, 500],
        )
        assert np.allclose(np.diag(bank.gram()), 1.0)

    def test_rejects_the_null_hypothesis(self):
        with pytest.raises(ValueError, match="FaultType.NONE has no signature"):
            build_signature_bank(TINY_CONFIG, {FaultType.NONE: FaultSpec()}, 10, [400])

    def test_rejects_empty_specs(self):
        with pytest.raises(ValueError, match="must not be empty"):
            build_signature_bank(TINY_CONFIG, {}, 10, [400])


class TestFaultSignature:
    def test_signature_is_a_unit_vector(self):
        phi = fault_signature(
            TINY_CONFIG, FaultSpec(FaultType.SENSOR_BIAS, 0, 1e-3, 1), 30, [400]
        )
        assert phi.shape == (60,)
        assert np.isclose(np.linalg.norm(phi), 1.0)

    def test_additive_sensor_signature_is_magnitude_independent(self):
        # For an additive fault the residual response is linear in the fault
        # size, so the normalised signature does not depend on it.
        small = fault_signature(
            TINY_CONFIG, FaultSpec(FaultType.SENSOR_BIAS, 0, 1e-4, 1), 40, [400]
        )
        large = fault_signature(
            TINY_CONFIG, FaultSpec(FaultType.SENSOR_BIAS, 0, 1e-2, 1), 40, [400]
        )
        assert np.allclose(small, large, atol=1e-9)

    def test_actuator_signature_does_depend_on_the_onset_phase(self):
        # The documented weakness of the classical isolator: a multiplicative
        # fault's signature depends on the commanded torque at onset.
        early = fault_signature(
            TINY_CONFIG, FaultSpec(FaultType.ACTUATOR_LOSS_OF_EFFECT, 0, 0.6, 0), 60, [400]
        )
        late = fault_signature(
            TINY_CONFIG, FaultSpec(FaultType.ACTUATOR_LOSS_OF_EFFECT, 0, 0.6, 0), 60, [550]
        )
        assert abs(float(early @ late)) < 0.999

    def test_long_window_bias_signature_matches_the_closed_form(self):
        # The tail of a long gyro-bias signature must approach the analytic
        # steady-state residual direction of fdiscope.analytic.
        kf = build_filter(loop_matrices(PlantConfig()))
        sigma = float(np.sqrt(PlantConfig().gyro_var_rad2_s2))
        direction, _ = normalised_bias_signature(kf, [0.0, 4.0 * sigma])
        phi = fault_signature(
            LoopConfig(n_steps=1600, noise=False),
            FaultSpec(FaultType.SENSOR_BIAS, 0, 4.0 * sigma, 1),
            800,
            [400],
        ).reshape(-1, 2)
        tail = phi[-1] / np.linalg.norm(phi[-1])
        assert np.allclose(tail, direction, atol=1e-6)

    def test_rejects_a_window_that_runs_off_the_end(self):
        with pytest.raises(ValueError, match="exceeds n_steps"):
            fault_signature(TINY_CONFIG, FaultSpec(FaultType.SENSOR_STUCK, 0, 0.0, 0), 200, [880])

    def test_rejects_a_bad_window(self):
        with pytest.raises(ValueError, match="window must be >= 1"):
            fault_signature(TINY_CONFIG, FaultSpec(FaultType.SENSOR_STUCK, 0, 0.0, 0), 0, [400])

    def test_rejects_empty_onsets(self):
        with pytest.raises(ValueError, match="onset_steps must not be empty"):
            fault_signature(TINY_CONFIG, FaultSpec(FaultType.SENSOR_STUCK, 0, 0.0, 0), 10, [])


class TestProperties:
    @settings(max_examples=30, deadline=None)
    @given(scale=st.floats(0.1, 50.0, allow_nan=False))
    def test_statistics_scale_quadratically_with_the_residual(self, scale):
        r = np.array([[1.0, -2.0], [0.5, 3.0]])
        base = glr_statistics(r, toy_bank())
        scaled = glr_statistics(scale * r, toy_bank())
        assert np.allclose(scaled, scale * scale * base, rtol=1e-10)

    @settings(max_examples=30, deadline=None)
    @given(
        a=st.floats(-5.0, 5.0, allow_nan=False),
        b=st.floats(-5.0, 5.0, allow_nan=False),
    )
    def test_isolation_is_invariant_to_the_overall_residual_sign(self, a, b):
        window = np.array([[a, 0.0], [0.0, b]])
        first = isolate_window(window, toy_bank(), alpha=0.5)
        second = isolate_window(-window, toy_bank(), alpha=0.5)
        assert first.fault is second.fault
        assert np.isclose(first.statistic, second.statistic)

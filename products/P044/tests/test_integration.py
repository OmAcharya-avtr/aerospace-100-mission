"""Integration and regression tests.

The integration test walks the whole pipeline the product exists for: pick a
link, average the scintillation over an aperture, correlate several
apertures, combine, measure the outage and the diversity order, estimate the
channel badly and compare the combiners. If any module's convention drifts
away from the others, this is where it shows.

The regression tests pin numbers that a seeded run must reproduce. They are
deliberately tight: they exist to catch a silent change in a formula or in
the RNG draw order, not to assert physics. Every expected value was produced
by running this code in this environment, and each is accompanied by the
tolerance and the reason it is what it is.
"""

from __future__ import annotations

import numpy as np
import pytest

from aperturediv import (
    aperture_averaging_factor,
    bpsk_ber_lognormal_sample,
    branch_mean_snr,
    combined_gain,
    correlation_matrix,
    diversity_order,
    effective_scintillation_index,
    equal_area_diameter,
    fresnel_scale,
    gamma_gamma_params_from_rytov,
    gamma_gamma_scintillation_index,
    irradiance_correlation_matrix,
    outage_probability,
    sample_correlated_lognormal,
)
from aperturediv.correlation import equispaced_positions
from aperturediv.datasets import make_combiner_dataset, split_dataset
from aperturediv.learned import (
    LearnedCombiner,
    egc_weights,
    fit_shrinkage_exponent,
    mrc_estimated_weights,
    mrc_true_weights,
    penalty_db,
)


class TestEndToEndLinkDesign:
    def test_full_pipeline(self):
        # 1.55 um over 2 km, total glass 0.2 m, point scintillation index 0.6.
        rho_c = fresnel_scale(1.55e-6, 2000.0)
        assert rho_c == pytest.approx(0.0556776436, rel=1e-9)

        # One aperture of 0.2 m averages hard; four of equal total area do not.
        si_one = effective_scintillation_index(0.6, 0.2, rho_c)
        d_each = equal_area_diameter(0.2, 4)
        si_each = effective_scintillation_index(0.6, d_each, rho_c)
        assert si_each > si_one
        assert si_one < 0.6
        assert aperture_averaging_factor(0.2, rho_c) < aperture_averaging_factor(d_each, rho_c)

        # Four apertures at twice their own diameter: correlation falls with pitch.
        pos = equispaced_positions(4, 2.0 * d_each)
        r_log = correlation_matrix(pos, rho_c)
        r_irr = irradiance_correlation_matrix(pos, rho_c, si_each)
        assert r_log[0, 1] >= r_irr[0, 1] - 1e-12
        assert r_log[0, 3] < r_log[0, 1]

        # Combine, with the total power held fixed across aperture counts.
        rng = np.random.default_rng(44044)
        irr = sample_correlated_lognormal(300_000, si_each, r_log, rng)
        snr_db = np.arange(0.0, 45.0, 0.5)
        orders = {}
        for n_ap in (1, 2, 4):
            gain = combined_gain(irr[:, :n_ap], "mrc")
            branch_db = 10.0 * np.log10(branch_mean_snr(10.0 ** 2.0, n_ap, "fixed_total"))
            assert branch_db <= 20.0 + 1e-9
            p = outage_probability(gain, snr_db, 5.0)
            assert np.all(np.diff(p) <= 0.0)
            orders[n_ap] = diversity_order(snr_db, p, window=(1e-4, 1e-2)).order
        assert orders[4] > orders[2] > orders[1]

        # Estimate the channel badly and compare combiners on held-out rows.
        data = make_combiner_dataset(
            20_000, n_apertures=4, si=si_each, aperture_spacing_m=2.0 * d_each,
            correlation_scale_m=rho_c, sigma_e_range=(0.0, 2.0), seed=44044,
        )
        train, val, test = split_dataset(data, n_train=12_000, n_validation=4_000)
        p_star = fit_shrinkage_exponent(val)
        model = LearnedCombiner(max_iter=60, random_state=44044).fit(train)
        h = test.amplitude_true
        pen_true = penalty_db(mrc_true_weights(test.irradiance_true), h).mean()
        pen_naive = penalty_db(mrc_estimated_weights(test.irradiance_estimated), h).mean()
        pen_egc = penalty_db(egc_weights(len(test), 4), h).mean()
        pen_learned = penalty_db(model.combine(test.irradiance_estimated, test.sigma_e)[0],
                                 h).mean()
        assert pen_true == pytest.approx(0.0, abs=1e-12)
        assert pen_learned >= pen_true
        assert pen_learned < pen_naive
        assert 0.0 <= p_star <= 1.0
        assert pen_egc > 0.0

    def test_gamma_gamma_and_lognormal_diverge_only_in_the_tail(self):
        # At matched scintillation index the two models agree to a couple of
        # per cent while the outage is large, and diverge monotonically into
        # the deep-fade tail, where gamma-gamma is the pessimistic one. Both
        # halves are checked, because only the second is usually mentioned and
        # the first is what makes a matched-si comparison legitimate at all.
        alpha, beta = gamma_gamma_params_from_rytov(0.3)
        si = gamma_gamma_scintillation_index(alpha, beta)
        from aperturediv.correlation import sample_correlated_gamma_gamma

        gg = sample_correlated_gamma_gamma(
            200_000, alpha, beta, np.eye(1), np.random.default_rng(1)
        )
        ln = sample_correlated_lognormal(200_000, si, np.eye(1), np.random.default_rng(1))
        ratios = []
        for snr_db in (2.0, 4.0, 6.0, 8.0, 10.0, 12.0):
            p_gg = float(outage_probability(combined_gain(gg, "mrc"), snr_db, 5.0))
            p_ln = float(outage_probability(combined_gain(ln, "mrc"), snr_db, 5.0))
            ratios.append(p_gg / p_ln)
        # High-outage end: the two models agree within 2 %.
        assert ratios[0] == pytest.approx(1.0, abs=0.02)
        assert ratios[2] == pytest.approx(1.0, abs=0.02)
        # Deep tail: gamma-gamma is more pessimistic, by more than 3x at 1e-3.
        assert ratios[-1] > 3.0
        # And the divergence is monotone once it starts.
        assert all(y > x for x, y in zip(ratios[3:], ratios[4:], strict=False))


class TestRegression:
    """Pinned values from seeded runs in this environment.

    A failure here means a formula, a draw order or a default changed. It
    does not by itself mean the new behaviour is wrong, but it must be
    explained before the pin is moved.
    """

    def test_aperture_averaging_pin(self):
        # Deterministic quadrature, so this is exact to the solver tolerance.
        assert aperture_averaging_factor(1.0, 1.0) == pytest.approx(0.79417571, rel=1e-7)
        assert aperture_averaging_factor(2.0, 1.0) == pytest.approx(0.47622239, rel=1e-7)
        assert aperture_averaging_factor(20.0, 1.0) == pytest.approx(0.00943616, rel=1e-6)

    def test_gamma_gamma_params_pin(self):
        alpha, beta = gamma_gamma_params_from_rytov(1.0)
        assert alpha == pytest.approx(4.393859025, rel=1e-9)
        assert beta == pytest.approx(2.563631980, rel=1e-9)
        assert gamma_gamma_scintillation_index(alpha, beta) == pytest.approx(
            0.706438496, rel=1e-9
        )

    def test_irradiance_correlation_pin(self):
        pos = equispaced_positions(4, 0.05)
        r = irradiance_correlation_matrix(pos, 0.10, 0.6)
        # Adjacent log correlation exp(-0.25) = 0.7788008; at si = 0.6 the
        # irradiance correlation drops to 0.7290.
        # Adjacent log correlation exp(-(0.05/0.10)^2) = exp(-0.25) = 0.7788008;
        # at si = 0.6 (s^2 = ln 1.6) the irradiance correlation drops to 0.73669.
        assert r[0, 1] == pytest.approx(0.7366863, rel=1e-6)
        # Outermost pair: log exp(-(0.15/0.10)^2) = exp(-2.25) = 0.1053992,
        # irradiance 0.0846426.
        assert r[0, 3] == pytest.approx(0.0846426, rel=1e-6)

    def test_sample_ber_pin(self):
        # Seeded sample BER; this pins both the formula and the RNG draw order.
        res = bpsk_ber_lognormal_sample(10.0, 0.3, 2_000_000, 44044, chunk_size=2_000_000)
        assert res.n_errors == 1180
        assert res.ber == pytest.approx(5.90e-4, rel=1e-9)

    def test_outage_pin(self):
        rng = np.random.default_rng(44044)
        irr = sample_correlated_lognormal(200_000, 0.9, np.eye(4), rng)
        gain = combined_gain(irr, "mrc")
        # Fully seeded, so this is an exact count, not an estimate:
        # 228 of 200000 realisations below the threshold at 5 dB branch SNR.
        assert float(outage_probability(gain, 5.0, 5.0)) == pytest.approx(
            228 / 200_000, rel=1e-12
        )
        # At 8 dB and above the outage is already below the 1/200000 floor, so
        # the function must return exactly 0 rather than something small.
        assert float(outage_probability(gain, 8.0, 5.0)) == 0.0

    def test_diversity_order_pin(self):
        rng = np.random.default_rng(44044)
        irr = sample_correlated_lognormal(400_000, 0.9, np.eye(4), rng)
        snr_db = np.arange(0.0, 50.0, 0.25)
        res = diversity_order(snr_db, outage_probability(combined_gain(irr, "mrc"), snr_db, 5.0),
                              window=(1e-4, 1e-2))
        assert res.order == pytest.approx(7.630, abs=0.02)
        assert res.n_points == 10
        assert res.residual_rms < 0.05

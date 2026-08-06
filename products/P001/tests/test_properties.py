"""Property-based tests (Hypothesis) for algebraic identities in BeamTwin."""

from __future__ import annotations

import math

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from beamtwin.budget import (
    LinkParams,
    beam_radius,
    compute_budget,
    db_from_fraction,
    dbm_from_watts,
    gaussian_divergence_half_angle,
    geometric_capture_fraction,
    pointing_loss_fraction,
    rayleigh_range,
    watts_from_dbm,
)
from beamtwin.channel import mean_pointing_loss_fraction, rytov_variance_plane_wave
from beamtwin.stats import analytic_fade_probability_lognormal

SETTINGS = settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

wavelengths = st.floats(min_value=400e-9, max_value=10e-6, allow_nan=False)
waists = st.floats(min_value=1e-3, max_value=0.5, allow_nan=False)
ranges = st.floats(min_value=1.0, max_value=100_000.0, allow_nan=False)
fractions = st.floats(min_value=1e-6, max_value=1.0, allow_nan=False)
powers_dbm = st.floats(min_value=-60.0, max_value=40.0, allow_nan=False)


class TestConversionProperties:
    @given(p=powers_dbm)
    @SETTINGS
    def test_dbm_watt_roundtrip(self, p):
        assert dbm_from_watts(watts_from_dbm(p)) == pytest.approx(p, abs=1e-9)

    @given(f=fractions)
    @SETTINGS
    def test_db_from_fraction_non_negative(self, f):
        assert db_from_fraction(f) >= -1e-12

    @given(a=fractions, b=fractions)
    @SETTINGS
    def test_db_of_product_is_sum_of_dbs(self, a, b):
        # -10log10(ab) = -10log10(a) + -10log10(b): losses add in dB.
        assume(a * b > 1e-300)
        assert db_from_fraction(a * b) == pytest.approx(
            db_from_fraction(a) + db_from_fraction(b), abs=1e-9
        )


class TestGaussianBeamProperties:
    @given(lam=wavelengths, w0=waists, z=ranges)
    @SETTINGS
    def test_beam_radius_at_least_waist(self, lam, w0, z):
        assert beam_radius(lam, w0, z) >= w0 - 1e-15

    @given(lam=wavelengths, w0=waists, z=ranges)
    @SETTINGS
    def test_beam_radius_monotone_in_range(self, lam, w0, z):
        assert beam_radius(lam, w0, 2 * z) >= beam_radius(lam, w0, z)

    @given(lam=wavelengths, w0=waists)
    @SETTINGS
    def test_divergence_times_rayleigh_equals_waist(self, lam, w0):
        # theta * z_R = (lambda/(pi w0)) * (pi w0^2/lambda) = w0.
        theta = gaussian_divergence_half_angle(lam, w0)
        z_r = rayleigh_range(lam, w0)
        assert theta * z_r == pytest.approx(w0, rel=1e-9)

    @given(lam=wavelengths, w0=waists, z=ranges)
    @SETTINGS
    def test_beam_radius_formula_identity(self, lam, w0, z):
        z_r = rayleigh_range(lam, w0)
        assert beam_radius(lam, w0, z) == pytest.approx(
            w0 * math.sqrt(1 + (z / z_r) ** 2), rel=1e-12
        )


class TestCaptureProperties:
    @given(
        w=st.floats(min_value=1e-3, max_value=10.0),
        a=st.floats(min_value=1e-4, max_value=5.0),
    )
    @SETTINGS
    def test_capture_in_open_unit_interval(self, w, a):
        eta = geometric_capture_fraction(w, a)
        assert 0.0 < eta <= 1.0

    @given(
        w=st.floats(min_value=1e-2, max_value=5.0),
        a=st.floats(min_value=1e-3, max_value=1.0),
    )
    @SETTINGS
    def test_capture_monotone_in_aperture(self, w, a):
        assert geometric_capture_fraction(w, 2 * a) >= geometric_capture_fraction(w, a)

    @given(
        w=st.floats(min_value=1e-2, max_value=5.0),
        d=st.floats(min_value=0.0, max_value=5.0),
    )
    @SETTINGS
    def test_pointing_loss_in_unit_interval(self, w, d):
        assert 0.0 <= pointing_loss_fraction(d, w) <= 1.0

    @given(
        w=st.floats(min_value=1e-2, max_value=5.0),
        d=st.floats(min_value=1e-4, max_value=2.0),
    )
    @SETTINGS
    def test_pointing_loss_monotone_in_offset(self, w, d):
        assert pointing_loss_fraction(2 * d, w) <= pointing_loss_fraction(d, w)

    @given(
        w=st.floats(min_value=1e-2, max_value=5.0),
        s=st.floats(min_value=0.0, max_value=5.0),
    )
    @SETTINGS
    def test_mean_pointing_loss_in_unit_interval(self, w, s):
        assert 0.0 < mean_pointing_loss_fraction(s, w) <= 1.0


class TestChannelProperties:
    @given(
        cn2=st.floats(min_value=0.0, max_value=1e-12),
        lam=wavelengths,
        z=ranges,
    )
    @SETTINGS
    def test_rytov_non_negative(self, cn2, lam, z):
        assert rytov_variance_plane_wave(cn2, lam, z) >= 0.0

    @given(
        cn2=st.floats(min_value=1e-18, max_value=1e-13),
        lam=wavelengths,
        z=st.floats(min_value=10.0, max_value=50_000.0),
    )
    @SETTINGS
    def test_rytov_linear_in_cn2(self, cn2, lam, z):
        a = rytov_variance_plane_wave(cn2, lam, z)
        b = rytov_variance_plane_wave(3 * cn2, lam, z)
        assert b == pytest.approx(3 * a, rel=1e-9)


class TestFadeProbabilityProperties:
    @given(
        m=st.floats(min_value=-30.0, max_value=40.0),
        s=st.floats(min_value=1e-3, max_value=2.0),
    )
    @SETTINGS
    def test_analytic_probability_in_unit_interval(self, m, s):
        assert 0.0 <= analytic_fade_probability_lognormal(m, s) <= 1.0

    @given(
        m=st.floats(min_value=0.0, max_value=30.0),
        s=st.floats(min_value=1e-2, max_value=1.5),
    )
    @SETTINGS
    def test_analytic_monotone_decreasing_in_margin(self, m, s):
        assert analytic_fade_probability_lognormal(m + 1.0, s) <= (
            analytic_fade_probability_lognormal(m, s) + 1e-15
        )


class TestBudgetProperties:
    @given(
        rng_m=st.floats(min_value=100.0, max_value=50_000.0),
        att=st.floats(min_value=0.0, max_value=20.0),
        tx=st.floats(min_value=-10.0, max_value=40.0),
    )
    @SETTINGS
    def test_budget_terms_are_consistent(self, rng_m, att, tx):
        p = LinkParams(range_m=rng_m, attenuation_db_per_km=att, tx_power_dbm=tx)
        b = compute_budget(p)
        total = (
            tx
            - b.tx_optics_loss_db
            - b.geometric_loss_db
            - b.pointing_loss_db
            - b.atmospheric_loss_db
            - b.rx_optics_loss_db
        )
        assert b.received_power_dbm == pytest.approx(total, abs=1e-9)
        assert b.margin_negative == (b.margin_db < 0.0)

    @given(
        rng_m=st.floats(min_value=100.0, max_value=30_000.0),
        att=st.floats(min_value=0.0, max_value=10.0),
    )
    @SETTINGS
    def test_atmospheric_loss_is_linear_in_range(self, rng_m, att):
        b = compute_budget(LinkParams(range_m=rng_m, attenuation_db_per_km=att))
        assert b.atmospheric_loss_db == pytest.approx(att * rng_m / 1000.0, rel=1e-12)

    @given(
        rng_m=st.floats(min_value=500.0, max_value=30_000.0),
        extra_db=st.floats(min_value=0.1, max_value=20.0),
    )
    @SETTINGS
    def test_margin_shifts_one_for_one_with_tx_power(self, rng_m, extra_db):
        a = compute_budget(LinkParams(range_m=rng_m, tx_power_dbm=10.0)).margin_db
        b = compute_budget(LinkParams(range_m=rng_m, tx_power_dbm=10.0 + extra_db)).margin_db
        assert b - a == pytest.approx(extra_db, abs=1e-9)

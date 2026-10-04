"""Uncertainty-analysis helpers."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from constellink.constellation import GroundStation, walker_delta
from constellink.synthdata import default_optical_terminal, default_rf_terminal
from constellink.uncertainty import (
    grid_step_convergence,
    monte_carlo_capacity,
    perturb_mean_anomaly,
    window_edge_sensitivity,
)

EPOCH = datetime(2026, 4, 1, tzinfo=UTC)
STATION = GroundStation("AWARUA", -46.53, 168.38, 0.01, 10.0)


@pytest.fixture(scope="module")
def const():
    return walker_delta(24, 4, 1, 53.0, 550.0, EPOCH, max_epoch_age_days=2.0)


def test_perturb_mean_anomaly_changes_only_the_mean_anomaly(const):
    sat = const.satellites[0]
    pert = perturb_mean_anomaly(sat, 0.01)
    assert pert.satrec.mo == pytest.approx((sat.satrec.mo + 0.01) % (2 * np.pi),
                                           rel=1e-12)
    for field in ("inclo", "nodeo", "ecco", "argpo", "no_kozai", "bstar"):
        assert getattr(pert.satrec, field) == pytest.approx(
            getattr(sat.satrec, field), rel=1e-12, abs=1e-15)


def test_perturb_mean_anomaly_moves_the_position_along_track(const):
    sat = const.satellites[0]
    # dM = ds / a, so a 10 km along-track error should move the satellite by
    # roughly 10 km. Checked loosely because SGP4 is not a pure two-body
    # propagator.
    a_km = float(sat.satrec.a) * 6378.135
    pert = perturb_mean_anomaly(sat, 10.0 / a_km)
    r0, _ = sat.propagate([EPOCH])
    r1, _ = pert.propagate([EPOCH])
    assert float(np.linalg.norm(r1[0] - r0[0])) == pytest.approx(10.0, rel=0.05)


def test_window_edge_sensitivity_scales_with_the_error(const):
    small = window_edge_sensitivity(const, "W01-04", STATION, EPOCH,
                                    EPOCH + timedelta(hours=6), 30.0, 2.0)
    large = window_edge_sensitivity(const, "W01-04", STATION, EPOCH,
                                    EPOCH + timedelta(hours=6), 30.0, 20.0)
    assert small.n_windows_nominal > 0
    assert small.n_windows_nominal == large.n_windows_nominal
    assert large.max_abs_edge_shift_s > small.max_abs_edge_shift_s
    # Roughly linear in the applied error (first order), within a factor of 2.
    ratio = large.max_abs_edge_shift_s / small.max_abs_edge_shift_s
    assert 5.0 < ratio < 20.0


def test_window_edge_sensitivity_reports_the_mean_anomaly_perturbation(const):
    es = window_edge_sensitivity(const, "W01-04", STATION, EPOCH,
                                 EPOCH + timedelta(hours=6), 30.0, 7.0)
    a_km = float(const.satellite("W01-04").satrec.a) * 6378.135
    assert es.delta_m_rad == pytest.approx(7.0 / a_km, rel=1e-12)
    assert es.along_track_error_km == pytest.approx(7.0)
    assert es.d_t_open_s.shape == es.d_t_close_s.shape == es.d_duration_s.shape


def test_window_edge_sensitivity_rejects_bad_error(const):
    with pytest.raises(ValueError):
        window_edge_sensitivity(const, "W01-04", STATION, EPOCH,
                                EPOCH + timedelta(hours=1), 60.0, 0.0)


def test_edge_sensitivity_nan_when_nothing_matched():
    # A station nothing passes over inside a short horizon: the result is
    # reported as NaN, not as zero, so "no data" cannot be read as "no shift".
    const = walker_delta(4, 2, 1, 0.0, 550.0, EPOCH, max_epoch_age_days=2.0)
    polar = GroundStation("POLE", 89.0, 0.0, 0.0, 10.0)
    es = window_edge_sensitivity(const, "W00-00", polar, EPOCH,
                                 EPOCH + timedelta(minutes=30), 60.0, 1.0)
    assert es.n_windows_nominal == 0
    assert np.isnan(es.max_abs_edge_shift_s)


def test_grid_step_convergence_shape_and_table(const):
    gc = grid_step_convergence(const, "W00-00", "W02-02", EPOCH,
                               EPOCH + timedelta(hours=2), [30.0, 60.0, 120.0])
    assert gc.step_s.shape == gc.n_windows.shape == gc.total_duration_s.shape
    assert gc.step_s.shape == (3,)
    text = gc.format_table()
    assert "step [s]" in text
    assert text.count("\n") == 4


def test_grid_step_convergence_is_stable_for_this_link(const):
    # The bisection decouples edge accuracy from the grid step, so the total
    # duration must agree across steps to much better than the step itself.
    gc = grid_step_convergence(const, "W00-00", "W02-02", EPOCH,
                               EPOCH + timedelta(hours=2),
                               [10.0, 30.0, 60.0, 120.0])
    assert len(set(gc.n_windows.tolist())) == 1
    spread = float(gc.total_duration_s.max() - gc.total_duration_s.min())
    assert spread < 0.2


@pytest.mark.parametrize("steps", [[], [0.0], [60.0, -1.0]])
def test_grid_step_convergence_validation(const, steps):
    with pytest.raises(ValueError):
        grid_step_convergence(const, "W00-00", "W02-02", EPOCH,
                              EPOCH + timedelta(hours=1), steps)


def test_monte_carlo_capacity_rf_summary():
    d = monte_carlo_capacity(1500.0, default_rf_terminal(),
                             {"tx_gain_dbi": 0.5}, n_draws=500, seed=3)
    assert d.samples_bps.shape == (500,)
    assert d.n_draws == 500
    assert d.seed == 3
    assert d.std_bps > 0.0
    assert d.sem_bps == pytest.approx(d.std_bps / np.sqrt(500), rel=1e-12)
    assert d.percentiles[5.0] < d.percentiles[50.0] < d.percentiles[95.0]
    assert "Monte Carlo capacity" in d.format_summary()


def test_monte_carlo_zero_sigma_is_deterministic():
    d = monte_carlo_capacity(1500.0, default_rf_terminal(),
                             {"tx_gain_dbi": 0.0}, n_draws=200, seed=1)
    assert d.std_bps == pytest.approx(0.0, abs=1e-6)


def test_monte_carlo_spread_matches_the_analytic_quadrature():
    # For a pure dB-domain sum the achievable rate is log-normal, so the
    # p95/p5 ratio in dB must match 2 * 1.6449 * sqrt(sum of sigma^2).
    sigmas = {"tx_power_dbw": 0.4, "tx_gain_dbi": 0.6,
              "rx_g_over_t_db_per_k": 0.8}
    d = monte_carlo_capacity(1500.0, default_rf_terminal(), sigmas,
                             n_draws=8000, seed=17)
    measured_db = 10.0 * np.log10(d.percentiles[95.0] / d.percentiles[5.0])
    expected_db = 2.0 * 1.6449 * np.sqrt(sum(v ** 2 for v in sigmas.values()))
    assert measured_db == pytest.approx(expected_db, rel=0.05)


def test_monte_carlo_capacity_optical():
    d = monte_carlo_capacity(1500.0, default_optical_terminal(),
                             {"pointing_error_rad": 1e-6}, n_draws=300, seed=5)
    assert np.all(d.samples_bps > 0.0)
    assert d.mean_bps > 0.0


def test_monte_carlo_is_deterministic_for_a_seed():
    a = monte_carlo_capacity(1500.0, default_rf_terminal(),
                             {"tx_gain_dbi": 0.5}, n_draws=200, seed=9)
    b = monte_carlo_capacity(1500.0, default_rf_terminal(),
                             {"tx_gain_dbi": 0.5}, n_draws=200, seed=9)
    assert np.array_equal(a.samples_bps, b.samples_bps)


def test_monte_carlo_rejects_unknown_field():
    with pytest.raises(KeyError, match="unknown terminal fields"):
        monte_carlo_capacity(1500.0, default_rf_terminal(), {"nope": 1.0},
                             n_draws=200)


@pytest.mark.parametrize("kwargs", [{"range_km": 0.0}, {"n_draws": 10}])
def test_monte_carlo_validation(kwargs):
    base = {"range_km": 1500.0, "terminal": default_rf_terminal(),
            "sigmas": {"tx_gain_dbi": 0.5}, "n_draws": 200}
    base.update(kwargs)
    with pytest.raises(ValueError):
        monte_carlo_capacity(**base)


def test_monte_carlo_rejects_negative_sigma():
    with pytest.raises(ValueError, match="sigma"):
        monte_carlo_capacity(1500.0, default_rf_terminal(),
                             {"tx_gain_dbi": -0.5}, n_draws=200)

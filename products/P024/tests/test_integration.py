"""End-to-end integration: scenarios to trained scheduler to held-out comparison.

Deliberately small so it fits in a unit-test budget; the full-size campaign
lives in ``validation/learned_vs_fixed_ci.py``.
"""

from __future__ import annotations

import numpy as np

from detumblesim import (
    BDotController,
    CrossProductController,
    DetumbleConfig,
    FixedGainPolicy,
    GainScheduler,
    Magnetorquer,
    PowerLawGainPolicy,
    ScheduledGainPolicy,
    controllability_report,
    detumble_time_first_order,
    fit_power_law_gain,
    inertia_from_diagonal,
    magnetic_torque,
    oracle_gain,
    orbit_field_moments,
    paired_difference_ci,
    run_policy,
    sample_scenarios,
    simulate_detumble,
    training_rows,
)
from detumblesim.orbit import CircularOrbit

FAST = {"duration_s": 12000.0, "control_dt_s": 4.0, "substeps": 1}


class TestPhysicsPipeline:
    def test_bdot_and_cross_product_both_detumble(self):
        cfg = DetumbleConfig(
            inertia=inertia_from_diagonal(0.05, 0.05, 0.05),
            orbit=CircularOrbit(altitude_km=500.0, inclination_deg=97.4),
            magnetorquer=Magnetorquer.isotropic(0.2),
            omega0_rad_s=np.radians([8.0, -6.0, 5.0]),
            duration_s=23000.0, control_dt_s=2.0, substeps=1,
            target_rate_rad_s=np.radians(1.0), stop_when_detumbled=True,
        )
        bdot = simulate_detumble(cfg, BDotController(gain=1.5e5))
        cross = simulate_detumble(cfg, CrossProductController(gain=6.0e-5))
        assert bdot.detumbled and cross.detumbled
        for r in (bdot, cross):
            dots = np.sum(r.torque_nm * r.b_body_t, axis=1)
            scale = np.linalg.norm(r.torque_nm, axis=1) * np.linalg.norm(r.b_body_t, axis=1)
            assert np.all(np.abs(dots) <= 1e-10 * (scale + 1e-30))

    def test_analytic_model_brackets_an_unsaturated_run(self):
        orbit = CircularOrbit(altitude_km=500.0, inclination_deg=97.4)
        j = 0.05
        w0 = np.radians([6.0, -5.0, 5.7])
        target = np.radians(1.0)
        cfg = DetumbleConfig(
            inertia=inertia_from_diagonal(j, j, j), orbit=orbit,
            magnetorquer=Magnetorquer.isotropic(50.0),  # saturation disabled
            omega0_rad_s=w0, duration_s=60000.0, control_dt_s=4.0, substeps=1,
            target_rate_rad_s=target, stop_when_detumbled=True,
        )
        r = simulate_detumble(cfg, FixedGainPolicy(1.0e4))
        assert r.saturated_fraction == 0.0
        mom = orbit_field_moments(orbit, 2000, 10.0 * orbit.period_s)
        rate0 = float(np.linalg.norm(w0))
        fast = detumble_time_first_order(j, 1e4, mom, rate0, target, "fastest")
        slow = detumble_time_first_order(j, 1e4, mom, rate0, target, "slowest")
        assert fast < r.detumble_time_s < slow

    def test_equatorial_orbit_leaves_a_residual_about_the_weak_axis(self):
        eq = CircularOrbit(altitude_km=500.0, inclination_deg=0.0)
        pol = CircularOrbit(altitude_km=500.0, inclination_deg=97.4)
        common = {
            "inertia": inertia_from_diagonal(0.05, 0.05, 0.05),
            "magnetorquer": Magnetorquer.isotropic(0.2),
            "omega0_rad_s": np.radians([5.0, 5.0, 5.0]),
            "duration_s": 23000.0, "control_dt_s": 2.0, "substeps": 1,
            "target_rate_rad_s": np.radians(1.0),
        }
        r_eq = simulate_detumble(DetumbleConfig(orbit=eq, **common), FixedGainPolicy(1.5e5))
        r_pol = simulate_detumble(DetumbleConfig(orbit=pol, **common), FixedGainPolicy(1.5e5))
        assert r_eq.rate_norm_rad_s[-1] > r_pol.rate_norm_rad_s[-1]
        assert controllability_report(eq, 2000, 10 * eq.period_s).weighted_eigenvalues[0] < (
            controllability_report(pol, 2000, 10 * pol.period_s).weighted_eigenvalues[0]
        )

    def test_torque_matches_the_dipole_cross_field_definition(self):
        cfg = DetumbleConfig(
            inertia=inertia_from_diagonal(0.05, 0.06, 0.04),
            orbit=CircularOrbit(), magnetorquer=Magnetorquer.isotropic(0.2),
            omega0_rad_s=np.radians([8.0, -6.0, 5.0]),
            duration_s=1000.0, control_dt_s=2.0, substeps=1,
        )
        r = simulate_detumble(cfg, FixedGainPolicy(1e5))
        for i in (1, 50, 200):
            assert np.allclose(
                r.torque_nm[i], magnetic_torque(r.dipole_am2[i], r.b_body_t[i])
            )


class TestLearningPipeline:
    def test_train_and_evaluate_end_to_end(self):
        gains = np.geomspace(1e4, 1e6, 5)
        train = sample_scenarios(5, 90000)
        best, grid = [], []
        for s in train:
            bg, _, costs = oracle_gain(s, gains, **FAST)
            best.append(bg)
            grid.append(costs)
        k_fixed = float(gains[int(np.argmin(np.mean(grid, axis=0)))])

        coef, rms = fit_power_law_gain(train, np.array(best))
        assert coef.shape == (3,) and np.isfinite(rms)

        rows_x, rows_y = [], []
        for s, bg in zip(train, best, strict=True):
            res, _ = run_policy(s, FixedGainPolicy(k_fixed), **FAST)
            x, y = training_rows(s, bg, k_fixed, res, window_length=30, stride=20)
            if x.size:
                rows_x.append(x)
                rows_y.append(y)
        x = np.vstack(rows_x)
        y = np.concatenate(rows_y)
        assert x.shape[0] > 20
        sch = GainScheduler(n_estimators=40).fit(x, y)

        test = sample_scenarios(6, 95000)
        costs = {"fixed": [], "powerlaw": [], "learned": []}
        for s in test:
            m_max = float(np.min(s.magnetorquer.max_dipole_am2))
            costs["fixed"].append(run_policy(s, FixedGainPolicy(k_fixed), **FAST)[1].cost)
            costs["powerlaw"].append(
                run_policy(
                    s, PowerLawGainPolicy(coef, m_max, s.inertia_scale_kgm2), **FAST
                )[1].cost
            )
            pol = ScheduledGainPolicy(
                sch, k_fixed, m_max, s.inertia_scale_kgm2, window=30, update_every=20
            )
            costs["learned"].append(run_policy(s, pol, **FAST)[1].cost)
            assert pol.gain_history
            assert all(0.0 < c <= 1.0 for _, _, c in pol.gain_history)

        for v in costs.values():
            assert len(v) == 6 and all(np.isfinite(v))
        d = paired_difference_ci(costs["learned"], costs["fixed"])
        assert d.n == 6 and np.isfinite(d.mean)

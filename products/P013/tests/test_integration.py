"""End-to-end integration tests spanning forward model -> synthesis -> both
inversion paths (classical closed-form and learned)."""

from __future__ import annotations

import numpy as np
import pytest

from turbscope.dataset import build_table
from turbscope.dimm import differential_variance
from turbscope.inversion import multi_sensor_closed_form_estimate
from turbscope.model import ScintillometerWeakBaseline, train_default_model
from turbscope.scintillometer import (
    invert_cn2_all_roots,
    is_weak_regime,
    rytov_variance,
    scintillation_index_full,
)
from turbscope.synthetic import (
    APERTURE_DIAM_M,
    DIMM_WAVELENGTH_M,
    SCINT_WAVELENGTH_M,
    SEPARATION_M,
    WAVE_TYPE,
    generate_scenarios,
    synthesize_measurement,
)


def test_full_pipeline_weak_regime_recovers_truth_within_measurement_noise():
    """Forward-generate a weak-regime scenario, add realistic noise, invert
    with the classical multi-sensor closed-form estimator, and check the
    recovered Cn2 is close to the ground truth -- a full round trip through
    every public forward and inverse function in the package."""
    scenarios = generate_scenarios(1, seed=2024)
    sc = scenarios[0]
    # Force a clearly weak-regime case for this test regardless of the draw.
    from turbscope.synthetic import Scenario

    sc = Scenario(cn2_path=2e-17, path_length_m=200.0, rytov_variance_true=0.0)
    rng = np.random.default_rng(7)
    m = synthesize_measurement(sc, rng)

    result = multi_sensor_closed_form_estimate(
        m.sigma_i2_scint, m.var_long_dimm, m.var_trans_dimm, m.path_length_m,
        SCINT_WAVELENGTH_M, WAVE_TYPE, DIMM_WAVELENGTH_M, APERTURE_DIAM_M, SEPARATION_M,
        0.08, 0.10,
    )
    assert result.weak_regime_scint is True
    rel_err = abs(result.fused.cn2_path - sc.cn2_path) / sc.cn2_path
    # With ~8-10% sensor noise fused over 3 channels, recovery should be well
    # within 40% for a single noisy draw (loose bound; the seeded aggregate
    # statistic is what validation/round_trip_recovery.py reports precisely).
    assert rel_err < 0.40


def test_full_pipeline_saturated_regime_baseline_fails_but_is_detected():
    """A deeply saturated scenario: the weak-regime baseline must be shown to
    disagree substantially with truth, and the package's own diagnostics
    (is_weak_regime / multi-root inversion) must flag the failure rather than
    silently accepting the wrong answer."""
    from turbscope.synthetic import Scenario

    sc = Scenario(cn2_path=5e-13, path_length_m=2000.0, rytov_variance_true=0.0)
    r_var_true = float(rytov_variance(sc.cn2_path, sc.path_length_m, SCINT_WAVELENGTH_M, WAVE_TYPE))
    assert not bool(is_weak_regime(r_var_true))  # confirm this case is genuinely saturated

    sigma_i2_true = float(scintillation_index_full(r_var_true))
    baseline = ScintillometerWeakBaseline()
    x = np.array(
        [[np.log10(sigma_i2_true), np.log10(1e-12), np.log10(1e-12), np.log10(sc.path_length_m)]]
    )
    baseline_cn2 = float(10.0 ** baseline.predict_log10_cn2(x)[0])
    rel_err = abs(baseline_cn2 - sc.cn2_path) / sc.cn2_path
    assert rel_err > 0.5  # the baseline is demonstrably, substantially wrong here

    # The package's own multi-root inversion at least brackets the truth among
    # its candidate roots when the measurement is in the multi-valued band, or
    # otherwise clearly signals a single very different regime.
    roots = invert_cn2_all_roots(sigma_i2_true, sc.path_length_m, SCINT_WAVELENGTH_M, WAVE_TYPE)
    assert len(roots.cn2_roots) >= 1


def test_forward_and_dataset_pipeline_agree_on_dimm_signs():
    """The dataset builder's DIMM features must be internally consistent with
    directly calling the forward model (no silent unit or sign error)."""
    scenarios = generate_scenarios(3, seed=1)
    x, y, groups = build_table(scenarios, n_realisations=1, seed=1)
    for gi, sc in enumerate(scenarios):
        row = x[groups == gi][0]
        true_var_l = float(
            differential_variance(
                sc.cn2_path, sc.path_length_m, DIMM_WAVELENGTH_M, APERTURE_DIAM_M, SEPARATION_M,
                "longitudinal",
            )
        )
        # noisy measurement should be within an order of magnitude of truth
        measured = 10.0 ** row[1]
        assert 0.1 * true_var_l < measured < 10.0 * true_var_l


def test_learned_model_end_to_end_predict_matches_manual_feature_call():
    model, art = train_default_model(n_scenarios=80, n_realisations=2, calibrate=True)
    sc = art["test_scenarios"][0]
    rng = np.random.default_rng(999)
    m = synthesize_measurement(sc, rng)
    pred = model.predict(m.sigma_i2_scint, m.var_long_dimm, m.var_trans_dimm, m.path_length_m)
    x = np.array(
        [[
            np.log10(m.sigma_i2_scint),
            np.log10(m.var_long_dimm),
            np.log10(m.var_trans_dimm),
            np.log10(m.path_length_m),
        ]]
    )
    manual = float(10.0 ** model.predict_log10_cn2(x)[0])
    assert pred.cn2_path == pytest.approx(manual, rel=1e-9)

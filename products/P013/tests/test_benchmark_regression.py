"""Benchmark and regression tests.

These lock in (a) the physics constants that everything else is built on and
(b) the headline benchmark result, so that a future change that silently moves a
number fails the suite.  The reference values were produced by running this
configuration in this repository; the full-size numbers live in
``validation/VALIDATION.md``.
"""

from __future__ import annotations

import numpy as np
import pytest

from turbscope import (
    PathGeometry,
    TurbScopeModel,
    generate_dataset,
    saturation_peak,
    scintillation_branches,
)
from turbscope.model import BASELINES, split_dataset
from turbscope.scintillation import RYTOV_COEFFICIENT

# Reduced configuration so the suite stays fast; 1200 scenarios, 150 trees.
N_SCENARIOS = 1200
DATA_SEED = 424242
SPLIT_SEED = 17


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


@pytest.fixture(scope="module")
def benchmark():
    data = generate_dataset(N_SCENARIOS, seed=DATA_SEED)
    idx_fit, idx_cal, idx_test = split_dataset(len(data), seed=SPLIT_SEED)
    model = TurbScopeModel(n_estimators=150)
    model.fit(data.x[idx_fit], data.y[idx_fit])
    model.calibrate(data.x[idx_cal], data.y[idx_cal])
    test = data.take(idx_test)
    pred = model.predict(test.x)
    scores = {"learned model": pred.log10_cn2}
    for baseline in BASELINES:
        scores[baseline.name] = baseline.predict(test)
    return test, pred, scores


def test_physics_constants_are_pinned():
    """Regression guard on the derived Rytov constants and the saturation peak."""
    assert RYTOV_COEFFICIENT == pytest.approx(2.2522625262, rel=1e-10)
    b_peak, s_peak = saturation_peak(0.0)
    assert b_peak == pytest.approx(7.296562, rel=1e-5)
    assert s_peak == pytest.approx(1.6921035, rel=1e-6)
    # A reading of sigma_I^2 = 1.5 admits exactly two turbulence strengths.
    branches = scintillation_branches(1.5, 0.0)
    assert len(branches) == 2
    assert branches[0] == pytest.approx(2.893647, rel=1e-5)
    assert branches[1] == pytest.approx(31.54215, rel=1e-5)


def test_regression_rmse_by_regime(benchmark):
    test, _, scores = benchmark
    reg = test.regimes()
    weak = reg == 0
    saturated = reg == 3
    assert weak.sum() > 100 and saturated.sum() > 10

    # Reference values from an actual run of this configuration.
    reference = {
        ("learned model", "weak"): 0.05362,
        ("learned model", "saturated"): 0.18446,
        ("weak closed form (point scintillometer)", "weak"): 0.02265,
        ("weak closed form (point scintillometer)", "saturated"): 0.99021,
        ("saturation-aware inversion (point scintillometer)", "weak"): 0.02271,
        ("saturation-aware inversion (point scintillometer)", "saturated"): 0.66506,
        ("DIMM closed form (coherence kernel)", "weak"): 0.20167,
    }
    for (name, band), expected in reference.items():
        mask = weak if band == "weak" else saturated
        assert _rmse(scores[name][mask], test.y[mask]) == pytest.approx(expected, rel=2e-3)


def test_closed_form_beats_the_learned_model_in_the_weak_regime(benchmark):
    """The documented negative result: an unbiased analytic estimator wins in weak turbulence."""
    test, _, scores = benchmark
    weak = test.regimes() == 0
    analytic = _rmse(scores["weak closed form (point scintillometer)"][weak], test.y[weak])
    learned = _rmse(scores["learned model"][weak], test.y[weak])
    assert analytic < learned, "if this ever flips, re-read validation/VALIDATION.md section 5"


def test_learned_model_beats_the_closed_form_only_where_saturation_bites(benchmark):
    test, _, scores = benchmark
    saturated = test.regimes() == 3
    key = "weak closed form (point scintillometer)"
    analytic = _rmse(scores[key][saturated], test.y[saturated])
    learned = _rmse(scores["learned model"][saturated], test.y[saturated])
    assert learned < analytic


def test_interval_coverage_is_near_nominal(benchmark):
    test, pred, _ = benchmark
    coverage = float(np.mean((test.y >= pred.log10_lower) & (test.y <= pred.log10_upper)))
    assert coverage == pytest.approx(0.8667, abs=0.005)


def test_dataset_generation_is_deterministic():
    a = generate_dataset(60, seed=31337)
    b = generate_dataset(60, seed=31337)
    assert np.array_equal(a.x, b.x)
    assert np.array_equal(a.y, b.y)
    c = generate_dataset(60, seed=31338)
    assert not np.array_equal(a.x, c.x)


def test_model_fit_is_deterministic():
    data = generate_dataset(300, seed=515)
    x, y = data.x[:200], data.y[:200]
    preds = []
    for _ in range(2):
        m = TurbScopeModel(n_estimators=60).fit(x, y)
        preds.append(m.predict_log10(data.x[200:])[1])
    assert float(np.max(np.abs(preds[0] - preds[1]))) == 0.0


def test_forward_model_timing_budget():
    """The generator must stay well inside the 3-minute compute budget."""
    import time

    start = time.perf_counter()
    generate_dataset(500, seed=808)
    elapsed = time.perf_counter() - start
    assert elapsed < 30.0, f"dataset generation took {elapsed:.1f} s"


def test_path_geometry_repr_is_stable():
    p = PathGeometry(1000.0, 1.55e-6)
    assert "length_m=1000.0" in repr(p)

"""Dataset determinism, ML reproducibility, API contract, and interval-coverage tests."""

import numpy as np
import pytest

from fogcast import FogCastModel, generate_dataset, kim_attenuation_db_km, split_indices

N_SAMPLES = 6000
SEED = 42


@pytest.fixture(scope="module")
def data():
    return generate_dataset(n_samples=N_SAMPLES, seed=SEED)


@pytest.fixture(scope="module")
def trained_model():
    """Train once per module (~6 s) and share across tests."""
    return FogCastModel.train_default(n_samples=N_SAMPLES, seed=SEED)


class TestDataset:
    def test_deterministic_regeneration(self):
        """Same seed => bit-identical dataset (committed-script reproducibility)."""
        a = generate_dataset(n_samples=500, seed=7)
        b = generate_dataset(n_samples=500, seed=7)
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])

    def test_different_seed_differs(self):
        a = generate_dataset(n_samples=500, seed=7)
        b = generate_dataset(n_samples=500, seed=8)
        assert not np.array_equal(a["attenuation_db_km"], b["attenuation_db_km"])

    def test_ranges_and_positivity(self, data):
        assert np.all(data["visibility_km"] >= 0.05) and np.all(data["visibility_km"] <= 50.0)
        assert np.all(data["wavelength_nm"] >= 600.0) and np.all(data["wavelength_nm"] <= 1700.0)
        assert np.all(data["rh_percent"] >= 40.0) and np.all(data["rh_percent"] <= 100.0)
        assert np.all(data["attenuation_db_km"] > 0.0)

    def test_split_is_deterministic_and_disjoint(self):
        tr1, va1, te1 = split_indices(N_SAMPLES, seed=SEED)
        tr2, va2, te2 = split_indices(N_SAMPLES, seed=SEED)
        np.testing.assert_array_equal(tr1, tr2)
        np.testing.assert_array_equal(te1, te2)
        all_idx = np.concatenate([tr1, va1, te1])
        assert len(np.unique(all_idx)) == N_SAMPLES
        assert len(tr1) == 4200 and len(va1) == 900 and len(te1) == 900

    def test_truth_correlates_with_kim(self, data):
        """Sanity: synthetic truth is a perturbed Kim model, so it must track Kim."""
        kim = np.asarray(kim_attenuation_db_km(data["visibility_km"], data["wavelength_nm"]))
        ratio = data["attenuation_db_km"] / kim
        # Perturbations are bounded multiplicative effects; median ratio near 1.
        assert 0.9 < np.median(ratio) < 1.2


class TestReproducibility:
    def test_same_seed_same_predictions(self):
        """ML pipeline reproducibility: same seed => identical predictions."""
        m1 = FogCastModel.train_default(n_samples=1500, seed=11)
        m2 = FogCastModel.train_default(n_samples=1500, seed=11)
        v = np.array([0.2, 0.8, 3.0, 12.0])
        lam = np.array([850.0, 1310.0, 1550.0, 1550.0])
        rh = np.array([95.0, 80.0, 60.0, 50.0])
        p1, lo1, hi1 = m1.predict(v, lam, rh)
        p2, lo2, hi2 = m2.predict(v, lam, rh)
        np.testing.assert_array_equal(p1, p2)
        np.testing.assert_array_equal(lo1, lo2)
        np.testing.assert_array_equal(hi1, hi2)


class TestPredictAPI:
    def test_scalar_in_scalar_out(self, trained_model):
        out = trained_model.predict(1.0, 1550.0, 85.0)
        assert isinstance(out, tuple) and len(out) == 3
        point, lower, upper = out
        assert isinstance(point, float)
        assert lower <= point <= upper

    def test_point_only(self, trained_model):
        point = trained_model.predict(1.0, 1550.0, 85.0, return_interval=False)
        assert isinstance(point, float)
        assert point > 0.0

    def test_interval_ordering_on_grid(self, trained_model):
        v = np.geomspace(0.05, 50.0, 40)
        p, lo, hi = trained_model.predict(v, np.full_like(v, 1550.0), np.full_like(v, 80.0))
        assert np.all(lo <= p) and np.all(p <= hi)
        assert np.all(p > 0.0)

    def test_prediction_physically_plausible(self, trained_model):
        """Dense fog must attenuate far more than clear haze (learned monotone trend)."""
        dense = trained_model.predict(0.1, 1550.0, 95.0, return_interval=False)
        clear = trained_model.predict(20.0, 1550.0, 50.0, return_interval=False)
        assert dense > 50.0 * clear

    def test_input_validation(self, trained_model):
        with pytest.raises(ValueError):
            trained_model.predict(-1.0, 1550.0, 80.0)
        with pytest.raises(ValueError):
            trained_model.predict(1.0, 100.0, 80.0)
        with pytest.raises(ValueError):
            trained_model.predict(1.0, 1550.0, 150.0)
        with pytest.raises(ValueError):
            trained_model.predict(1.0, 1550.0, -5.0)

    def test_unfitted_model_raises(self):
        with pytest.raises(RuntimeError):
            FogCastModel().predict(1.0, 1550.0, 80.0)

    def test_fit_rejects_bad_targets(self):
        m = FogCastModel()
        with pytest.raises(ValueError):
            m.fit([1.0, 2.0], [850.0, 850.0], [50.0, 50.0], [1.0, -3.0])
        with pytest.raises(ValueError):
            m.fit([1.0, 2.0], [850.0, 850.0], [50.0, 50.0], [1.0])


class TestIntervalCoverage:
    def test_coverage_near_nominal_on_held_out_data(self, trained_model, data):
        """Empirical 90 % PI coverage on the held-out test split within +/- 5 pp.

        The model never saw the test split (train/val/test = 70/15/15, fixed seed).
        """
        _, _, idx_test = split_indices(N_SAMPLES, seed=SEED)
        y = data["attenuation_db_km"][idx_test]
        _, lo, hi = trained_model.predict(
            data["visibility_km"][idx_test],
            data["wavelength_nm"][idx_test],
            data["rh_percent"][idx_test],
        )
        coverage = float(np.mean((y >= lo) & (y <= hi)))
        assert 0.85 <= coverage <= 0.95, f"coverage {coverage:.3f} outside [0.85, 0.95]"

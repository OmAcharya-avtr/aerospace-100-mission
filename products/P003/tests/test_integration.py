"""End-to-end integration test: simulate -> build dataset -> train -> predict."""

import numpy as np
import pytest

from scintinet import SimParams, Surrogate, rytov_baseline, simulate_scintillation


def test_generate_train_predict_end_to_end():
    """Miniature version of the full campaign -> surrogate pipeline.

    Uses a reduced grid (128^2, 4 screens, 3 realizations) so the whole test
    runs in well under a minute on 2 CPU cores.
    """
    cn2_vals = (3e-16, 1e-15)
    length_vals = (1000.0, 2000.0)
    lam = 1.55e-6
    apertures = (0.02, 0.06)

    rows, targets = [], []
    seed = 100
    for cn2 in cn2_vals:
        for ell in length_vals:
            p = SimParams(
                cn2=cn2, wavelength=lam, path_length=ell,
                aperture_diameters=apertures,
                grid_size=128, grid_width=0.4, n_screens=4, n_realizations=3,
            )
            r = simulate_scintillation(p, seed=seed)
            seed += 1
            for dia in apertures:
                rows.append([cn2, ell, lam, dia])
                targets.append(r.sigma_i2_aperture[dia])

    x = np.array(rows)
    y = np.array(targets)
    assert np.all(y > 0.0), "weak-turbulence sims must yield positive sigma_I^2"

    surrogate = Surrogate(n_members=3, hidden_layer_sizes=(16,), random_state=0)
    surrogate.fit(x, y)
    mean, std = surrogate.predict(x, return_std=True)

    # Pipeline soundness: finite, positive, correct shapes, usable uncertainty.
    assert mean.shape == y.shape
    assert std.shape == y.shape
    assert np.all(np.isfinite(mean)) and np.all(mean > 0.0)
    assert np.all(np.isfinite(std)) and np.all(std >= 0.0)

    # The surrogate must reproduce its own (tiny) training set to within a
    # factor of ~2 in every point — a loose but real end-to-end check.
    ratio = mean / y
    assert np.all(ratio > 0.5) and np.all(ratio < 2.0)

    # Analytic baseline runs on the same design matrix without error and is
    # the same order of magnitude as simulation (weak-regime consistency).
    base = rytov_baseline(x)
    assert np.all(base > 0.0)
    assert np.median(base / y) == pytest.approx(1.0, abs=0.8)

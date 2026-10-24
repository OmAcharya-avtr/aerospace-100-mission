"""The synthetic noise processes, against their closed-form Allan deviations."""

from __future__ import annotations

import numpy as np
import pytest

from timetransfer.errors import ConfigurationError
from timetransfer.noise import (
    NOISE_KINDS,
    random_walk_fm_phase,
    synthetic_phase,
    theoretical_adev,
    theory_slope,
    white_fm_phase,
    white_pm_phase,
)


def test_theory_slopes_are_the_textbook_values() -> None:
    assert theory_slope("white_pm") == -1.0
    assert theory_slope("white_fm") == -0.5
    assert theory_slope("random_walk_fm") == 0.5


def test_theory_slope_rejects_an_unknown_kind() -> None:
    with pytest.raises(ConfigurationError, match="unknown noise kind"):
        theory_slope("pink")


def test_theoretical_adev_known_answers() -> None:
    # white PM : sqrt(3) * sigma_x / tau = 1.7320508 * 1e-9 / 10 = 1.7320508e-10
    # white FM : sqrt(q1 / tau) = sqrt(1e-20 / 10) = 1e-10.5 = 3.1622777e-11
    # RW FM    : sqrt(q2 * tau / 3) = sqrt(1e-24 * 10 / 3) = sqrt(3.3333e-24)
    #                                                      = 1.8257419e-12
    tau = np.asarray([10.0])
    assert theoretical_adev("white_pm", tau, level=1e-9)[0] == pytest.approx(
        1.7320508075688772e-10, rel=1e-12
    )
    assert theoretical_adev("white_fm", tau, level=1e-20)[0] == pytest.approx(
        3.1622776601683794e-11, rel=1e-12
    )
    assert theoretical_adev("random_walk_fm", tau, level=1e-24)[0] == pytest.approx(
        1.8257418583505536e-12, rel=1e-12
    )


def test_white_pm_variance_matches_the_declared_sigma() -> None:
    x = white_pm_phase(40000, sigma_x_s=1e-9, seed=3)
    assert float(np.std(x, ddof=1)) == pytest.approx(1e-9, rel=0.02)


def test_white_fm_increment_variance_matches_q1_tau0() -> None:
    q1, tau0 = 1e-20, 2.0
    x = white_fm_phase(40000, tau0_s=tau0, q1_s2_per_s=q1, seed=5)
    assert float(np.var(np.diff(x), ddof=1)) == pytest.approx(q1 * tau0, rel=0.03)
    assert x[0] == 0.0


def test_random_walk_fm_phase_increment_variance_matches_the_exact_covariance() -> None:
    # For the integrated Wiener process the phase increment over one step has
    # variance q2 * h^3 / 3 only when the frequency starts at zero, which it
    # does for the first step. The first increment is therefore the clean test.
    q2, h = 1e-20, 1.0
    firsts = []
    for seed in range(3000):
        x = random_walk_fm_phase(3, tau0_s=h, q2_per_s=q2, seed=seed)
        firsts.append(x[1] - x[0])
    assert float(np.var(np.asarray(firsts), ddof=1)) == pytest.approx(q2 * h**3 / 3.0, rel=0.08)


@pytest.mark.parametrize("kind", NOISE_KINDS)
def test_synthetic_phase_is_deterministic_in_the_seed(kind: str) -> None:
    level = {"white_pm": 1e-9, "white_fm": 1e-20, "random_walk_fm": 1e-24}[kind]
    a = synthetic_phase(kind, 500, tau0_s=1.0, level=level, seed=12)
    b = synthetic_phase(kind, 500, tau0_s=1.0, level=level, seed=12)
    c = synthetic_phase(kind, 500, tau0_s=1.0, level=level, seed=13)
    assert a.tobytes() == b.tobytes()
    assert a.tobytes() != c.tobytes()


def test_synthetic_phase_rejects_an_unknown_kind() -> None:
    with pytest.raises(ConfigurationError, match="unknown noise kind"):
        synthetic_phase("flicker_pm", 100, tau0_s=1.0, level=1e-9, seed=0)


@pytest.mark.parametrize(
    ("fn", "kwargs", "match"),
    [
        (white_pm_phase, {"n": 2, "sigma_x_s": 1e-9, "seed": 0}, "n must be"),
        (white_pm_phase, {"n": 10, "sigma_x_s": 0.0, "seed": 0}, "sigma_x_s"),
        (white_fm_phase, {"n": 10, "tau0_s": 0.0, "q1_s2_per_s": 1e-20, "seed": 0}, "tau0_s"),
        (white_fm_phase, {"n": 10, "tau0_s": 1.0, "q1_s2_per_s": 0.0, "seed": 0}, "q1_s2_per_s"),
        (
            random_walk_fm_phase,
            {"n": 10, "tau0_s": 1.0, "q2_per_s": -1.0, "seed": 0},
            "q2_per_s",
        ),
    ],
)
def test_noise_generators_validate(fn, kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ConfigurationError, match=match):
        fn(**kwargs)


def test_theoretical_adev_rejects_non_positive_tau() -> None:
    with pytest.raises(ConfigurationError, match="tau_s"):
        theoretical_adev("white_fm", np.asarray([0.0]), level=1e-20)

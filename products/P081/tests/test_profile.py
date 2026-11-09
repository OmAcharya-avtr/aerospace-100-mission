"""Baseline 3: the flat profile, the grid, and the override of the covariance."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from twindiverge.attribution import CONFIDENCE_LEVEL
from twindiverge.identifiability import analyse
from twindiverge.plant import NOMINAL, perturb
from twindiverge.profile import (
    FLAT_TOLERANCE,
    GRID_HALF_WIDTH,
    GRID_MIN_HALF_WIDTH,
    GRID_WALD_MULTIPLE,
    WARM_START_FLOOR,
    chi_square,
    fit,
    grid_half_width_for,
    profile_direction,
    profile_parameters,
)
from twindiverge.telemetry import measure

CRITICAL = float(stats.norm.ppf(0.5 + CONFIDENCE_LEVEL / 2.0))


def _observation(telemetry, offset, seed=11):
    return measure(telemetry, perturb(NOMINAL, offset), seed=seed)[0]


def test_every_profile_is_flat_on_a_rate_only_set_and_overrides_the_covariance(short_set):
    """The headline of this module. A change in any one parameter is cancelled
    exactly by a compensating move along the scale direction, so the profile
    cannot move; the pseudo-inverse diagonal of the Fisher information
    nevertheless reports finite half-widths. Exact in exact arithmetic; the
    measured maxima below are rounding."""
    offset = np.zeros(5)
    offset[0] = 0.06
    report = analyse(short_set, NOMINAL)
    results = profile_parameters(
        short_set,
        _observation(short_set, offset),
        CRITICAL * report.pinv_diagonal_not_a_bound,
        points=3,
    )
    assert len(results) == 5
    for result in results:
        assert result.max_delta_chi2 < FLAT_TOLERANCE, result.label
        assert result.flat
        assert result.interval is None
        assert np.isfinite(result.wald_half_width)
        assert result.overrides_covariance
        assert not np.isfinite(result.covariance_ratio)


def test_a_full_rank_set_gives_curved_profiles_that_score_the_covariance(torque_set):
    """Baseline 3 against the linearisation. Where the profile closes an
    interval, the ratio to the Wald half-width is the score."""
    offset = np.zeros(5)
    offset[0] = 0.02
    report = analyse(torque_set, NOMINAL)
    results = profile_parameters(
        torque_set,
        _observation(torque_set, offset),
        CRITICAL * report.pinv_diagonal_not_a_bound,
        points=3,
    )
    curved = [result for result in results if not result.flat]
    assert len(curved) == 5
    for result in curved:
        assert result.max_delta_chi2 > FLAT_TOLERANCE
        assert not result.overrides_covariance


def test_the_grid_half_width_follows_the_linearised_width():
    assert grid_half_width_for(float("inf")) == pytest.approx(
        GRID_HALF_WIDTH, rel=0.0, abs=0.0
    )
    assert grid_half_width_for(1.0) == pytest.approx(GRID_HALF_WIDTH, rel=0.0, abs=0.0)
    assert grid_half_width_for(1.0e-12) == pytest.approx(
        GRID_MIN_HALF_WIDTH, rel=0.0, abs=0.0
    )
    assert grid_half_width_for(0.01) == pytest.approx(
        GRID_WALD_MULTIPLE * 0.01, rel=1e-14, abs=0.0
    )


def test_the_chi_square_of_the_truth_is_near_its_row_count(short_set):
    offset = np.zeros(5)
    offset[1] = 0.03
    observed = _observation(short_set, offset)
    value = chi_square(short_set, observed, offset)
    assert abs(value - short_set.rows) < 4.0 * np.sqrt(2.0 * short_set.rows)


def test_the_fit_reaches_at_least_the_chi_square_of_the_truth(short_set):
    offset = np.zeros(5)
    offset[1] = 0.03
    observed = _observation(short_set, offset)
    _, fitted = fit(short_set, observed)
    assert fitted <= chi_square(short_set, observed, offset) + 1e-6


def test_profile_direction_accepts_a_combination_not_only_an_axis(short_set):
    """A telemetry set with a null space has nothing worth profiling along the
    axes, and the identifiable combinations are exactly as profilable.

    The grid here is centred on the unperturbed twin rather than on the fitted
    optimum, so the profile rises steeply on one side and the threshold is
    crossed once. One side is the ordinary outcome on this plant and the
    one-sided half-width is what the data support; the two-sided ``interval``
    is correctly ``None`` and that is asserted rather than worked around.
    """
    report = analyse(short_set, NOMINAL)
    best = report.identifiable_basis[0]
    wald = CRITICAL * float(report.direction_crb_relative[0])
    result = profile_direction(
        short_set,
        _observation(short_set, np.zeros(5)),
        best,
        "best combination",
        wald,
        half_width=grid_half_width_for(wald),
        points=5,
    )
    assert not result.flat
    assert result.sides_crossed == 1
    assert result.interval is None
    assert np.isfinite(result.one_sided_half_width)
    assert result.one_sided_ratio > 0.0


def test_profiling_the_scale_direction_is_exactly_flat(short_set):
    """Equation (6) again, from the other side: the null direction itself."""
    report = analyse(short_set, NOMINAL)
    result = profile_direction(
        short_set,
        _observation(short_set, np.zeros(5)),
        report.null_basis[0],
        "scale direction",
        float("inf"),
        points=5,
    )
    assert result.flat
    assert result.max_delta_chi2 < FLAT_TOLERANCE
    assert not result.overrides_covariance
    assert np.isnan(result.covariance_ratio)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"points": 4}, "odd integer"),
        ({"points": 1}, "odd integer"),
        ({"half_width": 0.0}, "half_width"),
    ],
)
def test_profile_direction_validates_its_grid(short_set, kwargs: dict, message: str):
    with pytest.raises(ValueError, match=message):
        profile_direction(
            short_set,
            _observation(short_set, np.zeros(5)),
            np.eye(5)[0],
            "log J",
            0.01,
            **kwargs,
        )


def test_profile_direction_validates_its_direction(short_set):
    observed = _observation(short_set, np.zeros(5))
    with pytest.raises(ValueError, match="5 components"):
        profile_direction(short_set, observed, np.zeros(3), "x", 0.01)
    with pytest.raises(ValueError, match="non-zero"):
        profile_direction(short_set, observed, np.zeros(5), "x", 0.01)


def test_profile_parameters_validates_its_widths(short_set):
    with pytest.raises(ValueError, match="5 Wald half-widths"):
        profile_parameters(short_set, _observation(short_set, np.zeros(5)), np.ones(3))


def test_the_warm_start_floor_is_below_every_genuine_warm_start():
    """The floor only ever snaps a start that is already indistinguishable from
    zero. See the constant's own documentation for the measured scipy
    behaviour it exists for."""
    assert 0.0 < WARM_START_FLOOR < 1.0e-9

"""Feature extraction, including the signed and zero ``B*`` cases."""

from __future__ import annotations

import numpy as np
import pytest
from sgp4.api import WGS72 as SGP4_WGS72
from sgp4.api import Satrec

from orbitdrift.constants import WGS72
from orbitdrift.features import (
    ALL_FEATURE_NAMES,
    FEATURE_NAMES,
    design_matrix,
    epoch_jd,
    features_from_satrec,
    period_s_from_satrec,
    seconds_to_days,
)


def _satrec(bstar: float, *, no_kozai: float = 0.0011) -> Satrec:
    satrec = Satrec()
    satrec.sgp4init(
        SGP4_WGS72, "i", 99999, 27774.0, bstar, 0.0, 0.0, 0.001, 0.5, 0.9, 0.3, no_kozai, 0.0
    )
    return satrec


def test_feature_names_and_array_agree_in_order_and_length() -> None:
    features = features_from_satrec(_satrec(1e-4))
    array = features.as_array()
    assert array.shape == (len(FEATURE_NAMES),)
    assert ALL_FEATURE_NAMES == (*FEATURE_NAMES, "epoch_age_days")
    assert array[FEATURE_NAMES.index("bstar")] == pytest.approx(1e-4, rel=1e-4, abs=0.0)


def test_positive_negative_and_zero_bstar(moderate_drag_satrec: Satrec) -> None:
    """``B*`` is signed, so it is carried as a value, a log magnitude and a sign.

    A single ``log10(B*)`` column would be undefined for the negative and the
    zero cases, both of which are legal two-line element sets.
    """
    positive = features_from_satrec(_satrec(1e-4))
    assert positive.bstar_sign == 1.0
    assert positive.log10_abs_bstar == pytest.approx(-4.0, rel=0.0, abs=1e-3)

    negative = features_from_satrec(_satrec(-1e-4))
    assert negative.bstar_sign == -1.0
    assert negative.log10_abs_bstar == pytest.approx(-4.0, rel=0.0, abs=1e-3)

    zero = features_from_satrec(_satrec(0.0))
    assert zero.bstar_sign == 0.0
    assert zero.log10_abs_bstar == -30.0
    assert np.isfinite(zero.log10_drag_parameter)

    real = features_from_satrec(moderate_drag_satrec)
    assert real.bstar > 0.0
    assert real.perigee_altitude_km == pytest.approx(377.0, rel=0.0, abs=5.0)


def test_altitudes_are_scaled_by_the_wgs72_radius(moderate_drag_satrec: Satrec) -> None:
    """``satrec.altp`` and ``satrec.alta`` are in Earth radii of the model SGP4 used."""
    features = features_from_satrec(moderate_drag_satrec)
    assert features.perigee_altitude_km == pytest.approx(
        float(moderate_drag_satrec.altp) * WGS72.radius_km, rel=1e-13, abs=0.0
    )
    assert features.apogee_altitude_km >= features.perigee_altitude_km


def test_period_from_mean_motion_known_answer() -> None:
    """``no_kozai`` is rad/min, so the period is ``2 pi / n`` minutes.

    Hand-calculated at n = 0.0011 rad/min: 2 pi / 0.0011 = 5711.98666... min
    = 342719.2 s, and the feature reports minutes.
    """
    features = features_from_satrec(_satrec(1e-4, no_kozai=0.0011))
    assert features.period_min == pytest.approx(2.0 * np.pi / 0.0011, rel=1e-9, abs=0.0)
    assert period_s_from_satrec(_satrec(1e-4, no_kozai=0.0011)) == pytest.approx(
        60.0 * 2.0 * np.pi / 0.0011, rel=1e-9, abs=0.0
    )


def test_period_refuses_a_non_positive_mean_motion() -> None:
    class Fake:
        no_kozai = 0.0

    with pytest.raises(ValueError, match="is not positive"):
        period_s_from_satrec(Fake())  # type: ignore[arg-type]


def test_epoch_jd_sums_the_split_julian_date() -> None:
    satrec = _satrec(1e-4)
    assert epoch_jd(satrec) == float(satrec.jdsatepoch) + float(satrec.jdsatepochF)


def test_design_matrix_broadcasts_and_validates() -> None:
    static = np.tile(features_from_satrec(_satrec(1e-4)).as_array(), (3, 1))
    ages = np.array([1.0, 2.0, 3.0])
    matrix, names = design_matrix(static, ages)
    assert matrix.shape == (3, len(ALL_FEATURE_NAMES))
    assert names == ALL_FEATURE_NAMES
    assert np.allclose(matrix[:, -1], ages)

    with pytest.raises(ValueError, match="same length"):
        design_matrix(static, np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="must have 10 columns"):
        design_matrix(np.zeros((3, 4)), ages)


def test_seconds_to_days() -> None:
    assert float(seconds_to_days(86400.0)) == 1.0
    assert np.allclose(seconds_to_days(np.array([0.0, 43200.0])), [0.0, 0.5])

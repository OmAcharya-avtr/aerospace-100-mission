"""Tests for Kolmogorov Zernike statistics and Noll residual variances."""

from __future__ import annotations

import numpy as np
import pytest

from zernkit import (
    KOLMOGOROV_PSD_CONSTANT,
    NOLL_PSD_CONSTANT,
    NOLL_TABLE_IV,
    coefficient_variance,
    coefficient_variance_noll,
    residual_variance,
    residual_variance_asymptotic,
)


def test_psd_constants() -> None:
    assert NOLL_PSD_CONSTANT == 0.023
    # 0.490 / (2 pi)^(5/3) = 0.490 / 21.3937... = 0.022904...
    assert KOLMOGOROV_PSD_CONSTANT == pytest.approx(0.0229037, rel=1e-4)


def test_variance_depends_only_on_radial_order() -> None:
    """Every mode of a given n has the same Kolmogorov variance."""
    # Noll j = 4, 5, 6 are all n = 2 (defocus and the two astigmatisms).
    v = [coefficient_variance_noll(j) for j in (4, 5, 6)]
    assert v[0] == pytest.approx(v[1]) == pytest.approx(v[2])


def test_low_order_variances_against_noll_table_differences() -> None:
    """Consecutive Delta_J differences are the per-mode variances."""
    # tip/tilt: (Delta_1 - Delta_3)/2 = (1.0299 - 0.134)/2 = 0.44795
    assert coefficient_variance(1) == pytest.approx(0.44795, rel=0.02)
    # defocus: Delta_3 - Delta_4 = 0.134 - 0.111 = 0.023
    assert coefficient_variance(2) == pytest.approx(0.023, rel=0.03)
    # coma/trefoil (n=3): (Delta_6 - Delta_10)/4 = (0.0648 - 0.0401)/4 = 0.006175
    assert coefficient_variance(3) == pytest.approx(0.006175, rel=0.03)
    # n=4: (Delta_10 - Delta_15)/5 = (0.0401 - 0.0279)/5 = 0.00244
    assert coefficient_variance(4) == pytest.approx(0.00244, rel=0.05)


def test_residual_variance_matches_published_table_within_one_percent() -> None:
    worst = 0.0
    for j, published in NOLL_TABLE_IV.items():
        computed = residual_variance(j)
        worst = max(worst, abs(computed - published) / published)
    assert worst < 0.01, f"worst relative deviation {worst:.4%} exceeds 1 %"


def test_unrounded_psd_constant_improves_agreement() -> None:
    """Using 0.490/(2pi)^(5/3) instead of Noll's rounded 0.023 halves the gap."""

    def worst(constant: float) -> float:
        return max(
            abs(residual_variance(j, psd_constant=constant) - p) / p
            for j, p in NOLL_TABLE_IV.items()
        )

    assert worst(KOLMOGOROV_PSD_CONSTANT) < worst(NOLL_PSD_CONSTANT)
    assert worst(KOLMOGOROV_PSD_CONSTANT) < 0.006


def test_residual_variance_is_monotonically_decreasing() -> None:
    values = [residual_variance(j) for j in range(1, 60)]
    assert all(b < a for a, b in zip(values[:-1], values[1:], strict=True))


def test_residual_variance_equals_sum_of_remaining_coefficient_variances() -> None:
    """Delta_J - Delta_{J+1} must equal <a_{J+1}^2> exactly."""
    for j in range(1, 30):
        step = residual_variance(j) - residual_variance(j + 1)
        assert step == pytest.approx(coefficient_variance_noll(j + 1), rel=1e-9)


def test_d_over_r0_scaling_is_five_thirds() -> None:
    base = residual_variance(4, d_over_r0=1.0)
    scaled = residual_variance(4, d_over_r0=8.0)
    assert scaled / base == pytest.approx(8.0 ** (5.0 / 3.0), rel=1e-12)


def test_asymptotic_matches_table_at_large_j() -> None:
    # Delta_21 published 0.0208; 0.2944 * 21^(-sqrt(3)/2) = 0.02110
    assert residual_variance_asymptotic(21) == pytest.approx(0.0208, rel=0.02)
    assert residual_variance_asymptotic(20) == pytest.approx(NOLL_TABLE_IV[20], rel=0.03)


def test_asymptotic_scaling() -> None:
    assert residual_variance_asymptotic(4, d_over_r0=2.0) / residual_variance_asymptotic(
        4
    ) == pytest.approx(2.0 ** (5.0 / 3.0))


def test_truncation_of_the_sum_is_negligible() -> None:
    fine = residual_variance(5, n_max=400_000)
    coarse = residual_variance(5, n_max=20_000)
    assert abs(fine - coarse) < 1e-7


def test_hand_check_delta_1_against_structure_function_value() -> None:
    """Independent route: the piston-removed Kolmogorov variance over a disc.

    sigma^2 = (1/2) <D_phi(|r1 - r2|)> with D_phi(r) = 6.883877 (r/r0)^(5/3)
    (Fried 1965 / Noll 1976) and <s^(5/3)> = 0.29995 for two uniform points in
    a disc of diameter 1 gives sigma^2 = 1.03242 (D/r0)^(5/3). The Zernike
    series must agree with that, and both sit just above Noll's tabulated
    1.0299 because of the rounded spectral constant.
    """
    computed = residual_variance(1, psd_constant=KOLMOGOROV_PSD_CONSTANT)
    assert computed == pytest.approx(1.03242, rel=0.005)


def test_input_validation() -> None:
    with pytest.raises(ValueError, match="n must be >= 1"):
        coefficient_variance(0)
    with pytest.raises(TypeError):
        coefficient_variance(2.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="piston"):
        coefficient_variance_noll(1)
    with pytest.raises(ValueError, match="j_removed must be >= 1"):
        residual_variance(0)
    with pytest.raises(TypeError):
        residual_variance(3.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="d_over_r0"):
        residual_variance(3, d_over_r0=0.0)
    with pytest.raises(ValueError, match="d_over_r0"):
        coefficient_variance(2, d_over_r0=-1.0)
    with pytest.raises(ValueError, match="n_max"):
        residual_variance(3, n_max=5)
    with pytest.raises(ValueError, match="n_max"):
        residual_variance(200, n_max=10)
    with pytest.raises(ValueError, match="j_removed must be >= 1"):
        residual_variance_asymptotic(0)
    with pytest.raises(TypeError):
        residual_variance_asymptotic(2.5)  # type: ignore[arg-type]


def test_table_is_reference_data_only() -> None:
    """The published table must never be used to produce computed values."""
    assert set(NOLL_TABLE_IV) == set(range(1, 22))
    assert NOLL_TABLE_IV[1] == 1.0299
    computed = np.array([residual_variance(j) for j in NOLL_TABLE_IV])
    published = np.array(list(NOLL_TABLE_IV.values()))
    assert not np.array_equal(computed, published)

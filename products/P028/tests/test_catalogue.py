"""Synthetic catalogue generation and preparation."""

from __future__ import annotations

import numpy as np
import pytest

from skymatch.catalogue import (
    StarCatalogue,
    expected_close_pairs,
    generate_catalogue,
    predicted_count,
    remove_close_pairs,
)
from skymatch.geometry import angular_separation


class TestKnownAnswers:
    def test_predicted_count_at_the_reference_magnitude(self) -> None:
        # Eq. C1 at m = 6.0 with the defaults:
        # 4800 * (10^0 - 10^(0.52 * (-1.5 - 6))) = 4800 * (1 - 1.25893e-4)
        # = 4800 - 0.60429 = 4799.3957.
        assert predicted_count(6.0) == pytest.approx(4799.39571, rel=1e-6)

    def test_predicted_count_grows_by_the_slope(self) -> None:
        # One magnitude fainter multiplies the count by 10^0.52 = 3.3113.
        ratio = predicted_count(7.0) / predicted_count(6.0)
        assert ratio == pytest.approx(10.0**0.52, rel=1e-3)

    def test_expected_close_pairs_by_hand(self) -> None:
        # C(100, 2) * (1 - cos(60 deg)) / 2 = 4950 * 0.5 / 2 = 1237.5.
        assert expected_close_pairs(100, np.pi / 3) == pytest.approx(1237.5)

    def test_expected_close_pairs_is_zero_at_zero_separation(self) -> None:
        assert expected_close_pairs(1000, 0.0) == 0.0

    def test_generated_count_matches_the_model(self) -> None:
        cat = generate_catalogue(6.0, seed=1)
        assert cat.n_stars == round(predicted_count(6.0))

    def test_density_is_count_over_four_pi(self) -> None:
        cat = generate_catalogue(5.0, seed=1)
        assert cat.density_per_steradian == pytest.approx(cat.n_stars / (4 * np.pi))
        assert cat.expected_in_solid_angle(4 * np.pi) == pytest.approx(cat.n_stars)


class TestBehaviour:
    def test_determinism(self) -> None:
        a = generate_catalogue(5.0, seed=42)
        b = generate_catalogue(5.0, seed=42)
        assert np.array_equal(a.ra, b.ra)
        assert np.array_equal(a.magnitude, b.magnitude)

    def test_different_seeds_differ(self) -> None:
        a = generate_catalogue(5.0, seed=42)
        b = generate_catalogue(5.0, seed=43)
        assert not np.array_equal(a.ra, b.ra)

    def test_sorted_by_magnitude(self, small_catalogue: StarCatalogue) -> None:
        assert np.all(np.diff(small_catalogue.magnitude) >= 0.0)

    def test_vectors_are_unit(self, small_catalogue: StarCatalogue) -> None:
        assert np.allclose(np.linalg.norm(small_catalogue.vectors, axis=1), 1.0)

    def test_magnitudes_inside_the_limits(self, small_catalogue: StarCatalogue) -> None:
        assert small_catalogue.magnitude.min() >= -1.5
        assert small_catalogue.magnitude.max() <= 5.0

    def test_brighter_than_filters(self, small_catalogue: StarCatalogue) -> None:
        cut = small_catalogue.brighter_than(3.0)
        assert cut.n_stars < small_catalogue.n_stars
        assert cut.magnitude.max() < 3.0
        assert cut.magnitude_limit == 3.0

    def test_stars_within_agrees_with_brute_force(
        self, small_catalogue: StarCatalogue
    ) -> None:
        direction = small_catalogue.vectors[0]
        radius = np.radians(10.0)
        got = set(small_catalogue.stars_within(direction, radius).tolist())
        want = {
            i
            for i in range(small_catalogue.n_stars)
            if angular_separation(small_catalogue.vectors[i], direction)[0] <= radius
        }
        assert got == want

    def test_remove_close_pairs_leaves_nothing_below_the_threshold(self) -> None:
        cat = generate_catalogue(7.0, seed=3)
        sep = np.radians(0.1)
        prepared, removed = remove_close_pairs(cat, sep)
        assert removed > 0
        assert prepared.n_stars == cat.n_stars - removed
        assert prepared.removed_close_pairs == removed
        # both members of every close pair go, so nothing survives below sep
        from scipy.spatial import cKDTree

        tree = cKDTree(prepared.vectors)
        assert tree.query_pairs(2.0 * np.sin(0.5 * sep), output_type="ndarray").size == 0

    def test_remove_close_pairs_noop_at_zero(self, small_catalogue: StarCatalogue) -> None:
        prepared, removed = remove_close_pairs(small_catalogue, 0.0)
        assert removed == 0
        assert prepared is small_catalogue

    def test_generate_with_preparation(self) -> None:
        cat = generate_catalogue(6.5, seed=4, min_separation_rad=np.radians(0.05))
        assert cat.min_separation_rad == pytest.approx(np.radians(0.05))
        assert cat.n_stars <= round(predicted_count(6.5))


class TestValidation:
    def test_rejects_limit_below_minimum(self) -> None:
        with pytest.raises(ValueError, match="must exceed magnitude_min"):
            generate_catalogue(-2.0)

    def test_rejects_absurd_limit(self) -> None:
        with pytest.raises(ValueError, match="exceeds the supported range"):
            generate_catalogue(15.0)

    def test_rejects_non_finite_limit(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            generate_catalogue(np.nan)

    def test_rejects_too_few_stars(self) -> None:
        with pytest.raises(ValueError, match="at least 4"):
            generate_catalogue(-1.0)

    def test_predicted_count_rejects_bad_slope(self) -> None:
        with pytest.raises(ValueError, match="slope"):
            predicted_count(6.0, slope=0.0)

    def test_expected_close_pairs_rejects_negative(self) -> None:
        with pytest.raises(ValueError, match="n_stars"):
            expected_close_pairs(-1, 0.1)
        with pytest.raises(ValueError, match="separation_rad"):
            expected_close_pairs(10, -0.1)

    def test_remove_close_pairs_rejects_negative(
        self, small_catalogue: StarCatalogue
    ) -> None:
        with pytest.raises(ValueError, match="min_separation_rad"):
            remove_close_pairs(small_catalogue, -1.0)

    def test_stars_within_rejects_negative_radius(
        self, small_catalogue: StarCatalogue
    ) -> None:
        with pytest.raises(ValueError, match="radius_rad"):
            small_catalogue.stars_within(np.array([0.0, 0.0, 1.0]), -1.0)

    def test_expected_in_solid_angle_rejects_negative(
        self, small_catalogue: StarCatalogue
    ) -> None:
        with pytest.raises(ValueError, match="solid_angle_sr"):
            small_catalogue.expected_in_solid_angle(-1.0)

"""Equivalence-class and boundary coverage, including the margin reporting."""

from __future__ import annotations

import math

import pytest

from scenariocov import BoundarySpec, EquivalenceClass, boundary_coverage, class_coverage


class TestEquivalenceClassDeclaration:
    def test_interval_membership_is_half_open_by_default(self) -> None:
        cls = EquivalenceClass("x", "mid", lower=1.0, upper=2.0)
        assert cls.contains(1.0)
        assert cls.contains(1.999)
        assert not cls.contains(2.0)
        assert not cls.contains(0.999)

    def test_closed_upper_includes_the_top(self) -> None:
        cls = EquivalenceClass("x", "top", lower=1.0, upper=2.0, closed_upper=True)
        assert cls.contains(2.0)

    def test_categorical_membership(self) -> None:
        cls = EquivalenceClass("light", "degraded", labels=("dusk", "dark"))
        assert cls.contains("dusk")
        assert not cls.contains("sun")
        assert not cls.is_interval

    def test_declaring_both_forms_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="not both"):
            EquivalenceClass("x", "bad", lower=0.0, upper=1.0, labels=("a",))

    def test_declaring_neither_form_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="not both"):
            EquivalenceClass("x", "bad")

    def test_inverted_interval_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="lower < upper"):
            EquivalenceClass("x", "bad", lower=2.0, upper=1.0)

    def test_non_finite_bounds_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            EquivalenceClass("x", "bad", lower=0.0, upper=math.inf)

    def test_round_trip_through_dict(self) -> None:
        for cls in (
            EquivalenceClass("x", "mid", lower=1.0, upper=2.0, closed_upper=True),
            EquivalenceClass("c", "set", labels=("a", "b"), expected=False),
        ):
            assert EquivalenceClass.from_dict(cls.to_dict()) == cls


class TestClassCoverage:
    def test_hand_counted_coverage(self) -> None:
        classes = [
            EquivalenceClass("x", "low", lower=0.0, upper=1.0),
            EquivalenceClass("x", "high", lower=1.0, upper=2.0),
        ]
        records = [{"x": 0.2}, {"x": 0.7}, {"x": 0.9}]
        result = class_coverage(classes, records)
        # 3 records in 'low', 0 in 'high': 1 of 2 classes covered.
        assert [h.count for h in result.hits] == [3, 0]
        assert (result.covered, result.total) == (1, 2)
        assert result.fraction == pytest.approx(0.5, abs=1e-12)
        assert [c.name for c in result.uncovered] == ["high"]

    def test_overlapping_classes_each_count_the_record(self) -> None:
        classes = [
            EquivalenceClass("x", "wide", lower=0.0, upper=10.0),
            EquivalenceClass("x", "narrow", lower=0.0, upper=1.0),
        ]
        result = class_coverage(classes, [{"x": 0.5}])
        assert [h.count for h in result.hits] == [1, 1]

    def test_missing_parameter_raises(self) -> None:
        classes = [EquivalenceClass("x", "low", lower=0.0, upper=1.0)]
        with pytest.raises(ValueError, match="has no parameter"):
            class_coverage(classes, [{"y": 1.0}])

    def test_no_classes_declared_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one equivalence class"):
            class_coverage([], [{"x": 1.0}])

    def test_the_committed_campaign_misses_exactly_the_planted_class(
        self, declared_classes, rendezvous_campaign
    ) -> None:
        classes, _ = declared_classes
        result = class_coverage(classes, rendezvous_campaign)
        # Planted gap G4: nav_position_error >= 1.5 m was never executed.
        assert [(c.parameter, c.name) for c in result.uncovered] == [
            ("nav_position_error", "excessive")
        ]
        assert (result.covered, result.total) == (14, 15)


class TestBoundarySpecDeclaration:
    @pytest.mark.parametrize("tolerance", [0.0, -1.0, math.inf, math.nan])
    def test_invalid_tolerance(self, tolerance: float) -> None:
        with pytest.raises(ValueError, match="tolerance must be positive"):
            BoundarySpec("x", "b", value=1.0, tolerance=tolerance)

    def test_non_finite_value(self) -> None:
        with pytest.raises(ValueError, match="must be finite"):
            BoundarySpec("x", "b", value=math.nan, tolerance=1.0)

    def test_round_trip_through_dict(self) -> None:
        spec = BoundarySpec("x", "b", value=1.0, tolerance=0.1)
        assert BoundarySpec.from_dict(spec.to_dict()) == spec


class TestBoundaryCoverage:
    def test_hand_worked_hit_and_margin(self) -> None:
        spec = BoundarySpec("x", "limit", value=1.0, tolerance=0.1)
        hits = boundary_coverage([spec], [{"x": 1.05}, {"x": 3.0}])
        assert hits[0].count == 1
        assert hits[0].nearest_distance == pytest.approx(0.05, abs=1e-12)
        assert hits[0].margin == pytest.approx(0.05, abs=1e-12)
        assert hits[0].covered

    def test_a_near_miss_reports_a_negative_margin(self) -> None:
        spec = BoundarySpec("x", "limit", value=2.0, tolerance=0.02)
        hits = boundary_coverage([spec], [{"x": 2.021}])
        assert not hits[0].covered
        assert hits[0].margin < 0.0
        # The point of reporting the margin: this is a 1-part-in-20 miss, not a
        # campaign that ignored the boundary.
        assert hits[0].nearest_distance == pytest.approx(0.021, abs=1e-12)

    def test_empty_campaign_gives_an_infinite_nearest_distance(self) -> None:
        spec = BoundarySpec("x", "limit", value=1.0, tolerance=0.1)
        hits = boundary_coverage([spec], [])
        assert hits[0].count == 0
        assert math.isinf(hits[0].nearest_distance)

    def test_missing_parameter_raises(self) -> None:
        spec = BoundarySpec("x", "limit", value=1.0, tolerance=0.1)
        with pytest.raises(ValueError, match="has no parameter"):
            boundary_coverage([spec], [{"y": 1.0}])

    def test_the_committed_campaign_misses_the_planted_boundary(
        self, declared_classes, rendezvous_campaign
    ) -> None:
        _, boundaries = declared_classes
        hits = {h.spec.name: h for h in boundary_coverage(boundaries, rendezvous_campaign)}
        # Planted gap G5: nothing within 2.5 m of the 200 m handover boundary,
        # while the declared tolerance is 2.0 m.
        assert not hits["handover_200m"].covered
        assert hits["handover_200m"].nearest_distance > 2.0

    def test_the_tumble_boundary_is_the_near_tie_the_margin_exists_for(
        self, declared_classes, rendezvous_campaign
    ) -> None:
        # Measured on the committed fixture: nearest approach to the 2.0 deg/s
        # tumble boundary is about 0.021 deg/s against a declared tolerance of
        # 0.020 deg/s, so the hit flag is False by about one part in twenty of
        # the tolerance. The assertion is on the distance, with a tolerance;
        # the boolean is deliberately NOT asserted, because it is a strict
        # comparison sitting next to the threshold.
        _, boundaries = declared_classes
        hits = {h.spec.name: h for h in boundary_coverage(boundaries, rendezvous_campaign)}
        hit = hits["tumble_limit_2p0"]
        assert hit.nearest_distance == pytest.approx(0.021, abs=5e-3)
        assert abs(hit.margin) < 0.5 * hit.spec.tolerance

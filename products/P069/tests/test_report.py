"""The assembled campaign report."""

from __future__ import annotations

import pytest

from scenariocov import Parameter, ScenarioSpace, build_report


class TestBuildReport:
    def test_rejects_an_empty_campaign(self, tiny_space: ScenarioSpace) -> None:
        with pytest.raises(ValueError, match="nothing to report"):
            build_report(tiny_space, [])

    def test_sections_present_on_the_committed_fixture(
        self, rendezvous_space, rendezvous_campaign, declared_classes
    ) -> None:
        classes, boundaries = declared_classes
        report = build_report(rendezvous_space, rendezvous_campaign, 2, classes, boundaries)
        assert report.n_scenarios == 240
        assert report.one_way.covered == report.one_way.total == 23
        assert (report.tway.covered, report.tway.total) == (223, 226)
        assert report.occupancy is not None
        assert report.classes is not None
        assert len(report.boundaries) == 4

    def test_classes_and_boundaries_are_optional(self, rendezvous_space, rendezvous_campaign):
        report = build_report(rendezvous_space, rendezvous_campaign)
        assert report.classes is None
        assert report.boundaries == ()

    def test_occupancy_is_skipped_for_a_purely_categorical_space(self) -> None:
        space = ScenarioSpace(
            [Parameter.categorical("a", ["x", "y"]), Parameter.categorical("b", ["p", "q"])]
        )
        report = build_report(space, [{"a": "x", "b": "p"}])
        assert report.occupancy is None

    def test_explicit_occupancy_parameters(self, rendezvous_space, rendezvous_campaign) -> None:
        report = build_report(
            rendezvous_space, rendezvous_campaign,
            occupancy_parameters=["closing_speed", "lighting"],
        )
        assert report.occupancy is not None
        assert report.occupancy.shape == (4, 3)

    def test_strength_three_is_available(self, rendezvous_space, rendezvous_campaign) -> None:
        report = build_report(rendezvous_space, rendezvous_campaign, strength=3)
        assert report.tway.t == 3
        assert report.tway.total == 1230


class TestRendering:
    @pytest.fixture
    def report(self, rendezvous_space, rendezvous_campaign, declared_classes):
        classes, boundaries = declared_classes
        return build_report(rendezvous_space, rendezvous_campaign, 2, classes, boundaries)

    def test_discretisation_is_stated(self, report) -> None:
        text = report.discretisation_text()
        assert "continuous, 4 bins" in text
        assert "m/s" in text
        assert "categorical, 3 levels" in text

    def test_text_contains_the_headline_numbers(self, report) -> None:
        text = report.to_text()
        assert "223/226" in text
        assert "an assumption imposed by this report" in text
        assert "closing_speed=[0.2,0.5] & lighting=eclipse" in text
        assert "UNCOVERED CLASS  nav_position_error/excessive" in text
        assert "MISS range_at_handover/handover_200m" in text

    def test_uncovered_listing_is_capped_and_says_so(self, tiny_space: ScenarioSpace) -> None:
        report = build_report(tiny_space, [{"mode": "safe", "speed": 0.5, "light": "sun"}])
        text = report.to_text(max_uncovered=2)
        assert "more, available in report.tway.uncovered" in text

    def test_negative_cap_is_rejected(self, report) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            report.to_text(max_uncovered=-1)

    def test_edge_ambiguity_is_mentioned(self, report) -> None:
        assert "within tolerance of a bin edge" in report.to_text()

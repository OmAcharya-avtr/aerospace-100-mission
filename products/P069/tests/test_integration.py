"""End-to-end: declare a space, plant a gap, report it, propose into it, benchmark.

One test walks the whole pipeline on a space small enough that every number in
it is hand-checkable, which is the point: the integration test is not a smoke
test, it asserts the planted gap is found and then closed.
"""

from __future__ import annotations

import numpy as np
import pytest

from scenariocov import (
    BoundarySpec,
    EquivalenceClass,
    GapFinder,
    GreedyMaxMinStrategy,
    LearnedGapFinderStrategy,
    Parameter,
    ScenarioSpace,
    UniformRandomStrategy,
    build_report,
    coverage_benchmark,
    tway_coverage,
)


@pytest.fixture
def planted_case():
    """A 3-parameter space and a campaign that deliberately avoids one corner.

    Space: speed with 3 bins over [0, 3], mode with 2 levels, light with 2
    levels, so |T_2| = 3*2 + 3*2 + 2*2 = 16. The campaign covers every pair
    except (speed in [2,3], light = dark), which is planted empty.
    """
    space = ScenarioSpace(
        [
            Parameter.continuous("speed", [0.0, 1.0, 2.0, 3.0], units="m/s"),
            Parameter.categorical("mode", ["safe", "nominal"]),
            Parameter.categorical("light", ["sun", "dark"]),
        ]
    )
    records = []
    for speed in (0.5, 1.5, 2.5):
        for mode in ("safe", "nominal"):
            for light in ("sun", "dark"):
                if speed > 2.0 and light == "dark":
                    continue  # the planted gap
                records.append({"speed": speed, "mode": mode, "light": light})
    return space, records


class TestPlantedGapIsFoundAndClosed:
    def test_the_report_names_exactly_the_planted_pair(self, planted_case) -> None:
        space, records = planted_case
        report = build_report(space, records, strength=2)
        assert report.tway.total == 16
        assert report.tway.n_uncovered == 1
        assert report.tway.uncovered[0].as_text() == "speed=[2,3] & light=dark"

    def test_class_and_boundary_coverage_see_what_the_tuples_cannot(
        self, planted_case
    ) -> None:
        space, records = planted_case
        classes = [
            EquivalenceClass("speed", "low", lower=0.0, upper=1.0),
            EquivalenceClass("speed", "very_high", lower=2.9, upper=3.0, closed_upper=True),
        ]
        boundaries = [BoundarySpec("speed", "limit_2", value=2.0, tolerance=0.01)]
        report = build_report(space, records, 2, classes, boundaries)
        # No scenario above 2.9 and none within 0.01 of 2.0, and neither fact
        # is visible to the 2-way tuple report, which is fully covered on the
        # speed bins involved.
        assert [c.name for c in report.classes.uncovered] == ["very_high"]
        assert not report.boundaries[0].covered
        assert report.boundaries[0].nearest_distance == pytest.approx(0.5, abs=1e-12)

    def test_the_gap_finder_proposes_into_the_planted_corner(self, planted_case) -> None:
        space, records = planted_case
        pool = [
            {"speed": 2.7, "mode": "safe", "light": "dark"},
            {"speed": 0.4, "mode": "safe", "light": "sun"},
            {"speed": 1.4, "mode": "nominal", "light": "sun"},
        ]
        finder = GapFinder(n_bootstrap=8, random_state=1).fit(space.normalise(records))
        proposals = finder.propose(space.normalise(pool), 1)
        assert proposals[0].candidate_index == 0
        # The uncertainty output exists and is in range.
        assert 0.0 <= proposals[0].sparsity_agreement <= 1.0
        assert proposals[0].log_density_sd >= 0.0

    def test_executing_the_proposal_closes_the_gap(self, planted_case) -> None:
        space, records = planted_case
        extended = [*records, {"speed": 2.7, "mode": "safe", "light": "dark"}]
        result = tway_coverage(space, space.encode_many(extended), t=2)
        assert result.covered == result.total == 16
        assert result.uncovered == ()


class TestPipelineOnTheCommittedFixture:
    def test_report_then_benchmark_runs_end_to_end(
        self, rendezvous_space, rendezvous_campaign, candidate_pool
    ) -> None:
        report = build_report(rendezvous_space, rendezvous_campaign)
        assert report.tway.n_uncovered == 3
        result = coverage_benchmark(
            rendezvous_space,
            rendezvous_campaign,
            candidate_pool,
            [
                UniformRandomStrategy(),
                GreedyMaxMinStrategy(),
                LearnedGapFinderStrategy(n_bootstrap=0),
            ],
            n_start=20,
            n_steps=8,
            pool_size=60,
            seeds=(0, 1, 2, 3),
        )
        for curves in result.by_strategy.values():
            assert curves.curves.shape == (4, 9)
            assert np.all(np.diff(curves.curves, axis=1) >= -1e-12)

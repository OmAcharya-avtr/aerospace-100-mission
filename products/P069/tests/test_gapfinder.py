"""The learned gap finder: fitting, scoring, uncertainty output and validation."""

from __future__ import annotations

import numpy as np
import pytest

from scenariocov import GapFinder


@pytest.fixture
def clustered_training() -> np.ndarray:
    """200 points in the lower-left corner of the unit square, seed 20261016."""
    rng = np.random.default_rng(20261016)
    return rng.uniform(0.0, 0.3, size=(200, 2))


class TestConstruction:
    @pytest.mark.parametrize("bandwidth", ["cross-validated", "", "auto"])
    def test_unknown_bandwidth_string(self, bandwidth: str) -> None:
        with pytest.raises(ValueError, match="scott"):
            GapFinder(bandwidth=bandwidth)

    @pytest.mark.parametrize("bandwidth", [0.0, -1.0, float("inf")])
    def test_invalid_numeric_bandwidth(self, bandwidth: float) -> None:
        with pytest.raises(ValueError, match="positive and finite"):
            GapFinder(bandwidth=bandwidth)

    def test_negative_bootstrap(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            GapFinder(n_bootstrap=-1)

    @pytest.mark.parametrize("quantile", [0.0, -0.1, 1.5])
    def test_invalid_quantile(self, quantile: float) -> None:
        with pytest.raises(ValueError, match=r"\(0, 1\]"):
            GapFinder(sparsity_quantile=quantile)


class TestFitValidation:
    def test_scoring_before_fitting_raises(self) -> None:
        with pytest.raises(RuntimeError, match="call fit"):
            GapFinder().score_candidates(np.zeros((2, 2)))

    def test_one_dimensional_input_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            GapFinder().fit(np.zeros(5))

    def test_one_scenario_is_not_enough(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            GapFinder().fit(np.zeros((1, 3)))

    def test_non_finite_training_data(self) -> None:
        data = np.zeros((4, 2))
        data[0, 0] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            GapFinder().fit(data)

    def test_candidate_width_must_match(self, clustered_training: np.ndarray) -> None:
        finder = GapFinder(n_bootstrap=0).fit(clustered_training)
        with pytest.raises(ValueError, match="candidates must have shape"):
            finder.score_candidates(np.zeros((3, 5)))

    def test_empty_candidate_set(self, clustered_training: np.ndarray) -> None:
        finder = GapFinder(n_bootstrap=0).fit(clustered_training)
        with pytest.raises(ValueError, match="at least one row"):
            finder.score_candidates(np.zeros((0, 2)))

    def test_non_finite_candidates(self, clustered_training: np.ndarray) -> None:
        finder = GapFinder(n_bootstrap=0).fit(clustered_training)
        with pytest.raises(ValueError, match="non-finite"):
            finder.score_candidates(np.array([[np.inf, 0.0]]))


class TestScoring:
    def test_density_is_higher_inside_the_cluster_than_outside(self, clustered_training) -> None:
        finder = GapFinder(n_bootstrap=0).fit(clustered_training)
        scores = finder.score_candidates(np.array([[0.15, 0.15], [0.9, 0.9]]))
        # Inside the cluster versus the far corner. A margin, not an equality.
        assert scores.log_density[0] > scores.log_density[1] + 1.0

    def test_is_fitted_flag(self, clustered_training: np.ndarray) -> None:
        finder = GapFinder(n_bootstrap=0)
        assert not finder.is_fitted
        finder.fit(clustered_training)
        assert finder.is_fitted

    def test_without_bootstrap_the_uncertainty_fields_are_nan(self, clustered_training) -> None:
        finder = GapFinder(n_bootstrap=0).fit(clustered_training)
        scores = finder.score_candidates(np.array([[0.5, 0.5]]))
        assert np.isnan(scores.log_density_sd).all()
        assert np.isnan(scores.sparsity_agreement).all()

    def test_bootstrap_produces_a_finite_spread_and_confidence(self, clustered_training) -> None:
        finder = GapFinder(n_bootstrap=12, random_state=3).fit(clustered_training)
        scores = finder.score_candidates(np.array([[0.15, 0.15], [0.95, 0.95]]))
        assert np.all(np.isfinite(scores.log_density_sd))
        assert np.all(scores.log_density_sd >= 0.0)
        assert np.all((scores.sparsity_agreement >= 0.0) & (scores.sparsity_agreement <= 1.0))
        # The far corner is in the sparse decile of every resample; the cluster
        # centre is in none of them.
        assert scores.sparsity_agreement[1] > scores.sparsity_agreement[0]

    def test_one_bootstrap_replicate_gives_zero_spread(self, clustered_training) -> None:
        finder = GapFinder(n_bootstrap=1).fit(clustered_training)
        scores = finder.score_candidates(np.array([[0.5, 0.5]]))
        assert scores.log_density_sd[0] == pytest.approx(0.0, abs=0.0)

    def test_the_same_seed_reproduces_the_same_scores(self, clustered_training) -> None:
        candidates = np.array([[0.5, 0.5], [0.8, 0.2]])
        a = GapFinder(n_bootstrap=8, random_state=11).fit(clustered_training)
        b = GapFinder(n_bootstrap=8, random_state=11).fit(clustered_training)
        assert a.score_candidates(candidates).log_density == pytest.approx(
            b.score_candidates(candidates).log_density, abs=1e-12
        )


class TestProposals:
    def test_proposals_are_in_the_sparse_region(self, clustered_training: np.ndarray) -> None:
        rng = np.random.default_rng(5)
        candidates = np.vstack([rng.uniform(0.0, 0.3, (40, 2)), rng.uniform(0.7, 1.0, (10, 2))])
        finder = GapFinder(n_bootstrap=0).fit(clustered_training)
        picked = finder.proposed_indices(candidates, 2)
        # Both picks should come from the 10 far-corner candidates, indices
        # 40 and above. Only two are asked for because the minimum-separation
        # rule limits how many well-separated proposals a 0.3-wide corner can
        # hold at a fitted bandwidth of about 0.41.
        assert all(i >= 40 for i in picked)

    def test_min_separation_zero_disables_the_rule(self, clustered_training) -> None:
        rng = np.random.default_rng(5)
        candidates = np.vstack([rng.uniform(0.0, 0.3, (40, 2)), rng.uniform(0.7, 1.0, (10, 2))])
        finder = GapFinder(n_bootstrap=0, min_separation=0.0).fit(clustered_training)
        picked = finder.proposed_indices(candidates, 5)
        assert all(i >= 40 for i in picked)

    def test_negative_min_separation_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="min_separation"):
            GapFinder(min_separation=-0.1)

    def test_fitted_bandwidth_is_reported(self, clustered_training: np.ndarray) -> None:
        finder = GapFinder(n_bootstrap=0, bandwidth=0.25).fit(clustered_training)
        assert finder.fitted_bandwidth == pytest.approx(0.25, abs=1e-12)
        scott = GapFinder(n_bootstrap=0).fit(clustered_training)
        # Scott's rule on 200 points in 2 dimensions: 200 ** (-1 / 6) = 0.4135.
        assert scott.fitted_bandwidth == pytest.approx(200 ** (-1 / 6), rel=1e-6)

    def test_fitted_bandwidth_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError, match="before reading"):
            _ = GapFinder().fitted_bandwidth

    def test_proposals_are_distinct_and_ordered(self, clustered_training: np.ndarray) -> None:
        rng = np.random.default_rng(6)
        candidates = rng.uniform(0.0, 1.0, (30, 2))
        proposals = GapFinder(n_bootstrap=0).fit(clustered_training).propose(candidates, 6)
        indices = [p.candidate_index for p in proposals]
        assert len(set(indices)) == 6
        assert [p.order for p in proposals] == list(range(6))

    def test_non_sequential_mode_takes_the_lowest_density_block(self, clustered_training) -> None:
        rng = np.random.default_rng(7)
        candidates = rng.uniform(0.0, 1.0, (30, 2))
        finder = GapFinder(n_bootstrap=0).fit(clustered_training)
        scores = finder.score_candidates(candidates)
        expected = list(np.argsort(scores.log_density, kind="stable")[:4])
        picked = [p.candidate_index for p in finder.propose(candidates, 4, sequential=False)]
        assert picked == [int(i) for i in expected]

    def test_sequential_mode_spreads_the_batch(self, clustered_training: np.ndarray) -> None:
        # Two distinct empty corners. A sequential batch of 2 must not put both
        # proposals in the same corner.
        candidates = np.array([[0.9, 0.9], [0.91, 0.9], [0.9, 0.91], [0.05, 0.95], [0.9, 0.89]])
        finder = GapFinder(n_bootstrap=0).fit(clustered_training)
        picked = set(finder.proposed_indices(candidates, 2))
        assert 3 in picked

    def test_batch_size_validation(self, clustered_training: np.ndarray) -> None:
        finder = GapFinder(n_bootstrap=0).fit(clustered_training)
        candidates = np.zeros((3, 2))
        for n in (0, 4):
            with pytest.raises(ValueError, match="n_proposals must satisfy"):
                finder.propose(candidates, n)

    def test_density_report_renders_one_row_per_candidate(self, clustered_training) -> None:
        finder = GapFinder(n_bootstrap=4).fit(clustered_training)
        text = finder.density_report(np.array([[0.1, 0.1], [0.9, 0.9]]), ["near", "far"])
        assert text.count("\n") == 2
        assert "near" in text and "far" in text

    def test_density_report_checks_label_count(self, clustered_training: np.ndarray) -> None:
        finder = GapFinder(n_bootstrap=0).fit(clustered_training)
        with pytest.raises(ValueError, match="one entry per candidate"):
            finder.density_report(np.zeros((2, 2)), ["only-one"])


class TestOnTheCommittedFixture:
    def test_the_batch_reaches_the_planted_gaps(
        self, rendezvous_space, rendezvous_campaign, candidate_pool
    ) -> None:
        # The planted gaps G1, G2 and G3 are all in the fast / far / eclipse /
        # camera-plus-failed-thruster corner. A batch of 5 proposals should
        # land at least one scenario in a planted-gap region.
        finder = GapFinder(n_bootstrap=0, random_state=0)
        finder.fit(rendezvous_space.normalise(rendezvous_campaign))
        picked = finder.proposed_indices(rendezvous_space.normalise(candidate_pool), 5)
        chosen = [candidate_pool[i] for i in picked]
        in_gap = [
            r
            for r in chosen
            if (float(r["closing_speed"]) >= 0.20 and r["lighting"] == "eclipse")
            or (r["thruster_health"] == "one_failed" and r["sensor_mode"] == "camera_only")
            or (float(r["closing_speed"]) >= 0.20 and float(r["range_at_handover"]) >= 200.0)
        ]
        assert len(in_gap) >= 1

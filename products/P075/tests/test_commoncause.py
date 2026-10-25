"""The beta-factor rewrite: both limits exact, every refusal tested.

The two limits are design claims rather than approximations. At ``beta = 0``
the rewrite must reproduce the independent diagram's number bit for bit,
because it does not create a common-cause block at all. At ``beta = 1`` the
group must reduce to a single member, because a group in which every failure is
shared is not redundant.
"""

from __future__ import annotations

import math

import pytest

from reliablock.commoncause import (
    ScaledHazard,
    apply_common_cause,
    common_cause_event_name,
    group_beta_effects,
    strip_common_cause,
)
from reliablock.diagram import Block, Diagram, KOfN, Parallel, Series, Standby
from reliablock.distributions import Exponential, FixedProbability, Lognormal, Weibull
from reliablock.quantify import quantify
from reliablock.structure import minimal_cut_sets

from conftest import abs_, exponential_block, rel

MISSION = 10_000.0


class TestScaledHazard:
    def test_scales_the_cumulative_hazard(self) -> None:
        base = Exponential(1.0e-5)
        scaled = ScaledHazard(base=base, factor=0.3)
        assert scaled.cumulative_hazard(MISSION) == rel(
            0.3 * base.cumulative_hazard(MISSION), 1e-15
        )

    def test_factor_zero_never_fails(self) -> None:
        scaled = ScaledHazard(base=Exponential(1.0), factor=0.0)
        assert scaled.reliability(1.0e6) == 1.0
        assert scaled.unreliability(1.0e6) == 0.0

    def test_factor_one_is_the_base(self) -> None:
        base = Weibull(shape=2.0, scale_hours=5000.0)
        scaled = ScaledHazard(base=base, factor=1.0)
        for t in (1.0, 1000.0, 50_000.0):
            assert scaled.cumulative_hazard(t) == base.cumulative_hazard(t)

    def test_density_uses_its_own_survival_not_the_base_s(self) -> None:
        # f_scaled(t) = factor * h_base(t) * R_scaled(t). Multiplying the
        # base's density by the factor would use the base's survival term and
        # be wrong for every factor other than 1.
        base = Exponential(1.0e-4)
        scaled = ScaledHazard(base=base, factor=0.4)
        expected = 0.4 * 1.0e-4 * scaled.reliability(MISSION)
        assert scaled.density(MISSION) == rel(expected, 1e-12)
        # A negative assertion, so the tolerance must be explicit for the same
        # reason: a default absolute tolerance of 1e-12 on densities of order
        # 1e-5 would make "not approximately equal" nearly always true and the
        # check vacuous in the other direction.
        assert scaled.density(MISSION) != rel(0.4 * base.density(MISSION), 1e-6)

    def test_weibull_scaling_has_an_equivalent_parameter_change(self) -> None:
        # Scaling a Weibull's hazard by c is the same as scaling eta by
        # c**(-1/shape). Stated in the docstring, so it is checked.
        base = Weibull(shape=1.7, scale_hours=4000.0)
        factor = 0.35
        scaled = ScaledHazard(base=base, factor=factor)
        equivalent = Weibull(shape=1.7, scale_hours=4000.0 * factor ** (-1.0 / 1.7))
        for t in (10.0, 1000.0, 20_000.0):
            assert scaled.cumulative_hazard(t) == rel(equivalent.cumulative_hazard(t), 1e-12)

    def test_lognormal_scaling_has_no_equivalent_parameter_change(self) -> None:
        # The claim in the docstring, checked rather than asserted: no (mu,
        # sigma) reproduces a scaled lognormal hazard, so the hazard is the
        # right thing to scale.
        base = Lognormal(log_median=9.0, log_sigma=0.5)
        scaled = ScaledHazard(base=base, factor=0.5)
        # A scaled lognormal is not a lognormal: its hazard ratio to the base
        # is constant, which no lognormal pair achieves.
        ratios = [
            scaled.cumulative_hazard(t) / base.cumulative_hazard(t) for t in (100.0, 1e4, 1e6)
        ]
        assert all(r == rel(0.5, 1e-12) for r in ratios)

    def test_inherits_ages_with_time(self) -> None:
        assert ScaledHazard(base=Exponential(1e-5), factor=0.5).ages_with_time is True
        assert ScaledHazard(base=FixedProbability(0.1), factor=0.5).ages_with_time is False

    @pytest.mark.parametrize("bad", [-0.1, 1.1])
    def test_factor_out_of_range_is_rejected(self, bad: float) -> None:
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            ScaledHazard(base=Exponential(1e-5), factor=bad)

    def test_non_distribution_base_is_rejected(self) -> None:
        with pytest.raises(TypeError, match="FailureDistribution"):
            ScaledHazard(base=0.5, factor=0.5)  # type: ignore[arg-type]


class TestLimits:
    def test_beta_zero_creates_no_common_cause_block(self, parallel_pair: Diagram) -> None:
        rewritten = apply_common_cause(parallel_pair)
        assert rewritten.block_names() == parallel_pair.block_names()
        assert quantify(rewritten, MISSION).reliability == quantify(
            parallel_pair, MISSION
        ).reliability

    def test_beta_one_reduces_a_parallel_pair_to_one_member(self) -> None:
        rate = 1.0e-5
        diagram = Diagram(
            "fully shared",
            Parallel(
                items=(exponential_block("a", rate), exponential_block("b", rate)),
                beta_factor=1.0,
                label="pair",
            ),
        )
        assert quantify(apply_common_cause(diagram), MISSION).reliability == rel(
            Exponential(rate).reliability(MISSION), 1e-15
        )

    def test_beta_one_reduces_a_three_of_four_group_to_one_member(self) -> None:
        rate = 1.0e-5
        diagram = Diagram(
            "fully shared",
            KOfN(
                k=3,
                items=tuple(exponential_block(f"u{i}", rate) for i in range(4)),
                beta_factor=1.0,
                label="group",
            ),
        )
        assert quantify(apply_common_cause(diagram), MISSION).reliability == rel(
            Exponential(rate).reliability(MISSION), 1e-15
        )

    def test_the_split_preserves_a_member_s_total_hazard(self) -> None:
        rate, beta = 1.0e-5, 0.17
        diagram = apply_common_cause(
            Diagram(
                "split",
                Parallel(
                    items=(exponential_block("a", rate), exponential_block("b", rate)),
                    beta_factor=beta,
                    label="pair",
                ),
            )
        )
        hazards = {b.name: b.distribution.cumulative_hazard(MISSION) for b in diagram.blocks()}
        total = Exponential(rate).cumulative_hazard(MISSION)
        assert hazards["pair::common_cause"] == rel(beta * total, 1e-15)
        assert hazards["a"] == rel((1.0 - beta) * total, 1e-15)
        assert hazards["a"] + hazards["pair::common_cause"] == rel(total, 1e-15)


class TestStructuralEffect:
    def test_a_beta_factor_adds_exactly_one_single_point_of_failure(
        self, beta_pair: Diagram
    ) -> None:
        before = minimal_cut_sets(strip_common_cause(beta_pair))
        after = minimal_cut_sets(apply_common_cause(beta_pair))
        assert [sorted(c) for c in before] == [["a", "b"]]
        assert [sorted(c) for c in after] == [["pair::common_cause"], ["a", "b"]]

    def test_the_event_is_named_after_the_group_label(self, beta_pair: Diagram) -> None:
        node = beta_pair.root
        assert common_cause_event_name(node, 0) == "pair::common_cause"

    def test_an_unlabelled_group_gets_a_deterministic_name(self) -> None:
        diagram = Diagram(
            "unlabelled",
            Parallel(
                items=(exponential_block("a", 1e-5), exponential_block("b", 1e-5)),
                beta_factor=0.1,
            ),
        )
        names = apply_common_cause(diagram).block_names()
        assert "parallel#0::common_cause" in names

    def test_strip_preserves_everything_except_the_beta(self, beta_pair: Diagram) -> None:
        stripped = strip_common_cause(beta_pair)
        assert stripped.declared_betas == {}
        assert stripped.block_names() == beta_pair.block_names()

    def test_the_rewrite_does_not_mutate_the_input(self, beta_pair: Diagram) -> None:
        before = beta_pair.to_yaml()
        apply_common_cause(beta_pair)
        strip_common_cause(beta_pair)
        assert beta_pair.to_yaml() == before


class TestRefusals:
    def test_non_identical_members_are_refused(self) -> None:
        diagram = Diagram(
            "mixed group",
            Parallel(
                items=(exponential_block("a", 1.0e-5), exponential_block("b", 2.0e-5)),
                beta_factor=0.1,
                label="pair",
            ),
        )
        with pytest.raises(ValueError, match="do not share one distribution"):
            apply_common_cause(diagram)

    def test_the_refusal_names_the_alternative_model(self) -> None:
        diagram = Diagram(
            "mixed group",
            Parallel(
                items=(exponential_block("a", 1.0e-5), exponential_block("b", 2.0e-5)),
                beta_factor=0.1,
                label="pair",
            ),
        )
        with pytest.raises(ValueError, match="alpha-factor or multiple-Greek-letter"):
            apply_common_cause(diagram)

    def test_nested_members_are_refused(self) -> None:
        diagram = Diagram(
            "nested group",
            Parallel(
                items=(
                    Series(
                        items=(exponential_block("a1", 1e-5), exponential_block("a2", 1e-5))
                    ),
                    Series(
                        items=(exponential_block("b1", 1e-5), exponential_block("b2", 1e-5))
                    ),
                ),
                beta_factor=0.1,
                label="branches",
            ),
        )
        with pytest.raises(ValueError, match="nested structure"):
            apply_common_cause(diagram)

    def test_a_standby_node_cannot_declare_a_beta(self) -> None:
        # The declaration simply has no such field: a standby node is already a
        # dependent-failure model.
        with pytest.raises(TypeError):
            Standby(
                primary=exponential_block("p", 1e-5),
                standbys=(exponential_block("s", 1e-5),),
                switch_reliability=0.99,
                beta_factor=0.1,  # type: ignore[call-arg]
            )

    def test_identical_weibull_members_are_accepted(self) -> None:
        shared = Weibull(shape=1.9, scale_hours=6000.0)
        diagram = Diagram(
            "weibull pair",
            Parallel(
                items=(
                    Block(name="a", distribution=shared),
                    Block(name="b", distribution=Weibull(shape=1.9, scale_hours=6000.0)),
                ),
                beta_factor=0.2,
                label="pair",
            ),
        )
        result = quantify(apply_common_cause(diagram), MISSION)
        assert 0.0 < result.reliability < 1.0


class TestGroupEffects:
    def test_a_parallel_group_is_reported_conservative(self, beta_pair: Diagram) -> None:
        effects = group_beta_effects(beta_pair, MISSION)
        assert len(effects) == 1
        assert effects[0].conservative is True
        assert effects[0].ratio > 1.0
        assert effects[0].k == 1
        assert effects[0].members == 2

    def test_a_k_of_n_group_can_be_reported_non_conservative(self) -> None:
        # The known inadequacy of the beta-factor model for groups larger than
        # two, surfaced on the declaration rather than left in a document. The
        # hazard here is 0.3 per member, which is past the crossing measured in
        # validation/validate_commoncause_output.txt.
        rate = 3.0e-4
        hours = 1000.0
        diagram = Diagram(
            "three of four",
            KOfN(
                k=3,
                items=tuple(exponential_block(f"u{i}", rate) for i in range(4)),
                beta_factor=0.1,
                label="group",
            ),
        )
        effect = group_beta_effects(diagram, hours)[0]
        assert effect.conservative is False
        assert effect.ratio < 1.0
        assert "DECREASES" in str(effect)

    def test_no_declared_beta_gives_no_effects(self, parallel_pair: Diagram) -> None:
        assert group_beta_effects(parallel_pair, MISSION) == ()

    def test_ratio_is_infinite_when_the_independent_figure_is_zero(self) -> None:
        diagram = Diagram(
            "never fails",
            Parallel(
                items=(exponential_block("a", 0.0), exponential_block("b", 0.0)),
                beta_factor=0.1,
                label="pair",
            ),
        )
        assert group_beta_effects(diagram, MISSION)[0].ratio == math.inf


class TestSystemLevelComparison:
    def test_the_two_diagrams_differ_only_in_the_assumption(self) -> None:
        from reliablock.cases import case, mission

        diagram = case("smallsat_bus")
        independent = strip_common_cause(diagram)
        with_cc = apply_common_cause(diagram)
        assert independent.block_names() == diagram.block_names()
        extra = set(with_cc.block_names()) - set(diagram.block_names())
        assert extra == {
            "power::common_cause",
            "reaction_wheels::common_cause",
            "receivers::common_cause",
        }
        q_ind = quantify(independent, mission("three_year")).unreliability
        q_cc = quantify(with_cc, mission("three_year")).unreliability
        assert q_cc > q_ind

    def test_the_notes_record_what_was_assumed(self, beta_pair: Diagram) -> None:
        assert any("independent" in n for n in strip_common_cause(beta_pair).notes)
        assert any("common_cause" in n for n in apply_common_cause(beta_pair).notes)

    def test_a_fixed_probability_member_splits_too(self) -> None:
        diagram = Diagram(
            "one shot pair",
            Parallel(
                items=(
                    Block(name="a", distribution=FixedProbability(1e-3)),
                    Block(name="b", distribution=FixedProbability(1e-3)),
                ),
                beta_factor=0.25,
                label="pair",
            ),
        )
        result = quantify(apply_common_cause(diagram), 0.0)
        independent = quantify(strip_common_cause(diagram), 0.0)
        assert result.unreliability > independent.unreliability
        # The split preserves the member's total hazard.
        hazards = {
            b.name: b.distribution.cumulative_hazard(0.0)
            for b in apply_common_cause(diagram).blocks()
        }
        total = FixedProbability(1e-3).cumulative_hazard(0.0)
        assert hazards["a"] + hazards["pair::common_cause"] == abs_(total, 1e-18)

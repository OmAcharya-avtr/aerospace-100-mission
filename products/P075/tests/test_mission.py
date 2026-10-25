"""Mission profiles: the effective-age model, and what it refuses to age."""

from __future__ import annotations

import math

import pytest

from reliablock.diagram import Diagram, Series
from reliablock.distributions import Exponential, Weibull
from reliablock.mission import MissionPhase, MissionProfile, load_mission
from reliablock.quantify import quantify

from conftest import abs_, exponential_block, fixed_block, rel


def _two_block_diagram() -> Diagram:
    return Diagram(
        "two blocks",
        Series(items=(exponential_block("a", 1.0e-5), exponential_block("b", 2.0e-5))),
    )


class TestEffectiveAge:
    def test_single_phase_is_elapsed_time(self) -> None:
        profile = MissionProfile.from_hours(1000.0)
        assert profile.effective_ages(_two_block_diagram()) == {"a": 1000.0, "b": 1000.0}

    def test_duty_cycle_scales_the_age(self) -> None:
        profile = MissionProfile(
            phases=(MissionPhase("p", 1000.0, duty_cycles={"a": 0.25}),)
        )
        ages = profile.effective_ages(_two_block_diagram())
        assert ages["a"] == rel(250.0, 1e-15)
        assert ages["b"] == rel(1000.0, 1e-15)

    def test_stress_factor_scales_the_age(self) -> None:
        profile = MissionProfile(
            phases=(MissionPhase("p", 1000.0, stress_factors={"a": 3.0}),)
        )
        assert profile.effective_ages(_two_block_diagram())["a"] == rel(3000.0, 1e-15)

    def test_duty_and_stress_multiply(self) -> None:
        profile = MissionProfile(
            phases=(
                MissionPhase("p", 1000.0, duty_cycles={"a": 0.5}, stress_factors={"a": 3.0}),
            )
        )
        assert profile.effective_ages(_two_block_diagram())["a"] == rel(1500.0, 1e-15)

    def test_phases_accumulate(self) -> None:
        profile = MissionProfile(
            phases=(
                MissionPhase("boost", 10.0, stress_factors={"a": 5.0}),
                MissionPhase("cruise", 990.0, duty_cycles={"a": 0.1}),
            )
        )
        # 10 * 5 + 990 * 0.1 = 50 + 99 = 149
        assert profile.effective_ages(_two_block_diagram())["a"] == rel(149.0, 1e-14)
        assert profile.total_hours == rel(1000.0, 1e-15)

    def test_default_duty_cycle_applies_to_unnamed_blocks(self) -> None:
        profile = MissionProfile(
            phases=(MissionPhase("p", 1000.0, default_duty_cycle=0.5, duty_cycles={"a": 1.0}),)
        )
        ages = profile.effective_ages(_two_block_diagram())
        assert ages["a"] == rel(1000.0, 1e-15)
        assert ages["b"] == rel(500.0, 1e-15)

    def test_the_model_is_exact_for_an_exponential_block(self) -> None:
        # For a constant hazard rate the effective-age sum IS the integral of
        # the hazard rate over the mission, so the model is exact. That is
        # asserted here and the non-exponential case is measured below.
        rate = 1.0e-5
        phases = (
            MissionPhase("a", 100.0, stress_factors={"x": 2.0}),
            MissionPhase("b", 900.0, duty_cycles={"x": 0.5}),
        )
        profile = MissionProfile(phases=phases)
        diagram = Diagram("one", Series(items=(exponential_block("x", rate),)))
        hazard = rate * (100.0 * 2.0 + 900.0 * 0.5)
        assert quantify(diagram, profile).hazard == rel(hazard, 1e-14)

    def test_the_model_is_an_approximation_for_a_weibull_block(self) -> None:
        # Cumulative-exposure (Nelson) ageing: the block advances along its own
        # hazard curve at a scaled rate. That is NOT the same as integrating a
        # time-varying hazard rate in calendar time, and the two differ
        # whenever the shape is not 1. The difference is reported, not assumed
        # small.
        from reliablock.diagram import Block

        shape, scale = 2.5, 5000.0
        block = Block(name="x", distribution=Weibull(shape=shape, scale_hours=scale))
        diagram = Diagram("one", Series(items=(block,)))
        profile = MissionProfile(
            phases=(
                MissionPhase("hot", 500.0, stress_factors={"x": 4.0}),
                MissionPhase("cold", 500.0, stress_factors={"x": 1.0}),
            )
        )
        cumulative_exposure = quantify(diagram, profile).hazard
        # Calendar-time integration of the scaled hazard RATE instead:
        # integral of s(t) * h(t) dt with h(t) = (shape/scale)(t/scale)^(shape-1)
        from scipy.integrate import quad

        def rate(t: float) -> float:
            stress = 4.0 if t < 500.0 else 1.0
            return stress * (shape / scale) * (t / scale) ** (shape - 1.0)

        calendar, _ = quad(rate, 0.0, 1000.0, points=[500.0])
        assert cumulative_exposure != rel(calendar, 1e-3)
        # Both are positive and of the same order; the point is that they
        # differ, and which one is meant has to be declared.
        assert 0.0 < cumulative_exposure
        assert 0.0 < calendar


class TestOneShotRefusal:
    def test_a_duty_cycle_on_a_one_shot_block_is_refused(self) -> None:
        diagram = Diagram("deploy", Series(items=(fixed_block("d", 1e-3),)))
        profile = MissionProfile(phases=(MissionPhase("p", 100.0, duty_cycles={"d": 0.5}),))
        with pytest.raises(ValueError, match="not a model of anything"):
            profile.effective_ages(diagram)

    def test_a_stress_factor_on_a_one_shot_block_is_refused(self) -> None:
        diagram = Diagram("deploy", Series(items=(fixed_block("d", 1e-3),)))
        profile = MissionProfile(phases=(MissionPhase("p", 100.0, stress_factors={"d": 2.0}),))
        with pytest.raises(ValueError, match="probability per"):
            profile.effective_ages(diagram)

    def test_a_one_shot_block_is_otherwise_unaffected_by_the_profile(self) -> None:
        diagram = Diagram("deploy", Series(items=(fixed_block("d", 1.5e-3),)))
        for hours in (0.0, 100.0, 1e6):
            assert quantify(diagram, hours).unreliability == rel(1.5e-3, 1e-14)


class TestValidation:
    def test_a_profile_with_no_phases_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="no phases"):
            MissionProfile(phases=())

    def test_a_negative_duration_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="duration_hours must be >= 0"):
            MissionPhase("p", -1.0)

    def test_an_empty_phase_name_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty string"):
            MissionPhase("  ", 1.0)

    @pytest.mark.parametrize("bad", [-0.1, 1.1])
    def test_a_duty_cycle_out_of_range_is_rejected(self, bad: float) -> None:
        with pytest.raises(ValueError, match="duty cycle"):
            MissionPhase("p", 1.0, duty_cycles={"a": bad})

    def test_a_negative_stress_factor_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="stress factor"):
            MissionPhase("p", 1.0, stress_factors={"a": -1.0})

    @pytest.mark.parametrize("bad", [-0.1, 1.1])
    def test_a_default_duty_cycle_out_of_range_is_rejected(self, bad: float) -> None:
        with pytest.raises(ValueError, match="default_duty_cycle"):
            MissionPhase("p", 1.0, default_duty_cycle=bad)

    def test_a_negative_from_hours_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be >= 0"):
            MissionProfile.from_hours(-1.0)


class TestDeclarationLoading:
    def test_round_trip_through_yaml(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        source = MissionProfile(
            name="phased",
            phases=(
                MissionPhase("boost", 10.0, stress_factors={"a": 5.0}),
                MissionPhase("cruise", 990.0, duty_cycles={"a": 0.1}, default_duty_cycle=0.8),
            ),
        )
        path = tmp_path / "m.yaml"
        path.write_text(source.to_yaml())
        assert load_mission(path).to_yaml() == source.to_yaml()

    def test_unknown_top_level_key_is_rejected(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        path = tmp_path / "m.yaml"
        path.write_text("name: x\nphases: []\nextra: 1\n")
        with pytest.raises(ValueError, match="does not accept"):
            load_mission(path)

    def test_unknown_phase_key_is_rejected(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        path = tmp_path / "m.yaml"
        path.write_text(
            "phases:\n  - name: p\n    duration_hours: 1.0\n    duty_cycel: {a: 0.5}\n"
        )
        with pytest.raises(ValueError, match="does not accept"):
            load_mission(path)

    def test_missing_phases_is_rejected(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        path = tmp_path / "m.yaml"
        path.write_text("name: x\n")
        with pytest.raises(ValueError, match="requires 'phases'"):
            load_mission(path)

    def test_scalar_document_is_rejected(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        path = tmp_path / "m.yaml"
        path.write_text("a string\n")
        with pytest.raises(ValueError, match="mapping at the top level"):
            load_mission(path)


class TestBreakdown:
    def test_breakdown_has_one_row_per_phase_and_block(self) -> None:
        profile = MissionProfile(
            phases=(MissionPhase("a", 1.0), MissionPhase("b", 2.0))
        )
        rows = profile.breakdown(_two_block_diagram())
        assert len(rows) == 4
        assert {r["phase"] for r in rows} == {"a", "b"}
        assert {r["block"] for r in rows} == {"a", "b"}

    def test_breakdown_effective_hours_sum_to_the_effective_age(self) -> None:
        profile = MissionProfile(
            phases=(
                MissionPhase("a", 10.0, stress_factors={"a": 5.0}),
                MissionPhase("b", 990.0, duty_cycles={"a": 0.1}),
            )
        )
        diagram = _two_block_diagram()
        rows = profile.breakdown(diagram)
        total = sum(r["effective_hours"] for r in rows if r["block"] == "a")
        assert total == abs_(profile.effective_ages(diagram)["a"], 1e-12)

    def test_zero_duration_mission_is_certain_success(self) -> None:
        diagram = _two_block_diagram()
        assert quantify(diagram, MissionProfile.from_hours(0.0)).reliability == 1.0

    def test_exponential_mttf_sanity(self) -> None:
        # At t = MTTF the reliability of a single exponential block is 1/e.
        rate = 1.0e-5
        diagram = Diagram("one", Series(items=(exponential_block("x", rate),)))
        assert quantify(diagram, Exponential(rate).mttf_hours()).reliability == rel(
            1.0 / math.e, 1e-12
        )

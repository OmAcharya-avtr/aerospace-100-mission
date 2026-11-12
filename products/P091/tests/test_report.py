"""The assembled analysis and its rendering."""

from __future__ import annotations

import pytest

from thermalbudget.provenance import Origin
from thermalbudget.report import analyse, render


@pytest.fixture(scope="module")
def analysis(no_throttle_campaign):
    return analyse(no_throttle_campaign, n_resamples=2000)


def test_the_analysis_runs_every_build_step_in_order(analysis):
    assert analysis.baseline1 is not None
    assert analysis.baseline2 is not None
    assert analysis.baseline3 is not None
    assert analysis.headline is not None
    assert analysis.learned is not None
    assert analysis.learned_forest is not None
    assert analysis.verdict is not None
    assert analysis.origin is Origin.SYNTHETIC


def test_every_run_including_idle_and_cooldown_gets_an_equilibrium_verdict(
    analysis, no_throttle_campaign
):
    assert set(analysis.equilibrium) == set(no_throttle_campaign.traces)


def test_the_analysis_can_skip_the_learned_step(no_throttle_campaign):
    classical = analyse(no_throttle_campaign, n_resamples=500, run_learned=False)
    assert classical.learned is None
    assert classical.verdict is None
    assert any("skipped by request" in note for note in classical.notes)


def test_the_analysis_refuses_a_campaign_without_the_runs_it_needs(
    no_throttle_campaign, tmp_path
):
    from thermalbudget.trace import Campaign

    only_bursts = Campaign(
        directory=tmp_path,
        origin=Origin.SYNTHETIC,
        manifest={},
        traces={
            label: no_throttle_campaign[label]
            for label in ("burst_1.00_a", "burst_1.00_b")
        },
    )
    with pytest.raises(ValueError, match="no duty-1.0 soak"):
        analyse(only_bursts, n_resamples=100)
    only_soaks = Campaign(
        directory=tmp_path,
        origin=Origin.SYNTHETIC,
        manifest={},
        traces={label: no_throttle_campaign[label] for label in no_throttle_campaign.traces
                if label.startswith("soak_1.00")},
    )
    with pytest.raises(ValueError, match="no burst"):
        analyse(only_soaks, n_resamples=100)


def test_the_render_states_the_origin_in_the_header_and_on_the_headline(analysis):
    text = render(analysis)
    assert "origin           : SYNTHETIC" in text
    assert "not a measurement of anything" in text
    assert "DOES NOT MATCH" in text  # synthetic manifest is not the declared board


def test_the_render_contains_every_section_a_reader_needs(analysis):
    text = render(analysis)
    for heading in (
        "STEP 1  equilibrium test",
        "STEP 2  baseline 1",
        "STEP 3  baseline 2",
        "STEP 4  baseline 3",
        "HEADLINE  cold-burst overstatement",
        "STEP 5  learned thermal-state predictor",
        "STRUCTURALLY UNMEASURABLE in this campaign",
        "NOTES AND REFUSALS",
        "UNMEASURED, BY NAME",
    ):
        assert heading in text


def test_the_render_prints_the_unmeasurable_ambient_axis_with_its_reason(analysis):
    text = render(analysis)
    assert "held_out_ambient" in text
    assert "not approximated by the held-out duty-cycle split" in text


def test_the_render_names_every_unmeasured_claim(analysis):
    from thermalbudget import limits

    text = render(analysis)
    for claim in limits.UNMEASURED_BY_NAME:
        assert claim.key in text


def test_the_render_carries_the_safety_statement(analysis):
    text = render(analysis)
    assert "not flight-qualified, not certified and not approved" in text
    for forbidden in ("flight-safe", "mission-ready", "production-ready", "certified for"):
        assert forbidden not in text


def test_the_render_states_the_run_level_granularity(analysis):
    text = render(analysis)
    assert "distinct multisets" in text


def test_the_render_states_the_degenerate_clock_relation(analysis):
    text = render(analysis)
    assert "DEGENERATE" in text
    assert "reduces to baseline 1" in text


def test_the_throttled_render_states_an_identifiable_relation(throttled_campaign):
    text = render(analyse(throttled_campaign, n_resamples=500, run_learned=False))
    assert "IDENTIFIABLE" in text
    assert "reduces to baseline 1" not in text

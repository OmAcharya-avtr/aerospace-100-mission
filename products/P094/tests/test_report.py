"""The headline assembly, the verdicts, and the scope statement that travels with it.

SPEC §6 requires the unpinned clock to be stated WITH the headline rather than
in a limitations list. The only way to make that true of every surface is to
have one string and quote it, so these tests assert that the one string says
what it must and that the headline carries it.
"""

from __future__ import annotations

import pytest

from hiltimingdiff.report import (
    MEASUREMENT_MODEL,
    RESEARCH_GRADE,
    SCOPE_LINE,
    Verdict,
    headline_lines,
    spread_verdict,
    transfer_verdict,
)

MARKETING_WORDS = (
    "revolutionary",
    "cutting-edge",
    "powerful",
    "seamless",
    "game-changing",
    "world-class",
    "best-in-class",
    "effortless",
    "blazing",
)

FORBIDDEN_CLAIMS = (
    "flight-safe",
    "flight safe",
    "flight-qualified for",
    "certified for",
    "mission-ready",
    "mission ready",
    "production-ready",
    "production ready",
)


def test_scope_line_names_the_unpinned_governor():
    assert "schedutil" in SCOPE_LINE
    assert "unpinned" in SCOPE_LINE


def test_scope_line_names_the_power_mode():
    assert "MAXN_SUPER" in SCOPE_LINE


def test_scope_line_names_the_measured_drift():
    assert "7.52" in SCOPE_LINE


def test_measurement_model_states_the_unpinned_clock_in_capitals():
    assert "EVERY\nHARDWARE FIGURE HERE IS A FIGURE AT AN UNPINNED" in MEASUREMENT_MODEL


def test_measurement_model_names_the_subsampling_of_the_raw_table():
    assert "subsampled every third frame" in MEASUREMENT_MODEL


def test_measurement_model_names_the_drift_ratio_to_full_precision():
    assert "1.0752408447278534" in MEASUREMENT_MODEL


def test_measurement_model_names_the_frame_counts():
    assert "90000" in MEASUREMENT_MODEL
    assert "810000" in MEASUREMENT_MODEL


def test_measurement_model_names_the_campaign_seed():
    assert "20261115" in MEASUREMENT_MODEL


def test_measurement_model_says_simulation_figures_move_between_runs():
    assert "move between runs" in MEASUREMENT_MODEL


def test_research_grade_statement_is_the_required_one():
    assert RESEARCH_GRADE == (
        "Research-grade. Not flight-qualified, not certified, and not approved for "
        "operational aerospace use."
    )


@pytest.mark.parametrize("word", MARKETING_WORDS)
def test_measurement_model_carries_no_marketing_word(word):
    assert word not in MEASUREMENT_MODEL.lower()


@pytest.mark.parametrize("word", MARKETING_WORDS)
def test_scope_line_carries_no_marketing_word(word):
    assert word not in SCOPE_LINE.lower()


def test_measurement_model_carries_no_exclamation_mark():
    assert "!" not in MEASUREMENT_MODEL


def test_scope_line_carries_no_exclamation_mark():
    assert "!" not in SCOPE_LINE


# --- verdicts -------------------------------------------------------------


def test_verdict_label_for_a_held_expectation():
    verdict = Verdict(
        name="x", statement="y", threshold=1.0, measured=2.0, direction=">", held=True
    )
    assert verdict.label == "CONFIRMED"


def test_verdict_label_for_a_refuted_expectation():
    verdict = Verdict(
        name="x", statement="y", threshold=1.0, measured=0.5, direction=">", held=False
    )
    assert verdict.label == "REFUTED"


def test_verdict_line_names_the_threshold_and_the_measurement():
    verdict = Verdict(
        name="n", statement="s", threshold=3.0, measured=1114.86, direction=">", held=True
    )
    line = verdict.line()
    assert "CONFIRMED" in line
    assert "3" in line
    assert "1114.86" in line


def test_spread_verdict_is_confirmed_on_this_campaign(divergence_result):
    verdict = spread_verdict(divergence_result)
    assert verdict.held is True
    assert verdict.label == "CONFIRMED"
    assert verdict.threshold == 3.0


def test_spread_verdict_measures_the_spread(divergence_result):
    verdict = spread_verdict(divergence_result)
    assert verdict.measured == divergence_result.spread


def test_spread_verdict_quotes_the_specification_sentence(divergence_result):
    assert "stage-dependently wrong" in spread_verdict(divergence_result).statement


def test_transfer_verdict_is_refuted_on_this_campaign(baseline2):
    # SPEC §7's SECONDARY pre-registration predicted a held-out transfer error
    # above 0.5. The measured error is far below it, so the expectation is
    # REFUTED and published as refuted. Pinned here so it cannot be softened.
    # The boolean is a strict threshold comparison on a live wall-clock
    # measurement, so the BOUND is asserted first and the boolean is asserted
    # as the consistent function of it rather than as a literal of its own.
    # Measured band published in validation/VALIDATION.md.
    verdict = transfer_verdict(baseline2)
    assert verdict.threshold == 0.5
    assert verdict.measured < verdict.threshold
    assert verdict.held == (verdict.measured > verdict.threshold)
    assert verdict.label == "REFUTED"


def test_transfer_verdict_measures_the_declared_split_error(baseline2):
    assert transfer_verdict(baseline2).measured == baseline2.declared.mean_relative_error


def test_transfer_verdict_quotes_the_specification_sentence(baseline2):
    assert "transfer poorly" in transfer_verdict(baseline2).statement


# --- the rendered headline ------------------------------------------------


@pytest.fixture(scope="module")
def lines(
    campaign,
    units,
    simulation,
    ns_per_op,
    divergence_result,
    baseline1,
    baseline2,
    baseline3,
    comparison,
    confound,
    covariation,
    census,
):
    from hiltimingdiff.report import Headline
    from hiltimingdiff.sources import SOURCE_MD5

    headline = Headline(
        campaign=campaign,
        units=units,
        simulation=simulation,
        ns_per_op=ns_per_op,
        divergence=divergence_result,
        baseline1=baseline1,
        baseline2=baseline2,
        baseline3=baseline3,
        drift=comparison,
        confounding=confound,
        frequency=covariation,
        gate=census,
        spread_verdict=spread_verdict(divergence_result),
        transfer_verdict=transfer_verdict(baseline2),
        source_md5=dict(SOURCE_MD5),
    )
    return headline_lines(headline)


def test_headline_begins_with_the_product_name(lines):
    assert lines[0] == "HEADLINE -- P094 HilTimingDiff"


def test_headline_states_the_scope_on_its_second_line(lines):
    assert lines[1] == SCOPE_LINE


def test_headline_ends_with_the_research_grade_statement(lines):
    assert lines[-1] == RESEARCH_GRADE


@pytest.mark.parametrize("number", ["1.", "2.", "3.", "4.", "5."])
def test_headline_carries_all_five_spec_outputs(lines, number):
    assert any(line.startswith(number) for line in lines)


@pytest.mark.parametrize(
    "stage", ["ingest", "background", "threshold", "centroid", "gate", "track"]
)
def test_headline_names_every_stage(lines, stage):
    assert any(stage in line for line in lines)


def test_headline_names_the_resampling_unit(lines):
    assert any("pass-unit" in line for line in lines)


def test_headline_reports_both_verdicts(lines, divergence_result, baseline2):
    # Both pre-registrations are NAMED, and each label is asserted to be the one
    # its own measurement implies. "REFUTED" was previously asserted as a bare
    # substring, which made it an assertion about a strict threshold comparison
    # on a live wall-clock measurement.
    text = "\n".join(lines)
    assert "SPEC §7 primary (divergence spread)" in text
    assert "SPEC §7 secondary (baseline 2 held-out transfer error)" in text
    assert spread_verdict(divergence_result).label in text
    assert transfer_verdict(baseline2).label in text


def test_headline_carries_no_exclamation_mark(lines):
    assert "!" not in "\n".join(lines)


@pytest.mark.parametrize("word", MARKETING_WORDS)
def test_headline_carries_no_marketing_word(lines, word):
    assert word not in "\n".join(lines).lower()


@pytest.mark.parametrize("claim", FORBIDDEN_CLAIMS)
def test_headline_makes_no_forbidden_claim(lines, claim):
    text = "\n".join(lines).lower()
    if claim in ("certified for",):
        assert "not certified" in text
    assert claim not in text.replace("not certified", "")


def test_headline_names_the_declared_and_measured_orders(lines):
    text = "\n".join(lines)
    assert "declared order" in text
    assert "measured order" in text


def test_headline_reports_the_permutation(lines):
    assert any("permutation" in line for line in lines)

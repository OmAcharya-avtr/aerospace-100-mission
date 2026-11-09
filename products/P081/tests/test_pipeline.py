"""The ordered pipeline and the learned study, on a small declared split."""

from __future__ import annotations

import numpy as np
import pytest

from twindiverge.identifiability import analyse
from twindiverge.pipeline import (
    METHODS,
    analyse_case,
    learned_study,
    montecarlo_directions,
)
from twindiverge.plant import NOMINAL, PARAMETER_NAMES, simulate_with_sensitivities
from twindiverge.synthetic import generate_cases
from twindiverge.telemetry import TELEMETRY_SETS

TRAIN = tuple(range(8000, 8060))
TEST = tuple(range(8100, 8112))
NULL = tuple(range(8200, 8212))


@pytest.fixture(scope="module")
def study(short_set):
    """One learned study for every test that only reads it.

    Fitting two classifiers and analysing five populations costs about a second;
    the tests below read different parts of the same result and there is nothing
    for a shared instance to leak, because none of them mutates it.
    """
    return learned_study(short_set, TRAIN, TEST, NULL)


def test_a_single_parameter_case_is_detected_refused_and_declared_ambiguous(short_set):
    case = generate_cases((8300,), "single")[0]
    analysis = analyse_case(short_set, case)
    assert analysis.attribution.detected
    assert analysis.gated_answer == "REFUSED"
    assert analysis.ambiguity_declared
    assert analysis.truth_name == PARAMETER_NAMES[case.label]


def test_the_short_record_confuses_the_viscous_and_drag_coefficients(short_set):
    """Case 8300 moves ``b``. On the 20 s record the constrained route answers
    ``c_d`` instead, because the two sensitivity columns are collinear to the
    precision the record supports: this is the practical unidentifiability that
    the WEAK verdict of baseline 1 reports for the same set. The error is pinned
    here rather than smoothed over, and the same case on the 200 s record is the
    test below."""
    case = generate_cases((8300,), "single")[0]
    assert PARAMETER_NAMES[case.label] == "b"
    assert analyse_case(short_set, case).constrained_answer == "c_d"


def test_the_long_record_gets_the_same_case_right(full_set):
    case = generate_cases((8300,), "single")[0]
    assert analyse_case(full_set, case).constrained_answer == PARAMETER_NAMES[case.label]


def test_a_none_case_is_reported_as_no_attributable_change(short_set):
    case = generate_cases((8301,), "none")[0]
    analysis = analyse_case(short_set, case)
    assert not analysis.attribution.detected
    assert analysis.gated_answer == "UNIDENTIFIABLE"
    assert analysis.truth_name == "UNIDENTIFIABLE"


def test_a_scale_case_is_reported_as_no_attributable_change(short_set):
    case = generate_cases((8302,), "scale")[0]
    analysis = analyse_case(short_set, case)
    assert not analysis.attribution.detected
    assert analysis.gated_answer == "UNIDENTIFIABLE"


def test_a_full_rank_set_answers_instead_of_refusing(torque_set):
    case = generate_cases((8303,), "single")[0]
    analysis = analyse_case(torque_set, case)
    assert analysis.gated_answer == PARAMETER_NAMES[case.label]
    assert not analysis.ambiguity_declared


def test_the_cached_report_and_twin_change_nothing(short_set, short_report, short_twin):
    case = generate_cases((8304,), "single")[0]
    slow = analyse_case(short_set, case)
    fast = analyse_case(short_set, case, short_report, short_twin)
    assert slow.features == pytest.approx(fast.features, rel=0.0, abs=0.0)
    assert slow.attribution.residual_chi2 == pytest.approx(
        fast.attribution.residual_chi2, rel=0.0, abs=0.0
    )
    assert slow.gated_answer == fast.gated_answer


def test_the_learned_study_scores_every_method_on_every_population(study):
    assert study.train_cases == len(TRAIN)
    assert study.test_cases == len(TEST)
    assert {block.population for block in study.populations} == {
        "single",
        "weak",
        "equivalent",
        "scale",
        "none",
    }
    for block in study.populations:
        assert set(block.methods) == set(METHODS)
        for score in block.methods.values():
            assert 0.0 <= score.accuracy.value <= 1.0
            assert 0.0 <= score.confident_error.value <= 1.0
    assert study.feature_importances.shape == (16,)
    assert study.setup_seconds > 0.0
    assert study.analytic_seconds_per_case > 0.0
    assert study.learned_seconds_per_case > 0.0


def test_the_study_refuses_overlapping_splits(short_set):
    with pytest.raises(ValueError, match="overlap"):
        learned_study(short_set, TRAIN, TRAIN, NULL)
    with pytest.raises(ValueError, match="overlap"):
        learned_study(short_set, TRAIN, TEST, TEST)


def test_the_analytic_path_refuses_everything_on_a_rate_only_set(study):
    """Accuracy zero on single-parameter cases is the honest answer there, not a
    failure, and it is asserted so that a future change cannot quietly turn the
    refusal into a guess."""
    single = study.population("single")
    assert single.methods["analytic-gated"].refusal_rate.value == pytest.approx(
        1.0, rel=0.0, abs=0.0
    )
    assert single.methods["analytic-gated"].accuracy.value == pytest.approx(
        0.0, rel=0.0, abs=0.0
    )
    assert single.methods["analytic-gated"].ambiguity_declared.value == pytest.approx(
        1.0, rel=0.0, abs=0.0
    )


def test_the_learned_component_declares_no_ambiguity_anywhere(study):
    for block in study.populations:
        for method in ("learned-parameter", "learned-refusal"):
            assert block.methods[method].ambiguity_declared.value == pytest.approx(
                0.0, rel=0.0, abs=0.0
            )


def test_the_study_cannot_find_an_unknown_population(study):
    with pytest.raises(KeyError, match="no population named"):
        study.population("invented")


def test_montecarlo_directions_returns_one_row_per_seed(torque_set):
    amplitudes, bound = montecarlo_directions(torque_set, NOMINAL, tuple(range(8400, 8404)))
    report = analyse(torque_set, NOMINAL)
    assert amplitudes.shape == (4, report.rank)
    assert bound.shape == (report.rank,)
    assert np.all(bound > 0.0)
    assert np.all(np.isfinite(amplitudes))


def test_the_bound_is_the_reciprocal_singular_values(short_set):
    from twindiverge.telemetry import design_matrix

    _, bound = montecarlo_directions(short_set, NOMINAL, (8410, 8411, 8412, 8413))
    matrix, _ = design_matrix(short_set, NOMINAL, trajectory=simulate_with_sensitivities(
        NOMINAL, short_set.campaign
    ))
    values = np.linalg.svd(matrix, compute_uv=False)
    report = analyse(short_set, NOMINAL)
    assert bound == pytest.approx(1.0 / values[: report.rank], rel=1e-12, abs=0.0)


def test_every_declared_telemetry_set_runs_through_the_pipeline():
    case = generate_cases((8500,), "single")[0]
    for telemetry in TELEMETRY_SETS.values():
        analysis = analyse_case(telemetry, case)
        assert analysis.gated_answer in (
            "UNIDENTIFIABLE",
            "REFUSED",
            *PARAMETER_NAMES,
        )

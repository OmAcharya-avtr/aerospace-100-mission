"""Every input-validation path raises an actionable error."""

from __future__ import annotations

import pytest

from safemodetree import (
    Condition,
    DetectionAssumption,
    Fault,
    FaultResponsePolicy,
    LearnedFaultClassifier,
    Literal,
    Mode,
    ObservabilityModel,
    ObservationChannel,
    Precondition,
    Response,
    SafeModeSpec,
    brier_score,
    load_spec,
    load_spec_text,
    make_observations,
    random_declaration,
    reliability_curve,
    split_observations,
)


def test_bad_identifier_rejected():
    with pytest.raises(ValueError, match="not a valid identifier"):
        Mode(id="2bad")
    with pytest.raises(TypeError, match="must be a string"):
        Mode(id=7)  # type: ignore[arg-type]


def test_unknown_severity_rejected():
    with pytest.raises(ValueError, match="MIL-STD-1629A"):
        Fault(id="f", severity="V")


def test_negative_latency_rejected():
    with pytest.raises(ValueError, match="must be >= 0 s"):
        Fault(id="f", detection_latency_s=-1.0)


def test_negative_cost_rejected():
    with pytest.raises(ValueError, match="must be >= 0 min"):
        Response(id="r", to_mode="m", cost_min=-0.5)


def test_bool_is_not_a_number():
    with pytest.raises(TypeError, match="real number"):
        Response(id="r", to_mode="m", cost_min=True)


def test_channel_direction_must_be_above_or_below():
    with pytest.raises(ValueError, match="'above' or 'below'"):
        ObservationChannel(
            id="c", unit="V", nominal=0.0, threshold=1.0, faulted=2.0,
            direction="sideways", noise_sd=0.5,
        )


def test_channel_ordering_is_checked_in_both_directions():
    with pytest.raises(ValueError, match="nominal < threshold < faulted"):
        ObservationChannel(
            id="c", unit="V", nominal=3.0, threshold=1.0, faulted=2.0,
            direction="above", noise_sd=0.5,
        )
    with pytest.raises(ValueError, match="faulted < threshold < nominal"):
        ObservationChannel(
            id="c", unit="V", nominal=0.0, threshold=1.0, faulted=2.0,
            direction="below", noise_sd=0.5,
        )


def test_channel_noise_sd_must_be_positive():
    with pytest.raises(ValueError, match="noise_sd must be > 0"):
        ObservationChannel(
            id="c", unit="V", nominal=0.0, threshold=1.0, faulted=2.0,
            direction="above", noise_sd=0.0,
        )


def test_signature_width_is_checked():
    channel = ObservationChannel(
        id="c", unit="V", nominal=0.0, threshold=1.0, faulted=2.0,
        direction="above", noise_sd=0.5,
    )
    with pytest.raises(ValueError, match="expected 1"):
        ObservabilityModel(channels=(channel,), signatures={"f": (1, 0)})
    with pytest.raises(ValueError, match="only 0 or 1"):
        ObservabilityModel(channels=(channel,), signatures={"f": (2,)})


def test_declaration_without_safe_mode_is_rejected(minimal_yaml):
    with pytest.raises(ValueError, match="declares no safe mode"):
        load_spec_text(minimal_yaml.replace("    safe: true", ""))


def test_unknown_initial_mode_is_rejected(minimal_yaml):
    with pytest.raises(ValueError, match="initial_mode"):
        load_spec_text(minimal_yaml.replace("initial_mode: ops", "initial_mode: nowhere"))


def test_response_to_undeclared_mode_is_rejected(minimal_yaml):
    with pytest.raises(ValueError, match="which is not declared"):
        load_spec_text(minimal_yaml.replace("to_mode: hold", "to_mode: elsewhere"))


def test_response_for_undeclared_fault_is_rejected(minimal_yaml):
    with pytest.raises(ValueError, match="which is not declared"):
        load_spec_text(minimal_yaml.replace("fault: f_one", "fault: f_two"))


def test_precondition_over_undeclared_condition_is_rejected(minimal_yaml):
    text = minimal_yaml.replace(
        "    cost_min: 7.5", "    cost_min: 7.5\n    precondition: [no_such_condition]"
    )
    with pytest.raises(ValueError, match="references condition"):
        load_spec_text(text)


def test_unknown_key_is_rejected_not_ignored(minimal_yaml):
    with pytest.raises(ValueError, match="unknown key"):
        load_spec_text(minimal_yaml.replace("name: minimal", "name: minimal\nnmae: typo"))


def test_missing_required_key_is_named(minimal_yaml):
    with pytest.raises(ValueError, match="missing required key 'channels'"):
        load_spec_text(
            minimal_yaml.split("channels:")[0]
            + "signatures:\n  f_one:\n    c_one: 1\n"
        )


def test_signature_for_undeclared_fault_is_rejected(minimal_yaml):
    with pytest.raises(ValueError, match="exactly one signature per fault"):
        load_spec_text(
            minimal_yaml.replace(
                "signatures:\n  f_one:\n    c_one: 1",
                "signatures:\n  f_one:\n    c_one: 1\n  f_ghost:\n    c_one: 1",
            )
        )


def test_signature_naming_undeclared_channel_is_rejected(minimal_yaml):
    with pytest.raises(ValueError, match="undeclared channel"):
        load_spec_text(minimal_yaml.replace("    c_one: 1", "    c_two: 1"))


def test_all_zero_signature_is_rejected(minimal_yaml):
    with pytest.raises(ValueError, match="all-zero signature"):
        load_spec_text(minimal_yaml.replace("    c_one: 1", "    c_one: 0"))


def test_duplicate_mode_id_is_rejected(minimal_yaml):
    with pytest.raises(ValueError, match="duplicate mode id"):
        load_spec_text(minimal_yaml.replace("  - id: hold", "  - id: ops\n  - id: hold"))


def test_condition_listing_undeclared_mode_is_rejected(minimal_yaml):
    text = minimal_yaml.replace(
        "faults:", "conditions:\n  - id: k\n    available_in_modes: [ghost]\nfaults:"
    )
    with pytest.raises(ValueError, match="which is not declared"):
        load_spec_text(text)


def test_list_field_given_as_bare_string_is_rejected(minimal_yaml):
    text = minimal_yaml.replace(
        "    detection_latency_s: 5.0", "    detection_latency_s: 5.0\n    occurs_in_modes: ops"
    )
    with pytest.raises(ValueError, match="not a bare string"):
        load_spec_text(text)


def test_load_spec_on_missing_file_raises_oserror(tmp_path):
    with pytest.raises(OSError):
        load_spec(tmp_path / "nothing.yaml")


def test_detection_assumption_rejects_negative_budget():
    with pytest.raises(ValueError, match="must be >= 0 s"):
        DetectionAssumption(latency_budget_s=-1.0)


def test_policy_rejects_negative_threshold(minimal):
    with pytest.raises(ValueError, match="must be >= 0"):
        FaultResponsePolicy(minimal, confidence_threshold=-0.1)
    with pytest.raises(TypeError, match="real number"):
        FaultResponsePolicy(minimal, confidence_threshold="high")  # type: ignore[arg-type]


def test_policy_rejects_unknown_mode_and_fault(minimal):
    policy = FaultResponsePolicy(minimal)
    with pytest.raises(KeyError):
        policy.decide("nowhere", "f_one", 1.0)
    with pytest.raises(KeyError):
        policy.decide("ops", "f_ghost", 1.0)


def test_make_observations_rejects_bad_arguments(minimal):
    with pytest.raises(ValueError, match="n_per_fault must be > 0"):
        make_observations(minimal, n_per_fault=0)
    with pytest.raises(ValueError, match="noise_scale must be > 0"):
        make_observations(minimal, noise_scale=0.0)
    with pytest.raises(ValueError, match="nominal_fraction must be in"):
        make_observations(minimal, nominal_fraction=1.0)


def test_split_observations_rejects_empty_test_split(minimal):
    observations = make_observations(minimal, n_per_fault=20, seed=1)
    with pytest.raises(ValueError, match="test split is"):
        split_observations(observations, train_fraction=0.8, calibration_fraction=0.3)


def test_reliability_curve_rejects_bad_inputs():
    with pytest.raises(ValueError, match="must lie in"):
        reliability_curve([1.5], [True])
    with pytest.raises(ValueError, match="at least one prediction"):
        reliability_curve([], [])
    with pytest.raises(ValueError, match="n_bins must be >= 1"):
        reliability_curve([0.5], [True], n_bins=0)
    with pytest.raises(ValueError, match="entries"):
        reliability_curve([0.5, 0.5], [True])


def test_brier_score_rejects_unnormalised_rows():
    with pytest.raises(ValueError, match="sum to 1"):
        brier_score([[0.5, 0.2]], ["a"], ("a", "b"))


def test_brier_score_rejects_unknown_label():
    with pytest.raises(ValueError, match="not one of the given classes"):
        brier_score([[0.5, 0.5]], ["c"], ("a", "b"))


def test_learned_classifier_rejects_bad_settings():
    with pytest.raises(ValueError, match="calibration must be"):
        LearnedFaultClassifier(calibration="magic")
    with pytest.raises(ValueError, match="regularisation must be > 0"):
        LearnedFaultClassifier(regularisation=0.0)
    with pytest.raises(ValueError, match="cv must be >= 2"):
        LearnedFaultClassifier(cv=1)


def test_learned_classifier_refuses_to_predict_before_fit():
    with pytest.raises(RuntimeError, match="not fitted"):
        LearnedFaultClassifier().predict_proba([[0.0]])


def test_learned_classifier_rejects_single_class():
    model = LearnedFaultClassifier(calibration="none")
    with pytest.raises(ValueError, match="at least two classes"):
        model.fit([[0.0], [1.0]], ["a", "a"])


def test_learned_classifier_rejects_class_smaller_than_cv():
    model = LearnedFaultClassifier(calibration="isotonic", cv=5)
    with pytest.raises(ValueError, match="fewer than cv"):
        model.fit([[0.0], [1.0], [2.0]], ["a", "a", "b"])


def test_random_declaration_rejects_degenerate_sizes():
    with pytest.raises(ValueError, match="n_modes must be >= 2"):
        random_declaration(0, n_modes=1)
    with pytest.raises(ValueError, match="n_faults must be >= 1"):
        random_declaration(0, n_faults=0)
    with pytest.raises(ValueError, match="n_channels must be >= 1"):
        random_declaration(0, n_channels=0)


def test_spec_requires_at_least_one_mode_and_fault():
    channel = ObservationChannel(
        id="c", unit="V", nominal=0.0, threshold=1.0, faulted=2.0,
        direction="above", noise_sd=0.5,
    )
    with pytest.raises(ValueError, match="at least one mode"):
        SafeModeSpec(
            name="x",
            modes=(),
            faults=(Fault(id="f"),),
            responses=(),
            observability=ObservabilityModel(channels=(channel,), signatures={"f": (1,)}),
            initial_mode="m",
        )
    with pytest.raises(ValueError, match="at least one fault"):
        SafeModeSpec(
            name="x",
            modes=(Mode(id="m", safe=True),),
            faults=(),
            responses=(),
            observability=ObservabilityModel(channels=(channel,), signatures={}),
            initial_mode="m",
        )


def test_precondition_rejects_non_literal_contents():
    with pytest.raises(TypeError, match="Literal instances"):
        Precondition(("a",))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="must be a tuple"):
        Precondition([Literal("a")])  # type: ignore[arg-type]


def test_condition_rejects_non_tuple_mode_list():
    with pytest.raises(TypeError, match="must be a tuple"):
        Condition(id="k", available_in_modes=["m"])  # type: ignore[arg-type]

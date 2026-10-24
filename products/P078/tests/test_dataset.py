"""The generator: clean by construction, and its timing arithmetic unit-checked."""

from __future__ import annotations

import pytest

from cmdguard.dataset import (
    ACTIVITY_WEIGHTS,
    BASE_LIMITS,
    MODES,
    PREDICATES,
    CorpusEntry,
    GeneratorConfig,
    _required_dt,
    base_dictionary,
    base_machine,
    build_corpus,
    generate_clean_case,
    generate_defective_case,
    verify_clean,
)
from cmdguard.model import SequenceStep
from cmdguard.trace import forward_execute


def test_base_dictionary_declares_every_predicate_as_an_effect() -> None:
    dictionary = base_dictionary()
    written = {name for spec in dictionary.commands.values() for name in spec.effects}
    assert set(PREDICATES) <= written


def test_base_dictionary_has_the_documented_shape() -> None:
    dictionary = base_dictionary()
    assert len(dictionary.commands) == 29
    assert set(dictionary.resource_limits.limits) == set(BASE_LIMITS)
    irreversible = [s.identifier for s in dictionary.commands.values() if s.irreversible]
    assert sorted(irreversible) == ["ANT_DEPLOY", "MEM_SCRUB", "PROP_BURN", "PROP_VALVE_OPEN"]
    assert dictionary.exclusion_members("high_power") == (
        "DOWNLINK_START",
        "PAYLOAD_SCAN",
        "PROP_BURN",
    )


def test_base_machine_is_consistent() -> None:
    machine = base_machine()
    assert machine.modes == MODES
    for mode in MODES:
        assert machine.allows(mode, mode)
    assert machine.allows("NOMINAL", "SCIENCE")
    assert not machine.allows("SAFE", "SCIENCE")


def test_base_dictionary_rejects_non_positive_scales() -> None:
    with pytest.raises(ValueError, match="must both be > 0"):
        base_dictionary(draw_scale=0.0)


@pytest.mark.parametrize("seed", range(12))
def test_generated_case_is_clean_by_construction(seed: int) -> None:
    """The negative control, per seed. A failure here is a generator defect."""
    case = generate_clean_case(seed)
    report = verify_clean(case)
    assert report.n_violations == 0
    assert len(case.sequence) > 0


@pytest.mark.parametrize("seed", range(6))
def test_generation_is_deterministic(seed: int) -> None:
    first = generate_clean_case(seed)
    second = generate_clean_case(seed)
    assert [(s.t_s, s.command) for s in first.sequence] == [
        (s.t_s, s.command) for s in second.sequence
    ]
    assert forward_execute(first).fingerprint() == forward_execute(second).fingerprint()


def test_generated_sequences_reach_the_interesting_commands() -> None:
    """A uniform walk produced sequences with no payload scan at all.

    The goal-directed generator exists because of that measurement, so the
    property it was written for is asserted here.
    """
    used: set[str] = set()
    for seed in range(20):
        used.update(step.command for step in generate_clean_case(seed).sequence)
    for command in ("PAYLOAD_SCAN", "DOWNLINK_START", "PROP_BURN", "ANT_DEPLOY"):
        assert command in used


def test_activity_weights_name_declared_commands() -> None:
    declared = set(base_dictionary().identifiers)
    assert {name for name, _ in ACTIVITY_WEIGHTS} <= declared
    assert all(weight >= 1 for _, weight in ACTIVITY_WEIGHTS)


def test_generator_config_validation() -> None:
    with pytest.raises(ValueError, match="n_steps=0"):
        GeneratorConfig(n_steps=0)
    with pytest.raises(ValueError, match="unverified_fraction"):
        GeneratorConfig(unverified_fraction=1.5)
    with pytest.raises(ValueError, match="dt range"):
        GeneratorConfig(dt_min_s=10.0, dt_max_s=1.0)
    with pytest.raises(ValueError, match="draw_jitter"):
        GeneratorConfig(draw_jitter=1.0)


def test_short_configuration_is_honoured() -> None:
    case = generate_clean_case(1, GeneratorConfig(n_steps=4))
    assert len(case.sequence) <= 4


def test_required_dt_is_hand_checkable() -> None:
    """Hand calculation, stated in full.

    PWR_BUS_CHECK declares min_gap_after_s = 2.0 and min_interval_s = 30.0.
    After a single PWR_BUS_CHECK at t = 0, the gap owed to the next command is
    2.0 s. For another PWR_BUS_CHECK the interval owed is 30.0 s measured from
    t = 0, which from the last step (also t = 0) is 30.0 - 0 = 30.0 s, and the
    maximum of the two is 30.0 s. With the last step at t = 25 instead, the
    interval owed from the last step is 30.0 - 25.0 = 5.0 s, and the maximum
    against the 2.0 s gap is 5.0 s.
    """
    dictionary = base_dictionary()
    assert _required_dt(dictionary, [], "PWR_BUS_CHECK") == 0.0
    one = [SequenceStep(t_s=0.0, command="PWR_BUS_CHECK")]
    assert _required_dt(dictionary, one, "SSR_DISARM") == pytest.approx(2.0, abs=1e-12)
    assert _required_dt(dictionary, one, "PWR_BUS_CHECK") == pytest.approx(30.0, abs=1e-12)
    two = [
        SequenceStep(t_s=0.0, command="PWR_BUS_CHECK"),
        SequenceStep(t_s=25.0, command="SSR_DISARM"),
    ]
    assert _required_dt(dictionary, two, "PWR_BUS_CHECK") == pytest.approx(5.0, abs=1e-12)


def test_required_dt_never_negative() -> None:
    dictionary = base_dictionary()
    steps = [
        SequenceStep(t_s=0.0, command="PWR_BUS_CHECK"),
        SequenceStep(t_s=5000.0, command="SSR_DISARM"),
    ]
    assert _required_dt(dictionary, steps, "PWR_BUS_CHECK") >= 0.0


def test_verify_clean_raises_on_a_defective_case() -> None:
    clean, defective, record = generate_defective_case(4, "undeclared_typo")
    assert record.applied
    with pytest.raises(AssertionError, match="was generated as clean but reports"):
        verify_clean(defective)
    assert verify_clean(clean).clean


def test_build_corpus_shape() -> None:
    corpus = build_corpus(range(3), families=("undeclared_typo", "swap_enabler"))
    assert len(corpus) == 3 * 3
    assert sum(1 for entry in corpus if entry.record is None) == 3
    assert all(isinstance(entry, CorpusEntry) for entry in corpus)


def test_build_corpus_without_clean_cases() -> None:
    corpus = build_corpus(range(2), families=("undeclared_typo",), include_clean=False)
    assert len(corpus) == 2
    assert all(entry.record is not None for entry in corpus)


def test_corpus_reports_match_their_cases() -> None:
    for entry in build_corpus(range(2), families=("swap_enabler",)):
        assert entry.report.case_name == entry.case.name
        assert entry.clean == entry.report.clean

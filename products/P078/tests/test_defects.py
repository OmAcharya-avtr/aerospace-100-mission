"""The injected defect population: applicability, determinism, honesty."""

from __future__ import annotations

import pytest

from cmdguard.dataset import generate_clean_case
from cmdguard.defects import FAMILIES, inject, inject_random, unverified_predicates
from cmdguard.trace import forward_execute


@pytest.fixture(scope="module")
def clean_cases() -> tuple:
    return tuple(generate_clean_case(seed) for seed in range(6))


def test_family_list_is_the_documented_nine() -> None:
    assert len(FAMILIES) == 9
    assert FAMILIES[-1] == "undeclared_typo"
    assert len(set(FAMILIES)) == 9


@pytest.mark.parametrize("family", FAMILIES)
def test_every_family_is_applicable_somewhere(family: str, clean_cases: tuple) -> None:
    applied = [inject(case, family, seed)[1].applied for seed, case in enumerate(clean_cases)]
    assert any(applied), f"{family} was a no-op on every clean case in the fixture"


@pytest.mark.parametrize("family", FAMILIES)
def test_injection_is_deterministic(family: str, clean_cases: tuple) -> None:
    for case in clean_cases:
        first, record_a = inject(case, family, 11)
        second, record_b = inject(case, family, 11)
        assert record_a == record_b
        assert forward_execute(first).fingerprint() == forward_execute(second).fingerprint()


@pytest.mark.parametrize("family", FAMILIES)
def test_a_noop_injection_returns_the_case_unchanged(family: str, clean_cases: tuple) -> None:
    for case in clean_cases:
        edited, record = inject(case, family, 3)
        if not record.applied:
            assert edited.sequence == case.sequence
            assert edited.name == case.name


def test_applied_injection_renames_the_case(clean_cases: tuple) -> None:
    edited, record = inject(clean_cases[0], "undeclared_typo", 5)
    assert record.applied
    assert edited.name == f"{clean_cases[0].name}+undeclared_typo+5"


def test_undeclared_typo_always_produces_a_finding(clean_cases: tuple) -> None:
    """The control family: a per-command lint finds this one every time."""
    from cmdguard.lint import static_lint

    for case in clean_cases:
        edited, record = inject(case, "undeclared_typo", 2)
        assert record.applied
        assert "undeclared_command" in static_lint(edited).kind_counts()


def test_unknown_family_raises_keyerror(clean_cases: tuple) -> None:
    with pytest.raises(KeyError, match="unknown defect family"):
        inject(clean_cases[0], "no_such_family", 1)


def test_inject_random_picks_a_declared_family(clean_cases: tuple) -> None:
    for seed in range(12):
        _, record = inject_random(clean_cases[0], seed)
        assert record.family in FAMILIES


def test_record_render_is_one_line(clean_cases: tuple) -> None:
    _, record = inject(clean_cases[0], "undeclared_typo", 1)
    rendered = record.render()
    assert "\n" not in rendered
    assert "applied" in rendered


def test_compress_gap_has_no_noops_on_the_fixture(clean_cases: tuple) -> None:
    """Regression: an early version used min_interval_s as if it were a gap.

    That arithmetic error made 18 of 40 injections no-ops. The fix narrowed the
    family to ``min_gap_after_s`` only, and this test pins the behaviour.
    """
    for seed, case in enumerate(clean_cases):
        _, record = inject(case, "compress_gap", seed)
        assert record.applied


def test_irreversible_early_reaches_the_gate_kind(clean_cases: tuple) -> None:
    """At least one seed in the fixture produces the gate-blocking kind."""
    found = False
    for seed in range(40):
        case = generate_clean_case(seed)
        edited, record = inject(case, "irreversible_early", seed)
        if record.applied and forward_execute(edited).gate_blocking:
            found = True
            break
    assert found


def test_unverified_predicates_helper(clean_cases: tuple) -> None:
    for case in clean_cases:
        names = unverified_predicates(case)
        assert all(case.initial.verity(name).value == "unknown" for name in names)

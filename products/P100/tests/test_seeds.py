"""Finding and rewriting the seed literals a script declares."""

from __future__ import annotations

import ast

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from seedaudit import seeds


def _int_literals(source: str) -> list[int]:
    """Every integer literal in a source string, in source order."""
    return [
        node.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node.value if isinstance(node, ast.Constant) else None, int)
        and not isinstance(node.value, bool)
    ]


def test_every_recognised_seeding_call_is_found() -> None:
    for source, expected in (
        ("rng = np.random.default_rng(12345)\n", 12345),
        ("rng = default_rng(7)\n", 7),
        ("np.random.seed(99)\n", 99),
        ("import random\nrandom.seed(4242)\n", 4242),
        ("s = np.random.SeedSequence(2026)\n", 2026),
        ("b = np.random.PCG64(2026)\n", 2026),
    ):
        scan = seeds.find_seed_literals(source)
        assert scan.count == 1, source
        assert scan.literals[0].value == expected
        assert scan.literals[0].kind == "call"


def test_a_user_function_named_seed_is_not_a_seeding_call() -> None:
    assert seeds.find_seed_literals("helper.seed(3)\n").count == 0
    assert seeds.find_seed_literals("x = my_seed(3)\n").count == 0


def test_seed_and_random_state_keywords_are_found_and_siblings_are_not() -> None:
    scan = seeds.find_seed_literals("m = Forest(n_estimators=100, random_state=11)\n")
    assert scan.count == 1
    assert scan.literals[0].value == 11
    assert scan.literals[0].kind == "keyword"
    rewritten, _ = seeds.rewrite_seeds("m = Forest(n_estimators=100, random_state=11)\n")
    assert "n_estimators=100" in rewritten
    assert seeds.find_seed_literals("run(seed=5)\n").count == 1


def test_upper_case_seed_assignments_are_found_in_every_shape() -> None:
    sources = ("SEED = 930093\n", "ANALYSIS_SEED = 1\n", "SEED_CONTROL = 5150\n", "SEEDS = 4\n")
    for source in sources:
        scan = seeds.find_seed_literals(source)
        assert scan.count == 1, source
        assert scan.literals[0].kind == "assign"


def test_a_lower_case_seed_or_an_unrelated_constant_is_not_rewritten() -> None:
    # A lower-case `seed` is usually a parameter being passed through; editing
    # it in one place while its callers pass their own value would change some
    # call sites and not others. A tolerance is an integer literal too.
    assert seeds.find_seed_literals("seed = 42\n").count == 0
    assert seeds.find_seed_literals("TOLERANCE = 5\nMAX_ITERATIONS = 1000\n").count == 0


def test_a_seed_position_that_is_not_an_integer_literal_is_reported_unresolved() -> None:
    # Reporting it as rewritten when it was not would be the audit lying about
    # its own coverage.
    boolean = seeds.find_seed_literals("run(seed=True)\n")
    assert boolean.count == 0 and boolean.unresolved
    through_name = seeds.find_seed_literals("SEED = 1\nrng = np.random.default_rng(SEED)\n")
    assert through_name.count == 1
    assert any(entry[1] == "call" for entry in through_name.unresolved)
    none_valued = seeds.find_seed_literals("m = Forest(random_state=None)\n")
    assert none_valued.count == 0 and len(none_valued.unresolved) == 1


def test_the_rewrite_adds_the_declared_perturbation() -> None:
    assert seeds.PERTURBATION == 104729
    rewritten, scan = seeds.rewrite_seeds("rng = np.random.default_rng(1)\n")
    assert scan.count == 1
    assert f"default_rng({1 + seeds.PERTURBATION})" in rewritten


def test_a_zero_perturbation_is_the_identity_and_still_scans() -> None:
    source = "SEED = 930093\nrng = np.random.default_rng(12345)\n"
    rewritten, scan = seeds.rewrite_seeds(source, 0)
    assert rewritten == source
    assert scan.count == 2


def test_the_rewrite_keeps_distinct_seeds_distinct() -> None:
    # Collapsing several declared seeds onto one constant could make two
    # independent streams identical and produce a change that is an artifact
    # of the rewrite rather than a dependence on the seed.
    rewritten, _ = seeds.rewrite_seeds("A_SEED = 1\nB_SEED = 2\n")
    values = sorted(_int_literals(rewritten))
    assert values == [1 + seeds.PERTURBATION, 2 + seeds.PERTURBATION]
    assert values[0] != values[1]


def test_the_rewrite_handles_several_seeds_on_one_line() -> None:
    rewritten, scan = seeds.rewrite_seeds("run(seed=1, random_state=2)\n")
    assert scan.count == 2
    assert f"seed={1 + seeds.PERTURBATION}" in rewritten
    assert f"random_state={2 + seeds.PERTURBATION}" in rewritten


def test_the_rewrite_leaves_the_file_parseable_and_a_seedless_file_untouched() -> None:
    ast.parse(seeds.rewrite_seeds("import numpy as np\nSEED = 7\nx = 1 + 2\n")[0])
    source = "x = 1\ny = x + 2\n"
    rewritten, scan = seeds.rewrite_seeds(source)
    assert rewritten == source and scan.count == 0


def test_the_rewrite_does_not_reach_inside_a_string_or_a_comment() -> None:
    # The rewrite works from the syntax tree, not from a regular expression.
    quoted, _ = seeds.rewrite_seeds('MESSAGE = "default_rng(12345)"\nnp.random.seed(1)\n')
    assert 'MESSAGE = "default_rng(12345)"' in quoted
    commented, _ = seeds.rewrite_seeds("# np.random.seed(12345) was here\nnp.random.seed(1)\n")
    assert "# np.random.seed(12345) was here" in commented


def test_literals_are_sorted_by_position_and_deduplicated() -> None:
    scan = seeds.find_seed_literals("B_SEED = 2\nA_SEED = 1\n")
    assert [literal.line for literal in scan.literals] == [1, 2]
    nested = seeds.find_seed_literals("np.random.default_rng(seed=9)\n")
    positions = {(item.line, item.col, item.end_col) for item in nested.literals}
    assert len(positions) == nested.count


def test_input_validation() -> None:
    with pytest.raises(TypeError, match="must be a str"):
        seeds.find_seed_literals(None)  # type: ignore[arg-type]
    with pytest.raises(SyntaxError):
        # A file silently skipped is coverage the audit would be overstating.
        seeds.find_seed_literals("def (:\n")
    with pytest.raises(ValueError, match="non-negative"):
        seeds.rewrite_seeds("SEED = 1\n", -1)
    with pytest.raises(TypeError, match="must be an int"):
        seeds.rewrite_seeds("SEED = 1\n", 1.5)  # type: ignore[arg-type]


@given(value=st.integers(min_value=0, max_value=2**31 - 1))
@settings(max_examples=200, deadline=None)
def test_property_the_rewrite_shifts_every_literal_by_the_perturbation(value: int) -> None:
    rewritten, scan = seeds.rewrite_seeds(f"rng = np.random.default_rng({value})\n")
    assert scan.count == 1
    assert _int_literals(rewritten) == [value + seeds.PERTURBATION]


@given(
    value=st.integers(min_value=0, max_value=10**9),
    offset=st.integers(min_value=0, max_value=10**6),
)
@settings(max_examples=150, deadline=None)
def test_property_rewriting_twice_equals_rewriting_by_the_sum(value: int, offset: int) -> None:
    source = f"SEED = {value}\n"
    once, _ = seeds.rewrite_seeds(source, offset)
    assert seeds.rewrite_seeds(once, offset)[0] == seeds.rewrite_seeds(source, 2 * offset)[0]


def test_regression_a_product_style_seed_header_is_rewritten_correctly() -> None:
    # Pinned literals: this is the shape the mission's own scripts use.
    source = (
        "SEED_ANALYSIS = 930093\n"
        "SEED_FIXTURE = 930093\n"
        "SEED_CONTROL = 5150\n"
        "rng = np.random.default_rng(SEED_ANALYSIS)\n"
    )
    rewritten, scan = seeds.rewrite_seeds(source)
    assert scan.count == 3
    assert "SEED_ANALYSIS = 1034822" in rewritten
    assert "SEED_CONTROL = 109879" in rewritten
    assert "default_rng(SEED_ANALYSIS)" in rewritten

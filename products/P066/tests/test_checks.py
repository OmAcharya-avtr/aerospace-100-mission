"""Each of the ten checks, raised deliberately and in isolation."""

from __future__ import annotations

import pytest

from budgetroll import (
    Budget,
    Correlation,
    Item,
    MarginPolicy,
    Node,
    has_blocking,
    run_checks,
    severity_counts,
)
from budgetroll.checks import CODES, DIVERGENCE_RTOL, MARGIN_RTOL

POLICY = MarginPolicy(
    name="t",
    description="check policy",
    source="declared in tests/test_checks.py",
    levels={"mature": 0.05, "new": 0.20},
)


def codes(budget: Budget) -> list[str]:
    return [f.code for f in run_checks(budget)]


def leaf(ident, value=1.0, uncertainty=0.1, maturity="mature", **kwargs):
    return Item(
        id=ident, name=ident, value=value, uncertainty=uncertainty, maturity=maturity, **kwargs
    )


def wrap(root, correlations=(), k_sigma=1.0):
    return Budget(
        id="b",
        name="B",
        unit="kg",
        quantity="mass",
        policy=POLICY,
        root=root,
        correlations=correlations,
        k_sigma=k_sigma,
    )


def test_br001_rss_with_a_declared_correlation():
    root = Node(id="root", name="root", combination="rss", children=(leaf("a"), leaf("b")))
    budget = wrap(root, (Correlation("a", "b", 0.5, "same supplier"),))
    assert "BR001" in codes(budget)
    finding = next(f for f in run_checks(budget) if f.code == "BR001")
    assert finding.severity == "error"
    assert "a-b rho=+0.5" in finding.message


def test_br001_is_silent_when_the_declared_rho_is_zero():
    root = Node(id="root", name="root", combination="rss", children=(leaf("a"), leaf("b")))
    budget = wrap(root, (Correlation("a", "b", 0.0, "independent and said so"),))
    assert "BR001" not in codes(budget)


def test_br002_error_when_the_hierarchy_understates():
    root = Node(id="root", name="root", combination="rss", children=(leaf("a"), leaf("b")))
    budget = wrap(root, (Correlation("a", "b", 0.9, "same supplier"),))
    finding = next(f for f in run_checks(budget) if f.code == "BR002")
    assert finding.severity == "error"
    assert "understates" in finding.message


def test_br002_info_when_the_hierarchy_overstates():
    root = Node(id="root", name="root", combination="linear", children=(leaf("a"), leaf("b")))
    finding = next(f for f in run_checks(wrap(root)) if f.code == "BR002")
    assert finding.severity == "info"
    assert "overstates" in finding.message


def test_br002_is_silent_within_the_divergence_tolerance():
    root = Node(id="root", name="root", combination="rss", children=(leaf("a"), leaf("b")))
    assert "BR002" not in codes(wrap(root))
    assert DIVERGENCE_RTOL > 0.0


def test_br003_error_on_a_real_overrun():
    root = Node(
        id="root",
        name="root",
        combination="rss",
        children=(leaf("a", value=10.0), leaf("b", value=10.0)),
        allocation=5.0,
    )
    finding = next(f for f in run_checks(wrap(root)) if f.code == "BR003")
    assert finding.severity == "error"


def test_br003_warning_inside_the_rounding_band():
    root = Node(
        id="root",
        name="root",
        combination="rss",
        children=(
            leaf("a", value=0.1, uncertainty=0.0),
            leaf("b", value=0.1, uncertainty=0.0),
            leaf("c", value=0.1, uncertainty=0.0),
        ),
        allocation=0.3 * 1.05,
    )
    finding = next(f for f in run_checks(wrap(root)) if f.code == "BR003")
    assert finding.severity == "warning"
    assert MARGIN_RTOL > 0.0


def test_br003_is_silent_with_margin_to_spare():
    root = Node(
        id="root",
        name="root",
        combination="rss",
        children=(leaf("a"), leaf("b")),
        allocation=100.0,
    )
    assert "BR003" not in codes(wrap(root))


def test_br004_margin_lost_at_k_sigma():
    root = Node(
        id="root",
        name="root",
        combination="rss",
        children=(leaf("a", value=1.0, uncertainty=3.0), leaf("b", value=1.0, uncertainty=4.0)),
        allocation=3.0,
    )
    finding = next(f for f in run_checks(wrap(root)) if f.code == "BR004")
    assert finding.severity == "warning"
    # predicted 2.1, allocation 3.0 so margin 0.9; uncertainty 5.0 so at
    # k = 1 the margin is 3.0 - 7.1 = -4.1.
    assert "-4.1" in finding.message


def test_br004_is_silent_when_the_budget_closes_at_k_sigma():
    root = Node(
        id="root",
        name="root",
        combination="rss",
        children=(leaf("a"), leaf("b")),
        allocation=100.0,
    )
    assert "BR004" not in codes(wrap(root))


def test_br005_contingency_override():
    root = Node(
        id="root",
        name="root",
        combination="rss",
        children=(leaf("a", contingency=0.4), leaf("b")),
    )
    finding = next(f for f in run_checks(wrap(root)) if f.code == "BR005")
    assert finding.severity == "warning"
    assert finding.subject == "a"


def test_br005_is_silent_when_the_override_matches_the_policy():
    root = Node(
        id="root",
        name="root",
        combination="rss",
        children=(leaf("a", contingency=0.05), leaf("b")),
    )
    assert "BR005" not in codes(wrap(root))


def test_br006_zero_uncertainty_at_an_immature_level():
    root = Node(
        id="root",
        name="root",
        combination="rss",
        children=(leaf("a", uncertainty=0.0, maturity="new"), leaf("b")),
    )
    finding = next(f for f in run_checks(wrap(root)) if f.code == "BR006")
    assert finding.severity == "warning"
    assert finding.subject == "a"


def test_br006_is_silent_at_the_most_mature_level():
    root = Node(
        id="root",
        name="root",
        combination="rss",
        children=(leaf("a", uncertainty=0.0, maturity="mature"), leaf("b")),
    )
    assert "BR006" not in codes(wrap(root))


def test_br007_single_child_rule():
    inner = Node(id="inner", name="inner", combination="rss", children=(leaf("a"),))
    root = Node(id="root", name="root", combination="linear", children=(inner, leaf("b")))
    finding = next(f for f in run_checks(wrap(root)) if f.code == "BR007")
    assert finding.severity == "info"
    assert finding.subject == "inner"


def test_br007_is_silent_for_a_single_child_declared_linear():
    inner = Node(id="inner", name="inner", combination="linear", children=(leaf("a"),))
    root = Node(id="root", name="root", combination="linear", children=(inner, leaf("b")))
    assert "BR007" not in codes(wrap(root))


def test_br008_inadmissible_matrix():
    root = Node(
        id="root",
        name="root",
        combination="linear",
        children=(leaf("a"), leaf("b"), leaf("c")),
    )
    budget = wrap(
        root,
        (
            Correlation("a", "b", 0.9, "r"),
            Correlation("a", "c", 0.9, "r"),
        ),
    )
    finding = next(f for f in run_checks(budget) if f.code == "BR008")
    assert finding.severity == "error"
    assert "not positive semi-definite" in finding.message


def test_br008_is_silent_for_an_admissible_matrix():
    root = Node(
        id="root",
        name="root",
        combination="linear",
        children=(leaf("a"), leaf("b"), leaf("c")),
    )
    budget = wrap(root, (Correlation("a", "b", 0.5, "r"),))
    assert "BR008" not in codes(budget)


def test_br009_correlated_rule_with_nothing_to_correlate():
    root = Node(
        id="root", name="root", combination="correlated", children=(leaf("a"), leaf("b"))
    )
    finding = next(f for f in run_checks(wrap(root)) if f.code == "BR009")
    assert finding.severity == "error"


def test_br009_is_silent_when_a_correlation_applies():
    root = Node(
        id="root", name="root", combination="correlated", children=(leaf("a"), leaf("b"))
    )
    budget = wrap(root, (Correlation("a", "b", 0.5, "r"),))
    assert "BR009" not in codes(budget)


def test_br010_shared_driver_without_a_declared_correlation():
    root = Node(
        id="root",
        name="root",
        combination="rss",
        children=(leaf("a", common_driver="vendor"), leaf("b", common_driver="vendor")),
    )
    finding = next(f for f in run_checks(wrap(root)) if f.code == "BR010")
    assert finding.severity == "warning"
    assert finding.subject == "a,b"


def test_br010_is_silent_when_the_correlation_is_declared():
    root = Node(
        id="root",
        name="root",
        combination="linear",
        children=(leaf("a", common_driver="vendor"), leaf("b", common_driver="vendor")),
    )
    budget = wrap(root, (Correlation("a", "b", 0.5, "r"),))
    assert "BR010" not in codes(budget)


def test_br010_pairs_within_a_group_of_three():
    root = Node(
        id="root",
        name="root",
        combination="linear",
        children=(
            leaf("a", common_driver="vendor"),
            leaf("b", common_driver="vendor"),
            leaf("c", common_driver="vendor"),
        ),
    )
    # Three items in one group give three unordered pairs.
    assert sum(1 for c in codes(wrap(root)) if c == "BR010") == 3


def test_every_code_has_a_description():
    assert sorted(CODES) == [f"BR{n:03d}" for n in range(1, 11)]
    assert all(text and not text.endswith(".") for text in CODES.values())


def test_severity_counts_and_blocking_on_a_clean_budget(clean_mass):
    findings = run_checks(clean_mass)
    assert findings == []
    assert severity_counts(findings) == {"error": 0, "warning": 0, "info": 0}
    assert has_blocking(findings) is False


def test_findings_are_sorted_by_severity_then_code(planted):
    findings = run_checks(planted)
    order = {"error": 0, "warning": 1, "info": 2}
    keys = [(order[f.severity], f.code, f.subject) for f in findings]
    assert keys == sorted(keys)


def test_finding_str_is_readable(planted):
    text = str(run_checks(planted)[0])
    assert text.startswith("[ERROR")
    assert "BR" in text


def test_run_checks_accepts_a_precomputed_rollup(planted):
    from budgetroll import rollup

    assert run_checks(planted, rollup(planted)) == run_checks(planted)


@pytest.mark.parametrize("code", sorted(CODES))
def test_planted_case_raises_every_code(planted, code):
    assert code in codes(planted)

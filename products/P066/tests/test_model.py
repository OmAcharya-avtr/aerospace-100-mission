"""Input validation on the typed model."""

from __future__ import annotations

import pytest

from budgetroll import Budget, Correlation, Item, MarginPolicy, Node, leaves, walk
from budgetroll.errors import DeclarationError
from budgetroll.model import depths, parents

POLICY = MarginPolicy(
    name="p",
    description="d",
    source="declared in tests/test_model.py",
    levels={"low": 0.05, "high": 0.30},
)


def leaf(ident: str, value: float = 1.0, **kwargs) -> Item:
    kwargs.setdefault("maturity", "low")
    return Item(id=ident, name=ident, value=value, **kwargs)


def test_policy_rejects_empty_levels():
    with pytest.raises(DeclarationError, match="declares no maturity levels"):
        MarginPolicy(name="p", description="d", source="s", levels={})


def test_policy_rejects_negative_contingency():
    with pytest.raises(DeclarationError, match="must be >= 0"):
        MarginPolicy(name="p", description="d", source="s", levels={"a": -0.1})


def test_policy_rejects_non_numeric_contingency():
    with pytest.raises(DeclarationError, match="must be a number"):
        MarginPolicy(name="p", description="d", source="s", levels={"a": "0.1"})


def test_policy_rejects_boolean_contingency():
    with pytest.raises(DeclarationError, match="must be a number"):
        MarginPolicy(name="p", description="d", source="s", levels={"a": True})


def test_policy_unknown_maturity_lists_known_levels():
    with pytest.raises(DeclarationError, match="declared levels are: high, low"):
        POLICY.contingency_for("middling")


def test_policy_most_mature_level():
    assert POLICY.most_mature_level == "low"


def test_policy_most_mature_level_breaks_ties_by_name():
    policy = MarginPolicy(
        name="p", description="d", source="s", levels={"b": 0.1, "a": 0.1, "c": 0.5}
    )
    assert policy.most_mature_level == "a"


def test_correlation_rejects_self_pair():
    with pytest.raises(DeclarationError, match="with itself"):
        Correlation("a", "a", 0.5, "because")


@pytest.mark.parametrize("rho", [-1.0001, 1.0001, 2.0, -3.0])
def test_correlation_rejects_out_of_range_rho(rho):
    with pytest.raises(DeclarationError, match=r"must lie in \[-1, 1\]"):
        Correlation("a", "b", rho, "because")


@pytest.mark.parametrize("rho", [-1.0, -0.5, 0.0, 0.5, 1.0])
def test_correlation_accepts_endpoints_and_interior(rho):
    assert Correlation("a", "b", rho, "because").rho == rho


def test_correlation_requires_rationale():
    with pytest.raises(DeclarationError, match="non-empty rationale"):
        Correlation("a", "b", 0.5, "   ")


def test_item_requires_maturity_or_contingency():
    with pytest.raises(DeclarationError, match="one of 'maturity' or 'contingency'"):
        Item(id="a", name="a", value=1.0)


def test_item_rejects_negative_contingency():
    with pytest.raises(DeclarationError, match="contingency must be >= 0"):
        Item(id="a", name="a", value=1.0, contingency=-0.01)


def test_item_rejects_negative_uncertainty():
    with pytest.raises(DeclarationError, match="1 sigma magnitude"):
        Item(id="a", name="a", value=1.0, maturity="low", uncertainty=-0.1)


def test_item_accepts_negative_value_for_log_domain_budgets():
    assert Item(id="a", name="a", value=-2.5, maturity="low").value == -2.5


def test_node_rejects_unknown_combination():
    with pytest.raises(DeclarationError, match="combination must be one of"):
        Node(id="n", name="n", combination="average", children=(leaf("a"),))


def test_node_rejects_empty_children():
    with pytest.raises(DeclarationError, match="at least one child"):
        Node(id="n", name="n", combination="rss", children=())


def test_budget_rejects_duplicate_identifiers():
    root = Node(id="root", name="root", combination="rss", children=(leaf("a"), leaf("a")))
    with pytest.raises(DeclarationError, match="duplicate identifier 'a'"):
        Budget(id="b", name="b", unit="kg", quantity="mass", policy=POLICY, root=root)


def test_budget_rejects_dangling_correlation():
    root = Node(id="root", name="root", combination="rss", children=(leaf("a"), leaf("b")))
    with pytest.raises(DeclarationError, match="is not a leaf item"):
        Budget(
            id="b",
            name="b",
            unit="kg",
            quantity="mass",
            policy=POLICY,
            root=root,
            correlations=(Correlation("a", "zz", 0.5, "because"),),
        )


def test_budget_rejects_correlation_to_an_internal_node():
    inner = Node(id="inner", name="inner", combination="rss", children=(leaf("a"), leaf("b")))
    root = Node(id="root", name="root", combination="rss", children=(inner, leaf("c")))
    with pytest.raises(DeclarationError, match="is not a leaf item"):
        Budget(
            id="b",
            name="b",
            unit="kg",
            quantity="mass",
            policy=POLICY,
            root=root,
            correlations=(Correlation("inner", "c", 0.5, "because"),),
        )


def test_budget_rejects_duplicate_correlation_pair():
    root = Node(id="root", name="root", combination="rss", children=(leaf("a"), leaf("b")))
    with pytest.raises(DeclarationError, match="declared more than once"):
        Budget(
            id="b",
            name="b",
            unit="kg",
            quantity="mass",
            policy=POLICY,
            root=root,
            correlations=(
                Correlation("a", "b", 0.5, "because"),
                Correlation("b", "a", 0.2, "because"),
            ),
        )


def test_budget_rejects_negative_k_sigma():
    root = Node(id="root", name="root", combination="rss", children=(leaf("a"),))
    with pytest.raises(DeclarationError, match="k_sigma must be >= 0"):
        Budget(
            id="b", name="b", unit="kg", quantity="mass", policy=POLICY, root=root, k_sigma=-1.0
        )


def test_walk_is_preorder():
    inner = Node(id="inner", name="inner", combination="rss", children=(leaf("a"), leaf("b")))
    root = Node(id="root", name="root", combination="rss", children=(inner, leaf("c")))
    assert [n.id for n in walk(root)] == ["root", "inner", "a", "b", "c"]


def test_leaves_are_left_to_right():
    inner = Node(id="inner", name="inner", combination="rss", children=(leaf("a"), leaf("b")))
    root = Node(id="root", name="root", combination="rss", children=(inner, leaf("c")))
    assert [n.id for n in leaves(root)] == ["a", "b", "c"]


def test_depths_and_parents():
    inner = Node(id="inner", name="inner", combination="rss", children=(leaf("a"),))
    root = Node(id="root", name="root", combination="rss", children=(inner, leaf("c")))
    assert depths(root) == {"root": 0, "inner": 1, "a": 2, "c": 1}
    assert parents(root) == {"root": None, "inner": "root", "a": "inner", "c": "root"}


def test_is_leaf_flags():
    assert leaf("a").is_leaf is True
    assert Node(id="n", name="n", combination="rss", children=(leaf("a"),)).is_leaf is False

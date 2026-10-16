"""Parser acceptance and rejection paths.

Every rejection names the offending YAML key path; the tests assert on that
text, because a rejection that does not say where the problem is costs more
than it saves.
"""

from __future__ import annotations

import pytest

from budgetroll import load, loads
from budgetroll.errors import DeclarationError

MINIMAL = """
policies:
  p:
    description: d
    source: s
    levels:
      low: 0.05
budget:
  id: b
  name: B
  unit: kg
  quantity: mass
  policy: p
  root:
    id: root
    name: Root
    combination: rss
    children:
      - id: a
        name: A
        value: 1.0
        maturity: low
        uncertainty: 0.1
      - id: c
        name: C
        value: 2.0
        maturity: low
        uncertainty: 0.2
"""


def test_minimal_document_parses():
    budget = loads(MINIMAL)
    assert budget.id == "b"
    assert budget.unit == "kg"
    assert budget.k_sigma == 1.0
    assert [item.id for item in budget.root.children] == ["a", "c"]


def test_defaults_are_applied():
    budget = loads(MINIMAL)
    assert budget.correlations == ()
    assert budget.root.allocation is None
    assert budget.root.independence_rationale == ""


REJECTIONS = [
    ("empty document", "", "the document is empty"),
    ("not a mapping", "- a\n- b", "expected a mapping"),
    (
        "unknown top-level key",
        MINIMAL + "\nextras: 1\n",
        "unknown key\\(s\\) 'extras'",
    ),
    (
        "unknown budget key",
        MINIMAL.replace("  unit: kg", "  unit: kg\n  colour: blue"),
        "budget: unknown key\\(s\\) 'colour'",
    ),
    (
        "unknown node key",
        MINIMAL.replace("    combination: rss", "    combination: rss\n    weight: 3"),
        "budget.root: unknown key\\(s\\) 'weight'",
    ),
    (
        "unknown item key",
        MINIMAL.replace("        maturity: low\n        uncertainty: 0.1", "        maturity: low\n        sigma: 0.1"),
        r"budget.root.children\[0\]: unknown key\(s\) 'sigma'",
    ),
    (
        "missing budget id",
        MINIMAL.replace("  id: b\n", ""),
        "budget: required key 'id' is missing",
    ),
    (
        "missing item value",
        MINIMAL.replace("        value: 1.0\n", ""),
        r"budget.root.children\[0\]: required key 'value' is missing",
    ),
    (
        "missing policies",
        MINIMAL.split("budget:")[1].join(["budget:", ""]),
        "required key 'policies' is missing",
    ),
    (
        "empty policies",
        "policies: {}\n" + MINIMAL.split("budget:", 1)[1].join(["budget:", ""]),
        "at least one margin policy must be declared",
    ),
    (
        "undeclared policy reference",
        MINIMAL.replace("  policy: p", "  policy: q"),
        "budget.policy references 'q'",
    ),
    (
        "unknown maturity",
        MINIMAL.replace("        maturity: low\n        uncertainty: 0.1", "        maturity: mid\n        uncertainty: 0.1"),
        "maturity 'mid' is not declared by margin policy",
    ),
    (
        "root is an item",
        MINIMAL.replace(
            "    combination: rss\n    children:", "    value: 1.0\n    maturity: low\n    unused:"
        ),
        "unknown key",
    ),
    (
        "empty children list",
        MINIMAL.split("    children:")[0] + "    children: []\n",
        "expected a non-empty list",
    ),
    (
        "children not a list",
        MINIMAL.split("    children:")[0] + "    children: 3\n",
        "expected a non-empty list",
    ),
    (
        "non-numeric value",
        MINIMAL.replace("        value: 1.0", "        value: heavy"),
        "expected a number",
    ),
    (
        "boolean value",
        MINIMAL.replace("        value: 1.0", "        value: true"),
        "expected a number",
    ),
    (
        "infinite value",
        MINIMAL.replace("        value: 1.0", "        value: .inf"),
        "expected a finite number",
    ),
    (
        "nan uncertainty",
        MINIMAL.replace("        uncertainty: 0.1", "        uncertainty: .nan"),
        "expected a finite number",
    ),
    (
        "empty name",
        MINIMAL.replace("  name: B", '  name: ""'),
        "expected a non-empty string",
    ),
    (
        "duplicate key in one mapping",
        MINIMAL.replace("        maturity: low\n        uncertainty: 0.1", "        maturity: low\n        maturity: low\n        uncertainty: 0.1"),
        "duplicate key 'maturity'",
    ),
    (
        "duplicate identifier",
        MINIMAL.replace("      - id: c\n        name: C", "      - id: a\n        name: C"),
        "duplicate identifier 'a'",
    ),
    (
        "unknown combination",
        MINIMAL.replace("    combination: rss", "    combination: average"),
        "combination must be one of",
    ),
    (
        "negative uncertainty",
        MINIMAL.replace("        uncertainty: 0.1", "        uncertainty: -0.1"),
        "1 sigma magnitude",
    ),
    (
        "correlation with one item",
        MINIMAL + "\ncorrelations:\n  - items: [a]\n    rho: 0.5\n    rationale: r\n",
        "exactly two leaf identifiers",
    ),
    (
        "correlation with three items",
        MINIMAL + "\ncorrelations:\n  - items: [a, c, a]\n    rho: 0.5\n    rationale: r\n",
        "exactly two leaf identifiers",
    ),
    (
        "correlation unknown key",
        MINIMAL + "\ncorrelations:\n  - items: [a, c]\n    rho: 0.5\n    rationale: r\n    why: x\n",
        r"correlations\[0\]: unknown key\(s\) 'why'",
    ),
    (
        "correlation out of range",
        MINIMAL + "\ncorrelations:\n  - items: [a, c]\n    rho: 1.5\n    rationale: r\n",
        r"must lie in \[-1, 1\]",
    ),
    (
        "correlation missing rationale",
        MINIMAL + "\ncorrelations:\n  - items: [a, c]\n    rho: 0.5\n",
        "required key 'rationale' is missing",
    ),
    (
        "dangling correlation",
        MINIMAL + "\ncorrelations:\n  - items: [a, zz]\n    rho: 0.5\n    rationale: r\n",
        "is not a leaf item",
    ),
    (
        "policy level not numeric",
        MINIMAL.replace("      low: 0.05", "      low: small"),
        "policies.p.levels.low: expected a number",
    ),
    (
        "policy missing source",
        MINIMAL.replace("    source: s\n", ""),
        "policies.p: required key 'source' is missing",
    ),
    (
        "policy unknown key",
        MINIMAL.replace("    source: s", "    source: s\n    author: x"),
        r"policies.p: unknown key\(s\) 'author'",
    ),
    (
        "negative k_sigma",
        MINIMAL.replace("  policy: p", "  policy: p\n  k_sigma: -2"),
        "k_sigma must be >= 0",
    ),
    (
        "invalid yaml",
        "policies: [\n",
        "not valid YAML",
    ),
    (
        "node is not a mapping",
        MINIMAL.split("    children:")[0] + "    children:\n      - 3\n",
        "expected a mapping",
    ),
]


@pytest.mark.parametrize(
    ("label", "text", "pattern"), REJECTIONS, ids=[row[0] for row in REJECTIONS]
)
def test_rejection_paths(label, text, pattern):
    with pytest.raises(DeclarationError, match=pattern):
        loads(text)


def test_rejection_count_is_at_least_thirty():
    # Guards against a parametrisation that silently shrinks.
    assert len(REJECTIONS) >= 30


def test_load_missing_file(tmp_path):
    with pytest.raises(DeclarationError, match="no such budget declaration"):
        load(tmp_path / "absent.yaml")


def test_load_directory(tmp_path):
    with pytest.raises(DeclarationError, match="is a directory"):
        load(tmp_path)


def test_load_non_utf8(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_bytes(b"\xff\xfe\x00budget:")
    with pytest.raises(DeclarationError, match="not valid UTF-8"):
        load(path)


def test_load_round_trips_a_shipped_case(cases_dir):
    budget = load(cases_dir / "clean_mass.yaml")
    assert budget.id == "clean_mass"
    assert budget.unit == "kg"


def test_item_with_explicit_contingency_and_no_maturity_parses():
    text = MINIMAL.replace("        maturity: low\n        uncertainty: 0.1", "        contingency: 0.3\n        uncertainty: 0.1")
    budget = loads(text)
    assert budget.root.children[0].contingency == 0.3
    assert budget.root.children[0].maturity is None


def test_common_driver_is_parsed():
    text = MINIMAL.replace("        uncertainty: 0.1", "        uncertainty: 0.1\n        common_driver: vendor_x")
    assert loads(text).root.children[0].common_driver == "vendor_x"

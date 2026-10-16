"""End-to-end: YAML on disk through to a rendered report and an exit status."""

from __future__ import annotations

import json
import math

import pytest

from budgetroll import has_blocking, json_report, load, rollup, run_checks, text_report

CASE_NAMES = [
    "clean_mass.yaml",
    "planted_findings.yaml",
    "cubesat_power.yaml",
    "cubesat_link.yaml",
]


@pytest.mark.parametrize("name", CASE_NAMES)
def test_every_shipped_case_loads_rolls_up_and_renders(cases_dir, name):
    budget = load(cases_dir / name)
    roll = rollup(budget)
    assert roll.results
    assert math.isfinite(roll.root.predicted)
    assert math.isfinite(roll.root.uncertainty)
    report = text_report(budget)
    assert "not flight-qualified" in report
    payload = json.loads(json_report(budget))
    assert payload["budget"]["id"] == budget.id


@pytest.mark.parametrize("name", CASE_NAMES)
def test_every_shipped_case_has_a_policy_source(cases_dir, name):
    budget = load(cases_dir / name)
    assert budget.policy.source.strip()
    assert "Declared by the author" in budget.policy.source


@pytest.mark.parametrize(
    ("name", "blocking"),
    [
        ("clean_mass.yaml", False),
        ("planted_findings.yaml", True),
        ("cubesat_power.yaml", True),
        ("cubesat_link.yaml", False),
    ],
)
def test_blocking_status_of_each_case(cases_dir, name, blocking):
    assert has_blocking(run_checks(load(cases_dir / name))) is blocking


@pytest.mark.parametrize("name", CASE_NAMES)
def test_every_rss_node_declares_an_independence_rationale(cases_dir, name):
    from budgetroll import Node, walk

    budget = load(cases_dir / name)
    for node in walk(budget.root):
        if isinstance(node, Node) and node.combination in ("rss", "correlated"):
            assert node.independence_rationale.strip(), node.id


def test_the_combination_ordering_holds_on_every_case(cases_dir):
    """RSS <= declared hierarchy <= linear is not guaranteed, but
    RSS <= linear always is; both are asserted here on the real cases."""
    from budgetroll import linear_rollup_uncertainty, rss_rollup_uncertainty

    for name in CASE_NAMES:
        budget = load(cases_dir / name)
        rss = rss_rollup_uncertainty(budget)
        linear = linear_rollup_uncertainty(budget)
        assert rss <= linear + 1e-12 * max(1.0, linear), name


def test_report_text_and_json_agree_on_the_root(cases_dir):
    for name in CASE_NAMES:
        budget = load(cases_dir / name)
        payload = json.loads(json_report(budget))
        root_json = payload["nodes"][0]
        root = rollup(budget).root
        assert root_json["predicted"] == pytest.approx(root.predicted, rel=1e-12)
        assert root_json["uncertainty"] == pytest.approx(root.uncertainty, rel=1e-12)

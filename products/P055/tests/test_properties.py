"""Hypothesis property tests over generated assurance cases.

The properties here are graph identities, not numerical ones, so they are exact
rather than tolerance-bounded:

* a graph generated as a DAG never reports a cycle;
* adding an orphan node never changes the unsupported-claim findings;
* the renderer emits exactly one declaration per node and one per edge;
* reachability is monotone in the root set.

Generated cases are small (at most 14 nodes) so the whole module runs in a
couple of seconds; the properties do not get truer with larger graphs, only
slower.
"""

from __future__ import annotations

from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from assuregraph import (
    check_orphan_nodes,
    check_unsupported_claims,
    find_cycles,
    node_depths,
    parse_case,
    reachable_from,
    render_mermaid,
    run_checks,
    sha256_file,
)

_FAST = settings(max_examples=60, deadline=None)


@st.composite
def dag_document(draw: Any) -> dict[str, Any]:
    """Generate a well-formed GSN case that is a DAG by construction.

    Claims are laid out in a line ``G0, G1, ... Gn-1`` alternating Goal and
    Strategy, and every ``SupportedBy`` edge runs from a lower index to a higher
    one, which makes a cycle impossible. Solutions, Contexts and Assumptions are
    attached as leaves under randomly chosen claims, respecting the Core GSN
    relationship table.
    """
    claim_count = draw(st.integers(min_value=1, max_value=8))
    kinds = ["goal" if i % 2 == 0 else "strategy" for i in range(claim_count)]
    nodes: list[dict[str, Any]] = [
        {"id": f"N{i}", "type": kinds[i], "statement": f"claim {i}"} for i in range(claim_count)
    ]
    edges: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    for target in range(1, claim_count):
        source = draw(st.integers(min_value=0, max_value=target - 1))
        key = (f"N{source}", f"N{target}", "supported_by")
        if key not in seen:
            seen.add(key)
            edges.append({"from": key[0], "to": key[1], "type": "supported_by"})

    leaf_count = draw(st.integers(min_value=0, max_value=6))
    for index in range(leaf_count):
        leaf_kind = draw(st.sampled_from(["solution", "context", "assumption", "justification"]))
        leaf_id = f"L{index}"
        entry: dict[str, Any] = {
            "id": leaf_id,
            "type": leaf_kind,
            "statement": f"leaf {index}",
        }
        if leaf_kind == "solution":
            entry["evidence"] = {"path": f"evidence/{leaf_id}.txt"}
        nodes.append(entry)
        parent = f"N{draw(st.integers(min_value=0, max_value=claim_count - 1))}"
        relation = "supported_by" if leaf_kind == "solution" else "in_context_of"
        key = (parent, leaf_id, relation)
        if key not in seen:
            seen.add(key)
            edges.append({"from": parent, "to": leaf_id, "type": relation})

    # Mark some claims undeveloped so the INFO path is exercised too.
    for entry in nodes[:claim_count]:
        if draw(st.booleans()):
            entry["undeveloped"] = True

    return {"name": "generated", "nodes": nodes, "edges": edges, "top_goals": ["N0"]}


@given(dag_document())
@_FAST
def test_a_dag_never_reports_a_cycle(doc: dict[str, Any]) -> None:
    case = parse_case(doc, base_dir="/nonexistent-base")
    assert find_cycles(case) == []


@given(dag_document())
@_FAST
def test_a_dag_run_reports_no_cycle_finding(doc: dict[str, Any]) -> None:
    case = parse_case(doc, base_dir="/nonexistent-base")
    assert run_checks(case).by_check("cycles") == ()


@given(dag_document(), st.integers(min_value=1, max_value=4))
@_FAST
def test_adding_orphans_never_changes_unsupported_claim_findings(
    doc: dict[str, Any], orphan_count: int
) -> None:
    base = parse_case(doc, base_dir="/nonexistent-base")
    before = {(f.severity, f.node_ids) for f in check_unsupported_claims(base)}

    grown = {
        "name": doc["name"],
        "nodes": [
            *doc["nodes"],
            *(
                {
                    "id": f"ORPHAN{i}",
                    "type": "context",
                    "statement": f"detached context {i}",
                }
                for i in range(orphan_count)
            ),
        ],
        "edges": list(doc["edges"]),
        "top_goals": doc["top_goals"],
    }
    after = {(f.severity, f.node_ids) for f in check_unsupported_claims(parse_case(grown))}
    assert after == before


@given(dag_document(), st.integers(min_value=1, max_value=4))
@_FAST
def test_adding_orphans_increases_the_orphan_count_by_exactly_that_many(
    doc: dict[str, Any], orphan_count: int
) -> None:
    base = parse_case(doc, base_dir="/nonexistent-base")
    before = len(check_orphan_nodes(base))
    grown = {
        "name": doc["name"],
        "nodes": [
            *doc["nodes"],
            *(
                {"id": f"ORPHAN{i}", "type": "context", "statement": "detached"}
                for i in range(orphan_count)
            ),
        ],
        "edges": list(doc["edges"]),
        "top_goals": doc["top_goals"],
    }
    after = len(check_orphan_nodes(parse_case(grown)))
    assert after == before + orphan_count


@given(dag_document())
@_FAST
def test_renderer_emits_one_line_per_node_and_per_edge(doc: dict[str, Any]) -> None:
    case = parse_case(doc, base_dir="/nonexistent-base")
    lines = [ln.strip() for ln in render_mermaid(case).splitlines() if ln.strip().startswith("n_")]
    node_lines = [ln for ln in lines if " --" not in ln]
    edge_lines = [ln for ln in lines if " --" in ln]
    assert len(node_lines) == len(case.nodes)
    assert len(edge_lines) == len(case.edges)


@given(dag_document())
@_FAST
def test_reachability_is_monotone_in_the_root_set(doc: dict[str, Any]) -> None:
    case = parse_case(doc, base_dir="/nonexistent-base")
    ids = list(case.nodes)
    small = reachable_from(case, ids[:1])
    large = reachable_from(case, ids[:2])
    assert small <= large


@given(dag_document())
@_FAST
def test_depth_never_exceeds_the_node_count(doc: dict[str, Any]) -> None:
    case = parse_case(doc, base_dir="/nonexistent-base")
    depths = node_depths(case)
    assert not depths or max(depths.values()) < len(case.nodes)


@given(dag_document())
@_FAST
def test_node_count_and_edge_count_survive_parsing(doc: dict[str, Any]) -> None:
    case = parse_case(doc, base_dir="/nonexistent-base")
    assert len(case.nodes) == len({n["id"] for n in doc["nodes"]})
    assert len(case.edges) == len(doc["edges"])


@given(dag_document())
@_FAST
def test_coverage_counts_sum_to_the_node_total(doc: dict[str, Any]) -> None:
    case = parse_case(doc, base_dir="/nonexistent-base")
    cov = run_checks(case).coverage
    assert sum(cov.node_counts.values()) == len(case.nodes)
    assert sum(cov.edge_counts.values()) == len(case.edges)


@given(dag_document())
@_FAST
def test_claims_supported_plus_unsupported_equals_claims_total(doc: dict[str, Any]) -> None:
    case = parse_case(doc, base_dir="/nonexistent-base")
    report = run_checks(case)
    cov = report.coverage
    unsupported = len(report.by_check("unsupported_claims"))
    assert cov.claims_supported + unsupported == cov.claims_total


@given(st.binary(min_size=0, max_size=4096))
@_FAST
def test_sha256_file_is_64_hex_characters(tmp_path_factory, payload: bytes) -> None:
    target = tmp_path_factory.mktemp("hash") / "a.bin"
    target.write_bytes(payload)
    digest = sha256_file(str(target))
    assert len(digest) == 64
    assert set(digest) <= set("0123456789abcdef")


@given(dag_document())
@_FAST
def test_node_order_does_not_change_the_finding_set(doc: dict[str, Any]) -> None:
    base = parse_case(doc, base_dir="/nonexistent-base")
    reversed_doc = {
        "name": doc["name"],
        "nodes": list(reversed(doc["nodes"])),
        "edges": list(reversed(doc["edges"])),
        "top_goals": doc["top_goals"],
    }
    other = parse_case(reversed_doc, base_dir="/nonexistent-base")
    as_set = lambda case: {  # noqa: E731
        (f.check, f.severity, f.node_ids) for f in run_checks(case).findings
    }
    assert as_set(base) == as_set(other)

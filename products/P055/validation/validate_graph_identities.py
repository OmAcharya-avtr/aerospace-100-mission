"""Randomised check of the graph identities the checks rest on.

Three identities, each checked over many random graphs with a fixed seed so the
run is reproducible:

1. **A DAG never reports a cycle.** Graphs are generated with every
   ``SupportedBy`` edge running from a lower node index to a higher one, which
   makes a cycle impossible by construction. Any cycle reported is a false
   positive in :func:`assuregraph.graph.find_cycles`.
2. **A graph with a planted cycle always reports one.** One back-edge is added
   to an otherwise acyclic graph; at least one cyclic region must be reported,
   and the reported walk must be a closed walk whose consecutive pairs are all
   real edges of the graph.
3. **Adding an orphan never changes the unsupported-claim findings, and raises
   the orphan count by exactly the number added.** This is the independence
   property the two checks are supposed to have.

Method: Tarjan's algorithm (Tarjan 1972, "Depth-first search and linear graph
algorithms", SIAM Journal on Computing 1(2)) for the components; breadth-first
search for reachability. Both are exact, so these are pass/fail identities with
no tolerance, not numerical comparisons.

Run: ``python validation/validate_graph_identities.py``
Output: ``validation/validate_graph_identities_output.txt``
"""

from __future__ import annotations

import random
from typing import Any

import _bootstrap  # noqa: F401

from assuregraph import (
    check_orphan_nodes,
    check_unsupported_claims,
    find_cycles,
    parse_case,
)

SEED = 20261008
TRIALS = 400


def random_dag(rng: random.Random, max_claims: int = 14) -> dict[str, Any]:
    """Generate a well-formed GSN case that is acyclic by construction."""
    claim_count = rng.randint(1, max_claims)
    nodes: list[dict[str, Any]] = [
        {
            "id": f"N{i}",
            "type": "goal" if i % 2 == 0 else "strategy",
            "statement": f"claim {i}",
            **({"undeveloped": True} if rng.random() < 0.3 else {}),
        }
        for i in range(claim_count)
    ]
    edges: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for target in range(1, claim_count):
        for _ in range(rng.randint(1, 2)):
            source = rng.randint(0, target - 1)
            key = (f"N{source}", f"N{target}", "supported_by")
            if key not in seen:
                seen.add(key)
                edges.append({"from": key[0], "to": key[1], "type": key[2]})
    for index in range(rng.randint(0, 6)):
        kind = rng.choice(["solution", "context", "assumption", "justification"])
        leaf_id = f"L{index}"
        entry: dict[str, Any] = {"id": leaf_id, "type": kind, "statement": f"leaf {index}"}
        if kind == "solution":
            entry["evidence"] = {"path": f"evidence/{leaf_id}.txt"}
        nodes.append(entry)
        parent = f"N{rng.randint(0, claim_count - 1)}"
        relation = "supported_by" if kind == "solution" else "in_context_of"
        key = (parent, leaf_id, relation)
        if key not in seen:
            seen.add(key)
            edges.append({"from": key[0], "to": key[1], "type": key[2]})
    return {"name": "random dag", "nodes": nodes, "edges": edges, "top_goals": ["N0"]}


def plant_a_cycle(document: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    """Add one back-edge between two claims, creating at least one cycle."""
    claims = [n["id"] for n in document["nodes"] if n["id"].startswith("N")]
    if len(claims) < 2:
        return None
    existing = {(e["from"], e["to"], e["type"]) for e in document["edges"]}
    forward = [
        (e["from"], e["to"])
        for e in document["edges"]
        if e["type"] == "supported_by" and e["from"].startswith("N") and e["to"].startswith("N")
    ]
    if not forward:
        return None
    source, target = rng.choice(forward)
    key = (target, source, "supported_by")
    if key in existing:
        return None
    grown = {
        "name": document["name"],
        "nodes": list(document["nodes"]),
        "edges": [*document["edges"], {"from": key[0], "to": key[1], "type": key[2]}],
        "top_goals": document["top_goals"],
    }
    return grown


def add_orphans(document: dict[str, Any], count: int) -> dict[str, Any]:
    """Add ``count`` detached Context nodes, reachable from nothing."""
    return {
        "name": document["name"],
        "nodes": [
            *document["nodes"],
            *(
                {"id": f"ORPHAN{i}", "type": "context", "statement": "detached"}
                for i in range(count)
            ),
        ],
        "edges": list(document["edges"]),
        "top_goals": document["top_goals"],
    }


def main() -> int:
    rng = random.Random(SEED)
    lines = [
        "Graph identities over random GSN cases -- assuregraph 0.1.0",
        "",
        f"seed = {SEED}, trials = {TRIALS}. Identities are exact; no tolerance applies.",
        "",
    ]

    false_positive_cycles = 0
    dag_nodes = 0
    dag_edges = 0
    for _ in range(TRIALS):
        document = random_dag(rng)
        case = parse_case(document, base_dir="/nonexistent-base")
        dag_nodes += len(case.nodes)
        dag_edges += len(case.edges)
        if find_cycles(case):
            false_positive_cycles += 1

    planted = 0
    missed_cycles = 0
    bad_walks = 0
    for _ in range(TRIALS):
        document = random_dag(rng)
        grown = plant_a_cycle(document, rng)
        if grown is None:
            continue
        planted += 1
        case = parse_case(grown, base_dir="/nonexistent-base")
        cycles = find_cycles(case)
        if not cycles:
            missed_cycles += 1
            continue
        real_edges = {(e.source, e.target) for e in case.edges}
        for walk in cycles:
            if walk[0] != walk[-1]:
                bad_walks += 1
                break
            if any((a, b) not in real_edges for a, b in zip(walk[:-1], walk[1:], strict=True)):
                bad_walks += 1
                break

    unsupported_changed = 0
    orphan_count_wrong = 0
    for _ in range(TRIALS):
        document = random_dag(rng)
        base = parse_case(document, base_dir="/nonexistent-base")
        before_unsupported = {
            (f.severity, f.node_ids) for f in check_unsupported_claims(base)
        }
        before_orphans = len(check_orphan_nodes(base))
        added = rng.randint(1, 5)
        grown = parse_case(add_orphans(document, added), base_dir="/nonexistent-base")
        after_unsupported = {
            (f.severity, f.node_ids) for f in check_unsupported_claims(grown)
        }
        after_orphans = len(check_orphan_nodes(grown))
        if after_unsupported != before_unsupported:
            unsupported_changed += 1
        if after_orphans != before_orphans + added:
            orphan_count_wrong += 1

    failures = false_positive_cycles + missed_cycles + bad_walks + unsupported_changed
    failures += orphan_count_wrong
    lines.extend(
        [
            "1. A DAG never reports a cycle",
            f"   DAGs generated                       : {TRIALS}",
            f"   mean nodes per DAG                   : {dag_nodes / TRIALS:.2f}",
            f"   mean edges per DAG                   : {dag_edges / TRIALS:.2f}",
            f"   DAGs for which a cycle was reported  : {false_positive_cycles}",
            f"   false-positive rate                  : {false_positive_cycles / TRIALS:.6f}",
            "",
            "2. A planted cycle is always found, and the reported walk is a real walk",
            f"   graphs with a back-edge planted      : {planted}",
            f"   graphs where no cycle was reported   : {missed_cycles}",
            f"   false-negative rate                  : "
            f"{(missed_cycles / planted) if planted else float('nan'):.6f}",
            f"   reported walks that were not closed, "
            f"or used a non-existent edge             : {bad_walks}",
            "",
            "3. Orphans are independent of the unsupported-claim check",
            f"   trials                               : {TRIALS}",
            f"   trials where the unsupported-claim finding set changed : "
            f"{unsupported_changed}",
            f"   trials where the orphan count moved by the wrong amount : "
            f"{orphan_count_wrong}",
            "",
            f"FAILURES: {failures}",
        ]
    )
    target = __file__.replace(".py", "_output.txt")
    with open(target, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

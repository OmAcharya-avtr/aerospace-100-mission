"""Generate the committed replayed-change-history corpora. Run ONCE, then commit.

This is a DATA GENERATOR and it is deliberately not under ``validation/``. The
release gate runs ``sorted(glob('validation/*.py'))``, so alphabetically
adjacent validation scripts run back to back; a validation script that
regenerated a corpus another validation script pins by hash would be collided
by the gate's own ordering. That has blocked a product of this mission before.
So: this script writes ``data/corpus_layered.json``,
``data/corpus_cyclic.json`` and ``data/corpus_manifest.json``, those files are
committed, and ``validation/validate_corpus_pin.py`` only READS and hashes
them.

Determinism: one declared seed, ``numpy.random.default_rng`` (PCG64), no other
source of randomness, no dependence on dict iteration order anywhere (every
set is sorted before use). Re-running this script reproduces the committed
files byte for byte, and ``validate_corpus_pin.py`` checks the committed
digests against the committed files.

WHAT IS BEING MODELLED
----------------------
A layered artifact set of the shape an airborne-software assurance case rests
on: configuration and source artifacts at layer 0, built or derived artifacts
at layer 1, and verification reports and analyses at layer 2. Dependency edges
point from a dependent to what it was derived from, which is the direction
DO-178C (RTCA, 2011) traceability runs in when it requires that verification
results be traceable to the requirements, design and code they were produced
against. The numbers here model nothing measured: they are a synthetic corpus
whose only purpose is that each item's true staleness is known by construction.

THE SECOND CORPUS IS CYCLIC ON PURPOSE. An assurance-case dependency graph can
contain a cycle through a shared artifact -- a generated interface header a
tool both reads and writes, or two models each calibrated against the other.
``corpus_cyclic.json`` contains a three-artifact cycle so the whole pipeline is
exercised on a graph that is not a directed acyclic graph, rather than the
cyclic path being tested only in unit tests.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))

from evidencestale.graph import DependencyGraph  # noqa: E402
from evidencestale.model import ChangeEvent  # noqa: E402
from evidencestale.replay import ChangeHistory, true_stale_labels  # noqa: E402

#: The one declared seed for the whole corpus.
SEED = 980098

#: Layer sizes. 80 + 80 + 80 = 240 artifacts.
LAYER_SIZES = (80, 80, 80)

#: Maximum number of direct dependents any artifact may have, enforced during
#: construction. This is the ``d`` the independent bound consumes.
MAX_DEPENDENTS = 2

#: Maximum number of direct dependencies any artifact may declare.
MAX_DEPENDENCIES = 2

#: Declared maximum impact depth. Three layers with edges only between
#: adjacent layers gives a longest dependency path of 2, so the closure and the
#: impact set are both reached within 2 levels. The claim is CHECKED by
#: validate_corpus_pin.py against the generated graph, not assumed.
DECLARED_MAX_IMPACT_DEPTH = 2

#: Evidence items per artifact. Every artifact is the subject of this many
#: items, so the maximum items per subject -- the ``m`` the bound consumes --
#: is exactly this number.
ITEMS_PER_ARTIFACT = 2

#: Revision at which the audit is performed.
EVALUATION_TIME = 2000

#: Evidence production revisions are drawn from this inclusive range.
PRODUCTION_RANGE = (1, 600)

#: Number of change events in the main corpus.
EVENT_COUNT = 250

#: Event-kind mixture. A ``revert`` is a write whose content is a value the
#: artifact held earlier, which is NOT a change in value and so is not
#: staleness; it is in the corpus because an event-counting method gets it
#: wrong and a value-comparing method does not.
KIND_WEIGHTS = {"write": 0.08, "revert": 0.06, "touch": 0.86}

#: Per-layer probability that an event lands on that layer. Weighted towards
#: the lower layers because that is where a change reaches an upper-layer
#: artifact through a dependency, which is the case baseline 1 misses.
LAYER_EVENT_WEIGHTS = (0.80, 0.15, 0.05)

#: Fraction of true edges withheld from the declared edge set, which is what an
#: incomplete dependency declaration looks like.
HIDDEN_EDGE_FRACTION = 0.30

#: Solutions per leaf goal, and leaf goals per strategy, in the GSN structure.
SOLUTIONS_PER_LEAF_GOAL = 4
LEAF_GOALS_PER_STRATEGY = 10


def artifact_names() -> tuple[list[str], list[list[str]]]:
    """All artifact identifiers and the per-layer lists."""
    layers: list[list[str]] = []
    index = 0
    for layer, size in enumerate(LAYER_SIZES):
        names = [f"a{index + offset:03d}L{layer}" for offset in range(size)]
        index += size
        layers.append(names)
    return [name for layer in layers for name in layer], layers


def build_edges(rng: np.random.Generator, layers: list[list[str]]) -> list[tuple[str, str]]:
    """Edges from each artifact to 1..MAX_DEPENDENCIES artifacts one layer below.

    Dependents per artifact are capped at :data:`MAX_DEPENDENTS` by rejecting a
    candidate whose capacity is exhausted. The capacity check is what makes the
    declared ``d`` true of the generated graph rather than hoped for.
    """
    edges: list[tuple[str, str]] = []
    dependents: dict[str, int] = {}
    for layer in range(1, len(layers)):
        below = layers[layer - 1]
        for dependent in layers[layer]:
            wanted = int(rng.integers(1, MAX_DEPENDENCIES + 1))
            order = list(rng.permutation(len(below)))
            chosen: list[str] = []
            for position in order:
                candidate = below[position]
                if dependents.get(candidate, 0) >= MAX_DEPENDENTS:
                    continue
                chosen.append(candidate)
                dependents[candidate] = dependents.get(candidate, 0) + 1
                if len(chosen) == wanted:
                    break
            for dependency in chosen:
                edges.append((dependent, dependency))
    return sorted(edges)


def build_events(
    rng: np.random.Generator, layers: list[list[str]], count: int, horizon: int
) -> list[ChangeEvent]:
    """A change history of ``count`` events on distinct ``(time, artifact)`` pairs."""
    all_names = [name for layer in layers for name in layer]
    weights = np.zeros(len(all_names), dtype=float)
    offset = 0
    for layer, names in enumerate(layers):
        weights[offset : offset + len(names)] = LAYER_EVENT_WEIGHTS[layer] / len(names)
        offset += len(names)
    weights /= weights.sum()

    kinds = sorted(KIND_WEIGHTS)
    kind_probabilities = np.asarray([KIND_WEIGHTS[kind] for kind in kinds], dtype=float)
    kind_probabilities /= kind_probabilities.sum()

    history: dict[str, list[str]] = {name: [f"{name}:v000"] for name in all_names}
    used: set[tuple[int, str]] = set()
    events: list[ChangeEvent] = []
    attempts = 0
    while len(events) < count and attempts < count * 20:
        attempts += 1
        time = int(rng.integers(1, horizon + 1))
        artifact = all_names[int(rng.choice(len(all_names), p=weights))]
        if (time, artifact) in used:
            continue
        used.add((time, artifact))
        kind = kinds[int(rng.choice(len(kinds), p=kind_probabilities))]
        if kind == "touch":
            events.append(ChangeEvent(time=time, artifact=artifact, kind="touch"))
            continue
        if kind == "revert" and len(history[artifact]) > 1:
            previous = history[artifact][int(rng.integers(0, len(history[artifact]) - 1))]
            events.append(
                ChangeEvent(time=time, artifact=artifact, kind="write", content=previous)
            )
            continue
        token = f"{artifact}:v{len(history[artifact]):03d}"
        history[artifact].append(token)
        events.append(ChangeEvent(time=time, artifact=artifact, kind="write", content=token))
    return sorted(events, key=lambda event: (event.time, event.artifact, event.kind))


def build_assurance_case(
    evidence_rows: list[dict[str, object]],
) -> dict[str, object]:
    """A GSN structure over the evidence: one top goal, strategies, leaf goals.

    Nodes and links are the subset of GSN this product models, and the loader
    validates them against the GSN restrictions in
    ``src/evidencestale/model.py``.
    """
    nodes: dict[str, str] = {"G000": "Goal", "C000": "Context", "A000": "Assumption"}
    links: list[tuple[str, str, str]] = [
        ("InContextOf", "G000", "C000"),
        ("InContextOf", "G000", "A000"),
    ]
    leaf_goals: list[str] = []
    for position, row in enumerate(evidence_rows):
        solution = str(row["solution"])
        nodes[solution] = "Solution"
        leaf_index = position // SOLUTIONS_PER_LEAF_GOAL
        leaf_goal = f"G1{leaf_index:03d}"
        if leaf_goal not in nodes:
            nodes[leaf_goal] = "Goal"
            leaf_goals.append(leaf_goal)
        links.append(("SupportedBy", leaf_goal, solution))
        row["goal"] = leaf_goal
    for leaf_index, leaf_goal in enumerate(leaf_goals):
        strategy = f"S{leaf_index // LEAF_GOALS_PER_STRATEGY:03d}"
        if strategy not in nodes:
            nodes[strategy] = "Strategy"
            links.append(("SupportedBy", "G000", strategy))
        links.append(("SupportedBy", strategy, leaf_goal))
    return {"nodes": nodes, "links": [list(link) for link in links]}


def generate_layered() -> dict[str, object]:
    """The main corpus: three layers, acyclic, 240 artifacts, 480 evidence items."""
    rng = np.random.default_rng(SEED)
    all_names, layers = artifact_names()
    edges = build_edges(rng, layers)
    events = build_events(rng, layers, EVENT_COUNT, EVALUATION_TIME)

    hidden_count = int(round(HIDDEN_EDGE_FRACTION * len(edges)))
    hidden_positions = sorted(
        int(position)
        for position in rng.choice(len(edges), size=hidden_count, replace=False)
    )
    hidden = {edges[position] for position in hidden_positions}
    declared = [edge for edge in edges if edge not in hidden]

    evidence_rows: list[dict[str, object]] = []
    counter = 0
    for name in all_names:
        for _repeat in range(ITEMS_PER_ARTIFACT):
            produced = int(rng.integers(PRODUCTION_RANGE[0], PRODUCTION_RANGE[1] + 1))
            evidence_rows.append(
                {
                    "identifier": f"e{counter:04d}",
                    "subject": name,
                    "produced_at": produced,
                    "solution": f"Sn{counter:04d}",
                    "goal": "",
                }
            )
            counter += 1
    case = build_assurance_case(evidence_rows)

    true_graph = DependencyGraph.build(all_names, edges)
    history = ChangeHistory(
        all_names, {name: f"{name}:v000" for name in all_names}, events
    )
    labels = true_stale_labels(
        history,
        true_graph,
        [
            (str(row["identifier"]), str(row["subject"]), int(row["produced_at"]))
            for row in evidence_rows
        ],
        EVALUATION_TIME,
    )
    for row in evidence_rows:
        row["true_stale"] = bool(labels[str(row["identifier"])])

    return {
        "schema": "evidencestale/corpus/1",
        "name": "layered",
        "seed": SEED,
        "artifacts": all_names,
        "initial_contents": {name: f"{name}:v000" for name in all_names},
        "true_edges": [list(edge) for edge in edges],
        "declared_edges": [list(edge) for edge in declared],
        "events": [
            {
                "time": event.time,
                "artifact": event.artifact,
                "kind": event.kind,
                "content": event.content,
            }
            for event in events
        ],
        "evidence": evidence_rows,
        "assurance_case": case,
        "evaluation_time": EVALUATION_TIME,
        "declared_max_dependents": MAX_DEPENDENTS,
        "declared_max_impact_depth": DECLARED_MAX_IMPACT_DEPTH,
    }


def generate_cyclic() -> dict[str, object]:
    """A deliberately cyclic corpus: 12 artifacts with a three-artifact cycle.

    The cycle is ``x -> y -> z -> x``, which models a generated interface
    artifact that a tool both reads and writes. The closure of any cycle member
    contains the whole cycle, which is the correct answer: each member really
    is reachable from each other member.
    """
    names = [f"c{index:02d}" for index in range(12)]
    edges = [
        ("c00", "c01"),
        ("c01", "c02"),
        ("c02", "c00"),
        ("c03", "c00"),
        ("c04", "c01"),
        ("c05", "c02"),
        ("c06", "c03"),
        ("c07", "c04"),
        ("c08", "c05"),
        ("c09", "c06"),
        ("c10", "c07"),
        ("c11", "c08"),
    ]
    events = [
        ChangeEvent(time=5, artifact="c01", kind="write", content="c01:v001"),
        ChangeEvent(time=7, artifact="c09", kind="touch"),
        ChangeEvent(time=9, artifact="c02", kind="write", content="c02:v001"),
        ChangeEvent(time=11, artifact="c02", kind="write", content="c02:v000"),
        ChangeEvent(time=13, artifact="c05", kind="touch"),
        ChangeEvent(time=15, artifact="c08", kind="write", content="c08:v001"),
    ]
    evidence_rows: list[dict[str, object]] = []
    for counter, name in enumerate(names):
        evidence_rows.append(
            {
                "identifier": f"y{counter:02d}",
                "subject": name,
                "produced_at": 1 + 2 * (counter % 4),
                "solution": f"Sc{counter:02d}",
                "goal": "",
            }
        )
    case = build_assurance_case(evidence_rows)
    graph = DependencyGraph.build(names, edges)
    history = ChangeHistory(names, {name: f"{name}:v000" for name in names}, events)
    labels = true_stale_labels(
        history,
        graph,
        [
            (str(row["identifier"]), str(row["subject"]), int(row["produced_at"]))
            for row in evidence_rows
        ],
        20,
    )
    for row in evidence_rows:
        row["true_stale"] = bool(labels[str(row["identifier"])])
    return {
        "schema": "evidencestale/corpus/1",
        "name": "cyclic",
        "seed": SEED,
        "artifacts": names,
        "initial_contents": {name: f"{name}:v000" for name in names},
        "true_edges": [list(edge) for edge in edges],
        "declared_edges": [list(edge) for edge in edges],
        "events": [
            {
                "time": event.time,
                "artifact": event.artifact,
                "kind": event.kind,
                "content": event.content,
            }
            for event in events
        ],
        "evidence": evidence_rows,
        "assurance_case": case,
        "evaluation_time": 20,
        "declared_max_dependents": 2,
        # The longest dependency path here runs c09 -> c06 -> c03 -> c00 and
        # then around the three-artifact cycle c00 -> c01 -> c02, which is five
        # breadth-first levels. 6 is declared so the constant is a true bound on
        # the generated graph; validate_corpus_pin.py measures it rather than
        # trusting it, and an earlier value of 4 was caught by that check.
        "declared_max_impact_depth": 6,
    }


def write(document: dict[str, object], name: str) -> tuple[str, int, str]:
    """Write ``document`` to ``data/<name>`` and return (name, bytes, digest)."""
    target = ROOT / "data" / name
    text = json.dumps(document, indent=1, sort_keys=True) + "\n"
    target.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return name, len(text.encode("utf-8")), digest


def main() -> int:
    """Generate both corpora, write them and write the manifest."""
    lines: list[str] = []

    def say(text: str = "") -> None:
        lines.append(text)
        print(text)

    say("Corpus generation -- P098 EvidenceStale")
    say("")
    say(f"seed: {SEED}")
    say(f"layer sizes: {LAYER_SIZES}")
    say(f"max dependents per artifact (d): {MAX_DEPENDENTS}")
    say(f"declared max impact depth (L): {DECLARED_MAX_IMPACT_DEPTH}")
    say(f"evidence items per artifact (m): {ITEMS_PER_ARTIFACT}")
    say(f"events requested: {EVENT_COUNT}")
    say(f"event kind weights: {KIND_WEIGHTS}")
    say(f"hidden edge fraction: {HIDDEN_EDGE_FRACTION}")
    say(f"evaluation revision: {EVALUATION_TIME}")
    say("")

    manifest: dict[str, object] = {"seed": SEED, "files": {}}
    for document, name in (
        (generate_layered(), "corpus_layered.json"),
        (generate_cyclic(), "corpus_cyclic.json"),
    ):
        file_name, size, digest = write(document, name)
        stale = sum(1 for row in document["evidence"] if row["true_stale"])  # type: ignore[index]
        total = len(document["evidence"])  # type: ignore[arg-type]
        manifest["files"][file_name] = {  # type: ignore[index]
            "sha256": digest,
            "bytes": size,
            "artifacts": len(document["artifacts"]),  # type: ignore[arg-type]
            "true_edges": len(document["true_edges"]),  # type: ignore[arg-type]
            "declared_edges": len(document["declared_edges"]),  # type: ignore[arg-type]
            "events": len(document["events"]),  # type: ignore[arg-type]
            "evidence": total,
            "true_stale": stale,
            "true_stale_fraction": stale / total,
        }
        say(f"{file_name}")
        say(f"  bytes            : {size}")
        say(f"  sha256           : {digest}")
        say(f"  artifacts        : {len(document['artifacts'])}")  # type: ignore[arg-type]
        say(f"  true edges       : {len(document['true_edges'])}")  # type: ignore[arg-type]
        say(f"  declared edges   : {len(document['declared_edges'])}")  # type: ignore[arg-type]
        say(f"  events           : {len(document['events'])}")  # type: ignore[arg-type]
        say(f"  evidence items   : {total}")
        say(f"  true stale items : {stale}")
        say(f"  true stale frac  : {stale / total:.6f}")
        say("")

    manifest_text = json.dumps(manifest, indent=1, sort_keys=True) + "\n"
    (ROOT / "data" / "corpus_manifest.json").write_text(manifest_text, encoding="utf-8")
    say("data/corpus_manifest.json written")
    say(f"  bytes : {len(manifest_text.encode('utf-8'))}")
    say("")
    say("These files are COMMITTED. No validation script regenerates them;")
    say("validation/validate_corpus_pin.py reads and hashes them only.")
    (ROOT / "data" / "generate_corpus_output.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

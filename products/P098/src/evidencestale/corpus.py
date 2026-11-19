"""Loading and validating the committed replayed-change-history corpus.

The corpus is generated ONCE by ``data/generate_corpus.py``, committed, and
hashed. No validation script regenerates it. That separation is deliberate: the
release gate runs ``sorted(glob('validation/*.py'))``, so alphabetically
adjacent scripts run back to back, and a script that regenerates data another
script pins would collide with it.

The corpus carries both a TRUE edge set and a DECLARED edge set. The declared
set is a strict subset, which is what an incomplete dependency declaration
looks like in practice. The ground-truth labels are always over the true set.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .graph import DependencyGraph
from .model import AssuranceCase, ChangeEvent, EvidenceItem
from .replay import ChangeHistory

#: Schema identifier written by the generator and required by the loader.
SCHEMA = "evidencestale/corpus/1"

#: Keys every corpus document must carry.
REQUIRED_KEYS: tuple[str, ...] = (
    "schema",
    "name",
    "seed",
    "artifacts",
    "initial_contents",
    "true_edges",
    "declared_edges",
    "events",
    "evidence",
    "assurance_case",
    "evaluation_time",
    "declared_max_dependents",
    "declared_max_impact_depth",
)


@dataclass(frozen=True)
class Corpus:
    """A replayed change history with its graphs, evidence and argument.

    Attributes
    ----------
    name:
        Short corpus name, used in reports.
    seed:
        The generator seed, recorded so a reader can regenerate the file.
    true_graph:
        The complete dependency graph. Ground truth is computed over this.
    declared_graph:
        The subset a tool would actually see. Equal to the true graph only when
        the declarations are complete.
    history:
        The replayed change history.
    evidence:
        Evidence items, each carrying its by-construction staleness label.
    case:
        The GSN argument structure the evidence supports.
    evaluation_time:
        Revision at which the audit is performed.
    declared_max_dependents:
        The structural constant the independent bound consumes as ``d``.
    declared_max_impact_depth:
        The structural constant the independent bound consumes as ``L``.
    """

    name: str
    seed: int
    true_graph: DependencyGraph
    declared_graph: DependencyGraph
    history: ChangeHistory
    evidence: tuple[EvidenceItem, ...]
    case: AssuranceCase
    evaluation_time: int
    declared_max_dependents: int
    declared_max_impact_depth: int

    def graph(self, declarations: str) -> DependencyGraph:
        """The graph named by ``declarations``: ``"complete"`` or ``"partial"``."""
        if declarations == "complete":
            return self.true_graph
        if declarations == "partial":
            return self.declared_graph
        raise ValueError(
            f"declarations={declarations!r} is not a view this corpus provides. Pass "
            f"'complete' for the true edge set or 'partial' for the declared subset."
        )

    def hidden_edge_count(self) -> int:
        """Number of true edges missing from the declared set."""
        return len(set(self.true_graph.edges) - set(self.declared_graph.edges))

    def subjects(self) -> tuple[str, ...]:
        """Subject artifact of each evidence item, in evidence order."""
        return tuple(item.subject for item in self.evidence)

    def labels(self) -> tuple[int, ...]:
        """By-construction staleness labels as 0/1, in evidence order."""
        return tuple(int(bool(item.true_stale)) for item in self.evidence)


def _require(document: Mapping[str, object], path: Path) -> None:
    missing = [key for key in REQUIRED_KEYS if key not in document]
    if missing:
        raise ValueError(
            f"corpus file {path.name} is missing required keys {missing}. It was not "
            f"written by data/generate_corpus.py, or it was written by an "
            f"incompatible version; regenerate it with that script."
        )
    schema = document["schema"]
    if schema != SCHEMA:
        raise ValueError(
            f"corpus file {path.name} declares schema {schema!r}; this loader reads "
            f"{SCHEMA!r}. Regenerate the corpus with data/generate_corpus.py."
        )


def load_corpus(path: str | Path) -> Corpus:
    """Load and validate a corpus JSON document.

    Parameters
    ----------
    path:
        Path to the corpus file. Relative paths are resolved against the
        process working directory, so callers under ``validation/`` and
        ``examples/`` must have changed directory to the product root first --
        which the shared ``_bootstrap`` module does.

    Raises
    ------
    ValueError
        On a missing key, an unknown schema, or any structural inconsistency
        the data types themselves reject.
    """
    target = Path(path)
    if not target.is_file():
        raise ValueError(
            f"corpus file {target.name} does not exist at the given path. The "
            f"committed corpus lives in data/ beside data/generate_corpus.py; run "
            f"that script to regenerate it, or check the path."
        )
    document = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(
            f"corpus file {target.name} does not contain a JSON object at the top level"
        )
    _require(document, target)

    artifacts = tuple(str(name) for name in document["artifacts"])
    true_graph = DependencyGraph.build(artifacts, document["true_edges"])
    declared_graph = DependencyGraph.build(artifacts, document["declared_edges"])
    hidden = set(declared_graph.edges) - set(true_graph.edges)
    if hidden:
        raise ValueError(
            f"corpus file {target.name} declares {len(hidden)} edges that are not in "
            f"the true edge set, beginning {sorted(hidden)[:4]}. The declared set must "
            f"be a subset of the true set: an incomplete declaration omits edges, it "
            f"does not invent them."
        )
    events = tuple(
        ChangeEvent(
            time=int(entry["time"]),
            artifact=str(entry["artifact"]),
            kind=str(entry["kind"]),  # type: ignore[arg-type]
            content=None if entry.get("content") is None else str(entry["content"]),
        )
        for entry in document["events"]
    )
    history = ChangeHistory(
        artifacts,
        {str(k): str(v) for k, v in dict(document["initial_contents"]).items()},
        events,
    )
    evidence = tuple(
        EvidenceItem(
            identifier=str(entry["identifier"]),
            subject=str(entry["subject"]),
            produced_at=int(entry["produced_at"]),
            goal=str(entry["goal"]),
            solution=str(entry["solution"]),
            true_stale=None if entry.get("true_stale") is None else bool(entry["true_stale"]),
        )
        for entry in document["evidence"]
    )
    case_document = dict(document["assurance_case"])
    case = AssuranceCase(
        nodes={str(k): str(v) for k, v in dict(case_document["nodes"]).items()},
        links=tuple(
            (str(link[0]), str(link[1]), str(link[2])) for link in case_document["links"]
        ),
    )
    evaluation_time = int(document["evaluation_time"])
    for item in evidence:
        if item.produced_at > evaluation_time:
            raise ValueError(
                f"evidence {item.identifier!r} was produced at revision "
                f"{item.produced_at}, after the corpus evaluation revision "
                f"{evaluation_time}. Evidence cannot be audited before it exists."
            )
    return Corpus(
        name=str(document["name"]),
        seed=int(document["seed"]),
        true_graph=true_graph,
        declared_graph=declared_graph,
        history=history,
        evidence=evidence,
        case=case,
        evaluation_time=evaluation_time,
        declared_max_dependents=int(document["declared_max_dependents"]),
        declared_max_impact_depth=int(document["declared_max_impact_depth"]),
    )


#: Committed corpus file names, relative to the product root.
MAIN_CORPUS = "data/corpus_layered.json"
CYCLIC_CORPUS = "data/corpus_cyclic.json"
MANIFEST = "data/corpus_manifest.json"

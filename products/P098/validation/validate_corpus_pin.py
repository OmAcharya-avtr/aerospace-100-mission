"""Pin the committed corpora by digest, and check their declared structure.

This script READS and HASHES. It never writes to ``data/``. The corpora are
generated once by ``data/generate_corpus.py``, which is deliberately not under
``validation/`` because the release gate runs
``sorted(glob('validation/*.py'))`` and a script that regenerated data another
script pins would be collided by the gate's own ordering.

Three things are checked:

1. the SHA-256 digest and byte length of every committed corpus file against
   the committed manifest;
2. the structural constants the independent bound in
   ``src/evidencestale/bounds.py`` consumes -- the maximum number of direct
   dependents and the maximum impact depth -- against the graph as generated,
   because the bound is only valid if the declared constants are true of the
   graph;
3. that the committed by-construction labels agree with the labels the replay
   recomputes from the committed event log.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import hashlib  # noqa: E402
import json  # noqa: E402

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()

from evidencestale.baselines import committed_labels, ground_truth  # noqa: E402
from evidencestale.corpus import CYCLIC_CORPUS, MAIN_CORPUS, MANIFEST, load_corpus  # noqa: E402
from evidencestale.hashing import (  # noqa: E402
    collision_bound_exponent,
    collision_probability_bound,
)


def main() -> int:
    """Hash the committed corpora and check their declared structure."""
    out = _bootstrap.Tee("validate_corpus_pin_output.txt")
    out("Corpus pin -- P098 EvidenceStale")
    out("")
    out("this script reads and hashes. It never writes to data/. The corpora are")
    out("generated once by data/generate_corpus.py, which is not under validation/")
    out("because the gate runs sorted(glob('validation/*.py')) and adjacent scripts")
    out("run back to back.")
    out("")
    out(f"fast path (EVIDENCESTALE_FAST=1): {_bootstrap.FAST}")
    out(f"manifest: {MANIFEST}")

    manifest = json.loads((ROOT / MANIFEST).read_text(encoding="utf-8"))
    out(f"generator seed recorded in the manifest: {manifest['seed']}")

    out.rule("1. COMMITTED DIGESTS")
    for name in sorted(manifest["files"]):
        entry = manifest["files"][name]
        text = (ROOT / "data" / name).read_text(encoding="utf-8")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        size = len(text.encode("utf-8"))
        out("")
        out(f"data/{name}")
        out(f"  committed sha256 : {entry['sha256']}")
        out(f"  recomputed sha256: {digest}")
        out(f"  committed bytes  : {entry['bytes']}")
        out(f"  recomputed bytes : {size}")
        out.check(f"data/{name} digest matches the manifest", digest == entry["sha256"])
        out.check(f"data/{name} byte length matches the manifest", size == entry["bytes"])

    out.rule("2. DECLARED STRUCTURE AGAINST THE GENERATED GRAPH")
    out("the independent bound consumes two structural constants. They are only")
    out("valid inputs if they are true of the graph, so they are measured here")
    out("rather than trusted.")
    for path in (MAIN_CORPUS, CYCLIC_CORPUS):
        corpus = load_corpus(ROOT / path)
        graph = corpus.true_graph
        dependents = graph.max_out_degree_of_impact_graph()
        depth = max(graph.closure_depth(name) for name in graph.nodes)
        closure_sizes = [len(graph.closure(name)) for name in graph.nodes]
        out("")
        out(f"{path}  (corpus {corpus.name!r})")
        out(f"  artifacts                      : {len(graph.nodes)}")
        out(f"  true edges                     : {len(graph.edges)}")
        out(f"  declared edges                 : {len(corpus.declared_graph.edges)}")
        out(f"  hidden edges                   : {corpus.hidden_edge_count()}")
        out(f"  change events                  : {len(corpus.history.events)}")
        out(f"  evidence items                 : {len(corpus.evidence)}")
        out(f"  evaluation revision            : {corpus.evaluation_time}")
        out(f"  declared max dependents (d)    : {corpus.declared_max_dependents}")
        out(f"  measured max dependents        : {dependents}")
        out(f"  declared max impact depth (L)  : {corpus.declared_max_impact_depth}")
        out(f"  measured max closure depth     : {depth}")
        out(f"  max closure size               : {max(closure_sizes)}")
        out(f"  mean closure size              : {sum(closure_sizes) / len(closure_sizes):.6f}")
        out(f"  acyclic                        : {graph.is_acyclic()}")
        out(f"  cycles reported                : {graph.cycles()}")
        out.check(
            f"{path}: measured max dependents does not exceed the declared d",
            dependents <= corpus.declared_max_dependents,
            f"{dependents} <= {corpus.declared_max_dependents}",
        )
        out.check(
            f"{path}: measured max closure depth does not exceed the declared L",
            depth <= corpus.declared_max_impact_depth,
            f"{depth} <= {corpus.declared_max_impact_depth}",
        )
        out.check(
            f"{path}: the closure depth limit never bound",
            depth < 64,
            f"deepest closure {depth} against the default limit 64",
        )

    out.rule("3. COMMITTED LABELS AGAINST THE REPLAYED GROUND TRUTH")
    out("the generator wrote true_stale while it built the history. The replay")
    out("recomputes it from the committed event log. Two computations over two")
    out("representations of one definition; disagreement would mean the committed")
    out("corpus does not describe the history it ships.")
    for path in (MAIN_CORPUS, CYCLIC_CORPUS):
        corpus = load_corpus(ROOT / path)
        committed = committed_labels(corpus.evidence)
        replayed = ground_truth(
            corpus.evidence,
            corpus.history,
            corpus.true_graph,
            corpus.evaluation_time,
        )
        disagree = sum(
            1
            for a, b in zip(committed.as_array(), replayed.as_array(), strict=True)
            if a != b
        )
        out("")
        out(f"{path}")
        out(f"  committed stale items : {committed.stale_count} of {committed.total}")
        out(f"  replayed stale items  : {replayed.stale_count} of {replayed.total}")
        out(f"  committed fraction    : {committed.stale_fraction:.6f}")
        out(f"  replayed fraction     : {replayed.stale_fraction:.6f}")
        out(f"  items disagreeing     : {disagree}")
        out.check(
            f"{path}: committed labels agree with the replayed ground truth item by item",
            disagree == 0,
            f"{disagree} disagree",
        )

    out.rule("4. DIGEST COLLISION BOUND FOR THE CORPUS ACTUALLY USED")
    out("one of the three conditions the closure comparison's exactness rests on.")
    for path in (MAIN_CORPUS, CYCLIC_CORPUS):
        corpus = load_corpus(ROOT / path)
        contents = set(corpus.history.initial.values())
        for event in corpus.history.events:
            if event.content is not None:
                contents.add(event.content)
        count = len(contents)
        bound = collision_probability_bound(count)
        exponent = collision_bound_exponent(count)
        out("")
        out(f"{path}")
        out(f"  distinct contents over the whole history : {count}")
        out(f"  pairs                                    : {count * (count - 1) // 2}")
        out(f"  union bound on any pair colliding        : {bound:.6e}")
        out(f"  log10 of that bound                      : {exponent:.6f}")
        out.check(
            f"{path}: the pairwise collision bound is below 1e-60",
            bound < 1e-60,
            f"{bound:.3e}",
        )
    out("")
    out("this is a bound on an idealised model of SHA-256 (FIPS 180-4), not a proof")
    out("about SHA-256, and it says nothing about an adversary choosing contents.")
    return out.finish()


if __name__ == "__main__":
    sys.exit(main())

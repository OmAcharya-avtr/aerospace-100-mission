"""Corpus loading, validation and the committed digests."""

from __future__ import annotations

import hashlib
import json

import pytest

from evidencestale.corpus import (
    CYCLIC_CORPUS,
    MAIN_CORPUS,
    MANIFEST,
    SCHEMA,
    load_corpus,
)


def test_committed_digests_match_the_manifest(product_root) -> None:
    manifest = json.loads((product_root / MANIFEST).read_text(encoding="utf-8"))
    for name, entry in manifest["files"].items():
        text = (product_root / "data" / name).read_text(encoding="utf-8")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        assert digest == entry["sha256"], name
        assert len(text.encode("utf-8")) == entry["bytes"], name


def test_manifest_counts_match_the_loaded_corpora(product_root) -> None:
    manifest = json.loads((product_root / MANIFEST).read_text(encoding="utf-8"))
    for name, entry in manifest["files"].items():
        corpus = load_corpus(product_root / "data" / name)
        assert len(corpus.true_graph.nodes) == entry["artifacts"]
        assert len(corpus.true_graph.edges) == entry["true_edges"]
        assert len(corpus.declared_graph.edges) == entry["declared_edges"]
        assert len(corpus.history.events) == entry["events"]
        assert len(corpus.evidence) == entry["evidence"]
        assert sum(corpus.labels()) == entry["true_stale"]


def test_layered_corpus_shape(layered_corpus) -> None:
    assert layered_corpus.name == "layered"
    assert len(layered_corpus.true_graph.nodes) == 240
    assert len(layered_corpus.evidence) == 480
    assert layered_corpus.evaluation_time == 2000
    assert layered_corpus.declared_max_dependents == 2
    assert layered_corpus.declared_max_impact_depth == 2
    assert layered_corpus.hidden_edge_count() == 75
    assert sum(layered_corpus.labels()) == 194


def test_every_artifact_carries_exactly_two_evidence_items(layered_corpus) -> None:
    counts: dict[str, int] = {}
    for item in layered_corpus.evidence:
        counts[item.subject] = counts.get(item.subject, 0) + 1
    assert set(counts) == set(layered_corpus.true_graph.nodes)
    assert set(counts.values()) == {2}


def test_corpus_schema_and_views(layered_corpus) -> None:
    assert SCHEMA == "evidencestale/corpus/1"
    assert layered_corpus.graph("complete") is layered_corpus.true_graph
    assert layered_corpus.graph("partial") is layered_corpus.declared_graph
    with pytest.raises(ValueError, match="not a view this corpus provides"):
        layered_corpus.graph("other")


def test_declared_edges_are_a_subset_of_the_true_edges(layered_corpus) -> None:
    assert set(layered_corpus.declared_graph.edges) < set(layered_corpus.true_graph.edges)


def test_cyclic_corpus_shape(cyclic_corpus) -> None:
    assert cyclic_corpus.name == "cyclic"
    assert len(cyclic_corpus.true_graph.nodes) == 12
    assert cyclic_corpus.true_graph.is_acyclic() is False
    assert cyclic_corpus.hidden_edge_count() == 0


def test_corpus_paths_are_the_committed_ones() -> None:
    assert MAIN_CORPUS == "data/corpus_layered.json"
    assert CYCLIC_CORPUS == "data/corpus_cyclic.json"
    assert MANIFEST == "data/corpus_manifest.json"


def test_missing_file_rejected_with_an_actionable_message(tmp_path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        load_corpus(tmp_path / "nope.json")


def test_missing_key_rejected(tmp_path) -> None:
    target = tmp_path / "bad.json"
    target.write_text(json.dumps({"schema": SCHEMA}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing required keys"):
        load_corpus(target)


def test_wrong_schema_rejected(tmp_path, product_root) -> None:
    document = json.loads((product_root / CYCLIC_CORPUS).read_text(encoding="utf-8"))
    document["schema"] = "evidencestale/corpus/0"
    target = tmp_path / "old.json"
    target.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="declares schema"):
        load_corpus(target)


def test_non_object_document_rejected(tmp_path) -> None:
    target = tmp_path / "list.json"
    target.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object at the top level"):
        load_corpus(target)


def test_declared_edge_outside_the_true_set_rejected(tmp_path, product_root) -> None:
    document = json.loads((product_root / CYCLIC_CORPUS).read_text(encoding="utf-8"))
    document["declared_edges"] = [*document["declared_edges"], ["c00", "c11"]]
    target = tmp_path / "invented.json"
    target.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="it does not invent them"):
        load_corpus(target)


def test_evidence_produced_after_the_evaluation_revision_rejected(
    tmp_path, product_root
) -> None:
    document = json.loads((product_root / CYCLIC_CORPUS).read_text(encoding="utf-8"))
    document["evidence"][0]["produced_at"] = document["evaluation_time"] + 1
    target = tmp_path / "future.json"
    target.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="before it exists"):
        load_corpus(target)

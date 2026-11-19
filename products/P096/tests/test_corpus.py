"""The committed corpus: hash, columns, feature causality, the two splits."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from handoverloss.corpus import (
    CORPUS_CSV,
    CORPUS_MANIFEST,
    FEATURE_COLUMNS,
    PROVENANCE_COLUMNS,
    TARGET_COLUMNS,
    CorpusAbsentError,
    Split,
    check_feature_causality,
    load_corpus,
    station_pair_split,
    time_split,
)


def test_corpus_hash_matches_its_manifest(corpus):
    declared = json.loads(Path(CORPUS_MANIFEST).read_text())
    assert corpus.sha256 == declared["sha256"]
    assert len(corpus) == declared["events"]


def test_corpus_stays_under_one_megabyte():
    # The mission's build guide forbids committing data over 1 MB. The generator
    # is committed either way; this check is on what a cold clone carries.
    assert Path(CORPUS_CSV).stat().st_size < 1_000_000


def test_no_feature_is_a_post_handover_quantity():
    # A realised duration or a realised loss among the features would make the
    # learned component a measurement dressed as a prediction.
    assert check_feature_causality() == []


def test_feature_target_and_provenance_columns_are_disjoint():
    assert not set(FEATURE_COLUMNS) & set(TARGET_COLUMNS)
    assert not set(FEATURE_COLUMNS) & set(PROVENANCE_COLUMNS)


def test_shapes_and_signs(corpus):
    assert corpus.features.shape == (len(corpus), len(FEATURE_COLUMNS))
    assert corpus.targets.shape == (len(corpus), 3)
    assert np.all(corpus.targets >= 0.0)
    assert np.isfinite(corpus.features).all()


def test_all_three_mechanisms_dominate_some_events(corpus):
    # A corpus in which one answer is nearly always right cannot distinguish a
    # predictor from a constant. Every mechanism must dominate at least a tenth
    # of the corpus for the dominance question to be non-degenerate.
    counts = corpus.dominance_counts()
    assert sum(counts.values()) == len(corpus)
    for name, count in counts.items():
        assert count >= 0.10 * len(corpus), f"{name} dominates only {count} events"


def test_dominant_column_agrees_with_the_targets(corpus):
    names = np.asarray(corpus.table["dominant"], dtype=object)
    from handoverloss.sequence import MECHANISMS

    recomputed = np.asarray([MECHANISMS[int(i)] for i in corpus.dominant_index], dtype=object)
    assert np.array_equal(names, recomputed)


def test_conservation_column_matches_the_target_sum(corpus):
    # The committed conservation total and the committed per-mechanism sum are
    # two independent accountings written by the generator. They must agree to
    # within the accumulation error of the slot sums.
    declared = json.loads(Path(CORPUS_MANIFEST).read_text())
    slots = int(declared["slots_per_event"])
    bucketed = corpus.targets.sum(axis=1)
    conserved = np.asarray(corpus.table["conservation_total_bits"], dtype=float)
    produced = np.asarray(corpus.table["produced_bits"], dtype=float)
    delivered = np.asarray(corpus.table["delivered_bits"], dtype=float)
    slack = 4.0 * slots * (2.0**-53) * (produced + delivered + bucketed)
    assert np.all(np.abs(conserved - bucketed) <= slack)


def test_station_pair_split_holds_out_whole_pairs(corpus):
    split = station_pair_split(corpus, held_out_pairs=4)
    assert split.name == "station_pairs"
    test_pairs = set(corpus.pair_id[split.test].tolist())
    train_pairs = set(corpus.pair_id[split.train].tolist())
    assert not test_pairs & train_pairs
    assert test_pairs == set(split.held_out)
    assert split.train.size + split.test.size == len(corpus)


def test_time_split_is_forward_only(corpus):
    split = time_split(corpus, held_out_epochs=3)
    assert split.name == "time"
    assert corpus.epoch[split.train].max() < corpus.epoch[split.test].min()
    assert split.train.size + split.test.size == len(corpus)


def test_splits_are_deterministic(corpus):
    first = station_pair_split(corpus)
    second = station_pair_split(corpus)
    assert np.array_equal(first.test, second.test)


@pytest.mark.parametrize("held_out", [0, -1, 16, 99])
def test_station_pair_split_bounds(corpus, held_out):
    with pytest.raises(ValueError, match="held_out_pairs must lie"):
        station_pair_split(corpus, held_out_pairs=held_out)


@pytest.mark.parametrize("held_out", [0, -1, 8, 99])
def test_time_split_bounds(corpus, held_out):
    with pytest.raises(ValueError, match="held_out_epochs must lie"):
        time_split(corpus, held_out_epochs=held_out)


def test_split_rejects_a_leak():
    with pytest.raises(ValueError, match="leaks"):
        Split(
            name="broken",
            train=np.array([0, 1, 2]),
            test=np.array([2, 3]),
            held_out=(1,),
            description="",
        )


def test_split_rejects_an_empty_side():
    with pytest.raises(ValueError, match="empty side"):
        Split(
            name="broken",
            train=np.array([], dtype=int),
            test=np.array([1]),
            held_out=(1,),
            description="",
        )


def test_missing_corpus_raises_with_the_generator_named(tmp_path):
    with pytest.raises(CorpusAbsentError, match="generate_corpus.py"):
        load_corpus(tmp_path / "absent.csv", tmp_path / "absent.json")


def test_hash_mismatch_is_refused(tmp_path):
    original = Path(CORPUS_CSV).read_text()
    csv = tmp_path / "handovers.csv"
    csv.write_text(original)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"sha256": "0" * 64}) + "\n")
    with pytest.raises(ValueError, match="is not the one the committed numbers"):
        load_corpus(csv, manifest)
    # The same file loads when the check is switched off, so the refusal is
    # about the hash and not about the file being unreadable.
    assert len(load_corpus(csv, manifest, verify_hash=False)) > 0


def test_malformed_row_is_refused(tmp_path):
    lines = Path(CORPUS_CSV).read_text().splitlines()
    csv = tmp_path / "handovers.csv"
    csv.write_text("\n".join([lines[0], lines[1], "1,2,3"]) + "\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"sha256": "x"}) + "\n")
    with pytest.raises(ValueError, match="fields against a header"):
        load_corpus(csv, manifest, verify_hash=False)


def test_header_only_file_is_refused(tmp_path):
    csv = tmp_path / "handovers.csv"
    csv.write_text(Path(CORPUS_CSV).read_text().splitlines()[0] + "\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"sha256": "x"}) + "\n")
    with pytest.raises(ValueError, match="no data rows"):
        load_corpus(csv, manifest, verify_hash=False)


def test_missing_column_is_refused(tmp_path):
    lines = Path(CORPUS_CSV).read_text().splitlines()
    header = lines[0].split(",")
    drop = header.index("utilisation")
    new_lines = []
    for line in lines:
        fields = line.split(",")
        del fields[drop]
        new_lines.append(",".join(fields))
    csv = tmp_path / "handovers.csv"
    csv.write_text("\n".join(new_lines) + "\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"sha256": "x"}) + "\n")
    with pytest.raises(ValueError, match="missing columns"):
        load_corpus(csv, manifest, verify_hash=False)

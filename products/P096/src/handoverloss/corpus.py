"""Loading the committed handover corpus and cutting it two ways: by pair, by time.

THE TWO SPLITS, AND WHY THEY ARE SCORED SEPARATELY
--------------------------------------------------

A predictor of handover loss can be wrong in two different ways and a single
random split hides both.

* **A new station pair.** Each pair has its own geometry, its own typical
  atmospheric conditions and its own protocol configuration. A model that has
  seen every pair in training may be interpolating within each pair's own
  cluster rather than learning the mechanism. :func:`station_pair_split` holds
  out whole pairs, so every test event comes from a pair the model never saw.
* **A later time.** Conditions drift across epochs. :func:`time_split` holds out
  the LAST epochs, so every test event is later than every training event.

The two are reported separately throughout this package and are never averaged,
because they answer different operational questions: "can I deploy this at a
station I have no history for" and "will it still be right next month".

THE FILE
--------

``data/corpus/handovers.csv``, written by ``data/generate_corpus.py`` with fixed
seeds and hashed into ``data/corpus/manifest.json``. The CSV is committed
because it is under 1 MB; the generator is committed because the CSV is not the
evidence, the reproducibility is.

Columns fall into three groups and the grouping is enforced by
:data:`FEATURE_COLUMNS`, :data:`TARGET_COLUMNS` and :data:`PROVENANCE_COLUMNS`:

* **features** are quantities observable BEFORE the handover begins. No realised
  duration and no realised loss is a feature. That restriction is what makes the
  learned component a predictor rather than a measurement, and
  :func:`check_feature_causality` asserts it.
* **targets** are the ground-truth per-mechanism losses from the trace.
* **provenance** identifies the event, its pair and its epoch.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .sequence import MECHANISMS

#: Default location of the committed corpus, relative to the repository root.
CORPUS_CSV = "data/corpus/handovers.csv"

#: Default location of its manifest.
CORPUS_MANIFEST = "data/corpus/manifest.json"

#: Pre-handover observables. Every one of these is knowable before the outgoing
#: link breaks. Order is fixed because the learned model's feature importances
#: are reported against it.
FEATURE_COLUMNS: tuple[str, ...] = (
    "source_rate_bps",
    "downlink_rate_bps",
    "buffer_capacity_bits",
    "initial_occupancy_bits",
    "occupancy_fraction",
    "headroom_s",
    "utilisation",
    "declared_ramp_s",
    "declared_restart_s",
    "acquisition_median_s",
    "acquisition_log_sigma",
    "acquisition_abort_probability",
    "acquisition_attempt_window_s",
    "elevation_deg",
    "turbulence_cn2_log10",
    "beacon_margin_db",
    "sky_cover_index",
    "source_burst_cv",
)

#: Ground-truth per-mechanism loss, bits, from the trace. Not features.
TARGET_COLUMNS: tuple[str, ...] = tuple(f"loss_{name}_bits" for name in MECHANISMS)

#: Identity and realised-outcome columns. Realised durations live here, not in
#: FEATURE_COLUMNS, because they are not knowable before the handover.
PROVENANCE_COLUMNS: tuple[str, ...] = (
    "event_id",
    "pair_id",
    "station_from",
    "station_to",
    "epoch",
    "realised_acquisition_s",
    "realised_restart_s",
    "horizon_s",
    "produced_bits",
    "delivered_bits",
    "final_occupancy_bits",
    "conservation_total_bits",
    "dominant",
)


class CorpusAbsentError(FileNotFoundError):
    """Raised when the committed corpus is not on disk, naming what was looked for."""


@dataclass(frozen=True)
class Corpus:
    """The committed corpus as parallel arrays, with its split helpers.

    Attributes
    ----------
    features:
        ``(n, len(FEATURE_COLUMNS))`` float64 array of pre-handover observables.
    targets:
        ``(n, 3)`` float64 array of ground-truth per-mechanism loss, bits.
    pair_id:
        ``(n,)`` int array identifying the station pair of each event.
    epoch:
        ``(n,)`` int array identifying the time epoch of each event.
    table:
        The whole CSV as a dict of column name to array, including provenance.
    sha256:
        Hex digest of the CSV as read.
    """

    features: np.ndarray
    targets: np.ndarray
    pair_id: np.ndarray
    epoch: np.ndarray
    table: dict[str, np.ndarray]
    sha256: str

    def __len__(self) -> int:
        return int(self.features.shape[0])

    @property
    def dominant_index(self) -> np.ndarray:
        """``(n,)`` int array: which mechanism dominates each event's ground truth."""
        return np.argmax(self.targets, axis=1)

    def dominance_counts(self) -> dict[str, int]:
        """How many events each mechanism dominates, keyed by mechanism name."""
        index = self.dominant_index
        return {name: int(np.sum(index == column)) for column, name in enumerate(MECHANISMS)}


@dataclass(frozen=True)
class Split:
    """One train/test cut, with the reason it was cut that way.

    Attributes
    ----------
    name:
        ``"station_pairs"`` or ``"time"``.
    train, test:
        Integer index arrays into the corpus.
    held_out:
        The pair ids or epochs held out, as a sorted tuple.
    description:
        One line naming the operational question this split answers.
    """

    name: str
    train: np.ndarray
    test: np.ndarray
    held_out: tuple[int, ...]
    description: str

    def __post_init__(self) -> None:
        if self.train.size == 0 or self.test.size == 0:
            raise ValueError(
                f"Split {self.name!r} has an empty side: {self.train.size} train, "
                f"{self.test.size} test"
            )
        overlap = np.intersect1d(self.train, self.test)
        if overlap.size:
            raise ValueError(
                f"Split {self.name!r} leaks {overlap.size} event(s) between train and "
                "test; the sides must be disjoint"
            )


def _read_csv(path: Path) -> tuple[dict[str, np.ndarray], str]:
    """Read the corpus CSV into column arrays and return its sha256 hex digest."""
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    lines = raw.decode("utf-8").splitlines()
    if len(lines) < 2:
        raise ValueError(f"{path.name} has no data rows")
    header = lines[0].split(",")
    columns: dict[str, list[str]] = {name: [] for name in header}
    for line in lines[1:]:
        if not line.strip():
            continue
        fields = line.split(",")
        if len(fields) != len(header):
            raise ValueError(
                f"{path.name}: row with {len(fields)} fields against a header of "
                f"{len(header)}"
            )
        for name, value in zip(header, fields, strict=True):
            columns[name].append(value)
    table: dict[str, np.ndarray] = {}
    for name, values in columns.items():
        try:
            table[name] = np.asarray([float(v) for v in values], dtype=float)
        except ValueError:
            table[name] = np.asarray(values, dtype=object)
    return table, digest


def load_corpus(
    csv_path: str | Path = CORPUS_CSV,
    manifest_path: str | Path = CORPUS_MANIFEST,
    *,
    verify_hash: bool = True,
) -> Corpus:
    """Load the committed corpus, checking its hash against the manifest.

    Parameters
    ----------
    csv_path, manifest_path:
        Paths, interpreted relative to the current working directory. Every
        caller in this repository runs with the working directory set to the
        repository root by ``validation/_bootstrap.py`` or ``examples/_common.py``,
        so the defaults are relative and no absolute path is ever constructed.
    verify_hash:
        When true, the CSV's sha256 must match ``manifest.json``.

    Returns
    -------
    Corpus

    Raises
    ------
    CorpusAbsentError
        If the CSV or the manifest is missing, naming both paths and the
        generator that writes them.
    ValueError
        On a hash mismatch, a missing column, or a malformed row.
    """
    csv = Path(csv_path)
    manifest = Path(manifest_path)
    if not csv.is_file() or not manifest.is_file():
        raise CorpusAbsentError(
            f"the committed corpus is absent. Expected {CORPUS_CSV} and "
            f"{CORPUS_MANIFEST}. Regenerate both, deterministically, with "
            "`python3 data/generate_corpus.py` from the repository root. Nothing was "
            "substituted: this package has no synthetic fallback at load time, because "
            "a fallback would produce a figure that looks like the committed corpus and "
            "is not it."
        )
    table, digest = _read_csv(csv)
    declared = json.loads(manifest.read_text())
    if verify_hash:
        expected = declared.get("sha256", "")
        if expected != digest:
            raise ValueError(
                f"{CORPUS_CSV} has sha256 {digest} but {CORPUS_MANIFEST} declares "
                f"{expected}. The corpus on disk is not the one the committed numbers "
                "came from; regenerate it with `python3 data/generate_corpus.py`."
            )
    missing = [
        name
        for name in (*FEATURE_COLUMNS, *TARGET_COLUMNS, *PROVENANCE_COLUMNS)
        if name not in table
    ]
    if missing:
        raise ValueError(f"{CORPUS_CSV} is missing columns: {missing}")
    features = np.stack([table[name] for name in FEATURE_COLUMNS], axis=1)
    targets = np.stack([table[name] for name in TARGET_COLUMNS], axis=1)
    return Corpus(
        features=np.ascontiguousarray(features, dtype=float),
        targets=np.ascontiguousarray(targets, dtype=float),
        pair_id=table["pair_id"].astype(int),
        epoch=table["epoch"].astype(int),
        table=table,
        sha256=digest,
    )


def station_pair_split(corpus: Corpus, held_out_pairs: int = 4) -> Split:
    """Hold out whole station pairs: the highest-numbered ``held_out_pairs`` of them.

    Deterministic, with no shuffling, so the split is a property of the corpus
    and not of a seed. Highest-numbered rather than random because the generator
    assigns pair ids in a fixed order and the choice must not be re-rollable:
    re-rolling a split until a model wins is the way a held-out score stops
    meaning anything.
    """
    unique = np.unique(corpus.pair_id)
    if held_out_pairs < 1 or held_out_pairs >= unique.size:
        raise ValueError(
            f"held_out_pairs must lie in [1, {unique.size - 1}] for a corpus with "
            f"{unique.size} pairs, got {held_out_pairs!r}"
        )
    held = tuple(int(p) for p in unique[-held_out_pairs:])
    mask = np.isin(corpus.pair_id, held)
    return Split(
        name="station_pairs",
        train=np.flatnonzero(~mask),
        test=np.flatnonzero(mask),
        held_out=held,
        description=(
            f"{held_out_pairs} of {unique.size} station pairs held out entirely; every "
            "test event comes from a pair absent from training. Answers: can this be "
            "deployed at a station pair with no history."
        ),
    )


def time_split(corpus: Corpus, held_out_epochs: int = 3) -> Split:
    """Hold out the last ``held_out_epochs`` epochs, so test is strictly later.

    Deterministic and forward-only: no test event precedes any training event.
    """
    unique = np.unique(corpus.epoch)
    if held_out_epochs < 1 or held_out_epochs >= unique.size:
        raise ValueError(
            f"held_out_epochs must lie in [1, {unique.size - 1}] for a corpus with "
            f"{unique.size} epochs, got {held_out_epochs!r}"
        )
    held = tuple(int(e) for e in unique[-held_out_epochs:])
    mask = np.isin(corpus.epoch, held)
    return Split(
        name="time",
        train=np.flatnonzero(~mask),
        test=np.flatnonzero(mask),
        held_out=held,
        description=(
            f"the last {held_out_epochs} of {unique.size} epochs held out; every test "
            "event is later than every training event. Answers: will this still be "
            "right next month."
        ),
    )


def check_feature_causality() -> list[str]:
    """Names in :data:`FEATURE_COLUMNS` that are not pre-handover observables.

    Returns an empty list when the restriction holds. A realised duration, a
    realised loss or a delivered volume appearing among the features would make
    the learned component a measurement dressed as a prediction, so the check is
    by name and is asserted by ``tests/test_corpus.py``.
    """
    forbidden_markers = ("realised_", "loss_", "delivered_", "produced_", "dominant")
    return [
        name
        for name in FEATURE_COLUMNS
        if any(marker in name for marker in forbidden_markers)
    ]

"""Shared fixtures. Generated records are session-scoped because they are the
slowest thing in the suite and nothing mutates them.
"""

from __future__ import annotations

import numpy as np
import pytest

from telemgap import classify, dataset, features
from telemgap.learned import LearnedClassifier
from telemgap.record import ChannelSpec, RecordSchema, TelemetryRecord


@pytest.fixture(scope="session")
def small_channel() -> ChannelSpec:
    """One 16-bit unsigned channel with round numbers, for known-answer tests.

    ``scale = 0.5``, ``offset = 10.0``, so raw 0 decodes to 10.0 and raw 100 to
    60.0. Declared range [12.0, 60.0].
    """
    return ChannelSpec(
        name="probe",
        unit="K",
        scale=0.5,
        offset=10.0,
        low=12.0,
        high=60.0,
        max_slew=1.0,
        max_curvature=0.25,
        noise_sigma=0.0,
        word_bits=16,
        signed=False,
    )


@pytest.fixture(scope="session")
def small_schema(small_channel: ChannelSpec) -> RecordSchema:
    """Cadence 1 s, jitter 0.1 s, counter modulus 256, one channel."""
    return RecordSchema(
        cadence_s=1.0,
        jitter_s=0.1,
        counter_modulus=256,
        channels=(small_channel,),
        has_validity_flag=True,
        has_raw_words=True,
    )


def build_record(
    schema: RecordSchema,
    raws: list[int],
    timestamps: list[float] | None = None,
    counters: list[int] | None = None,
    validity: list[bool] | None = None,
) -> TelemetryRecord:
    """Build a one-channel record from raw words, with nominal time and counters."""
    n = len(raws)
    channel = schema.channels[0]
    words = np.asarray(raws, dtype=np.uint64)
    values = channel.decode(words.astype(np.int64)).reshape(n, 1)
    return TelemetryRecord(
        schema=schema,
        timestamps=np.arange(n, dtype=np.float64) if timestamps is None else np.asarray(timestamps),
        counters=np.arange(n, dtype=np.int64) % schema.counter_modulus
        if counters is None
        else np.asarray(counters),
        values=values,
        raw_words=words.reshape(n, 1),
        validity=np.ones(n, dtype=bool) if validity is None else np.asarray(validity, dtype=bool),
        provenance={"time_source": "test"},
    )


@pytest.fixture(scope="session")
def record_builder():
    """The :func:`build_record` helper, as a fixture.

    A fixture rather than a cross-module import: with pytest's default
    ``prepend`` import mode and no ``__init__.py`` in this directory,
    ``from .conftest import build_record`` does not resolve, and adding an
    ``__init__.py`` to make it resolve changes how the whole suite is imported.
    """
    return build_record


@pytest.fixture
def nominal_record(small_schema: RecordSchema) -> TelemetryRecord:
    """Forty frames of a clean ramp inside the declared range."""
    raws = [40 + index for index in range(40)]
    return build_record(small_schema, raws)


@pytest.fixture(scope="session")
def generated_small() -> dataset.GeneratedRecord:
    """A 1,500-frame generated record with every injection class present."""
    return dataset.generate(samples=1500, seed=4242, n_gaps=6, n_corruptions=40)


@pytest.fixture(scope="session")
def generated_train() -> dataset.GeneratedRecord:
    """The training split at a size the suite can afford."""
    return dataset.training_record(samples=3000)


@pytest.fixture(scope="session")
def generated_test() -> dataset.GeneratedRecord:
    """The held-out split at a size the suite can afford."""
    return dataset.test_record(samples=3000)


@pytest.fixture(scope="session")
def trained_model(generated_train: dataset.GeneratedRecord) -> LearnedClassifier:
    """One fitted forest, shared by every learned-model test."""
    matrix, _ = features.build(generated_train.record)
    labels = features.truth_labels(generated_train.value_truth)
    return LearnedClassifier(n_estimators=40).fit(matrix, labels)


@pytest.fixture(scope="session")
def report_small(generated_small: dataset.GeneratedRecord) -> classify.IntegrityReport:
    """One strengthened-mode integrity report over ``generated_small``."""
    return classify.classify(generated_small.record)

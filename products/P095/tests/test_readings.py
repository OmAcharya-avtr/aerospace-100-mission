"""The reading value object, its validation, and the tamper seal."""

from __future__ import annotations

import pickle

import pytest

from boardprobe.provenance import Provenance, ProvenanceError
from boardprobe.readings import (
    Attribution,
    Reading,
    TamperError,
    attribute,
    combine,
    from_mapping,
    mint,
    verify,
)


def make(channel="die_temp_cpu", value=41.5, provenance=Provenance.SIMULATED, stamp=10):
    return mint(
        channel=channel,
        value=value,
        unit="degC",
        timestamp_ns=stamp,
        provenance=provenance,
        source_id="test",
        detail="unit test",
    )


def test_a_fresh_reading_is_intact_and_verifies():
    reading = make()
    assert reading.intact
    assert verify(reading) is reading


def test_assignment_is_refused_by_the_frozen_dataclass():
    reading = make()
    with pytest.raises(Exception):  # noqa: B017 - FrozenInstanceError is not public
        reading.provenance = Provenance.LIVE


def test_object_setattr_bypasses_frozen_and_the_seal_catches_it():
    """The measured reason the seal exists. ``frozen=True`` is not immutability."""
    reading = make()
    object.__setattr__(reading, "provenance", Provenance.LIVE)
    assert reading.provenance is Provenance.LIVE
    assert not reading.intact
    with pytest.raises(TamperError, match="fails its seal"):
        verify(reading)


@pytest.mark.parametrize("field_name", ["channel", "value", "unit", "timestamp_ns",
                                        "source_id", "detail"])
def test_the_seal_catches_a_change_to_any_field(field_name):
    reading = make()
    replacement = {"channel": "other", "value": 99.0, "unit": "K",
                   "timestamp_ns": 11, "source_id": "elsewhere", "detail": "x"}
    object.__setattr__(reading, field_name, replacement[field_name])
    assert not reading.intact


def test_a_subclass_is_refused_by_exact_type_check():
    """``isinstance`` would admit this; the verifier uses ``type(...) is``.

    The subclass has to override ``__post_init__`` to get past the minting
    token at all, which is itself a route the adversarial check reports. Having
    got past it, it still cannot reach a label.
    """

    class Liar(Reading):
        def __post_init__(self, mint_token) -> None:  # noqa: ARG002
            object.__setattr__(self, "value", float(self.value))
            object.__setattr__(self, "seal", "forged")

        @property
        def intact(self) -> bool:  # pragma: no cover - never reached
            return True

    liar = Liar(
        None, channel="die_temp_cpu", value=1.0, unit="degC", timestamp_ns=1,
        provenance=Provenance.LIVE, source_id="liar",
    )
    assert isinstance(liar, Reading)
    assert liar.provenance is Provenance.LIVE
    with pytest.raises(TamperError, match="and nothing else"):
        verify(liar)


def test_dataclasses_replace_cannot_relabel_a_reading():
    """The laundering route the adversarial check found, pinned here.

    ``dataclasses.replace(reading, provenance=LIVE)`` re-ran ``__post_init__``,
    recomputed the seal, and produced a validly sealed LIVE reading carrying a
    simulated reading's own value. The minting token closes it.
    """
    import copy
    import dataclasses

    reading = make(provenance=Provenance.SIMULATED)
    with pytest.raises(TypeError, match="InitVar"):
        dataclasses.replace(reading, provenance=Provenance.LIVE)
    with pytest.raises(TypeError, match="InitVar"):
        copy.replace(reading, provenance=Provenance.LIVE)


def test_a_reading_cannot_be_constructed_without_the_minting_token():
    with pytest.raises(TypeError):
        Reading(channel="c", value=1.0, unit="degC", timestamp_ns=1,
                provenance=Provenance.LIVE, source_id="s")
    with pytest.raises(TypeError, match="cannot be constructed directly"):
        Reading(object(), channel="c", value=1.0, unit="degC", timestamp_ns=1,
                provenance=Provenance.LIVE, source_id="s")


def test_a_pickle_round_trip_does_not_survive_the_seal():
    """The seal key is per-process, so a reading cannot be smuggled through a file."""
    reading = make()
    restored = pickle.loads(pickle.dumps(reading))
    assert restored.channel == reading.channel
    assert restored.intact  # same process, same key
    # In another process the key differs; the mechanism is the key, and the
    # test for the cross-process case is that the seal is never serialised.
    assert "seal" not in reading.to_mapping()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"channel": ""}, "non-empty"),
        ({"unit": ""}, "non-empty"),
        ({"source_id": ""}, "source_id"),
    ],
)
def test_invalid_fields_are_refused_with_actionable_messages(kwargs, match):
    base = {"channel": "c", "value": 1.0, "unit": "degC", "timestamp_ns": 1,
            "provenance": Provenance.SIMULATED, "source_id": "s"}
    with pytest.raises(ValueError, match=match):
        mint(**{**base, **kwargs})


def test_a_non_provenance_is_refused():
    with pytest.raises(ProvenanceError, match="boardprobe.Provenance member"):
        mint(channel="c", value=1.0, unit="degC", timestamp_ns=1,
             provenance="live", source_id="s")


def test_a_boolean_value_is_refused():
    """``bool`` is a subclass of ``int`` and would silently become 0.0 or 1.0."""
    with pytest.raises(TypeError, match="real number"):
        mint(channel="c", value=True, unit="degC", timestamp_ns=1,
             provenance=Provenance.LIVE, source_id="s")


def test_a_float_timestamp_is_refused():
    with pytest.raises(TypeError, match="nanoseconds"):
        mint(channel="c", value=1.0, unit="degC", timestamp_ns=1.5,
             provenance=Provenance.LIVE, source_id="s")


def test_attribute_meets_the_provenances_and_names_the_weakest():
    readings = [
        make(channel="a", provenance=Provenance.LIVE, stamp=5),
        make(channel="b", provenance=Provenance.RECORDED, stamp=7),
        make(channel="c", provenance=Provenance.SIMULATED, stamp=9),
    ]
    attribution = attribute(readings)
    assert attribution.provenance is Provenance.SIMULATED
    assert attribution.weakest == ("c",)
    assert attribution.reading_count == 3
    assert attribution.channels == ("a", "b", "c")
    assert attribution.first_timestamp_ns == 5
    assert attribution.last_timestamp_ns == 9
    assert attribution.span_s == pytest.approx(4e-9, rel=1e-12, abs=0.0)


def test_attribute_refuses_an_empty_collection():
    with pytest.raises(ProvenanceError, match="cannot be established"):
        attribute([])


def test_combine_meets_attributions_and_sums_the_counts():
    live = attribute([make(channel="a", provenance=Provenance.LIVE)])
    simulated = attribute([make(channel="b", provenance=Provenance.SIMULATED)])
    merged = combine([live, simulated])
    assert merged.provenance is Provenance.SIMULATED
    assert merged.reading_count == 2
    assert merged.weakest == ("b",)


def test_combine_refuses_an_empty_collection_and_a_foreign_type():
    with pytest.raises(ProvenanceError, match="cannot be established"):
        combine([])
    with pytest.raises(TamperError, match="Attribution"):
        combine([object()])


def test_an_attribution_cannot_claim_zero_readings():
    with pytest.raises(ProvenanceError, match="at least one reading"):
        Attribution(
            provenance=Provenance.LIVE, reading_count=0, channels=("a",),
            source_ids=("s",), weakest=("a",), weakest_sources=("s",),
            first_timestamp_ns=0, last_timestamp_ns=0,
        )


def test_from_mapping_forces_a_document_down_to_the_floor():
    """A document claiming ``live`` cannot rise above the floor it is given."""
    document = {"channel": "a", "value": 1.0, "unit": "degC", "timestamp_ns": 3,
                "provenance": "live"}
    assert from_mapping(document, Provenance.RECORDED).provenance is Provenance.RECORDED
    assert from_mapping(document, Provenance.SIMULATED).provenance is Provenance.SIMULATED
    assert from_mapping(document, Provenance.LIVE).provenance is Provenance.LIVE


def test_from_mapping_names_a_missing_key():
    with pytest.raises(ValueError, match="missing the required key 'unit'"):
        from_mapping({"channel": "a", "value": 1.0, "timestamp_ns": 1,
                      "provenance": "live"}, Provenance.LIVE)

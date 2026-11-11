"""Trace documents: what they may claim, and how a malformed one is reported."""

from __future__ import annotations

import json

import pytest

from boardprobe.provenance import Provenance
from boardprobe.trace import TRACE_SCHEMA, Trace, TraceError, TraceSample


def minimal(**overrides):
    base = {
        "schema": TRACE_SCHEMA,
        "board_model": "NVIDIA Jetson Orin Nano Developer Kit",
        "origin": "a hand-built document for this test, not a recording",
        "content_provenance": "recorded",
        "channels": ["die_temp_cpu"],
        "samples": [
            {"channel": "die_temp_cpu", "value": 41.0, "unit": "degC",
             "timestamp_ns": 0}
        ],
    }
    return {**base, **overrides}


def test_a_document_may_not_declare_live_content():
    """The one-line route from a text editor to a measured label, closed."""
    with pytest.raises(TraceError, match="not allowed in a trace"):
        Trace.from_mapping(minimal(content_provenance="live"))


def test_a_document_may_declare_recorded_or_simulated():
    for value in ("recorded", "simulated"):
        trace = Trace.from_mapping(minimal(content_provenance=value))
        assert trace.content_provenance is Provenance(value)


def test_an_unknown_provenance_string_is_refused():
    with pytest.raises(TraceError, match="is not a provenance"):
        Trace.from_mapping(minimal(content_provenance="measured"))


def test_a_wrong_schema_is_refused_rather_than_guessed_at():
    with pytest.raises(TraceError, match="refusing to guess"):
        Trace.from_mapping(minimal(schema="boardprobe/trace/2"))


@pytest.mark.parametrize("key", ["board_model", "origin", "content_provenance",
                                 "channels", "samples"])
def test_a_missing_key_is_named(key):
    payload = minimal()
    del payload[key]
    with pytest.raises(TraceError, match=key):
        Trace.from_mapping(payload)


def test_an_empty_sample_list_is_refused():
    with pytest.raises(TraceError, match="no samples"):
        Trace.from_mapping(minimal(samples=[]))


def test_an_empty_origin_is_refused():
    with pytest.raises(TraceError, match="non-empty origin"):
        Trace.from_mapping(minimal(origin="   "))


def test_a_sample_on_an_undeclared_channel_is_refused():
    payload = minimal(
        samples=[{"channel": "die_temp_gpu", "value": 1.0, "unit": "degC",
                  "timestamp_ns": 0}]
    )
    with pytest.raises(TraceError, match="inconsistent with itself"):
        Trace.from_mapping(payload)


def test_a_malformed_sample_names_its_index():
    payload = minimal(samples=[{"channel": "die_temp_cpu", "value": 1.0}])
    with pytest.raises(TraceError, match="sample 0 of the trace is malformed"):
        Trace.from_mapping(payload)


def test_a_partially_written_document_is_reported_as_truncated(tmp_path):
    """The failure mode docs/RECOVERY.md calls 'a partially written trace'."""
    path = tmp_path / "truncated.json"
    path.write_text(json.dumps(minimal())[:120])
    with pytest.raises(TraceError, match="not valid JSON"):
        Trace.from_file(path)


def test_a_missing_file_is_reported_by_name_only(tmp_path):
    path = tmp_path / "absent.json"
    with pytest.raises(TraceError, match="absent.json"):
        Trace.from_file(path)


def test_a_top_level_list_is_refused(tmp_path):
    path = tmp_path / "list.json"
    path.write_text("[1, 2, 3]")
    with pytest.raises(TraceError, match="at the top"):
        Trace.from_file(path)


def test_a_round_trip_preserves_the_content_provenance(idle_trace):
    reloaded = Trace.from_mapping(json.loads(idle_trace.to_json()))
    assert reloaded.content_provenance is idle_trace.content_provenance
    assert reloaded.sample_count == idle_trace.sample_count
    assert reloaded.origin == idle_trace.origin


def test_samples_for_names_what_is_present(idle_trace):
    with pytest.raises(TraceError, match="channels present are"):
        idle_trace.samples_for("no_such_channel")


def test_a_directly_constructed_trace_cannot_claim_live():
    with pytest.raises(TraceError, match="not allowed in a trace"):
        Trace(
            board_model="board",
            origin="direct construction",
            content_provenance=Provenance.LIVE,
            channels=("a",),
            samples=(TraceSample("a", 1.0, "degC", 0),),
        )

"""The committed trace fixtures, re-synthesised and compared value by value.

A fixture that cannot be regenerated is a number with no provenance, which in
this product would be an embarrassment. Each committed trace is re-synthesised
from its declared seed by ``boardprobe.sources.synthesise_trace`` and compared
sample by sample, so the fixture cannot drift from the generator that is said
to have produced it -- and a change in numpy's PCG64 stream or in
``Generator.standard_normal`` fails here, which is why ``numpy`` carries a
minor-version ceiling in ``pyproject.toml``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from boardprobe.orin_nano import ORIN_NANO
from boardprobe.provenance import Provenance
from boardprobe.sources import synthesise_trace
from boardprobe.trace import Trace

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"

IDLE_CHANNELS = (
    "die_temp_cpu", "die_temp_gpu", "die_temp_junction",
    "rail_vdd_in_voltage", "rail_vdd_in_current",
    "rail_cpu_gpu_cv_voltage", "rail_cpu_gpu_cv_current",
    "perf_instructions", "perf_cpu_cycles", "cpu0_frequency",
)

CASES = (
    ("orin_nano_idle_simulated_trace.json", IDLE_CHANNELS, 24, 20260101,
     Provenance.SIMULATED),
    ("replay_path_recorded_trace.json", IDLE_CHANNELS[:5], 12, 20260202,
     Provenance.RECORDED),
)


@pytest.mark.parametrize(("name", "channels", "count", "seed", "content"), CASES)
def test_the_committed_fixture_matches_its_declared_generator(
    name, channels, count, seed, content
):
    committed = Trace.from_file(FIXTURES / name)
    regenerated = synthesise_trace(
        ORIN_NANO, channels, count, seed=seed, origin=committed.origin,
        content_provenance=content, notes=committed.notes,
    )
    assert committed.channels == regenerated.channels
    assert committed.sample_count == regenerated.sample_count
    assert committed.content_provenance is content
    for left, right in zip(committed.samples, regenerated.samples, strict=True):
        assert left.channel == right.channel
        assert left.unit == right.unit
        assert left.timestamp_ns == right.timestamp_ns
        assert left.value == pytest.approx(right.value, rel=1e-15, abs=0.0)


@pytest.mark.parametrize("name", [case[0] for case in CASES])
def test_every_fixture_declares_its_origin_loudly(name):
    """Both fixtures say in the file that they are not board measurements."""
    trace = Trace.from_file(FIXTURES / name)
    assert "SYNTHESISED IN A CONTAINER" in trace.origin
    assert "NOT A RECORDING" in trace.origin
    assert trace.notes


@pytest.mark.parametrize("name", [case[0] for case in CASES])
def test_no_fixture_claims_live_content(name):
    trace = Trace.from_file(FIXTURES / name)
    assert trace.content_provenance is not Provenance.LIVE


@pytest.mark.parametrize("name", [case[0] for case in CASES])
def test_every_fixture_is_valid_json_with_the_declared_schema(name):
    payload = json.loads((FIXTURES / name).read_text())
    assert payload["schema"] == "boardprobe/trace/1"
    assert payload["board_model"] == ORIN_NANO.model


@pytest.mark.parametrize("name", [case[0] for case in CASES])
def test_every_fixture_is_present_and_small_enough_to_track(name):
    path = FIXTURES / name
    assert path.is_file()
    assert path.stat().st_size < 200_000


@pytest.mark.parametrize("name", [case[0] for case in CASES])
def test_no_fixture_sample_carries_a_channel_the_board_table_lacks(name):
    trace = Trace.from_file(FIXTURES / name)
    for channel in trace.channels:
        assert ORIN_NANO.channel(channel).name == channel

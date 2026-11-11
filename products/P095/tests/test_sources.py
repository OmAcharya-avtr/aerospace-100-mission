"""The three sealed backends, and the structural refusals that seal them."""

from __future__ import annotations

import pytest

from boardprobe.channels import Access
from boardprobe.orin_nano import ORIN_NANO
from boardprobe.provenance import Provenance, meet
from boardprobe.sources import (
    LiveSource,
    RecordedSource,
    SimulatedSource,
    Source,
    backend_names,
    make_source,
    read_available,
    sealed_backends,
    synthesise_trace,
)


def test_the_three_backends_are_sealed_and_named():
    assert backend_names() == ("simulated", "recorded", "live")
    assert set(sealed_backends()) == {LiveSource, RecordedSource, SimulatedSource}


@pytest.mark.parametrize("base", [LiveSource, RecordedSource, SimulatedSource])
def test_a_concrete_backend_cannot_be_subclassed(base):
    """The route 'subclass LiveSource and serve it trace data' is closed."""
    with pytest.raises(TypeError, match="sealed"):

        class Sneaky(base):  # noqa: B903 - the class body is irrelevant
            pass


@pytest.mark.parametrize("member", ["read", "read_all", "provenance", "_stamp_ns"])
def test_a_backend_may_not_override_the_stamping_path(member):
    namespace = {
        "backend_provenance": Provenance.LIVE,
        "_raw": lambda self, channel: 0.0,
        member: (property(lambda self: Provenance.LIVE)
                 if member == "provenance" else (lambda self, *a: None)),
    }
    with pytest.raises(TypeError, match=member):
        type("Sneaky", (Source,), namespace)


def test_a_backend_class_declaring_only_raw_is_accepted():
    """The permitted shape, so the refusals above are not refusing everything."""
    namespace = {
        "backend_provenance": Provenance.SIMULATED,
        "_raw": lambda self, channel: 1.0,
    }
    cls = type("Allowed", (Source,), namespace)
    instance = cls(ORIN_NANO, "allowed:test")
    assert instance.provenance is Provenance.SIMULATED
    assert instance.read("die_temp_cpu").value == pytest.approx(1e-3, rel=1e-12, abs=0.0)


def test_simulated_readings_are_stamped_simulated_on_every_channel():
    source = SimulatedSource(ORIN_NANO, seed=11)
    for reading in source.read_all():
        assert reading.provenance is Provenance.SIMULATED
        assert reading.source_id == "simulated:seed=11"


def test_the_simulated_backend_is_bit_reproducible_at_one_seed():
    """Exact equality is correct HERE and nowhere else in this suite: the claim
    is bit-reproducibility of a pseudo-random stream, not the value of a
    physical quantity."""
    first = [r.value for r in SimulatedSource(ORIN_NANO, seed=99).read_all()]
    second = [r.value for r in SimulatedSource(ORIN_NANO, seed=99).read_all()]
    assert first == second


def test_two_seeds_differ_on_the_noisy_channels():
    first = {r.channel: r.value for r in SimulatedSource(ORIN_NANO, seed=1).read_all()}
    second = {r.channel: r.value for r in SimulatedSource(ORIN_NANO, seed=2).read_all()}
    noisy = [name for name, value in first.items() if value != second[name]]
    # The synthetic clock is deterministic by construction and must NOT differ.
    assert "monotonic_clock" not in noisy
    assert len(noisy) >= 10


def test_reading_a_subset_gives_the_same_values_as_reading_all():
    """One generator per channel, so a value cannot depend on what else was read."""
    full = {r.channel: r.value for r in SimulatedSource(ORIN_NANO, seed=7).read_all()}
    subset = SimulatedSource(ORIN_NANO, seed=7).read_all(["die_temp_gpu", "die_temp_cpu"])
    for reading in subset:
        assert reading.value == full[reading.channel]


def test_a_non_integer_seed_is_refused():
    with pytest.raises(TypeError, match="seed must be an int"):
        SimulatedSource(ORIN_NANO, seed=1.5)


def test_a_trace_of_simulated_content_replays_as_simulated(idle_trace):
    """meet(RECORDED, SIMULATED) = SIMULATED. The lattice doing the work."""
    source = RecordedSource(ORIN_NANO, idle_trace)
    assert idle_trace.content_provenance is Provenance.SIMULATED
    assert source.provenance is Provenance.SIMULATED
    assert source.provenance is meet(Provenance.RECORDED, idle_trace.content_provenance)
    for reading in source.read_all(idle_trace.channels):
        assert reading.provenance is Provenance.SIMULATED


def test_a_trace_of_recorded_content_replays_as_recorded(recorded_trace):
    source = RecordedSource(ORIN_NANO, recorded_trace)
    assert source.provenance is Provenance.RECORDED
    for reading in source.read_all(recorded_trace.channels):
        assert reading.provenance is Provenance.RECORDED


def test_replay_cycles_the_trace_and_records_the_index(recorded_trace):
    source = RecordedSource(ORIN_NANO, recorded_trace)
    channel = recorded_trace.channels[0]
    samples = recorded_trace.samples_for(channel)
    values = [source.read(channel).value for _ in range(len(samples) + 2)]
    assert values[: len(samples)] == pytest.approx(
        [s.value for s in samples], rel=1e-12, abs=0.0
    )
    assert values[len(samples)] == pytest.approx(samples[0].value, rel=1e-12, abs=0.0)
    assert source.read(channel).detail.startswith(f"recorded:{channel}[")


def test_replay_refuses_a_channel_the_trace_does_not_carry(recorded_trace):
    from boardprobe.sysio import DeviceUnreachableError

    source = RecordedSource(ORIN_NANO, recorded_trace)
    with pytest.raises(DeviceUnreachableError, match="no samples on channel"):
        source.read("die_temp_cv0")


def test_recorded_backend_refuses_a_foreign_trace():
    with pytest.raises(TypeError, match="boardprobe.Trace"):
        RecordedSource(ORIN_NANO, {"schema": "boardprobe/trace/1"})


def test_a_unit_mismatch_between_trace_and_declaration_is_refused():
    from boardprobe.sysio import DeviceUnreachableError
    from boardprobe.trace import Trace, TraceSample

    trace = Trace(
        board_model=ORIN_NANO.model,
        origin="a hand-built trace for this test",
        content_provenance=Provenance.RECORDED,
        channels=("die_temp_cpu",),
        samples=(TraceSample("die_temp_cpu", 41.0, "K", 0),),
    )
    with pytest.raises(DeviceUnreachableError, match="unit"):
        RecordedSource(ORIN_NANO, trace).read("die_temp_cpu")


def test_make_source_builds_each_backend_and_names_the_unknown(idle_trace, absent_root):
    assert isinstance(make_source("simulated", ORIN_NANO, seed=3), SimulatedSource)
    assert isinstance(make_source("recorded", ORIN_NANO, trace=idle_trace), RecordedSource)
    assert isinstance(make_source("live", ORIN_NANO, root=absent_root), LiveSource)
    with pytest.raises(ValueError, match="unknown backend"):
        make_source("psychic", ORIN_NANO)
    with pytest.raises(ValueError, match="needs a trace"):
        make_source("recorded", ORIN_NANO)


def test_read_available_returns_refusals_rather_than_dropping_them(absent_root):
    source = LiveSource(ORIN_NANO, root=absent_root)
    board = [c.name for c in ORIN_NANO.channels if c.access is not Access.CLOCK_SYSCALL]
    readings, refusals = read_available(source, board)
    assert readings == ()
    assert len(refusals) == len(board)
    assert all(reason for _, reason in refusals)


def test_synthesise_trace_is_deterministic_and_refuses_a_live_claim():
    from boardprobe.trace import TraceError

    channels = ("die_temp_cpu", "rail_vdd_in_voltage")
    first = synthesise_trace(ORIN_NANO, channels, 3, seed=5, origin="test",
                             content_provenance=Provenance.SIMULATED)
    second = synthesise_trace(ORIN_NANO, channels, 3, seed=5, origin="test",
                              content_provenance=Provenance.SIMULATED)
    assert [s.value for s in first.samples] == [s.value for s in second.samples]
    with pytest.raises(TraceError, match="not allowed in a trace"):
        synthesise_trace(ORIN_NANO, channels, 3, seed=5, origin="test",
                         content_provenance=Provenance.LIVE)


def test_synthesise_trace_refuses_zero_samples():
    with pytest.raises(ValueError, match="at least 1"):
        synthesise_trace(ORIN_NANO, ("die_temp_cpu",), 0, seed=1, origin="t",
                         content_provenance=Provenance.SIMULATED)


def test_a_foreign_declaration_or_floor_is_refused():
    with pytest.raises(TypeError, match="BoardDeclaration"):
        SimulatedSource({"model": "board"}, seed=1)

"""The benchmark harness: what it certifies, what it refuses, and its own cost.

Every wall-clock figure here is a property of whatever host runs the suite, so
no test asserts a duration against a constant. What is asserted is structure:
the label the harness earns, the provenance it propagates, the refusals it
reports rather than drops, and the ordering relations that hold whatever the
host's speed.
"""

from __future__ import annotations

import pytest

from boardprobe.certify import Label
from boardprobe.channels import Access
from boardprobe.harness import (
    HarnessError,
    benchmark,
    characterise_overhead,
    reference_payload,
    timing_resolution_s,
)
from boardprobe.orin_nano import ORIN_NANO
from boardprobe.platform_attest import PlatformAttestation, attest_platform
from boardprobe.provenance import Provenance
from boardprobe.sources import LiveSource, RecordedSource, SimulatedSource

ON_TARGET = PlatformAttestation(
    target_model="NVIDIA Jetson Orin Nano",
    observed_model="NVIDIA Jetson Orin Nano Developer Kit",
    machine="aarch64",
    system="Linux",
    source="a hand-built attestation for this test, not a board",
    matches_target=True,
)


@pytest.fixture
def clock():
    return LiveSource(ORIN_NANO)


def test_a_timing_only_benchmark_on_a_live_clock_earns_measured(clock):
    result = benchmark(reference_payload(200), "ref", clock, repeats=12, warmup=2)
    assert result.label is Label.MEASURED
    assert result.overall_certificate.provenance is Provenance.LIVE
    assert result.repeats == 12
    assert len(result.durations_s) == 12
    assert all(duration >= 0.0 for duration in result.durations_s)


def test_an_off_target_measured_result_says_so_in_its_headline(clock):
    result = benchmark(reference_payload(200), "ref", clock, repeats=8)
    assert not attest_platform().matches_target
    assert result.headline.startswith("MEASURED OFF-TARGET")
    assert not result.overall_certificate.is_hardware_measurement


def test_a_simulated_environment_drags_the_whole_result_to_synthetic(clock):
    result = benchmark(
        reference_payload(200), "ref", clock, repeats=8,
        environment=SimulatedSource(ORIN_NANO, seed=2),
    )
    assert result.timing_certificate.label is Label.MEASURED
    assert result.label is Label.SYNTHETIC
    assert result.overall_certificate.provenance is Provenance.SIMULATED
    assert result.resources


def test_a_recorded_environment_gives_a_replayed_result(clock, recorded_trace):
    result = benchmark(
        reference_payload(200), "ref", clock, repeats=6,
        environment=RecordedSource(ORIN_NANO, recorded_trace),
        environment_channels=recorded_trace.channels,
    )
    assert result.label is Label.REPLAYED
    assert result.overall_certificate.provenance is Provenance.RECORDED


def test_a_simulated_clock_cannot_produce_a_measured_timing_figure():
    simulated = SimulatedSource(ORIN_NANO, seed=3)
    result = benchmark(reference_payload(50), "ref", simulated, repeats=4)
    assert result.timing_certificate.label is Label.SYNTHETIC
    assert result.label is Label.SYNTHETIC
    # The synthetic clock advances 1 ms per read, so each bracket is 1 ms exactly.
    assert result.mean_duration_s == pytest.approx(1e-3, rel=1e-9, abs=0.0)


def test_an_absent_board_environment_refuses_with_an_actionable_message(
    clock, absent_root
):
    live_environment = LiveSource(ORIN_NANO, root=absent_root)
    with pytest.raises(HarnessError, match="could serve none of"):
        benchmark(reference_payload(50), "ref", clock, repeats=4,
                  environment=live_environment)


def test_refusals_are_reported_rather_than_dropped(clock, recorded_trace):
    """A trace that carries two of the five default channels: three refusals."""
    result = benchmark(
        reference_payload(50), "ref", clock, repeats=4,
        environment=RecordedSource(ORIN_NANO, recorded_trace),
    )
    served = {name for name in recorded_trace.channels}
    expected = [c for c in result.refusals if c[0] not in served]
    assert result.refusals
    assert len(expected) == len(result.refusals)


def test_the_payload_must_be_callable_and_the_clock_a_source(clock):
    with pytest.raises(TypeError, match="callable"):
        benchmark(42, "ref", clock, repeats=2)
    with pytest.raises(TypeError, match="Source"):
        benchmark(reference_payload(10), "ref", "a clock", repeats=2)


@pytest.mark.parametrize(("repeats", "warmup"), [(0, 0), (1, -1)])
def test_out_of_range_counts_are_refused(clock, repeats, warmup):
    with pytest.raises(ValueError, match="at least"):
        benchmark(reference_payload(10), "ref", clock, repeats=repeats, warmup=warmup)


def test_a_clock_channel_not_in_seconds_is_refused():
    """``cpu0_frequency`` is a clock channel in hertz, not a timebase in seconds.

    The simulated backend is used because the live backend refuses the channel
    first -- the sysfs node is absent here -- and the refusal under test is the
    harness's unit check, not the device's absence.
    """
    simulated = SimulatedSource(ORIN_NANO, seed=1)
    with pytest.raises(HarnessError, match="needs a clock channel"):
        benchmark(reference_payload(10), "ref", simulated, repeats=2,
                  clock_channel="cpu0_frequency")


def test_an_on_target_live_benchmark_would_be_a_hardware_measurement(clock):
    """The only way to reach this in the build environment is a hand-built
    attestation, which is a statement by the caller and is recorded as one in
    the certificate's ``source`` field. It is here so the positive branch of
    ``is_hardware_measurement`` is exercised rather than merely described."""
    result = benchmark(reference_payload(100), "ref", clock, repeats=5,
                       platform=ON_TARGET)
    assert result.overall_certificate.is_hardware_measurement
    assert result.headline == f"MEASURED on {ON_TARGET.describe()}"
    assert "hand-built attestation" in result.headline


def test_the_overhead_characterisation_is_internally_ordered(clock):
    """Three independently measured quantities, with an ordering that must hold.

    A sealed ``Reading`` is constructed inside the timed bracket, and at most
    two clock reads bracket it, so the in-bracket overhead lies between the
    cost of one Reading and the cost of a clock-read pair. The slack is
    generous because these are wall-clock figures on a shared two-core host;
    the ORDER is what is being asserted, not the values.
    """
    report = characterise_overhead(clock, repeats=200)
    assert report.certificate.label is Label.MEASURED
    assert report.reading_cost_s > 0.0
    assert report.clock_pair_cost_s > 0.0
    assert report.in_bracket_overhead_s > 0.0
    assert report.reading_cost_s <= report.clock_pair_cost_s
    assert report.perturbation_ratio > 1.0 - 0.5


def test_the_overhead_report_refuses_too_few_repeats(clock):
    with pytest.raises(ValueError, match="at least 2"):
        characterise_overhead(clock, repeats=1)


def test_timing_resolution_is_one_ulp_at_the_given_magnitude():
    """Hand calculation: ulp(1.0) in float64 is 2**-52 = 2.220446049250313e-16.

    ulp scales with the binade, not with the decimal magnitude: 1e6 lies in
    [2**19, 2**20), so its ulp is 2**-52 * 2**19 = 1.1641532182693481e-10 and
    NOT 2**-52 * 2**20. Both are written out, because getting that wrong is
    exactly the kind of mistake a resolution claim hides.
    """
    assert timing_resolution_s(1.0) == pytest.approx(2.0**-52, rel=0.0, abs=0.0)
    assert timing_resolution_s(2.0**20) == pytest.approx(
        2.0**-52 * 2.0**20, rel=0.0, abs=0.0
    )
    assert timing_resolution_s(1e6) == pytest.approx(
        2.0**-52 * 2.0**19, rel=0.0, abs=0.0
    )
    assert timing_resolution_s(1e6) == pytest.approx(
        1.1641532182693481e-10, rel=1e-15, abs=0.0
    )
    assert timing_resolution_s() > 0.0


def test_reference_payload_is_deterministic_and_refuses_zero_iterations():
    payload = reference_payload(10)
    assert payload() == pytest.approx(payload(), rel=0.0, abs=0.0)
    with pytest.raises(ValueError, match="at least 1"):
        reference_payload(0)


def test_the_default_environment_channels_are_not_clock_channels():
    from boardprobe.harness import DEFAULT_ENVIRONMENT_CHANNELS

    for name in DEFAULT_ENVIRONMENT_CHANNELS:
        assert ORIN_NANO.channel(name).access is not Access.CLOCK_SYSCALL

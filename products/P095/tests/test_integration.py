"""End-to-end: readings to a certified report, through every backend.

These exercise the whole pipeline the way a user does -- choose a backend,
read channels, aggregate, derive, certify, render -- and assert the one thing
that must hold at every step: the label on the output follows the weakest
input, and nothing anywhere produces a bare ``MEASURED`` off the declared
target.
"""

from __future__ import annotations

import pytest

from boardprobe import (
    ORIN_NANO,
    RAIL_POWER_PAIRS,
    Label,
    LiveSource,
    Provenance,
    RecordedSource,
    SimulatedSource,
    attest_platform,
    attribute,
    benchmark,
    certify,
    combine,
    derive,
    label_for,
    rail_power,
    read_available,
    reference_payload,
    render_benchmark,
    render_certificate,
    render_readings,
    render_statistics,
    summarise,
)


def test_a_full_simulated_campaign_ends_in_a_synthetic_certificate():
    source = SimulatedSource(ORIN_NANO, seed=31)
    thermal = [source.read("die_temp_cpu") for _ in range(16)]
    statistics = summarise(thermal)
    powers = [
        rail_power(source.read(voltage), source.read(current), name)
        for name, voltage, current in RAIL_POWER_PAIRS
    ]
    headroom = derive(
        "headroom to 85 degC", 85.0 - statistics[1].value, "degC",
        [statistics[1], *powers],
    )
    certificate = certify(
        "campaign summary",
        combine([headroom.attribution]),
        label_for(headroom.provenance),
        attest_platform(),
    )
    assert certificate.label is Label.SYNTHETIC
    assert not certificate.is_hardware_measurement
    rendered = "\n".join(
        (render_readings(thermal[:3]), render_statistics([*statistics, headroom]),
         render_certificate(certificate))
    )
    assert "simulated" in rendered
    assert "SYNTHETIC" in rendered


def test_a_replayed_campaign_ends_in_a_replayed_certificate(recorded_trace):
    source = RecordedSource(ORIN_NANO, recorded_trace)
    readings, refusals = read_available(source, recorded_trace.channels)
    assert refusals == ()
    attribution = attribute(readings)
    certificate = certify(
        "replayed campaign", attribution, label_for(attribution.provenance),
        attest_platform(),
    )
    assert certificate.label is Label.REPLAYED
    assert certificate.provenance is Provenance.RECORDED


def test_mixing_a_live_clock_with_a_replayed_environment_gives_replayed(
    recorded_trace,
):
    clock = LiveSource(ORIN_NANO)
    result = benchmark(
        reference_payload(100), "mixed", clock, repeats=6,
        environment=RecordedSource(ORIN_NANO, recorded_trace),
        environment_channels=recorded_trace.channels,
    )
    assert result.timing_certificate.label is Label.MEASURED
    assert result.label is Label.REPLAYED
    rendered = render_benchmark(result)
    assert "REPLAYED" in rendered
    assert "MEASURED OFF-TARGET" not in rendered.splitlines()[1]


def test_adding_one_simulated_channel_to_a_replayed_set_drops_it_to_synthetic(
    recorded_trace,
):
    recorded = RecordedSource(ORIN_NANO, recorded_trace)
    simulated = SimulatedSource(ORIN_NANO, seed=2)
    readings = [
        *recorded.read_all(recorded_trace.channels),
        simulated.read("die_temp_cv0"),
    ]
    attribution = attribute(readings)
    assert attribution.provenance is Provenance.SIMULATED
    assert attribution.weakest == ("die_temp_cv0",)


def test_no_rendered_output_anywhere_contains_a_bare_measured_off_target():
    """Every ``MEASURED`` this build can produce names the platform beside it."""
    clock = LiveSource(ORIN_NANO)
    result = benchmark(reference_payload(100), "timing only", clock, repeats=6)
    rendered = render_benchmark(result)
    for line in rendered.splitlines():
        if "MEASURED" in line:
            assert "OFF-TARGET" in line
            assert attest_platform().target_model in line


def test_every_public_name_is_importable_from_the_package_root():
    import boardprobe

    for name in boardprobe.__all__:
        assert hasattr(boardprobe, name), name


def test_the_package_declares_the_exact_validation_level_phrase():
    import boardprobe

    assert boardprobe.VALIDATION_LEVEL == "Level 3, hardware-pending"


def test_the_declared_rail_power_pairs_resolve_to_declared_channels():
    for name, voltage, current in RAIL_POWER_PAIRS:
        assert ORIN_NANO.channel(voltage).unit == "V"
        assert ORIN_NANO.channel(current).unit == "A"
        assert name.endswith("_power")


def test_the_whole_pipeline_refuses_a_measured_label_at_every_entry_point():
    """Six entry points, one refusal. The choke point has no bypass."""
    from boardprobe.certify import LabelViolation

    source = SimulatedSource(ORIN_NANO, seed=8)
    readings = [source.read("die_temp_cpu") for _ in range(4)]
    attempts = [
        lambda: certify("a", attribute(readings), Label.MEASURED, attest_platform()),
        lambda: certify("b", combine([attribute(readings)]), Label.MEASURED,
                        attest_platform()),
        lambda: certify("c", summarise(readings)[1].attribution, Label.MEASURED,
                        attest_platform()),
        lambda: certify("d", derive("x", 1.0, "W", [summarise(readings)[1]]).attribution,
                        Label.MEASURED, attest_platform()),
        lambda: certify("e", rail_power(source.read("rail_vdd_in_voltage"),
                                        source.read("rail_vdd_in_current"),
                                        "p").attribution,
                        Label.MEASURED, attest_platform()),
        lambda: certify("f", attribute([*readings, source.read("die_temp_gpu")]),
                        Label.MEASURED, attest_platform()),
    ]
    for attempt in attempts:
        with pytest.raises(LabelViolation):
            attempt()

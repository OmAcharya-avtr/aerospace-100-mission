"""The choke point: every route to a label, and every refusal."""

from __future__ import annotations

import pytest

from boardprobe.certify import (
    Certificate,
    Label,
    LabelViolation,
    certify,
    label_for,
)
from boardprobe.platform_attest import PlatformAttestation, attest_platform
from boardprobe.provenance import ORDER, Provenance
from boardprobe.readings import attribute, mint

TARGET = PlatformAttestation(
    target_model="NVIDIA Jetson Orin Nano",
    observed_model="NVIDIA Jetson Orin Nano Developer Kit",
    machine="aarch64",
    system="Linux",
    source="a hand-built attestation for this test, not a board",
    matches_target=True,
)


def reading(provenance, channel="die_temp_cpu"):
    return mint(channel=channel, value=1.0, unit="degC", timestamp_ns=1,
                   provenance=provenance, source_id=f"{provenance.value}:test")


@pytest.mark.parametrize(
    ("provenance", "label"),
    [
        (Provenance.LIVE, Label.MEASURED),
        (Provenance.RECORDED, Label.REPLAYED),
        (Provenance.SIMULATED, Label.SYNTHETIC),
    ],
)
def test_label_for_is_the_declared_mapping(provenance, label):
    assert label_for(provenance) is label


def test_only_live_maps_to_measured():
    measured = [p for p in ORDER if label_for(p) is Label.MEASURED]
    assert measured == [Provenance.LIVE]


@pytest.mark.parametrize("provenance", list(ORDER))
def test_the_admitted_label_is_granted(provenance):
    attribution = attribute([reading(provenance)])
    certificate = certify("subject", attribution, label_for(provenance), TARGET)
    assert certificate.label is label_for(provenance)
    assert certificate.provenance is provenance


@pytest.mark.parametrize("provenance", [Provenance.RECORDED, Provenance.SIMULATED])
def test_measured_is_refused_on_anything_but_live(provenance):
    attribution = attribute([reading(provenance)])
    with pytest.raises(LabelViolation) as error:
        certify("the figure", attribution, Label.MEASURED, TARGET)
    message = str(error.value)
    assert "cannot be labelled MEASURED" in message
    assert "die_temp_cpu" in message
    assert f"{provenance.value}:test" in message


def test_one_simulated_reading_in_a_live_set_refuses_the_label():
    attribution = attribute(
        [reading(Provenance.LIVE, "a"), reading(Provenance.LIVE, "b"),
         reading(Provenance.SIMULATED, "c")]
    )
    with pytest.raises(LabelViolation, match="Offending channel"):
        certify("mixed", attribution, Label.MEASURED, TARGET)


def test_a_certificate_cannot_be_constructed_directly():
    attribution = attribute([reading(Provenance.LIVE)])
    with pytest.raises(TypeError, match="cannot be constructed directly"):
        Certificate(object(), "subject", Label.MEASURED, attribution, TARGET)


def test_a_foreign_attribution_is_refused():
    class Fake:
        provenance = Provenance.LIVE
        weakest = ("a",)
        weakest_sources = ("s",)
        reading_count = 1

    with pytest.raises(TypeError, match="Attribution"):
        certify("subject", Fake(), Label.MEASURED, TARGET)


def test_a_subclass_of_attribution_is_refused():
    from boardprobe.readings import Attribution

    class Sneaky(Attribution):
        pass

    sneaky = Sneaky(
        provenance=Provenance.LIVE, reading_count=1, channels=("a",),
        source_ids=("s",), weakest=("a",), weakest_sources=("s",),
        first_timestamp_ns=0, last_timestamp_ns=0,
    )
    with pytest.raises(TypeError, match="subclass is refused"):
        certify("subject", sneaky, Label.MEASURED, TARGET)


def test_a_string_label_is_refused():
    attribution = attribute([reading(Provenance.LIVE)])
    with pytest.raises(TypeError, match="boardprobe.Label member"):
        certify("subject", attribution, "MEASURED", TARGET)


def test_an_empty_subject_is_refused():
    attribution = attribute([reading(Provenance.LIVE)])
    with pytest.raises(TypeError, match="non-empty string"):
        certify("   ", attribution, Label.MEASURED, TARGET)


def test_a_foreign_platform_is_refused():
    attribution = attribute([reading(Provenance.LIVE)])
    with pytest.raises(TypeError, match="PlatformAttestation"):
        certify("subject", attribution, Label.MEASURED, "a jetson, honestly")


def test_an_off_target_measurement_cannot_render_a_bare_measured():
    here = attest_platform()
    assert not here.matches_target
    attribution = attribute([reading(Provenance.LIVE)])
    certificate = certify("subject", attribution, Label.MEASURED, here)
    assert certificate.label is Label.MEASURED
    assert certificate.headline.startswith("MEASURED OFF-TARGET")
    assert here.target_model in certificate.headline
    assert not certificate.is_hardware_measurement
    assert not certificate.on_declared_target


def test_only_live_on_the_declared_target_is_a_hardware_measurement():
    attribution = attribute([reading(Provenance.LIVE)])
    on_target = certify("subject", attribution, Label.MEASURED, TARGET)
    assert on_target.is_hardware_measurement
    assert on_target.headline == f"MEASURED on {TARGET.describe()}"
    replayed = certify(
        "subject", attribute([reading(Provenance.RECORDED)]), Label.REPLAYED, TARGET
    )
    assert not replayed.is_hardware_measurement


def test_the_mapping_carries_the_headline_and_never_a_bare_label():
    here = attest_platform()
    certificate = certify(
        "subject", attribute([reading(Provenance.LIVE)]), Label.MEASURED, here
    )
    mapping = certificate.to_mapping()
    assert mapping["label"] == "MEASURED"
    assert mapping["headline"].startswith("MEASURED OFF-TARGET")
    assert mapping["is_hardware_measurement"] is False
    assert mapping["declared_target"] == here.target_model

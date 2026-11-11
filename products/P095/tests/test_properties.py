"""Property-based tests over the whole propagation path, not just the lattice.

The lattice laws are in ``tests/test_provenance_properties.py``. What is here
is the claim that the propagation through readings, attributions, statistics
and the choke point agrees with the lattice for every shape of input
Hypothesis can build: the label granted is always the label the meet admits,
and the other two are always refused.

Every falsifying example these tests have found is pinned by literal inputs in
``tests/test_regressions.py``, because ``.hypothesis/`` is gitignored and an
example found by one run is otherwise lost on a cold clone.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from boardprobe.certify import Label, LabelViolation, certify, label_for
from boardprobe.platform_attest import attest_platform
from boardprobe.provenance import ORDER, Provenance, meet_all, rank
from boardprobe.readings import attribute, combine, from_mapping, mint
from boardprobe.statistics import derive, summarise

PLATFORM = attest_platform()
SETTINGS = settings(max_examples=250, deadline=None)

provenances = st.sampled_from(ORDER)
channel_names = st.sampled_from(["die_temp_cpu", "die_temp_gpu", "rail_vdd_in_voltage"])
values = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)
wide_values = st.floats(allow_nan=False, allow_infinity=False, width=64)


def build(items):
    """Readings from ``(channel, provenance, value)`` triples."""
    return [
        mint(channel=channel, value=value, unit="degC", timestamp_ns=index,
                provenance=provenance, source_id=f"{provenance.value}:{index}")
        for index, (channel, provenance, value) in enumerate(items)
    ]


reading_specs = st.lists(
    st.tuples(channel_names, provenances, values), min_size=1, max_size=10
)


@SETTINGS
@given(reading_specs)
def test_the_attribution_provenance_is_the_meet_of_its_readings(items):
    readings = build(items)
    attribution = attribute(readings)
    assert attribution.provenance is meet_all([r.provenance for r in readings])


@SETTINGS
@given(reading_specs)
def test_the_weakest_channels_are_exactly_those_at_the_meet(items):
    readings = build(items)
    attribution = attribute(readings)
    expected = []
    for reading in readings:
        if reading.provenance is attribution.provenance and reading.channel not in expected:
            expected.append(reading.channel)
    assert list(attribution.weakest) == expected


@SETTINGS
@given(reading_specs)
def test_exactly_one_label_is_granted_and_the_other_two_refused(items):
    readings = build(items)
    attribution = attribute(readings)
    admitted = label_for(attribution.provenance)
    granted = certify("subject", attribution, admitted, PLATFORM)
    assert granted.label is admitted
    for label in Label:
        if label is admitted:
            continue
        with pytest.raises(LabelViolation):
            certify("subject", attribution, label, PLATFORM)


@SETTINGS
@given(reading_specs)
def test_measured_is_granted_only_when_every_reading_is_live(items):
    readings = build(items)
    attribution = attribute(readings)
    all_live = all(r.provenance is Provenance.LIVE for r in readings)
    if all_live:
        assert certify("s", attribution, Label.MEASURED, PLATFORM).label is Label.MEASURED
    else:
        with pytest.raises(LabelViolation):
            certify("s", attribution, Label.MEASURED, PLATFORM)


@SETTINGS
@given(reading_specs, st.integers(min_value=1, max_value=9))
def test_propagation_through_two_levels_equals_one_level(items, split):
    """Splitting a reading set, attributing each part and combining gives the
    same provenance as attributing the whole set. Associativity, through the
    types rather than over the enum."""
    readings = build(items)
    assume(0 < split < max(2, len(readings)))
    cut = min(split, len(readings) - 1) if len(readings) > 1 else 1
    left, right = readings[:cut], readings[cut:]
    if not left or not right:
        return
    one_level = attribute(readings).provenance
    two_level = combine([attribute(left), attribute(right)]).provenance
    assert one_level is two_level


@SETTINGS
@given(reading_specs)
def test_a_derived_figure_is_never_stronger_than_its_weakest_part(items):
    readings = build(items)
    parts = [attribute([reading]) for reading in readings]
    derived = derive("derived", 1.0, "W", parts)
    assert rank(derived.provenance) == min(rank(part.provenance) for part in parts)


@SETTINGS
@given(st.lists(values, min_size=2, max_size=40), provenances)
def test_the_mean_lies_between_the_min_and_the_max(sample, provenance):
    readings = build([("die_temp_cpu", provenance, value) for value in sample])
    statistics = {s.name: s.value for s in summarise(readings)}
    # A tolerance on both sides: the mean is a sum of rounded quantities and
    # can land one rounding step outside the closed interval.
    span = max(abs(statistics["min"]), abs(statistics["max"]), 1.0)
    slack = 8.0 * np.finfo(float).eps * span * len(sample)
    assert statistics["min"] - slack <= statistics["mean"] <= statistics["max"] + slack


@SETTINGS
@given(st.lists(wide_values, min_size=2, max_size=12), provenances)
def test_the_mean_of_wide_magnitude_values_stays_finite_or_is_reported(
    sample, provenance
):
    """Deliberately unbounded magnitudes. A mean that overflows to infinity is a
    real defect in a reporting layer -- Hypothesis found one here during this
    build -- so every aggregate must come back finite OR be refused by name.
    Silently returning infinity is the outcome this test exists to forbid."""
    readings = build([("die_temp_cpu", provenance, value) for value in sample])
    try:
        statistics = {s.name: s.value for s in summarise(readings)}
    except ValueError as error:
        assert "not representable as a float64" in str(error)
        return
    for name, value in statistics.items():
        assert np.isfinite(value), f"the {name} of {sample} came back as {value}"


@SETTINGS
@given(provenances, provenances, values)
def test_a_document_can_never_rise_above_the_floor_it_is_given(claim, floor, value):
    document = {"channel": "die_temp_cpu", "value": value, "unit": "degC",
                "timestamp_ns": 1, "provenance": claim.value}
    rebuilt = from_mapping(document, floor)
    assert rank(rebuilt.provenance) <= rank(floor)
    assert rank(rebuilt.provenance) <= rank(claim)


@SETTINGS
@given(channel_names, provenances, values, st.sampled_from(
    ["channel", "value", "unit", "timestamp_ns", "source_id", "detail"]
))
def test_any_single_field_mutation_breaks_the_seal(channel, provenance, value, field):
    reading = mint(channel=channel, value=value, unit="degC", timestamp_ns=3,
                      provenance=provenance, source_id="s", detail="d")
    replacement = {"channel": channel + "_x", "value": value + 1.0, "unit": "K",
                   "timestamp_ns": 4, "source_id": "s_x", "detail": "d_x"}[field]
    object.__setattr__(reading, field, replacement)
    assert not reading.intact

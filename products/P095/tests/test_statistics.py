"""Aggregates, and the attribution they are obliged to carry."""

from __future__ import annotations

import numpy as np
import pytest

from boardprobe.orin_nano import ORIN_NANO
from boardprobe.provenance import Provenance
from boardprobe.readings import attribute, mint
from boardprobe.sources import SimulatedSource
from boardprobe.statistics import Statistic, derive, rail_power, summarise


def readings(values, provenance=Provenance.SIMULATED, channel="die_temp_cpu", unit="degC"):
    return [
        mint(channel=channel, value=value, unit=unit, timestamp_ns=index,
                provenance=provenance, source_id="test")
        for index, value in enumerate(values)
    ]


def test_known_answer_by_hand():
    """Values 40, 42, 44, 46. Mean 43; sd with ddof=1 is sqrt(80/3) = 2.581988897...

    Hand calculation: deviations -3, -1, 1, 3; squares 9, 1, 1, 9; sum 20;
    divided by n-1 = 3 gives 20/3; sqrt(20/3) = 2.581988897471611.
    Median is (42+44)/2 = 43. The 0.95 quantile with numpy's default linear
    interpolation over 4 points is 44 + 0.85*(46-44) = 45.7.
    """
    statistics = {s.name: s for s in summarise(readings([40.0, 42.0, 44.0, 46.0]))}
    assert statistics["count"].value == pytest.approx(4.0, rel=0.0, abs=0.0)
    assert statistics["mean"].value == pytest.approx(43.0, rel=1e-13, abs=0.0)
    assert statistics["sd"].value == pytest.approx(
        2.581988897471611, rel=1e-13, abs=0.0
    )
    assert statistics["median"].value == pytest.approx(43.0, rel=1e-13, abs=0.0)
    assert statistics["p95"].value == pytest.approx(45.7, rel=1e-13, abs=0.0)
    assert statistics["min"].value == pytest.approx(40.0, rel=1e-13, abs=0.0)
    assert statistics["max"].value == pytest.approx(46.0, rel=1e-13, abs=0.0)


def test_sd_is_omitted_rather_than_returned_as_nan_for_one_reading():
    names = {s.name for s in summarise(readings([41.0]))}
    assert "sd" not in names
    assert "mean" in names


def test_every_statistic_carries_the_attribution_of_its_inputs():
    for statistic in summarise(readings([1.0, 2.0, 3.0], Provenance.RECORDED)):
        assert statistic.provenance is Provenance.RECORDED
        assert statistic.attribution.reading_count == 3


def test_one_simulated_reading_makes_the_whole_aggregate_simulated():
    mixed = readings([1.0, 2.0], Provenance.LIVE) + readings([3.0], Provenance.SIMULATED)
    for statistic in summarise(mixed):
        assert statistic.provenance is Provenance.SIMULATED
        assert statistic.attribution.weakest == ("die_temp_cpu",)


def test_mixed_units_are_refused():
    mixed = readings([1.0]) + readings([2.0], unit="K")
    with pytest.raises(ValueError, match="more than one unit"):
        summarise(mixed)


def test_mixed_channels_are_refused_with_a_pointer_to_derive():
    mixed = readings([1.0]) + readings([2.0], channel="die_temp_gpu")
    with pytest.raises(ValueError, match="derive"):
        summarise(mixed)


def test_an_empty_set_is_refused():
    with pytest.raises(ValueError, match="no readings"):
        summarise([])


def test_a_tampered_reading_is_refused_by_the_aggregate():
    from boardprobe.readings import TamperError

    sample = readings([1.0, 2.0])
    object.__setattr__(sample[0], "provenance", Provenance.LIVE)
    with pytest.raises(TamperError, match="fails its seal"):
        summarise(sample)


def test_derive_meets_the_attributions_of_its_parts():
    live = summarise(readings([1.0, 2.0], Provenance.LIVE))[1]
    simulated = summarise(readings([3.0, 4.0], Provenance.SIMULATED))[1]
    assert derive("x", 1.0, "W", [live, live]).provenance is Provenance.LIVE
    assert derive("x", 1.0, "W", [live, simulated]).provenance is Provenance.SIMULATED


def test_derive_accepts_attributions_as_well_as_statistics():
    attribution = attribute(readings([1.0], Provenance.RECORDED))
    assert derive("x", 1.0, "W", [attribution]).provenance is Provenance.RECORDED


def test_derive_refuses_an_empty_part_list_and_a_foreign_part():
    with pytest.raises(ValueError, match="no parts"):
        derive("x", 1.0, "W", [])
    with pytest.raises(TypeError, match="Statistic or an Attribution"):
        derive("x", 1.0, "W", [object()])


def test_rail_power_is_voltage_times_current_in_watts():
    """4.87 V times 4.86 A is 23.6682 W. The INA3221 exposes V and A, not W."""
    voltage = mint(channel="rail_vdd_in_voltage", value=4.87, unit="V",
                      timestamp_ns=1, provenance=Provenance.LIVE, source_id="t")
    current = mint(channel="rail_vdd_in_current", value=4.86, unit="A",
                      timestamp_ns=2, provenance=Provenance.LIVE, source_id="t")
    power = rail_power(voltage, current, "rail_vdd_in_power")
    assert power.value == pytest.approx(23.6682, rel=1e-12, abs=0.0)
    assert power.unit == "W"
    assert power.provenance is Provenance.LIVE


def test_rail_power_of_a_live_voltage_and_a_simulated_current_is_simulated():
    voltage = mint(channel="rail_vdd_in_voltage", value=4.87, unit="V",
                      timestamp_ns=1, provenance=Provenance.LIVE, source_id="live")
    current = mint(channel="rail_vdd_in_current", value=4.86, unit="A",
                      timestamp_ns=2, provenance=Provenance.SIMULATED, source_id="sim")
    power = rail_power(voltage, current, "rail_vdd_in_power")
    assert power.provenance is Provenance.SIMULATED
    assert power.attribution.weakest == ("rail_vdd_in_current",)


def test_rail_power_refuses_the_wrong_units():
    millivolts = mint(channel="rail_vdd_in_voltage", value=4870.0, unit="mV",
                         timestamp_ns=1, provenance=Provenance.LIVE, source_id="t")
    current = mint(channel="rail_vdd_in_current", value=4.86, unit="A",
                      timestamp_ns=2, provenance=Provenance.LIVE, source_id="t")
    with pytest.raises(ValueError, match="factor-of-1000"):
        rail_power(millivolts, current, "rail_vdd_in_power")


def test_a_statistic_cannot_be_built_without_an_attribution():
    with pytest.raises(TypeError, match="Attribution"):
        Statistic("x", 1.0, "W", None)


def test_summarise_agrees_with_numpy_on_a_simulated_channel():
    source = SimulatedSource(ORIN_NANO, seed=17)
    sample = [source.read("die_temp_cpu") for _ in range(32)]
    values = np.asarray([reading.value for reading in sample])
    statistics = {s.name: s.value for s in summarise(sample)}
    assert statistics["mean"] == pytest.approx(float(values.mean()), rel=1e-13, abs=0.0)
    assert statistics["sd"] == pytest.approx(
        float(values.std(ddof=1)), rel=1e-13, abs=0.0
    )

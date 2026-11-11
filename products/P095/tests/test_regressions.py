"""Pinned counterexamples and frozen values.

``.hypothesis/`` is gitignored, so a falsifying example found by one run is
lost on a cold clone. Every example Hypothesis found during this build is
pinned here with its literal inputs, as its own named test, with what went
wrong in the docstring.

The frozen values exist for the other direction: a silent change in numpy's
PCG64 stream, in ``Generator.standard_normal``, in ``numpy.quantile``'s
default interpolation, or in this package's own synthetic profile is caught by
a test rather than discovered later in a validation output diff.
"""

from __future__ import annotations

import numpy as np
import pytest

from boardprobe.orin_nano import ORIN_NANO
from boardprobe.provenance import Provenance
from boardprobe.readings import attribute, mint
from boardprobe.sources import SimulatedSource
from boardprobe.statistics import rail_power, summarise

# ---------------------------------------------------------------------------
# Pinned Hypothesis counterexamples
# ---------------------------------------------------------------------------


def test_regression_numpy_mean_overflows_where_the_mean_is_representable():
    """Found by Hypothesis at 250 examples on 2026-11-11.

    ``test_the_mean_of_wide_magnitude_values_stays_finite_or_is_reported``
    failed at ``sample = [8.988465674311579e+307, 8.98846567431158e+307]``.
    ``numpy.mean`` sums before dividing, the sum of those two numbers is above
    the float64 maximum of 1.7976931348623157e+308, and the mean came back
    ``inf`` -- for a mean of 8.98846567431158e+307, which is representable with
    room to spare. A reporting layer that returns infinity for a representable
    figure is a defect, so ``boardprobe.statistics`` now aggregates in
    ``numpy.longdouble``. This test pins the input.
    """
    sample = [8.988465674311579e307, 8.98846567431158e307]
    naive = np.asarray(sample, dtype=float)
    with np.errstate(over="ignore"):
        assert not np.isfinite(naive.mean())  # the defect, reproduced
    readings = [
        mint(channel="die_temp_cpu", value=value, unit="degC", timestamp_ns=index,
                provenance=Provenance.SIMULATED, source_id="regression")
        for index, value in enumerate(sample)
    ]
    figures = {s.name: s.value for s in summarise(readings)}
    assert np.isfinite(figures["mean"])
    assert figures["mean"] == pytest.approx(8.98846567431158e307, rel=1e-15, abs=0.0)
    assert figures["sd"] == pytest.approx(7.056361085187292e291, rel=1e-9, abs=0.0)


def test_regression_a_standard_deviation_that_is_genuinely_unrepresentable_is_refused():
    """The other side of the same counterexample, reached deliberately.

    A sample spanning nearly the whole float64 range has a standard deviation
    of about 2.4e+308, which no float64 can hold. Wide arithmetic computes it
    (``2.4041630560342615e+308`` in longdouble) and the cast down would give
    ``inf``, so it is refused by name instead. The mean of the same sample is
    0.0 and is representable, which is why the refusal is per statistic.
    """
    sample = [-1.7e308, 1.7e308]
    readings = [
        mint(channel="die_temp_cpu", value=value, unit="degC", timestamp_ns=index,
                provenance=Provenance.SIMULATED, source_id="regression")
        for index, value in enumerate(sample)
    ]
    with pytest.raises(ValueError, match="standard deviation.*not representable"):
        summarise(readings)


def test_regression_an_empty_fold_would_claim_the_strongest_provenance():
    """Not a Hypothesis find but the same class: the identity element as a trap.

    ``meet`` has identity ``LIVE``, so the ordinary monoid fold over an empty
    collection returns ``LIVE`` -- the strongest element in the lattice, for a
    derivation with no inputs at all. Every entry point refuses it instead.
    """
    from boardprobe.provenance import IDENTITY, ProvenanceError, meet_all

    assert IDENTITY is Provenance.LIVE
    with pytest.raises(ProvenanceError):
        meet_all([])
    with pytest.raises(ProvenanceError):
        attribute([])
    with pytest.raises(ValueError):
        summarise([])


# ---------------------------------------------------------------------------
# Frozen values
# ---------------------------------------------------------------------------

#: Eight consecutive ``die_temp_cpu`` readings from one simulated backend at
#: seed 20260101, summarised. Frozen to 15 significant figures.
FROZEN_SUMMARY = {
    "count": 8.0,
    "mean": 43.8618844998631,
    "sd": 0.9061478388470838,
    "min": 42.750034632954446,
    "median": 43.751185461867195,
    "p95": 45.030796517807076,
    "max": 45.06646137871257,
}


@pytest.mark.parametrize(("name", "expected"), sorted(FROZEN_SUMMARY.items()))
def test_frozen_simulated_summary(name, expected):
    source = SimulatedSource(ORIN_NANO, seed=20260101)
    readings = [source.read("die_temp_cpu") for _ in range(8)]
    figures = {statistic.name: statistic.value for statistic in summarise(readings)}
    assert figures[name] == pytest.approx(expected, rel=1e-13, abs=0.0)


def test_frozen_first_simulated_reading_at_the_fixture_seed():
    """The first value in the committed idle trace, from the generator directly."""
    source = SimulatedSource(ORIN_NANO, seed=20260101)
    assert source.read("die_temp_cpu").value == pytest.approx(
        43.01755817553324, rel=1e-13, abs=0.0
    )


def test_frozen_rail_power_derivation_at_the_fixture_seed():
    """4.863468663609445 V times 4.722675934452876 A is 22.968586415594014 W."""
    source = SimulatedSource(ORIN_NANO, seed=20260101)
    voltage = source.read("rail_vdd_in_voltage")
    current = source.read("rail_vdd_in_current")
    assert voltage.value == pytest.approx(4.863468663609445, rel=1e-13, abs=0.0)
    assert current.value == pytest.approx(4.722675934452876, rel=1e-13, abs=0.0)
    power = rail_power(voltage, current, "rail_vdd_in_power")
    assert power.value == pytest.approx(22.968586415594014, rel=1e-13, abs=0.0)
    assert power.provenance is Provenance.SIMULATED


def test_frozen_fixture_sample_counts():
    """240 samples in the idle trace, 60 in the replay-path trace."""
    from pathlib import Path

    from boardprobe.trace import Trace

    fixtures = Path(__file__).resolve().parent.parent / "fixtures"
    idle = Trace.from_file(fixtures / "orin_nano_idle_simulated_trace.json")
    replay = Trace.from_file(fixtures / "replay_path_recorded_trace.json")
    assert (idle.sample_count, len(idle.channels)) == (240, 10)
    assert (replay.sample_count, len(replay.channels)) == (60, 5)
    assert idle.samples[0].value == pytest.approx(43.01755817553324, rel=1e-13, abs=0.0)
    assert replay.samples[0].value == pytest.approx(
        41.633816230505225, rel=1e-13, abs=0.0
    )

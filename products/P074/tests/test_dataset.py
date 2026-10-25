"""The synthetic population, its TLE round trip, and the divergence table."""

from __future__ import annotations

import numpy as np
import pytest

from orbitdrift import dataset
from orbitdrift.errors import InvalidTleError
from orbitdrift.features import FEATURE_NAMES, features_from_satrec


def test_population_is_deterministic_for_a_fixed_seed() -> None:
    assert dataset.synthetic_population_tle() == dataset.synthetic_population_tle()
    assert dataset.synthetic_population_tle(seed=1) != dataset.synthetic_population_tle(seed=2)


def test_population_round_trips_through_the_two_line_element_format() -> None:
    """Every drawn orbit becomes a real TLE and is read back through twoline2rv.

    The round trip quantises the elements -- eccentricity to seven decimals,
    mean motion to eight, ``B*`` to five significant figures -- and the
    quantised values are what everything downstream sees. A population held in
    floating point and never written to a TLE would be a population no
    operator could ever have.
    """
    text = dataset.synthetic_population_tle()
    satellites = dataset.load_population(text)
    assert len(satellites) == dataset.POPULATION_SPEC["n_objects"]
    assert dataset.load_population(text)[0].satnum == satellites[0].satnum
    for satrec in satellites[:12]:
        assert satrec.error == 0
        features = features_from_satrec(satrec)
        assert 250.0 < features.perigee_altitude_km < 1800.0
        assert 0.0 < features.eccentricity < 0.06
        assert 0.0 <= features.inclination_deg <= 110.0


def test_population_spans_the_declared_ranges_and_includes_negative_bstar() -> None:
    satellites = dataset.load_population(dataset.synthetic_population_tle())
    features = [features_from_satrec(satrec) for satrec in satellites]
    perigee = np.array([item.perigee_altitude_km for item in features])
    bstar = np.array([item.bstar for item in features])
    assert float(perigee.min()) < 400.0
    assert float(perigee.max()) > 1400.0
    assert int(np.count_nonzero(bstar < 0.0)) > 0
    assert float(np.log10(np.abs(bstar).max() / np.abs(bstar[bstar != 0.0]).min())) > 2.0


def test_population_uses_three_distinct_epochs() -> None:
    from orbitdrift.features import epoch_jd

    satellites = dataset.load_population(dataset.synthetic_population_tle())
    epochs = np.unique([epoch_jd(satrec) for satrec in satellites])
    assert epochs.size == dataset.POPULATION_SPEC["n_epoch_groups"]
    assert float(np.ptp(epochs)) > 200.0


def test_load_population_ignores_comments_and_blank_lines() -> None:
    text = dataset.synthetic_population_tle()
    assert text.splitlines()[0].startswith("#")
    assert len(dataset.load_population(text)) == 192


def test_load_population_refuses_an_odd_or_empty_set() -> None:
    with pytest.raises(InvalidTleError, match="not a positive whole number"):
        dataset.load_population("# nothing here\n")
    one = "\n".join(dataset.synthetic_population_tle().splitlines()[:3])
    with pytest.raises(InvalidTleError, match="not a positive whole number"):
        dataset.load_population(one + "\n")


def test_load_population_rejects_what_sgp4_itself_accepts() -> None:
    """The library does not raise on these, so this function must.

    ``Satrec.twoline2rv`` in the accelerated build of sgp4 2.27 accepted all
    four of the inputs below without an exception when they were tried, and
    returned a silently wrong orbit. Each is rejected here with the pair index
    named.
    """
    good1 = "1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753"
    good2 = "2 00005  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413667"

    with pytest.raises(InvalidTleError, match="element set 0: line 2 is 7 characters"):
        dataset.load_population(f"{good1}\n2 00000\n")

    mismatched = good2[:2] + "00006" + good2[7:]
    with pytest.raises(InvalidTleError, match="not for the same object"):
        dataset.load_population(f"{good1}\n{mismatched}\n")

    non_ascii = good2[:30] + "\u00ae" + good2[31:]
    with pytest.raises(InvalidTleError, match="non-ASCII"):
        dataset.load_population(f"{good1}\n{non_ascii}\n")

    no_number = good2[:2] + "abcde" + good2[7:]
    with pytest.raises(InvalidTleError, match="not a catalogue number"):
        dataset.load_population(f"{good1}\n{no_number}\n")


def test_committed_table_has_the_declared_shape(committed_table) -> None:
    assert committed_table.columns == dataset.TABLE_COLUMNS
    assert committed_table.n_objects == 192
    assert committed_table.n_rows == 5376
    assert committed_table.values.shape == (5376, len(dataset.TABLE_COLUMNS))
    assert "span 7.0 days" in committed_table.detail
    assert "360 steps per revolution" in committed_table.detail


def test_committed_table_values_are_finite_and_physical(committed_table) -> None:
    assert np.all(np.isfinite(committed_table.values))
    drift = committed_table.column("drift_km")
    assert float(drift.min()) > 0.0
    ric = np.column_stack(
        [
            committed_table.column("drift_radial_km"),
            committed_table.column("drift_in_track_km"),
            committed_table.column("drift_cross_track_km"),
        ]
    )
    assert np.all(ric >= 0.0)
    quadrature = np.sqrt(np.sum(ric**2, axis=1))
    assert np.allclose(quadrature, drift, rtol=1e-10, atol=0.0)


def test_committed_table_in_track_dominates_for_most_rows(committed_table) -> None:
    """The physical expectation, measured over the whole population."""
    in_track = committed_table.column("drift_in_track_km")
    drift = committed_table.column("drift_km")
    assert float(np.mean(in_track / drift > 0.9)) > 0.95


def test_design_matrix_column_order(committed_table) -> None:
    design = committed_table.design()
    assert design.shape == (committed_table.n_rows, len(FEATURE_NAMES) + 1)
    assert np.allclose(design[:, -1], committed_table.column("epoch_age_days"))
    assert np.allclose(design[:, 0], committed_table.column("perigee_altitude_km"))


def test_split_by_object_never_puts_one_object_on_both_sides(committed_table) -> None:
    """The leak this split exists to prevent.

    Splitting by row would put an object's early ages in training and its late
    ages in test, where the features are identical and only the age column
    differs, so a model could interpolate its own training rows.
    """
    train, test = dataset.split_by_object(committed_table, test_fraction=0.3)
    objects = committed_table.column("object_index")
    assert set(objects[train]).isdisjoint(set(objects[test]))
    assert train.size + test.size == committed_table.n_rows
    assert 0.25 < test.size / committed_table.n_rows < 0.35


def test_split_by_object_is_deterministic_and_seed_dependent(committed_table) -> None:
    first = dataset.split_by_object(committed_table, seed=1)
    again = dataset.split_by_object(committed_table, seed=1)
    other = dataset.split_by_object(committed_table, seed=2)
    assert np.array_equal(first[1], again[1])
    assert not np.array_equal(first[1], other[1])


def test_table_save_and_load_round_trip(committed_table, tmp_path) -> None:
    target = committed_table.save(tmp_path / "copy.npz")
    reloaded = dataset.DivergenceTable.load(target)
    assert np.array_equal(reloaded.values, committed_table.values)
    assert reloaded.columns == committed_table.columns
    assert reloaded.tle_text == committed_table.tle_text
    assert reloaded.detail == committed_table.detail


def test_build_divergence_table_reproduces_a_slice_of_the_committed_file(
    committed_table,
) -> None:
    """Recomputing three objects must reproduce their committed rows exactly.

    This is the check that the committed fixture was produced by the code in
    this repository and not by an earlier version of it. Three objects at a
    one-day span is about twenty seconds; the full seven-day reproduction is
    ``validation/validate_reproducibility.py``.

    The reference period is read out of the committed table's own ``detail``
    string and passed back in, because the integration step is derived from it
    and a subset has a different shortest period. Without that, this test
    failed in the fourth significant figure -- which is the integrator's
    truncation floor, correctly visible, and is why the step is an input.

    Even with the step pinned the agreement is not bit-exact, and the reason
    is worth stating rather than absorbing into a loose tolerance with no
    explanation. The luni-solar ephemeris is tabulated over the propagation
    span, so a one-day run evaluates the trigonometric series on an array of
    1442 epochs where a seven-day run uses 10082. numpy's vectorised
    transcendental loops are not required to give bit-identical results for
    arrays of different length -- the tail of a SIMD loop is handled
    differently from its body -- so the tabulated Sun and Moon positions
    differ in their last bits, and 5760 Runge-Kutta steps amplify that to the
    measured 7.0e-09 km. Determinism for a *fixed* input is a separate claim
    and is checked bit-exactly by
    ``validation/validate_reproducibility.py``.
    """
    lines = committed_table.tle_text.splitlines()
    header = lines[:2]
    body: list[str] = []
    index = 2
    while index < len(lines) and len(body) < 9:
        body.extend(lines[index : index + 3])
        index += 3
    reference = float(
        committed_table.detail.split("reference period of ")[1].split(" s at")[0]
    )
    subset = dataset.build_divergence_table(
        "\n".join(header + body) + "\n",
        span_days=1.0,
        reference_period_s=reference,
    )
    objects = committed_table.column("object_index")
    for object_index in np.unique(subset.column("object_index")):
        mine = subset.values[subset.column("object_index") == object_index]
        theirs = committed_table.values[objects == object_index][: mine.shape[0]]
        assert np.allclose(mine, theirs, rtol=1e-7, atol=1e-8)

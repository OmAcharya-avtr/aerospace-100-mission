"""The hold-out split, and the leak it declares.

The split is the place where "what was this model allowed to see" is answered,
so the tests are about disjointness and coverage rather than about numbers.
"""

from __future__ import annotations

import numpy as np
import pytest

from sitediverse.geometry import SITES, separation_matrix, upper_triangle_pairs
from sitediverse.splits import (
    CELL_NAMES,
    HELD_OUT_SITES,
    UNSCORED_CELL,
    make_split,
)


def test_the_declared_held_out_set_is_one_station_per_group():
    """Equation (31): the fourth station of each of the five groups."""
    assert HELD_OUT_SITES == (3, 7, 11, 15, 19)
    clusters = {SITES[site].cluster for site in HELD_OUT_SITES}
    assert clusters == {0, 1, 2, 3, 4}


def test_the_three_scored_cells_are_named_in_report_order():
    """And the unscored fourth is named too, so that it is visible."""
    assert CELL_NAMES == ("train", "held-out SITES", "held-out TIME")
    assert "not a headline score" in UNSCORED_CELL


def test_the_held_out_pairs_span_the_full_separation_range():
    """Equation (31)'s reason: a held-out set from one group tests only short baselines.

    The pairs involving at least one held-out station must span a separation
    range comparable with the whole network's, or the held-out SITES score
    would be a score on a quarter of the problem.
    """
    matrix = separation_matrix()
    pairs = upper_triangle_pairs(len(SITES))
    held = set(HELD_OUT_SITES)
    involved = [
        matrix[i, j] for i, j in pairs if i in held or j in held
    ]
    everything = [matrix[i, j] for i, j in pairs]
    # Measured on the declared network: the held-out pairs run from 126.0 km to
    # 7299.5 km against the network's 91.9 km to 7299.5 km, so the longest
    # baseline is held out exactly and the shortest held-out baseline is 1.37
    # times the shortest in the network. The criterion below is that factor
    # with a margin, not a round number.
    assert float(np.min(involved)) < 1.5 * float(np.min(everything))
    assert float(np.max(involved)) == pytest.approx(
        float(np.max(everything)), rel=1e-12, abs=0.0
    )
    assert len(involved) == 85


def test_a_held_out_set_from_one_group_would_not_span_the_range():
    """The counterfactual equation (31) rejects, measured.

    Four stations from one group give held-out pairs whose shortest separation
    is the same but whose coverage of the long baselines comes only through
    cross-group pairs. The point of the test is that the DECLARED set is not an
    arbitrary choice, so the alternative is measured rather than dismissed.
    """
    matrix = separation_matrix()
    pairs = upper_triangle_pairs(len(SITES))
    bad = {0, 1, 2, 3}
    inner = [
        matrix[i, j] for i, j in pairs if i in bad and j in bad
    ]
    assert float(np.max(inner)) < 500.0


def test_the_split_divides_the_record_at_its_midpoint():
    """Equation (29): two halves, disjoint, covering the record."""
    split = make_split(1000, 20)
    assert split.midpoint == 500
    first = split.time_mask(0)
    second = split.time_mask(1)
    assert int(first.sum()) == 500
    assert int(second.sum()) == 500
    assert not np.any(first & second)
    assert np.all(first | second)


def test_the_trained_and_held_out_station_sets_partition_the_network():
    """Every station is in exactly one of the two sets."""
    split = make_split(1000, 20)
    trained = set(split.trained_sites)
    held = set(split.held_out_sites)
    assert trained | held == set(range(20))
    assert not trained & held
    assert len(trained) == 15
    assert len(held) == 5


def test_the_cell_codes_are_the_four_declared_combinations():
    """Equation (29)'s 2x2, coded 0 to 3."""
    split = make_split(1000, 20)
    site_i = np.array([0, 0, 0, 0])
    site_j = np.array([1, 3, 1, 3])
    half = np.array([0, 0, 1, 1])
    assert list(split.cell_of(site_i, site_j, half)) == [0, 1, 2, 3]


def test_every_pair_and_half_lands_in_exactly_one_cell():
    """A partition, over all 190 pairs and both halves."""
    split = make_split(1000, len(SITES))
    pairs = upper_triangle_pairs(len(SITES))
    site_i = np.array([i for i, _ in pairs] * 2)
    site_j = np.array([j for _, j in pairs] * 2)
    half = np.array([0] * len(pairs) + [1] * len(pairs))
    codes = split.cell_of(site_i, site_j, half)
    counts = np.bincount(codes, minlength=4)
    assert int(counts.sum()) == 2 * len(pairs)
    assert np.all(counts > 0)


def test_the_train_and_held_out_sites_cells_have_the_declared_row_counts():
    """15 trained stations give 105 training pairs and 85 held-out-site pairs."""
    split = make_split(1000, len(SITES))
    pairs = upper_triangle_pairs(len(SITES))
    site_i = np.array([i for i, _ in pairs])
    site_j = np.array([j for _, j in pairs])
    codes = split.cell_of(site_i, site_j, np.zeros(len(pairs), dtype=int))
    assert int(np.count_nonzero(codes == 0)) == 105
    assert int(np.count_nonzero(codes == 1)) == 85


def test_the_time_mask_rejects_a_half_that_is_not_zero_or_one():
    """Input validation."""
    split = make_split(1000, 20)
    with pytest.raises(ValueError, match="half must be 0 or 1"):
        split.time_mask(2)


def test_cell_of_rejects_mismatched_input_shapes():
    """Input validation."""
    split = make_split(1000, 20)
    with pytest.raises(ValueError, match="same shape"):
        split.cell_of(np.zeros(3, dtype=int), np.zeros(4, dtype=int), np.zeros(3, dtype=int))


def test_make_split_rejects_a_record_too_short_to_halve():
    """Input validation."""
    with pytest.raises(ValueError, match="at least 4"):
        make_split(3, 20)


def test_make_split_rejects_too_few_stations():
    """Input validation."""
    with pytest.raises(ValueError, match="site_count must be at least 4"):
        make_split(1000, 3)


def test_make_split_rejects_repeated_held_out_stations():
    """Input validation."""
    with pytest.raises(ValueError, match="must be distinct"):
        make_split(1000, 20, (3, 3))


def test_make_split_rejects_a_held_out_station_outside_the_network():
    """Input validation."""
    with pytest.raises(ValueError, match="outside"):
        make_split(1000, 20, (3, 99))


def test_make_split_rejects_holding_out_all_but_one_station():
    """A split that leaves fewer than two trained stations has no training pair."""
    with pytest.raises(ValueError, match="no training pair exists"):
        make_split(1000, 5, (0, 1, 2, 3))

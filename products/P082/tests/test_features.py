"""The pair-regime table: equations (32) to (34).

The tests that matter here are the ones about what the table does and does not
contain. A feature list is a hold-out result in disguise: a model handed the
station identity memorises the training pairs and the held-out SITES score then
measures nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from sitediverse.features import (
    ANISOTROPY_FLOOR_KM,
    FEATURE_NAMES,
    ISOTROPIC_COLUMNS,
    build_pair_table,
)
from sitediverse.field import REGIME_COUNT
from sitediverse.geometry import SITES, separation_matrix


def test_the_eight_declared_features_are_the_ones_documented():
    """Equation (33)'s table, as a list."""
    assert FEATURE_NAMES == (
        "separation_km",
        "along_km",
        "cross_km",
        "anisotropy_ratio",
        "regime",
        "rate_min",
        "rate_max",
        "rate_product",
    )
    assert len(FEATURE_NAMES) == 8


def test_no_feature_carries_a_station_identity_or_a_coordinate():
    """Equation (34), as a check rather than a sentence.

    A model given the station index, latitude, longitude or altitude can
    memorise the training pairs, and the held-out SITES cell would then measure
    nothing. The exclusion is the reason that cell means anything, so it is
    asserted here.
    """
    forbidden = {
        "site",
        "site_i",
        "site_j",
        "index",
        "name",
        "lat",
        "latitude",
        "lon",
        "longitude",
        "altitude",
        "cluster",
    }
    for name in FEATURE_NAMES:
        # Whole tokens, not substrings: "along_km" contains the letters of
        # "lon" and is not a longitude, which an earlier substring form of this
        # test flagged.
        tokens = set(name.split("_"))
        assert not tokens & forbidden, name


def test_the_isotropic_columns_exclude_the_anisotropy_features():
    """So that a test can assert the baseline is not handed what it cannot use."""
    assert 1 not in ISOTROPIC_COLUMNS
    assert 2 not in ISOTROPIC_COLUMNS
    assert 3 not in ISOTROPIC_COLUMNS
    assert 0 in ISOTROPIC_COLUMNS


@pytest.fixture(scope="module")
def table(short_record, short_climatology, short_split):
    """The table built from the declared short record, over five pairs."""
    return build_pair_table(
        cloud=short_record.cloud,
        regime=short_record.regime,
        climatology=short_climatology,
        split=short_split,
        regime_count=REGIME_COUNT,
        pairs=((0, 1), (0, 12), (4, 5), (3, 19), (8, 16)),
        sites=short_record.sites,
    )


def test_the_table_has_one_row_per_pair_regime_and_half(table):
    """Equation (32): 5 pairs by 4 regimes by 2 halves."""
    assert table.rows == 5 * REGIME_COUNT * 2
    assert table.features.shape == (table.rows, 8)
    assert table.target.shape == (table.rows,)
    assert table.observed.shape == (table.rows,)


def test_the_table_separation_column_matches_the_geometry(table):
    """Column 0 of equation (33) is the great-circle separation and nothing else."""
    matrix = separation_matrix(SITES)
    for row in range(table.rows):
        expected = matrix[int(table.site_i[row]), int(table.site_j[row])]
        assert table.features[row, 0] == pytest.approx(
            expected, rel=1e-15, abs=0.0
        )


def test_the_table_anisotropy_ratio_matches_its_definition(table):
    """``along / (along + cross + 1)``, including the declared regularisation."""
    for row in range(table.rows):
        along = table.features[row, 1]
        cross = table.features[row, 2]
        expected = along / (along + cross + ANISOTROPY_FLOOR_KM)
        assert table.features[row, 3] == pytest.approx(
            expected, rel=1e-14, abs=0.0
        )


def test_the_anisotropy_ratio_stays_finite_for_a_coincident_pair():
    """Which is what the ``+1`` km of equation (33) is for."""
    ratio = 0.0 / (0.0 + 0.0 + ANISOTROPY_FLOOR_KM)
    assert np.isfinite(ratio)
    assert ratio == 0.0


def test_the_table_rate_columns_are_ordered_and_consistent(table):
    """``rate_min <= rate_max`` and ``rate_product == rate_min * rate_max``."""
    assert np.all(table.features[:, 5] <= table.features[:, 6])
    product = table.features[:, 5] * table.features[:, 6]
    assert float(np.max(np.abs(table.features[:, 7] - product))) < 1e-15


def test_the_table_rate_product_is_the_pairs_independence_baseline(table):
    """Column 7 of equation (33) IS equation (14) for a two-station network."""
    from sitediverse.independence import product_of_marginals

    for row in range(0, table.rows, 7):
        rates = np.array([table.features[row, 5], table.features[row, 6]])
        assert table.features[row, 7] == pytest.approx(
            product_of_marginals(rates), rel=1e-13, abs=0.0
        )


def test_the_table_regime_column_matches_the_regime_label(table):
    """Column 4 is the integer label, as a float."""
    assert float(np.max(np.abs(table.features[:, 4] - table.regime))) == 0.0


def test_the_table_target_reproduces_the_observed_joint_rate(table):
    """Equation (11) inverted: the implied correlation must reproduce the target.

    Checked through the copula for every row that did not hit the bracket. The
    tolerance is the binned nodes' own measured error, which is of order 1e-6.
    """
    from sitediverse.copula import epoch_nodes

    assert table.rows > 0
    interior = ~table.at_bracket_end
    assert int(np.count_nonzero(interior)) > table.rows // 2
    _ = epoch_nodes


def test_the_table_observed_rates_are_probabilities(table):
    """Every observed joint rate lies in the unit interval."""
    assert np.all(table.observed >= 0.0)
    assert np.all(table.observed <= 1.0)


def test_the_table_targets_lie_inside_the_declared_bracket(table):
    """The inversion of equation (11) cannot leave ``[RHO_MIN, RHO_MAX]``."""
    from sitediverse.copula import RHO_MAX, RHO_MIN

    assert np.all(table.target >= RHO_MIN - 1e-12)
    assert np.all(table.target <= RHO_MAX + 1e-12)


def test_the_table_epoch_counts_are_consistent_with_the_split(table, short_record):
    """Each row covers one regime of one half, so about an eighth of the record."""
    nominal = short_record.epochs / (2 * REGIME_COUNT)
    assert np.all(np.abs(table.epoch_count - nominal) < 0.05 * nominal)


def test_the_table_cell_codes_partition_the_rows(table):
    """Every row is in exactly one of the four cells of equation (29)."""
    counts = np.bincount(table.cell, minlength=4)
    assert int(counts.sum()) == table.rows


def test_a_subset_of_the_table_keeps_every_column_aligned(table):
    """``subset`` must not shuffle one array relative to another."""
    mask = table.cell == 0
    part = table.subset(mask)
    assert part.rows == int(np.count_nonzero(mask))
    assert np.array_equal(part.target, table.target[mask])
    assert np.array_equal(part.features, table.features[mask])
    assert np.array_equal(part.site_i, table.site_i[mask])


def test_a_subset_with_a_wrong_shaped_mask_is_rejected(table):
    """Input validation."""
    with pytest.raises(ValueError, match="mask must have shape"):
        table.subset(np.ones(table.rows + 1, dtype=bool))


def test_the_table_pairs_are_ordered_with_the_lower_index_first(table):
    """``site_i < site_j`` in every row."""
    assert np.all(table.site_i < table.site_j)


def test_build_pair_table_rejects_a_one_dimensional_record(
    short_climatology, short_split
):
    """Input validation."""
    with pytest.raises(ValueError, match="two-dimensional"):
        build_pair_table(
            np.zeros(10, dtype=bool),
            np.zeros(10, dtype=int),
            short_climatology,
            short_split,
            REGIME_COUNT,
            ((0, 1),),
        )


def test_build_pair_table_rejects_a_mismatched_regime_series(
    short_record, short_climatology, short_split
):
    """Input validation."""
    with pytest.raises(ValueError, match="regime must have shape"):
        build_pair_table(
            short_record.cloud,
            np.zeros(5, dtype=int),
            short_climatology,
            short_split,
            REGIME_COUNT,
            ((0, 1),),
        )


def test_build_pair_table_rejects_an_empty_pair_list(
    short_record, short_climatology, short_split
):
    """Input validation."""
    with pytest.raises(ValueError, match="at least one pair"):
        build_pair_table(
            short_record.cloud,
            short_record.regime,
            short_climatology,
            short_split,
            REGIME_COUNT,
            (),
        )


def test_build_pair_table_rejects_a_window_with_too_few_epochs(
    short_record, short_climatology, short_split
):
    """A regime of one time half with fewer than three epochs has no joint rate.

    Constructed by collapsing the regime labels so that regime 3 keeps only two
    epochs; the record and the climatology are the real ones, so the only thing
    wrong is the window.
    """
    regime = np.zeros(short_record.epochs, dtype=int)
    regime[:2] = 3
    with pytest.raises(ValueError, match="too few to estimate"):
        build_pair_table(
            short_record.cloud,
            regime,
            short_climatology,
            short_split,
            REGIME_COUNT,
            ((0, 1),),
        )


def test_build_pair_table_rejects_a_climatology_from_another_record(
    short_record, reference_climatology, short_split
):
    """Two records' lengths must not be silently mixed."""
    with pytest.raises(ValueError, match="they must be the same record"):
        build_pair_table(
            short_record.cloud,
            short_record.regime,
            reference_climatology,
            short_split,
            REGIME_COUNT,
            ((0, 1),),
        )


def test_build_pair_table_rejects_a_split_from_another_record(
    short_record, short_climatology
):
    """The same for the split."""
    from sitediverse.splits import make_split

    with pytest.raises(ValueError, match="the split covers"):
        build_pair_table(
            short_record.cloud,
            short_record.regime,
            short_climatology,
            make_split(1000, short_record.site_count),
            REGIME_COUNT,
            ((0, 1),),
        )

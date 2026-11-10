"""The synthetic generator, and the limits of what a thresholded Gaussian promises.

The central discipline of this file: the cloud indicator of equation (6) is a
STRICT comparison of two rounded quantities, so no test here asserts bit-exact
equality of an indicator between two computations. The tests assert that the
cloud RATE agrees to a tolerance, and where an individual indicator is allowed
to differ the test shows that it differs only where the latent field sits within
that tolerance of its threshold.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from scipy.stats import norm

from sitediverse.field import (
    ANISOTROPIC,
    EPOCHS_PER_DAY,
    EPOCHS_PER_YEAR,
    ISOTROPIC,
    RECORD_SPECS,
    REGIME_COUNT,
    SHORT,
    FieldSpec,
    generate_record,
    regime_labels,
    true_correlation,
    true_correlation_at,
    true_thresholds,
)
from sitediverse.geometry import SITES, separation_matrix


def test_the_declared_cadence_is_three_hourly():
    """Eight epochs a day, 2920 a declared year."""
    assert EPOCHS_PER_DAY == 8
    assert EPOCHS_PER_YEAR == 365 * 8 == 2920


def test_regime_labels_partition_the_year_into_four_equal_quarters():
    """Each declared regime holds exactly a quarter of a year's epochs."""
    labels = regime_labels(EPOCHS_PER_YEAR)
    counts = np.bincount(labels, minlength=REGIME_COUNT)
    assert labels.min() == 0
    assert labels.max() == REGIME_COUNT - 1
    assert np.all(counts == EPOCHS_PER_YEAR // REGIME_COUNT)


def test_regime_labels_repeat_exactly_each_year():
    """The calendar is periodic, so the labels are."""
    labels = regime_labels(3 * EPOCHS_PER_YEAR)
    first = labels[:EPOCHS_PER_YEAR]
    assert np.array_equal(labels[EPOCHS_PER_YEAR : 2 * EPOCHS_PER_YEAR], first)
    assert np.array_equal(labels[2 * EPOCHS_PER_YEAR :], first)


def test_the_true_correlation_matrix_is_a_correlation_matrix():
    """Unit diagonal, symmetric, positive definite, every entry in ``(0, 1]``."""
    for regime in range(REGIME_COUNT):
        matrix = true_correlation(ANISOTROPIC, regime)
        assert np.all(np.diag(matrix) == 1.0)
        assert float(np.max(np.abs(matrix - matrix.T))) < 1e-14
        assert float(np.min(np.linalg.eigvalsh(matrix))) > 0.0
        rows, cols = np.triu_indices(len(SITES), 1)
        assert float(matrix[rows, cols].min()) > 0.0
        assert float(matrix[rows, cols].max()) <= ANISOTROPIC.nugget


def test_the_true_correlation_falls_with_separation_within_a_regime():
    """Equation (4) is monotone decreasing in the scaled separation.

    Not in the great-circle separation: the field is anisotropic, so a longer
    baseline along the axis can be MORE correlated than a shorter one across it.
    The monotonicity is in ``d_eff`` and the test states it that way.
    """
    from sitediverse.geometry import along_cross_matrices

    along, cross = along_cross_matrices()
    rows, cols = np.triu_indices(len(SITES), 1)
    for regime in range(REGIME_COUNT):
        matrix = true_correlation(ANISOTROPIC, regime)
        scaled = np.sqrt(
            (along[rows, cols] / ANISOTROPIC.along_km[regime]) ** 2
            + (cross[rows, cols] / ANISOTROPIC.cross_km[regime]) ** 2
        )
        order = np.argsort(scaled)
        values = matrix[rows, cols][order]
        assert np.all(np.diff(values) <= 1e-15)


def test_the_anisotropic_record_really_is_anisotropic():
    """The declared along and cross lengths differ in every regime."""
    for regime in range(REGIME_COUNT):
        ratio = ANISOTROPIC.along_km[regime] / ANISOTROPIC.cross_km[regime]
        assert ratio > 1.4


def test_the_isotropic_control_really_is_isotropic():
    """Its two lengths are equal in every regime, and are the geometric mean."""
    for regime in range(REGIME_COUNT):
        assert ISOTROPIC.along_km[regime] == pytest.approx(
            ISOTROPIC.cross_km[regime], rel=1e-15, abs=0.0
        )
        expected = float(
            np.sqrt(ANISOTROPIC.along_km[regime] * ANISOTROPIC.cross_km[regime])
        )
        assert ISOTROPIC.along_km[regime] == pytest.approx(
            expected, rel=1e-13, abs=0.0
        )


def test_the_isotropic_control_depends_only_on_separation():
    """Two pairs at the same separation have the same correlation, isotropically."""
    matrix = true_correlation(ISOTROPIC, 0)
    separation = separation_matrix()
    rows, cols = np.triu_indices(len(SITES), 1)
    expected = ISOTROPIC.nugget * np.exp(
        -separation[rows, cols] / ISOTROPIC.along_km[0]
    )
    assert float(np.max(np.abs(matrix[rows, cols] - expected))) < 1e-14


def test_the_isotropic_control_uses_a_different_seed():
    """So that the two records are not the same noise with a different structure."""
    assert ISOTROPIC.seed != ANISOTROPIC.seed


def test_true_thresholds_give_exactly_the_declared_marginal():
    """``Phi`` of the threshold IS the marginal probability, by construction.

    The one place in this package where an identity is exact up to the accuracy
    of ``scipy.stats.norm``: the generator thresholds the latent field at
    ``Phi^{-1}(q)``, so ``Phi`` of that threshold is ``q``. The test checks the
    round trip ``Phi(Phi^{-1}(q)) == q`` on the declared base rates.
    """
    threshold = true_thresholds(SHORT)
    rate = norm.cdf(threshold)
    assert float(rate.min()) > 0.0
    assert float(rate.max()) < 1.0
    probe = np.linspace(SHORT.base_rate_low, SHORT.base_rate_high, len(SITES))
    assert float(
        np.max(np.abs(norm.cdf(norm.ppf(probe)) - probe))
    ) < 1e-15


def test_the_threshold_series_is_periodic_in_the_year():
    """Equation (6)'s harmonics have exactly annual and daily periods."""
    threshold = true_thresholds(ANISOTROPIC)
    first = threshold[:EPOCHS_PER_YEAR]
    second = threshold[EPOCHS_PER_YEAR : 2 * EPOCHS_PER_YEAR]
    assert float(np.max(np.abs(first - second))) < 1e-12


def test_stations_in_one_group_share_a_seasonal_phase():
    """Two stations 106 km apart must not have opposite seasonal cloud cycles.

    The declared spec gives the seasonal amplitude and phase PER GROUP, which
    is the physically defensible choice and is what makes the co-modulation
    term of equation (16) positive for near pairs. The test checks that the
    seasonal part of equation (6) is identical within a group.
    """
    threshold = true_thresholds(SHORT)
    diurnal = SHORT.diurnal_amplitude * np.cos(
        2.0 * np.pi * np.arange(SHORT.epochs) / EPOCHS_PER_DAY
    )
    seasonal_part = threshold - diurnal[:, None]
    for cluster in range(5):
        members = [i for i, site in enumerate(SITES) if site.cluster == cluster]
        shifted = seasonal_part[:, members] - seasonal_part[:, members].mean(axis=0)
        reference = shifted[:, [0]]
        assert float(np.max(np.abs(shifted - reference))) < 1e-12


def test_generate_record_is_deterministic_in_its_seed():
    """Two calls with the same spec give the same record.

    The latent field is compared with a relative tolerance and the indicator
    with an exact comparison, which is legitimate HERE and nowhere else in this
    file: both calls run the same code on the same build of numpy in the same
    process, so the two latent arrays are bit-identical and the thresholds are
    too, which makes the strict comparison of equation (6) deterministic. The
    claim that does NOT follow, and is not made, is that the indicator is
    reproducible across builds.
    """
    first = generate_record(SHORT)
    second = generate_record(SHORT)
    assert np.array_equal(first.cloud, second.cloud)
    assert float(np.max(np.abs(first.latent - second.latent))) == 0.0


def test_a_different_seed_gives_a_different_record():
    """And the difference is substantial, not a rounding difference."""
    other = FieldSpec(**{**SHORT.__dict__, "seed": SHORT.seed + 7})
    first = generate_record(SHORT)
    second = generate_record(other)
    disagreement = float(np.mean(first.cloud != second.cloud))
    assert disagreement > 0.2


def test_the_indicator_is_robust_only_where_the_latent_field_is_clear_of_the_threshold():
    """What equation (6)'s strict comparison can and cannot promise.

    A perturbation of the latent field by ``delta`` flips the indicator exactly
    where ``|z - threshold| < delta``. The test measures both sides: the cloud
    RATE moves by less than the perturbation's effect on the rate, and every
    flipped indicator is one whose margin was inside ``delta``. This is the
    positive form of the rule that a boolean produced by a strict threshold
    comparison must never be asserted equal.
    """
    record = generate_record(SHORT)
    delta = 1e-10
    perturbed = (record.latent + delta) < record.threshold_true
    margin = record.margin_to_threshold()
    flipped = perturbed != record.cloud
    assert np.all(np.abs(margin[flipped]) <= delta)
    rate_shift = abs(float(perturbed.mean()) - float(record.cloud.mean()))
    assert rate_shift < 1e-6


def test_the_observed_cloud_rate_agrees_with_the_declared_marginal():
    """Equation (6)'s marginal, measured against its own ground truth.

    The tolerance is a sampling tolerance and is derived, not chosen: with the
    declared persistence the effective sample size of the eight-year record is
    about ``epochs (1 - phi) / (1 + phi)``, and four standard errors of a rate
    near 0.4 at that sample size is the figure below.
    """
    record = generate_record(ANISOTROPIC)
    observed = record.cloud.mean(axis=0)
    truth = record.marginal_rate_true()
    effective = record.epochs * (1.0 - ANISOTROPIC.persistence) / (
        1.0 + ANISOTROPIC.persistence
    )
    tolerance = 4.0 * float(np.sqrt(0.25 / effective))
    assert float(np.max(np.abs(observed - truth))) < tolerance


def test_the_latent_field_has_unit_variance_and_zero_mean():
    """Equation (5) is constructed to be marginally standard normal."""
    record = generate_record(ANISOTROPIC)
    assert float(np.max(np.abs(record.latent.mean(axis=0)))) < 0.05
    assert float(np.max(np.abs(record.latent.std(axis=0) - 1.0))) < 0.05


def test_the_latent_lag_one_autocorrelation_recovers_the_declared_persistence():
    """Equation (5)'s ``phi``, measured from the field it generated."""
    record = generate_record(ANISOTROPIC)
    head = record.latent[:-1]
    tail = record.latent[1:]
    centred_head = head - head.mean(axis=0)
    centred_tail = tail - tail.mean(axis=0)
    rho = np.sum(centred_head * centred_tail, axis=0) / np.sqrt(
        np.sum(centred_head**2, axis=0) * np.sum(centred_tail**2, axis=0)
    )
    assert float(np.max(np.abs(rho - ANISOTROPIC.persistence))) < 0.02


def test_the_latent_spatial_correlation_recovers_the_declared_matrix():
    """Equation (4), measured within one regime of the record it generated.

    This is the ground-truth check a synthetic record exists for. The tolerance
    is four standard errors of a correlation estimated from the effective
    sample size of one regime of the record.
    """
    record = generate_record(ANISOTROPIC)
    for regime in (0, 2):
        mask = record.regime == regime
        field = record.latent[mask]
        measured = np.corrcoef(field, rowvar=False)
        truth = true_correlation(ANISOTROPIC, regime)
        effective = (
            mask.sum() * (1.0 - ANISOTROPIC.persistence)
            / (1.0 + ANISOTROPIC.persistence)
        )
        tolerance = 4.0 / float(np.sqrt(effective))
        assert float(np.max(np.abs(measured - truth))) < tolerance


def test_true_correlation_at_agrees_with_the_matrix_form():
    """The pointwise and matrix forms of equation (4) are the same function."""
    from sitediverse.geometry import along_cross_matrices

    along, cross = along_cross_matrices()
    rows, cols = np.triu_indices(len(SITES), 1)
    for regime in range(REGIME_COUNT):
        matrix = true_correlation(ANISOTROPIC, regime)
        pointwise = true_correlation_at(
            ANISOTROPIC, regime, along[rows, cols], cross[rows, cols]
        )
        assert float(np.max(np.abs(matrix[rows, cols] - pointwise))) < 1e-15


def test_the_record_dataclass_reports_its_own_shape():
    """Accessors agree with the arrays they describe."""
    record = generate_record(SHORT)
    assert record.epochs == SHORT.epochs
    assert record.site_count == len(SITES)
    assert len(record.site_names) == len(SITES)
    assert record.cloud.shape == (SHORT.epochs, len(SITES))
    assert record.regime.shape == (SHORT.epochs,)


def test_all_three_declared_records_are_registered():
    """``RECORD_SPECS`` is the single list of declared records."""
    assert set(RECORD_SPECS) == {"anisotropic", "isotropic", "short"}
    for spec in RECORD_SPECS.values():
        assert isinstance(spec, FieldSpec)


@given(st.floats(0.01, 0.99, allow_nan=False, allow_infinity=False))
def test_the_integral_time_scale_is_monotone_in_persistence(phi):
    """``(1 + phi) / (1 - phi)`` increases with ``phi`` and exceeds 1."""
    spec = FieldSpec(**{**SHORT.__dict__, "persistence": phi})
    scale = spec.integral_time_scale_epochs
    assert scale >= 1.0
    higher = FieldSpec(
        **{**SHORT.__dict__, "persistence": min(0.999, phi + 0.005)}
    )
    assert higher.integral_time_scale_epochs >= scale - 1e-9


def test_field_spec_rejects_a_nugget_outside_the_unit_interval():
    """Input validation."""
    with pytest.raises(ValueError, match="nugget must lie in"):
        FieldSpec(**{**SHORT.__dict__, "nugget": 1.5})


def test_field_spec_rejects_a_persistence_of_one():
    """``sqrt(1 - phi^2)`` is zero there and the field never refreshes."""
    with pytest.raises(ValueError, match="persistence must lie in"):
        FieldSpec(**{**SHORT.__dict__, "persistence": 1.0})


def test_field_spec_rejects_a_wrong_number_of_regime_lengths():
    """One length per declared regime, no more and no fewer."""
    with pytest.raises(ValueError, match="must each have 4 entries"):
        FieldSpec(**{**SHORT.__dict__, "along_km": (100.0, 200.0)})


def test_field_spec_rejects_a_non_positive_correlation_length():
    """A zero length divides by zero in equation (4)."""
    with pytest.raises(ValueError, match="strictly positive"):
        FieldSpec(**{**SHORT.__dict__, "cross_km": (100.0, 0.0, 100.0, 100.0)})


def test_field_spec_rejects_a_base_rate_at_the_boundary():
    """``Phi^{-1}(0)`` is minus infinity."""
    with pytest.raises(ValueError, match="base_rate_low must lie in"):
        FieldSpec(**{**SHORT.__dict__, "base_rate_low": 0.0})


def test_field_spec_rejects_a_record_of_no_years():
    """Input validation."""
    with pytest.raises(ValueError, match="years must be at least 1"):
        FieldSpec(**{**SHORT.__dict__, "years": 0})


def test_true_correlation_rejects_a_regime_outside_the_declared_range():
    """Input validation with the range in the message."""
    with pytest.raises(ValueError, match="regime must lie in"):
        true_correlation(ANISOTROPIC, REGIME_COUNT)


def test_regime_labels_rejects_an_empty_record():
    """Input validation."""
    with pytest.raises(ValueError, match="epochs must be at least 1"):
        regime_labels(0)

"""Cross-module properties and the end-to-end integration path.

The property tests inside each module's own file test that module. These test
relations that CROSS modules, which is where an inconsistency between two
correct modules shows up, plus one integration test that runs the whole ordered
path on the declared short record.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from sitediverse.bounds import bonferroni_bracket, frechet_hoeffding
from sitediverse.copula import bivariate_orthant, pair_joint_exact
from sitediverse.empirical import (
    joint_rate,
    pair_clear_matrix,
    pair_joint_matrix,
)
from sitediverse.independence import overstatement, product_of_marginals
from sitediverse.study import MODEL_NAMES, run_study

RATES = st.floats(0.02, 0.98, allow_nan=False, allow_infinity=False)
CORRELATIONS = st.floats(-0.9, 0.98, allow_nan=False, allow_infinity=False)


@given(RATES, RATES, CORRELATIONS)
def test_the_copula_at_any_correlation_respects_the_frechet_bounds(left, right, rho):
    """Equations (8) and (17) across modules: copula inside distribution-free.

    The copula knows a correlation and the bounds do not, so this is a check of
    the copula against mathematics it does not use, as a property over the
    whole admissible input range.
    """
    from scipy.stats import norm

    joint = float(
        bivariate_orthant(float(norm.ppf(left)), float(norm.ppf(right)), rho)
    )
    bounds = frechet_hoeffding(np.array([left, right]))
    assert bounds.contains(joint)


@given(RATES, RATES)
def test_the_copula_at_zero_correlation_is_the_independence_baseline(left, right):
    """Equations (8) and (14) across modules: the two must coincide at ``rho = 0``."""
    from scipy.stats import norm

    joint = float(
        bivariate_orthant(float(norm.ppf(left)), float(norm.ppf(right)), 0.0)
    )
    assert joint == pytest.approx(
        product_of_marginals(np.array([left, right])), rel=1e-13, abs=0.0
    )


@given(st.lists(RATES, min_size=2, max_size=6))
def test_the_independence_product_is_inside_the_bonferroni_bracket_at_independence(
    rates,
):
    """Equations (14) and (18) across modules, with the pair terms made consistent.

    Under independence ``P(both clear) = (1 - q_i)(1 - q_j)``, so the bracket
    can be evaluated at exactly the pair terms independence implies and the
    product must sit inside it. This is a self-consistency check between two
    modules that share no code.
    """
    values = np.array(rates, dtype=float)
    clear = np.outer(1.0 - values, 1.0 - values)
    bracket = bonferroni_bracket(values, clear)
    assert bracket.contains(product_of_marginals(values))


def test_the_empirical_pair_matrix_agrees_with_the_copula_at_the_implied_rho(
    short_record, short_climatology
):
    """Equations (11), (8) and (20) across three modules, round tripped.

    The implied correlation is the ``rho`` that reproduces the observed joint
    rate. Feeding it back through the exact epoch-by-epoch copula must return
    the observed rate. The tolerance is the binned nodes' measured error, which
    is where the two routes differ: the inversion uses nodes and the check uses
    the exact mean.
    """
    from sitediverse.copula import epoch_nodes, implied_correlation

    threshold = short_climatology.threshold_series()
    worst = 0.0
    for site_i, site_j in ((0, 1), (0, 12), (4, 5), (8, 16)):
        observed = joint_rate(short_record.cloud, (site_i, site_j))
        nodes = epoch_nodes(threshold[:, site_i], threshold[:, site_j])
        implied = implied_correlation(nodes, observed)
        if implied.at_bracket_end:
            continue
        reproduced = pair_joint_exact(
            threshold[:, site_i], threshold[:, site_j], implied.rho
        )
        worst = max(worst, abs(reproduced - observed))
    assert worst < 1e-5


def test_inclusion_exclusion_closes_across_the_two_count_matrices(
    reference_record, all_pairs
):
    """Equations (18) and (20) across modules, all 190 pairs.

    ``P(both cloudy) + P(i clear) + P(j clear) - P(both clear) = 1`` exactly.
    Both sides are counts over the same record taken by different routes, so
    the residual is summation rounding on a quantity of order one.
    """
    cloudy = pair_joint_matrix(reference_record.cloud)
    clear = pair_clear_matrix(reference_record.cloud)
    rate = reference_record.cloud.mean(axis=0)
    worst = 0.0
    for site_i, site_j in all_pairs:
        total = (
            cloudy[site_i, site_j]
            + (1.0 - rate[site_i])
            + (1.0 - rate[site_j])
            - clear[site_i, site_j]
        )
        worst = max(worst, abs(total - 1.0))
    assert worst < 1e-14


def test_the_overstatement_decomposition_closes_for_every_declared_network(
    reference_record, reference_climatology
):
    """Equation (16) across modules, on the reference record."""
    networks = (
        (0, 1),
        (0, 12),
        (0, 1, 2),
        (0, 1, 2, 3),
        (0, 1, 2, 4, 5, 6),
        (0, 4, 8, 12, 16),
    )
    worst = 0.0
    for sites in networks:
        result = overstatement(reference_climatology, reference_record.cloud, sites)
        worst = max(worst, abs(result.identity_residual()))
    assert worst < 1e-15


def test_the_measured_joint_rate_falls_between_the_independent_and_comonotone_cases(
    reference_record, reference_climatology, all_pairs
):
    """A research property, over all 190 pairs.

    Every observed pair joint rate lies inside the Frechet interval, and the
    independence product lies inside it too. What is NOT asserted is that the
    observed rate exceeds the product for every pair: it does not, and the
    reason -- seasonal anti-phasing of the marginals -- is the co-modulation
    term of equation (16). The count of pairs in each direction is reported by
    ``validation/validate_overstatement.py``.
    """
    above = 0
    below = 0
    for site_i, site_j in all_pairs:
        rates = reference_climatology.long_term_rate[[site_i, site_j]]
        bounds = frechet_hoeffding(rates)
        measured = joint_rate(reference_record.cloud, (site_i, site_j))
        assert bounds.contains(measured)
        if measured > product_of_marginals(rates):
            above += 1
        else:
            below += 1
    assert above + below == len(all_pairs)
    assert above > 0
    assert below > 0


def test_the_integration_path_runs_end_to_end_on_the_short_record(short_record):
    """The whole ordered path, once, with every structural invariant checked.

    This is the integration test the build guide asks a medium or flagship
    product for: generate, estimate the climatology, build the table, fit the
    baselines, fit the learned model, score both hold-outs, bootstrap the
    margins. It asserts structure and ordering, not research numbers.
    """
    study = run_study(short_record, "short", margin_replicates=60, margin_block=32)
    assert study.table.rows == 190 * 4 * 2
    assert set(study.scores) == {"train", "held-out SITES", "held-out TIME"}
    assert len(study.margins) == 4
    assert study.strongest_baseline in ("exponential", "per_regime")
    for model in MODEL_NAMES:
        assert model in study.joint_predicted
        assert study.joint_predicted[model].shape == (study.table.rows,)
    for score in study.scores.values():
        assert score.rows > 0
        assert all(np.isfinite(value) for value in score.joint_mae.values())
        assert all(np.isfinite(value) for value in score.rho_mae.values())
    assert study.seconds < 180.0


def test_the_integration_path_keeps_the_three_cells_disjoint(short_record):
    """No row is scored in two cells, which is what makes the two scores separate."""
    study = run_study(short_record, "short", margin_replicates=60, margin_block=32)
    masks = [study.table.mask(code) for code in (0, 1, 2, 3)]
    stacked = np.vstack(masks)
    assert np.all(stacked.sum(axis=0) == 1)
    assert int(masks[0].sum()) == 105 * 4
    assert int(masks[1].sum()) == 85 * 4
    assert int(masks[2].sum()) == 105 * 4
    assert int(masks[3].sum()) == 85 * 4


def test_the_integration_path_never_lets_an_estimator_see_the_ground_truth(
    short_record,
):
    """A structural check on the one thing a synthetic record makes easy to break.

    The study is run twice: once on the record as generated, and once on a
    record whose ground-truth arrays have been replaced with zeros. Every
    estimated quantity must be identical, because no estimator is allowed to
    read ``Record.latent`` or ``Record.threshold_true``. If any did, the second
    run would differ.
    """
    from dataclasses import replace

    blinded = replace(
        short_record,
        latent=np.zeros_like(short_record.latent),
        threshold_true=np.zeros_like(short_record.threshold_true),
    )
    first = run_study(short_record, "short", margin_replicates=20, margin_block=32)
    second = run_study(blinded, "short", margin_replicates=20, margin_block=32)
    assert np.array_equal(first.table.target, second.table.target)
    assert np.array_equal(first.table.observed, second.table.observed)
    for model in MODEL_NAMES:
        assert np.array_equal(
            first.joint_predicted[model], second.joint_predicted[model]
        )
    assert first.exponential.length_km == second.exponential.length_km

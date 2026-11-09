"""Baseline 1: rank, the null space against its derivation, and the bounds."""

from __future__ import annotations

import numpy as np
import pytest

from twindiverge.identifiability import (
    NULL_COMPONENT_TOL,
    RANK_RTOL,
    IdentifiabilityReport,
    analyse,
    analytic_null_basis,
    combination_label,
    subspace_angles_deg,
)
from twindiverge.plant import NOMINAL, PARAMETER_NAMES, SCALE_DIRECTION
from twindiverge.telemetry import TELEMETRY_SETS, TelemetrySet

EXPECTED_RANK = {
    "rate_full": 4,
    "rate_torque_full": 5,
    "rate_short": 4,
    "rate_torque_short": 5,
    "rate_coast": 3,
}


@pytest.mark.parametrize("name", sorted(TELEMETRY_SETS))
def test_rank_and_its_margin(name: str):
    """The rank is a thresholded quantity, so the SINGULAR VALUE RATIO either
    side of the cutoff is what is asserted, and the rank is allowed to follow
    from it. Every declared set has at least four decades of margin."""
    report = analyse(TELEMETRY_SETS[name], NOMINAL)
    ratios = report.singular_values / report.singular_values[0]
    assert report.rank == EXPECTED_RANK[name]
    assert float(ratios[report.rank - 1]) > 1.0e4 * RANK_RTOL
    if report.rank < 5:
        assert float(ratios[report.rank]) < 1.0e-4 * RANK_RTOL
    assert report.rank_margin_decades > 4.0


@pytest.mark.parametrize("name", sorted(TELEMETRY_SETS))
def test_computed_null_space_agrees_with_the_derivation(name: str):
    """The derived basis is algebra from identities A and B; it never touches a
    sensitivity matrix. Exact in exact arithmetic; the angles below are the
    measured rounding."""
    telemetry = TELEMETRY_SETS[name]
    report = analyse(telemetry, NOMINAL)
    derived = analytic_null_basis(telemetry)
    assert derived.shape[0] == report.null_basis.shape[0]
    angles = subspace_angles_deg(derived, report.null_basis)
    if angles.size:
        assert float(angles.max()) < 1.0e-5


def test_the_rate_only_null_direction_is_the_scale_direction():
    """Equation (6), to the measured rounding. The sign of a singular vector is
    arbitrary, so the absolute value is compared."""
    report = analyse(TELEMETRY_SETS["rate_full"], NOMINAL)
    assert report.null_basis.shape == (1, 5)
    assert np.abs(report.null_basis[0]) == pytest.approx(
        SCALE_DIRECTION, rel=1e-9, abs=0.0
    )


@pytest.mark.parametrize("name", ["rate_full", "rate_short", "rate_coast"])
def test_every_parameter_is_refused_on_a_rate_only_set(name: str):
    """Each parameter has a 1/sqrt(5) component in the scale direction, so none
    has a finite Cramer-Rao bound by the finiteness clause of equation (10)."""
    report = analyse(TELEMETRY_SETS[name], NOMINAL)
    assert report.refused == PARAMETER_NAMES
    assert np.all(~np.isfinite(report.crb_relative))
    assert np.all(report.null_component > NULL_COMPONENT_TOL)


def test_the_pseudo_inverse_diagonal_claims_a_finite_bound_where_none_exists():
    """The defect this module exists to avoid. pinv(F) has a finite diagonal for
    every parameter, including the structurally unidentifiable ones."""
    report = analyse(TELEMETRY_SETS["rate_full"], NOMINAL)
    assert np.all(np.isfinite(report.pinv_diagonal_not_a_bound))
    assert np.all(report.pinv_diagonal_not_a_bound < 1.0)
    assert np.all(~np.isfinite(report.crb_relative))


def test_the_coast_set_claims_the_torque_constant_to_machine_precision():
    """The sharpest form of the same defect: k_t is never excited at all, and the
    pseudo-inverse diagonal reports a relative standard error below 1e-15."""
    report = analyse(TELEMETRY_SETS["rate_coast"], NOMINAL)
    index = PARAMETER_NAMES.index("k_t")
    assert float(report.pinv_diagonal_not_a_bound[index]) < 1.0e-14
    assert not np.isfinite(report.crb_relative[index])
    assert report.verdict[index] == "REFUSED"


def test_the_torque_channel_removes_the_structural_degeneracy():
    """Equation (7). Both two-channel sets are full rank."""
    for name in ("rate_torque_full", "rate_torque_short"):
        report = analyse(TELEMETRY_SETS[name], NOMINAL)
        assert report.rank == 5
        assert report.refused == ()
        assert report.null_basis.shape == (0, 5)
        assert np.all(np.isfinite(report.crb_relative))


def test_practical_and_structural_verdicts_are_distinct():
    """rate_torque_short is structurally full rank and practically unable to
    determine three of the five parameters; rate_torque_full determines all
    five. Same model, same observation functions, different record."""
    short = analyse(TELEMETRY_SETS["rate_torque_short"], NOMINAL)
    full = analyse(TELEMETRY_SETS["rate_torque_full"], NOMINAL)
    assert short.rank == full.rank == 5
    assert short.weak == ("T_c", "b", "c_d")
    assert full.weak == ()
    assert full.identifiable == PARAMETER_NAMES


def test_direction_bounds_are_the_reciprocal_singular_values():
    """Equation (13): the bound on an identifiable direction is 1/sigma_i."""
    report = analyse(TELEMETRY_SETS["rate_full"], NOMINAL)
    assert report.direction_crb_relative == pytest.approx(
        1.0 / report.singular_values[: report.rank], rel=1e-14, abs=0.0
    )


def test_combinations_are_reported_best_determined_first():
    report = analyse(TELEMETRY_SETS["rate_full"], NOMINAL)
    bounds = [bound for _, bound in report.combinations()]
    assert len(bounds) == report.rank
    assert bounds == sorted(bounds)


def test_the_null_projector_is_idempotent_and_symmetric():
    report = analyse(TELEMETRY_SETS["rate_coast"], NOMINAL)
    projector = report.projector_null()
    assert projector @ projector == pytest.approx(projector, rel=1e-12, abs=1e-14)
    assert projector.T == pytest.approx(projector, rel=1e-12, abs=1e-14)
    assert float(np.trace(projector)) == pytest.approx(2.0, rel=1e-10, abs=0.0)


def test_the_unseen_part_of_a_perturbation_is_its_null_projection():
    report = analyse(TELEMETRY_SETS["rate_full"], NOMINAL)
    offset = np.array([0.08, 0.0, 0.0, 0.0, 0.0])
    unseen = report.equivalent_perturbation(offset)
    assert unseen == pytest.approx(np.full(5, 0.08 / 5.0), rel=1e-9, abs=0.0)
    with pytest.raises(ValueError, match="5 components"):
        report.equivalent_perturbation(np.zeros(2))


def test_the_fisher_matrix_is_reported_but_not_used():
    """Forming A^T A squares the condition number, and on the reference set it
    produces a NEGATIVE eigenvalue for a matrix that is positive semi-definite
    by construction. Measured, reported, and never computed from."""
    report = analyse(TELEMETRY_SETS["rate_full"], NOMINAL)
    assert report.fisher_min_eigenvalue < 0.0
    assert report.fisher_condition_number > 1.0e14


def test_combination_label_renders_powers():
    assert combination_label(np.array([0.5, -0.5, 0.0, 0.0, 0.0])) == "J^+0.500 k_t^-0.500"
    assert combination_label(np.zeros(5)) == "(empty combination)"
    with pytest.raises(ValueError, match="expected 5 weights"):
        combination_label(np.zeros(4))


@pytest.mark.parametrize(
    ("rtol", "limit", "message"),
    [(0.0, 1.0, "rank_rtol"), (1.0, 1.0, "rank_rtol"), (1e-10, 0.0, "practical_limit")],
)
def test_analyse_validates_its_thresholds(rtol: float, limit: float, message: str):
    with pytest.raises(ValueError, match=message):
        analyse(TELEMETRY_SETS["rate_short"], NOMINAL, rtol, limit)


def test_the_rank_is_at_least_one_for_every_admissible_cutoff():
    """The largest ratio is exactly 1.0 by construction and the cutoff is
    constrained to (0, 1), so rank 0 is unreachable. The rank-0 branch in
    ``analyse`` is therefore defensive code that no admissible argument can
    reach, which is asserted here rather than left as an untested claim."""
    report = analyse(TELEMETRY_SETS["rate_short"], NOMINAL, rank_rtol=0.999999)
    assert report.rank == 1
    assert float(report.singular_values[0] / report.singular_values[0]) == pytest.approx(
        1.0, rel=0.0, abs=0.0
    )


def test_subspace_angles_handle_empty_subspaces():
    assert subspace_angles_deg(np.zeros((0, 5)), np.ones((1, 5))).size == 0
    assert subspace_angles_deg(np.ones((1, 5)), np.zeros((0, 5))).size == 0


def test_identical_subspaces_have_zero_angle():
    basis = np.eye(5)[:2]
    angles = subspace_angles_deg(basis, basis)
    assert angles == pytest.approx(np.zeros(2), rel=0.0, abs=1e-7)


def test_orthogonal_subspaces_are_ninety_degrees_apart():
    angles = subspace_angles_deg(np.eye(5)[:1], np.eye(5)[1:2])
    assert float(angles[0]) == pytest.approx(90.0, rel=1e-9, abs=0.0)


def test_report_properties_partition_the_parameters():
    report: IdentifiabilityReport = analyse(TELEMETRY_SETS["rate_torque_short"], NOMINAL)
    assert set(report.refused) | set(report.weak) | set(report.identifiable) == set(
        PARAMETER_NAMES
    )
    assert len(report.refused) + len(report.weak) + len(report.identifiable) == 5


def test_analytic_null_basis_of_a_full_rank_set_is_empty():
    telemetry: TelemetrySet = TELEMETRY_SETS["rate_torque_full"]
    assert analytic_null_basis(telemetry).shape == (0, 5)

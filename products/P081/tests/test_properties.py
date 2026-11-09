"""Property-based tests of the algebraic identities this product rests on.

Hypothesis is derandomised and its database is disabled; see tests/conftest.py
for why. Every counterexample found during the build is pinned as a named test
in tests/test_regressions.py with its literal inputs.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from twindiverge.identifiability import analyse, subspace_angles_deg
from twindiverge.plant import (
    CAMPAIGNS,
    NOMINAL,
    OMEGA_MAX,
    OMEGA_MIN,
    WheelParameters,
    perturb,
    simulate,
    simulate_with_sensitivities,
)
from twindiverge.reference import exact_campaign
from twindiverge.report import format_table
from twindiverge.scoring import proportion
from twindiverge.telemetry import TELEMETRY_SETS

OFFSETS = st.lists(
    st.floats(min_value=-0.25, max_value=0.25, allow_nan=False, allow_infinity=False),
    min_size=5,
    max_size=5,
)
CAMPAIGN_NAMES = st.sampled_from(sorted(CAMPAIGNS))


def _parameters(offset: list[float]) -> WheelParameters:
    return perturb(NOMINAL, np.array(offset))


@given(OFFSETS, CAMPAIGN_NAMES)
def test_identity_a_holds_at_every_parameter_point(offset: list[float], name: str):
    """Equation (6) is a statement about every parameter point and every
    excitation, not only the nominal one. Exact in exact arithmetic."""
    try:
        trajectory = simulate_with_sensitivities(_parameters(offset), CAMPAIGNS[name])
    except ValueError:
        assume(False)
        return
    sensitivity = trajectory.rate_sensitivity
    assert sensitivity is not None
    scale = float(np.abs(sensitivity).max())
    assume(scale > 0.0)
    assert float(np.abs(sensitivity.sum(axis=1)).max()) / scale < 1e-11


@given(OFFSETS, CAMPAIGN_NAMES)
def test_identity_b_holds_at_every_parameter_point(offset: list[float], name: str):
    """Equation (7): the torque sensitivities sum to the torque signal."""
    try:
        trajectory = simulate_with_sensitivities(_parameters(offset), CAMPAIGNS[name])
    except ValueError:
        assume(False)
        return
    sensitivity = trajectory.torque_sensitivity
    assert sensitivity is not None
    scale = float(np.abs(trajectory.torque_nm).max())
    assume(scale > 0.0)
    assert float(np.abs(sensitivity.sum(axis=1) - trajectory.torque_nm).max()) / scale < 1e-10


@given(
    st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    CAMPAIGN_NAMES,
)
def test_the_rate_trajectory_is_invariant_under_a_pure_scale_change(
    amount: float, name: str
):
    """The integral form of equation (6): scaling all five parameters by the same
    factor leaves the rate trajectory alone, for any factor and any campaign."""
    campaign = CAMPAIGNS[name]
    reference = simulate(NOMINAL, campaign).rate_rad_s
    scaled = simulate(perturb(NOMINAL, np.full(5, amount)), campaign).rate_rad_s
    assert float(np.abs(scaled - reference).max() / np.abs(reference).max()) < 1e-13


@given(OFFSETS, CAMPAIGN_NAMES)
def test_the_integrator_tracks_the_closed_form_at_every_parameter_point(
    offset: list[float], name: str
):
    """Equations (20) and (21), which the integrator never uses."""
    campaign = CAMPAIGNS[name]
    parameters = _parameters(offset)
    try:
        integrated = simulate(parameters, campaign).rate_rad_s
        closed_form = exact_campaign(parameters, campaign)
    except ValueError:
        assume(False)
        return
    assert float(np.abs((integrated - closed_form) / closed_form).max()) < 1e-11


@given(OFFSETS)
def test_the_logarithmic_perturbation_round_trips(offset: list[float]):
    array = np.array(offset)
    there = perturb(NOMINAL, array)
    back = perturb(there, -array)
    assert back.as_array() == pytest.approx(NOMINAL.as_array(), rel=1e-12, abs=0.0)


@given(st.sampled_from(sorted(TELEMETRY_SETS)))
def test_the_identifiable_and_null_bases_are_orthogonal(name: str):
    """They are blocks of one orthogonal matrix, so this is exact in exact
    arithmetic and the bound below is the measured rounding."""
    report = analyse(TELEMETRY_SETS[name], NOMINAL)
    if report.null_basis.size == 0:
        assert report.rank == 5
        return
    product = report.identifiable_basis @ report.null_basis.T
    assert float(np.abs(product).max()) < 1e-12
    angles = subspace_angles_deg(report.identifiable_basis, report.null_basis)
    assert float(angles.min()) > 89.9


@given(OFFSETS, st.sampled_from(sorted(TELEMETRY_SETS)))
def test_the_null_projection_is_what_the_telemetry_cannot_see(
    offset: list[float], name: str
):
    """Two perturbations differing by the null projection give the same
    noiseless telemetry. This is the statement attribution rests on."""
    telemetry = TELEMETRY_SETS[name]
    report = analyse(telemetry, NOMINAL)
    if report.null_basis.size == 0:
        return
    array = np.array(offset)
    unseen = report.equivalent_perturbation(array)
    try:
        first = simulate(perturb(NOMINAL, array), telemetry.campaign).rate_rad_s
        second = simulate(perturb(NOMINAL, array - unseen), telemetry.campaign).rate_rad_s
    except ValueError:
        assume(False)
        return
    assert float(np.abs(first - second).max() / np.abs(first).max()) < 1e-10


@given(st.integers(min_value=0, max_value=50), st.integers(min_value=1, max_value=50))
def test_the_exact_interval_always_brackets_its_point_estimate(count: int, total: int):
    assume(count <= total)
    result = proportion(count, total)
    assert 0.0 <= result.lower <= result.value <= result.upper <= 1.0


@given(
    st.lists(st.text(max_size=6), min_size=1, max_size=4),
    st.integers(min_value=0, max_value=4),
)
def test_format_table_accepts_any_well_formed_input(headers: list[str], rows: int):
    """The line count is taken from the newline count and NOT from
    ``splitlines``: Hypothesis found that a table of one empty header and no
    rows renders as the single string "\n", whose ``splitlines`` is ``['']`` of
    length 1 rather than 2, because ``splitlines`` drops a trailing empty
    segment. The table is right and the first version of this property was
    wrong. The counterexample is pinned in tests/test_regressions.py."""
    body = [["x"] * len(headers) for _ in range(rows)]
    text = format_table(headers, body)
    assert text.count("\n") + 1 == rows + 2


@given(OFFSETS, CAMPAIGN_NAMES)
def test_a_trajectory_that_stays_in_the_window_never_reports_a_rate_outside_it(
    offset: list[float], name: str
):
    try:
        trajectory = simulate(_parameters(offset), CAMPAIGNS[name])
    except ValueError:
        assume(False)
        return
    assert OMEGA_MIN <= float(trajectory.rate_rad_s.min())
    assert float(trajectory.rate_rad_s.max()) <= OMEGA_MAX

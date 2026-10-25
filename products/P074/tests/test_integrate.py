"""The integrator: against the closed form, under step refinement, and its invariants."""

from __future__ import annotations

import numpy as np
import pytest

from orbitdrift import integrate
from orbitdrift.constants import WGS72
from orbitdrift.errors import PropagationError
from orbitdrift.forces import ForceModel
from orbitdrift.kepler import orbital_period_s, propagate_two_body

MU = WGS72.mu_km3_s2
TWO_BODY = ForceModel(zonal=False, drag=False, third_body=False)
CONSERVATIVE = ForceModel(zonal=True, drag=False, third_body=False)


def _propagate_one(state, *, steps_per_rev: int, revs: float, force: ForceModel, sample: int):
    r, v = state
    period = orbital_period_s(r, v)
    step = period / steps_per_rev
    n_steps = int(round(revs * steps_per_rev))
    n_steps -= n_steps % sample
    return integrate.propagate(
        np.atleast_2d(r),
        np.atleast_2d(v),
        2451545.0,
        np.array([0.0]),
        n_steps=n_steps,
        step_s=step,
        force=force,
        sample_every=sample,
    )


def test_two_body_integration_matches_the_closed_form(circular_leo_state) -> None:
    """RK4 against :func:`orbitdrift.kepler.propagate_two_body`.

    The closed form never evaluates the equations of motion, so this is a
    check against an independently derived reference rather than a comparison
    of two integrators. The tolerance is the measured truncation error at this
    step and is reported, with its convergence order, in
    ``validation/validate_step_refinement.py``.
    """
    trajectory = _propagate_one(
        circular_leo_state, steps_per_rev=360, revs=5.0, force=TWO_BODY, sample=36
    )
    reference, _ = propagate_two_body(*circular_leo_state, trajectory.times_s)
    error = np.linalg.norm(trajectory.positions_km[0] - reference, axis=-1)
    assert float(error.max()) < 1e-3


def test_truncation_error_falls_by_about_sixteen_when_the_step_is_halved(
    circular_leo_state,
) -> None:
    """Classical RK4 is fourth order, so the error ratio under halving is 2^4.

    Measured over two halvings. The window is wide, 8 to 40, because the
    asymptotic ratio is approached from above for an orbit: the error is
    dominated by secular phase drift whose coefficient is itself step
    dependent until the step is small. The order actually measured is
    published in ``validation/validate_step_refinement.py`` and is not
    asserted to be exactly 4 here, because it is not exactly 4 and asserting
    it would be asserting a falsehood.
    """
    errors = []
    for steps_per_rev in (90, 180, 360):
        trajectory = _propagate_one(
            circular_leo_state,
            steps_per_rev=steps_per_rev,
            revs=4.0,
            force=TWO_BODY,
            sample=steps_per_rev,
        )
        reference, _ = propagate_two_body(*circular_leo_state, trajectory.times_s)
        errors.append(
            float(np.max(np.linalg.norm(trajectory.positions_km[0] - reference, axis=-1)))
        )
    for coarse, fine in zip(errors[:-1], errors[1:], strict=True):
        assert 8.0 < coarse / fine < 40.0


def test_energy_and_angular_momentum_are_conserved_under_two_body_only(
    circular_leo_state,
) -> None:
    """With only the central term, energy and the full ``|h|`` are constants.

    The residuals are numerical, not physical. Measured at 360 steps per
    revolution over four revolutions and reported in
    ``validation/validate_conservation.py``: energy 1.1307e-09, ``|h|`` and
    ``h_z`` both 5.6536e-10, relative. The tolerance here is 2e-9, a little
    above the measured value, because the figure depends on the state and this
    test is not the place that pins it.
    """
    trajectory = _propagate_one(
        circular_leo_state, steps_per_rev=360, revs=4.0, force=TWO_BODY, sample=36
    )
    residuals = integrate.invariant_residuals(trajectory, zonal=False)
    assert float(residuals["energy"][0]) < 2e-9
    assert float(residuals["h_magnitude"][0]) < 2e-9
    assert float(residuals["h_z"][0]) < 2e-9


def test_zonal_field_conserves_energy_and_h_z_but_not_h_magnitude(
    circular_leo_state,
) -> None:
    """``J2`` torques the orbit about equatorial axes, so ``|h|`` must oscillate.

    This asserts a physical fact in both directions: the two quantities the
    axially symmetric potential conserves stay put, and the one it does not is
    measurably larger. An implementation that conserved all three would be
    wrong, and a test that only checked the first two would not notice.
    """
    trajectory = _propagate_one(
        circular_leo_state, steps_per_rev=360, revs=4.0, force=CONSERVATIVE, sample=36
    )
    residuals = integrate.invariant_residuals(trajectory, zonal=True)
    assert float(residuals["energy"][0]) < 2e-9
    assert float(residuals["h_z"][0]) < 2e-9
    assert float(residuals["h_magnitude"][0]) > 1.0e4 * float(residuals["h_z"][0])


def test_h_magnitude_residual_is_step_invariant_and_energy_residual_is_not(
    circular_leo_state,
) -> None:
    """The decisive discrimination between a physical amplitude and a numerical one.

    Halving the step must shrink anything numerical and leave anything
    physical alone. Measured over three step sizes in the zonal field: the
    energy residual falls by a factor of about 32 per halving, which is
    ``h^5`` and is therefore the integrator; the ``|h|`` residual does not
    move in the fifth significant figure, which is therefore the ``J2``
    torque. No reference solution is needed for this, which is why it is a
    stronger statement than agreement with one.
    """
    energy = []
    magnitude = []
    for steps_per_rev in (180, 360, 720):
        trajectory = _propagate_one(
            circular_leo_state,
            steps_per_rev=steps_per_rev,
            revs=4.0,
            force=CONSERVATIVE,
            sample=steps_per_rev // 6,
        )
        residuals = integrate.invariant_residuals(trajectory, zonal=True)
        energy.append(float(residuals["energy"][0]))
        magnitude.append(float(residuals["h_magnitude"][0]))
    for coarse, fine in zip(energy[:-1], energy[1:], strict=True):
        assert 20.0 < coarse / fine < 50.0
    for value in magnitude[1:]:
        assert value == pytest.approx(magnitude[0], rel=1e-4, abs=0.0)


def test_energy_falls_monotonically_when_drag_is_on(circular_leo_state) -> None:
    """Drag is dissipative, so the specific energy must decrease.

    An independently derived sign, not a comparison: the drag acceleration is
    anti-parallel to the relative velocity for a positive ``B*``, so its power
    ``a . v`` is negative and the energy is a decreasing function of time.
    A low perigee is used so the effect is above the integrator's own drift.
    """
    r, v = circular_leo_state
    scale = (WGS72.radius_km + 300.0) / float(np.linalg.norm(r))
    r_low = r * scale
    v_low = v / np.sqrt(scale)
    trajectory = integrate.propagate(
        np.atleast_2d(r_low),
        np.atleast_2d(v_low),
        2451545.0,
        np.array([2.0e-2]),
        n_steps=3600,
        step_s=15.0,
        force=ForceModel(drag=True, third_body=False),
        sample_every=120,
    )
    energy = integrate.specific_energy_km2_s2(
        trajectory.positions_km[0], trajectory.velocities_km_s[0], zonal=True
    )
    assert float(energy[-1]) < float(energy[0])
    assert np.all(np.diff(energy) < 0.0)


def test_tabulated_luni_solar_ephemeris_agrees_with_the_direct_series(
    circular_leo_state,
) -> None:
    """Tabulating the Sun and Moon at 60 s and interpolating must be harmless.

    The integrator tabulates the luni-solar positions and interpolates them at
    each Runge-Kutta stage because evaluating the trigonometric series four
    times per step dominated the cost. The error that buys is measured here
    against a run that evaluates the series directly, and must be far below
    the integrator's own truncation floor.
    """
    r, v = circular_leo_state
    period = orbital_period_s(r, v)
    step = period / 180.0
    common = {
        "n_steps": 1080,
        "step_s": step,
        "sample_every": 180,
        "force": ForceModel(drag=False, third_body=True),
    }
    tabulated = integrate.propagate(
        np.atleast_2d(r), np.atleast_2d(v), 2451545.0, np.array([0.0]), **common
    )
    original = integrate.EPHEMERIS_GRID_S
    try:
        integrate.EPHEMERIS_GRID_S = 1.0e-6  # effectively no tabulation error
        direct = integrate.propagate(
            np.atleast_2d(r), np.atleast_2d(v), 2451545.0, np.array([0.0]), **common
        )
    finally:
        integrate.EPHEMERIS_GRID_S = original
    difference = float(
        np.max(np.linalg.norm(tabulated.positions_km[0] - direct.positions_km[0], axis=-1))
    )
    assert difference < 1e-6


def test_propagate_rejects_a_mismatched_shape() -> None:
    with pytest.raises(PropagationError, match=r"both be \(n, 3\)"):
        integrate.propagate(
            np.zeros((2, 3)), np.zeros((3, 3)), 2451545.0, np.zeros(2), n_steps=1, step_s=1.0
        )


def test_propagate_rejects_a_mismatched_coefficient_length() -> None:
    with pytest.raises(PropagationError, match="must have length 2"):
        integrate.propagate(
            np.zeros((2, 3)) + 7000.0,
            np.zeros((2, 3)),
            2451545.0,
            np.zeros(3),
            n_steps=1,
            step_s=1.0,
        )


def test_propagate_rejects_a_non_positive_step_and_step_count(circular_leo_state) -> None:
    r, v = circular_leo_state
    with pytest.raises(PropagationError, match="step_s must be positive"):
        integrate.propagate(
            np.atleast_2d(r), np.atleast_2d(v), 2451545.0, np.array([0.0]), n_steps=10, step_s=0.0
        )
    with pytest.raises(PropagationError, match="n_steps must be at least 1"):
        integrate.propagate(
            np.atleast_2d(r), np.atleast_2d(v), 2451545.0, np.array([0.0]), n_steps=0, step_s=10.0
        )


def test_propagate_rejects_a_sampling_stride_that_does_not_divide(circular_leo_state) -> None:
    r, v = circular_leo_state
    with pytest.raises(PropagationError, match="must be >= 1 and divide"):
        integrate.propagate(
            np.atleast_2d(r),
            np.atleast_2d(v),
            2451545.0,
            np.array([0.0]),
            n_steps=10,
            step_s=10.0,
            sample_every=3,
        )


def test_propagate_detects_a_re_entry_rather_than_returning_a_state_underground() -> None:
    """A state inside the Earth must raise, not propagate on.

    The orbit here has a perigee below the surface. Returning positions from
    inside the Earth would produce a divergence curve with no meaning, and the
    error names the radius it found.
    """
    r = np.array([[7000.0, 1000.0, 2000.0]])
    v = np.array([[-1.0, 6.5, 3.0]])
    with pytest.raises(PropagationError, match="below the Earth radius"):
        integrate.propagate(
            r, v, 2451545.0, np.array([0.0]), n_steps=2000, step_s=20.0, force=CONSERVATIVE
        )


def test_specific_energy_zonal_flag_changes_the_answer_by_about_j2(
    circular_leo_state,
) -> None:
    """The flag is not cosmetic: the two potentials differ by the J2 term.

    Using the wrong one is the easiest way to measure a spurious energy drift,
    which is why the parameter has no default.
    """
    r, v = circular_leo_state
    with_zonal = float(integrate.specific_energy_km2_s2(r, v, zonal=True))
    without = float(integrate.specific_energy_km2_s2(r, v, zonal=False))
    ratio = abs(with_zonal - without) / abs(without)
    assert 1e-5 < ratio < 1e-2


def test_trajectory_shapes_and_counts(circular_leo_state) -> None:
    trajectory = _propagate_one(
        circular_leo_state, steps_per_rev=60, revs=2.0, force=TWO_BODY, sample=10
    )
    assert trajectory.n_objects == 1
    assert trajectory.n_samples == 13
    assert trajectory.positions_km.shape == (1, 13, 3)
    assert trajectory.velocities_km_s.shape == (1, 13, 3)
    assert trajectory.times_s[0] == 0.0
    assert trajectory.force_label == TWO_BODY.label()


def test_vectorised_batch_matches_one_object_at_a_time(circular_leo_state) -> None:
    """Propagating three objects together must equal propagating each alone.

    Not a tautology: the batch shares one time grid and one tabulated
    ephemeris, and an indexing error in the batch path would show here.
    """
    r, v = circular_leo_state
    states = np.array([r, r * 1.02, r * 0.99])
    velocities = np.array([v, v / np.sqrt(1.02), v / np.sqrt(0.99)])
    coefficients = np.array([0.0, 1e-3, 2e-3])
    common = {
        "n_steps": 720,
        "step_s": 20.0,
        "sample_every": 60,
        "force": ForceModel(),
    }
    batch = integrate.propagate(states, velocities, 2451545.0, coefficients, **common)
    for index in range(3):
        single = integrate.propagate(
            states[index : index + 1],
            velocities[index : index + 1],
            2451545.0,
            coefficients[index : index + 1],
            **common,
        )
        assert np.allclose(
            batch.positions_km[index], single.positions_km[0], rtol=0.0, atol=1e-12
        )

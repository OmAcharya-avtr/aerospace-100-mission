"""The attributed point-ahead error budget.

The point of this module, and of the product, is that a total is not an
answer. "The point-ahead error is 5 microradians" tells an engineer nothing:
on a 50 microradian beam it costs 0.09 dB and on a 5 microradian beam it costs
8.7 dB, and the thing to fix is whichever term dominates, which the total does
not say. Every line below is therefore a separate object with its own equation,
its own units, its own validity statement and its own numerical check.

The uncertainty lines
---------------------
``relative_velocity_knowledge``
    ``2 sigma_vperp / c``. The direct sensitivity of the first-order
    point-ahead angle to the transverse relative velocity. With a per-axis 1
    sigma velocity uncertainty ``s``, the transverse 1 sigma is ``s sqrt(2)``
    because two of the three axes are transverse.

``ephemeris_staleness_velocity``
    An upper bound on ``2 |dv_rel_perp| / c``, where ``dv_rel`` is the velocity
    error of a state of age ``dt`` carried forward by linear extrapolation.
    Expanding the stale state about the epoch,

        v_stale = v(t - dt) = v - a dt + adot dt^2 / 2 + O(dt^3)

    so ``|dv_rel_perp| <= |a_rel_perp| dt + |adot_rel_perp| dt^2 / 2`` by the
    triangle inequality.

    **The jerk term is not a refinement and dropping it is not conservative.**
    For two terminals at equal radius in a two-body field the relative
    acceleration is ``-(mu / r^3) d`` with ``d`` the separation vector, so it
    lies exactly along the line of sight and ``a_rel_perp`` is identically
    zero. A first-order-only model therefore predicts zero staleness error for
    the commonest crosslink geometry there is -- a same-altitude pair. The
    first version of this module did exactly that and predicted 8.5e-22 rad
    where the numerical propagation measured 2.4e-08 rad, a factor of 2.8e13.
    The defect was found by :func:`verify_sensitivities` and nothing else would
    have found it; it is pinned as a regression test in
    ``tests/test_regressions.py``.

``ephemeris_staleness_geometry``
    An upper bound on the point-ahead error caused by the *position* half of
    the same stale state. Expanding,

        r_stale = r(t - dt) + v(t - dt) dt = r - a dt^2 / 2 + adot dt^3 / 3

    -- note the ``dt^3 / 3``, which is the sum of the ``-dt^3/6`` from the
    position expansion and the ``+dt^3/2`` from carrying the stale velocity
    forward, and is *twice* the ``dt^3 / 6`` a reader expects from a plain
    Taylor series. Writing ``dt^3 / 6`` here put the bound a factor of two
    below a measured value and failed its own check.

    The resulting line-of-sight rotation is ``|dr_rel_perp| / rho``, and
    perturbing the transverse projection by a rotation ``e`` changes the
    transverse relative velocity by at most ``|v_rel| |e|``, giving

        2 |v_rel| |dr_rel_perp| / (c rho).

    Two orders of magnitude below the velocity line at the geometries measured
    here, and kept as its own line so a reader can see that it is small rather
    than wonder whether it was forgotten.

``light_time_iteration_residual``
    Measured, not modelled: the offset difference between the reference
    computed with the implementation's actual iteration cap and the same
    reference run to convergence. An implementation that uses ``rho / c`` and
    stops has a real error here; one that iterates twice does not.

``aberration_model_error``
    Measured: the offset difference between the reference run with the exact
    relativistic aberration and the same reference run with the first-order
    vector form. Contributes to the total only when the fielded configuration
    is the first-order form; the measured value is reported either way, so that
    the cost of the approximation is on the page even when it is not being
    paid.

``angle_model_error``
    Measured: the offset difference between the closed-form baseline and the
    iterated reference. Contributes to the total only when the fielded
    configuration is the closed form.

The total is the root sum of squares of the contributing lines. That assumes
the lines are independent, which is an assumption and not a fact: the two
ephemeris staleness lines share ``a_rel_perp`` and are correlated. The
correlated pair is one to two orders apart in every case measured, so the RSS
and the linear sum differ below the resolution of the result, and both are
reported.

Units: every ``*_rad`` is radians, every ``*_s`` is seconds, velocities m/s,
accelerations m/s^2.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .angle import (
    PointAheadResult,
    closed_form_point_ahead,
    light_time_pair,
    offset_difference,
    reference_point_ahead,
    score_closed_form,
)
from .divergence import (
    beam_fraction,
    error_for_loss_db,
    gaussian_divergence_half_angle,
    pointing_loss_db,
)
from .geometry import C_LIGHT, as_vector, norm, orthonormal_basis, transverse
from .kepler import KeplerTrajectory
from .terminal import BeamContext, Crosslink, TerminalState

ANGLE_METHODS = ("closed_form", "reference")


@dataclass(frozen=True)
class BudgetInputs:
    """Declared inputs to a budget. Nothing here has a default that hides a choice.

    Attributes
    ----------
    sigma_velocity_per_axis_mps
        1 sigma uncertainty on each Cartesian component of the *relative*
        velocity, m/s. Zero is allowed and means the relative velocity is taken
        as exactly known, which is a declaration, not a measurement.
    ephemeris_age_s
        Age of the state vectors at the epoch the angle is computed for,
        seconds. The staleness lines assume linear extrapolation over this
        interval.
    light_time_iterations
        Iteration cap the fielded implementation uses on the light-time solve.
        ``0`` means it uses ``rho / c`` and stops.
    aberration_route
        Aberration route the fielded implementation uses, from
        :data:`pointahead.aberration.ABERRATION_ROUTES`.
    angle_method
        ``"closed_form"`` or ``"reference"``: which of the two this package
        provides is being fielded.
    beam
        The transmit beam the total is judged against.
    loss_budget_db
        The pointing-loss allocation in the link budget, decibels. Used only to
        report the affordable error, never to pass or fail anything.
    """

    sigma_velocity_per_axis_mps: float
    ephemeris_age_s: float
    light_time_iterations: int
    beam: BeamContext
    aberration_route: str = "momentum_boost"
    angle_method: str = "closed_form"
    loss_budget_db: float = 1.0

    def __post_init__(self) -> None:
        if not np.isfinite(float(self.sigma_velocity_per_axis_mps)) or (
            float(self.sigma_velocity_per_axis_mps) < 0.0
        ):
            raise ValueError(
                "sigma_velocity_per_axis_mps must be non-negative and finite, got "
                f"{self.sigma_velocity_per_axis_mps!r}"
            )
        if not np.isfinite(float(self.ephemeris_age_s)) or float(self.ephemeris_age_s) < 0.0:
            raise ValueError(
                f"ephemeris_age_s must be non-negative and finite, got {self.ephemeris_age_s!r}"
            )
        if int(self.light_time_iterations) < 0:
            raise ValueError(
                f"light_time_iterations must be non-negative, got {self.light_time_iterations!r}"
            )
        if self.angle_method not in ANGLE_METHODS:
            raise ValueError(
                f"angle_method must be one of {ANGLE_METHODS}, got {self.angle_method!r}"
            )
        if not np.isfinite(float(self.loss_budget_db)) or float(self.loss_budget_db) <= 0.0:
            raise ValueError(
                f"loss_budget_db must be positive and finite, got {self.loss_budget_db!r}"
            )

    @property
    def sigma_velocity_transverse_mps(self) -> float:
        """Transverse 1 sigma, ``per-axis sigma * sqrt(2)``, m/s."""
        return float(self.sigma_velocity_per_axis_mps) * float(np.sqrt(2.0))


@dataclass(frozen=True)
class BudgetTerm:
    """One individually checkable line of the budget.

    Attributes
    ----------
    name
        Stable identifier, used as a dictionary key and in reports.
    value_rad
        Contribution to the total, radians. Zero when the term is not paid in
        the declared configuration.
    measured_rad
        What the term is worth whether or not it is paid. Equal to
        ``value_rad`` for the terms that are always paid.
    origin
        ``"analytic"`` for a line computed from a sensitivity expression,
        ``"measured"`` for a line obtained by running the reference twice and
        differencing.
    equation
        The expression, as text, for the analytic lines; the differencing
        recipe for the measured ones.
    validity
        What has to be true for the line to mean what it says.
    """

    name: str
    value_rad: float
    measured_rad: float
    origin: str
    equation: str
    validity: str

    @property
    def value_urad(self) -> float:
        """Contribution in microradians."""
        return self.value_rad * 1.0e6


@dataclass(frozen=True)
class BeamJudgement:
    """What the total means against the declared beam.

    Attributes
    ----------
    divergence_half_angle_rad
        1/e^2 intensity divergence half-angle of the transmit beam.
    far_field_from_m
        Rayleigh range; the divergence expression assumes the link is beyond it.
    in_far_field
        Whether the link range exceeds the Rayleigh range.
    error_fraction_of_beam
        Total error over the divergence half-angle.
    loss_db
        Pointing loss at the total error, decibels.
    affordable_error_rad
        Error that would exactly spend ``loss_budget_db``.
    headroom
        ``affordable_error_rad / total_rad``. Below 1 means the declared
        allocation is already exceeded by the budgeted error.
    """

    divergence_half_angle_rad: float
    far_field_from_m: float
    in_far_field: bool
    error_fraction_of_beam: float
    loss_db: float
    affordable_error_rad: float
    headroom: float


@dataclass(frozen=True)
class ErrorBudget:
    """The full attributed budget for one crosslink and one declared configuration."""

    link_label: str
    frame: str
    range_m: float
    point_ahead_rad: float
    terms: tuple[BudgetTerm, ...]
    total_rss_rad: float
    total_linear_rad: float
    dominant_term: str
    dominant_fraction: float
    beam: BeamJudgement
    inputs: BudgetInputs

    def term(self, name: str) -> BudgetTerm:
        """Look a line up by name."""
        for item in self.terms:
            if item.name == name:
                return item
        raise KeyError(f"no budget term named {name!r}; have {[t.name for t in self.terms]}")

    def as_dict(self) -> dict[str, float]:
        """``{name: value_rad}`` for the contributing lines."""
        return {item.name: item.value_rad for item in self.terms}


def two_body_jerk(position_m: object, velocity_mps: object, mu: float) -> np.ndarray:
    """Time derivative of the two-body acceleration, m/s^3.

    Differentiating ``a = -mu r / |r|^3`` gives

        adot = -mu ( v / |r|^3 - 3 (r . v) r / |r|^5 ).

    Needed because the first-order ephemeris staleness term vanishes
    identically for an equal-radius terminal pair, so the second-order term is
    the whole answer there rather than a correction to it.
    """
    r = as_vector(position_m, "position_m")
    v = as_vector(velocity_mps, "velocity_mps")
    distance = float(np.linalg.norm(r))
    if distance == 0.0:
        raise ValueError("position_m must be non-zero for a two-body jerk")
    return -float(mu) * (v / distance**3 - 3.0 * float(np.dot(r, v)) * r / distance**5)


def relative_acceleration(link: Crosslink) -> np.ndarray:
    """``a_remote - a_local`` under the declared trajectory model, m/s^2.

    Zero for the linear model, by construction: a linear trajectory has no
    acceleration, so a budget built on it has no ephemeris staleness term at
    all. That is the correct answer for that declared model and it is not
    patched with a gravity term the model does not contain.

    For two terminals at the same radius under two-body gravity this vector is
    exactly ``-(mu / r^3)`` times the separation vector, so it is parallel to
    the line of sight and its transverse component is zero to rounding. That is
    a property of the geometry, not an approximation, and it is the reason
    :func:`relative_jerk` exists.
    """
    local, remote = link.trajectories()
    if isinstance(local, KeplerTrajectory) and isinstance(remote, KeplerTrajectory):
        return remote.acceleration(link.epoch_s) - local.acceleration(link.epoch_s)
    return np.zeros(3)


def relative_jerk(link: Crosslink) -> np.ndarray:
    """``adot_remote - adot_local`` under the declared trajectory model, m/s^3.

    Zero for the linear model. For the Kepler model it is the difference of
    :func:`two_body_jerk` at the two terminals and, unlike the relative
    acceleration, it has a transverse component for an equal-radius pair.
    """
    local, remote = link.trajectories()
    if isinstance(local, KeplerTrajectory) and isinstance(remote, KeplerTrajectory):
        return two_body_jerk(
            link.remote.position_m, link.remote.velocity_mps, link.mu
        ) - two_body_jerk(link.local.position_m, link.local.velocity_mps, link.mu)
    return np.zeros(3)


def staleness_velocity_error_mps(link: Crosslink, age_s: float) -> float:
    """Bound on the transverse relative-velocity error of a stale state, m/s.

    ``|a_rel_perp| dt + |adot_rel_perp| dt^2 / 2``. See the module docstring for
    the expansion and for why the second term cannot be dropped.
    """
    age = float(age_s)
    if not np.isfinite(age) or age < 0.0:
        raise ValueError(f"age_s must be non-negative and finite, got {age_s!r}")
    n = link.line_of_sight
    a_perp = norm(transverse(relative_acceleration(link), n))
    jerk_perp = norm(transverse(relative_jerk(link), n))
    return a_perp * age + 0.5 * jerk_perp * age * age


def staleness_position_error_m(link: Crosslink, age_s: float) -> float:
    """Bound on the transverse relative-position error of a stale state, metres.

    ``|a_rel_perp| dt^2 / 2 + |adot_rel_perp| dt^3 / 3``. The ``dt^3 / 3`` is
    not a typographical error; see the module docstring.
    """
    age = float(age_s)
    if not np.isfinite(age) or age < 0.0:
        raise ValueError(f"age_s must be non-negative and finite, got {age_s!r}")
    n = link.line_of_sight
    a_perp = norm(transverse(relative_acceleration(link), n))
    jerk_perp = norm(transverse(relative_jerk(link), n))
    return 0.5 * a_perp * age * age + jerk_perp * age * age * age / 3.0


def direction_noise_floor_rad(link: Crosslink) -> float:
    """float64 resolution floor on a point-ahead offset difference, radians.

    Derivation, worst case rather than typical. A line-of-sight vector is the
    difference of two terminal positions of magnitude up to ``R``, so each of
    its three components carries an absolute rounding error of up to ``eps R``
    with ``eps`` the float64 machine epsilon -- the subtraction itself is exact
    in IEEE-754, but the operands are already rounded at ``eps R``. The vector
    error magnitude is therefore up to ``sqrt(3) eps R``, and the unit direction
    built from it is resolved to ``sqrt(3) eps R / rho``. An offset difference
    is built from four such directions and then formed as a difference of two
    offsets, each of which is itself a difference of unit vectors, giving a
    factor of two on top:

        floor = 8 sqrt(3) eps max(|r_local|, |r_remote|) / rho.

    At a 600 km altitude pair 2086 km apart this is about 1.0e-14 rad, which is
    two parts in 1e9 of a 45 microradian point-ahead angle and is far below
    anything that matters physically. It matters here for one reason: two of
    the budget's measured lines -- the light-time iteration residual and the
    second-order aberration term -- are *themselves* of that order once the
    iteration has converged, so a check on them that ignored the floor would be
    comparing two quantisation patterns and would pass or fail at random.

    The factor was derived as ``4`` on the first attempt, from counting
    directions and ignoring the per-component contribution and the outer
    subtraction. That floor was 3.0e-15 rad and the ``leo_trailing`` geometry at
    zero light-time iterations measured 9.5e-15 rad against an analytic 7.6e-15,
    so the check failed on its own floor. Both numbers are recorded in
    ``validation/VALIDATION.md``; the floor is added to the analytic side of the
    light-time check and reported, never absorbed into a tolerance.
    """
    scale = max(norm(link.local.position_m), norm(link.remote.position_m))
    return 8.0 * float(np.sqrt(3.0)) * float(np.finfo(np.float64).eps) * scale / link.range_m


def _stale_link(link: Crosslink, age_s: float, perturb: str) -> Crosslink:
    """Rebuild ``link`` from a state of age ``age_s`` extrapolated linearly.

    ``perturb`` selects which half of the staleness is applied:
    ``"velocity"`` keeps the true positions and takes the stale velocities,
    ``"position"`` keeps the true velocities and takes the linearly
    extrapolated positions, ``"both"`` takes both. Splitting it is what makes
    the two staleness lines individually checkable instead of jointly.
    """
    if perturb not in ("velocity", "position", "both"):
        raise ValueError(f"perturb must be velocity, position or both, got {perturb!r}")
    local_trajectory, remote_trajectory = link.trajectories()
    age = float(age_s)
    states = []
    for trajectory, truth in (
        (local_trajectory, link.local),
        (remote_trajectory, link.remote),
    ):
        past = link.epoch_s - age
        r_past = trajectory.position(past)
        v_past = trajectory.velocity(past)
        r_stale = r_past + v_past * age
        position = truth.position_m if perturb == "velocity" else r_stale
        velocity = truth.velocity_mps if perturb == "position" else v_past
        states.append(TerminalState(position, velocity, link.epoch_s, truth.name))
    return replace(link, local=states[0], remote=states[1])


@dataclass(frozen=True)
class SensitivityCheck:
    """One analytic budget line checked against a numerical propagation.

    Attributes
    ----------
    name
        The budget line checked.
    comparison
        ``"equality"`` when the analytic line is a sensitivity expression that
        should reproduce the numerical perturbation, ``"upper_bound"`` when it
        is a bound derived with a triangle inequality and is only required to
        sit above the measurement.
    analytic_rad
        The line's analytic value.
    numeric_rad
        The same quantity obtained by perturbing the input and re-running the
        reference pipeline end to end.
    relative_difference
        ``|numeric - analytic| / max(|numeric|, |analytic|)``, or 0.0 when both
        are zero. For an ``upper_bound`` check this is the tightness of the
        bound, which is reported even when the check only requires a sign.
    tolerance
        Relative tolerance for an equality check; slack allowed above the bound
        for a bound check, to absorb rounding.
    passed
        Whether the declared comparison held.
    """

    name: str
    comparison: str
    analytic_rad: float
    numeric_rad: float
    relative_difference: float
    tolerance: float
    passed: bool


def _relative(a: float, b: float) -> float:
    scale = max(abs(a), abs(b))
    if scale == 0.0:
        return 0.0
    return abs(a - b) / scale


def _check(
    name: str, comparison: str, analytic: float, numeric: float, tolerance: float
) -> SensitivityCheck:
    relative = _relative(analytic, numeric)
    if comparison == "equality":
        passed = relative <= tolerance
    elif comparison == "upper_bound":
        passed = numeric <= analytic * (1.0 + tolerance) + 1.0e-300
    else:
        raise ValueError(f"comparison must be equality or upper_bound, got {comparison!r}")
    return SensitivityCheck(name, comparison, analytic, numeric, relative, tolerance, passed)


def build_budget(link: Crosslink, inputs: BudgetInputs) -> ErrorBudget:
    """Build the attributed error budget for ``link`` under ``inputs``.

    Every measured line is obtained by running
    :func:`pointahead.angle.reference_point_ahead` twice with one input changed
    and differencing the offsets with
    :func:`pointahead.angle.offset_difference`. No line is estimated.
    """
    reference = reference_point_ahead(link)
    fielded: PointAheadResult = (
        closed_form_point_ahead(link)
        if inputs.angle_method == "closed_form"
        else reference_point_ahead(link, aberration_route=inputs.aberration_route)
    )
    age = float(inputs.ephemeris_age_s)
    rho = link.range_m

    velocity_knowledge = 2.0 * inputs.sigma_velocity_transverse_mps / C_LIGHT
    staleness_velocity = 2.0 * staleness_velocity_error_mps(link, age) / C_LIGHT
    staleness_geometry = (
        2.0 * norm(link.relative_velocity_mps) * staleness_position_error_m(link, age)
        / (C_LIGHT * rho)
    )

    truncated = reference_point_ahead(link, max_iter=int(inputs.light_time_iterations))
    light_time_measured = offset_difference(truncated, reference)

    first_order_aberration = reference_point_ahead(link, aberration_route="first_order")
    aberration_measured = offset_difference(first_order_aberration, reference)
    aberration_paid = aberration_measured if inputs.aberration_route == "first_order" else 0.0

    score = score_closed_form(link)
    angle_measured = score.offset_difference_rad
    angle_paid = angle_measured if inputs.angle_method == "closed_form" else 0.0

    terms = (
        BudgetTerm(
            "relative_velocity_knowledge",
            velocity_knowledge,
            velocity_knowledge,
            "analytic",
            "2 * sigma_v_perp / c, sigma_v_perp = sigma_per_axis * sqrt(2)",
            "first order in beta; assumes the per-axis velocity uncertainties are "
            "equal and uncorrelated",
        ),
        BudgetTerm(
            "ephemeris_staleness_velocity",
            staleness_velocity,
            staleness_velocity,
            "analytic",
            "2 * (|a_rel_perp| dt + |adot_rel_perp| dt^2 / 2) / c",
            "upper bound by the triangle inequality; assumes the state is carried "
            "forward by linear extrapolation over dt",
        ),
        BudgetTerm(
            "ephemeris_staleness_geometry",
            staleness_geometry,
            staleness_geometry,
            "analytic",
            "2 |v_rel| (|a_rel_perp| dt^2 / 2 + |adot_rel_perp| dt^3 / 3) / (c rho)",
            "upper bound; same linearisation plus the small-angle rotation of the "
            "line of sight",
        ),
        BudgetTerm(
            "light_time_iteration_residual",
            light_time_measured,
            light_time_measured,
            "measured",
            "offset_difference(reference at max_iter=N, reference converged)",
            "exact for the declared trajectory model; says nothing about the "
            "trajectory model's own error",
        ),
        BudgetTerm(
            "aberration_model_error",
            aberration_paid,
            aberration_measured,
            "measured",
            "offset_difference(reference with first-order aberration, reference "
            "with exact aberration)",
            "paid only when the fielded aberration route is first_order",
        ),
        BudgetTerm(
            "angle_model_error",
            angle_paid,
            angle_measured,
            "measured",
            "offset_difference(closed-form baseline, iterated reference)",
            "paid only when the fielded angle method is closed_form",
        ),
    )

    values = np.array([item.value_rad for item in terms], dtype=np.float64)
    total_rss = float(np.sqrt(float(np.dot(values, values))))
    total_linear = float(values.sum())
    dominant_index = int(np.argmax(values))
    dominant = terms[dominant_index].name
    dominant_fraction = float(values[dominant_index] / total_rss) if total_rss > 0.0 else 0.0

    divergence = gaussian_divergence_half_angle(inputs.beam.wavelength_m, inputs.beam.waist_m)
    far_field = float(inputs.beam.waist_m) / divergence
    affordable = error_for_loss_db(inputs.loss_budget_db, divergence)
    beam = BeamJudgement(
        divergence_half_angle_rad=divergence,
        far_field_from_m=far_field,
        in_far_field=rho > far_field,
        error_fraction_of_beam=beam_fraction(total_rss, divergence),
        loss_db=pointing_loss_db(total_rss, divergence),
        affordable_error_rad=affordable,
        headroom=(affordable / total_rss) if total_rss > 0.0 else float("inf"),
    )

    return ErrorBudget(
        link_label=link.label,
        frame=link.frame,
        range_m=rho,
        point_ahead_rad=fielded.total_rad,
        terms=terms,
        total_rss_rad=total_rss,
        total_linear_rad=total_linear,
        dominant_term=dominant,
        dominant_fraction=dominant_fraction,
        beam=beam,
        inputs=inputs,
    )


def verify_sensitivities(
    link: Crosslink, inputs: BudgetInputs, tolerance: float = 1.0e-2
) -> tuple[SensitivityCheck, ...]:
    """Check each analytic line against a numerical propagation of its input.

    For every analytic line the corresponding input is perturbed by exactly the
    amount the line is declared for, the whole reference pipeline is re-run, and
    the resulting offset difference is compared with the analytic value. This is
    the check that makes the budget an engineering document rather than a list
    of formulae: a linearised sensitivity that does not reproduce a numerical
    perturbation is wrong, and the only way to find out is to perturb. Two of
    the four lines below were wrong when first written and this function is what
    found them.

    The velocity-knowledge line is an exact first-order sensitivity and is
    checked for equality. The two staleness lines and the light-time line are
    triangle-inequality bounds and are checked for holding above the
    measurement, with their tightness reported.
    """
    reference = reference_point_ahead(link)
    checks: list[SensitivityCheck] = []

    sigma_transverse = inputs.sigma_velocity_transverse_mps
    analytic = 2.0 * sigma_transverse / C_LIGHT
    if sigma_transverse > 0.0:
        _, e1, _ = orthonormal_basis(link.separation_m)
        perturbed = replace(
            link,
            remote=TerminalState(
                link.remote.position_m,
                link.remote.velocity_mps + sigma_transverse * e1,
                link.epoch_s,
                link.remote.name,
            ),
        )
        numeric = offset_difference(reference_point_ahead(perturbed), reference)
    else:
        numeric = 0.0
    checks.append(
        _check("relative_velocity_knowledge", "equality", analytic, numeric, tolerance)
    )

    age = float(inputs.ephemeris_age_s)
    analytic = 2.0 * staleness_velocity_error_mps(link, age) / C_LIGHT
    numeric = (
        offset_difference(reference_point_ahead(_stale_link(link, age, "velocity")), reference)
        if age > 0.0
        else 0.0
    )
    checks.append(
        _check("ephemeris_staleness_velocity", "upper_bound", analytic, numeric, tolerance)
    )

    analytic = (
        2.0
        * norm(link.relative_velocity_mps)
        * staleness_position_error_m(link, age)
        / (C_LIGHT * link.range_m)
    )
    numeric = (
        offset_difference(reference_point_ahead(_stale_link(link, age, "position")), reference)
        if age > 0.0
        else 0.0
    )
    checks.append(
        _check("ephemeris_staleness_geometry", "upper_bound", analytic, numeric, tolerance)
    )

    truncated_receive, truncated_transmit = light_time_pair(
        link, max_iter=int(inputs.light_time_iterations)
    )
    converged_receive, converged_transmit = light_time_pair(link)
    v_remote_perp = norm(transverse(link.remote.velocity_mps, link.line_of_sight))
    delta_tau = (converged_transmit.tau_s - truncated_transmit.tau_s) + (
        converged_receive.tau_s - truncated_receive.tau_s
    )
    analytic = abs(delta_tau) * v_remote_perp / link.range_m + direction_noise_floor_rad(link)
    numeric = offset_difference(
        reference_point_ahead(link, max_iter=int(inputs.light_time_iterations)), reference
    )
    checks.append(
        _check("light_time_iteration_residual", "upper_bound", analytic, numeric, tolerance)
    )
    return tuple(checks)

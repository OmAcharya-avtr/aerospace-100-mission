"""Point-ahead angle: the closed-form baseline and the iterated reference.

What the angle is
-----------------
At one event at the local terminal, two directions matter: the direction the
received beam *appears* to come from, and the direction the transmit beam must
*leave in*. They are not the same, because the far terminal moves while light
is in transit, and because the local terminal is itself moving and therefore
sees both directions aberrated. The angle between them is the point-ahead
angle, and the transmit beam has to lead.

The decomposition, and why the split is frame-dependent
-------------------------------------------------------
Work first in the chosen inertial frame. Let ``n`` be the unit line of sight
from local to remote at the event epoch, ``rho`` the range, and ``tau`` the
light time. Then

    apparent source direction  ~  n - v_remote_perp / c
    required transmit direction ~  n + v_remote_perp / c

so in the inertial frame the two differ by ``2 v_remote_perp / c``. This is the
**light-time lead**: it is produced entirely by the remote terminal's motion
during transit and it does not involve the local terminal's velocity at all.

Now transform both photon directions into the local terminal's instantaneous
rest frame, which is the frame its optics actually live in. Aberration
(:mod:`pointahead.aberration`) displaces a propagation direction by
``-beta_perp`` to first order. The incoming photon propagates along ``-n`` and
the outgoing along ``+n``, so the *apparent source* direction moves by
``+beta_local_perp`` and the *transmit* direction by ``-beta_local_perp``. Two
separate, equal, first-order terms:

    stellar aberration (received beam)   : -v_local_perp / c on the point-ahead
    velocity aberration (transmit beam)  : -v_local_perp / c on the point-ahead

Summing the three gives the standard textbook result,

    theta_pa  =  2 |(v_remote - v_local)_perp| / c  =  2 |v_rel_perp| / c,

which for two low Earth orbit terminals on crossing planes is of order 50
microradians.

The consequence worth stating plainly, because it is easy to get wrong: the
**total** depends only on the relative transverse velocity and is therefore the
same in every inertial frame to first order, but the **split** between the
light-time lead and the two aberration terms is not -- it moves entirely into
the light-time lead in the remote terminal's rest frame and entirely into the
aberration terms in a frame where the remote terminal is at rest relative to
the line of sight. Every decomposition this module returns therefore carries
the frame label it was computed in. ``validation/validate_frames.py`` measures
both halves of that statement.

Terminology note. Usage in the optical-communications literature is not
uniform: "velocity aberration" is used both for the total required lead
``2 v_rel_perp / c`` (as it is in satellite laser ranging, where the return
beam is a retroreflection) and for the transmit-side half of the aberration
pair. This module uses the second meaning, names the field explicitly, and
reports the total as ``total_rad`` so that no reader has to guess which
convention a number follows.

References for the structure above: Rindler, *Relativity*, 2nd ed., OUP 2006,
section 3.6 for aberration; Moyer, JPL Monograph 2, Wiley 2003, chapter 8 for
the light-time iteration; Hemmati (ed.), *Deep Space Optical Communications*,
JPL Deep-Space Communications and Navigation Series, Wiley 2006, chapter 1 for
the point-ahead requirement in a link design. No numerical value in this
repository is taken from any of them.

Units: angles in radians, velocities in m/s, positions in metres.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .aberration import aberrate, beta_from_velocity
from .geometry import C_LIGHT, angle_between, orthonormal_basis, transverse, unit
from .lighttime import LightTimeSolution, solve_light_time
from .terminal import Crosslink


@dataclass(frozen=True)
class PointAheadResult:
    """A point-ahead angle with the directions it came from.

    Attributes
    ----------
    total_rad
        Angle between the apparent source direction and the required transmit
        direction, radians, both expressed in the local terminal's frame.
    offset_rad
        Two-component transverse offset of the transmit direction from the
        apparent source direction, in the ``(e1, e2)`` basis returned by
        :func:`pointahead.geometry.orthonormal_basis` about the line of sight.
        Its norm equals ``total_rad`` to within the small-angle error of the
        projection. The *components* depend on the arbitrary but deterministic
        choice of ``e1``; the norm does not.
    source_direction
        Unit vector toward the apparent position of the remote terminal, in the
        local terminal's frame.
    transmit_direction
        Unit vector the transmit beam must leave along, in the local
        terminal's frame.
    method
        ``"closed_form"`` or ``"reference"``.
    frame
        Label of the inertial frame the geometry was supplied in.
    """

    total_rad: float
    offset_rad: np.ndarray
    source_direction: np.ndarray
    transmit_direction: np.ndarray
    method: str
    frame: str

    @property
    def total_urad(self) -> float:
        """Point-ahead angle in microradians."""
        return self.total_rad * 1.0e6

    @property
    def offset_vector(self) -> np.ndarray:
        """``transmit_direction - source_direction`` as a 3-vector, radians.

        Both directions are unit vectors separated by a few tens of
        microradians, so their difference is transverse to the line of sight to
        within that angle and its norm equals ``total_rad`` to the same order.
        It is the basis-free form of ``offset_rad`` and it is what
        :func:`offset_difference` compares, because it is the quantity a
        terminal applies open loop on top of a received-beam track.
        """
        return self.transmit_direction - self.source_direction


def offset_difference(first: PointAheadResult, second: PointAheadResult) -> float:
    """Norm of the difference of two point-ahead offsets, radians.

    This is the error metric used throughout the budget, and the choice is
    deliberate. A terminal that tracks the incoming beam closes the loop on the
    *apparent source direction*, so an error common to the source and transmit
    directions -- a wrong line of sight from a stale ephemeris, for instance --
    is removed by the tracker and is not a point-ahead error. What survives is
    the error in the *offset* between them, which is what this returns.

    The open-loop alternative, the angle between the two transmit directions, is
    also reported where it matters, labelled as such, because a terminal that
    points from ephemeris alone does pay the common part.
    """
    return float(np.linalg.norm(first.offset_vector - second.offset_vector))


@dataclass(frozen=True)
class PointAheadDecomposition:
    """First-order attribution of a point-ahead angle to its three sources.

    Each ``*_rad`` field is the signed transverse offset contributed by that
    term, projected onto the ``(e1, e2)`` basis about the line of sight, with
    ``*_magnitude_rad`` its norm. ``total_rad`` is the norm of the vector sum,
    and ``closure_rad`` is how far the vector sum is from the reference
    solution -- it is the part the first-order attribution does not explain,
    and it is reported rather than absorbed.

    The split is frame-dependent; see the module docstring. ``frame`` records
    which frame it is a split in.
    """

    light_time_lead: np.ndarray
    stellar_aberration: np.ndarray
    velocity_aberration: np.ndarray
    total_rad: float
    closure_rad: float
    frame: str

    @property
    def light_time_lead_magnitude_rad(self) -> float:
        """Norm of the light-time lead contribution, radians."""
        return float(np.linalg.norm(self.light_time_lead))

    @property
    def stellar_aberration_magnitude_rad(self) -> float:
        """Norm of the received-beam aberration contribution, radians."""
        return float(np.linalg.norm(self.stellar_aberration))

    @property
    def velocity_aberration_magnitude_rad(self) -> float:
        """Norm of the transmit-beam aberration contribution, radians."""
        return float(np.linalg.norm(self.velocity_aberration))

    def as_lines(self) -> tuple[tuple[str, float], ...]:
        """``(name, magnitude_rad)`` pairs in a fixed order, for reports."""
        return (
            ("light_time_lead", self.light_time_lead_magnitude_rad),
            ("stellar_aberration", self.stellar_aberration_magnitude_rad),
            ("velocity_aberration", self.velocity_aberration_magnitude_rad),
        )


def _project(vector: np.ndarray, e1: np.ndarray, e2: np.ndarray) -> np.ndarray:
    return np.array([float(np.dot(vector, e1)), float(np.dot(vector, e2))])


def closed_form_point_ahead(link: Crosslink) -> PointAheadResult:
    """Closed-form first-order point-ahead angle. The baseline.

    Implements ``theta_pa = 2 (v_rel)_perp / c`` with ``perp`` taken about the
    instantaneous line of sight, and builds the two directions by applying half
    the offset either side of it. Validity: first order in ``beta``, and it
    assumes the remote terminal moves in a straight line at constant velocity
    over the light time, so its error grows with range through the trajectory
    curvature. Both error sources are measured against
    :func:`reference_point_ahead` and reported as budget lines rather than
    bounded by argument.

    Cost: no trajectory evaluation and no iteration.
    """
    n, e1, e2 = orthonormal_basis(link.separation_m)
    v_perp = transverse(link.relative_velocity_mps, n)
    offset_vector = 2.0 * v_perp / C_LIGHT
    half = 0.5 * offset_vector
    source = unit(n - half, "closed-form source direction")
    transmit = unit(n + half, "closed-form transmit direction")
    return PointAheadResult(
        total_rad=angle_between(source, transmit),
        offset_rad=_project(offset_vector, e1, e2),
        source_direction=source,
        transmit_direction=transmit,
        method="closed_form",
        frame=link.frame,
    )


def reference_point_ahead(
    link: Crosslink,
    tol_s: float = 1.0e-15,
    max_iter: int = 30,
    aberration_route: str = "momentum_boost",
) -> PointAheadResult:
    """Iterated light-time point-ahead angle with exact aberration. The reference.

    Pipeline, in order:

    1. Solve the receive light time against the remote trajectory, giving the
       retarded remote position.
    2. Solve the transmit light time, giving the advanced remote position.
    3. Form the two inertial-frame photon propagation directions.
    4. Aberrate both into the local terminal's rest frame with the exact
       relativistic transformation and the local terminal's velocity.
    5. Take the angle between the apparent source direction and the transmit
       direction.

    Parameters
    ----------
    link
        The crosslink, carrying both states and the declared trajectory model.
    tol_s, max_iter
        Passed to :func:`pointahead.lighttime.solve_light_time`. ``max_iter=0``
        or ``1`` deliberately truncates the iteration and is how the
        light-time-iteration budget line is measured.
    aberration_route
        One of :data:`pointahead.aberration.ABERRATION_ROUTES`. The default is
        exact. Passing ``"first_order"`` is how the aberration-model budget
        line is measured.

    Notes
    -----
    This is the reference the closed form is scored against. It is not a truth
    model: it is exact in special relativity for the declared trajectory model,
    and it inherits every error in that model. A two-body remote trajectory is
    wrong by the oblateness and drag terms it omits, and over an ephemeris age
    of hours that dominates everything else in the budget.
    """
    local_trajectory, remote_trajectory = link.trajectories()
    del local_trajectory
    epoch = link.epoch_s
    receive = solve_light_time(
        remote_trajectory, link.local.position_m, epoch, "receive", tol_s, max_iter
    )
    transmit = solve_light_time(
        remote_trajectory, link.local.position_m, epoch, "transmit", tol_s, max_iter
    )
    beta_local = beta_from_velocity(link.local.velocity_mps)

    incoming_inertial = unit(
        link.local.position_m - receive.remote_position_m, "incoming propagation direction"
    )
    outgoing_inertial = unit(
        transmit.remote_position_m - link.local.position_m, "outgoing propagation direction"
    )
    incoming_local = aberrate(incoming_inertial, beta_local, aberration_route)
    outgoing_local = aberrate(outgoing_inertial, beta_local, aberration_route)

    source = -incoming_local
    transmit_direction = outgoing_local
    n, e1, e2 = orthonormal_basis(link.separation_m)
    offset_vector = transmit_direction / float(np.dot(transmit_direction, n)) - source / float(
        np.dot(source, n)
    )
    return PointAheadResult(
        total_rad=angle_between(source, transmit_direction),
        offset_rad=_project(offset_vector, e1, e2),
        source_direction=source,
        transmit_direction=transmit_direction,
        method="reference",
        frame=link.frame,
    )


def light_time_pair(
    link: Crosslink, tol_s: float = 1.0e-15, max_iter: int = 30
) -> tuple[LightTimeSolution, LightTimeSolution]:
    """``(receive, transmit)`` light-time solutions for ``link``."""
    _, remote_trajectory = link.trajectories()
    return (
        solve_light_time(
            remote_trajectory, link.local.position_m, link.epoch_s, "receive", tol_s, max_iter
        ),
        solve_light_time(
            remote_trajectory, link.local.position_m, link.epoch_s, "transmit", tol_s, max_iter
        ),
    )


def decompose_point_ahead(link: Crosslink) -> PointAheadDecomposition:
    """Attribute the point-ahead offset to its three first-order sources.

    Returns the light-time lead ``2 v_remote_perp / c``, and the two equal
    aberration contributions ``-v_local_perp / c`` each, as transverse offset
    vectors in the ``(e1, e2)`` basis about the line of sight, together with
    the closure against :func:`reference_point_ahead`.

    The closure is the honest part of this function. It is non-zero and it is
    the sum of everything the first-order attribution leaves out: trajectory
    curvature over the light time, second-order aberration, and the difference
    between the instantaneous and light-time-corrected line of sight.
    """
    n, e1, e2 = orthonormal_basis(link.separation_m)
    remote_perp = transverse(link.remote.velocity_mps, n)
    local_perp = transverse(link.local.velocity_mps, n)
    lead = _project(2.0 * remote_perp / C_LIGHT, e1, e2)
    stellar = _project(-local_perp / C_LIGHT, e1, e2)
    velocity = _project(-local_perp / C_LIGHT, e1, e2)
    total_vector = lead + stellar + velocity
    reference = reference_point_ahead(link)
    return PointAheadDecomposition(
        light_time_lead=lead,
        stellar_aberration=stellar,
        velocity_aberration=velocity,
        total_rad=float(np.linalg.norm(total_vector)),
        closure_rad=float(np.linalg.norm(total_vector - reference.offset_rad)),
        frame=link.frame,
    )


@dataclass(frozen=True)
class BaselineScore:
    """How the closed-form baseline scores against the iterated reference.

    Attributes
    ----------
    closed_form_rad, reference_rad
        The two point-ahead magnitudes, radians.
    offset_difference_rad
        Norm of the difference of the two offset vectors. This is the
        closed-loop error, the one a terminal that tracks the received beam
        actually pays, and it is the line that enters the error budget.
    transmit_direction_rad
        Angle between the two commanded transmit directions. This is the
        open-loop error and it is larger, because it also contains the
        difference between the instantaneous line of sight and the
        light-time-corrected one.
    magnitude_difference_rad
        ``|closed_form_rad - reference_rad|``. Reported because it is the
        number a reader will compute from the two magnitudes, and it is
        *smaller* than the offset difference whenever the two offsets differ in
        direction rather than in size. Quoting it alone would understate the
        error, which is why all three are returned.
    """

    closed_form_rad: float
    reference_rad: float
    offset_difference_rad: float
    transmit_direction_rad: float
    magnitude_difference_rad: float


def score_closed_form(
    link: Crosslink, tol_s: float = 1.0e-15, max_iter: int = 30
) -> BaselineScore:
    """Score the closed-form baseline against the iterated light-time reference.

    Three error metrics are returned rather than one; see :class:`BaselineScore`
    for why none of them alone is the honest answer.
    """
    baseline = closed_form_point_ahead(link)
    reference = reference_point_ahead(link, tol_s, max_iter)
    return BaselineScore(
        closed_form_rad=baseline.total_rad,
        reference_rad=reference.total_rad,
        offset_difference_rad=offset_difference(baseline, reference),
        transmit_direction_rad=angle_between(
            baseline.transmit_direction, reference.transmit_direction
        ),
        magnitude_difference_rad=abs(baseline.total_rad - reference.total_rad),
    )

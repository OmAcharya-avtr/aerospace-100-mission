"""Reaction-wheel array model: torque and momentum envelopes, and saturation.

A wheel array is ``m`` wheels with unit spin axes ``a_i`` fixed in the body.
Each wheel stores momentum ``h_i a_i`` and applies ``-h_i_dot a_i`` to the
spacecraft, so with the distribution matrix ``A = [a_1 ... a_m]`` (3 x m) the
body torque delivered is

    tau_body = -A h_dot,      h_body_wheels = A h

and the constraints are per-wheel: ``|h_i| <= h_max`` and
``|h_i_dot| <= tau_max``. Both are box constraints in wheel space, which makes
the reachable body sets zonotopes.

The two capability questions a slew planner asks are directional:

    tau_cap(u)  = max { t >= 0 : t u = A c, |c_i| <= tau_max }
    h_cap(u)    = max { t >= 0 : t u = A h, |h_i| <= h_max }

Both are linear programmes, but the feasible set ``{A c : |c_i| <= bound}`` is
a zonotope, so both have a closed form: every facet of a zonotope is spanned by
a pair of generators, giving facet normals ``n_ij = a_i x a_j`` and support
``h(n) = bound * sum_k |a_k · n|``, whence

    cap(u) = min over facets of  h(n_f) / |u · n_f|

That is what :meth:`WheelArray.max_torque_along` evaluates -- about 60 times
faster than the equivalent linear programme, which is retained as
:meth:`WheelArray.max_torque_along_lp` and used to cross-check it (agreement
1.665e-16 over 200 random directions, `validation/validate_wheel_envelope.py`).

The commonly used minimum-norm shortcut ``t = tau_max / max_i |(A^+ u)_i|`` is
a *lower* bound on the same quantity and is provided separately as
:meth:`WheelArray.pseudo_inverse_capability` so the gap can be measured rather
than assumed away.

Units: torque N*m, momentum N*m*s, angles rad. Wheel axes are dimensionless
unit vectors in the body frame.

References
----------
B. Wie, *Space Vehicle Dynamics and Control*, 2nd ed., AIAA (2008), Sec. 7.3 --
    reaction-wheel arrays, distribution matrices and momentum envelopes.
J. R. Wertz (ed.), *Spacecraft Attitude Determination and Control*, Reidel
    (1978), Sec. 6.6 -- momentum-exchange actuators.
F. L. Markley and J. L. Crassidis, *Fundamentals of Spacecraft Attitude
    Determination and Control*, Springer (2014), Sec. 7.2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import linprog

from .attitude import unit_vector

__all__ = ["WheelArray", "orthogonal_wheels", "pyramid_wheels"]

_PYRAMID_ANGLE = math.atan(math.sqrt(2.0))
"""Elevation of a standard four-wheel pyramid from the +z axis, arctan(sqrt 2)
= 54.735610317245346 deg -- the angle at which the array's momentum envelope is
closest to a sphere (Wie 2008, Sec. 7.3)."""


@dataclass(frozen=True)
class WheelArray:
    """A reaction-wheel array with per-wheel torque and momentum limits.

    Parameters
    ----------
    axes : array_like
        Spin axes, shape ``(m, 3)``, normalised on construction. Body frame.
    max_torque : float
        Per-wheel torque limit [N*m], ``> 0``. Same for every wheel.
    max_momentum : float
        Per-wheel momentum limit [N*m*s], ``> 0``.
    name : str
        Label.

    Notes
    -----
    Wheel *dynamics* are not modelled: no motor time constant, no bearing
    drag, no zero-speed friction, no wheel-speed-dependent torque roll-off.
    The array is a static capability envelope. See README limitations.
    """

    axes: NDArray[np.float64]
    max_torque: float
    max_momentum: float
    name: str = "wheels"

    def __post_init__(self) -> None:
        a = np.atleast_2d(np.asarray(self.axes, dtype=float))
        if a.ndim != 2 or a.shape[1] != 3:
            raise ValueError(f"axes must have shape (m, 3), got {a.shape}")
        if a.shape[0] < 1:
            raise ValueError("a wheel array needs at least one wheel")
        object.__setattr__(self, "axes", unit_vector(a))
        for field_name in ("max_torque", "max_momentum"):
            v = float(getattr(self, field_name))
            if not math.isfinite(v) or v <= 0.0:
                raise ValueError(f"{field_name} must be finite and > 0, got {v}")
            object.__setattr__(self, field_name, v)
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string")

    @property
    def n_wheels(self) -> int:
        """Number of wheels."""
        return int(self.axes.shape[0])

    @property
    def distribution(self) -> NDArray[np.float64]:
        """Distribution matrix ``A``, shape ``(3, m)``: columns are the axes."""
        return self.axes.T.copy()

    @property
    def rank(self) -> int:
        """Rank of ``A``. Below 3 the array cannot produce torque in some
        direction and the planner reports that direction as unreachable."""
        return int(np.linalg.matrix_rank(self.distribution, tol=1e-10))

    def _facets(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Facet normals and unit support values of the wheel-space zonotope.

        The set ``{A c : |c_i| <= 1}`` is a zonotope whose every facet is
        spanned by a pair of generators, so its facet normals are the
        normalised cross products ``a_i x a_j`` and its support in direction
        ``n`` is ``sum_k |a_k · n|`` (Ziegler, *Lectures on Polytopes*, 1995,
        Sec. 7.3). The zonotope is therefore the intersection of the slabs
        ``|y · n_f| <= h_f``, which turns the directional-capability LP into a
        minimum over at most ``m (m-1) / 2`` scalar ratios.

        Cached on first use. Returns ``(normals (f, 3), h_unit (f,))`` with
        ``h_unit`` the support for a unit per-wheel bound; multiply by the
        actual bound. Empty arrays when the array is rank-deficient, in which
        case the LP path is used instead.
        """
        cached = getattr(self, "_facet_cache", None)
        if cached is not None:
            return cached
        normals: list[NDArray[np.float64]] = []
        for i in range(self.n_wheels):
            for j in range(i + 1, self.n_wheels):
                n = np.cross(self.axes[i], self.axes[j])
                nn = float(np.linalg.norm(n))
                if nn > 1e-12:
                    normals.append(n / nn)
        if not normals or self.rank < 3:
            out = (np.zeros((0, 3)), np.zeros(0))
        else:
            nrm = np.asarray(normals)
            h = np.sum(np.abs(self.axes @ nrm.T), axis=0)
            out = (nrm, h)
        object.__setattr__(self, "_facet_cache", out)
        return out

    def envelope(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Facet normals ``(f, 3)`` and unit-bound supports ``(f,)``.

        Public because the planner evaluates thousands of directional
        capabilities per solve and wants the arrays once rather than a
        validated call per query. Empty arrays for a rank-deficient array.
        """
        n, h = self._facets()
        return n.copy(), h.copy()

    def _directional_capability(self, direction: ArrayLike, bound: float) -> float:
        """Exact ``max{t >= 0 : t u = A c, |c_i| <= bound}``.

        Uses the zonotope facet description when the array has rank 3, and
        falls back to :meth:`_directional_capability_lp` otherwise. The two
        agree to 1e-12 over random arrays and directions
        (`validation/validate_wheel_envelope.py`).
        """
        u = unit_vector(direction).reshape(3)
        normals, h_unit = self._facets()
        if normals.shape[0] == 0:
            return self._directional_capability_lp(u, bound)
        denom = np.abs(normals @ u)
        active = denom > 1e-12
        if not np.any(active):
            return self._directional_capability_lp(u, bound)
        return float(np.min(bound * h_unit[active] / denom[active]))

    def _directional_capability_lp(self, direction: ArrayLike, bound: float) -> float:
        """The same quantity by linear programming; the independent route."""
        u = unit_vector(direction).reshape(3)
        a = self.distribution
        m = self.n_wheels
        a_eq = np.hstack([a, -u.reshape(3, 1)])
        cost = np.zeros(m + 1)
        cost[-1] = -1.0
        bounds = [(-bound, bound)] * m + [(0.0, None)]
        res = linprog(cost, A_eq=a_eq, b_eq=np.zeros(3), bounds=bounds, method="highs")
        if not res.success:
            # The box is compact so the LP is always bounded, and c = 0, t = 0
            # is always feasible, so failure here means a solver problem and is
            # reported rather than silently returned as zero capability.
            raise RuntimeError(f"wheel capability LP failed: {res.message}")
        return float(res.x[-1])

    def max_torque_along_lp(self, direction: ArrayLike) -> float:
        """:meth:`max_torque_along` computed by linear programming instead."""
        return self._directional_capability_lp(direction, self.max_torque)

    def max_momentum_along_lp(self, direction: ArrayLike) -> float:
        """:meth:`max_momentum_along` computed by linear programming instead."""
        return self._directional_capability_lp(direction, self.max_momentum)

    def max_torque_along(self, direction: ArrayLike) -> float:
        """Largest body torque [N*m] deliverable along ``direction``.

        Exact linear-programming answer over the per-wheel torque box.
        Returns 0.0 for a direction outside the column space of ``A``.
        """
        return self._directional_capability(direction, self.max_torque)

    def max_momentum_along(self, direction: ArrayLike) -> float:
        """Largest body momentum [N*m*s] storable along ``direction``. Exact LP."""
        return self._directional_capability(direction, self.max_momentum)

    def pseudo_inverse_capability(self, direction: ArrayLike, momentum: bool = False) -> float:
        """Minimum-norm-allocation capability along ``direction``.

        ``bound / max_i |(A^+ u)_i|`` -- the number a spreadsheet gets by
        distributing the demand with the pseudo-inverse. It is a lower bound on
        :meth:`max_torque_along` because the minimum-norm allocation is only one
        of the feasible allocations. `validation/validate_wheel_envelope.py`
        measures the gap.
        """
        u = unit_vector(direction).reshape(3)
        c = self.pseudo_inverse() @ u
        peak = float(np.max(np.abs(c)))
        if peak < 1e-15:
            return 0.0
        bound = self.max_momentum if momentum else self.max_torque
        return bound / peak

    def pseudo_inverse(self) -> NDArray[np.float64]:
        """Moore-Penrose pseudo-inverse of ``A``, shape ``(m, 3)``, cached.

        Cached because the simulator allocates a torque at every RK4 stage and
        recomputing an SVD there dominated the run time.
        """
        cached = getattr(self, "_pinv_cache", None)
        if cached is None:
            cached = np.linalg.pinv(self.distribution)
            object.__setattr__(self, "_pinv_cache", cached)
        return cached

    def allocate(self, body_torque: ArrayLike) -> NDArray[np.float64]:
        """Minimum-norm wheel torques [N*m] for a commanded body torque.

        ``c = -A^+ tau`` (the sign is the reaction: wheels spin up opposite to
        the torque they impart). No bounds are enforced here -- use
        :meth:`torque_feasible` to ask whether the command is deliverable at
        all, which is a different question from whether the minimum-norm
        allocation happens to fit.
        """
        tau = np.asarray(body_torque, dtype=float).reshape(3)
        return -self.pseudo_inverse() @ tau

    def torque_feasible(self, body_torque: ArrayLike, tol: float = 1e-9) -> bool:
        """``True`` if some allocation within the torque box delivers the command."""
        tau = np.asarray(body_torque, dtype=float).reshape(3)
        mag = float(np.linalg.norm(tau))
        if mag <= tol:
            return True
        return self.max_torque_along(tau) >= mag - tol

    def momentum_feasible(self, body_momentum: ArrayLike, tol: float = 1e-9) -> bool:
        """``True`` if the array can store the commanded body momentum."""
        h = np.asarray(body_momentum, dtype=float).reshape(3)
        mag = float(np.linalg.norm(h))
        if mag <= tol:
            return True
        return self.max_momentum_along(h) >= mag - tol


def pyramid_wheels(
    max_torque: float,
    max_momentum: float,
    elevation: float = _PYRAMID_ANGLE,
    name: str = "pyramid",
) -> WheelArray:
    """Standard skewed four-wheel pyramid.

    Four wheels at ``elevation`` from the +z body axis, azimuths 0, 90, 180 and
    270 deg. The default elevation ``arctan(sqrt 2)`` = 54.7356 deg is the
    isotropic choice (Wie 2008, Sec. 7.3): the momentum envelope is then as
    close to spherical as a four-wheel pyramid gets, and any single wheel
    failure leaves a rank-3 array.

    Parameters
    ----------
    max_torque, max_momentum : float
        Per-wheel limits [N*m], [N*m*s].
    elevation : float
        Angle from +z [rad], in ``(0, pi/2)``.
    """
    el = float(elevation)
    if not 0.0 < el < 0.5 * math.pi:
        raise ValueError(f"elevation must lie in (0, pi/2) rad, got {el}")
    az = np.array([0.0, 0.5, 1.0, 1.5]) * math.pi
    axes = np.stack(
        [np.sin(el) * np.cos(az), np.sin(el) * np.sin(az), np.full(4, math.cos(el))], axis=-1
    )
    return WheelArray(axes=axes, max_torque=max_torque, max_momentum=max_momentum, name=name)


def orthogonal_wheels(
    max_torque: float, max_momentum: float, name: str = "orthogonal"
) -> WheelArray:
    """Three wheels along the body axes. ``A = I``, so capability is the box."""
    return WheelArray(
        axes=np.eye(3), max_torque=max_torque, max_momentum=max_momentum, name=name
    )

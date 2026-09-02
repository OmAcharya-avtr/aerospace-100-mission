"""Single-gimbal control-moment-gyro (SGCMG) array geometry and momentum map.

Conventions
-----------
All vectors are expressed in the spacecraft **body frame**, right-handed,
orthonormal, with the same active-rotation convention as QuatKit (P007):
``v_body = R v_frame`` for a rotation matrix ``R``.  No quaternion appears in
this module; the array geometry is expressed directly as body-frame unit
vectors, so the convention only matters when a caller rotates the array.

A single-gimbal CMG ``i`` is described by

* a **gimbal axis** ``g_i`` (unit, fixed in the body frame),
* a **reference axis** ``c_i`` (unit, perpendicular to ``g_i``), the direction
  of the rotor angular momentum at gimbal angle ``delta_i = 0``,
* a **transverse axis** ``s_i = g_i x c_i`` (unit), completing the right-handed
  triad, and
* a rotor angular momentum magnitude ``h0_i`` [N*m*s], assumed constant (the
  rotor runs at fixed speed; variable-speed CMGs are out of scope).

The rotor momentum direction at gimbal angle ``delta_i`` is the right-handed
rotation of ``c_i`` about ``g_i``::

    h_hat_i(delta_i) = c_i cos(delta_i) + s_i sin(delta_i)                  (1)

and the array momentum map is::

    h(delta) = sum_i h0_i h_hat_i(delta_i)                    [N*m*s]       (2)

Its Jacobian, the matrix that every steering law inverts, is::

    A(delta) = dh/ddelta,   A[:, i] = h0_i (g_i x h_hat_i)
                                    = h0_i (-c_i sin d_i + s_i cos d_i)     (3)

with units [N*m*s/rad].  The CMG output torque is ``tau_cmg = dh/dt =
A(delta) ddelta/dt`` and, by conservation of angular momentum on a rigid
vehicle with no external torque, the torque delivered **to the spacecraft** is
its negative::

    tau_body = -A(delta) ddelta/dt                            [N*m]         (4)

Every public function in this package that takes or returns a ``torque``
means ``tau_body`` of equation (4).

Source: standard SGCMG formulation as given by Margulies & Aubrun (1978),
Bedrossian et al. (1990), Wie (2008) and the survey of Kurokawa (2007).
Assumptions: rigid vehicle, rigid gimbals, constant rotor speed, no gimbal
friction or flexibility, no gimbal-angle measurement error, and gimbal rate as
the commanded quantity (a rate-servo inner loop is assumed ideal).
Validity: any gimbal angle; the model is exact for the stated assumptions and
carries no small-angle approximation.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "CMGArray",
    "general_array",
    "pyramid_array",
    "roof_array",
    "STANDARD_PYRAMID_SKEW_DEG",
]

#: Skew angle of the textbook four-CMG pyramid, in degrees.  ``arctan(4/3)``.
#: This is the value used for the pyramid array throughout the SGCMG
#: literature (Wie 2008; Kurokawa 2007) because it makes the array's momentum
#: envelope close to spherical.  It is a default, not a physical constant.
STANDARD_PYRAMID_SKEW_DEG: float = float(np.degrees(np.arctan2(4.0, 3.0)))

_UNIT_TOL = 1e-9


def _as_unit_rows(vectors: ArrayLike, name: str) -> NDArray[np.float64]:
    """Return ``vectors`` as an ``(n, 3)`` float array of unit rows."""
    arr = np.asarray(vectors, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{name} must have shape (n, 3), got {arr.shape}")
    norms = np.linalg.norm(arr, axis=1)
    if np.any(norms < _UNIT_TOL):
        bad = int(np.argmin(norms))
        raise ValueError(f"{name} row {bad} has near-zero length {norms[bad]:.3e}")
    return arr / norms[:, None]


def _orthogonal_to(axis: NDArray[np.float64]) -> NDArray[np.float64]:
    """A unit vector perpendicular to ``axis`` (deterministic choice)."""
    trial = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    out = np.cross(trial, axis)
    return out / np.linalg.norm(out)


@dataclass(frozen=True)
class CMGArray:
    """A single-gimbal CMG array: geometry, momentum map and Jacobian.

    Parameters
    ----------
    gimbal_axes
        ``(n, 3)`` body-frame gimbal axes ``g_i``, normalised on construction.
    ref_axes
        ``(n, 3)`` body-frame reference axes ``c_i``: the rotor momentum
        direction at ``delta_i = 0``.  Each row is normalised and then the
        component along ``g_i`` is removed; a row parallel to its gimbal axis
        raises ``ValueError``.
    rotor_momenta
        ``(n,)`` rotor angular momentum magnitudes ``h0_i`` [N*m*s], all > 0.
    names
        Per-CMG labels, defaulting to ``("cmg1", ...)``.
    locked
        ``(n,)`` boolean mask.  ``True`` marks a CMG whose gimbal can no longer
        move (a failed gimbal actuator).  Its rotor momentum still contributes
        to ``momentum`` but its column is removed from ``jacobian``.
    """

    gimbal_axes: NDArray[np.float64]
    ref_axes: NDArray[np.float64]
    rotor_momenta: NDArray[np.float64]
    names: tuple[str, ...] = field(default=())
    locked: NDArray[np.bool_] = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        g = _as_unit_rows(self.gimbal_axes, "gimbal_axes")
        c = _as_unit_rows(self.ref_axes, "ref_axes")
        if g.shape != c.shape:
            raise ValueError(
                f"gimbal_axes {g.shape} and ref_axes {c.shape} must have the same shape"
            )
        n = g.shape[0]
        if n < 2:
            raise ValueError(f"a CMG array needs at least 2 CMGs, got {n}")

        # Remove any component of c along g, then re-normalise.
        c = c - (np.sum(c * g, axis=1))[:, None] * g
        residual = np.linalg.norm(c, axis=1)
        if np.any(residual < 1e-6):
            bad = int(np.argmin(residual))
            raise ValueError(
                f"ref_axes row {bad} is parallel to its gimbal axis; the rotor momentum "
                "direction must be perpendicular to the gimbal axis"
            )
        c = c / residual[:, None]

        h0 = np.asarray(self.rotor_momenta, dtype=float).reshape(-1)
        if h0.shape[0] != n:
            raise ValueError(f"rotor_momenta must have length {n}, got {h0.shape[0]}")
        if np.any(h0 <= 0.0):
            raise ValueError("every rotor momentum must be strictly positive [N*m*s]")

        names = tuple(self.names) if self.names else tuple(f"cmg{i + 1}" for i in range(n))
        if len(names) != n:
            raise ValueError(f"names must have length {n}, got {len(names)}")

        locked = (
            np.zeros(n, dtype=bool)
            if self.locked is None
            else np.asarray(self.locked, dtype=bool).reshape(-1)
        )
        if locked.shape[0] != n:
            raise ValueError(f"locked must have length {n}, got {locked.shape[0]}")

        for arr in (g, c, h0, locked):
            arr.flags.writeable = False
        object.__setattr__(self, "gimbal_axes", g)
        object.__setattr__(self, "ref_axes", c)
        object.__setattr__(self, "rotor_momenta", h0)
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "locked", locked)

    # -- basic geometry ---------------------------------------------------

    @property
    def n_cmgs(self) -> int:
        """Number of CMGs in the array."""
        return int(self.gimbal_axes.shape[0])

    @property
    def n_free(self) -> int:
        """Number of CMGs whose gimbal can still move."""
        return int(np.count_nonzero(~self.locked))

    @property
    def transverse_axes(self) -> NDArray[np.float64]:
        """``(n, 3)`` transverse axes ``s_i = g_i x c_i`` (unit, body frame)."""
        return np.cross(self.gimbal_axes, self.ref_axes)

    @property
    def total_momentum_capacity(self) -> float:
        """``sum_i h0_i`` [N*m*s]: the largest momentum magnitude the array can hold."""
        return float(np.sum(self.rotor_momenta))

    @property
    def free_indices(self) -> NDArray[np.intp]:
        """Indices of CMGs whose gimbal can still move."""
        return np.flatnonzero(~self.locked)

    # -- momentum map and Jacobian ----------------------------------------

    def rotor_directions(self, deltas: ArrayLike) -> NDArray[np.float64]:
        """Unit rotor momentum directions ``h_hat_i(delta_i)``, equation (1).

        Parameters
        ----------
        deltas
            ``(n,)`` gimbal angles [rad].

        Returns
        -------
        ``(n, 3)`` unit vectors in the body frame.
        """
        d = self._check_deltas(deltas)
        return self.ref_axes * np.cos(d)[:, None] + self.transverse_axes * np.sin(d)[:, None]

    def momentum(self, deltas: ArrayLike) -> NDArray[np.float64]:
        """Total array angular momentum ``h(delta)`` [N*m*s], equation (2)."""
        return self.rotor_momenta @ self.rotor_directions(deltas)

    def jacobian(self, deltas: ArrayLike, free_only: bool = True) -> NDArray[np.float64]:
        """Momentum-map Jacobian ``A = dh/ddelta`` [N*m*s/rad], equation (3).

        Parameters
        ----------
        deltas
            ``(n,)`` gimbal angles [rad].
        free_only
            When ``True`` (the default) only the columns of CMGs whose gimbal
            can still move are returned, so the result has shape
            ``(3, n_free)``.  ``False`` returns all ``n`` columns, which is what
            a finite-difference check of the full momentum map needs.
        """
        d = self._check_deltas(deltas)
        cols = self.rotor_momenta[:, None] * (
            -self.ref_axes * np.sin(d)[:, None] + self.transverse_axes * np.cos(d)[:, None]
        )
        full = cols.T
        return full[:, self.free_indices] if free_only else full

    def torque(self, deltas: ArrayLike, gimbal_rates: ArrayLike) -> NDArray[np.float64]:
        """Body torque delivered for a gimbal-rate command, equation (4).

        ``tau_body = -A(delta) ddelta/dt`` [N*m].  ``gimbal_rates`` has length
        ``n_free`` (free gimbals only) or ``n`` (full vector; locked entries are
        ignored).
        """
        rates = np.asarray(gimbal_rates, dtype=float).reshape(-1)
        if rates.shape[0] == self.n_cmgs:
            rates = rates[self.free_indices]
        elif rates.shape[0] != self.n_free:
            raise ValueError(
                f"gimbal_rates must have length {self.n_free} (free) or "
                f"{self.n_cmgs} (all), got {rates.shape[0]}"
            )
        return -(self.jacobian(deltas) @ rates)

    def expand_rates(self, gimbal_rates: ArrayLike) -> NDArray[np.float64]:
        """Scatter a length-``n_free`` rate vector into a length-``n`` vector."""
        rates = np.asarray(gimbal_rates, dtype=float).reshape(-1)
        if rates.shape[0] == self.n_cmgs:
            out = rates.copy()
            out[self.locked] = 0.0
            return out
        if rates.shape[0] != self.n_free:
            raise ValueError(
                f"gimbal_rates must have length {self.n_free} or {self.n_cmgs}, "
                f"got {rates.shape[0]}"
            )
        out = np.zeros(self.n_cmgs)
        out[self.free_indices] = rates
        return out

    # -- failure modelling ------------------------------------------------

    def with_locked(self, indices: ArrayLike) -> CMGArray:
        """Copy of this array with the listed gimbals locked (failed).

        A locked CMG keeps its rotor momentum contribution to ``momentum`` at
        whatever gimbal angle it is frozen at, but contributes no column to
        ``jacobian``, so the array loses one degree of freedom.
        """
        idx = np.atleast_1d(np.asarray(indices, dtype=int)).reshape(-1)
        if idx.size and (idx.min() < 0 or idx.max() >= self.n_cmgs):
            raise ValueError(f"locked index out of range for {self.n_cmgs} CMGs: {idx.tolist()}")
        locked = np.array(self.locked, dtype=bool)
        locked[idx] = True
        if int(np.count_nonzero(~locked)) < 1:
            raise ValueError("cannot lock every gimbal; at least one must remain free")
        return replace(self, locked=locked)

    def summary(self) -> str:
        """One-line-per-CMG human-readable description."""
        lines = [
            f"CMGArray: {self.n_cmgs} CMGs ({self.n_free} free), "
            f"capacity {self.total_momentum_capacity:.6g} N*m*s"
        ]
        for i in range(self.n_cmgs):
            g = self.gimbal_axes[i]
            c = self.ref_axes[i]
            state = "LOCKED" if self.locked[i] else "free  "
            lines.append(
                f"  {self.names[i]:>6} {state} h0={self.rotor_momenta[i]:.4g} "
                f"g=({g[0]:+.4f},{g[1]:+.4f},{g[2]:+.4f}) "
                f"c=({c[0]:+.4f},{c[1]:+.4f},{c[2]:+.4f})"
            )
        return "\n".join(lines)

    # -- internals --------------------------------------------------------

    def _check_deltas(self, deltas: ArrayLike) -> NDArray[np.float64]:
        d = np.asarray(deltas, dtype=float).reshape(-1)
        if d.shape[0] != self.n_cmgs:
            raise ValueError(f"deltas must have length {self.n_cmgs}, got {d.shape[0]}")
        if not np.all(np.isfinite(d)):
            raise ValueError("deltas must be finite")
        return d


def general_array(
    gimbal_axes: ArrayLike,
    ref_axes: ArrayLike | None = None,
    rotor_momentum: ArrayLike | float = 1.0,
    names: tuple[str, ...] | None = None,
) -> CMGArray:
    """Build a CMG array from arbitrary gimbal axes.

    Parameters
    ----------
    gimbal_axes
        ``(n, 3)`` body-frame gimbal axes; normalised internally.
    ref_axes
        ``(n, 3)`` reference axes.  ``None`` picks, for each CMG, a
        deterministic unit vector perpendicular to its gimbal axis, which only
        shifts the origin of that CMG's gimbal angle.
    rotor_momentum
        Scalar or ``(n,)`` rotor momenta [N*m*s].
    """
    g = np.asarray(gimbal_axes, dtype=float)
    if g.ndim != 2 or g.shape[1] != 3:
        raise ValueError(f"gimbal_axes must have shape (n, 3), got {g.shape}")
    n = g.shape[0]
    if ref_axes is None:
        norms = np.linalg.norm(g, axis=1)
        if np.any(norms < _UNIT_TOL):
            bad = int(np.argmin(norms))
            raise ValueError(f"gimbal_axes row {bad} has near-zero length {norms[bad]:.3e}")
        unit = g / norms[:, None]
        ref = np.array([_orthogonal_to(unit[i]) for i in range(n)])
    else:
        ref = np.asarray(ref_axes, dtype=float)
    h0 = np.broadcast_to(np.asarray(rotor_momentum, dtype=float), (n,)).astype(float)
    return CMGArray(g, ref, h0, names=names or ())


def pyramid_array(
    skew_angle_deg: float = STANDARD_PYRAMID_SKEW_DEG,
    rotor_momentum: float = 1.0,
    n_cmgs: int = 4,
) -> CMGArray:
    """The classical pyramid SGCMG array.

    ``n_cmgs`` gimbal axes are equally spaced in azimuth on a cone of half
    angle ``90 deg - skew_angle_deg`` about the body ``+z`` axis::

        g_i = (sin(b) cos(t_i), sin(b) sin(t_i), cos(b)),  t_i = 2 pi i / n
        c_i = (-sin(t_i), cos(t_i), 0)

    where ``b`` is the skew angle measured from the body ``x-y`` plane.  For
    ``n_cmgs = 4`` this reproduces the four-CMG pyramid used throughout the
    SGCMG literature (Wie 2008; Kurokawa 2007), whose momentum map is

        h = h0 * [ -cos(b) sin(d1) - cos(d2) + cos(b) sin(d3) + cos(d4),
                    cos(d1) - cos(b) sin(d2) - cos(d3) + cos(b) sin(d4),
                    sin(b) (sin(d1) + sin(d2) + sin(d3) + sin(d4)) ].

    Parameters
    ----------
    skew_angle_deg
        Skew angle ``b`` in degrees, strictly between 0 and 90.  The default
        ``STANDARD_PYRAMID_SKEW_DEG`` = 53.13 deg is the textbook value.
    rotor_momentum
        Rotor momentum ``h0`` of every CMG [N*m*s].
    n_cmgs
        Number of CMGs, at least 3.
    """
    if not 0.0 < skew_angle_deg < 90.0:
        raise ValueError(f"skew_angle_deg must lie in (0, 90), got {skew_angle_deg}")
    if n_cmgs < 3:
        raise ValueError(f"a pyramid array needs at least 3 CMGs, got {n_cmgs}")
    if rotor_momentum <= 0.0:
        raise ValueError(f"rotor_momentum must be positive [N*m*s], got {rotor_momentum}")
    beta = np.radians(skew_angle_deg)
    theta = 2.0 * np.pi * np.arange(n_cmgs) / n_cmgs
    g = np.column_stack(
        [np.sin(beta) * np.cos(theta), np.sin(beta) * np.sin(theta), np.full(n_cmgs, np.cos(beta))]
    )
    c = np.column_stack([-np.sin(theta), np.cos(theta), np.zeros(n_cmgs)])
    h0 = np.full(n_cmgs, float(rotor_momentum))
    return CMGArray(g, c, h0, names=tuple(f"cmg{i + 1}" for i in range(n_cmgs)))


def roof_array(
    skew_angle_deg: float = 45.0,
    rotor_momentum: float = 1.0,
    n_pairs: int = 2,
) -> CMGArray:
    """A roof-type SGCMG array: pairs of parallel gimbal axes.

    Pair ``k`` of two CMGs shares the gimbal axis

        g = (0, (-1)^k sin(b), cos(b))   for n_pairs = 2

    and in general the pairs' axes are equally spaced in azimuth on the same
    cone as :func:`pyramid_array`, two CMGs per axis.  Because the two members
    of a pair have parallel gimbal axes their momenta stay in a common plane,
    which is what makes the roof array's singular set different in character
    from the pyramid's; the taxonomy is Kurokawa's (2007).

    Parameters
    ----------
    skew_angle_deg
        Tilt ``b`` of each ridge from the body ``x-y`` plane, in ``(0, 90)``.
    rotor_momentum
        Rotor momentum of every CMG [N*m*s].
    n_pairs
        Number of parallel pairs, at least 2 (so at least 4 CMGs).
    """
    if not 0.0 < skew_angle_deg < 90.0:
        raise ValueError(f"skew_angle_deg must lie in (0, 90), got {skew_angle_deg}")
    if n_pairs < 2:
        raise ValueError(f"a roof array needs at least 2 pairs, got {n_pairs}")
    if rotor_momentum <= 0.0:
        raise ValueError(f"rotor_momentum must be positive [N*m*s], got {rotor_momentum}")
    beta = np.radians(skew_angle_deg)
    theta = np.pi * np.arange(n_pairs) / n_pairs
    axes = np.column_stack(
        [
            np.sin(beta) * np.cos(theta) * ((-1.0) ** np.arange(n_pairs)),
            np.sin(beta) * np.sin(theta) * ((-1.0) ** np.arange(n_pairs)),
            np.full(n_pairs, np.cos(beta)),
        ]
    )
    g = np.repeat(axes, 2, axis=0)
    refs = np.column_stack([-np.sin(theta), np.cos(theta), np.zeros(n_pairs)])
    c = np.repeat(refs, 2, axis=0)
    n = 2 * n_pairs
    h0 = np.full(n, float(rotor_momentum))
    names = tuple(f"pair{k + 1}{'ab'[j]}" for k in range(n_pairs) for j in range(2))
    return CMGArray(g, c, h0, names=names)

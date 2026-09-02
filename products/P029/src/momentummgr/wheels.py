"""Reaction-wheel array algebra: allocation, saturation margin and zero-speed avoidance.

A wheel array with ``n > 3`` wheels is redundant: the map from wheel momenta to body
momentum, ``h_body = A h_wheel`` with ``A`` the 3-by-n matrix of wheel spin axes, has an
``(n - 3)``-dimensional null space. Any null-space vector can be added to the wheel
momenta without changing the body momentum, and that freedom is what lets a wheel be
kept away from zero speed.

Why zero speed matters. Ball-bearing wheel friction is not linear through zero: it has a
Coulomb/stiction component whose sign flips with the direction of rotation, so a wheel
dwelling near zero speed delivers a torque error of the order of its own stiction torque
and injects broadband disturbance as it repeatedly crosses. Wheel models with this
behaviour and the recommendation to bias wheels away from zero are given in Sidi,
*Spacecraft Dynamics and Control*, and in Markley and Crassidis, *Fundamentals of
Spacecraft Attitude Determination and Control*. This module does not model the friction;
it provides the geometry that lets a scheduler avoid the region.

Units: wheel momenta N m s, wheel inertias kg m^2, speeds rad s^-1, axes dimensionless.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import _validate as _v

__all__ = [
    "WheelArray",
    "Allocation",
    "pyramid_four",
    "tetrahedral_four",
    "orthogonal_three",
    "count_zero_crossings",
]

PYRAMID_ISOTROPIC_HALF_ANGLE_RAD: float = float(np.arctan(np.sqrt(2.0)))
"""Cone half-angle 54.7356 deg for which the four-wheel pyramid is momentum-isotropic.

``A A^T = diag(2 sin^2 b, 2 sin^2 b, 4 cos^2 b)``, equal in all three axes when
``tan^2 b = 2``. Standard result, e.g. Markley and Crassidis, *Fundamentals of Spacecraft
Attitude Determination and Control*, chapter on actuators."""


@dataclass(frozen=True)
class Allocation:
    """Result of distributing a body momentum request over the wheels.

    Attributes
    ----------
    wheel_momentum_nms : (n,)
        Momentum stored in each wheel [N m s].
    null_coefficients : (n - 3,)
        Coefficients of the null-space basis that were added to the minimum-norm
        solution. All zero for a non-redundant array.
    min_abs_momentum_nms, max_abs_momentum_nms : float
        Smallest and largest wheel momentum magnitude [N m s]. The first is the
        zero-speed margin, the second the saturation margin.
    feasible : bool
        False when no null-space choice keeps every wheel inside its momentum limit, in
        which case the minimum-norm solution is returned unchanged and the caller must
        desaturate or accept clipping.
    """

    wheel_momentum_nms: NDArray[np.float64]
    null_coefficients: NDArray[np.float64]
    min_abs_momentum_nms: float
    max_abs_momentum_nms: float
    feasible: bool


@dataclass(frozen=True)
class WheelArray:
    """A set of fixed-axis reaction wheels.

    Attributes
    ----------
    axes : (n, 3)
        Unit spin axes in the body frame, dimensionless. At least three, and they must
        span three dimensions (rank 3), otherwise one body axis has no wheel authority.
    wheel_inertia_kg_m2 : (n,)
        Rotor inertia about the spin axis [kg m^2]. A scalar is broadcast.
    max_momentum_nms : float
        Per-wheel momentum limit [N m s], the saturation level.
    """

    axes: NDArray[np.float64]
    wheel_inertia_kg_m2: NDArray[np.float64]
    max_momentum_nms: float

    def __post_init__(self) -> None:
        axes = np.asarray(self.axes, dtype=float)
        if axes.ndim != 2 or axes.shape[1] != 3:
            raise ValueError(f"axes must have shape (n, 3), got shape {axes.shape}")
        if axes.shape[0] < 3:
            raise ValueError(f"at least 3 wheels are required, got {axes.shape[0]}")
        if not np.all(np.isfinite(axes)):
            raise ValueError("axes must be finite")
        norms = np.linalg.norm(axes, axis=1)
        if np.any(np.abs(norms - 1.0) > 1e-8):
            raise ValueError(
                "every wheel axis must be a unit vector; got norms "
                f"{np.round(norms, 6).tolist()}"
            )
        if np.linalg.matrix_rank(axes) < 3:
            raise ValueError(
                "wheel axes must span three dimensions (rank 3); the given set has rank "
                f"{int(np.linalg.matrix_rank(axes))}, so at least one body axis has no "
                "wheel authority"
            )
        j = np.broadcast_to(
            np.asarray(self.wheel_inertia_kg_m2, dtype=float), (axes.shape[0],)
        ).astype(float)
        if np.any(j <= 0.0) or not np.all(np.isfinite(j)):
            raise ValueError(f"wheel_inertia_kg_m2 must be finite and > 0, got {j.tolist()}")
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "wheel_inertia_kg_m2", j)
        object.__setattr__(
            self, "max_momentum_nms", _v.positive(self.max_momentum_nms, "max_momentum_nms")
        )

    @property
    def n_wheels(self) -> int:
        """Number of wheels."""
        return int(self.axes.shape[0])

    @property
    def distribution_matrix(self) -> NDArray[np.float64]:
        """``A`` [3, n], the map from wheel momenta to body momentum, dimensionless."""
        return self.axes.T.copy()

    @property
    def null_basis(self) -> NDArray[np.float64]:
        """Orthonormal basis of ``null(A)``, shape ``(n, n - 3)``.

        Computed by SVD of the distribution matrix. Empty for a non-redundant array.
        """
        _, _, vt = np.linalg.svd(self.distribution_matrix)
        return vt[3:].T.copy()

    @property
    def guaranteed_body_envelope_nms(self) -> float:
        r"""Largest body momentum magnitude the array can hold in **every** direction
        under minimum-norm allocation [N m s].

        For a unit direction :math:`\hat{\mathbf{u}}` the minimum-norm wheel vector is
        :math:`\mathbf{A}^{+}\hat{\mathbf{u}}`, whose largest component is bounded by
        the largest row norm of :math:`\mathbf{A}^{+}`, with equality when
        :math:`\hat{\mathbf{u}}` is along that row. Hence

        .. math:: h_{env} = \frac{h_{max}}{\max_i \lVert (\mathbf{A}^{+})_i \rVert}

        exactly, with no sampling. For the default isotropic pyramid
        :math:`\mathbf{A}^{+} = \tfrac34 \mathbf{A}^{\!\top}`, so
        :math:`h_{env} = \tfrac43 h_{max}`, i.e. 0.0667 N m s for the default
        0.05 N m s wheels.

        This is a **conservative** envelope: it assumes minimum-norm allocation. The true
        reachable set is the zonotope :math:`\{\mathbf{A}\mathbf{h} : |h_i| \le
        h_{max}\}`, which is larger and not a ball. Saturation fractions quoted
        throughout this package are referred to this conservative number, which is the
        one a sizing exercise should use.
        """
        pinv = np.linalg.pinv(self.distribution_matrix)
        return float(self.max_momentum_nms / np.max(np.linalg.norm(pinv, axis=1)))

    def body_momentum(self, wheel_momentum_nms: ArrayLike) -> NDArray[np.float64]:
        """Body momentum stored by the array [N m s], ``h_body = A h_wheel``."""
        h = np.asarray(wheel_momentum_nms, dtype=float)
        if h.shape[-1] != self.n_wheels:
            raise ValueError(
                f"wheel_momentum_nms must have trailing dimension {self.n_wheels}, "
                f"got shape {h.shape}"
            )
        return h @ self.axes

    def speeds_rad_s(self, wheel_momentum_nms: ArrayLike) -> NDArray[np.float64]:
        """Wheel speeds [rad s^-1], ``omega_i = h_i / J_i``."""
        h = np.asarray(wheel_momentum_nms, dtype=float)
        if h.shape[-1] != self.n_wheels:
            raise ValueError(
                f"wheel_momentum_nms must have trailing dimension {self.n_wheels}, "
                f"got shape {h.shape}"
            )
        return h / self.wheel_inertia_kg_m2

    def minimum_norm_allocation(self, h_body_nms: ArrayLike) -> NDArray[np.float64]:
        """Minimum-two-norm wheel momenta for a body request [N m s], ``A^+ h_body``.

        The Moore-Penrose pseudo-inverse solution, which is what most flight software
        uses when it has nothing else to optimise. It ignores zero speed entirely: for a
        symmetric array a body request near the null direction drives several wheels
        straight through zero.
        """
        h = _v.as_vector3(h_body_nms, "h_body_nms")
        return np.linalg.pinv(self.distribution_matrix) @ h

    def saturation_fraction(self, wheel_momentum_nms: ArrayLike) -> float:
        """Largest wheel momentum as a fraction of the per-wheel limit, dimensionless."""
        h = np.asarray(wheel_momentum_nms, dtype=float)
        return float(np.max(np.abs(h)) / self.max_momentum_nms)

    def allocate(
        self,
        h_body_nms: ArrayLike,
        avoid_zero_speed: bool = True,
        envelope_fraction: float = 1.0,
    ) -> Allocation:
        """Distribute a body momentum request, optionally biased away from zero speed.

        With ``avoid_zero_speed`` the null-space coefficient is chosen to **maximise the
        smallest wheel momentum magnitude** subject to every wheel staying inside
        ``max_momentum_nms``. For a one-dimensional null space (the four-wheel case) the
        objective ``min_i |h_i + alpha n_i|`` is a minimum of V-shaped piecewise-linear
        functions of the single coefficient ``alpha``, so its maximiser lies either at a
        crossing of two of those functions or at an endpoint of the feasible interval.
        Both candidate sets are finite and are enumerated exactly; there is no line
        search and no tolerance.

        For a null space of dimension two or more the same exact one-dimensional solver
        is applied to each basis direction in turn, cycling until no direction improves.
        That is a documented heuristic, not a proof of optimality; no four-wheel or
        three-wheel array reaches it.

        ``envelope_fraction`` in ``(0, 1]`` shrinks the box the search may use, to
        ``envelope_fraction * max_momentum_nms``. It exists because the unconstrained
        objective always parks wheels on the saturation boundary: maximising the smallest
        wheel momentum and preserving saturation margin are opposed, and this parameter
        is where the user states the trade rather than the library assuming it.

        **Limitation, stated because it is visible in the output.** The coefficient is
        chosen afresh on every call with no memory of the previous one, and the maximiser
        can switch between symmetric branches as the request moves. When it does, the
        commanded wheel momenta jump: on the three-orbit trajectory in
        ``validation/wheel_allocation.py`` the largest single-sample step is 0.0951 N m s
        at full envelope against 0.000060 N m s for minimum-norm allocation, i.e. nearly
        two whole wheel limits in one 7.9 s sample. Any flight use must rate-limit the
        null coefficient or add hysteresis to the branch choice. This function does not,
        so that the behaviour cannot be adopted unnoticed.

        Returns an :class:`Allocation`. When no feasible ``alpha`` exists the
        minimum-norm solution is returned with ``feasible=False``.
        """
        frac = _v.in_range(envelope_fraction, "envelope_fraction", 1e-6, 1.0)
        h_min_norm = self.minimum_norm_allocation(h_body_nms)
        basis = self.null_basis
        n_null = basis.shape[1]
        coeffs = np.zeros(max(n_null, 0))
        if not avoid_zero_speed or n_null == 0:
            return self._finish(h_min_norm, coeffs)
        h = h_min_norm.copy()
        for _ in range(max(1, 3 * n_null)):
            improved = False
            for k in range(n_null):
                alpha, ok = self._best_alpha(h, basis[:, k], frac)
                if not ok:
                    continue
                if alpha != 0.0:
                    h = h + alpha * basis[:, k]
                    coeffs[k] += alpha
                    improved = True
            if not improved:
                break
        return self._finish(h, coeffs)

    def _finish(
        self, h: NDArray[np.float64], coeffs: NDArray[np.float64]
    ) -> Allocation:
        mag = np.abs(h)
        return Allocation(
            wheel_momentum_nms=h,
            null_coefficients=coeffs,
            min_abs_momentum_nms=float(mag.min()),
            max_abs_momentum_nms=float(mag.max()),
            feasible=bool(mag.max() <= self.max_momentum_nms * (1.0 + 1e-12)),
        )

    def _best_alpha(
        self, p: NDArray[np.float64], nvec: NDArray[np.float64], envelope_fraction: float = 1.0
    ) -> tuple[float, bool]:
        """Exact maximiser of ``min_i |p_i + a n_i|`` over the feasible interval in ``a``."""
        hmax = self.max_momentum_nms * envelope_fraction
        lo, hi = -np.inf, np.inf
        for pi, ni in zip(p, nvec, strict=True):
            if abs(ni) < 1e-15:
                if abs(pi) > hmax:
                    return 0.0, False
                continue
            a1, a2 = (-hmax - pi) / ni, (hmax - pi) / ni
            lo, hi = max(lo, min(a1, a2)), min(hi, max(a1, a2))
        if lo > hi:
            return 0.0, False
        cands = [lo, hi, 0.0] if lo <= 0.0 <= hi else [lo, hi]
        n = p.size
        for i in range(n):
            for j in range(i + 1, n):
                for sgn in (1.0, -1.0):
                    denom = nvec[i] - sgn * nvec[j]
                    if abs(denom) < 1e-15:
                        continue
                    a = (sgn * p[j] - p[i]) / denom
                    if lo <= a <= hi:
                        cands.append(float(a))
        best_a, best_val = 0.0, -np.inf
        for a in cands:
            val = float(np.min(np.abs(p + a * nvec)))
            if val > best_val + 1e-18:
                best_val, best_a = val, float(a)
        return best_a, True


def pyramid_four(
    half_angle_rad: float = PYRAMID_ISOTROPIC_HALF_ANGLE_RAD,
    wheel_inertia_kg_m2: float = 1.0e-3,
    max_momentum_nms: float = 0.05,
) -> WheelArray:
    """Four wheels on a cone of half-angle ``half_angle_rad`` about body z.

    Axes ``(sin b cos t_k, sin b sin t_k, cos b)`` with ``t_k = 0, 90, 180, 270 deg``.
    The default half-angle 54.7356 deg makes the array momentum-isotropic; see
    :data:`PYRAMID_ISOTROPIC_HALF_ANGLE_RAD`. Default rotor inertia 1e-3 kg m^2 and
    0.05 N m s limit are smallsat-class values (they correspond to 50 rad s^-1, about
    478 rpm, at saturation); pass your own.
    """
    b = _v.in_range(half_angle_rad, "half_angle_rad", 1e-6, np.pi / 2 - 1e-6)
    t = np.arange(4) * (np.pi / 2.0)
    axes = np.column_stack([np.sin(b) * np.cos(t), np.sin(b) * np.sin(t), np.full(4, np.cos(b))])
    return WheelArray(axes=axes, wheel_inertia_kg_m2=wheel_inertia_kg_m2,
                      max_momentum_nms=max_momentum_nms)


def tetrahedral_four(
    wheel_inertia_kg_m2: float = 1.0e-3, max_momentum_nms: float = 0.05
) -> WheelArray:
    """Four wheels along the vertices of a regular tetrahedron, ``(+-1, +-1, +-1)/sqrt(3)``
    with an even number of minus signs. Momentum-isotropic like the default pyramid, with
    a different null direction."""
    raw = np.array([[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]])
    return WheelArray(
        axes=raw / np.sqrt(3.0),
        wheel_inertia_kg_m2=wheel_inertia_kg_m2,
        max_momentum_nms=max_momentum_nms,
    )


def orthogonal_three(
    wheel_inertia_kg_m2: float = 1.0e-3, max_momentum_nms: float = 0.05
) -> WheelArray:
    """Three wheels along the body axes. No redundancy, therefore **no** null space and no
    zero-speed avoidance: every wheel momentum is fixed by the body request. Provided so
    that the benefit of redundancy can be measured against its absence."""
    return WheelArray(
        axes=np.eye(3),
        wheel_inertia_kg_m2=wheel_inertia_kg_m2,
        max_momentum_nms=max_momentum_nms,
    )


def count_zero_crossings(
    wheel_momentum_history: ArrayLike, deadband_nms: float = 0.0
) -> NDArray[np.int64]:
    """Number of sign changes of each wheel's momentum over a history.

    ``wheel_momentum_history`` has shape ``(N, n_wheels)`` [N m s]. Samples whose
    magnitude is below ``deadband_nms`` are treated as unsigned and do not by themselves
    start or end a crossing; a crossing is counted when the sign changes between the last
    two samples that were outside the deadband. With the default zero deadband this is
    the plain count of sign changes.
    """
    h = np.asarray(wheel_momentum_history, dtype=float)
    if h.ndim != 2:
        raise ValueError(f"wheel_momentum_history must have shape (N, n), got {h.shape}")
    db = _v.non_negative(deadband_nms, "deadband_nms")
    counts = np.zeros(h.shape[1], dtype=np.int64)
    for k in range(h.shape[1]):
        col = h[:, k]
        outside = col[np.abs(col) > db]
        if outside.size < 2:
            continue
        counts[k] = int(np.count_nonzero(np.diff(np.sign(outside)) != 0))
    return counts

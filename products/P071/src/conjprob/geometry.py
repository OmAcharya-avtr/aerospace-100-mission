"""Close-approach geometry: time of closest approach, encounter plane, projection.

All quantities are SI: positions in metres, velocities in metres per second,
covariances in metres squared. Nothing in this module uses kilometres.

The encounter-plane (conjunction-plane) construction follows the short-encounter
formulation of Foster & Estes (1992), NASA/JSC-25898, as described in
Alfriend et al. (1999), "Probability of collision error analysis",
Space Debris 1(1), 21-35, and in Vallado (2013), *Fundamentals of Astrodynamics
and Applications*, 4th ed., chapter 11.

Assumptions, which are the product's validity regime and are checked elsewhere
in this package rather than assumed here:

* relative motion is rectilinear and uniform over the encounter
  (``rho(t) = rho_0 + v_rel * t``);
* the combined position covariance is constant over the encounter;
* velocity uncertainty is neglected;
* the encounter is short compared with the orbital period, so that the
  out-of-plane (relative-velocity) direction may be marginalised over an
  infinite time interval.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

_EYE3 = np.eye(3)


@dataclass(frozen=True)
class ObjectState:
    """Inertial state of one object with its 3x3 position covariance.

    Parameters
    ----------
    position_m
        Position, metres, shape ``(3,)``.
    velocity_mps
        Velocity, metres per second, shape ``(3,)``.
    covariance_m2
        Position covariance, metres squared, shape ``(3, 3)``. Must be
        symmetric to within ``1e-9`` relative and positive semi-definite.
    name
        Free-text label used only in report text.
    """

    position_m: NDArray[np.float64]
    velocity_mps: NDArray[np.float64]
    covariance_m2: NDArray[np.float64]
    name: str = "object"

    def __post_init__(self) -> None:
        object.__setattr__(self, "position_m", _as_vec3(self.position_m, "position_m"))
        object.__setattr__(self, "velocity_mps", _as_vec3(self.velocity_mps, "velocity_mps"))
        object.__setattr__(self, "covariance_m2", _as_cov3(self.covariance_m2))


def _as_vec3(value: ArrayLike, label: str) -> NDArray[np.float64]:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.shape != (3,):
        raise ValueError(f"{label} must have exactly 3 components, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} must be finite, got {arr!r}")
    return arr


def _as_cov3(value: ArrayLike) -> NDArray[np.float64]:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3, 3):
        raise ValueError(f"covariance_m2 must have shape (3, 3), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("covariance_m2 must be finite")
    scale = max(float(np.max(np.abs(arr))), 1.0)
    if not np.allclose(arr, arr.T, rtol=0.0, atol=1e-9 * scale):
        raise ValueError("covariance_m2 must be symmetric to within 1e-9 relative")
    sym = 0.5 * (arr + arr.T)
    eigenvalues = np.linalg.eigvalsh(sym)
    if eigenvalues.min() < -1e-9 * scale:
        raise ValueError(
            "covariance_m2 must be positive semi-definite; smallest eigenvalue is "
            f"{eigenvalues.min():.6e} m^2"
        )
    return sym


def relative_state(
    primary: ObjectState, secondary: ObjectState
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(relative position m, relative velocity m/s, combined covariance m^2)``.

    The relative quantities are secondary minus primary. The combined covariance
    is the sum of the two position covariances, which is the correct covariance
    of the difference only if the two estimates are uncorrelated; that
    assumption is standard in conjunction assessment and is stated in the README
    as a limitation.
    """
    r = secondary.position_m - primary.position_m
    v = secondary.velocity_mps - primary.velocity_mps
    cov = primary.covariance_m2 + secondary.covariance_m2
    return r, v, cov


def time_of_closest_approach(r_rel_m: ArrayLike, v_rel_mps: ArrayLike) -> float:
    """Time of closest approach in seconds relative to the state epoch.

    Exact for rectilinear uniform relative motion: ``t = -(r . v) / |v|^2``,
    obtained by setting the derivative of ``|r + v t|^2`` to zero. Negative
    values mean the closest approach is before the epoch.

    Raises
    ------
    ValueError
        If the relative speed is zero, in which case no closest approach is
        defined.
    """
    r = _as_vec3(r_rel_m, "r_rel_m")
    v = _as_vec3(v_rel_mps, "v_rel_mps")
    speed_sq = float(v @ v)
    if speed_sq <= 0.0:
        raise ValueError("relative velocity is zero; time of closest approach is undefined")
    return -float(r @ v) / speed_sq


def miss_vector(r_rel_m: ArrayLike, v_rel_mps: ArrayLike) -> NDArray[np.float64]:
    """Relative position at closest approach, metres, shape ``(3,)``.

    Equal to the component of ``r`` perpendicular to ``v``, so it lies in the
    encounter plane exactly (up to rounding) rather than approximately.
    """
    r = _as_vec3(r_rel_m, "r_rel_m")
    v = _as_vec3(v_rel_mps, "v_rel_mps")
    t = time_of_closest_approach(r, v)
    return r + v * t


def encounter_plane_basis(v_rel_mps: ArrayLike) -> NDArray[np.float64]:
    """Orthonormal basis of the encounter plane, shape ``(3, 2)``.

    The two columns span the plane perpendicular to the relative velocity. The
    construction is deterministic: the Cartesian axis least aligned with the
    relative velocity is orthogonalised against it to give the first column,
    and the second column completes a right-handed set. Pc is invariant to the
    in-plane rotation, which ``tests/test_properties.py`` checks.
    """
    v = _as_vec3(v_rel_mps, "v_rel_mps")
    speed = float(np.linalg.norm(v))
    if speed <= 0.0:
        raise ValueError("relative velocity is zero; encounter plane is undefined")
    n = v / speed
    seed = _EYE3[int(np.argmin(np.abs(n)))]
    x = seed - (seed @ n) * n
    x /= np.linalg.norm(x)
    y = np.cross(n, x)
    return np.column_stack((x, y))


def project_covariance(covariance_m2: ArrayLike, basis: ArrayLike) -> NDArray[np.float64]:
    """Marginal 2x2 covariance in the encounter plane, metres squared.

    ``C_p = B^T C B`` with ``B`` the ``(3, 2)`` plane basis. For a Gaussian this
    is the exact marginal over the out-of-plane direction, which is why the
    2-D formulation is exact under rectilinear motion over an infinite time
    interval and only approximate otherwise.
    """
    cov = _as_cov3(covariance_m2)
    b = np.asarray(basis, dtype=float)
    if b.shape != (3, 2):
        raise ValueError(f"basis must have shape (3, 2), got {b.shape}")
    projected = b.T @ cov @ b
    return 0.5 * (projected + projected.T)


@dataclass(frozen=True)
class PrincipalFrame:
    """Encounter-plane geometry in the principal axes of the projected covariance.

    Attributes
    ----------
    sigma_x_m, sigma_y_m
        Principal standard deviations, metres, with ``sigma_x_m >= sigma_y_m``
        by construction.
    x_bar_m, y_bar_m
        Components of the miss vector on the principal axes, metres.
    rotation
        ``(2, 2)`` matrix whose columns are the principal axes expressed in the
        input in-plane basis.
    """

    sigma_x_m: float
    sigma_y_m: float
    x_bar_m: float
    y_bar_m: float
    rotation: NDArray[np.float64]

    @property
    def miss_distance_m(self) -> float:
        """Miss distance in the encounter plane, metres."""
        return float(np.hypot(self.x_bar_m, self.y_bar_m))

    @property
    def aspect_ratio(self) -> float:
        """``sigma_x / sigma_y``, dimensionless, always >= 1."""
        return float(self.sigma_x_m / self.sigma_y_m)


def principal_frame(covariance_2d_m2: ArrayLike, miss_2d_m: ArrayLike) -> PrincipalFrame:
    """Diagonalise the projected covariance and express the miss vector in it.

    The analytic Pc formulations of Foster & Estes (1992), Chan (1997, 2008)
    and Alfano (2005) are all written for a diagonal encounter-plane
    covariance, so this rotation is a prerequisite for all three.

    Raises
    ------
    ValueError
        If the projected covariance is not strictly positive definite, since
        a zero principal variance makes the Gaussian singular and Pc
        ill-defined.
    """
    cov = np.asarray(covariance_2d_m2, dtype=float)
    if cov.shape != (2, 2):
        raise ValueError(f"covariance_2d_m2 must have shape (2, 2), got {cov.shape}")
    sym = 0.5 * (cov + cov.T)
    eigenvalues, eigenvectors = np.linalg.eigh(sym)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    if eigenvalues[-1] <= 0.0:
        raise ValueError(
            "projected covariance must be positive definite; smallest eigenvalue is "
            f"{eigenvalues[-1]:.6e} m^2"
        )
    miss = np.asarray(miss_2d_m, dtype=float).reshape(-1)
    if miss.shape != (2,):
        raise ValueError(f"miss_2d_m must have 2 components, got shape {miss.shape}")
    components = eigenvectors.T @ miss
    return PrincipalFrame(
        sigma_x_m=float(np.sqrt(eigenvalues[0])),
        sigma_y_m=float(np.sqrt(eigenvalues[1])),
        x_bar_m=float(components[0]),
        y_bar_m=float(components[1]),
        rotation=eigenvectors,
    )


def mahalanobis_distance(covariance_2d_m2: ArrayLike, miss_2d_m: ArrayLike) -> float:
    """Mahalanobis distance of the miss vector in the encounter plane, dimensionless.

    ``sqrt(m^T C_p^{-1} m)``. This is the standard dimensionless measure of how
    improbable the miss is given the covariance, and it is one of the two
    analytic screens the learned classifier in :mod:`conjprob.screening` is
    scored against.
    """
    frame = principal_frame(covariance_2d_m2, miss_2d_m)
    return float(
        np.hypot(frame.x_bar_m / frame.sigma_x_m, frame.y_bar_m / frame.sigma_y_m)
    )


@dataclass(frozen=True)
class EncounterGeometry:
    """Everything the Pc methods need, computed once from the two states.

    Attributes
    ----------
    tca_s
        Time of closest approach, seconds from the state epoch.
    miss_distance_m
        Distance at closest approach, metres.
    relative_speed_mps
        Relative speed, metres per second, constant under the rectilinear
        assumption.
    miss_vector_m
        Relative position at closest approach, metres, shape ``(3,)``.
    relative_velocity_mps
        Relative velocity, metres per second, shape ``(3,)``.
    combined_covariance_m2
        Sum of the two 3x3 position covariances, metres squared.
    basis
        ``(3, 2)`` encounter-plane basis.
    covariance_2d_m2
        ``(2, 2)`` projected covariance, metres squared.
    miss_2d_m
        Miss vector in the plane basis, metres, shape ``(2,)``.
    frame
        Principal-axis description of the in-plane geometry.
    """

    tca_s: float
    miss_distance_m: float
    relative_speed_mps: float
    miss_vector_m: NDArray[np.float64]
    relative_velocity_mps: NDArray[np.float64]
    combined_covariance_m2: NDArray[np.float64]
    basis: NDArray[np.float64]
    covariance_2d_m2: NDArray[np.float64]
    miss_2d_m: NDArray[np.float64]
    frame: PrincipalFrame

    @property
    def mahalanobis(self) -> float:
        """Mahalanobis distance of the miss vector in the encounter plane."""
        return float(
            np.hypot(
                self.frame.x_bar_m / self.frame.sigma_x_m,
                self.frame.y_bar_m / self.frame.sigma_y_m,
            )
        )

    @property
    def sigma_effective_m(self) -> float:
        """Geometric mean of the two principal standard deviations, metres."""
        return float(np.sqrt(self.frame.sigma_x_m * self.frame.sigma_y_m))

    @property
    def crossing_time_s(self) -> float:
        """Encounter crossing time ``sigma_eff / |v_rel|``, seconds.

        The time the relative motion takes to traverse one effective standard
        deviation. Used to non-dimensionalise the curvature and window
        violations measured in :mod:`conjprob.nonlinear`.
        """
        return float(self.sigma_effective_m / self.relative_speed_mps)


def encounter_geometry(primary: ObjectState, secondary: ObjectState) -> EncounterGeometry:
    """Build the full encounter geometry from two states with covariance.

    Units: metres, metres per second, metres squared in, seconds and metres out.
    """
    r, v, cov = relative_state(primary, secondary)
    tca = time_of_closest_approach(r, v)
    miss3 = r + v * tca
    basis = encounter_plane_basis(v)
    cov2 = project_covariance(cov, basis)
    miss2 = basis.T @ miss3
    frame = principal_frame(cov2, miss2)
    return EncounterGeometry(
        tca_s=tca,
        miss_distance_m=float(np.linalg.norm(miss3)),
        relative_speed_mps=float(np.linalg.norm(v)),
        miss_vector_m=miss3,
        relative_velocity_mps=v,
        combined_covariance_m2=cov,
        basis=basis,
        covariance_2d_m2=cov2,
        miss_2d_m=miss2,
        frame=frame,
    )

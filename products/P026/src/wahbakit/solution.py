"""The value type every solver returns, plus the Wahba loss and gain.

Wahba's loss for unit vectors (Wahba 1965, *SIAM Review* **7**(3), 409):

    L(A) = (1/2) sum_i w_i |b_i - A r_i|^2
         = sum_i w_i - trace(A B^T),   B = sum_i w_i b_i r_i^T          (Eq. S1)

Because the weights are normalised to sum to one, ``L = 1 - gain`` and the
optimal gain equals the largest Davenport eigenvalue ``lambda_max <= 1``.
Dimensionless.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .conventions import angle_between_dcm, quat_canonical
from .observations import Observability, VectorObservations

__all__ = ["AttitudeSolution", "wahba_gain", "wahba_loss"]


def wahba_gain(dcm: ArrayLike, obs: VectorObservations) -> float:
    """Gain ``trace(A B^T)`` (Eq. S1).  Dimensionless, ``<= sum_i w_i = 1``."""
    a = np.asarray(dcm, dtype=float)
    if a.shape != (3, 3):
        raise ValueError(f"dcm must have shape (3, 3), got {a.shape}")
    return float(np.sum(a * obs.attitude_profile_matrix()))


def wahba_loss(dcm: ArrayLike, obs: VectorObservations) -> float:
    """Wahba loss ``L(A)`` of Eq. S1.  Dimensionless, ``>= 0``."""
    return float(np.sum(obs.weights)) - wahba_gain(dcm, obs)


@dataclass(frozen=True)
class AttitudeSolution:
    """An attitude estimate and everything needed to judge it.

    Attributes
    ----------
    dcm : ndarray, shape (3, 3)
        Attitude matrix ``A`` with ``b_i ~= A r_i`` (reference-to-body).
    quaternion : ndarray, shape (4,)
        Scalar-first ``[w, x, y, z]`` with ``A = dcm_from_quat(q)`` and
        ``w >= 0``.
    method : str
        ``"triad"``, ``"q-method"``, ``"quest"`` or ``"olae"``.
    loss : float
        Wahba loss Eq. S1, dimensionless.
    gain : float
        ``trace(A B^T)``, dimensionless.
    n_observations : int
    weights : ndarray, shape (N,)
        Normalised Wahba weights actually used.
    residual_angles_rad : ndarray, shape (N,)
        Angle between ``b_i`` and ``A r_i``, in rad.
    observability : Observability or None
        Result of the degeneracy test, when it was run.
    lambda_max : float or None
        Largest Davenport eigenvalue.  Present for ``q-method`` and ``quest``,
        ``None`` for ``triad`` and ``olae``, which never form ``K``.
    diagnostics : dict
        Method-specific numbers; see each solver's docstring.  Always present,
        possibly empty.
    """

    dcm: NDArray[np.float64]
    quaternion: NDArray[np.float64]
    method: str
    loss: float
    gain: float
    n_observations: int
    weights: NDArray[np.float64]
    residual_angles_rad: NDArray[np.float64]
    observability: Observability | None = None
    lambda_max: float | None = None
    diagnostics: dict[str, float] = field(default_factory=dict)

    @property
    def residual_angles_deg(self) -> NDArray[np.float64]:
        """Per-observation residual angles in degrees."""
        return np.degrees(self.residual_angles_rad)

    def rotate(self, reference_vectors: ArrayLike) -> NDArray[np.float64]:
        """Map reference-frame vectors into the body frame: ``v_body = A v_ref``."""
        v = np.atleast_2d(np.asarray(reference_vectors, dtype=float))
        if v.shape[1] != 3:
            raise ValueError(f"reference_vectors must be (N, 3), got {v.shape}")
        return v @ self.dcm.T

    def angle_to(self, other: AttitudeSolution | ArrayLike) -> float:
        """Rotation angle between this attitude and another, in **rad**."""
        other_dcm = other.dcm if isinstance(other, AttitudeSolution) else other
        return angle_between_dcm(self.dcm, other_dcm)

    def __repr__(self) -> str:
        q = np.array2string(self.quaternion, precision=6)
        return (
            f"AttitudeSolution(method={self.method!r}, n={self.n_observations}, "
            f"q={q}, loss={self.loss:.6e}, "
            f"max_residual={np.max(self.residual_angles_deg):.6f} deg)"
        )


def build_solution(
    dcm: NDArray[np.float64],
    method: str,
    obs: VectorObservations,
    *,
    observability: Observability | None = None,
    lambda_max: float | None = None,
    diagnostics: dict[str, float] | None = None,
    quaternion: NDArray[np.float64] | None = None,
) -> AttitudeSolution:
    """Assemble an :class:`AttitudeSolution` from a DCM.  Internal helper."""
    from .conventions import quat_from_dcm

    q = quat_canonical(quaternion) if quaternion is not None else quat_from_dcm(dcm)
    gain = wahba_gain(dcm, obs)
    return AttitudeSolution(
        dcm=dcm,
        quaternion=q,
        method=method,
        loss=float(np.sum(obs.weights)) - gain,
        gain=gain,
        n_observations=obs.n,
        weights=obs.weights,
        residual_angles_rad=obs.residual_angles(dcm),
        observability=observability,
        lambda_max=lambda_max,
        diagnostics=dict(diagnostics or {}),
    )

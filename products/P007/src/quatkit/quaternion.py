"""Unit-quaternion attitude class (scalar-first ``[w, x, y, z]``).

Normalization policy (documented, enforced, tested)
---------------------------------------------------
``Quaternion`` objects always hold a **unit** quaternion:

* components within ``NORM_TOL = 1e-6`` of unit norm are accepted and silently
  re-normalized to machine precision;
* components further from unit norm **raise ValueError** unless the caller
  passes ``normalize=True`` explicitly.

Consequently every rotation performed through this class is guaranteed
norm-preserving; there is no way to feed a non-unit quaternion into
``rotate`` without either an explicit opt-in normalization or an exception.

Convention: Hamilton product (i j = k), active rotation
``v' = q ⊗ [0, v] ⊗ q*``. See :mod:`quatkit.core` for references
(Markley & Crassidis 2014; Shuster 1993).
"""

from __future__ import annotations

import numpy as np

from . import conversions as _conv
from . import core as _core

__all__ = ["NORM_TOL", "Quaternion"]

#: Accepted deviation from unit norm before construction raises (dimensionless).
NORM_TOL = 1e-6


class Quaternion:
    """Immutable unit quaternion, scalar-first ``[w, x, y, z]``, Hamilton convention.

    Represents an active rotation: ``q.rotate(v)`` returns
    ``q ⊗ [0, v] ⊗ q*``. Composition ``q2 * q1`` means "rotate by q1 first,
    then by q2".
    """

    __slots__ = ("_q",)

    def __init__(
        self, w: float, x: float, y: float, z: float, *, normalize: bool = False
    ) -> None:
        q = np.array([w, x, y, z], dtype=float)
        if not np.all(np.isfinite(q)):
            raise ValueError(f"quaternion components must be finite, got {q}")
        n = float(np.linalg.norm(q))
        if n < 1e-12:
            raise ValueError("zero quaternion cannot represent a rotation")
        if abs(n - 1.0) > NORM_TOL and not normalize:
            raise ValueError(
                f"|q| = {n:.6g} deviates from 1 by more than NORM_TOL={NORM_TOL}; "
                "pass normalize=True to accept and normalize non-unit input "
                "(documented normalize-or-raise policy)"
            )
        self._q = q / n
        self._q.setflags(write=False)

    # -- constructors -------------------------------------------------------
    @classmethod
    def identity(cls) -> Quaternion:
        """Identity (no-rotation) quaternion [1, 0, 0, 0]."""
        return cls(1.0, 0.0, 0.0, 0.0)

    @classmethod
    def from_array(cls, q: np.ndarray, *, normalize: bool = False) -> Quaternion:
        """From a length-4 array-like, scalar-first [w, x, y, z]. Same norm policy."""
        q = np.asarray(q, dtype=float)
        if q.shape != (4,):
            raise ValueError(f"expected shape (4,) scalar-first array, got {q.shape}")
        return cls(q[0], q[1], q[2], q[3], normalize=normalize)

    @classmethod
    def from_axis_angle(cls, axis: np.ndarray, angle: float) -> Quaternion:
        """From rotation axis (any nonzero 3-vector) and angle [rad]."""
        return cls.from_array(_conv.axis_angle_to_quat(axis, angle))

    @classmethod
    def from_dcm(cls, dcm: np.ndarray) -> Quaternion:
        """From a 3x3 rotation matrix (active convention, v' = R v)."""
        return cls.from_array(_conv.dcm_to_quat(dcm))

    @classmethod
    def from_euler_zyx(cls, yaw: float, pitch: float, roll: float) -> Quaternion:
        """From aerospace ZYX (yaw-pitch-roll) intrinsic Euler angles [rad]."""
        return cls.from_array(_conv.euler_zyx_to_quat(yaw, pitch, roll))

    @classmethod
    def from_rodrigues(cls, g: np.ndarray) -> Quaternion:
        """From a Rodrigues (Gibbs) vector g = â tan(θ/2)."""
        return cls.from_array(_conv.rodrigues_to_quat(g))

    @classmethod
    def from_mrp(cls, p: np.ndarray) -> Quaternion:
        """From Modified Rodrigues Parameters p = â tan(θ/4)."""
        return cls.from_array(_conv.mrp_to_quat(p))

    @classmethod
    def exp(cls, rotvec: np.ndarray) -> Quaternion:
        """Exponential map: rotation vector [rad] -> unit quaternion."""
        return cls.from_array(_core.quat_exp(rotvec))

    # -- accessors -----------------------------------------------------------
    @property
    def w(self) -> float:
        """Scalar part (first component)."""
        return float(self._q[0])

    @property
    def x(self) -> float:
        return float(self._q[1])

    @property
    def y(self) -> float:
        return float(self._q[2])

    @property
    def z(self) -> float:
        return float(self._q[3])

    @property
    def vec(self) -> np.ndarray:
        """Vector part (x, y, z) as a copy."""
        return self._q[1:].copy()

    def as_array(self) -> np.ndarray:
        """Return a copy of the components, scalar-first [w, x, y, z]."""
        return self._q.copy()

    @property
    def norm(self) -> float:
        """Always 1.0 to machine precision (class invariant)."""
        return float(np.linalg.norm(self._q))

    # -- algebra --------------------------------------------------------------
    def __mul__(self, other: Quaternion) -> Quaternion:
        """Hamilton product self ⊗ other ("rotate by other first, then self")."""
        if not isinstance(other, Quaternion):
            return NotImplemented
        return Quaternion.from_array(
            _core.quat_multiply(self._q, other._q), normalize=True
        )

    def conjugate(self) -> Quaternion:
        """Conjugate [w, -x, -y, -z] — the inverse rotation for unit quaternions."""
        return Quaternion.from_array(_core.quat_conjugate(self._q))

    def inverse(self) -> Quaternion:
        """Inverse rotation; identical to conjugate() for unit quaternions."""
        return self.conjugate()

    def normalized(self) -> Quaternion:
        """Return a freshly normalized copy (class invariant keeps |q| = 1 anyway)."""
        return Quaternion.from_array(_core.quat_normalize(self._q))

    def rotate(self, v: np.ndarray) -> np.ndarray:
        """Actively rotate vector(s) v (..., 3): v' = q ⊗ [0, v] ⊗ q*.

        Norm-preserving by the class unit-norm invariant. Units of v preserved.
        """
        return _core.quat_rotate(self._q, v)

    def log(self) -> np.ndarray:
        """Logarithmic map: minimal rotation vector [rad], |result| in [0, π]."""
        return _core.quat_log(self._q)

    def slerp(self, other: Quaternion, t: float | np.ndarray) -> Quaternion | np.ndarray:
        """SLERP from self (t=0) to other (t=1) along the shortest arc.

        Scalar t returns a Quaternion; an array of t values returns an
        (n, 4) scalar-first array.
        """
        out = _core.quat_slerp(self._q, other._q, t)
        if out.ndim == 1:
            return Quaternion.from_array(out, normalize=True)
        return out

    # -- conversions -----------------------------------------------------------
    def to_dcm(self) -> np.ndarray:
        """3x3 rotation matrix R with v' = R v (matches scipy Rotation.as_matrix)."""
        return _conv.quat_to_dcm(self._q)

    def to_euler_zyx(self) -> np.ndarray:
        """[yaw, pitch, roll] in radians, ZYX aerospace sequence.

        Emits GimbalLockWarning near pitch = ±90° (see quatkit.conversions).
        """
        return _conv.quat_to_euler_zyx(self._q)

    def to_axis_angle(self) -> tuple[np.ndarray, float]:
        """(unit axis, angle [rad] in [0, π]), minimal-angle representation."""
        return _conv.quat_to_axis_angle(self._q)

    def to_rodrigues(self) -> np.ndarray:
        """Rodrigues (Gibbs) vector; raises ValueError at 180° rotations."""
        return _conv.quat_to_rodrigues(self._q)

    def to_mrp(self) -> np.ndarray:
        """Modified Rodrigues Parameters (principal set, |p| <= 1)."""
        return _conv.quat_to_mrp(self._q)

    # -- comparison / repr -------------------------------------------------------
    def isclose(self, other: Quaternion, atol: float = 1e-9) -> bool:
        """True if both represent the same rotation within atol [rad].

        Accounts for the q ≡ -q double cover by comparing rotation angle.
        """
        dq = _core.quat_multiply(_core.quat_conjugate(self._q), other._q)
        angle = 2.0 * np.arctan2(np.linalg.norm(dq[1:]), abs(dq[0]))
        return bool(angle <= atol)

    def __repr__(self) -> str:
        w, x, y, z = self._q
        return f"Quaternion(w={w:.9g}, x={x:.9g}, y={y:.9g}, z={z:.9g})"

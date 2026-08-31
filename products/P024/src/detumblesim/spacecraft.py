"""Spacecraft inertia and magnetorquer hardware models."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray


def inertia_from_diagonal(ixx: float, iyy: float, izz: float) -> NDArray[np.float64]:
    """Diagonal inertia tensor [kg m^2] with a triangle-inequality check.

    A physically realisable rigid body must satisfy the triangle inequalities
    on its principal moments (Wertz 1978, sec. 15.1):
    ``Ixx + Iyy >= Izz`` and cyclic permutations.

    Raises
    ------
    ValueError
        If any moment is non-positive or the triangle inequalities fail.
    """
    vals = (float(ixx), float(iyy), float(izz))
    if any(v <= 0.0 or not np.isfinite(v) for v in vals):
        raise ValueError(f"principal moments must be positive and finite, got {vals}")
    a, b, c = vals
    if a + b < c or b + c < a or c + a < b:
        raise ValueError(
            "principal moments violate the triangle inequality "
            f"(Ixx+Iyy>=Izz and cyclic): {vals}"
        )
    return np.diag(vals)


def validate_inertia(inertia: ArrayLike) -> NDArray[np.float64]:
    """Return ``inertia`` as a validated symmetric positive-definite 3x3 array."""
    j = np.asarray(inertia, dtype=float)
    if j.shape != (3, 3):
        raise ValueError(f"inertia must have shape (3, 3), got {j.shape}")
    if not np.all(np.isfinite(j)):
        raise ValueError("inertia contains non-finite entries")
    if not np.allclose(j, j.T, rtol=0.0, atol=1e-12 * max(1.0, np.abs(j).max())):
        raise ValueError("inertia must be symmetric")
    eigs = np.linalg.eigvalsh(j)
    if eigs.min() <= 0.0:
        raise ValueError(f"inertia must be positive definite, eigenvalues {eigs}")
    return j


@dataclass(frozen=True)
class Magnetorquer:
    """Three-axis magnetorquer set with per-axis dipole limits.

    Parameters
    ----------
    max_dipole_am2 : ndarray, shape (3,)
        Per-axis magnetic dipole limit [A m^2].  All entries must be positive.
        Commanded dipoles are clipped **per axis** (the physical limit is a
        per-coil current limit, so the achievable set is a box, not a ball).
    """

    max_dipole_am2: NDArray[np.float64] = field(
        default_factory=lambda: np.array([0.2, 0.2, 0.2])
    )

    def __post_init__(self) -> None:
        m = np.asarray(self.max_dipole_am2, dtype=float)
        if m.shape != (3,):
            raise ValueError(f"max_dipole_am2 must have shape (3,), got {m.shape}")
        if not np.all(np.isfinite(m)) or np.any(m <= 0.0):
            raise ValueError(f"max_dipole_am2 entries must be positive, got {m}")
        object.__setattr__(self, "max_dipole_am2", m)

    @classmethod
    def isotropic(cls, max_dipole_am2: float) -> Magnetorquer:
        """Convenience constructor for an equal per-axis limit [A m^2]."""
        if not np.isfinite(max_dipole_am2) or max_dipole_am2 <= 0.0:
            raise ValueError(
                f"max_dipole_am2 must be positive, got {max_dipole_am2}"
            )
        return cls(np.full(3, float(max_dipole_am2)))

    def saturate(self, dipole_am2: ArrayLike) -> tuple[NDArray[np.float64], bool]:
        """Clip a commanded dipole to the per-axis box.

        Returns
        -------
        (clipped, saturated)
            ``clipped`` [A m^2] and a flag that is True if any axis was clipped.
        """
        m = np.asarray(dipole_am2, dtype=float)
        if m.shape != (3,):
            raise ValueError(f"dipole must have shape (3,), got {m.shape}")
        lim = self.max_dipole_am2
        clipped = np.clip(m, -lim, lim)
        return clipped, bool(np.any(np.abs(m) > lim))

    @property
    def max_norm_am2(self) -> float:
        """Largest achievable dipole magnitude (box corner) [A m^2]."""
        return float(np.linalg.norm(self.max_dipole_am2))

"""Deformable-mirror model: influence functions, stroke limits, fitting.

The mirror is represented by ``N_act`` actuators on a square grid.  The surface
produced by a command vector ``c`` is the linear superposition

    phi_DM(r) = sum_k c_k * I_k(r)                    [rad of phase]

with ``I_k`` the **influence function** of actuator ``k``.  Linear
superposition is the standard first-order DM model; real mirrors show
hysteresis and mild nonlinearity that this model does not represent (see
Limitations in the README).

Influence function
------------------
A Gaussian influence function is used, parameterised by the inter-actuator
*coupling* ``c_pitch`` — the fraction of an actuator's own stroke that appears
at its nearest neighbour::

    I(r) = exp( ln(c_pitch) * (r / d_act)^2 )

so that ``I(0) = 1`` and ``I(d_act) = c_pitch`` exactly.  Typical continuous
face-sheet piezo mirrors have 10-25 % coupling; the default here is 15 %.
Source for the Gaussian model and typical coupling values: R. K. Tyson,
*Principles of Adaptive Optics*, 3rd ed., CRC Press (2011), Ch. 5; Hardy
(1998), Sec. 6.2.

Units and sign
--------------
Commands and the resulting ``phi_DM`` are in **radians of phase** referred to
the sensing wavelength.  For a mirror in reflection the mechanical surface
displacement is ``phi * lambda / (4 pi)`` — a factor two smaller than the OPD,
because the beam traverses the sag twice.  ``stroke_rad`` is therefore a limit
on the *phase* the actuator can impose.

Fitting error
-------------
Correcting turbulence with a DM of actuator pitch ``d_act`` leaves a residual

    sigma^2_fit = a_F * (d_act / r0)^(5/3)              [rad^2]

with ``a_F`` between about 0.28 (continuous face-sheet, Hudgin 1977) and 0.34
(segmented / Gaussian influence functions).  See
:mod:`waveforge.errorbudget`; the coefficient measured for *this* DM model is
reported in ``validation/VALIDATION.md`` rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .pupil import PupilGrid, piston_removed

__all__ = ["DeformableMirror"]


@dataclass
class DeformableMirror:
    """Square-grid deformable mirror with Gaussian influence functions.

    Parameters
    ----------
    pupil:
        Pupil sampling grid.
    n_act:
        Actuators across the pupil diameter, ``>= 2``.  The actuator pitch is
        ``d_act = D / (n_act - 1)`` so that the outermost actuators sit on the
        pupil rim.
    coupling:
        Nearest-neighbour coupling in ``(0, 1)``; default 0.15.
    stroke_rad:
        Symmetric command limit ``|c_k| <= stroke_rad`` [rad of phase], or
        ``inf`` for an unlimited mirror.
    margin_actuators:
        Rings of actuators placed outside the pupil edge, ``>= 0``.  One ring
        (the default) is normal practice so that the edge is controllable.
    """

    pupil: PupilGrid
    n_act: int = 9
    coupling: float = 0.15
    stroke_rad: float = float("inf")
    margin_actuators: int = 1
    _influence: np.ndarray = field(init=False, repr=False)
    _positions: np.ndarray = field(init=False, repr=False)
    _factorisations: dict = field(init=False, default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if int(self.n_act) != self.n_act or self.n_act < 2:
            raise ValueError(f"n_act must be an integer >= 2, got {self.n_act!r}")
        if not (0.0 < self.coupling < 1.0):
            raise ValueError(f"coupling must lie in (0, 1), got {self.coupling!r}")
        if np.isnan(self.stroke_rad) or self.stroke_rad <= 0.0:
            raise ValueError(f"stroke_rad must be > 0, got {self.stroke_rad!r}")
        if int(self.margin_actuators) != self.margin_actuators or self.margin_actuators < 0:
            raise ValueError(
                f"margin_actuators must be a non-negative integer, got {self.margin_actuators!r}"
            )
        self._build()

    @property
    def pitch_m(self) -> float:
        """Inter-actuator pitch ``d_act = D / (n_act - 1)`` [m]."""
        return self.pupil.diameter_m / (self.n_act - 1)

    @property
    def positions_m(self) -> np.ndarray:
        """Actuator positions, shape ``(n_actuators, 2)`` in metres."""
        return self._positions

    @property
    def n_actuators(self) -> int:
        """Total actuator count including margin rings."""
        return self._influence.shape[0]

    @property
    def influence_matrix(self) -> np.ndarray:
        """``(n_actuators, pupil.n_valid)`` influence functions, dimensionless."""
        return self._influence

    def _build(self) -> None:
        pitch = self.pitch_m
        k = self.n_act + 2 * self.margin_actuators
        axis = (np.arange(k) - (k - 1) / 2.0) * pitch
        ax, ay = np.meshgrid(axis, axis, indexing="xy")
        self._positions = np.stack([ax.ravel(), ay.ravel()], axis=1)
        x, y = self.pupil.coords_m()
        xm, ym = x[self.pupil.mask], y[self.pupil.mask]
        log_c = np.log(self.coupling)
        r2 = (xm[None, :] - self._positions[:, 0:1]) ** 2 + (
            ym[None, :] - self._positions[:, 1:2]
        ) ** 2
        self._influence = np.exp(log_c * r2 / pitch**2)

    # -- surface ----------------------------------------------------------
    def surface(self, commands: np.ndarray, *, remove_piston: bool = True) -> np.ndarray:
        """Phase map produced by ``commands`` [rad], shape ``(n_pix, n_pix)``.

        Values outside the pupil are zero.
        """
        commands = np.asarray(commands, dtype=float)
        if commands.shape != (self.n_actuators,):
            raise ValueError(
                f"commands must have shape ({self.n_actuators},), got {commands.shape}"
            )
        flat = commands @ self._influence
        out = np.zeros((self.pupil.n_pix, self.pupil.n_pix), dtype=float)
        out[self.pupil.mask] = flat
        return piston_removed(out, self.pupil.mask) if remove_piston else out

    def clip(self, commands: np.ndarray) -> tuple[np.ndarray, float]:
        """Apply the stroke limit.

        Returns
        -------
        (clipped, saturated_fraction):
            The clipped command vector and the fraction of actuators that hit
            the limit this call, in ``[0, 1]``.
        """
        commands = np.asarray(commands, dtype=float)
        if commands.shape != (self.n_actuators,):
            raise ValueError(
                f"commands must have shape ({self.n_actuators},), got {commands.shape}"
            )
        if np.isinf(self.stroke_rad):
            return commands.copy(), 0.0
        saturated = np.abs(commands) > self.stroke_rad
        return (
            np.clip(commands, -self.stroke_rad, self.stroke_rad),
            float(np.count_nonzero(saturated)) / self.n_actuators,
        )

    # -- fitting ----------------------------------------------------------
    def fit(self, phase: np.ndarray, regularisation: float = 1e-9) -> np.ndarray:
        """Least-squares command vector that best reproduces ``phase``.

        Solves ``min_c || I^T c - phi ||^2 + mu ||c||^2`` over the illuminated
        pupil.  ``regularisation`` (``mu``, relative to the largest singular
        value squared) controls the unavoidable ill-conditioning coming from
        actuators outside the pupil, which are only weakly observed.

        This is the *best-case* fitting performance of the mirror: the residual
        it leaves is the DM fitting error.
        """
        phase = np.asarray(phase, dtype=float)
        if phase.shape != (self.pupil.n_pix, self.pupil.n_pix):
            raise ValueError(
                f"phase shape {phase.shape} does not match pupil "
                f"({self.pupil.n_pix}, {self.pupil.n_pix})"
            )
        if not np.isfinite(regularisation) or regularisation < 0.0:
            raise ValueError(f"regularisation must be finite and >= 0, got {regularisation!r}")
        target = phase[self.pupil.mask]
        # The normal-equations matrix depends only on the geometry, so its
        # Cholesky factor is computed once per regularisation value and reused;
        # a fit then costs one back-substitution instead of a full solve.
        key = float(regularisation)
        factor = self._factorisations.get(key)
        if factor is None:
            a = self._influence @ self._influence.T
            scale = float(np.trace(a)) / self.n_actuators
            a = a + regularisation * scale * np.eye(self.n_actuators)
            factor = cho_factor(a, lower=True)
            self._factorisations[key] = factor
        return cho_solve(factor, self._influence @ target)

    def fitting_residual(self, phase: np.ndarray, regularisation: float = 1e-9) -> np.ndarray:
        """``phase - surface(fit(phase))`` over the pupil, piston removed [rad]."""
        commands = self.fit(phase, regularisation)
        return piston_removed(phase - self.surface(commands), self.pupil.mask)

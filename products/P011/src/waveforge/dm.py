"""Deformable-mirror influence-function model with finite actuator stroke.

Model
-----
A continuous-facesheet DM is represented by a set of actuators on a square grid
of pitch ``d_act`` [m], each contributing a fixed **influence function** scaled
by its command. The mirror-induced phase is linear in the commands:

```
phi_DM(x, y) = sum_a c_a * F(|r - r_a| / d_act)          [rad]
```

The influence function used here is the standard Gaussian model with a
specified inter-actuator coupling ``c_coupling`` -- the fraction of an
actuator's own stroke that appears at the neighbouring actuator:

```
F(u) = exp( ln(c_coupling) * u^2 ),      F(0) = 1,  F(1) = c_coupling
```

Typical measured coupling for continuous-facesheet mirrors is 5-20 %
(Hardy, J. W. 1998, *Adaptive Optics for Astronomical Telescopes*, Oxford
University Press, ch. 6, discusses facesheet influence functions and coupling);
15 % is the default here. *Assumptions:* linear, time-invariant, identical and
translation-invariant influence functions, no hysteresis, no creep, no plate
dynamics. *Validity:* small strokes; the model says nothing about the mirror's
mechanical resonances or about print-through.

Units and sign convention. Commands ``c_a`` and the resulting ``phi_DM`` are in
**radians of optical phase at the working wavelength**, defined as the phase
the mirror *adds* to the beam. A correction removes ``phi_atm`` by commanding
``phi_DM = -phi_atm``. Mechanical surface displacement is
``z = -phi_DM * lambda / (4 pi)`` for a normal-incidence reflection (the factor
2 from the double pass is included).

Fitting error
-------------
Because the DM can only produce phase in the span of its influence functions,
the uncorrectable remainder for Kolmogorov turbulence scales as

```
sigma_fit^2 = mu * (d_act / r0)^(5/3)         [rad^2]
```

with ``mu = 0.34`` for a continuous-facesheet mirror (Hudgin, R. H. 1977,
"Wave-front compensation error due to finite corrector-element size",
*JOSA* **67**, 393-395). ``mu`` depends on the influence-function shape;
``validation/validate_fitting.py`` measures both the exponent and the
coefficient produced by *this* model rather than assuming Hudgin's value.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .pupil import Pupil

__all__ = ["DeformableMirror", "HUDGIN_FITTING_COEFF"]

# Hudgin (1977), JOSA 67, 393: continuous-facesheet corrector.
HUDGIN_FITTING_COEFF: float = 0.34


def _check_positive(name: str, value: float) -> float:
    v = float(value)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if v <= 0.0:
        raise ValueError(f"{name} must be > 0, got {v!r}")
    return v


class DeformableMirror:
    """Square-grid continuous-facesheet DM on a :class:`~waveforge.pupil.Pupil`.

    Parameters
    ----------
    pupil:
        Pupil grid.
    n_act:
        Actuators across the pupil diameter [-]. Must be >= 2. Actuators are
        placed on a square grid of pitch ``D / (n_act - 1)`` spanning the full
        diameter (Fried geometry when paired with ``n_sub = n_act - 1``).
    coupling:
        Inter-actuator coupling ``F(1)`` [-], in ``(0, 0.9)``. Default 0.15.
    stroke:
        Maximum command magnitude [rad of optical phase]. ``None`` (default)
        means unlimited. Must be > 0 if given.
    edge_margin:
        Actuators whose centre lies further than ``edge_margin * pitch``
        outside the pupil radius are removed [-]. Default 1.0 (one ring of
        slaved actuators outside the aperture is kept, standard practice).

    Attributes
    ----------
    n_actuators:
        Number of retained actuators [-].
    influence:
        ``(n_valid_pupil_samples, n_actuators)`` influence matrix [-].
    """

    def __init__(
        self,
        pupil: Pupil,
        n_act: int,
        coupling: float = 0.15,
        stroke: float | None = None,
        edge_margin: float = 1.0,
    ) -> None:
        if not isinstance(pupil, Pupil):
            raise TypeError(f"pupil must be a Pupil, got {type(pupil).__name__}")
        n_act = int(n_act)
        if n_act < 2:
            raise ValueError(f"n_act must be >= 2, got {n_act}")
        c = float(coupling)
        if not (0.0 < c < 0.9):
            raise ValueError(f"coupling must be in (0, 0.9), got {coupling!r}")
        em = float(edge_margin)
        if not np.isfinite(em) or em < 0.0:
            raise ValueError(f"edge_margin must be >= 0, got {edge_margin!r}")
        self.pupil = pupil
        self.n_act = n_act
        self.coupling = c
        self.edge_margin = em
        self.stroke = None if stroke is None else _check_positive("stroke", stroke)

        self.pitch = pupil.diameter / (n_act - 1)
        axis = (np.arange(n_act) - (n_act - 1) / 2.0) * self.pitch
        ax, ay = np.meshgrid(axis, axis, indexing="xy")
        r_act = np.hypot(ax, ay)
        keep = r_act <= pupil.diameter / 2.0 + em * self.pitch
        self.actuator_x = ax[keep]
        self.actuator_y = ay[keep]
        self.n_actuators = int(self.actuator_x.size)

        px, py = pupil.coords()
        mx = px[pupil.mask]
        my = py[pupil.mask]
        u = np.hypot(
            mx[:, None] - self.actuator_x[None, :], my[:, None] - self.actuator_y[None, :]
        ) / self.pitch
        self.influence: NDArray[np.float64] = np.exp(np.log(c) * u**2)
        self._pinv: NDArray[np.float64] | None = None

    # ------------------------------------------------------------------ shapes
    def influence_function(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        """Gaussian influence function ``F(u) = coupling^(u^2)`` [-], ``u`` in pitches."""
        return np.exp(np.log(self.coupling) * np.asarray(u, dtype=np.float64) ** 2)

    def clip(self, commands: NDArray[np.float64]) -> NDArray[np.float64]:
        """Clip commands to the actuator stroke limit [rad]."""
        c = np.asarray(commands, dtype=np.float64)
        if c.shape != (self.n_actuators,):
            raise ValueError(f"commands must have shape {(self.n_actuators,)}, got {c.shape}")
        if self.stroke is None:
            return c.copy()
        return np.clip(c, -self.stroke, self.stroke)

    def saturated(self, commands: NDArray[np.float64]) -> NDArray[np.bool_]:
        """Boolean mask of actuators at or beyond the stroke limit."""
        c = np.asarray(commands, dtype=np.float64)
        if self.stroke is None:
            return np.zeros(c.shape, dtype=bool)
        return np.abs(c) >= self.stroke * (1.0 - 1.0e-12)

    def shape(self, commands: NDArray[np.float64], apply_stroke: bool = True):
        """Phase map produced by ``commands``, shape ``(n_grid, n_grid)`` [rad].

        With ``apply_stroke`` (default) the commands are clipped to the stroke
        limit first, so the returned surface is the one the hardware can make.
        """
        c = self.clip(commands) if apply_stroke else np.asarray(commands, dtype=np.float64)
        out = np.zeros((self.pupil.n_grid, self.pupil.n_grid), dtype=np.float64)
        out[self.pupil.mask] = self.influence @ c
        return out

    # ------------------------------------------------------------------ fitting
    @property
    def pinv(self) -> NDArray[np.float64]:
        """Least-squares fit matrix, shape ``(n_actuators, n_valid_samples)``."""
        if self._pinv is None:
            self._pinv = np.linalg.pinv(self.influence, rcond=1.0e-8)
        return self._pinv

    def fit(self, phase: NDArray[np.float64], apply_stroke: bool = False):
        """Least-squares commands reproducing ``phase`` [rad] -> commands [rad].

        The fit is unconstrained by default; pass ``apply_stroke=True`` to clip
        the result, which is *not* the constrained optimum (see README
        Limitations).
        """
        arr = np.asarray(phase, dtype=np.float64)
        n = self.pupil.n_grid
        if arr.shape != (n, n):
            raise ValueError(f"phase must have shape {(n, n)}, got {arr.shape}")
        c = self.pinv @ arr[self.pupil.mask]
        return self.clip(c) if apply_stroke else c

    def fitting_residual(self, phase: NDArray[np.float64]) -> NDArray[np.float64]:
        """``phase`` minus the best DM reproduction of it [rad], piston removed."""
        cmd = self.fit(phase, apply_stroke=False)
        return self.pupil.piston_removed(np.asarray(phase, dtype=np.float64) - self.shape(cmd, False))

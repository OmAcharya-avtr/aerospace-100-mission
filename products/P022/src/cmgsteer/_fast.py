"""A fused single-SVD steering step, used only to keep dataset generation affordable.

Every quantity the rollouts need -- the SR-inverse gimbal rates, the
singularity measure, the unit null vector and the manipulability gradient --
comes out of one singular value decomposition of ``A``, instead of the three
that the public code path computes independently.  The result is numerically
identical to the public path to round-off; ``tests/test_dataset.py`` pins that
agreement, and nothing but :mod:`cmgsteer.dataset` uses this module.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .arrays import CMGArray

__all__ = ["FastStepper"]


class FastStepper:
    """Fused SR-inverse + null-motion step for one array.

    Parameters
    ----------
    array
        The CMG array.  Only arrays with no locked gimbals and at least four
        free gimbals have a non-trivial null space; with exactly three the
        stepper still works and the null motion is zero.
    lam0, mu
        Adaptive robustness parameters, as in
        :func:`cmgsteer.steering.robustness_parameter`.
    max_gimbal_rate
        Symmetric rate limit [rad/s], applied by clipping, or ``None``.
    """

    def __init__(
        self,
        array: CMGArray,
        lam0: float,
        mu: float,
        max_gimbal_rate: float | None = None,
    ) -> None:
        self.array = array
        self.lam0 = float(lam0)
        self.mu = float(mu)
        self.max_gimbal_rate = max_gimbal_rate
        self._idx = array.free_indices
        self._c = array.ref_axes[self._idx]
        self._s = array.transverse_axes[self._idx]
        self._h0 = array.rotor_momenta[self._idx]
        self._h0_all = array.rotor_momenta
        self._c_all = array.ref_axes
        self._s_all = array.transverse_axes
        self._h0_mean = float(np.mean(array.rotor_momenta))
        self._n_free = int(self._idx.size)

    def momentum(self, deltas: NDArray[np.float64]) -> NDArray[np.float64]:
        """Array momentum [N*m*s], same value as :meth:`CMGArray.momentum`."""
        sd = np.sin(deltas)
        cd = np.cos(deltas)
        return self._h0_all @ (self._c_all * cd[:, None] + self._s_all * sd[:, None])

    def step(
        self,
        deltas: NDArray[np.float64],
        torque: NDArray[np.float64],
        coefficient: float,
        max_null_rate: float,
        gradient_gain: float | None = None,
    ) -> NDArray[np.float64]:
        """Full gimbal-rate command [rad/s] for one step, length ``n_cmgs``.

        ``coefficient`` is the constant null-motion coefficient.  When
        ``gradient_gain`` is not ``None`` the classical gradient policy is used
        instead, with ``coefficient`` reinterpreted as the rate cap.
        """
        d_free = deltas[self._idx]
        sd = np.sin(d_free)
        cd = np.cos(d_free)
        hhat = self._c * cd[:, None] + self._s * sd[:, None]
        jac = (self._h0[:, None] * (-self._c * sd[:, None] + self._s * cd[:, None])).T
        u, sv, vt = np.linalg.svd(jac)
        measure = float(np.prod(sv))
        lam = self.lam0 * self._h0_mean**2 * np.exp(-self.mu * measure / self._h0_mean**3)
        rates = vt[:3].T @ ((sv / (sv**2 + lam)) * (u.T @ (-torque)))

        if self._n_free == 4 and (coefficient != 0.0 or gradient_gain is not None):
            null = vt[3]
            aprime = -(self._h0[:, None] * hhat)
            cofactor = np.array([np.prod(np.delete(sv, k)) for k in range(sv.size)])
            grad = ((aprime @ u) * vt[:3].T) @ cofactor
            dot = float(null @ grad)
            if dot < 0.0:
                null = -null
                dot = -dot
            if gradient_gain is None:
                rates = rates + max_null_rate * coefficient * null
            else:
                amount = gradient_gain * dot
                cap = max_null_rate
                peak = abs(amount) * float(np.max(np.abs(null)))
                if cap is not None and peak > cap:
                    amount *= cap / peak
                rates = rates + amount * null

        if self.max_gimbal_rate is not None:
            rates = np.clip(rates, -self.max_gimbal_rate, self.max_gimbal_rate)
        out = np.zeros(self.array.n_cmgs)
        out[self._idx] = rates
        return out

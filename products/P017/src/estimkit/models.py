"""Standard kinematic models used as filter test cases.

All models are discrete-time with a fixed sample interval ``dt`` [s].
State ordering is ``[position, velocity]`` in SI units ``[m, m/s]``.

References
----------
Bar-Shalom, Y., Rong Li, X. and Kirubarajan, T., *Estimation with
Applications to Tracking and Navigation*, Wiley 2001, Chapter 6
("Estimation for kinematic models") -- the continuous white-noise
acceleration (CWNA) and discrete white-noise acceleration (DWNA)
constant-velocity models below are the two standard forms given there.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "constant_velocity_cwna",
    "constant_velocity_dwna",
    "random_walk",
]


def _check_dt(dt: float) -> float:
    dt = float(dt)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be a positive finite number of seconds, got {dt}")
    return dt


def constant_velocity_cwna(dt: float, q_psd: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Constant-velocity model with continuous white-noise acceleration.

    The acceleration is modelled as continuous-time white noise of power
    spectral density ``q_psd``; integrating the continuous process noise
    over one sample interval gives

    .. math::
        F = \begin{bmatrix} 1 & T \\ 0 & 1\end{bmatrix}, \qquad
        Q = \tilde q \begin{bmatrix} T^{3}/3 & T^{2}/2 \\
                                     T^{2}/2 & T \end{bmatrix}

    Parameters
    ----------
    dt : float
        Sample interval ``T`` [s], must be > 0.
    q_psd : float
        Acceleration power spectral density :math:`\tilde q`
        [m^2/s^3], must be >= 0.

    Returns
    -------
    (F, Q) : (ndarray (2, 2), ndarray (2, 2))
        ``F`` dimensionless / s as appropriate; ``Q`` in
        [[m^2, m^2/s], [m^2/s, m^2/s^2]].

    Notes
    -----
    Source: Bar-Shalom, Rong Li & Kirubarajan 2001, Ch. 6 (CWNA model).
    Validity: manoeuvre-free or gently manoeuvring targets over the
    sample interval; :math:`\tilde q` is chosen so that
    :math:`\sqrt{\tilde q T}` is comparable to the expected velocity
    change per sample.
    """
    t = _check_dt(dt)
    q = float(q_psd)
    if not np.isfinite(q) or q < 0.0:
        raise ValueError(f"q_psd must be a non-negative finite number, got {q_psd}")
    f = np.array([[1.0, t], [0.0, 1.0]])
    qm = q * np.array([[t**3 / 3.0, t**2 / 2.0], [t**2 / 2.0, t]])
    return f, qm


def constant_velocity_dwna(dt: float, sigma_a: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Constant-velocity model with discrete white-noise acceleration.

    A single random acceleration, constant over each sample interval and
    independent between intervals, with standard deviation ``sigma_a``:

    .. math::
        Q = \sigma_a^{2}\,\Gamma\Gamma^{\mathsf{T}}, \qquad
        \Gamma = \begin{bmatrix} T^{2}/2 \\ T \end{bmatrix}
        \;\Rightarrow\;
        Q = \sigma_a^{2}\begin{bmatrix} T^{4}/4 & T^{3}/2 \\
                                        T^{3}/2 & T^{2}\end{bmatrix}

    Parameters
    ----------
    dt : float
        Sample interval ``T`` [s], must be > 0.
    sigma_a : float
        Standard deviation of the per-interval acceleration [m/s^2],
        must be >= 0.

    Returns
    -------
    (F, Q) : (ndarray (2, 2), ndarray (2, 2))

    Notes
    -----
    Source: Bar-Shalom, Rong Li & Kirubarajan 2001, Ch. 6 (DWNA model).
    ``Q`` is rank-1 by construction, hence positive *semi*-definite --
    a useful stress case for covariance handling.
    """
    t = _check_dt(dt)
    s = float(sigma_a)
    if not np.isfinite(s) or s < 0.0:
        raise ValueError(f"sigma_a must be a non-negative finite number, got {sigma_a}")
    f = np.array([[1.0, t], [0.0, 1.0]])
    gamma = np.array([[t**2 / 2.0], [t]])
    return f, (s**2) * (gamma @ gamma.T)


def random_walk(q: float, r: float) -> tuple[NDArray[np.float64], ...]:
    r"""Scalar random-walk-plus-noise model ``x_k = x_{k-1} + w``, ``z = x + v``.

    This is the model whose steady-state Riccati equation is solved by
    hand in ``validation/VALIDATION.md``:
    :math:`P^{-}_{\infty} = \tfrac{1}{2}\left(q + \sqrt{q^{2}+4qr}\right)`.

    Parameters
    ----------
    q : float
        Process-noise variance per step [x-units^2], must be >= 0.
    r : float
        Measurement-noise variance [x-units^2], must be > 0.

    Returns
    -------
    (F, H, Q, R) : tuple of 1x1 ndarrays
    """
    qv, rv = float(q), float(r)
    if not np.isfinite(qv) or qv < 0.0:
        raise ValueError(f"q must be a non-negative finite number, got {q}")
    if not np.isfinite(rv) or rv <= 0.0:
        raise ValueError(f"r must be a positive finite number, got {r}")
    return (
        np.array([[1.0]]),
        np.array([[1.0]]),
        np.array([[qv]]),
        np.array([[rv]]),
    )

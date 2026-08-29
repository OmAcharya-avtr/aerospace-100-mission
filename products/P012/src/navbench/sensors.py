r"""Sensor models with documented noise and bias behaviour.

Every model below states its parameterisation, its units, the discretisation
used, and the regime in which it is valid. All randomness comes from a
``numpy.random.Generator`` supplied by the caller, so every trajectory is
reproducible from its seed.

Gyroscope
---------
Continuous model (the standard "ARW + bias" rate-gyro model):

.. math::
    \tilde\omega(t) = \omega(t) + b(t) + n_v(t), \qquad
    \dot b(t) = -\frac{1}{\tau} b(t) + n_u(t)

* ``n_v`` white, PSD ``N²`` [(rad/s)²/Hz] — **angle random walk (ARW)**,
  quoted in datasheets as ``N`` in rad/√s (or °/√h).
* ``b`` a first-order Gauss-Markov (exponentially correlated) process with
  correlation time ``τ`` [s] and steady-state standard deviation ``σ_b``
  [rad/s] — the engineering stand-in for the **bias-instability** flicker
  floor. When ``τ = ∞`` the process degenerates to a **rate random walk** with
  parameter ``K`` [rad/s^{3/2}].

Exact discretisation at step ``Δt`` (Gelb 1974, *Applied Optimal Estimation*,
Table 4.1-1; Farrell 2008, *Aided Navigation*, §4.6):

.. math::
    \sigma_{n_v,\text{disc}} = N/\sqrt{\Delta t}, \qquad
    b_{k+1} = e^{-\Delta t/\tau} b_k + w_k,\quad
    \operatorname{var}(w_k)=\sigma_b^2\left(1-e^{-2\Delta t/\tau}\right)

For the random-walk limit ``b_{k+1} = b_k + w_k`` with
``std(w_k) = K√Δt``.

**Rate output versus delta-theta output — this matters.** Navigation-grade
gyros (fibre-optic, ring-laser, and most MEMS IMUs in strapdown use) report the
*integrated angle increment* ``Δθ_k = ∫_{t_{k-1}}^{t_k} ω dt`` over each sample
interval, not the instantaneous rate at the sample instant. The distinction is
not cosmetic:

* with **delta-theta** output, rectangular integration of ``Δθ_k/Δt`` in the
  filter is exact for the deterministic part, and the angle noise contributed
  by each step is white with variance ``N²Δt`` — exactly what the standard MEKF
  ``Q_d`` assumes;
* with **instantaneous-rate** output, holding one sample constant across the
  step leaves a deterministic ``O(‖ω̇‖Δt²)`` error per step that accumulates
  coherently (measured in ``validation/v4_attitude_mekf.py``: 180.9 arcsec over
  200 s at ``Δt = 0.1 s``). Averaging consecutive samples removes that drift but
  makes the per-step angle noise a *moving average* — correlated across steps
  and with half the single-step variance — which ``Q_d`` does not model, and
  which shows up as a measurable conservatism in NEES.

:class:`GyroParams` therefore carries a ``mode`` field. ``"delta_theta"`` (the
default) is the physically representative choice and the one used by the
validation; ``"rate"`` is retained so the integration-order effect can be
demonstrated rather than merely asserted.

**Honesty note.** True bias instability is 1/f (flicker) noise, which is *not*
a finite-order Markov process; the Gauss-Markov surrogate matches the Allan
deviation only near the bias-instability floor and is the standard aerospace
approximation. See IEEE Std 952-1997 (R2008) Annex C, and El-Sheimy, Hou & Niu
(2008), *IEEE Trans. Instrum. Meas.* **57**(1), 140–149.

Star tracker
------------
Attitude measurement expressed as a quaternion perturbed in the **body** frame:
``q̃ = q ⊗ δq(a_n)`` with ``a_n ~ N(0, diag(σ_x², σ_y², σ_z²))`` [rad]. Real
star trackers are 5–10× less accurate about the boresight than cross-boresight,
so the two are separate parameters. Source: Markley & Crassidis (2014) §4.3;
Liebe, C. C. (2002), "Accuracy Performance of Star Trackers — A Tutorial",
*IEEE Trans. Aerospace and Electronic Systems* **38**(2), 587–599.

Sun sensor / star vector
------------------------
Unit-vector observation in body axes, ``b̃ = A(q) r + n``, renormalised, with
``n ~ N(0, σ² I)`` [dimensionless]. This is the QUEST measurement model
(Shuster & Oh 1981, *J. Guidance and Control* **4**(1), 70–77). Valid for
``σ ≲ 0.1 rad``; the renormalisation makes the additive-Gaussian assumption
only approximately correct, which is why ``σ`` is small by construction.

Accelerometer
-------------
Specific force in body axes, ``f̃ = A(q)(a − g) + b_a + n_a`` [m/s²], with the
same ARW/bias structure as the gyro (velocity random walk + Gauss-Markov bias).
In free fall (the orbital case) ``a = g`` and the specific force is zero up to
the bias and noise — that is a feature, not a bug, and is why the orbital
navigation case uses GPS rather than the accelerometer for position.

GPS-like position fix
---------------------
``p̃ = p + n_p`` [m], ``n_p ~ N(0, σ_p² I)``, delivered every ``decimation``
steps, with an optional Bernoulli dropout and an optional scheduled outage
window. This deliberately omits correlated multipath and ionospheric errors;
see README Limitations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .quaternion import (
    attitude_matrix,
    quat_conjugate,
    quat_multiply,
    quat_normalize,
    quat_to_rotvec,
    small_angle_quat,
)
from .truth import AttitudeTruth, PositionTruth

__all__ = [
    "GyroParams",
    "GyroMeasurements",
    "StarTrackerParams",
    "VectorSensorParams",
    "AccelerometerParams",
    "GpsParams",
    "simulate_gyro",
    "simulate_star_tracker",
    "simulate_vector_sensor",
    "simulate_accelerometer",
    "simulate_gps",
    "allan_deviation",
]


def _rng(rng: np.random.Generator | int | None) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def _positive(value: float, name: str) -> float:
    v = float(value)
    if not np.isfinite(v) or v < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number, got {value!r}")
    return v


@dataclass(frozen=True)
class GyroParams:
    """Rate-gyro error parameters.

    Parameters
    ----------
    arw : float
        Angle random walk ``N`` [rad/√s]. A 0.1 °/√h gyro has
        ``N = 0.1·(π/180)/60 = 2.909e-5`` rad/√s.
    bias_sigma : float
        Steady-state standard deviation of the Gauss-Markov bias [rad/s].
    bias_tau : float
        Bias correlation time [s]. Use ``np.inf`` for a pure random walk, in
        which case ``rrw`` is used instead of ``bias_sigma``.
    rrw : float
        Rate random walk ``K`` [rad/s^{3/2}], used only when ``bias_tau`` is
        infinite.
    initial_bias : array_like or None
        Initial bias [rad/s]. ``None`` draws it from the stationary
        distribution (Gauss-Markov) or sets it to zero (random walk).
    scale_factor : array_like
        Per-axis multiplicative scale-factor error (dimensionless, added to 1).
    mode : {"delta_theta", "rate"}
        ``"delta_theta"`` (default) reports, for sample ``k >= 1``, the *average*
        rate over the interval ``[t_{k-1}, t_k]`` obtained from the exact
        rotation increment of the truth trajectory — the output of a
        rate-integrating gyro, divided by ``Δt``. ``"rate"`` reports the
        instantaneous truth rate at ``t_k``. See the module docstring for why
        the choice changes filter consistency.
    """

    arw: float = 3.0e-5
    bias_sigma: float = 4.85e-6
    bias_tau: float = 300.0
    rrw: float = 0.0
    initial_bias: NDArray[np.float64] | None = None
    scale_factor: tuple[float, float, float] = (0.0, 0.0, 0.0)
    mode: str = "delta_theta"

    def __post_init__(self) -> None:
        if self.mode not in ("delta_theta", "rate"):
            raise ValueError(f"mode must be 'delta_theta' or 'rate', got {self.mode!r}")
        _positive(self.arw, "arw")
        _positive(self.bias_sigma, "bias_sigma")
        _positive(self.rrw, "rrw")
        if self.bias_tau <= 0.0:
            raise ValueError(f"bias_tau must be > 0 (or np.inf), got {self.bias_tau}")
        if np.isinf(self.bias_tau) and self.rrw == 0.0 and self.bias_sigma > 0.0:
            raise ValueError(
                "bias_tau=inf selects the rate-random-walk model; set rrw > 0 "
                "(bias_sigma is ignored in that mode)"
            )


@dataclass(frozen=True)
class GyroMeasurements:
    """Simulated gyro output.

    Attributes
    ----------
    t : ndarray, shape (K,)
        Times [s].
    omega : ndarray, shape (K, 3)
        Measured body rate [rad/s].
    bias : ndarray, shape (K, 3)
        The realised (truth) bias [rad/s] — available for scoring only.
    """

    t: NDArray[np.float64]
    omega: NDArray[np.float64]
    bias: NDArray[np.float64]


def simulate_gyro(
    truth: AttitudeTruth, params: GyroParams, rng: np.random.Generator | int | None = None
) -> GyroMeasurements:
    r"""Sample the gyro model along an attitude truth trajectory.

    Discretisation as documented in the module docstring. Returns measurements
    at the same times as ``truth``.
    """
    g = _rng(rng)
    dt = truth.dt
    k = len(truth.t)
    sf = 1.0 + np.asarray(params.scale_factor, dtype=float).reshape(3)

    bias = np.zeros((k, 3))
    if np.isinf(params.bias_tau):
        b = np.zeros(3) if params.initial_bias is None else np.asarray(
            params.initial_bias, dtype=float
        ).reshape(3)
        sigma_u = params.rrw * np.sqrt(dt)
        for i in range(k):
            bias[i] = b
            b = b + sigma_u * g.standard_normal(3)
    else:
        phi = float(np.exp(-dt / params.bias_tau))
        sigma_w = params.bias_sigma * np.sqrt(max(1.0 - phi * phi, 0.0))
        b = (
            params.bias_sigma * g.standard_normal(3)
            if params.initial_bias is None
            else np.asarray(params.initial_bias, dtype=float).reshape(3)
        )
        for i in range(k):
            bias[i] = b
            b = phi * b + sigma_w * g.standard_normal(3)

    if params.mode == "delta_theta":
        base = np.empty_like(truth.omega)
        # Exact rotation increment of the truth over each interval, divided by dt.
        for i in range(1, k):
            dq = quat_multiply(quat_conjugate(truth.q[i - 1]), truth.q[i])
            base[i] = quat_to_rotvec(dq) / dt
        base[0] = base[1] if k > 1 else truth.omega[0]
        eff_bias = bias.copy()
        eff_bias[1:] = 0.5 * (bias[:-1] + bias[1:])
    else:
        base = truth.omega
        eff_bias = bias

    sigma_v = params.arw / np.sqrt(dt) if dt > 0 else 0.0
    noise = sigma_v * g.standard_normal((k, 3))
    omega = base * sf + eff_bias + noise
    return GyroMeasurements(t=truth.t.copy(), omega=omega, bias=bias)


@dataclass(frozen=True)
class StarTrackerParams:
    """Star-tracker attitude-measurement parameters.

    Parameters
    ----------
    sigma_cross : float
        1-σ error about the two cross-boresight axes [rad].
    sigma_boresight : float
        1-σ error about the boresight [rad]; typically 5–10× ``sigma_cross``.
    boresight_body : array_like, shape (3,)
        Boresight direction in body axes (unit vector).
    decimation : int
        Emit a measurement every ``decimation`` truth samples (>= 1).
    """

    sigma_cross: float = 1.0e-5
    sigma_boresight: float = 7.0e-5
    boresight_body: tuple[float, float, float] = (0.0, 0.0, 1.0)
    decimation: int = 1

    def __post_init__(self) -> None:
        _positive(self.sigma_cross, "sigma_cross")
        _positive(self.sigma_boresight, "sigma_boresight")
        if int(self.decimation) < 1:
            raise ValueError(f"decimation must be >= 1, got {self.decimation}")
        if float(np.linalg.norm(self.boresight_body)) < 1e-12:
            raise ValueError("boresight_body must be a non-zero vector")

    def noise_covariance(self) -> NDArray[np.float64]:
        """Measurement covariance ``R`` [rad²] in **body** axes, shape (3, 3)."""
        n = np.asarray(self.boresight_body, dtype=float).reshape(3)
        n = n / np.linalg.norm(n)
        return (
            self.sigma_cross ** 2 * (np.eye(3) - np.outer(n, n))
            + self.sigma_boresight ** 2 * np.outer(n, n)
        )


def simulate_star_tracker(
    truth: AttitudeTruth,
    params: StarTrackerParams,
    rng: np.random.Generator | int | None = None,
) -> tuple[NDArray[np.intp], NDArray[np.float64]]:
    r"""Simulate quaternion attitude measurements ``q̃ = q ⊗ δq(a_n)``.

    Returns
    -------
    indices : ndarray of int
        Truth-sample indices at which a measurement is available.
    quats : ndarray, shape (len(indices), 4)
        Measured quaternions, scalar-first, normalised.
    """
    g = _rng(rng)
    idx = np.arange(0, len(truth.t), int(params.decimation), dtype=np.intp)
    r = params.noise_covariance()
    chol = np.linalg.cholesky(r)
    out = np.zeros((len(idx), 4))
    for j, i in enumerate(idx):
        a_n = chol @ g.standard_normal(3)
        out[j] = quat_normalize(quat_multiply(truth.q[i], small_angle_quat(a_n)))
    return idx, out


@dataclass(frozen=True)
class VectorSensorParams:
    """Unit-vector sensor (sun sensor, magnetometer, single star vector).

    Parameters
    ----------
    sigma : float
        Per-component 1-σ noise [dimensionless] on the measured unit vector.
        Approximately the angular error in radians for small ``sigma``.
    reference_inertial : array_like, shape (3,)
        Known reference direction in inertial axes (unit vector).
    decimation : int
        Measurement decimation (>= 1).
    """

    sigma: float = 1.0e-3
    reference_inertial: tuple[float, float, float] = (1.0, 0.0, 0.0)
    decimation: int = 1

    def __post_init__(self) -> None:
        _positive(self.sigma, "sigma")
        if self.sigma > 0.1:
            raise ValueError(
                f"sigma={self.sigma} exceeds the 0.1 rad validity limit of the additive "
                "unit-vector noise model (Shuster & Oh 1981)"
            )
        if int(self.decimation) < 1:
            raise ValueError(f"decimation must be >= 1, got {self.decimation}")


def simulate_vector_sensor(
    truth: AttitudeTruth,
    params: VectorSensorParams,
    rng: np.random.Generator | int | None = None,
) -> tuple[NDArray[np.intp], NDArray[np.float64]]:
    """Simulate body-frame unit-vector observations of a known inertial direction."""
    g = _rng(rng)
    idx = np.arange(0, len(truth.t), int(params.decimation), dtype=np.intp)
    ref = np.asarray(params.reference_inertial, dtype=float).reshape(3)
    ref = ref / np.linalg.norm(ref)
    out = np.zeros((len(idx), 3))
    for j, i in enumerate(idx):
        b = attitude_matrix(truth.q[i]) @ ref + params.sigma * g.standard_normal(3)
        out[j] = b / np.linalg.norm(b)
    return idx, out


@dataclass(frozen=True)
class AccelerometerParams:
    """Accelerometer error parameters (same structure as the gyro).

    Parameters
    ----------
    vrw : float
        Velocity random walk [m/s/√s] (white specific-force noise PSD^(1/2)).
    bias_sigma : float
        Steady-state Gauss-Markov bias 1-σ [m/s²].
    bias_tau : float
        Bias correlation time [s].
    gravity : array_like, shape (3,)
        Gravity vector in the inertial frame [m/s²] used to form the specific
        force ``f = A(q)(a − g)``. Set to zeros for a free-fall/orbital case
        where the truth acceleration already *is* gravity.
    """

    vrw: float = 1.0e-3
    bias_sigma: float = 1.0e-3
    bias_tau: float = 600.0
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.80665)

    def __post_init__(self) -> None:
        _positive(self.vrw, "vrw")
        _positive(self.bias_sigma, "bias_sigma")
        if self.bias_tau <= 0.0:
            raise ValueError(f"bias_tau must be > 0, got {self.bias_tau}")


def simulate_accelerometer(
    position: PositionTruth,
    attitude: AttitudeTruth,
    params: AccelerometerParams,
    rng: np.random.Generator | int | None = None,
) -> NDArray[np.float64]:
    r"""Specific force in body axes [m/s²], shape (K, 3).

    ``f̃_k = A(q_k)(a_k − g) + b_k + n_k``. ``position`` and ``attitude`` must be
    sampled on the same time grid.
    """
    if len(position.t) != len(attitude.t):
        raise ValueError(
            f"position ({len(position.t)} samples) and attitude ({len(attitude.t)}) must share "
            "a time grid"
        )
    g = _rng(rng)
    dt = position.dt
    k = len(position.t)
    grav = np.asarray(params.gravity, dtype=float).reshape(3)
    phi = float(np.exp(-dt / params.bias_tau))
    sigma_w = params.bias_sigma * np.sqrt(max(1.0 - phi * phi, 0.0))
    b = params.bias_sigma * g.standard_normal(3)
    sigma_n = params.vrw / np.sqrt(dt)
    out = np.zeros((k, 3))
    for i in range(k):
        f_body = attitude_matrix(attitude.q[i]) @ (position.acc[i] - grav)
        out[i] = f_body + b + sigma_n * g.standard_normal(3)
        b = phi * b + sigma_w * g.standard_normal(3)
    return out


@dataclass(frozen=True)
class GpsParams:
    """GPS-like position-fix parameters.

    Parameters
    ----------
    sigma_pos : float
        Per-axis 1-σ position error [m].
    decimation : int
        A fix every ``decimation`` truth samples.
    dropout_prob : float
        Independent Bernoulli probability that a scheduled fix is lost.
    outage : tuple of (int, int) or None
        Half-open truth-index window ``[start, stop)`` during which no fix is
        produced (a scheduled outage, e.g. an urban canyon or an antenna
        occultation).
    """

    sigma_pos: float = 5.0
    decimation: int = 1
    dropout_prob: float = 0.0
    outage: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        _positive(self.sigma_pos, "sigma_pos")
        if int(self.decimation) < 1:
            raise ValueError(f"decimation must be >= 1, got {self.decimation}")
        if not 0.0 <= float(self.dropout_prob) <= 1.0:
            raise ValueError(f"dropout_prob must be in [0, 1], got {self.dropout_prob}")
        if self.outage is not None:
            a, b = self.outage
            if not 0 <= int(a) < int(b):
                raise ValueError(f"outage must be (start, stop) with 0 <= start < stop, got {self.outage}")


def simulate_gps(
    position: PositionTruth, params: GpsParams, rng: np.random.Generator | int | None = None
) -> tuple[NDArray[np.intp], NDArray[np.float64]]:
    """Simulate position fixes.

    Returns ``(indices, fixes)`` where ``fixes`` has shape ``(len(indices), 3)``
    in metres. Indices lost to dropout or an outage are simply absent.
    """
    g = _rng(rng)
    idx = np.arange(0, len(position.t), int(params.decimation), dtype=np.intp)
    if params.outage is not None:
        a, b = params.outage
        idx = idx[(idx < a) | (idx >= b)]
    if params.dropout_prob > 0.0:
        keep = g.random(len(idx)) >= params.dropout_prob
        idx = idx[keep]
    fixes = position.pos[idx] + params.sigma_pos * g.standard_normal((len(idx), 3))
    return idx, fixes


def allan_deviation(
    series: ArrayLike, dt: float, cluster_sizes: Sequence[int] | None = None
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Overlapping Allan deviation of a rate series.

    .. math::
        \sigma^2(\tau) = \frac{1}{2\tau^2 (N-2m+1)}
            \sum_{j=1}^{N-2m+1}\left(\theta_{j+2m} - 2\theta_{j+m} + \theta_j\right)^2

    with ``θ`` the cumulative integral of the rate (the angle) and ``τ = m Δt``.
    Source: IEEE Std 952-1997 (R2008), Annex C (Allan variance).

    Parameters
    ----------
    series : array_like, shape (N,)
        Rate samples [rad/s] (single axis).
    dt : float
        Sample interval [s].
    cluster_sizes : sequence of int, optional
        Cluster sizes ``m``; defaults to a log-spaced set up to ``N/5``.

    Returns
    -------
    tau : ndarray
        Cluster times [s].
    sigma : ndarray
        Allan deviation [rad/s].

    Notes
    -----
    The white-noise (ARW) branch is ``σ(τ) = N/√τ``; the rate-random-walk branch
    is ``σ(τ) = K √(τ/3)``. These asymptotes are what the validation script
    checks against.
    """
    x = np.asarray(series, dtype=float).reshape(-1)
    n = x.size
    if n < 16:
        raise ValueError(f"need at least 16 samples for an Allan deviation, got {n}")
    if dt <= 0.0:
        raise ValueError(f"dt must be > 0, got {dt}")
    theta = np.concatenate(([0.0], np.cumsum(x) * dt))
    if cluster_sizes is None:
        mmax = max(2, n // 5)
        ms = np.unique(np.round(np.logspace(0, np.log10(mmax), 24)).astype(int))
    else:
        ms = np.unique(np.asarray(cluster_sizes, dtype=int))
    ms = ms[(ms >= 1) & (2 * ms + 1 <= theta.size)]
    taus, devs = [], []
    for m in ms:
        tau = m * dt
        d = theta[2 * m:] - 2.0 * theta[m:-m] + theta[:-2 * m]
        var = float(np.sum(d * d)) / (2.0 * tau * tau * d.size)
        taus.append(tau)
        devs.append(np.sqrt(var))
    return np.asarray(taus), np.asarray(devs)

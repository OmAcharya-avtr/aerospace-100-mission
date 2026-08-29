"""Sensor models with documented noise and bias behaviour.

Every model takes an explicit ``numpy.random.Generator`` so that whole
scenarios are reproducible from a single seed.  Noise parameters follow the
terminology of **IEEE Std 952-2020** (*IEEE Standard Specification Format
Guide and Test Procedure for Single-Axis Interferometric Fiber Optic Gyros*),
Annex C, which defines angle random walk (ARW), rate random walk (RRW) and
bias instability through the Allan-variance slopes.

GYRO MODEL (Farrenkopf 1978; Markley & Crassidis 2014 §4.7.2)

    ω_meas(t) = ω_true(t) + b(t) + η_v(t)
    ḃ(t)      = η_u(t)

with independent zero-mean white noises

    E[η_v(t) η_vᵀ(t')] = σ_v² I δ(t − t')     σ_v in rad/s^{1/2}  (ARW)
    E[η_u(t) η_uᵀ(t')] = σ_u² I δ(t − t')     σ_u in rad/s^{3/2}  (RRW)

Sampled at step ``Δt`` the standard discrete realisation is

    ω_k = ω_true,k + b_k + (σ_v/√Δt) N(0, I)
    b_{k+1} = b_k + σ_u √Δt N(0, I)

(Markley & Crassidis 2014 Eqs. (4.53)-(4.54)).  Validity: Δt short compared
with the bias correlation time; the model has no bias-instability *flicker*
floor — the RRW term is the low-frequency mechanism represented here, and
the flicker (1/f) plateau of a real Allan deviation curve is **not** modelled.
That omission is stated in README Limitations.

Unit conversions used throughout:
    ARW quoted in deg/√hr  →  σ_v [rad/s^{1/2}] = ARW · (π/180) / 60
    RRW quoted in deg/hr^{3/2} → σ_u [rad/s^{3/2}] = RRW · (π/180) / 3600^{1.5}

STAR TRACKER (Markley & Crassidis 2014 §4.2; Shuster & Oh 1981)
Returns unit vectors of catalogued stars in the body frame, each perturbed by
a small rotation with per-axis standard deviation ``sigma_rad``.  The
cross-boresight/boresight anisotropy of a real star tracker is *not* modelled;
the noise is isotropic.  A full attitude-quaternion output mode is also
provided for filters that consume an attitude measurement directly.

SUN SENSOR (Markley & Crassidis 2014 §4.3) — same unit-vector model with a
larger sigma and an eclipse/field-of-view availability flag.

ACCELEROMETER — specific force ``f = a_inertial − g`` in body axes, with a
constant turn-on bias plus velocity random walk (white acceleration noise).

GPS-LIKE POSITION FIX — position (optionally position+velocity) in the same
inertial frame as the truth, with white Gaussian error and a per-epoch
dropout probability.  No ionospheric, multipath or clock modelling: this is a
*navigation-filter test signal*, not a GNSS simulator.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .attitude import dcm_from_quat, quat_from_small_angle, quat_multiply, quat_normalize

__all__ = [
    "DEG_PER_SQRT_HR_TO_SI",
    "DEG_PER_HR_1P5_TO_SI",
    "arw_deg_per_sqrt_hour_to_si",
    "rrw_deg_per_hour_1p5_to_si",
    "GyroModel",
    "StarTrackerModel",
    "SunSensorModel",
    "AccelerometerModel",
    "GpsModel",
    "GyroOutput",
    "VectorOutput",
    "GpsOutput",
]

#: deg/√hr → rad/s^{1/2}: (π/180) / √3600 = (π/180)/60.
DEG_PER_SQRT_HR_TO_SI = (np.pi / 180.0) / 60.0
#: deg/hr^{3/2} → rad/s^{3/2}: (π/180) / 3600^{3/2}.
DEG_PER_HR_1P5_TO_SI = (np.pi / 180.0) / 3600.0**1.5


def arw_deg_per_sqrt_hour_to_si(arw: float) -> float:
    """Angle random walk deg/√hr → σ_v in rad/s^{1/2} (IEEE Std 952-2020, Annex C)."""
    val = float(arw)
    if not np.isfinite(val) or val < 0.0:
        raise ValueError(f"arw must be finite and >= 0 deg/sqrt(hr), got {arw!r}")
    return val * DEG_PER_SQRT_HR_TO_SI


def rrw_deg_per_hour_1p5_to_si(rrw: float) -> float:
    """Rate random walk deg/hr^{3/2} → σ_u in rad/s^{3/2} (IEEE Std 952-2020, Annex C)."""
    val = float(rrw)
    if not np.isfinite(val) or val < 0.0:
        raise ValueError(f"rrw must be finite and >= 0 deg/hr^1.5, got {rrw!r}")
    return val * DEG_PER_HR_1P5_TO_SI


def _positive(name: str, value: float, allow_zero: bool = True) -> float:
    val = float(value)
    if not np.isfinite(val):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if val < 0.0 or (val == 0.0 and not allow_zero):
        bound = ">= 0" if allow_zero else "> 0"
        raise ValueError(f"{name} must be {bound}, got {value!r}")
    return val


@dataclass
class GyroOutput:
    """One gyro sample.

    Attributes
    ----------
    rate : ndarray, shape (3,)
        Measured body rate [rad/s].
    bias : ndarray, shape (3,)
        Bias realisation in effect for this sample [rad/s] (truth, for scoring).
    """

    rate: NDArray[np.float64]
    bias: NDArray[np.float64]


@dataclass
class VectorOutput:
    """One unit-vector observation.

    Attributes
    ----------
    body : ndarray, shape (3,)
        Measured unit vector in body axes (dimensionless).
    reference : ndarray, shape (3,)
        Known inertial-frame unit vector of the same object.
    valid : bool
        False when the object is unavailable (eclipse, out of field of view,
        or a simulated dropout); ``body`` is then all-NaN.
    """

    body: NDArray[np.float64]
    reference: NDArray[np.float64]
    valid: bool = True


@dataclass
class GpsOutput:
    """One GPS-like fix.

    Attributes
    ----------
    position : ndarray, shape (3,)
        Measured inertial position [m]; NaN when ``valid`` is False.
    velocity : ndarray or None, shape (3,)
        Measured inertial velocity [m/s] if the model provides it.
    valid : bool
        False on a simulated dropout epoch.
    """

    position: NDArray[np.float64]
    velocity: NDArray[np.float64] | None
    valid: bool = True


@dataclass
class GyroModel:
    """Rate-integrating gyro with angle random walk and rate random walk.

    Parameters
    ----------
    sigma_v : float
        Angle random walk σ_v [rad/s^{1/2}].  Use
        :func:`arw_deg_per_sqrt_hour_to_si` to convert a datasheet value.
    sigma_u : float
        Rate random walk σ_u [rad/s^{3/2}].  Drives the bias instability.
    dt : float
        Sample interval [s], > 0.
    bias0 : array_like, shape (3,), optional
        Initial (turn-on) bias [rad/s].  Default zero.
    scale_factor_ppm : float, optional
        Symmetric scale-factor error applied to all three axes [parts per
        million].  Default 0.  Included so that a *gross mis-specification*
        failure mode can be exercised; it is deterministic, not random.
    misalignment_rad : float, optional
        Magnitude of a fixed small-angle misalignment of the gyro triad with
        respect to body axes [rad], applied about a fixed axis (1,1,1)/√3.
        Default 0.

    Notes
    -----
    Steady-state Allan deviation slopes implied by this model
    (IEEE Std 952-2020 Annex C):  ``σ_A(τ) = sqrt(σ_v²/τ + σ_u² τ/3)`` —
    a −1/2 slope at short τ set by ARW and a +1/2 slope at long τ set by RRW.
    This is checked numerically in ``validation/v5_sensor_noise.py``.
    """

    sigma_v: float
    sigma_u: float
    dt: float
    bias0: ArrayLike = (0.0, 0.0, 0.0)
    scale_factor_ppm: float = 0.0
    misalignment_rad: float = 0.0
    _bias: NDArray[np.float64] = field(init=False, repr=False)
    _misalign: NDArray[np.float64] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.sigma_v = _positive("sigma_v", self.sigma_v)
        self.sigma_u = _positive("sigma_u", self.sigma_u)
        self.dt = _positive("dt", self.dt, allow_zero=False)
        self.scale_factor_ppm = float(self.scale_factor_ppm)
        if not np.isfinite(self.scale_factor_ppm):
            raise ValueError("scale_factor_ppm must be finite")
        mis = float(self.misalignment_rad)
        if not np.isfinite(mis):
            raise ValueError("misalignment_rad must be finite")
        b0 = np.asarray(self.bias0, dtype=float).reshape(3)
        if not np.all(np.isfinite(b0)):
            raise ValueError("bias0 must be finite")
        self._bias = b0.copy()
        axis = np.ones(3) / np.sqrt(3.0)
        self._misalign = dcm_from_quat(quat_from_small_angle(axis * mis))

    @property
    def bias(self) -> NDArray[np.float64]:
        """Current bias realisation [rad/s] (copy)."""
        return self._bias.copy()

    def reset(self, bias: ArrayLike | None = None) -> None:
        """Reset the internal bias random walk to ``bias`` (default ``bias0``)."""
        b = np.asarray(self.bias0 if bias is None else bias, dtype=float).reshape(3)
        if not np.all(np.isfinite(b)):
            raise ValueError("bias must be finite")
        self._bias = b.copy()

    def discrete_sigmas(self) -> tuple[float, float]:
        """Per-sample noise standard deviations ``(σ_rate, σ_bias_step)``.

        ``σ_rate = σ_v/√Δt`` [rad/s], ``σ_bias_step = σ_u √Δt`` [rad/s].
        """
        return self.sigma_v / np.sqrt(self.dt), self.sigma_u * np.sqrt(self.dt)

    def sample(self, omega_true: ArrayLike, rng: np.random.Generator) -> GyroOutput:
        """Draw one measurement and advance the bias random walk by one step."""
        w = np.asarray(omega_true, dtype=float).reshape(3)
        if not np.all(np.isfinite(w)):
            raise ValueError("omega_true must be finite")
        s_rate, s_bias = self.discrete_sigmas()
        bias_now = self._bias.copy()
        w_eff = self._misalign @ (w * (1.0 + 1e-6 * self.scale_factor_ppm))
        meas = w_eff + bias_now + s_rate * rng.standard_normal(3)
        self._bias = bias_now + s_bias * rng.standard_normal(3)
        return GyroOutput(rate=meas, bias=bias_now)

    def sample_series(
        self, omega_true: ArrayLike, rng: np.random.Generator
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Vectorised whole-run sampling.

        Parameters
        ----------
        omega_true : array_like, shape (N, 3)
            True body rates [rad/s].
        rng : numpy.random.Generator

        Returns
        -------
        rates : ndarray, shape (N, 3) [rad/s]
        biases : ndarray, shape (N, 3) [rad/s]
            Bias in effect at each sample (truth, for scoring).
        """
        w = np.atleast_2d(np.asarray(omega_true, dtype=float))
        if w.ndim != 2 or w.shape[1] != 3:
            raise ValueError(f"omega_true must have shape (N, 3), got {w.shape}")
        if not np.all(np.isfinite(w)):
            raise ValueError("omega_true must be finite")
        n = w.shape[0]
        s_rate, s_bias = self.discrete_sigmas()
        steps = s_bias * rng.standard_normal((n, 3))
        biases = self._bias + np.cumsum(np.vstack([np.zeros(3), steps[:-1]]), axis=0)
        w_eff = (self._misalign @ (w * (1.0 + 1e-6 * self.scale_factor_ppm)).T).T
        rates = w_eff + biases + s_rate * rng.standard_normal((n, 3))
        self._bias = biases[-1] + steps[-1]
        return rates, biases


@dataclass
class StarTrackerModel:
    """Star tracker producing body-frame unit vectors and/or an attitude quaternion.

    Parameters
    ----------
    sigma_rad : float
        Per-axis measurement standard deviation [rad], applied as a small
        rotation of the true line-of-sight.  Typical star trackers: 5-50 µrad
        cross-boresight.
    reference_vectors : array_like, shape (M, 3)
        Inertial unit vectors of the catalogued stars observed.  Rows are
        normalised on construction.
    dropout_prob : float, optional
        Per-epoch probability that the tracker returns no measurement (e.g.
        Sun/Earth in the field of view).  Default 0.

    Notes
    -----
    The measurement is generated as ``b = R(q)ᵀ r_i`` rotated by a small
    random rotation, i.e. the noise acts in the *body* frame.  Because the
    error rotation about the line of sight is unobservable in a single
    vector, the effective per-vector noise covariance is the QUEST measurement
    model ``σ²(I − b bᵀ)`` (Shuster & Oh 1981, *Three-Axis Attitude
    Determination from Vector Observations*, J. Guidance 4(1), 70-77) — a
    rank-2 covariance.  Filters here regularise it by adding ``σ² b bᵀ``,
    which is the standard treatment and is stated in the MEKF docstring.
    """

    sigma_rad: float
    reference_vectors: ArrayLike
    dropout_prob: float = 0.0
    _refs: NDArray[np.float64] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.sigma_rad = _positive("sigma_rad", self.sigma_rad)
        p = float(self.dropout_prob)
        if not np.isfinite(p) or not (0.0 <= p <= 1.0):
            raise ValueError(f"dropout_prob must be in [0, 1], got {self.dropout_prob!r}")
        self.dropout_prob = p
        refs = np.atleast_2d(np.asarray(self.reference_vectors, dtype=float))
        if refs.ndim != 2 or refs.shape[1] != 3:
            raise ValueError(f"reference_vectors must have shape (M, 3), got {refs.shape}")
        if refs.shape[0] == 0:
            raise ValueError("reference_vectors must contain at least one vector")
        norms = np.linalg.norm(refs, axis=1)
        if np.any(norms < 1e-12):
            raise ValueError("every reference vector must have norm >= 1e-12")
        self._refs = refs / norms[:, None]

    @property
    def n_vectors(self) -> int:
        """Number of catalogued reference vectors."""
        return int(self._refs.shape[0])

    @property
    def references(self) -> NDArray[np.float64]:
        """Normalised inertial reference unit vectors, shape (M, 3)."""
        return self._refs.copy()

    def sample(self, quat_true: ArrayLike, rng: np.random.Generator) -> list[VectorOutput]:
        """Observe all catalogued vectors at the true attitude ``quat_true``."""
        rot = dcm_from_quat(quat_true).T  # inertial -> body
        drop = self.dropout_prob > 0.0 and bool(rng.random() < self.dropout_prob)
        out: list[VectorOutput] = []
        for r_i in self._refs:
            b_true = rot @ r_i
            if drop:
                out.append(VectorOutput(np.full(3, np.nan), r_i, valid=False))
                continue
            err = quat_from_small_angle(self.sigma_rad * rng.standard_normal(3))
            b = dcm_from_quat(err).T @ b_true
            out.append(VectorOutput(b / np.linalg.norm(b), r_i, valid=True))
        return out

    def sample_quaternion(
        self, quat_true: ArrayLike, rng: np.random.Generator
    ) -> NDArray[np.float64]:
        """Full attitude output: ``q_meas = q_true ⊗ δq(σ n)`` with ``n ~ N(0, I₃)``.

        The perturbation is applied on the body side, matching the error
        definition used by the MEKF (``q = q̂ ⊗ δq``).
        """
        dq = quat_from_small_angle(self.sigma_rad * rng.standard_normal(3))
        return quat_normalize(quat_multiply(quat_normalize(quat_true), dq))


@dataclass
class SunSensorModel:
    """Coarse sun sensor: one body-frame unit vector with eclipse handling.

    Parameters
    ----------
    sigma_rad : float
        Per-axis noise standard deviation [rad] (coarse sensors: 1e-2 rad).
    sun_vector_inertial : array_like, shape (3,)
        Inertial unit vector to the Sun (normalised on construction).
    fov_half_angle_rad : float, optional
        Half-angle of the sensor's conical field of view about the +Z body
        axis [rad].  A measurement is invalid when the true Sun vector falls
        outside it.  Default π (always in view).
    eclipse_prob : float, optional
        Per-epoch probability of eclipse.  Default 0.
    """

    sigma_rad: float
    sun_vector_inertial: ArrayLike = (1.0, 0.0, 0.0)
    fov_half_angle_rad: float = float(np.pi)
    eclipse_prob: float = 0.0
    _sun: NDArray[np.float64] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.sigma_rad = _positive("sigma_rad", self.sigma_rad)
        fov = float(self.fov_half_angle_rad)
        if not np.isfinite(fov) or not (0.0 < fov <= np.pi):
            raise ValueError(
                f"fov_half_angle_rad must be in (0, pi], got {self.fov_half_angle_rad!r}"
            )
        self.fov_half_angle_rad = fov
        p = float(self.eclipse_prob)
        if not np.isfinite(p) or not (0.0 <= p <= 1.0):
            raise ValueError(f"eclipse_prob must be in [0, 1], got {self.eclipse_prob!r}")
        self.eclipse_prob = p
        s = np.asarray(self.sun_vector_inertial, dtype=float).reshape(3)
        n = float(np.linalg.norm(s))
        if n < 1e-12:
            raise ValueError("sun_vector_inertial must have norm >= 1e-12")
        self._sun = s / n

    @property
    def sun_inertial(self) -> NDArray[np.float64]:
        """Normalised inertial Sun unit vector."""
        return self._sun.copy()

    def sample(self, quat_true: ArrayLike, rng: np.random.Generator) -> VectorOutput:
        """Observe the Sun at the true attitude, honouring FOV and eclipse."""
        b_true = dcm_from_quat(quat_true).T @ self._sun
        eclipsed = self.eclipse_prob > 0.0 and bool(rng.random() < self.eclipse_prob)
        in_fov = float(np.arccos(np.clip(b_true[2], -1.0, 1.0))) <= self.fov_half_angle_rad
        if eclipsed or not in_fov:
            return VectorOutput(np.full(3, np.nan), self._sun, valid=False)
        err = quat_from_small_angle(self.sigma_rad * rng.standard_normal(3))
        b = dcm_from_quat(err).T @ b_true
        return VectorOutput(b / np.linalg.norm(b), self._sun, valid=True)


@dataclass
class AccelerometerModel:
    """Triad accelerometer measuring specific force in body axes.

    ``f_body = R(q)ᵀ (a_inertial − g_inertial) + b_a + η_a``

    Parameters
    ----------
    sigma_a : float
        White noise standard deviation of the sampled measurement [m/s²].
        For a velocity-random-walk figure ``VRW`` [m/s/√s] at step ``Δt``,
        ``sigma_a = VRW/√Δt``.
    bias : array_like, shape (3,), optional
        Constant turn-on bias [m/s²].  Default zero.
    gravity_inertial : array_like, shape (3,), optional
        Inertial gravity vector [m/s²].  Default zero (orbital free fall — a
        strapdown accelerometer in free fall measures only non-gravitational
        specific force).
    """

    sigma_a: float
    bias: ArrayLike = (0.0, 0.0, 0.0)
    gravity_inertial: ArrayLike = (0.0, 0.0, 0.0)
    _bias: NDArray[np.float64] = field(init=False, repr=False)
    _g: NDArray[np.float64] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.sigma_a = _positive("sigma_a", self.sigma_a)
        b = np.asarray(self.bias, dtype=float).reshape(3)
        g = np.asarray(self.gravity_inertial, dtype=float).reshape(3)
        if not (np.all(np.isfinite(b)) and np.all(np.isfinite(g))):
            raise ValueError("bias and gravity_inertial must be finite")
        self._bias, self._g = b.copy(), g.copy()

    def sample(
        self, quat_true: ArrayLike, accel_inertial: ArrayLike, rng: np.random.Generator
    ) -> NDArray[np.float64]:
        """Draw one specific-force sample [m/s²] in body axes."""
        a = np.asarray(accel_inertial, dtype=float).reshape(3)
        if not np.all(np.isfinite(a)):
            raise ValueError("accel_inertial must be finite")
        f_i = a - self._g
        return dcm_from_quat(quat_true).T @ f_i + self._bias + self.sigma_a * rng.standard_normal(3)


@dataclass
class GpsModel:
    """GPS-like inertial position (and optionally velocity) fix.

    Parameters
    ----------
    sigma_pos : float
        Per-axis position error standard deviation [m].
    sigma_vel : float, optional
        Per-axis velocity error standard deviation [m/s].  If ``None`` (the
        default) no velocity is reported.
    dropout_prob : float, optional
        Per-epoch outage probability (urban canyon, antenna occultation).
        Default 0.

    Notes
    -----
    Errors are white and isotropic.  Real GNSS errors are correlated in time
    (ionosphere, multipath) and anisotropic (DOP geometry); neither is
    modelled.  Stated in README Limitations.
    """

    sigma_pos: float
    sigma_vel: float | None = None
    dropout_prob: float = 0.0

    def __post_init__(self) -> None:
        self.sigma_pos = _positive("sigma_pos", self.sigma_pos, allow_zero=False)
        if self.sigma_vel is not None:
            self.sigma_vel = _positive("sigma_vel", self.sigma_vel, allow_zero=False)
        p = float(self.dropout_prob)
        if not np.isfinite(p) or not (0.0 <= p <= 1.0):
            raise ValueError(f"dropout_prob must be in [0, 1], got {self.dropout_prob!r}")
        self.dropout_prob = p

    @property
    def measurement_dim(self) -> int:
        """3 for position-only, 6 when velocity is reported."""
        return 3 if self.sigma_vel is None else 6

    def noise_covariance(self) -> NDArray[np.float64]:
        """Measurement noise covariance ``R`` [m², m²/s²] on the diagonal."""
        if self.sigma_vel is None:
            return self.sigma_pos**2 * np.eye(3)
        return np.diag(
            [self.sigma_pos**2] * 3 + [self.sigma_vel**2] * 3  # type: ignore[operator]
        )

    def sample(
        self,
        position_true: ArrayLike,
        rng: np.random.Generator,
        velocity_true: ArrayLike | None = None,
    ) -> GpsOutput:
        """Draw one fix; returns an invalid output on a dropout epoch."""
        p = np.asarray(position_true, dtype=float).reshape(3)
        if not np.all(np.isfinite(p)):
            raise ValueError("position_true must be finite")
        if self.dropout_prob > 0.0 and bool(rng.random() < self.dropout_prob):
            return GpsOutput(np.full(3, np.nan), None if self.sigma_vel is None else
                             np.full(3, np.nan), valid=False)
        pos = p + self.sigma_pos * rng.standard_normal(3)
        if self.sigma_vel is None:
            return GpsOutput(pos, None, valid=True)
        if velocity_true is None:
            raise ValueError("velocity_true is required when sigma_vel is set")
        v = np.asarray(velocity_true, dtype=float).reshape(3)
        if not np.all(np.isfinite(v)):
            raise ValueError("velocity_true must be finite")
        return GpsOutput(pos, v + self.sigma_vel * rng.standard_normal(3), valid=True)

"""Baseline 2: least-squares and Kalman estimation of offset, skew and drift.

The model being fitted
----------------------
The per-exchange symmetric offset estimates of :mod:`timetransfer.twoway` are
noisy, biased samples of the remote clock's time error,

    theta_hat_k = theta0 + a (t_k - t_ref) + 0.5 D (t_k - t_ref)^2
                  + A_k / 2 + e_k                                          [s]

where ``t_k`` is the exchange midpoint on the local timescale, ``A_k / 2`` is
the asymmetry bias of equation (5) in :mod:`timetransfer.twoway`, and ``e_k``
is timestamp quantisation and timestamping noise. Both estimators below fit the
polynomial; **neither removes the static part of ``A_k / 2``**, which is
constant and therefore absorbed into ``theta0`` exactly. That is not a defect
of the estimator, it is the identifiability result stated in
:mod:`timetransfer.linkmodel`: it is why the learned corrector in
:mod:`timetransfer.asym` exists and why its ceiling is what it is.

Least squares
-------------
Ordinary least squares on the design matrix ``[1, dt, dt^2 / 2]`` (or
``[1, dt]`` at order 1). The parameter covariance is the textbook
``sigma^2 (X'X)^-1`` with ``sigma^2 = RSS / (n - p)``; see Draper and Smith,
*Applied Regression Analysis*, 3rd ed. (1998), chapter 5. The standard
uncertainty of the fitted offset at an arbitrary epoch is
``sqrt(x' C x)`` for the design row ``x`` at that epoch, which is what
:meth:`LeastSquaresFit.offset_at` returns alongside the value.

The covariance is valid under independent, identically distributed residuals.
Queueing delay is serially correlated on a real link, which inflates the true
uncertainty above the OLS figure. :mod:`timetransfer.uncertainty` measures the
inflation from the data instead of assuming it away, and the measured coverage
is reported whether or not it is good.

Kalman
------
The two-state clock model: state ``[theta, a]``, transition

    F = [[1, dt], [0, 1]]

and process covariance

    Q = [[q1 dt + q2 dt^3 / 3,  q2 dt^2 / 2],
         [q2 dt^2 / 2,          q2 dt      ]]

with ``q1`` [s^2/s] the white-FM intensity and ``q2`` [1/s] the random-walk-FM
intensity. This is the standard clock model and the ``Q`` above is its exact
discrete-time process covariance; see Zucca and Tavella, "The clock model and
its relationship with the Allan and related variances", *IEEE Transactions on
Ultrasonics, Ferroelectrics and Frequency Control* 52(2):289-296 (2005),
section III, and Brown and Hwang, *Introduction to Random Signals and Applied
Kalman Filtering*, 4th ed. (2012), section 9.5 for the same model as a worked
example. A three-state variant adds drift ``D`` with

    F = [[1, dt, dt^2/2], [0, 1, dt], [0, 0, 1]]

and a third process intensity ``q3`` [1/s^3] on the drift random walk.

Measurement model: ``z_k = theta_k + v_k`` with ``R = r`` the measurement
variance. ``r`` is **not** taken from a datasheet: the default is computed from
the observed scatter of the symmetric offset estimates about a robust local
fit, which is what :func:`measurement_variance_from_data` does.

Validity range
--------------
Both estimators assume the polynomial order is adequate over the fitted span.
Order 2 over a span much longer than the oscillator's coherence time will
absorb low-frequency noise into the drift term and understate the uncertainty;
the Allan deviation in :mod:`timetransfer.allan` is the diagnostic for that,
not these estimators.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .errors import ConfigurationError, InsufficientDataError

__all__ = [
    "KalmanConfig",
    "KalmanResult",
    "LeastSquaresFit",
    "fit_least_squares",
    "kalman_skew",
    "measurement_variance_from_data",
]


def _design(dt: np.ndarray, order: int) -> np.ndarray:
    """Design matrix for the clock polynomial at ``order`` 1 or 2."""
    if order == 1:
        return np.column_stack([np.ones_like(dt), dt])
    if order == 2:
        return np.column_stack([np.ones_like(dt), dt, 0.5 * dt * dt])
    raise ConfigurationError(f"polynomial order must be 1 or 2, got {order!r}")


@dataclass(frozen=True)
class LeastSquaresFit:
    """An ordinary least-squares clock-polynomial fit.

    Attributes
    ----------
    params:
        ``[theta0, a]`` at order 1 or ``[theta0, a, D]`` at order 2, in
        ``[s, s/s, 1/s]``.
    covariance:
        Parameter covariance ``sigma^2 (X'X)^-1``, same ordering.
    residuals:
        Fit residuals [s].
    rss:
        Residual sum of squares [s^2].
    sigma_s:
        Residual standard deviation ``sqrt(RSS / (n - p))`` [s].
    n, order, dof:
        Points used, polynomial order, and ``n - p``.
    t_ref_s:
        Epoch the polynomial is expanded about [s].
    """

    params: np.ndarray
    covariance: np.ndarray
    residuals: np.ndarray
    rss: float
    sigma_s: float
    n: int
    order: int
    dof: int
    t_ref_s: float

    @property
    def offset0_s(self) -> float:
        """Fitted offset at ``t_ref_s`` [s]."""
        return float(self.params[0])

    @property
    def skew_s_per_s(self) -> float:
        """Fitted fractional frequency offset [s/s]."""
        return float(self.params[1])

    @property
    def drift_per_s(self) -> float:
        """Fitted drift [1/s]; ``0.0`` at order 1."""
        return float(self.params[2]) if self.order >= 2 else 0.0

    @property
    def offset0_stderr_s(self) -> float:
        """Standard uncertainty of the fitted offset at ``t_ref_s`` [s]."""
        return float(np.sqrt(max(0.0, self.covariance[0, 0])))

    @property
    def skew_stderr_s_per_s(self) -> float:
        """Standard uncertainty of the fitted skew [s/s]."""
        return float(np.sqrt(max(0.0, self.covariance[1, 1])))

    def offset_at(self, t_s: float) -> tuple[float, float]:
        """Fitted offset [s] and its standard uncertainty [s] at local time ``t_s``.

        The uncertainty is ``sqrt(x' C x)`` for the design row ``x`` at ``t_s``;
        it is the *statistical* term only and excludes quantisation and
        asymmetry. :mod:`timetransfer.uncertainty` combines all three.
        """
        dt = np.asarray([float(t_s) - self.t_ref_s], dtype=np.float64)
        row = _design(dt, self.order)[0]
        value = float(row @ self.params)
        variance = float(row @ self.covariance @ row)
        return value, float(np.sqrt(max(0.0, variance)))

    def detrended(self, t_s: np.ndarray, y_s: np.ndarray) -> np.ndarray:
        """``y_s`` with the fitted polynomial removed [s].

        This is the input the Allan deviation wants: the deterministic
        polynomial must come out before stability is computed, or the drift
        term dominates every long tau.
        """
        dt = np.asarray(t_s, dtype=np.float64) - self.t_ref_s
        return np.asarray(y_s, dtype=np.float64) - _design(dt, self.order) @ self.params

    def as_text(self) -> str:
        """Compact human-readable summary."""
        lines = [
            f"n points        : {self.n} (order {self.order}, dof {self.dof})",
            f"offset at t_ref : {self.offset0_s:.9e} +- {self.offset0_stderr_s:.3e} s",
            f"skew            : {self.skew_s_per_s:.9e} +- "
            f"{self.skew_stderr_s_per_s:.3e} s/s",
        ]
        if self.order >= 2:
            drift_err = float(np.sqrt(max(0.0, self.covariance[2, 2])))
            lines.append(f"drift           : {self.drift_per_s:.9e} +- {drift_err:.3e} 1/s")
        lines.append(f"residual sigma  : {self.sigma_s:.6e} s")
        return "\n".join(lines)


def fit_least_squares(
    t_s: np.ndarray, offset_s: np.ndarray, *, order: int = 2, t_ref_s: float | None = None
) -> LeastSquaresFit:
    """Least-squares clock-polynomial fit of ``offset_s`` against ``t_s``.

    Parameters
    ----------
    t_s:
        Local-clock epochs of the offset estimates [s].
    offset_s:
        Offset estimates [s], same length.
    order:
        1 for offset and skew, 2 to add drift.
    t_ref_s:
        Epoch to expand about [s]; defaults to the mean of ``t_s``, which
        decorrelates the offset and skew columns and is why the reported
        ``offset0_stderr_s`` is the smallest honest one available.

    Returns
    -------
    LeastSquaresFit

    Raises
    ------
    InsufficientDataError
        Fewer points than parameters plus one.
    """
    t = np.asarray(t_s, dtype=np.float64).reshape(-1)
    y = np.asarray(offset_s, dtype=np.float64).reshape(-1)
    if t.size != y.size:
        raise ConfigurationError(f"t_s and offset_s must match in length: {t.size} vs {y.size}")
    n_params = 2 if int(order) == 1 else 3
    if int(order) not in (1, 2):
        raise ConfigurationError(f"polynomial order must be 1 or 2, got {order!r}")
    if t.size < n_params + 1:
        raise InsufficientDataError(
            f"order-{order} fit needs at least {n_params + 1} points, got {t.size}",
            n_usable=int(t.size),
            n_required=n_params + 1,
        )
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(y)):
        raise ConfigurationError("t_s and offset_s must be finite")
    if float(np.ptp(t)) == 0.0:
        # Every epoch identical makes the design matrix rank deficient. The
        # pseudo-inverse would return a minimum-norm solution and a covariance
        # with a zero on the skew diagonal, i.e. a skew of exactly zero
        # reported with zero uncertainty. That is the worst available
        # behaviour, so it is refused. Found by a Hypothesis probe during the
        # build and pinned in tests/test_regressions.py.
        raise ConfigurationError(
            f"cannot fit a clock polynomial when every epoch is identical "
            f"(all {t.size} values are {float(t[0])!r} s); the skew is not identifiable"
        )
    ref = float(np.mean(t)) if t_ref_s is None else float(t_ref_s)
    dt = t - ref
    design = _design(dt, int(order))
    coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coeffs
    dof = int(t.size) - n_params
    rss = float(np.sum(resid * resid))
    sigma2 = rss / dof
    gram = design.T @ design
    cov = sigma2 * np.linalg.pinv(gram)
    return LeastSquaresFit(
        params=np.asarray(coeffs, dtype=np.float64),
        covariance=np.asarray(cov, dtype=np.float64),
        residuals=resid,
        rss=rss,
        sigma_s=float(np.sqrt(sigma2)),
        n=int(t.size),
        order=int(order),
        dof=dof,
        t_ref_s=ref,
    )


def measurement_variance_from_data(
    offset_s: np.ndarray, *, window: int = 5, floor_variance_s2: float = 1e-24
) -> float:
    """Measurement variance [s^2] estimated from the data, not from a datasheet.

    Takes the median absolute deviation of the offset estimates about a running
    median of width ``window`` and converts it with the Gaussian consistency
    factor ``1 / 0.6744897501960817`` (the 0.75 quantile of the standard
    normal; Rousseeuw and Croux, "Alternatives to the median absolute
    deviation", *Journal of the American Statistical Association*
    88(424):1273-1283, 1993). A running median is used rather than the global
    mean so that skew and drift do not inflate the estimate.

    Parameters
    ----------
    offset_s:
        Offset estimates [s], at least ``window`` of them.
    window:
        Odd running-median width, >= 3.
    floor_variance_s2:
        Lower bound on the returned variance [s^2], > 0. The default of
        ``1e-24`` exists only to keep a Kalman filter from dividing by zero and
        is **not** a physically meaningful floor. On a link with no queueing
        jitter the running-median residual is dominated by timestamp
        quantisation, the median absolute deviation collapses, and the returned
        variance hits this floor: ``validation/validate_skew_estimators.py``
        measures a mean normalised innovation of order ``1e5`` in that regime,
        i.e. a filter over-confident by a factor of several hundred. A caller on
        a quiet link should pass either the quantisation variance
        ``(delta/sqrt(12))**2`` here or an explicit
        :attr:`KalmanConfig.measurement_variance_s2`.

    Returns
    -------
    float
        Variance [s^2], never below ``floor_variance_s2``.
    """
    y = np.asarray(offset_s, dtype=np.float64).reshape(-1)
    w = int(window)
    if w < 3 or w % 2 == 0:
        raise ConfigurationError(f"window must be odd and >= 3, got {window!r}")
    if y.size < w:
        raise InsufficientDataError(
            f"measurement variance needs at least {w} points, got {y.size}",
            n_usable=int(y.size),
            n_required=w,
        )
    half = w // 2
    padded = np.pad(y, (half, half), mode="edge")
    strides = np.lib.stride_tricks.sliding_window_view(padded, w)
    running = np.median(strides, axis=-1)
    resid = y - running
    if not (float(floor_variance_s2) > 0.0):
        raise ConfigurationError(
            f"floor_variance_s2 must be > 0, got {floor_variance_s2!r}"
        )
    mad = float(np.median(np.abs(resid - np.median(resid))))
    sigma = mad / 0.6744897501960817
    return float(max(sigma * sigma, float(floor_variance_s2)))


@dataclass(frozen=True)
class KalmanConfig:
    """Process and measurement parameters for :func:`kalman_skew`.

    Attributes
    ----------
    q1_s2_per_s:
        White-FM intensity [s^2/s], >= 0.
    q2_per_s:
        Random-walk-FM intensity [1/s], >= 0.
    q3_per_s3:
        Drift random-walk intensity [1/s^3], >= 0; used only when
        ``n_states == 3``.
    measurement_variance_s2:
        ``R`` [s^2], > 0. ``None`` means take it from the data with
        :func:`measurement_variance_from_data`.
    n_states:
        2 for ``[theta, a]``, 3 for ``[theta, a, D]``.
    initial_offset_variance_s2, initial_skew_variance, initial_drift_variance:
        Diagonal of ``P0``. The defaults are deliberately loose so the filter
        is driven by the data rather than by the prior.
    """

    q1_s2_per_s: float = 0.0
    q2_per_s: float = 0.0
    q3_per_s3: float = 0.0
    measurement_variance_s2: float | None = None
    n_states: int = 2
    initial_offset_variance_s2: float = 1.0
    initial_skew_variance: float = 1e-6
    initial_drift_variance: float = 1e-18

    def __post_init__(self) -> None:
        for name in ("q1_s2_per_s", "q2_per_s", "q3_per_s3"):
            if float(getattr(self, name)) < 0.0:
                raise ConfigurationError(f"{name} must be >= 0, got {getattr(self, name)!r}")
        if self.n_states not in (2, 3):
            raise ConfigurationError(f"n_states must be 2 or 3, got {self.n_states!r}")
        if self.measurement_variance_s2 is not None and not (
            float(self.measurement_variance_s2) > 0.0
        ):
            raise ConfigurationError(
                f"measurement_variance_s2 must be > 0, got {self.measurement_variance_s2!r}"
            )
        for name in (
            "initial_offset_variance_s2",
            "initial_skew_variance",
            "initial_drift_variance",
        ):
            if not (float(getattr(self, name)) > 0.0):
                raise ConfigurationError(f"{name} must be > 0, got {getattr(self, name)!r}")


@dataclass(frozen=True)
class KalmanResult:
    """Filtered states and covariances from :func:`kalman_skew`.

    Attributes
    ----------
    t_s:
        Epochs [s].
    states:
        ``(n, n_states)`` filtered state estimates; column 0 is offset [s],
        column 1 skew [s/s], column 2 drift [1/s] when present.
    covariances:
        ``(n, n_states, n_states)`` filtered covariances.
    innovations:
        Measurement residuals [s].
    innovation_variances:
        Predicted innovation variances [s^2]; the normalised innovation
        squared, ``innovation^2 / innovation_variance``, is the filter's own
        consistency diagnostic and is reported by
        :attr:`normalised_innovation_mean`.
    measurement_variance_s2:
        The ``R`` actually used [s^2], whether supplied or taken from the data.
    config:
        The configuration used.
    """

    t_s: np.ndarray
    states: np.ndarray
    covariances: np.ndarray
    innovations: np.ndarray
    innovation_variances: np.ndarray
    measurement_variance_s2: float
    config: KalmanConfig

    @property
    def final_offset_s(self) -> float:
        """Filtered offset at the last epoch [s]."""
        return float(self.states[-1, 0])

    @property
    def final_offset_stderr_s(self) -> float:
        """Standard uncertainty of the filtered offset at the last epoch [s]."""
        return float(np.sqrt(max(0.0, self.covariances[-1, 0, 0])))

    @property
    def final_skew_s_per_s(self) -> float:
        """Filtered skew at the last epoch [s/s]."""
        return float(self.states[-1, 1])

    @property
    def final_skew_stderr_s_per_s(self) -> float:
        """Standard uncertainty of the filtered skew at the last epoch [s/s]."""
        return float(np.sqrt(max(0.0, self.covariances[-1, 1, 1])))

    @property
    def normalised_innovation_mean(self) -> float:
        """Mean of ``innovation^2 / innovation_variance``.

        A consistent filter gives a value near 1.0. Much above 1 means the
        filter is over-confident -- its ``Q`` or ``R`` is too small; much below
        means it is ignoring the data. This is reported rather than tuned
        against, because tuning a filter until its diagnostic looks right and
        then quoting the diagnostic is circular.
        """
        return float(np.mean(self.innovations**2 / self.innovation_variances))

    def as_text(self) -> str:
        """Compact human-readable summary."""
        return (
            f"n points              : {self.t_s.size} ({self.config.n_states} states)\n"
            f"R used                : {self.measurement_variance_s2:.6e} s^2\n"
            f"final offset          : {self.final_offset_s:.9e} +- "
            f"{self.final_offset_stderr_s:.3e} s\n"
            f"final skew            : {self.final_skew_s_per_s:.9e} +- "
            f"{self.final_skew_stderr_s_per_s:.3e} s/s\n"
            f"mean normalised innov : {self.normalised_innovation_mean:.4f} "
            f"(1.0 is consistent)"
        )


def _transition_and_q(dt: float, cfg: KalmanConfig) -> tuple[np.ndarray, np.ndarray]:
    """``(F, Q)`` for a step of ``dt`` seconds."""
    q1 = float(cfg.q1_s2_per_s)
    q2 = float(cfg.q2_per_s)
    if cfg.n_states == 2:
        f = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float64)
        q = np.array(
            [
                [q1 * dt + q2 * dt**3 / 3.0, q2 * dt**2 / 2.0],
                [q2 * dt**2 / 2.0, q2 * dt],
            ],
            dtype=np.float64,
        )
        return f, q
    q3 = float(cfg.q3_per_s3)
    f = np.array(
        [[1.0, dt, 0.5 * dt * dt], [0.0, 1.0, dt], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    q = np.array(
        [
            [
                q1 * dt + q2 * dt**3 / 3.0 + q3 * dt**5 / 20.0,
                q2 * dt**2 / 2.0 + q3 * dt**4 / 8.0,
                q3 * dt**3 / 6.0,
            ],
            [
                q2 * dt**2 / 2.0 + q3 * dt**4 / 8.0,
                q2 * dt + q3 * dt**3 / 3.0,
                q3 * dt**2 / 2.0,
            ],
            [q3 * dt**3 / 6.0, q3 * dt**2 / 2.0, q3 * dt],
        ],
        dtype=np.float64,
    )
    return f, q


def kalman_skew(
    t_s: np.ndarray,
    offset_s: np.ndarray,
    *,
    config: KalmanConfig | None = None,
) -> KalmanResult:
    """Run the clock-model Kalman filter over per-exchange offset estimates.

    Parameters
    ----------
    t_s:
        Local-clock epochs [s], non-decreasing.
    offset_s:
        Offset estimates [s], same length.
    config:
        Process and measurement parameters; the default is a two-state filter
        with zero process noise and ``R`` taken from the data, which reduces to
        a recursive least-squares fit of a straight line.

    Returns
    -------
    KalmanResult
    """
    t = np.asarray(t_s, dtype=np.float64).reshape(-1)
    y = np.asarray(offset_s, dtype=np.float64).reshape(-1)
    if t.size != y.size:
        raise ConfigurationError(f"t_s and offset_s must match in length: {t.size} vs {y.size}")
    cfg = KalmanConfig() if config is None else config
    if t.size < cfg.n_states + 1:
        raise InsufficientDataError(
            f"{cfg.n_states}-state filter needs at least {cfg.n_states + 1} points, "
            f"got {t.size}",
            n_usable=int(t.size),
            n_required=cfg.n_states + 1,
        )
    if np.any(np.diff(t) < 0.0):
        raise ConfigurationError("t_s must be non-decreasing")
    if cfg.measurement_variance_s2 is None:
        window = 5 if t.size >= 5 else 3
        r = measurement_variance_from_data(y, window=window)
    else:
        r = float(cfg.measurement_variance_s2)

    ns = int(cfg.n_states)
    x = np.zeros(ns, dtype=np.float64)
    x[0] = float(y[0])
    p = np.diag(
        [cfg.initial_offset_variance_s2, cfg.initial_skew_variance, cfg.initial_drift_variance][
            :ns
        ]
    ).astype(np.float64)
    h = np.zeros((1, ns), dtype=np.float64)
    h[0, 0] = 1.0
    states = np.zeros((t.size, ns), dtype=np.float64)
    covs = np.zeros((t.size, ns, ns), dtype=np.float64)
    innov = np.zeros(t.size, dtype=np.float64)
    innov_var = np.zeros(t.size, dtype=np.float64)

    for k in range(int(t.size)):
        if k > 0:
            dt = float(t[k] - t[k - 1])
            f, q = _transition_and_q(dt, cfg)
            x = f @ x
            p = f @ p @ f.T + q
        s = float((h @ p @ h.T)[0, 0]) + r
        resid = float(y[k]) - float((h @ x)[0])
        gain = (p @ h.T / s).reshape(ns)
        x = x + gain * resid
        p = p - np.outer(gain, h @ p)
        p = 0.5 * (p + p.T)
        states[k] = x
        covs[k] = p
        innov[k] = resid
        innov_var[k] = s

    return KalmanResult(
        t_s=t,
        states=states,
        covariances=covs,
        innovations=innov,
        innovation_variances=innov_var,
        measurement_variance_s2=r,
        config=cfg,
    )

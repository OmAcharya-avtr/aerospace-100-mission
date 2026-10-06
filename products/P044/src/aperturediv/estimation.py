"""Imperfect and stale channel-state estimates.

MRC with the true channel state is optimal by construction
(:mod:`aperturediv.combining`), so the only question a learned combiner can
honestly be asked is what to do when the state it is given is wrong. This
module is the error model, kept separate from the combiners so that no
combiner can see the truth by accident.

Error model
-----------
The estimate of branch ``k`` is formed from the channel as it was ``tau``
seconds ago, plus measurement noise, both in the log domain:

``ln Ihat_k = ln I_k(t - tau) + sigma_m u_k + c``

with ``u_k ~ N(0, 1)`` independent across branches and ``c`` a constant
chosen so that ``E[Ihat_k] = 1``, i.e. the estimate is unbiased in the mean.
The stale sample ``I_k(t - tau)`` is drawn jointly with ``I_k(t)``, with
temporal log-correlation ``rho_t = corr(ln I(t), ln I(t - tau))`` — one
scalar standing in for the temporal coherence of the scintillation, which
for a horizontal link is set by the transverse wind speed.

Error level
-----------
Both mechanisms enter only through the total log-domain error standard
deviation

``sigma_e = sqrt(2 s^2 (1 - rho_t) + sigma_m^2)``,  ``s^2 = ln(1 + si)``

derived in :func:`log_error_sigma` from
``var(ln Ihat - ln I) = var(s Z_then - s Z_now) + sigma_m^2``. So a stale
estimate and a noisy one are the same thing to the combiner at equal
``sigma_e``, which is why the sweeps in this package are parameterised by
``sigma_e`` alone and why staleness and measurement noise are not reported
as separate axes. ``sigma_e`` is reported in dB of irradiance,
``sigma_e_dB = (10 / ln 10) sigma_e`` (:func:`log_error_sigma_db`), because
that is the number a receiver designer has a feel for.

The equivalence is checked numerically in
``validation/validate_learned_combiner.py``.

Nothing here is flight-qualified or certified.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .channel import lognormal_sigma_log

__all__ = [
    "DB_PER_NEPER",
    "ChannelEstimate",
    "estimate_from_log_error",
    "estimate_stale_noisy",
    "log_error_sigma",
    "log_error_sigma_db",
    "measurement_sigma_for_target",
]

DB_PER_NEPER = 10.0 / np.log(10.0)  # 4.342944819..., irradiance dB per natural log unit


def log_error_sigma(si: float, temporal_correlation: float, measurement_sigma: float) -> float:
    """Total log-domain estimation error standard deviation ``sigma_e``.

    ``sigma_e = sqrt(2 s^2 (1 - rho_t) + sigma_m^2)`` with
    ``s = sqrt(ln(1 + si))``.

    Parameters
    ----------
    si
        Scintillation index of the marginal irradiance, > 0.
    temporal_correlation
        ``rho_t`` in ``[-1, 1]``; 1 means a fresh estimate.
    measurement_sigma
        ``sigma_m >= 0``, log-domain measurement noise standard deviation.
    """
    s2 = lognormal_sigma_log(si) ** 2
    rho = float(temporal_correlation)
    if not -1.0 <= rho <= 1.0:
        raise ValueError(f"temporal_correlation must lie in [-1, 1], got {rho!r}")
    sm = float(measurement_sigma)
    if not np.isfinite(sm) or sm < 0.0:
        raise ValueError(f"measurement_sigma must be finite and >= 0, got {sm!r}")
    return float(np.sqrt(2.0 * s2 * (1.0 - rho) + sm * sm))


def log_error_sigma_db(sigma_e: float) -> float:
    """Convert a log-domain error ``sigma_e`` to dB of irradiance."""
    return float(DB_PER_NEPER * float(sigma_e))


def measurement_sigma_for_target(si: float, temporal_correlation: float, target: float) -> float:
    """Measurement noise that gives a wanted total ``sigma_e``.

    Inverse of :func:`log_error_sigma` in ``sigma_m``. Raises if staleness
    alone already exceeds the target, which is a real condition — no
    estimator quality can recover a channel that has already decorrelated.
    """
    s2 = lognormal_sigma_log(si) ** 2
    rho = float(temporal_correlation)
    tgt = float(target)
    if tgt < 0.0:
        raise ValueError(f"target must be >= 0, got {tgt!r}")
    residual = tgt * tgt - 2.0 * s2 * (1.0 - rho)
    if residual < 0.0:
        raise ValueError(
            f"staleness alone gives sigma_e = {np.sqrt(2 * s2 * (1 - rho)):.4f}, "
            f"already above the target {tgt:.4f}; no measurement noise reaches it"
        )
    return float(np.sqrt(residual))


@dataclass(frozen=True)
class ChannelEstimate:
    """Paired truth and estimate for one batch of realisations.

    Attributes
    ----------
    irradiance_true
        Shape ``(n, L)``, the state at the time the symbol is received.
    irradiance_estimated
        Shape ``(n, L)``, what the receiver has, mean-unbiased.
    sigma_e
        Shape ``(n,)``, per-realisation log-domain error standard deviation.
    """

    irradiance_true: np.ndarray
    irradiance_estimated: np.ndarray
    sigma_e: np.ndarray

    @property
    def amplitude_true(self) -> np.ndarray:
        """``h = sqrt(I)``, shape ``(n, L)``."""
        return np.sqrt(self.irradiance_true)

    @property
    def sigma_e_db(self) -> np.ndarray:
        """Per-realisation error standard deviation in dB of irradiance."""
        return DB_PER_NEPER * self.sigma_e


def estimate_from_log_error(
    irradiance_true: np.ndarray,
    sigma_e: np.ndarray | float,
    rng: np.random.Generator,
) -> ChannelEstimate:
    """Estimate with a given total log-domain error, per realisation.

    ``ln Ihat = ln I + sigma_e u - sigma_e^2 / 2``, ``u ~ N(0, 1)``
    independent across branches. The ``-sigma_e^2/2`` term makes
    ``E[Ihat | I] = I``, so the estimate carries no systematic bias and any
    loss is attributable to its variance alone.

    Parameters
    ----------
    irradiance_true
        Shape ``(n, L)``, strictly positive.
    sigma_e
        Scalar, or shape ``(n,)`` for a per-realisation error level.
    rng
        ``numpy.random.Generator``.
    """
    i_true = np.asarray(irradiance_true, dtype=float)
    if i_true.ndim == 1:
        i_true = i_true[:, None]
    if i_true.ndim != 2:
        raise ValueError("irradiance_true must have shape (n, L)")
    if np.any(i_true <= 0.0):
        raise ValueError("irradiance_true must be strictly positive")
    se = np.asarray(sigma_e, dtype=float)
    if se.ndim == 0:
        se = np.full(i_true.shape[0], float(se))
    if se.shape != (i_true.shape[0],):
        raise ValueError("sigma_e must be a scalar or have shape (n,)")
    if np.any(se < 0.0) or np.any(~np.isfinite(se)):
        raise ValueError("sigma_e must be finite and >= 0")
    u = rng.standard_normal(i_true.shape)
    col = se[:, None]
    i_hat = i_true * np.exp(col * u - 0.5 * col * col)
    return ChannelEstimate(irradiance_true=i_true, irradiance_estimated=i_hat, sigma_e=se)


def estimate_stale_noisy(
    n_samples: int,
    si: float,
    log_correlation: np.ndarray,
    temporal_correlation: float,
    measurement_sigma: float,
    rng: np.random.Generator,
) -> ChannelEstimate:
    """Draw truth and a stale, noisy estimate from the full error model.

    The two lognormal fields (now, and ``tau`` ago) share the spatial
    correlation ``log_correlation`` and are linked in time by
    ``temporal_correlation``; measurement noise is then added per branch.
    Use this to confirm that the ``sigma_e`` parameterisation is sufficient;
    use :func:`estimate_from_log_error` for sweeps.
    """
    from .correlation import sample_correlated_lognormal  # local: avoids a cycle

    n = int(n_samples)
    s = lognormal_sigma_log(si)
    rho = float(temporal_correlation)
    if not -1.0 <= rho <= 1.0:
        raise ValueError(f"temporal_correlation must lie in [-1, 1], got {rho!r}")
    sm = float(measurement_sigma)
    if not np.isfinite(sm) or sm < 0.0:
        raise ValueError(f"measurement_sigma must be finite and >= 0, got {sm!r}")

    i_now = sample_correlated_lognormal(n, si, log_correlation, rng)
    z_now = (np.log(i_now) + 0.5 * s * s) / s
    # z_then shares the spatial correlation and is correlated in time with z_now.
    z_fresh = (
        np.log(sample_correlated_lognormal(n, si, log_correlation, rng)) + 0.5 * s * s
    ) / s
    z_then = rho * z_now + np.sqrt(max(1.0 - rho * rho, 0.0)) * z_fresh
    u = rng.standard_normal(z_now.shape)
    log_hat = s * z_then - 0.5 * s * s + sm * u - 0.5 * sm * sm
    se = log_error_sigma(si, rho, sm)
    return ChannelEstimate(
        irradiance_true=i_now,
        irradiance_estimated=np.exp(log_hat),
        sigma_e=np.full(n, se),
    )

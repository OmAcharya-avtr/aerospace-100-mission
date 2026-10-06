"""Channel-state estimates that are wrong, and the LLRs that account for it.

This module is the reason the package exists. A demapper needs the irradiance
``h``; a receiver has an *estimate* ``h_hat`` from pilots, from a slow power
monitor, or from a measurement that is one round trip old. Substituting
``h_hat`` into the known-CSI LLR of :func:`softdecode.llr.llr_ook_known_csi`
is what almost every implementation does, and it is wrong in a way that is
**asymmetric**: an estimate that is too high produces LLRs that are too
confident, and a decoder trusts them.

Error models
------------
``MultiplicativeCsiError(bias_db, jitter_db)``
    ``h_hat = h * exp(b + sigma_e * z)``, ``z ~ N(0, 1)``, with
    ``b = bias_db * ln(10) / 10`` and ``sigma_e = jitter_db * ln(10) / 10``.
    ``bias_db > 0`` over-estimates channel quality. This is the standard
    lognormal multiplicative gain-estimation error; it keeps ``h_hat > 0``,
    which matters because an irradiance estimate cannot be negative.

``StaleCsiError(correlation)``
    ``h_hat = exp(X_stale)`` where ``(log h, log h_hat)`` are jointly Gaussian
    with the same marginal and correlation ``rho``. This is the
    stale-estimate-of-age-tau case: for a log-amplitude process with
    autocorrelation ``rho(tau)``, the estimate taken ``tau`` ago has exactly
    this joint law. Only defined for lognormal fading, where ``log h`` is
    Gaussian.

Posterior-aware LLRs
--------------------
The LLR that is *optimal given what the receiver actually observes*, namely
``(y, h_hat)``, averages the likelihood over the posterior ``p(h | h_hat)``:

    p(y | b=1, h_hat) = integral N(y; a h, sigma**2) p(h | h_hat) dh

For lognormal fading both error models give a Gaussian posterior on
``log h`` in closed form, so the average is a Gauss-Hermite quadrature over
that posterior:

    MultiplicativeCsiError, lognormal prior ``log h ~ N(mu_x, sigma_x**2)``:
        s**2 = 1 / (1/sigma_x**2 + 1/sigma_e**2)
        m    = s**2 * ( mu_x / sigma_x**2 + (log h_hat - b) / sigma_e**2 )

    StaleCsiError, same prior:
        m    = mu_x + rho * (log h_hat - mu_x)
        s**2 = sigma_x**2 * (1 - rho**2)

For gamma-gamma fading no closed form exists, so
:func:`posterior_quadrature` builds the posterior numerically on a log-spaced
grid and returns it as a quadrature rule. The grid rule is checked against the
closed-form Gauss-Hermite rule on the lognormal case, where both apply, in
``validation/validate_quadrature.py``.

Honest accounting of information
--------------------------------
The posterior-aware LLR needs the error-model parameters (``bias_db``,
``jitter_db`` or ``rho``). It is therefore **not** free: it assumes the
receiver has characterised its own estimator. The learned corrector in
:mod:`softdecode.corrector` is given no such parameters and learns from pilot
bits instead. Every comparison in this repository states which information
each method used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .channel import GammaGammaFading, LognormalFading

_DB = np.log(10.0) / 10.0

__all__ = [
    "MultiplicativeCsiError",
    "StaleCsiError",
    "db_to_log_gain",
    "posterior_quadrature",
]


def db_to_log_gain(value_db: float) -> float:
    """Convert an irradiance offset in dB to a natural-log gain."""
    return float(value_db) * _DB


@dataclass(frozen=True)
class MultiplicativeCsiError:
    """Lognormal multiplicative channel-estimate error.

    Parameters
    ----------
    bias_db:
        Systematic irradiance error, dB. Positive over-estimates channel
        quality. May be zero.
    jitter_db:
        Standard deviation of the random part, dB. Must be >= 0; zero means a
        purely systematic error.
    """

    bias_db: float = 0.0
    jitter_db: float = 0.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.bias_db):
            raise ValueError(f"bias_db must be finite, got {self.bias_db!r}")
        if not np.isfinite(self.jitter_db) or self.jitter_db < 0.0:
            raise ValueError(f"jitter_db must be finite and >= 0, got {self.jitter_db!r}")

    @property
    def bias(self) -> float:
        """Natural-log bias ``b``."""
        return db_to_log_gain(self.bias_db)

    @property
    def jitter(self) -> float:
        """Natural-log jitter ``sigma_e``."""
        return db_to_log_gain(self.jitter_db)

    def estimate(self, h, rng: np.random.Generator) -> np.ndarray:
        """Draw ``h_hat`` given the true ``h``. One standard normal per sample."""
        h = np.asarray(h, dtype=float)
        if np.any(h < 0.0):
            raise ValueError("h must be non-negative")
        if self.jitter == 0.0:
            return h * np.exp(self.bias)
        return h * np.exp(self.bias + self.jitter * rng.standard_normal(h.shape))

    def log_posterior_moments(
        self, h_hat, fading: LognormalFading
    ) -> tuple[np.ndarray, np.ndarray]:
        """Mean and standard deviation of ``log h`` given ``h_hat``, lognormal prior."""
        if not isinstance(fading, LognormalFading):
            raise TypeError("closed-form posterior requires LognormalFading")
        o = np.log(np.asarray(h_hat, dtype=float)) - self.bias
        sx2 = fading.sigma_x**2
        if self.jitter == 0.0:
            return o, np.zeros_like(o)
        se2 = self.jitter**2
        s2 = 1.0 / (1.0 / sx2 + 1.0 / se2)
        m = s2 * (fading.mu_x / sx2 + o / se2)
        return m, np.full_like(m, np.sqrt(s2))

    def log_likelihood(self, h_hat, h) -> np.ndarray:
        """``log p(h_hat | h)`` up to a constant, for the numeric grid posterior."""
        if self.jitter == 0.0:
            raise ValueError("log_likelihood is undefined for a noiseless (jitter 0) estimate")
        lo = np.log(np.asarray(h_hat, dtype=float))
        lh = np.log(np.asarray(h, dtype=float))
        return -((lo - lh - self.bias) ** 2) / (2.0 * self.jitter**2)


@dataclass(frozen=True)
class StaleCsiError:
    """A channel estimate of stated age, as a correlated lognormal sample.

    Parameters
    ----------
    correlation:
        Correlation ``rho`` of ``log h`` between the measurement epoch and the
        decision epoch, in ``(0, 1]``. ``rho = 1`` is perfect CSI.
    """

    correlation: float = 1.0

    def __post_init__(self) -> None:
        r = float(self.correlation)
        if not np.isfinite(r) or not 0.0 < r <= 1.0:
            raise ValueError(f"correlation must lie in (0, 1], got {self.correlation!r}")

    def estimate(self, h, rng: np.random.Generator, fading: LognormalFading) -> np.ndarray:
        """Draw ``h_hat`` with the stated log-correlation to ``h``."""
        if not isinstance(fading, LognormalFading):
            raise TypeError("StaleCsiError is defined for LognormalFading only")
        h = np.asarray(h, dtype=float)
        x = np.log(h)
        rho = float(self.correlation)
        resid = np.sqrt(max(1.0 - rho * rho, 0.0)) * fading.sigma_x
        x_stale = fading.mu_x + rho * (x - fading.mu_x) + resid * rng.standard_normal(h.shape)
        return np.exp(x_stale)

    def log_posterior_moments(
        self, h_hat, fading: LognormalFading
    ) -> tuple[np.ndarray, np.ndarray]:
        """Mean and standard deviation of ``log h`` given a stale ``h_hat``."""
        if not isinstance(fading, LognormalFading):
            raise TypeError("StaleCsiError is defined for LognormalFading only")
        o = np.log(np.asarray(h_hat, dtype=float))
        rho = float(self.correlation)
        m = fading.mu_x + rho * (o - fading.mu_x)
        s = fading.sigma_x * np.sqrt(max(1.0 - rho * rho, 0.0))
        return m, np.full_like(m, s)


def posterior_quadrature(
    h_hat: float,
    fading: LognormalFading | GammaGammaFading,
    error: MultiplicativeCsiError | StaleCsiError,
    n: int = 32,
    grid_points: int = 241,
    grid_sigma: float = 6.0,
    weight_floor: float = 1e-14,
) -> tuple[np.ndarray, np.ndarray]:
    """Quadrature nodes and weights for ``p(h | h_hat)``.

    For :class:`LognormalFading` the closed-form Gaussian posterior on
    ``log h`` is used with ``n`` Gauss-Hermite nodes. For
    :class:`GammaGammaFading` the posterior is built on a ``grid_points``
    log-spaced grid spanning ``grid_sigma`` posterior-scale decades around the
    estimate and normalised by the trapezoid rule; nodes whose normalised
    weight falls below ``weight_floor`` are dropped and the rest renormalised.

    ``h_hat`` is a scalar. Returns ``(nodes, weights)`` with ``sum(w) = 1``.
    """
    hh = float(h_hat)
    if not np.isfinite(hh) or hh <= 0.0:
        raise ValueError(f"h_hat must be a finite positive number, got {h_hat!r}")
    if isinstance(fading, LognormalFading):
        m, s = error.log_posterior_moments(np.array(hh), fading)
        m = float(np.atleast_1d(m)[0])
        s = float(np.atleast_1d(s)[0])
        if s == 0.0:
            return np.array([np.exp(m)]), np.array([1.0])
        t, w = np.polynomial.hermite.hermgauss(int(n))
        return np.exp(m + np.sqrt(2.0) * s * t), w / np.sqrt(np.pi)
    if isinstance(fading, GammaGammaFading):
        if not isinstance(error, MultiplicativeCsiError):
            raise TypeError("gamma-gamma posteriors are implemented for MultiplicativeCsiError")
        if error.jitter == 0.0:
            return np.array([hh * np.exp(-error.bias)]), np.array([1.0])
        centre = np.log(hh) - error.bias
        span = grid_sigma * max(error.jitter, 0.25)
        lh = np.linspace(centre - span, centre + span, int(grid_points))
        h = np.exp(lh)
        log_prior = fading.log_pdf(h) + lh  # density in log h
        log_post = log_prior + error.log_likelihood(np.array(hh), h)
        log_post -= log_post.max()
        dens = np.exp(log_post)
        dlh = lh[1] - lh[0]
        w = dens * dlh
        w /= w.sum()
        keep = w > weight_floor
        if not np.any(keep):
            raise RuntimeError("posterior grid collapsed; widen grid_sigma")
        h, w = h[keep], w[keep]
        return h, w / w.sum()
    raise TypeError(f"unsupported fading model {type(fading).__name__}")


def _posterior_nodes_lognormal(
    h_hat: np.ndarray,
    fading: LognormalFading,
    error: MultiplicativeCsiError | StaleCsiError,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-sample posterior nodes ``(N, n)`` and shared weights ``(n,)``."""
    m, s = error.log_posterior_moments(h_hat, fading)
    scale = float(np.atleast_1d(s).reshape(-1)[0])
    t, w = np.polynomial.hermite.hermgauss(int(n))
    nodes = np.exp(m[:, None] + np.sqrt(2.0) * scale * t[None, :])
    return nodes, w / np.sqrt(np.pi)


def _posterior_grid_gammagamma(
    h_hat: np.ndarray,
    fading: GammaGammaFading,
    error: MultiplicativeCsiError,
    grid_points: int,
    grid_span: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Shared log-spaced grid ``(G,)`` with per-sample posterior weights ``(N, G)``."""
    if error.jitter == 0.0:
        raise ValueError("gamma-gamma posterior grid requires a nonzero jitter")
    lh = np.linspace(-grid_span, grid_span, int(grid_points))
    h = np.exp(lh)
    log_prior = fading.log_pdf(h) + lh
    lo = np.log(h_hat)[:, None] - error.bias
    log_post = log_prior[None, :] - (lo - lh[None, :]) ** 2 / (2.0 * error.jitter**2)
    log_post -= log_post.max(axis=1, keepdims=True)
    w = np.exp(log_post)
    w /= w.sum(axis=1, keepdims=True)
    return h, w


def csi_aware_llr_ook(
    y,
    amplitude: float,
    h_hat,
    fading: LognormalFading | GammaGammaFading,
    error: MultiplicativeCsiError | StaleCsiError,
    sigma: float = 1.0,
    nodes: int = 32,
    max_log: bool = False,
    grid_points: int = 161,
    grid_span: float = 5.0,
    chunk: int = 20000,
) -> np.ndarray:
    """OOK LLR that averages the likelihood over ``p(h | h_hat)``.

    This is the Bayes-optimal LLR given everything the receiver observes,
    ``(y, h_hat)``, under the stated error model. It needs the error-model
    parameters, which the learned corrector of :mod:`softdecode.corrector`
    does not.

    Parameters
    ----------
    y, h_hat:
        Equal-length 1-D arrays of samples and channel estimates.
    amplitude, sigma:
        As in :func:`softdecode.llr.llr_ook_known_csi`.
    nodes:
        Gauss-Hermite nodes for the lognormal closed-form posterior.
    max_log:
        Replace the ``logsumexp`` over posterior nodes by a ``max``.
    grid_points, grid_span:
        Log-spaced grid for the gamma-gamma posterior, spanning
        ``exp(-grid_span) .. exp(+grid_span)`` in ``h``.
    chunk:
        Samples processed per block, to bound peak memory.

    Returns
    -------
    ``(N,)`` LLR array, ``log P(0)/P(1)``.
    """
    from scipy import special  # local import keeps module import cheap

    y = np.asarray(y, dtype=float).ravel()
    hh = np.asarray(h_hat, dtype=float).ravel()
    if y.shape != hh.shape:
        raise ValueError(f"y and h_hat must have equal size, got {y.size} and {hh.size}")
    if np.any(hh <= 0.0):
        raise ValueError("h_hat must be strictly positive")
    a = float(amplitude)
    if not np.isfinite(a) or a <= 0.0:
        raise ValueError(f"amplitude must be a finite positive number, got {amplitude!r}")
    s = float(sigma)
    if not np.isfinite(s) or s <= 0.0:
        raise ValueError(f"sigma must be a finite positive number, got {sigma!r}")
    s2 = s * s
    reduce = np.max if max_log else (lambda x, axis: special.logsumexp(x, axis=axis))

    out = np.empty_like(y)
    for start in range(0, y.size, int(chunk)):
        sl = slice(start, start + int(chunk))
        yc, hc = y[sl], hh[sl]
        if isinstance(fading, LognormalFading):
            node, w = _posterior_nodes_lognormal(hc, fading, error, nodes)
            logw = np.log(w)[None, :]
        elif isinstance(fading, GammaGammaFading):
            if not isinstance(error, MultiplicativeCsiError):
                raise TypeError(
                    "gamma-gamma posteriors are implemented for MultiplicativeCsiError"
                )
            grid, wmat = _posterior_grid_gammagamma(hc, fading, error, grid_points, grid_span)
            node = np.broadcast_to(grid[None, :], (hc.size, grid.size))
            logw = np.log(np.maximum(wmat, 1e-300))
        else:
            raise TypeError(f"unsupported fading model {type(fading).__name__}")
        terms = logw - (yc[:, None] - a * node) ** 2 / (2.0 * s2)
        out[sl] = -(yc**2) / (2.0 * s2) - reduce(terms, axis=-1)
    return out

"""Fading-amplitude models for an atmospheric optical intensity channel.

Both models describe the *normalised irradiance* ``h >= 0`` of a single
receive aperture, with the normalisation ``E[h] = 1`` so that ``h`` multiplies
a nominal received amplitude rather than carrying an absolute level. Units:
``h`` is dimensionless.

Models
------
Lognormal
    ``h = exp(X)``, ``X ~ N(mu_x, sigma_x**2)``, with ``mu_x = -sigma_x**2 / 2``
    imposed by ``E[h] = 1``. The scintillation index is
    ``sigma_I**2 = Var[h] / E[h]**2 = exp(sigma_x**2) - 1``.
    Source: Andrews & Phillips, *Laser Beam Propagation through Random Media*,
    2nd ed., SPIE Press, 2005, chapter 9 (weak-fluctuation theory). Validity:
    weak turbulence, ``sigma_I**2`` up to roughly 0.3; the model is used here
    outside that range only as a parametric fading law, which the README says.

Gamma-gamma
    ``h = u * v`` with ``u ~ Gamma(alpha, 1/alpha)`` (large-scale) and
    ``v ~ Gamma(beta, 1/beta)`` (small-scale), independent, both unit-mean, so
    ``E[h] = 1``. The density is

        f(h) = 2 (alpha beta)**((alpha+beta)/2) / (Gamma(alpha) Gamma(beta))
               * h**((alpha+beta)/2 - 1) * K_{alpha-beta}(2 sqrt(alpha beta h))

    and the scintillation index is
    ``sigma_I**2 = 1/alpha + 1/beta + 1/(alpha beta)``.
    Source: M. A. Al-Habash, L. C. Andrews and R. L. Phillips, "Mathematical
    model for the irradiance probability density function of a laser beam
    propagating through turbulent media", *Optical Engineering* 40(8),
    1554-1562, 2001. Validity: ``alpha, beta > 0``; covers weak to strong
    turbulence.

Quadrature
----------
Every exact log-likelihood-ratio in :mod:`softdecode.llr` needs an expectation
over ``h``. Each model therefore exposes ``quadrature(n)`` returning nodes and
weights with ``sum(w) = 1`` and ``sum(w * h) = E[h] = 1`` to quadrature
accuracy:

* lognormal: Gauss-Hermite in ``X``, which is exact for polynomials in ``X``
  and converges geometrically for the smooth integrands used here;
* gamma-gamma: a trapezoid rule on the density in ``log h``, which converges
  to about 1e-8 in LLR at 201 points. The tensor product of two generalised
  Gauss-Laguerre rules is also provided
  (:meth:`GammaGammaFading.laguerre_quadrature`); it is exact for the moments
  but stalls near 1e-2 in LLR for small ``alpha, beta``, which is why it is
  not the default.

The quadrature is *verified against a Monte Carlo estimate of the same
quantity* in ``validation/validate_quadrature.py``; it is not assumed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import special

MAX_HERMITE_NODES = 300
"""Largest usable Gauss-Hermite rule; above this the weights underflow."""


def _check_positive(name: str, value: float) -> float:
    v = float(value)
    if not np.isfinite(v) or v <= 0.0:
        raise ValueError(f"{name} must be a finite positive number, got {value!r}")
    return v


@dataclass(frozen=True)
class LognormalFading:
    """Unit-mean lognormal irradiance.

    Parameters
    ----------
    sigma_i2:
        Scintillation index ``sigma_I**2 = Var[h]``, dimensionless, > 0.
    """

    sigma_i2: float

    def __post_init__(self) -> None:
        _check_positive("sigma_i2", self.sigma_i2)

    @property
    def sigma_x(self) -> float:
        """Standard deviation of ``log h``, dimensionless."""
        return float(np.sqrt(np.log1p(self.sigma_i2)))

    @property
    def mu_x(self) -> float:
        """Mean of ``log h``, fixed by ``E[h] = 1``: ``-sigma_x**2 / 2``."""
        return -0.5 * self.sigma_x**2

    @property
    def scintillation_index(self) -> float:
        """``Var[h] / E[h]**2``, dimensionless."""
        return float(self.sigma_i2)

    def mean(self) -> float:
        """``E[h]`` from the parameters (1 by construction)."""
        return float(np.exp(self.mu_x + 0.5 * self.sigma_x**2))

    def variance(self) -> float:
        """``Var[h]`` from the parameters."""
        return float(np.expm1(self.sigma_x**2) * np.exp(2 * self.mu_x + self.sigma_x**2))

    def pdf(self, h: np.ndarray) -> np.ndarray:
        """Density of ``h``, 1/dimensionless. Zero for ``h <= 0``."""
        h = np.asarray(h, dtype=float)
        out = np.zeros_like(h)
        pos = h > 0
        hp = h[pos]
        z = (np.log(hp) - self.mu_x) / self.sigma_x
        out[pos] = np.exp(-0.5 * z**2) / (hp * self.sigma_x * np.sqrt(2 * np.pi))
        return out

    def sample(self, size: int | tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
        """Draw ``h``. Draws one standard normal per sample."""
        z = rng.standard_normal(size)
        return np.exp(self.mu_x + self.sigma_x * z)

    def quadrature(self, n: int = 80) -> tuple[np.ndarray, np.ndarray]:
        """Gauss-Hermite nodes ``h`` and weights ``w``, ``sum(w) = 1``.

        ``n`` is capped at ``MAX_HERMITE_NODES`` because the Gauss-Hermite
        weights underflow to zero above roughly 300 nodes in double precision,
        which would silently corrupt the rule.
        """
        if int(n) < 2:
            raise ValueError(f"n must be at least 2, got {n!r}")
        if int(n) > MAX_HERMITE_NODES:
            raise ValueError(
                f"n must not exceed {MAX_HERMITE_NODES} (Gauss-Hermite weights "
                f"underflow above that in double precision), got {n!r}"
            )
        t, w = np.polynomial.hermite.hermgauss(int(n))
        h = np.exp(self.mu_x + np.sqrt(2.0) * self.sigma_x * t)
        return h, w / np.sqrt(np.pi)


@dataclass(frozen=True)
class GammaGammaFading:
    """Unit-mean gamma-gamma irradiance (Al-Habash et al. 2001).

    Parameters
    ----------
    alpha, beta:
        Effective numbers of large-scale and small-scale scatterers,
        dimensionless, > 0.
    """

    alpha: float
    beta: float

    def __post_init__(self) -> None:
        _check_positive("alpha", self.alpha)
        _check_positive("beta", self.beta)

    @property
    def scintillation_index(self) -> float:
        """``1/alpha + 1/beta + 1/(alpha beta)``, dimensionless."""
        a, b = self.alpha, self.beta
        return float(1.0 / a + 1.0 / b + 1.0 / (a * b))

    def mean(self) -> float:
        """``E[h] = 1`` by the unit-mean parameterisation."""
        return 1.0

    def variance(self) -> float:
        """``Var[h]``, equal to the scintillation index since ``E[h] = 1``."""
        return self.scintillation_index

    def log_pdf(self, h: np.ndarray) -> np.ndarray:
        """Natural log of the density, ``-inf`` for ``h <= 0``.

        Evaluated through ``scipy.special.kve(v, z) = K_v(z) exp(z)`` so that
        neither the ``(alpha beta)**((alpha+beta)/2)`` prefactor nor the
        exponentially small Bessel function has to be represented on its own.
        Evaluating the density directly overflows for ``alpha, beta`` above a
        few hundred; this form does not.
        """
        h = np.asarray(h, dtype=float)
        a, b = self.alpha, self.beta
        out = np.full(h.shape, -np.inf)
        pos = h > 0
        hp = h[pos]
        log_c = (
            np.log(2.0)
            + 0.5 * (a + b) * np.log(a * b)
            - special.gammaln(a)
            - special.gammaln(b)
        )
        z = 2.0 * np.sqrt(a * b * hp)
        with np.errstate(divide="ignore"):
            log_bessel = np.log(special.kve(a - b, z)) - z
        out[pos] = log_c + (0.5 * (a + b) - 1.0) * np.log(hp) + log_bessel
        return out

    def pdf(self, h: np.ndarray) -> np.ndarray:
        """Density of ``h`` from the Al-Habash et al. (2001) expression."""
        return np.exp(self.log_pdf(h))

    def sample(self, size: int | tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
        """Draw ``h`` as the product of two unit-mean gammas."""
        u = rng.gamma(self.alpha, 1.0 / self.alpha, size=size)
        v = rng.gamma(self.beta, 1.0 / self.beta, size=size)
        return u * v

    def log_grid_range(self) -> tuple[float, float]:
        """Default ``(log_lower, log_upper)`` for the trapezoid rule.

        Both limits come from the asymptotic forms of the Al-Habash et al.
        (2001) density, not from tuning.

        * ``h -> 0``: with ``s = min(alpha, beta)`` the density behaves as
          ``h**(s-1)``, so the density *in log h* behaves as ``h**s`` and the
          point where it falls to ``1e-14`` is ``log h = log(1e-14) / s``.
        * ``h -> infinity``: the density decays as
          ``exp(-2 sqrt(alpha beta h))``, so the point about 30 e-folds below
          the peak is ``h = (1 + 30 / sqrt(alpha beta))**2``.
        * the bulk: ``log h = log u + log v`` with ``u, v`` gamma, so
          ``E[log h] = psi(alpha) - log alpha + psi(beta) - log beta`` and
          ``Var[log h] = psi'(alpha) + psi'(beta)`` exactly. Ten standard
          deviations either side of that mean are always included.

        The two asymptotic limits bind for small ``alpha, beta``, where the
        density is heavy tailed; the bulk limit binds for large ``alpha,
        beta``, where the asymptotic forms apply only far outside the support
        that carries the mass. Taking the widest of each side covers both.
        """
        small = min(self.alpha, self.beta)
        product = self.alpha * self.beta
        mean_log = float(
            special.psi(self.alpha)
            - np.log(self.alpha)
            + special.psi(self.beta)
            - np.log(self.beta)
        )
        sd_log = float(np.sqrt(special.polygamma(1, self.alpha) + special.polygamma(1, self.beta)))
        lower = min(float(np.log(1e-14) / small), mean_log - 10.0 * sd_log, -0.05)
        upper = max(
            float(2.0 * np.log1p(30.0 / np.sqrt(product)) + 0.5),
            mean_log + 10.0 * sd_log,
            0.05,
        )
        return lower, upper

    def quadrature(
        self,
        n: int | None = None,
        log_lower: float | None = None,
        log_upper: float | None = None,
        step: float = 0.08,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Trapezoid rule for the density in ``log h``, ``sum(w) = 1``.

        ``n`` points spanning ``exp(log_lower) .. exp(log_upper)``. The
        default range comes from :meth:`log_grid_range` and the default ``n``
        from that range divided by ``step``, so the resolution is the same
        whatever the parameters. This rule is the default rather than the
        tensor Gauss-Laguerre rule of :meth:`laguerre_quadrature` because the
        Laguerre rule stalls near a relative LLR error of 1e-2 for small
        ``alpha, beta`` while this one reaches 1e-8; both are measured in
        ``validation/validate_quadrature.py``.
        """
        auto_lower, auto_upper = self.log_grid_range()
        lo = auto_lower if log_lower is None else float(log_lower)
        hi = auto_upper if log_upper is None else float(log_upper)
        if not lo < hi:
            raise ValueError(f"need log_lower < log_upper, got {lo!r}, {hi!r}")
        if n is None:
            if not np.isfinite(step) or step <= 0.0:
                raise ValueError(f"step must be a finite positive number, got {step!r}")
            n = int(min(40001, max(129, 2 * int(np.ceil((hi - lo) / (2.0 * step))) + 1)))
        if int(n) < 3:
            raise ValueError(f"n must be at least 3, got {n!r}")
        lh = np.linspace(lo, hi, int(n))
        h = np.exp(lh)
        step = lh[1] - lh[0]
        log_w = self.log_pdf(h) + lh
        log_w -= log_w.max()
        w = np.exp(log_w) * step
        w[0] *= 0.5
        w[-1] *= 0.5
        keep = w > 0.0
        if not np.any(keep):
            raise RuntimeError("gamma-gamma quadrature grid collapsed; widen the log range")
        h, w = h[keep], w[keep]
        return h, w / w.sum()

    def laguerre_quadrature(self, n: int = 40) -> tuple[np.ndarray, np.ndarray]:
        """Tensor product of two generalised Gauss-Laguerre rules.

        ``n`` nodes per gamma factor, so at most ``n**2`` nodes. Exact for the
        moments of each gamma factor, and therefore exact for ``E[h**k]``, but
        less accurate than :meth:`quadrature` for the peaked likelihood
        integrands this package evaluates.
        """
        if int(n) < 2:
            raise ValueError(f"n must be at least 2, got {n!r}")
        hu, wu = _gamma_quadrature(self.alpha, int(n))
        hv, wv = _gamma_quadrature(self.beta, int(n))
        h = np.outer(hu, hv).ravel()
        w = np.outer(wu, wv).ravel()
        return h, w


def _gamma_quadrature(shape: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Nodes/weights for a unit-mean ``Gamma(shape, 1/shape)`` measure.

    Generalised Gauss-Laguerre with weight ``x**(shape-1) exp(-x)``, rescaled
    by ``u = x / shape`` so that ``sum(w * u) = 1``.
    """
    x, w = special.roots_genlaguerre(n, shape - 1.0)
    w = w / special.gamma(shape)
    return x / shape, w / w.sum()


def amplitude_quadrature(
    fading: LognormalFading | GammaGammaFading, n: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and weights for the irradiance of either model.

    ``n`` defaults to 80 for lognormal (Gauss-Hermite); for gamma-gamma the
    default node count follows from :meth:`GammaGammaFading.log_grid_range`
    and a fixed log-step. Both defaults are set by the node-convergence table
    in ``validation/validate_quadrature.py``.
    """
    if isinstance(fading, LognormalFading):
        return fading.quadrature(80 if n is None else n)
    if isinstance(fading, GammaGammaFading):
        return fading.quadrature(n)
    raise TypeError(f"unsupported fading model {type(fading).__name__}")

"""Limit-state functions with reference failure probabilities.

A limit state is a function ``g`` of a vector of standard normal input
variables; the requirement is violated when ``g(x) <= 0``.  Working in
standard normal space ("u-space") is the usual convention in structural
reliability and is what makes the mean-shift tilting family of
:mod:`rareverify.tilting` a clean exponential family; a model whose inputs are
not Gaussian is assumed to have been transformed already, and no transformation
is implemented here.

Three limit states are provided, chosen so that the failure probability is
known to far better accuracy than any Monte-Carlo estimate of it, which is what
makes them usable as known-answer tests.

Linear Gaussian
    ``g(x) = beta - a . x`` with ``a`` a unit vector.  Then ``a . x`` is
    standard normal and ``P(g <= 0) = Phi(-beta)`` exactly, for any dimension
    and any ``beta``.  ``beta`` is the Hasofer-Lind reliability index and
    ``beta * a`` is the design point (the most probable point of the failure
    region).  Source: Hasofer, A. M. and Lind, N. C. (1974), "Exact and
    invariant second-moment code format", Journal of the Engineering Mechanics
    Division, ASCE, volume 100.  Dimensionless throughout.

Lognormal ratio
    ``g(x) = R(x) - S(x)`` with ``R = exp(mu_R + sigma_R x_0)`` a resistance
    and ``S = exp(mu_S + sigma_S x_1)`` a load, both in arbitrary consistent
    units.  ``g`` is nonlinear in ``x``, but the failure *event*
    ``{R <= S}`` is equivalent to ``{log R <= log S}``, which is linear in
    ``x``, so

        P(g <= 0) = Phi(-(mu_R - mu_S) / sqrt(sigma_R^2 + sigma_S^2))

    exactly.  This is the standard lognormal limit state of structural
    reliability; validity requires ``sigma_R, sigma_S > 0``.  It is a genuinely
    different code path from the linear case (exponentials are evaluated) with
    the same exact answer, which is why both are tested.

Rippled
    ``g(x) = beta - a . x + A sin(omega * b . x)`` with ``a`` and ``b``
    orthonormal.  With ``u = a . x`` and ``v = b . x`` independent standard
    normals, the failure event is ``u >= beta + A sin(omega v)``, so

        P(g <= 0) = E_v[ Phi(-beta - A sin(omega v)) ]

    a one-dimensional Gaussian expectation evaluated here by Gauss-Hermite
    quadrature.  The reference value is therefore semi-analytic, not
    closed-form; its convergence with node count is measured in
    validation/validate_quadrature.py.  The ripple makes the failure boundary
    non-planar, so the design point of the smooth part
    (``beta * a``) is no longer the most probable failure point.  That gap is
    the whole point of the rough instance: it is the case where a learned
    surrogate of ``g`` has something to contribute that the analytic smooth
    design point does not.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import numpy as np
from scipy import special, stats

__all__ = [
    "LimitState",
    "LinearGaussianLimitState",
    "LognormalRatioLimitState",
    "RippledLimitState",
    "limit_state_from_name",
]


def _unit(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm == 0.0:
        raise ValueError("direction vector must be non-zero")
    return np.asarray(v, dtype=float) / norm


def _as_samples(x: np.ndarray, dimension: int) -> np.ndarray:
    arr = np.atleast_2d(np.asarray(x, dtype=float))
    if arr.ndim != 2:
        raise ValueError(f"x must be 1- or 2-dimensional, got ndim={arr.ndim}")
    if arr.shape[1] != dimension:
        raise ValueError(
            f"x must have {dimension} columns to match the limit state dimension, "
            f"got shape {arr.shape}"
        )
    return arr


class LimitState(ABC):
    """Abstract limit state on standard normal inputs.

    Subclasses define ``g``; failure is ``g(x) <= 0``.  Every subclass exposes a
    reference failure probability accurate enough to be used as a known answer,
    and declares how that reference was obtained via :attr:`reference_kind`.
    """

    name: str
    dimension: int

    @abstractmethod
    def g(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the limit-state function.

        Parameters
        ----------
        x
            Array of shape ``(n, dimension)`` of standard normal variates, or a
            single row of shape ``(dimension,)``.

        Returns
        -------
        numpy.ndarray
            Shape ``(n,)``.  Units are those of the limit state; only the sign
            matters for the failure event.
        """

    @abstractmethod
    def analytic_probability(self) -> float:
        """Reference failure probability ``P(g(X) <= 0)``, dimensionless."""

    @property
    @abstractmethod
    def reference_kind(self) -> str:
        """How the reference probability was obtained.

        ``"closed-form"`` means an exact expression in the normal CDF;
        ``"quadrature"`` means a one-dimensional Gauss-Hermite evaluation whose
        truncation error is measured separately.
        """

    def failure(self, x: np.ndarray) -> np.ndarray:
        """Boolean failure indicator ``g(x) <= 0`` of shape ``(n,)``."""
        return self.g(x) <= 0.0

    def design_point(self) -> np.ndarray:
        """Analytically known design point, or the smooth-part one if rippled.

        Returns
        -------
        numpy.ndarray
            Shape ``(dimension,)``, in standard normal units.  This is the
            point a reliability analyst would compute without a surrogate, and
            it is exactly the mean shift the analytic importance-sampling
            baseline uses.
        """
        raise NotImplementedError

    def describe(self) -> str:
        """One-line summary. Contains no I/O."""
        return (
            f"{self.name} d={self.dimension} "
            f"p_ref={self.analytic_probability():.6e} ({self.reference_kind})"
        )


class LinearGaussianLimitState(LimitState):
    """``g(x) = beta - a . x``, exact ``P = Phi(-beta)``.

    Parameters
    ----------
    beta
        Reliability index (dimensionless), must be finite.  ``beta = 3.719``
        corresponds to ``p = 1e-4``.
    dimension
        Number of standard normal inputs, ``>= 1``.
    direction
        Unit direction ``a``; defaults to the first coordinate axis.  Any
        direction gives the same probability, which is itself a useful
        invariance test.
    """

    reference_kind = "closed-form"

    def __init__(
        self, beta: float = 3.719, dimension: int = 2, direction: np.ndarray | None = None
    ) -> None:
        beta = float(beta)
        if not math.isfinite(beta):
            raise ValueError(f"beta must be finite, got {beta}")
        if dimension < 1:
            raise ValueError(f"dimension must be at least 1, got {dimension}")
        self.name = "linear-gaussian"
        self.beta = beta
        self.dimension = int(dimension)
        if direction is None:
            a = np.zeros(self.dimension)
            a[0] = 1.0
        else:
            a = _unit(np.asarray(direction, dtype=float).ravel())
            if a.size != self.dimension:
                raise ValueError(
                    f"direction must have {self.dimension} entries, got {a.size}"
                )
        self.a = a

    def g(self, x: np.ndarray) -> np.ndarray:
        arr = _as_samples(x, self.dimension)
        return self.beta - arr @ self.a

    def analytic_probability(self) -> float:
        return float(stats.norm.cdf(-self.beta))

    def design_point(self) -> np.ndarray:
        return self.beta * self.a


class LognormalRatioLimitState(LimitState):
    """``g(x) = exp(mu_R + sigma_R x0) - exp(mu_S + sigma_S x1)``.

    Nonlinear in ``x``, with an exact failure probability because the failure
    event is linear in ``x``.

    Parameters
    ----------
    mu_r, sigma_r
        Log-mean and log-standard-deviation of the resistance; ``sigma_r > 0``.
    mu_s, sigma_s
        Log-mean and log-standard-deviation of the load; ``sigma_s > 0``.

    Notes
    -----
    Units of ``R`` and ``S`` are arbitrary but must be consistent; ``mu`` is in
    log-units of that quantity and ``sigma`` is dimensionless.  The defaults
    give ``beta = 3.7193`` and ``p = 1.0001e-4``, chosen to sit at the same
    order as the linear default so the two known-answer cases are comparable.
    """

    reference_kind = "closed-form"

    def __init__(
        self,
        mu_r: float = 1.190661,
        sigma_r: float = 0.20,
        mu_s: float = 0.0,
        sigma_s: float = 0.25,
    ) -> None:
        for value, label in ((sigma_r, "sigma_r"), (sigma_s, "sigma_s")):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{label} must be positive and finite, got {value}")
        for value, label in ((mu_r, "mu_r"), (mu_s, "mu_s")):
            if not math.isfinite(value):
                raise ValueError(f"{label} must be finite, got {value}")
        self.name = "lognormal-ratio"
        self.dimension = 2
        self.mu_r = float(mu_r)
        self.sigma_r = float(sigma_r)
        self.mu_s = float(mu_s)
        self.sigma_s = float(sigma_s)

    @property
    def beta(self) -> float:
        """Reliability index ``(mu_R - mu_S) / sqrt(sigma_R^2 + sigma_S^2)``."""
        return (self.mu_r - self.mu_s) / math.hypot(self.sigma_r, self.sigma_s)

    def g(self, x: np.ndarray) -> np.ndarray:
        arr = _as_samples(x, 2)
        resistance = np.exp(self.mu_r + self.sigma_r * arr[:, 0])
        load = np.exp(self.mu_s + self.sigma_s * arr[:, 1])
        return resistance - load

    def analytic_probability(self) -> float:
        return float(stats.norm.cdf(-self.beta))

    def design_point(self) -> np.ndarray:
        # minimise |x| subject to sigma_r x0 - sigma_s x1 = mu_s - mu_r.
        c = np.array([self.sigma_r, -self.sigma_s])
        h = self.mu_s - self.mu_r
        return c * h / float(c @ c)


class RippledLimitState(LimitState):
    """``g(x) = beta - a . x + A sin(omega * b . x)`` with ``a``, ``b`` orthonormal.

    Parameters
    ----------
    beta
        Reliability index of the smooth part (dimensionless).
    amplitude
        Ripple amplitude ``A`` in the units of ``g``; ``A = 0`` recovers the
        linear case exactly.
    frequency
        Ripple frequency ``omega`` in reciprocal standard-normal units.
    dimension
        Number of standard normal inputs, ``>= 2`` (two orthonormal directions
        are needed).
    quadrature_nodes
        Gauss-Hermite node count for the reference probability.  The default
        400 is justified by validation/validate_quadrature.py: it is converged
        to better than 2.6e-6 relative for ripple frequencies up to 12, where
        200 nodes reaches only 5.7e-5.  Raising the frequency further requires
        re-running that script before any tolerance stated in counting-noise
        floors can be trusted.

    Notes
    -----
    Valid for any finite ``beta``, ``A``, ``omega``.  The reference probability
    is a quadrature value, so known-answer tolerances for this instance must
    not be tighter than the quadrature error reported in validation.
    """

    reference_kind = "quadrature"

    def __init__(
        self,
        beta: float = 3.719,
        amplitude: float = 0.8,
        frequency: float = 1.5,
        dimension: int = 2,
        quadrature_nodes: int = 400,
    ) -> None:
        beta = float(beta)
        amplitude = float(amplitude)
        frequency = float(frequency)
        for value, label in (
            (beta, "beta"),
            (amplitude, "amplitude"),
            (frequency, "frequency"),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{label} must be finite, got {value}")
        if dimension < 2:
            raise ValueError(
                f"dimension must be at least 2 for a rippled limit state, got {dimension}"
            )
        if quadrature_nodes < 8:
            raise ValueError(f"quadrature_nodes must be at least 8, got {quadrature_nodes}")
        self.name = "rippled"
        self.beta = beta
        self.amplitude = amplitude
        self.frequency = frequency
        self.dimension = int(dimension)
        self.quadrature_nodes = int(quadrature_nodes)
        a = np.zeros(self.dimension)
        a[0] = 1.0
        b = np.zeros(self.dimension)
        b[1] = 1.0
        self.a = a
        self.b = b

    def g(self, x: np.ndarray) -> np.ndarray:
        arr = _as_samples(x, self.dimension)
        u = arr @ self.a
        v = arr @ self.b
        return self.beta - u + self.amplitude * np.sin(self.frequency * v)

    def analytic_probability(self, nodes: int | None = None) -> float:
        """Reference probability by Gauss-Hermite quadrature.

        Parameters
        ----------
        nodes
            Override the node count, used by the convergence study.
        """
        n = self.quadrature_nodes if nodes is None else int(nodes)
        if n < 8:
            raise ValueError(f"nodes must be at least 8, got {n}")
        # int f(v) phi(v) dv = (1/sqrt(pi)) sum w_i f(sqrt(2) x_i).
        # scipy.special.roots_hermite is used rather than
        # numpy.polynomial.hermite.hermgauss, which overflows to NaN at
        # n >= 400 on numpy 2.5.3; see validation/VALIDATION.md.
        hermite_nodes, hermite_weights = special.roots_hermite(n)
        v = math.sqrt(2.0) * hermite_nodes
        values = stats.norm.cdf(-self.beta - self.amplitude * np.sin(self.frequency * v))
        return float((hermite_weights @ values) / math.sqrt(math.pi))

    def design_point(self) -> np.ndarray:
        """Design point of the smooth part only, ``beta * a``.

        This is deliberately *not* the true most probable failure point of the
        rippled limit state.  It is what an analyst gets from the smooth model,
        and it is the baseline the learned surrogate has to beat.
        """
        return self.beta * self.a


def limit_state_from_name(name: str, **kwargs: float) -> LimitState:
    """Construct a limit state by name.

    Parameters
    ----------
    name
        ``"linear"`` / ``"linear-gaussian"``, ``"lognormal"`` /
        ``"lognormal-ratio"``, or ``"rippled"``.
    **kwargs
        Forwarded to the constructor.
    """
    key = name.lower()
    if key in ("linear", "linear-gaussian"):
        return LinearGaussianLimitState(**kwargs)  # type: ignore[arg-type]
    if key in ("lognormal", "lognormal-ratio"):
        return LognormalRatioLimitState(**kwargs)  # type: ignore[arg-type]
    if key == "rippled":
        return RippledLimitState(**kwargs)  # type: ignore[arg-type]
    raise ValueError(
        f"unknown limit state {name!r}; expected 'linear', 'lognormal' or 'rippled'"
    )

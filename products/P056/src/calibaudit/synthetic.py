"""Synthetic forecasts whose calibration is known in closed form.

Every number this package reports about an estimator's *bias* needs a
reference value that is not itself estimated. The construction here provides
one. A latent success probability is drawn from a Beta distribution,

    p ~ Beta(a, b),   o | p ~ Bernoulli(p),   f = g(p),

where ``g`` is a declared, strictly increasing distortion. Because ``f`` is a
deterministic monotone function of ``p``, conditioning on ``f`` is the same as
conditioning on ``p``, so

    E[o | f] = g^{-1}(f)

exactly. The whole calibration curve of the forecaster is therefore known, and
with it the population values of the Brier score, its three Murphy terms, the
logarithmic score and the expected calibration error:

    UNC = obar (1 - obar),  obar = a / (a + b)
    RES = Var(p) = a b / ((a + b)^2 (a + b + 1))
    REL = E_p[(g(p) - p)^2]
    BS  = REL - RES + UNC
    ECE = E_p[|g(p) - p|]
    LS  = E_p[-(p log g(p) + (1 - p) log(1 - g(p)))]

``UNC`` and ``RES`` are closed-form moments of the Beta distribution. The other
three are one-dimensional integrals against the Beta density, evaluated with
``scipy.integrate.quad``, which returns its own absolute-error estimate; that
estimate is carried on :class:`AnalyticTruth` and quoted as the tolerance
wherever these values are used as a reference.

``g = identity`` gives a *perfectly calibrated* forecaster, for which the
population ECE is exactly zero. Any nonzero ECE measured on such a sample is
estimator bias plus sampling noise, which is what
:mod:`calibaudit.ece` measures.

References
----------
Platt, J. C. (1999). "Probabilistic outputs for support vector machines and
comparisons to regularized likelihood methods." In *Advances in Large Margin
Classifiers*, MIT Press, 61-74. (the logistic distortion family used here)

Guo, C., Pleiss, G., Sun, Y. and Weinberger, K. Q. (2017). "On calibration of
modern neural networks." *ICML 2017*, PMLR 70, 1321-1330. arXiv:1706.04599.
(temperature scaling; ``temperature`` below is its inverse-direction twin)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import quad
from scipy.special import betaln, expit, logit

__all__ = [
    "SPEC_NAMES",
    "AnalyticTruth",
    "ForecastSample",
    "ForecastSpec",
    "analytic_truth",
    "calibration_map",
    "distortion",
    "get_spec",
    "sample_forecast",
    "spec_names",
]

_QUAD_LIMIT = 400
_QUAD_EPS = 1e-12


@dataclass(frozen=True)
class ForecastSpec:
    """A synthetic forecaster with a known calibration curve.

    Attributes
    ----------
    name
        Registry key.
    beta_a, beta_b
        Shape parameters of the latent Beta distribution, both > 0. Values
        below 1 put an integrable singularity at an endpoint; the shipped
        specs keep both at or above 1 so the quadrature error stays near
        machine precision.
    kind
        ``"identity"``, ``"temperature"`` or ``"logit_shift"``.
    param
        Temperature ``T > 0`` for ``"temperature"`` (``T < 1`` sharpens the
        forecast, ``T > 1`` shrinks it toward 0.5), or the additive logit
        offset for ``"logit_shift"``. Ignored for ``"identity"``.
    description
        One line, used by the CLI.
    """

    name: str
    beta_a: float
    beta_b: float
    kind: str
    param: float
    description: str

    def __post_init__(self) -> None:
        if self.beta_a <= 0.0 or self.beta_b <= 0.0:
            raise ValueError(
                f"beta_a and beta_b must be positive, got {self.beta_a!r} and {self.beta_b!r}"
            )
        if self.kind not in ("identity", "temperature", "logit_shift"):
            raise ValueError(
                f"unknown distortion kind {self.kind!r}; expected 'identity', "
                "'temperature' or 'logit_shift'"
            )
        if self.kind == "temperature" and self.param <= 0.0:
            raise ValueError(f"temperature must be positive, got {self.param!r}")

    @property
    def is_calibrated(self) -> bool:
        """True when the population ECE and reliability are exactly zero."""
        if self.kind == "identity":
            return True
        if self.kind == "temperature":
            return self.param == 1.0
        return self.param == 0.0


def distortion(spec: ForecastSpec, p: ArrayLike) -> np.ndarray:
    """``g(p)``: the forecast this spec issues when the truth is ``p``."""
    arr = np.asarray(p, dtype=float)
    if spec.kind == "identity":
        return arr.copy()
    if spec.kind == "temperature":
        return expit(logit(arr) / spec.param)
    return expit(logit(arr) + spec.param)


def calibration_map(spec: ForecastSpec, f: ArrayLike) -> np.ndarray:
    """``g^{-1}(f) = E[o | f]``: the exact calibration curve of this spec."""
    arr = np.asarray(f, dtype=float)
    if spec.kind == "identity":
        return arr.copy()
    if spec.kind == "temperature":
        return expit(logit(arr) * spec.param)
    return expit(logit(arr) - spec.param)


_SPECS: dict[str, ForecastSpec] = {
    s.name: s
    for s in (
        ForecastSpec(
            "calibrated",
            2.0,
            2.0,
            "identity",
            0.0,
            "perfectly calibrated, Beta(2,2) latent, base rate 0.5",
        ),
        ForecastSpec(
            "calibrated_uniform",
            1.0,
            1.0,
            "identity",
            0.0,
            "perfectly calibrated, uniform latent probability, base rate 0.5",
        ),
        ForecastSpec(
            "calibrated_rare",
            1.0,
            9.0,
            "identity",
            0.0,
            "perfectly calibrated, Beta(1,9) latent, base rate 0.1",
        ),
        ForecastSpec(
            "overconfident",
            2.0,
            2.0,
            "temperature",
            0.6,
            "sharpened logits (T = 0.6), the usual neural-network failure",
        ),
        ForecastSpec(
            "underconfident",
            2.0,
            2.0,
            "temperature",
            1.6,
            "shrunk logits (T = 1.6), forecasts pulled toward 0.5",
        ),
        ForecastSpec(
            "biased_high",
            2.0,
            2.0,
            "logit_shift",
            0.6,
            "logits offset by +0.6, a uniformly optimistic forecaster",
        ),
    )
}

#: Names of the shipped synthetic forecasters, in registry order.
SPEC_NAMES: tuple[str, ...] = tuple(_SPECS)


def spec_names() -> list[str]:
    """Names of the shipped synthetic forecasters."""
    return list(SPEC_NAMES)


def get_spec(name: str) -> ForecastSpec:
    """Look up a shipped spec by name.

    Raises
    ------
    KeyError
        If the name is not a shipped spec; the message lists the valid names.
    """
    try:
        return _SPECS[name]
    except KeyError:
        raise KeyError(f"unknown spec {name!r}; shipped specs are {SPEC_NAMES}") from None


@dataclass(frozen=True)
class ForecastSample:
    """One draw from a :class:`ForecastSpec`.

    ``true_probabilities`` is the latent ``p``; it is available because the
    data are synthetic and is the thing a real audit never has.
    """

    spec: ForecastSpec
    seed: int
    forecasts: np.ndarray = field(repr=False)
    outcomes: np.ndarray = field(repr=False)
    true_probabilities: np.ndarray = field(repr=False)

    @property
    def n_samples(self) -> int:
        return int(self.forecasts.size)


def sample_forecast(spec: ForecastSpec, n_samples: int, *, seed: int) -> ForecastSample:
    """Draw ``n_samples`` forecast-outcome pairs from ``spec``.

    Deterministic in ``(spec, n_samples, seed)``. The two random draws are
    taken in one call each, in the order latent probability then outcome;
    changing that order changes the stream and so changes every number.

    Raises
    ------
    ValueError
        If ``n_samples < 1``.
    """
    if n_samples < 1:
        raise ValueError(f"n_samples must be at least 1, got {n_samples!r}")
    rng = np.random.default_rng(int(seed))
    p = rng.beta(spec.beta_a, spec.beta_b, size=int(n_samples))
    o = (rng.random(int(n_samples)) < p).astype(float)
    f = distortion(spec, p)
    return ForecastSample(
        spec=spec, seed=int(seed), forecasts=f, outcomes=o, true_probabilities=p
    )


@dataclass(frozen=True)
class AnalyticTruth:
    """Population values for a :class:`ForecastSpec`.

    ``*_abserr`` fields are the absolute-error estimates returned by
    ``scipy.integrate.quad`` and are the tolerance to quote when comparing a
    measurement against the corresponding value. ``base_rate``,
    ``uncertainty`` and ``resolution`` are closed-form Beta moments with no
    quadrature error.
    """

    spec: ForecastSpec
    base_rate: float
    uncertainty: float
    resolution: float
    reliability: float
    reliability_abserr: float
    brier: float
    ece: float
    ece_abserr: float
    log_score: float
    log_score_abserr: float

    @property
    def brier_from_terms(self) -> float:
        """``REL - RES + UNC``, equal to ``brier`` by construction."""
        return float(self.reliability - self.resolution + self.uncertainty)


def _beta_integral(spec: ForecastSpec, func) -> tuple[float, float]:
    """Integrate ``func(p)`` against the Beta(a, b) density on (0, 1)."""
    a, b = spec.beta_a, spec.beta_b
    log_c = -betaln(a, b)

    def integrand(p: float) -> float:
        if p <= 0.0 or p >= 1.0:
            return 0.0
        dens = np.exp(log_c + (a - 1.0) * np.log(p) + (b - 1.0) * np.log1p(-p))
        return float(func(p) * dens)

    value, abserr = quad(integrand, 0.0, 1.0, limit=_QUAD_LIMIT, epsabs=_QUAD_EPS)
    return float(value), float(abserr)


def analytic_truth(spec: ForecastSpec) -> AnalyticTruth:
    """Population scores and calibration terms for ``spec``.

    Uses closed-form Beta moments where they exist and ``scipy.integrate.quad``
    for the three expectations that do not reduce.
    """
    a, b = spec.beta_a, spec.beta_b
    obar = a / (a + b)
    unc = obar * (1.0 - obar)
    res = a * b / ((a + b) ** 2 * (a + b + 1.0))

    if spec.is_calibrated:
        rel, rel_err = 0.0, 0.0
        ece, ece_err = 0.0, 0.0
    else:
        rel, rel_err = _beta_integral(spec, lambda p: (float(distortion(spec, p)) - p) ** 2)
        ece, ece_err = _beta_integral(spec, lambda p: abs(float(distortion(spec, p)) - p))

    def _log_term(p: float) -> float:
        g = float(np.clip(distortion(spec, p), 1e-300, 1.0 - 1e-16))
        return -(p * np.log(g) + (1.0 - p) * np.log1p(-g))

    ls, ls_err = _beta_integral(spec, _log_term)

    return AnalyticTruth(
        spec=spec,
        base_rate=float(obar),
        uncertainty=float(unc),
        resolution=float(res),
        reliability=float(rel),
        reliability_abserr=float(rel_err),
        brier=float(rel - res + unc),
        ece=float(ece),
        ece_abserr=float(ece_err),
        log_score=float(ls),
        log_score_abserr=float(ls_err),
    )

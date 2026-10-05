"""Injected synthetic staged pipelines with known latency parameters.

Why synthetic injection rather than measurement
-----------------------------------------------
This package predicts end-to-end latency from per-stage information. To check
a predictor you need to know the right answer. On a shared general-purpose
host you do not: a wall-clock measurement of a 270 us pipeline is dominated by
operating-system scheduling, and the same code measured twice can differ by an
order of magnitude. So every latency this package validates against is
*injected*: drawn from a distribution whose parameters are declared in the
test, with a fixed seed. The injected parameters are the ground truth, and the
correctness of the models is a statement about recovering them.

Stage model
-----------
Stage ``i`` has latency ``X_i > 0`` with declared mean ``m_i`` (s) and
standard deviation ``s_i`` (s). Two marginal families are supported:

``"constant"``
    ``X_i = m_i`` with probability 1 (requires ``s_i == 0``). Degenerate but
    useful: it makes the sum-of-stages model exact sample-by-sample.

``"lognormal"``
    ``X_i = exp(mu_i + sigma_i Z_i)`` with ``Z_i`` standard normal. The
    parameters follow from the declared moments by the standard lognormal
    moment relations

        sigma_i^2 = ln(1 + (s_i / m_i)^2)
        mu_i      = ln(m_i) - sigma_i^2 / 2

    Source: Johnson, Kotz & Balakrishnan (1994), *Continuous Univariate
    Distributions*, Vol. 1, 2nd ed., Ch. 14 (lognormal moments). Lognormal is
    the usual choice for a software latency marginal because it is positive,
    right-skewed and closed under products of independent multiplicative
    effects; it is a modelling choice, not a law of nature.

Dependence between stages
-------------------------
Dependence is injected with a Gaussian copula on the latent normals: the
vector ``Z`` is drawn from ``N(0, R)`` with ``R`` an equicorrelated latent
correlation matrix, ``R_ij = rho`` for ``i != j``. For lognormal marginals the
induced covariance in latency space is available in closed form,

    Cov(X_i, X_j) = m_i m_j (exp(rho sigma_i sigma_j) - 1)

so the dependence is not only injected but analytically known. Source: the
multivariate lognormal moment formulas in Johnson, Kotz & Balakrishnan (1994),
Ch. 14 Sec. 4; see also Aitchison & Brown (1957), *The Lognormal
Distribution*, Ch. 2. Validity range: ``R`` must be positive semi-definite,
which for the equicorrelated form requires ``-1/(K-1) <= rho <= 1`` with ``K``
stages.

Units
-----
All latencies and their standard deviations are in seconds. ``rho`` is
dimensionless.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

SUPPORTED_DISTRIBUTIONS = ("lognormal", "constant")


@dataclass(frozen=True)
class StageSpec:
    """One pipeline stage.

    Parameters
    ----------
    name:
        Stage label, free text.
    mean_s:
        Declared mean latency, seconds. Must be strictly positive.
    std_s:
        Declared standard deviation of the latency, seconds. Must be
        non-negative, and must be zero when ``dist == "constant"``.
    dist:
        Marginal family, one of :data:`SUPPORTED_DISTRIBUTIONS`.
    """

    name: str
    mean_s: float
    std_s: float
    dist: str = "lognormal"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("stage name must be a non-empty string")
        if not math.isfinite(self.mean_s) or self.mean_s <= 0.0:
            raise ValueError(
                f"stage {self.name!r}: mean_s must be finite and > 0, got {self.mean_s!r}"
            )
        if not math.isfinite(self.std_s) or self.std_s < 0.0:
            raise ValueError(
                f"stage {self.name!r}: std_s must be finite and >= 0, got {self.std_s!r}"
            )
        if self.dist not in SUPPORTED_DISTRIBUTIONS:
            raise ValueError(
                f"stage {self.name!r}: dist must be one of {SUPPORTED_DISTRIBUTIONS}, "
                f"got {self.dist!r}"
            )
        if self.dist == "constant" and self.std_s != 0.0:
            raise ValueError(f"stage {self.name!r}: dist='constant' requires std_s == 0")
        if self.dist == "lognormal" and self.std_s == 0.0:
            raise ValueError(
                f"stage {self.name!r}: dist='lognormal' requires std_s > 0; use dist='constant'"
            )

    @property
    def cv(self) -> float:
        """Coefficient of variation ``std_s / mean_s``, dimensionless."""
        return self.std_s / self.mean_s

    def lognormal_params(self) -> tuple[float, float]:
        """Return ``(mu, sigma)`` of the underlying normal, both dimensionless.

        Uses ``sigma^2 = ln(1 + cv^2)``, ``mu = ln(mean) - sigma^2 / 2``
        (Johnson, Kotz & Balakrishnan 1994, Ch. 14). Raises for a constant
        stage, which has no lognormal parameterisation.
        """
        if self.dist != "lognormal":
            raise ValueError(f"stage {self.name!r}: lognormal_params requires dist='lognormal'")
        sigma_sq = math.log1p(self.cv**2)
        sigma = math.sqrt(sigma_sq)
        return math.log(self.mean_s) - 0.5 * sigma_sq, sigma


@dataclass(frozen=True)
class PipelineSpec:
    """A staged pipeline: an ordered tuple of stages plus a latent correlation.

    Parameters
    ----------
    stages:
        Ordered stages. End-to-end latency is the sum of stage latencies; this
        is a serial pipeline with no overlap.
    latent_rho:
        Equicorrelation of the latent normals driving the lognormal stages.
        ``0.0`` gives independent stages. Constant stages are unaffected.
    """

    stages: tuple[StageSpec, ...]
    latent_rho: float = 0.0

    def __post_init__(self) -> None:
        if not self.stages:
            raise ValueError("a pipeline needs at least one stage")
        if not all(isinstance(s, StageSpec) for s in self.stages):
            raise TypeError("stages must all be StageSpec instances")
        k = len(self.stages)
        if not math.isfinite(self.latent_rho):
            raise ValueError("latent_rho must be finite")
        lower = -1.0 if k == 1 else -1.0 / (k - 1)
        if not (lower <= self.latent_rho <= 1.0):
            raise ValueError(
                f"latent_rho must lie in [{lower:.6f}, 1.0] for {k} stages to keep the "
                f"latent correlation matrix positive semi-definite, got {self.latent_rho!r}"
            )

    @property
    def n_stages(self) -> int:
        """Number of stages."""
        return len(self.stages)

    @property
    def stage_mean_s(self) -> np.ndarray:
        """Declared stage means, seconds, shape ``(K,)``."""
        return np.array([s.mean_s for s in self.stages], dtype=float)

    @property
    def stage_std_s(self) -> np.ndarray:
        """Declared stage standard deviations, seconds, shape ``(K,)``."""
        return np.array([s.std_s for s in self.stages], dtype=float)

    def injected_mean_s(self) -> float:
        """Exact end-to-end mean, seconds: ``sum(mean_s)`` by linearity of E[.]."""
        return float(self.stage_mean_s.sum())

    def injected_covariance(self) -> np.ndarray:
        """Exact end-to-end stage covariance matrix, s^2, shape ``(K, K)``.

        Diagonal entries are the declared variances. Off-diagonal entries use
        ``Cov(X_i, X_j) = m_i m_j (exp(rho sigma_i sigma_j) - 1)`` for
        lognormal pairs (Johnson, Kotz & Balakrishnan 1994, Ch. 14 Sec. 4) and
        are zero whenever either stage is constant.
        """
        k = self.n_stages
        cov = np.zeros((k, k), dtype=float)
        sigmas = np.zeros(k, dtype=float)
        for i, stage in enumerate(self.stages):
            cov[i, i] = stage.std_s**2
            if stage.dist == "lognormal":
                sigmas[i] = stage.lognormal_params()[1]
        means = self.stage_mean_s
        for i in range(k):
            for j in range(i + 1, k):
                if sigmas[i] == 0.0 or sigmas[j] == 0.0:
                    continue
                c = means[i] * means[j] * math.expm1(self.latent_rho * sigmas[i] * sigmas[j])
                cov[i, j] = c
                cov[j, i] = c
        return cov

    def injected_std_s(self) -> float:
        """Exact end-to-end standard deviation, seconds: ``sqrt(1' Cov 1)``."""
        return float(math.sqrt(self.injected_covariance().sum()))


def _latent_normals(k: int, n: int, rho: float, rng: np.random.Generator) -> np.ndarray:
    """Draw ``(n, k)`` latent normals with equicorrelation ``rho``.

    For ``rho == 0`` the draw is ``rng.standard_normal((n, k))``, filled in C
    order, which consumes the generator stream in the same order as a nested
    loop over samples and then stages. That ordering is part of the
    reproducibility contract and is relied on by the P033 cross-check.
    """
    z = rng.standard_normal((n, k))
    if rho == 0.0 or k == 1:
        return z
    # Equicorrelated factor model: Z_i = sqrt(rho) F + sqrt(1 - rho) E_i for
    # rho >= 0; for rho < 0 fall back to the Cholesky factor of R.
    if rho > 0.0:
        f = rng.standard_normal((n, 1))
        return math.sqrt(rho) * f + math.sqrt(1.0 - rho) * z
    r = np.full((k, k), rho, dtype=float)
    np.fill_diagonal(r, 1.0)
    chol = np.linalg.cholesky(r)
    return z @ chol.T


def sample_stage_latencies(
    spec: PipelineSpec, n_samples: int, seed: int
) -> np.ndarray:
    """Draw per-stage latencies.

    Parameters
    ----------
    spec:
        Pipeline to sample.
    n_samples:
        Number of end-to-end passes to draw. Must be >= 1.
    seed:
        Seed for ``numpy.random.default_rng``. The draw is fully determined by
        ``(spec, n_samples, seed)``.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_samples, K)``, seconds, strictly positive.
    """
    if not isinstance(n_samples, (int, np.integer)) or n_samples < 1:
        raise ValueError(f"n_samples must be an integer >= 1, got {n_samples!r}")
    rng = np.random.default_rng(seed)
    k = spec.n_stages
    z = _latent_normals(k, int(n_samples), float(spec.latent_rho), rng)
    out = np.empty((int(n_samples), k), dtype=float)
    for i, stage in enumerate(spec.stages):
        if stage.dist == "constant":
            out[:, i] = stage.mean_s
        else:
            mu, sigma = stage.lognormal_params()
            out[:, i] = np.exp(mu + sigma * z[:, i])
    return out


def sample_total_latency(spec: PipelineSpec, n_samples: int, seed: int) -> np.ndarray:
    """Draw end-to-end latencies, seconds, shape ``(n_samples,)``.

    End-to-end latency is the row sum of :func:`sample_stage_latencies`: a
    serial pipeline, no stage overlap, no queueing.
    """
    return sample_stage_latencies(spec, n_samples, seed).sum(axis=1)


def make_lognormal_pipeline(
    means_s: tuple[float, ...] | list[float],
    stds_s: tuple[float, ...] | list[float],
    latent_rho: float = 0.0,
    names: tuple[str, ...] | list[str] | None = None,
) -> PipelineSpec:
    """Build an all-lognormal :class:`PipelineSpec` from means and sds in seconds."""
    means_s = list(means_s)
    stds_s = list(stds_s)
    if len(means_s) != len(stds_s):
        raise ValueError(
            f"means_s and stds_s must match in length, got {len(means_s)} and {len(stds_s)}"
        )
    if names is None:
        names = [f"stage{i}" for i in range(len(means_s))]
    names = list(names)
    if len(names) != len(means_s):
        raise ValueError(f"names must match means_s in length, got {len(names)} and {len(means_s)}")
    stages = tuple(
        StageSpec(name=n, mean_s=m, std_s=s, dist="lognormal")
        for n, m, s in zip(names, means_s, stds_s, strict=True)
    )
    return PipelineSpec(stages=stages, latent_rho=latent_rho)

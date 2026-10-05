"""Synthetic pipeline populations, in two dependence regimes.

Every number this package validates comes from this generator. There is no
measured latency anywhere in the training or evaluation path, deliberately:
the host is a shared single-core container whose wall-clock timings move by
factors of several between runs, so a measured target would make every
reported error a statement about host load rather than about a model.

Two regimes
-----------
``"independent"``
    ``latent_rho = 0`` for every pipeline. Stage latencies are independent, so
    equation (2) of :mod:`latencynet.analytic` is exact and the analytic
    baseline is expected to win.

``"correlated"``
    ``latent_rho`` drawn from ``U(rho_low, rho_high)``, default
    ``U(0.35, 0.90)``. Variances no longer add, so the independence-mode
    analytic baseline systematically *under*-predicts the total spread and
    therefore the tail. This is the regime where a model with access to the
    dependence features has something to learn.

What one record contains
------------------------
For each pipeline:

* the declared stage parameters (the injected truth);
* a **probe trace** of ``n_probe`` passes, summarised by
  :func:`latencynet.features.summarise_probe_trace`. This is the only
  information any model is allowed to use;
* a **reference sample** of ``n_reference`` independent end-to-end passes, from
  a different seed stream, used *only* to compute the regression target. Its
  own Monte Carlo standard error is recorded per record, because it sets the
  floor below which model error cannot be resolved.

Target
------
``ln q_p`` for ``p`` in ``{0.99, 0.999}``, from the reference sample. Log
space, because latency spans two orders of magnitude across the population and
a relative error is the quantity an engineer cares about.

Compute budget
--------------
Dominated by the reference samples: ``n_pipelines * K_mean * n_reference``
lognormal draws. At the defaults (320 pipelines, mean 4 stages, 100000
reference passes for train/calibration and 400000 for test) that is of order
2e8 draws, about 6 s of a single core when the core is uncontended. All
generation is deterministic in ``seed``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .features import ProbeSummary, summarise_probe_trace
from .pipeline import PipelineSpec, make_lognormal_pipeline, sample_stage_latencies
from .tails import quantile

REGIMES = ("independent", "correlated")

#: Tail probabilities this package predicts.
TARGET_PROBABILITIES: tuple[float, ...] = (0.99, 0.999)


@dataclass(frozen=True)
class PipelineRecord:
    """One synthetic pipeline with its probe summary and its reference target.

    Attributes
    ----------
    index:
        Position in the generated population.
    regime:
        ``"independent"`` or ``"correlated"``.
    declared_mean_s, declared_std_s:
        Injected per-stage parameters, seconds.
    latent_rho:
        Injected latent equicorrelation, dimensionless.
    probe:
        Probe-trace summary; the only model input.
    reference_mean_s:
        Sample mean of the reference end-to-end passes, seconds.
    reference_quantile_s:
        Reference end-to-end quantiles, seconds, keyed by probability.
    reference_quantile_se_s:
        Order-statistic standard error of each reference quantile, seconds,
        estimated from the reference sample's own local density.
    n_reference:
        Reference sample size.
    """

    index: int
    regime: str
    declared_mean_s: tuple[float, ...]
    declared_std_s: tuple[float, ...]
    latent_rho: float
    probe: ProbeSummary
    reference_mean_s: float
    reference_quantile_s: dict[float, float]
    reference_quantile_se_s: dict[float, float]
    n_reference: int

    @property
    def n_stages(self) -> int:
        """Number of stages."""
        return len(self.declared_mean_s)

    def spec(self) -> PipelineSpec:
        """Rebuild the injected :class:`~latencynet.pipeline.PipelineSpec`."""
        return make_lognormal_pipeline(
            self.declared_mean_s, self.declared_std_s, latent_rho=self.latent_rho
        )


def _quantile_se(samples: np.ndarray, p: float) -> float:
    """Distribution-free standard uncertainty of a sample quantile, seconds.

    The rank of the order statistic bracketing ``q_p`` is binomial with mean
    ``n p`` and standard deviation ``sqrt(n p (1 - p))``, so a one-sigma
    rank interval is ``n p +/- sqrt(n p (1 - p))`` and the corresponding
    latency half-width ``(X_(s) - X_(r)) / 2`` is a distribution-free standard
    uncertainty for the quantile estimate. No density estimate and no
    parametric assumption are involved. Source: David & Nagaraja (2003),
    *Order Statistics*, 3rd ed., Sec. 7.1 (distribution-free confidence
    intervals for quantiles, large-sample normal approximation to the
    binomial rank distribution).

    Returns ``nan`` when the one-sigma rank interval does not fit inside the
    sample, which is the honest answer for a tail probability that the sample
    size cannot resolve.
    """
    arr = np.sort(np.asarray(samples, dtype=float).ravel())
    n = arr.size
    p = float(p)
    spread = math.sqrt(n * p * (1.0 - p))
    r = int(math.floor(n * p - spread))
    s = int(math.ceil(n * p + spread))
    if r < 1 or s > n or s <= r:
        return float("nan")
    return float(0.5 * (arr[s - 1] - arr[r - 1]))


def generate_record(
    index: int,
    regime: str,
    seed: int,
    n_probe: int = 256,
    n_reference: int = 100_000,
    n_stages_range: tuple[int, int] = (2, 6),
    stage_mean_range_s: tuple[float, float] = (2.0e-5, 5.0e-4),
    stage_cv_range: tuple[float, float] = (0.05, 0.60),
    rho_range: tuple[float, float] = (0.35, 0.90),
    probabilities: tuple[float, ...] = TARGET_PROBABILITIES,
) -> PipelineRecord:
    """Generate one synthetic pipeline record.

    ``seed`` seeds the pipeline's parameter draw; the probe trace and the
    reference sample use ``seed + 1`` and ``seed + 2`` so that no sample is
    shared between the model input and the target.
    """
    if regime not in REGIMES:
        raise ValueError(f"regime must be one of {REGIMES}, got {regime!r}")
    if n_probe < 8:
        raise ValueError(f"n_probe must be >= 8, got {n_probe}")
    if n_reference < 1000:
        raise ValueError(f"n_reference must be >= 1000 for a usable tail target, got {n_reference}")
    lo_k, hi_k = n_stages_range
    if not (1 <= lo_k <= hi_k):
        raise ValueError(f"n_stages_range must satisfy 1 <= lo <= hi, got {n_stages_range!r}")

    rng = np.random.default_rng(seed)
    k = int(rng.integers(lo_k, hi_k + 1))
    log_lo, log_hi = math.log(stage_mean_range_s[0]), math.log(stage_mean_range_s[1])
    means = np.exp(rng.uniform(log_lo, log_hi, size=k))
    cvs = rng.uniform(stage_cv_range[0], stage_cv_range[1], size=k)
    stds = means * cvs
    rho = 0.0 if regime == "independent" else float(rng.uniform(rho_range[0], rho_range[1]))

    spec = make_lognormal_pipeline(means, stds, latent_rho=rho)
    probe_trace = sample_stage_latencies(spec, n_probe, seed + 1)
    probe = summarise_probe_trace(probe_trace)
    ref_total = sample_stage_latencies(spec, n_reference, seed + 2).sum(axis=1)

    ref_q = {float(p): quantile(ref_total, p, method="linear") for p in probabilities}
    ref_se = {float(p): _quantile_se(ref_total, float(p)) for p in probabilities}

    return PipelineRecord(
        index=int(index),
        regime=regime,
        declared_mean_s=tuple(float(v) for v in means),
        declared_std_s=tuple(float(v) for v in stds),
        latent_rho=rho,
        probe=probe,
        reference_mean_s=float(ref_total.mean()),
        reference_quantile_s=ref_q,
        reference_quantile_se_s=ref_se,
        n_reference=int(n_reference),
    )


@dataclass(frozen=True)
class PipelineDataset:
    """A population of pipelines split into train, calibration and test.

    The split is by pipeline, never by pass: held out means a pipeline the
    model has never seen, which is the only split that answers the question
    "will this work on the next pipeline I profile".
    """

    regime: str
    train: tuple[PipelineRecord, ...]
    calibration: tuple[PipelineRecord, ...]
    test: tuple[PipelineRecord, ...]
    seed: int

    @property
    def n_total(self) -> int:
        """Total pipelines."""
        return len(self.train) + len(self.calibration) + len(self.test)

    def all_records(self) -> tuple[PipelineRecord, ...]:
        """Train, calibration and test records concatenated in that order."""
        return self.train + self.calibration + self.test


def feature_matrix(records: tuple[PipelineRecord, ...]) -> np.ndarray:
    """Stack probe feature vectors into ``(n_records, N_FEATURES)``."""
    if not records:
        raise ValueError("records must be non-empty")
    return np.array([r.probe.features for r in records], dtype=float)


def log_target(records: tuple[PipelineRecord, ...], p: float) -> np.ndarray:
    """Natural log of the reference quantile at ``p``, shape ``(n_records,)``."""
    if not records:
        raise ValueError("records must be non-empty")
    p = float(p)
    for r in records:
        if p not in r.reference_quantile_s:
            raise ValueError(f"record {r.index} has no reference quantile at p={p}")
    return np.log(np.array([r.reference_quantile_s[p] for r in records], dtype=float))


def build_dataset(
    regime: str,
    n_train: int = 200,
    n_calibration: int = 60,
    n_test: int = 100,
    seed: int = 20260402,
    n_probe: int = 256,
    n_reference: int = 100_000,
    n_reference_test: int | None = 400_000,
    **record_kwargs: object,
) -> PipelineDataset:
    """Generate a full train/calibration/test population for one regime.

    ``n_reference_test`` defaults to four times ``n_reference`` so that the
    scoring target carries less Monte Carlo noise than the training target,
    which is where the noise would otherwise be mistaken for model error.
    Each pipeline gets seed ``seed + 10 * index``, leaving room for the probe
    and reference streams.
    """
    if regime not in REGIMES:
        raise ValueError(f"regime must be one of {REGIMES}, got {regime!r}")
    for name, value in (("n_train", n_train), ("n_calibration", n_calibration), ("n_test", n_test)):
        if not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be an integer >= 1, got {value!r}")
    n_ref_test = int(n_reference if n_reference_test is None else n_reference_test)

    records: list[PipelineRecord] = []
    total = int(n_train) + int(n_calibration) + int(n_test)
    for i in range(total):
        is_test = i >= int(n_train) + int(n_calibration)
        records.append(
            generate_record(
                index=i,
                regime=regime,
                seed=int(seed) + 10 * i,
                n_probe=int(n_probe),
                n_reference=n_ref_test if is_test else int(n_reference),
                **record_kwargs,  # type: ignore[arg-type]
            )
        )
    a = int(n_train)
    b = a + int(n_calibration)
    return PipelineDataset(
        regime=regime,
        train=tuple(records[:a]),
        calibration=tuple(records[a:b]),
        test=tuple(records[b:]),
        seed=int(seed),
    )

"""The seeded scenario generator for the delay-asymmetry corrector.

What this generates, and the one assumption that decides the result
-------------------------------------------------------------------
Each scenario is one window of exchanges from a
:class:`~timetransfer.backends.simulated.SimulationSpec` whose parameters are
drawn from the distributions below, acquired through the ordinary HAL path so
that the training data comes from the same code the estimators run against.

The target is the window's mean asymmetry bias

    b = mean_k(d_ab,k - d_ba,k) / 2                                        [s]

which splits into a static and a queueing part. The static part is
**not** in the features (see :mod:`timetransfer.asym.features`), so whether any
corrector can recover it depends entirely on one declared modelling choice:

    floor_asymmetry = coupling * base_scale * (congestion_ab - congestion_ba)
                      + independent_asymmetry

``congestion_ab`` and ``congestion_ba`` are the per-direction queueing scale
multipliers, which *are* visible in the features. ``coupling`` therefore sets
how much of the static asymmetry is learnable, and ``independent_asymmetry`` is
the part that is not learnable by anything.

**This coupling is an assumption about networks, not a measurement of one.**
It is motivated -- an asymmetrically congested path tends to have both a
different queueing distribution and a different mean delay in the two
directions -- but no number here is fitted to a real trace. The default
``coupling = 1.0`` puts the coupled and independent components at comparable
size, which is deliberately the regime where the question "is a learned
corrector worth it" has a non-obvious answer. ``validation`` reports the result
at three couplings so the reader can see the whole curve rather than one point
chosen to flatter a model.

Distributions
-------------
Per scenario, with ``rng`` seeded from ``seed + scenario index``:

* ``offset0``             ``N(0, 1e-3)`` s
* ``skew``                ``N(0, 2e-6)`` s/s
* ``congestion_ab/ba``    ``Gamma(4, 0.25)``, mean 1, both > 0
* ``jitter_scale_*``      ``base_scale * congestion_*`` s
* ``jitter_shape_*``      ``U(1.2, 3.0)``, dimensionless
* ``independent_asym``    ``N(0, independent_sigma)`` s
* ``floor_mean``          ``U(2e-3, 2e-2)`` s, split by the asymmetry
* ``loss_probability``    ``U(0, 0.08)``
* clock noise             fixed ``q1 = 1e-20`` s^2/s, ``q2 = 1e-26`` 1/s

The clock noise is held fixed because the target does not depend on it and
varying it would only add label noise. Timestamp resolution is 1 ns and
timestamping noise is 20 ns, both fixed, because both are properties of the
measurement hardware and not of the scenario.

Determinism
-----------
Scenario ``i`` of a dataset built with ``seed`` depends only on ``seed + i``.
Rebuilding the dataset with the same seed reproduces it exactly, which
``validation/validate_reproducibility.py`` checks by digest. Nothing is cached.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..acquire import acquire
from ..backends import SimulatedBackend, SimulationSpec
from ..clockmodel import ClockSpec
from ..errors import ConfigurationError, InsufficientDataError
from ..exchange import ExchangeSet
from ..linkmodel import LinkSpec
from .features import FEATURE_NAMES, MIN_WINDOW, window_features

__all__ = ["AsymmetryDataset", "GeneratorConfig", "build_dataset", "generate_scenario"]


@dataclass(frozen=True)
class GeneratorConfig:
    """Declared parameters of the scenario generator.

    Attributes
    ----------
    window:
        Exchanges attempted per scenario, >= :data:`~timetransfer.asym.features.MIN_WINDOW`
        after losses.
    period_s:
        Exchange interval [s].
    base_jitter_scale_s:
        Gamma scale multiplied by the per-direction congestion factor [s].
    coupling:
        How strongly the static floor asymmetry follows the congestion
        difference, dimensionless, >= 0. ``0`` makes the static asymmetry
        entirely unlearnable.
    independent_sigma_s:
        Standard deviation of the part of the static asymmetry that is
        independent of anything observable [s].
    timestamp_noise_s:
        White-PM timestamping noise [s].
    resolution_ns:
        Timestamp quantisation [ns].
    max_loss_probability:
        Upper end of the per-scenario loss probability.
    """

    window: int = 150
    period_s: float = 1.0
    base_jitter_scale_s: float = 2.0e-4
    coupling: float = 1.0
    independent_sigma_s: float = 2.0e-4
    timestamp_noise_s: float = 20.0e-9
    resolution_ns: int = 1
    max_loss_probability: float = 0.08

    def __post_init__(self) -> None:
        if int(self.window) < MIN_WINDOW + 4:
            raise ConfigurationError(
                f"window must be at least {MIN_WINDOW + 4} to survive losses, "
                f"got {self.window!r}"
            )
        if float(self.coupling) < 0.0:
            raise ConfigurationError(f"coupling must be >= 0, got {self.coupling!r}")
        for name in ("base_jitter_scale_s", "independent_sigma_s"):
            if not (float(getattr(self, name)) > 0.0):
                raise ConfigurationError(f"{name} must be > 0, got {getattr(self, name)!r}")
        if not (0.0 <= float(self.max_loss_probability) < 0.5):
            raise ConfigurationError(
                f"max_loss_probability must be in [0, 0.5), got {self.max_loss_probability!r}"
            )


def generate_scenario(
    index: int, *, seed: int, config: GeneratorConfig
) -> tuple[ExchangeSet, dict[str, float]]:
    """One scenario: its exchanges and the truth about it.

    Parameters
    ----------
    index:
        Scenario index; the generator seed is ``seed + index``.
    seed:
        Base seed.
    config:
        Generator parameters.

    Returns
    -------
    tuple
        ``(exchanges, truth)``. ``truth`` carries ``bias_s`` (the target),
        ``static_bias_s``, ``queue_bias_s``, ``congestion_difference`` and the
        realised loss fraction.
    """
    scenario_seed = int(seed) + int(index)
    rng = np.random.Generator(np.random.PCG64(scenario_seed))
    congestion_ab = float(rng.gamma(4.0, 0.25))
    congestion_ba = float(rng.gamma(4.0, 0.25))
    scale_ab = float(config.base_jitter_scale_s) * congestion_ab
    scale_ba = float(config.base_jitter_scale_s) * congestion_ba
    shape_ab = float(rng.uniform(1.2, 3.0))
    shape_ba = float(rng.uniform(1.2, 3.0))
    independent = float(rng.normal(0.0, float(config.independent_sigma_s)))
    floor_asym = (
        float(config.coupling) * float(config.base_jitter_scale_s)
        * (congestion_ab - congestion_ba)
        + independent
    )
    floor_mean = float(rng.uniform(2.0e-3, 2.0e-2))
    half = 0.5 * floor_asym
    floor_ab = max(1.0e-6, floor_mean + half)
    floor_ba = max(1.0e-6, floor_mean - half)
    loss_p = float(rng.uniform(0.0, float(config.max_loss_probability)))
    spec = SimulationSpec(
        clock=ClockSpec(
            offset0_s=float(rng.normal(0.0, 1.0e-3)),
            skew_s_per_s=float(rng.normal(0.0, 2.0e-6)),
            q1_s2_per_s=1.0e-20,
            q2_per_s=1.0e-26,
        ),
        link=LinkSpec(
            floor_ab_s=floor_ab,
            floor_ba_s=floor_ba,
            jitter_shape_ab=shape_ab,
            jitter_shape_ba=shape_ba,
            jitter_scale_ab_s=scale_ab,
            jitter_scale_ba_s=scale_ba,
            loss_probability=loss_p,
        ),
        period_s=float(config.period_s),
        timestamp_noise_s=float(config.timestamp_noise_s),
        resolution_ns=int(config.resolution_ns),
        seed=scenario_seed,
    )
    backend = SimulatedBackend(spec)
    backend.open()
    report = acquire(backend, int(config.window), min_exchanges=MIN_WINDOW)
    backend.close()
    exchanges = report.exchanges
    if exchanges.true_asymmetry_s is None:
        raise ConfigurationError("simulated acquisition returned no truth column")
    bias = 0.5 * float(np.mean(exchanges.true_asymmetry_s))
    static_bias = 0.5 * floor_asym
    truth = {
        "bias_s": bias,
        "static_bias_s": static_bias,
        "queue_bias_s": bias - static_bias,
        "congestion_difference": congestion_ab - congestion_ba,
        "independent_asymmetry_s": independent,
        "loss_fraction": float(report.exchanges.loss_fraction),
        "n_exchanges": float(exchanges.n),
    }
    return exchanges, truth


@dataclass(frozen=True)
class AsymmetryDataset:
    """Features, targets and provenance for the asymmetry corrector.

    Attributes
    ----------
    features:
        ``(n_scenarios, 16)`` feature matrix; columns are
        :data:`~timetransfer.asym.features.FEATURE_NAMES`.
    target_s:
        ``(n_scenarios,)`` mean asymmetry bias [s]; the quantity to predict.
    static_bias_s, queue_bias_s:
        The target's decomposition [s], recorded for analysis and never used as
        a feature.
    congestion_difference:
        ``congestion_ab - congestion_ba`` per scenario, the latent the coupling
        acts through. Recorded so the measured learnable fraction can be
        checked against the declared coupling.
    seeds:
        Per-scenario seed, so any scenario can be regenerated alone.
    config:
        The generator configuration.
    base_seed:
        The base seed.
    """

    features: np.ndarray
    target_s: np.ndarray
    static_bias_s: np.ndarray
    queue_bias_s: np.ndarray
    congestion_difference: np.ndarray
    seeds: np.ndarray
    config: GeneratorConfig
    base_seed: int

    @property
    def n(self) -> int:
        """Number of scenarios."""
        return int(self.target_s.size)

    def split(self, *, train_fraction: float = 0.7) -> tuple[np.ndarray, np.ndarray]:
        """Index arrays for a deterministic scenario-wise train/test split.

        The split is by **scenario index order**, not shuffled: the first
        ``train_fraction`` of the scenarios train and the rest test. Because
        each scenario is an independent draw from the generator, an ordered
        split is as good as a shuffled one and it is reproducible without a
        second seed.
        """
        frac = float(train_fraction)
        if not (0.0 < frac < 1.0):
            raise ConfigurationError(f"train_fraction must be in (0, 1), got {frac!r}")
        n_train = int(round(frac * self.n))
        if n_train < 2 or self.n - n_train < 2:
            raise InsufficientDataError(
                f"split of {self.n} scenarios at {frac} leaves too few on one side",
                n_usable=self.n,
                n_required=8,
            )
        idx = np.arange(self.n)
        return idx[:n_train], idx[n_train:]

    def as_text(self) -> str:
        """Summary for a validation output file."""
        return (
            f"scenarios              : {self.n}\n"
            f"features               : {len(FEATURE_NAMES)}\n"
            f"base seed              : {self.base_seed}\n"
            f"window (attempted)     : {self.config.window}\n"
            f"coupling               : {self.config.coupling:.3f}\n"
            f"independent sigma      : {self.config.independent_sigma_s:.3e} s\n"
            f"target  mean / std     : {float(np.mean(self.target_s)):.6e} / "
            f"{float(np.std(self.target_s, ddof=1)):.6e} s\n"
            f"static  mean / std     : {float(np.mean(self.static_bias_s)):.6e} / "
            f"{float(np.std(self.static_bias_s, ddof=1)):.6e} s\n"
            f"queue   mean / std     : {float(np.mean(self.queue_bias_s)):.6e} / "
            f"{float(np.std(self.queue_bias_s, ddof=1)):.6e} s"
        )


def build_dataset(
    n_scenarios: int, *, seed: int = 20261024, config: GeneratorConfig | None = None
) -> AsymmetryDataset:
    """Generate ``n_scenarios`` scenarios and extract their features and targets.

    Parameters
    ----------
    n_scenarios:
        Number of scenarios, >= 8.
    seed:
        Base seed; scenario ``i`` uses ``seed + i``.
    config:
        Generator parameters; defaults to :class:`GeneratorConfig`.

    Returns
    -------
    AsymmetryDataset
    """
    if int(n_scenarios) < 8:
        raise ConfigurationError(f"n_scenarios must be >= 8, got {n_scenarios!r}")
    cfg = GeneratorConfig() if config is None else config
    rows: list[np.ndarray] = []
    target: list[float] = []
    static: list[float] = []
    queue: list[float] = []
    congestion: list[float] = []
    seeds: list[int] = []
    for i in range(int(n_scenarios)):
        exchanges, truth = generate_scenario(i, seed=int(seed), config=cfg)
        rows.append(window_features(exchanges))
        target.append(truth["bias_s"])
        static.append(truth["static_bias_s"])
        queue.append(truth["queue_bias_s"])
        congestion.append(truth["congestion_difference"])
        seeds.append(int(seed) + i)
    return AsymmetryDataset(
        features=np.asarray(rows, dtype=np.float64),
        target_s=np.asarray(target, dtype=np.float64),
        static_bias_s=np.asarray(static, dtype=np.float64),
        queue_bias_s=np.asarray(queue, dtype=np.float64),
        congestion_difference=np.asarray(congestion, dtype=np.float64),
        seeds=np.asarray(seeds, dtype=np.int64),
        config=cfg,
        base_seed=int(seed),
    )

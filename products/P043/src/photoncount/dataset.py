"""Synthetic data for the learned rate correction, generated deterministically.

Every row of this dataset is one simulated acquisition from
:mod:`photoncount.simulate`: a detector with a known dead-time model, dead time,
afterpulse probability and afterpulse delay, illuminated at a known true rate,
read out as per-window counts. The label is the true rate; the features are what
a receiver can actually know.

**The dimensionless design variable is ``x = n tau``**, not the rate. Everything
about dead time depends only on that product, so the sampling is log-uniform in
``x`` over a range that deliberately straddles the paralyzable maximum at
``x = 1``: below it the closed-form inversions are excellent, at it they are
ill-conditioned, and above it the lower-branch inversion is simply the wrong
root. A dataset that stops at ``x = 0.1`` cannot show anything interesting,
because there the textbook answer is right.

**Features (5).** All available from a real run plus the detector's declared
datasheet values:

====  ==========================  =========================================
 0    ``log10(m tau)``            observed rate times declared dead time
 1    ``fano_factor``             sample Var/mean of per-window counts
 2    ``afterpulse_probability``  declared ``p``
 3    ``log10(t_ap / tau)``       declared delay ratio
 4    ``is_paralyzable``          1.0 or 0.0, from the declared model
====  ==========================  =========================================

Feature 1 is the one doing physical work. Dead time anti-bunches counts and
pushes the Fano factor below 1; afterpulsing clusters them and pushes it above
1. The pair ``(m tau, Fano)`` is therefore the observable signature that
separates the two effects, and it is information no rate equation uses.

**A feature that was removed, and why.** An earlier version of this generator
also exposed the mean count per window. Because the window length is chosen as
``events_per_window / n`` to bound the simulation cost, the mean count per
window is ``events_per_window * m / n`` --- it contains the throughput ratio,
hence the answer. With it present the learned model reached the counting-noise
floor in every regime, including the regimes where the closed forms are exactly
right, which is the signature of a label leak rather than of a good model. The
feature is gone and the results below are from the 5 features above.

**Label.** ``log10(x) = log10(n tau)``. Working in the log of the dimensionless
product is what makes one model cover five decades of rate and two of dead time;
predicting the rate directly would be dominated by the scale.

**Honesty about the features.** Features 2, 3 and 4 are *declared* detector
parameters. The closed-form baselines in :mod:`photoncount.correction` can use
0 and 4 (and the composed baseline can use 2 as well), so the comparison is run
both ways: against the two textbook inversions, which structurally cannot accept
``p``, and against a composed closed form that gets exactly the same
information as the learned model. A learned model that only beats a baseline
denied half its inputs has not shown anything.

Compute: generating the default 6000-row training set and 2000-row test set is
roughly 4 x 10^7 simulated events. Measured in this build container (2 shared
cores) it runs well inside the 3-minute budget; the measured figure is in
``validation/validate_correction_output.txt``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .simulate import DetectorSpec, simulate_run

__all__ = [
    "FEATURE_NAMES",
    "SamplingRanges",
    "CorrectionDataset",
    "generate_dataset",
    "make_features",
]

#: Feature column names, in order. The single source of truth for the layout.
FEATURE_NAMES: tuple[str, ...] = (
    "log10_observed_x",
    "fano_factor",
    "afterpulse_probability",
    "log10_delay_ratio",
    "is_paralyzable",
)


@dataclass(frozen=True)
class SamplingRanges:
    """Ranges the generator samples from.

    Attributes
    ----------
    dead_time_s:
        ``(low, high)``, log-uniform, s.
    true_x:
        ``(low, high)`` for ``x = n tau``, log-uniform (-). The default
        ``(2e-3, 3.0)`` straddles the paralyzable maximum at ``x = 1``.
    afterpulse_probability:
        ``(low, high)``, uniform (-).
    delay_ratio:
        ``(low, high)`` for ``t_ap / tau``, log-uniform (-).
    events_per_window:
        Target expected primary events per window; sets ``window_s = k / n``.
    n_windows:
        Windows per row. Must be >= 2 for a Fano factor to exist.
    """

    dead_time_s: tuple[float, float] = (2e-8, 1e-6)
    true_x: tuple[float, float] = (2e-3, 3.0)
    afterpulse_probability: tuple[float, float] = (0.0, 0.15)
    delay_ratio: tuple[float, float] = (0.3, 30.0)
    events_per_window: float = 200.0
    n_windows: int = 25

    def __post_init__(self) -> None:
        for name in ("dead_time_s", "true_x", "delay_ratio"):
            lo, hi = getattr(self, name)
            if not (0.0 < lo < hi):
                raise ValueError(f"{name} must satisfy 0 < low < high, got {(lo, hi)!r}")
        lo, hi = self.afterpulse_probability
        if not (0.0 <= lo < hi < 1.0):
            raise ValueError(
                f"afterpulse_probability must satisfy 0 <= low < high < 1, got {(lo, hi)!r}"
            )
        if self.events_per_window < 10.0:
            raise ValueError(
                f"events_per_window must be >= 10, got {self.events_per_window!r}"
            )
        if self.n_windows < 2:
            raise ValueError(f"n_windows must be >= 2, got {self.n_windows!r}")


@dataclass
class CorrectionDataset:
    """A generated dataset.

    Attributes
    ----------
    features:
        ``(n, 5)`` float array, columns as :data:`FEATURE_NAMES`.
    label:
        ``(n,)`` float array, ``log10(n tau)``.
    true_rate_hz, observed_rate_hz, dead_time_s:
        ``(n,)`` arrays, counts/s and s, kept so a baseline can be applied and
        errors reported in rate rather than in log space.
    is_paralyzable, afterpulse_probability, delay_ratio, fano_factor, count_mean:
        ``(n,)`` arrays of the per-row conditions. ``count_mean`` is kept for
        reporting the counting-noise floor and is **not** a feature; see the
        module docstring.
    seed:
        The seed the rows were generated from.
    """

    features: np.ndarray
    label: np.ndarray
    true_rate_hz: np.ndarray
    observed_rate_hz: np.ndarray
    dead_time_s: np.ndarray
    is_paralyzable: np.ndarray
    afterpulse_probability: np.ndarray
    delay_ratio: np.ndarray
    fano_factor: np.ndarray
    count_mean: np.ndarray
    seed: int

    def __len__(self) -> int:
        return int(self.label.size)

    @property
    def true_x(self) -> np.ndarray:
        """``n tau`` per row (-)."""
        return self.true_rate_hz * self.dead_time_s

    @property
    def observed_x(self) -> np.ndarray:
        """``m tau`` per row (-)."""
        return self.observed_rate_hz * self.dead_time_s


def make_features(
    observed_rate_hz: np.ndarray,
    dead_time_s: np.ndarray,
    fano_factor: np.ndarray,
    afterpulse_probability: np.ndarray,
    delay_ratio: np.ndarray,
    is_paralyzable: np.ndarray,
) -> np.ndarray:
    """Assemble the ``(n, 5)`` feature matrix from measured and declared values.

    All inputs broadcast to a common length. A zero observed rate is floored at
    a small positive value before the log, because a window that registered
    nothing is informative (the rate is low) and must not become a NaN.
    """
    arrays = np.broadcast_arrays(
        np.asarray(observed_rate_hz, dtype=float),
        np.asarray(dead_time_s, dtype=float),
        np.asarray(fano_factor, dtype=float),
        np.asarray(afterpulse_probability, dtype=float),
        np.asarray(delay_ratio, dtype=float),
        np.asarray(is_paralyzable, dtype=float),
    )
    m, tau, fano, p_ap, ratio, par = (np.atleast_1d(a).astype(float) for a in arrays)
    floor = 1e-12
    obs_x = np.maximum(m * tau, floor)
    return np.column_stack(
        [
            np.log10(obs_x),
            np.nan_to_num(fano, nan=1.0, posinf=10.0, neginf=0.0),
            p_ap,
            np.log10(np.maximum(ratio, floor)),
            par,
        ]
    )


def generate_dataset(
    n_rows: int,
    seed: int,
    ranges: SamplingRanges | None = None,
) -> CorrectionDataset:
    """Generate ``n_rows`` simulated acquisitions. Deterministic in ``seed``.

    Each row draws ``tau``, ``x = n tau``, ``p``, ``t_ap/tau`` and a dead-time
    model, then simulates ``ranges.n_windows`` windows of
    ``ranges.events_per_window / n`` seconds and reduces them to the feature
    vector and label.
    """
    rows = int(n_rows)
    if rows < 1:
        raise ValueError(f"n_rows must be >= 1, got {rows!r}")
    rg = ranges if ranges is not None else SamplingRanges()
    rng = np.random.default_rng(int(seed))

    tau = 10.0 ** rng.uniform(*np.log10(rg.dead_time_s), size=rows)
    x_true = 10.0 ** rng.uniform(*np.log10(rg.true_x), size=rows)
    p_ap = rng.uniform(*rg.afterpulse_probability, size=rows)
    ratio = 10.0 ** rng.uniform(*np.log10(rg.delay_ratio), size=rows)
    is_par = rng.integers(0, 2, size=rows).astype(float)

    n_true = x_true / tau
    window_s = rg.events_per_window / n_true

    observed = np.empty(rows)
    fano = np.empty(rows)
    cmean = np.empty(rows)
    for i in range(rows):
        spec = DetectorSpec(
            dead_time_s=float(tau[i]),
            model="paralyzable" if is_par[i] > 0.5 else "nonparalyzable",
            afterpulse_probability=float(p_ap[i]),
            afterpulse_mean_delay_s=float(ratio[i] * tau[i]),
            cascading=True,
        )
        counts = np.empty(rg.n_windows, dtype=float)
        for w in range(rg.n_windows):
            counts[w] = simulate_run(float(n_true[i]), float(window_s[i]), spec, rng).registered
        mean = float(counts.mean())
        cmean[i] = mean
        observed[i] = mean / float(window_s[i])
        fano[i] = float(counts.var(ddof=1)) / mean if mean > 0.0 else 1.0

    features = make_features(observed, tau, fano, p_ap, ratio, is_par)
    return CorrectionDataset(
        features=features,
        label=np.log10(x_true),
        true_rate_hz=n_true,
        observed_rate_hz=observed,
        dead_time_s=tau,
        is_paralyzable=is_par,
        afterpulse_probability=p_ap,
        delay_ratio=ratio,
        fano_factor=fano,
        count_mean=cmean,
        seed=int(seed),
    )

"""Recovering the incident rate from observed counts: closed forms, then a model.

The question is simple to state and has no closed-form answer. A detector with
dead time ``tau`` **and** afterpulsing ``p`` reports ``m`` counts per second.
What was the incident rate ``n``?

**The closed forms, implemented first.** Each inverts one effect and assumes the
other is absent:

    non-paralyzable:  n = m / (1 - m tau)                     (D2), undefined at m tau >= 1
    paralyzable:      n = -W_0(-m tau) / tau                  (D5) lower branch,
                                                              undefined at m tau > 1/e

A fair comparison needs a third, because the two above structurally cannot
accept ``p``:

    composed:         n = invert_deadtime( m (1 - p) )        (A2) then (D2)/(D5)

The composed baseline gets **exactly the same information as the learned
model**. It is the one that matters. A learned model that beats only the two
textbook inversions has beaten baselines denied an input, which proves nothing.

**Why a learned correction might help at all.** The composition of the two
effects is not the product of their inverses: an afterpulse extends a
paralyzable detector's dead period, and a dead period suppresses an afterpulse,
so the order matters and neither inverse is right. Near ``m tau = 1/e`` the
paralyzable forward map is flat, so its inverse amplifies any model error
without bound, and above the maximum the lower branch is the wrong root. Those
are the regimes where a correction fitted on simulated truth has something to
add. Away from them the closed forms are excellent and are expected to win.

**Uncertainty.** :class:`RateCorrector` fits three quantile regressors (0.05,
0.5, 0.95) on the same features. The 0.5 model is the point estimate, so point
and interval come from one family. The interval's **measured** coverage on
held-out data is reported by :func:`interval_coverage` and is not assumed to be
0.90.

Everything here works in ``log10(n tau)``. Converting back,
``n = 10**prediction / tau``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import special
from sklearn.ensemble import GradientBoostingRegressor

from .dataset import FEATURE_NAMES, CorrectionDataset

__all__ = [
    "QUANTILES",
    "RateCorrector",
    "composed_baseline",
    "interval_coverage",
    "matched_baseline",
    "nonparalyzable_baseline",
    "paralyzable_baseline",
    "rate_error_metrics",
    "summarise_by_regime",
]

#: Quantiles the corrector fits: lower interval edge, median, upper edge.
QUANTILES: tuple[float, float, float] = (0.05, 0.5, 0.95)


def _arrays(*values: np.ndarray | float) -> tuple[np.ndarray, ...]:
    return tuple(np.atleast_1d(np.asarray(v, dtype=float)) for v in np.broadcast_arrays(*values))


def nonparalyzable_baseline(
    observed_rate_hz: np.ndarray | float, dead_time_s: np.ndarray | float
) -> np.ndarray:
    """(D2) ``n = m / (1 - m tau)``, counts/s. ``nan`` where ``m tau >= 1``.

    Returning ``nan`` rather than raising is deliberate: a baseline that cannot
    answer must be visible as a gap in the comparison, not as an exception that
    removes the row from the dataset.
    """
    m, tau = _arrays(observed_rate_hz, dead_time_s)
    x = m * tau
    out = np.full(m.shape, np.nan)
    ok = x < 1.0
    out[ok] = m[ok] / (1.0 - x[ok])
    return out


def paralyzable_baseline(
    observed_rate_hz: np.ndarray | float,
    dead_time_s: np.ndarray | float,
    branch: str = "lower",
) -> np.ndarray:
    """(D5) ``n = -W(-m tau)/tau``, counts/s. ``nan`` where ``m tau > 1/e``.

    ``branch="lower"`` uses ``W_0``, ``"upper"`` uses ``W_{-1}``. The lower
    branch is the usual choice and is wrong whenever the detector is driven past
    ``1/tau``; the dataset deliberately contains such rows.
    """
    if branch not in ("lower", "upper"):
        raise ValueError(f"branch must be 'lower' or 'upper', got {branch!r}")
    m, tau = _arrays(observed_rate_hz, dead_time_s)
    x = m * tau
    out = np.full(m.shape, np.nan)
    ok = x <= 1.0 / np.e
    k = 0 if branch == "lower" else -1
    if np.any(ok):
        w = special.lambertw(-x[ok], k=k)
        out[ok] = -np.real(w) / tau[ok]
    return out


def matched_baseline(
    observed_rate_hz: np.ndarray | float,
    dead_time_s: np.ndarray | float,
    is_paralyzable: np.ndarray | float,
    branch: str = "lower",
) -> np.ndarray:
    """Model-matched closed form: (D5) where paralyzable, (D2) where not. counts/s."""
    m, tau, par = _arrays(observed_rate_hz, dead_time_s, is_paralyzable)
    out = np.where(
        par > 0.5,
        paralyzable_baseline(m, tau, branch),
        nonparalyzable_baseline(m, tau),
    )
    return np.asarray(out, dtype=float)


def composed_baseline(
    observed_rate_hz: np.ndarray | float,
    dead_time_s: np.ndarray | float,
    is_paralyzable: np.ndarray | float,
    afterpulse_probability: np.ndarray | float,
    branch: str = "lower",
) -> np.ndarray:
    """Afterpulse inverse (A2) then the model-matched dead-time inverse. counts/s.

    ``m_primary = m (1 - p)`` removes the cascading afterpulse inflation
    exactly, *if* dead time were absent. It is not, so this is an
    approximation; it is nevertheless the strongest closed form available with
    the same inputs the learned model gets, and it is the baseline to beat.
    """
    m, tau, par, p = _arrays(
        observed_rate_hz, dead_time_s, is_paralyzable, afterpulse_probability
    )
    return matched_baseline(m * (1.0 - p), tau, par, branch)


def rate_error_metrics(
    estimate_hz: np.ndarray, truth_hz: np.ndarray, name: str = "estimator"
) -> dict[str, float | str]:
    """Relative-error summary of a rate estimator, over the rows it defined.

    Returns ``name``, ``n_total``, ``n_defined``, ``defined_fraction``,
    ``median_abs_rel_error``, ``p90_abs_rel_error``, ``mean_abs_log10_error``,
    ``median_signed_rel_error``. Relative error is
    ``(estimate - truth) / truth``, dimensionless. Rows the estimator returned
    ``nan`` for are excluded from the error statistics and counted in
    ``n_defined``: an estimator that answers only the easy half of the data must
    not look accurate by that alone.
    """
    est = np.asarray(estimate_hz, dtype=float)
    tru = np.asarray(truth_hz, dtype=float)
    if est.shape != tru.shape:
        raise ValueError(f"shape mismatch: {est.shape} vs {tru.shape}")
    defined = np.isfinite(est) & (est > 0.0)
    rel = np.full(est.shape, np.nan)
    rel[defined] = (est[defined] - tru[defined]) / tru[defined]
    n_def = int(defined.sum())
    if n_def == 0:
        return {
            "name": name,
            "n_total": float(est.size),
            "n_defined": 0.0,
            "defined_fraction": 0.0,
            "median_abs_rel_error": float("nan"),
            "p90_abs_rel_error": float("nan"),
            "mean_abs_log10_error": float("nan"),
            "median_signed_rel_error": float("nan"),
        }
    absrel = np.abs(rel[defined])
    return {
        "name": name,
        "n_total": float(est.size),
        "n_defined": float(n_def),
        "defined_fraction": n_def / est.size,
        "median_abs_rel_error": float(np.median(absrel)),
        "p90_abs_rel_error": float(np.percentile(absrel, 90)),
        "mean_abs_log10_error": float(
            np.mean(np.abs(np.log10(est[defined] / tru[defined])))
        ),
        "median_signed_rel_error": float(np.median(rel[defined])),
    }


def interval_coverage(
    lower_hz: np.ndarray, upper_hz: np.ndarray, truth_hz: np.ndarray
) -> dict[str, float]:
    """Measured coverage and width of an interval estimate.

    Returns ``coverage`` (fraction of rows with ``lower <= truth <= upper``),
    ``nominal`` (the interval's design coverage, 0.90 for :data:`QUANTILES`),
    ``median_relative_width`` (``(upper - lower) / truth``), and
    ``n``. Coverage is measured, never assumed.
    """
    lo = np.asarray(lower_hz, dtype=float)
    hi = np.asarray(upper_hz, dtype=float)
    tru = np.asarray(truth_hz, dtype=float)
    if not (lo.shape == hi.shape == tru.shape):
        raise ValueError("lower, upper and truth must have the same shape")
    inside = (tru >= lo) & (tru <= hi)
    return {
        "coverage": float(inside.mean()),
        "nominal": float(QUANTILES[2] - QUANTILES[0]),
        "median_relative_width": float(np.median((hi - lo) / tru)),
        "n": float(tru.size),
    }


@dataclass
class RateCorrector:
    """Learned incident-rate correction with a measured uncertainty interval.

    Three gradient-boosted quantile regressors on the six features of
    :mod:`photoncount.dataset`, predicting ``log10(n tau)``. The 0.5 model is
    the point estimate.

    Hyperparameters are deliberately small: 6 features, a few thousand rows and
    a 2-core shared container. ``n_estimators=250``, ``max_depth=3``,
    ``learning_rate=0.08``, ``subsample=1.0``, fixed ``random_state``. No search
    was run over them; they were chosen to fit inside the compute budget and
    are reported as such rather than as a tuned optimum.
    """

    n_estimators: int = 250
    max_depth: int = 3
    learning_rate: float = 0.08
    random_state: int = 0
    models_: dict[float, GradientBoostingRegressor] | None = None
    feature_names_: tuple[str, ...] = FEATURE_NAMES

    def fit(self, dataset: CorrectionDataset) -> RateCorrector:
        """Fit the three quantile models on ``dataset``. Returns ``self``."""
        if dataset.features.shape[1] != len(FEATURE_NAMES):
            raise ValueError(
                f"expected {len(FEATURE_NAMES)} features, got {dataset.features.shape[1]}"
            )
        self.models_ = {}
        for q in QUANTILES:
            model = GradientBoostingRegressor(
                loss="quantile",
                alpha=q,
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.random_state,
            )
            model.fit(dataset.features, dataset.label)
            self.models_[q] = model
        return self

    def _check_fitted(self) -> dict[float, GradientBoostingRegressor]:
        if self.models_ is None:
            raise RuntimeError("RateCorrector is not fitted; call fit() first")
        return self.models_

    def predict_log_x(self, features: np.ndarray) -> dict[str, np.ndarray]:
        """Predict ``log10(n tau)``. Keys ``lower``, ``median``, ``upper``.

        The quantile models are fitted independently, so a crossing (lower above
        upper) is possible in principle; the returned arrays are sorted per row
        so the interval is always well formed, and
        :func:`interval_coverage` measures what that costs.
        """
        models = self._check_fitted()
        x = np.atleast_2d(np.asarray(features, dtype=float))
        if x.shape[1] != len(FEATURE_NAMES):
            raise ValueError(f"expected {len(FEATURE_NAMES)} features, got {x.shape[1]}")
        raw = np.column_stack([models[q].predict(x) for q in QUANTILES])
        raw.sort(axis=1)
        return {"lower": raw[:, 0], "median": raw[:, 1], "upper": raw[:, 2]}

    def predict_rate(
        self, features: np.ndarray, dead_time_s: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Predict the incident rate, counts/s. Keys ``lower``, ``median``, ``upper``.

        ``n = 10**log10(n tau) / tau``, so the interval in rate is the interval
        in ``log10(n tau)`` mapped through a monotone function and is therefore
        the same probability statement.
        """
        tau = np.atleast_1d(np.asarray(dead_time_s, dtype=float))
        logs = self.predict_log_x(features)
        n = logs["median"].size
        if tau.size not in (1, n):
            raise ValueError(f"dead_time_s must have length 1 or {n}, got {tau.size}")
        return {k: 10.0**v / tau for k, v in logs.items()}

    def feature_importances(self) -> dict[str, float]:
        """Median gain-based importance of each feature across the three models (-).

        Gain importances are a diagnostic, not an explanation: they are not
        causal and they are unstable under correlated features.
        """
        models = self._check_fitted()
        stack = np.vstack([models[q].feature_importances_ for q in QUANTILES])
        med = np.median(stack, axis=0)
        return dict(zip(self.feature_names_, (float(v) for v in med), strict=True))

    def save(self, path: str | Path) -> None:
        """Persist with joblib. No model binary format: ``.joblib`` only."""
        import joblib

        target = Path(path)
        if target.suffix != ".joblib":
            raise ValueError(f"path must end in .joblib, got {target.name!r}")
        joblib.dump(
            {
                "version": 1,
                "quantiles": QUANTILES,
                "feature_names": list(self.feature_names_),
                "models": self._check_fitted(),
                "hyperparameters": {
                    "n_estimators": self.n_estimators,
                    "max_depth": self.max_depth,
                    "learning_rate": self.learning_rate,
                    "random_state": self.random_state,
                },
            },
            target,
            compress=3,
        )

    @classmethod
    def load(cls, path: str | Path) -> RateCorrector:
        """Load a corrector saved by :meth:`save`."""
        import joblib

        blob: dict[str, Any] = joblib.load(Path(path))
        if blob.get("version") != 1:
            raise ValueError(f"unsupported corrector version {blob.get('version')!r}")
        hp = blob["hyperparameters"]
        obj = cls(
            n_estimators=hp["n_estimators"],
            max_depth=hp["max_depth"],
            learning_rate=hp["learning_rate"],
            random_state=hp["random_state"],
        )
        obj.models_ = blob["models"]
        obj.feature_names_ = tuple(blob["feature_names"])
        return obj


def summarise_by_regime(
    estimates: dict[str, np.ndarray],
    truth_hz: np.ndarray,
    true_x: np.ndarray,
    afterpulse_probability: np.ndarray,
    is_paralyzable: np.ndarray,
) -> list[dict[str, float | str]]:
    """Per-regime error table for several estimators at once.

    Regimes, chosen because they are where the closed forms are expected to
    differ rather than to flatter any estimator:

    * ``x < 0.1`` --- the textbook regime, where the inversions are excellent
    * ``0.1 <= x < 0.7`` --- moderate loading
    * ``0.7 <= x <= 1.5`` --- around the paralyzable maximum at ``x = 1``
    * ``x > 1.5`` --- past the maximum, where the lower branch is the wrong root

    each split by ``p <= 0.02`` versus ``p > 0.02`` and by dead-time model, plus
    ``x > 1.5`` split by model, because the two behave completely differently
    there: the non-paralyzable inverse is still correct, the paralyzable lower
    branch is the wrong root.
    Returns one row per (regime, estimator) with the fields of
    :func:`rate_error_metrics` plus ``regime``.
    """
    x = np.asarray(true_x, dtype=float)
    p = np.asarray(afterpulse_probability, dtype=float)
    par = np.asarray(is_paralyzable, dtype=float) > 0.5
    bands = [
        ("x<0.1", x < 0.1),
        ("0.1<=x<0.7", (x >= 0.1) & (x < 0.7)),
        ("0.7<=x<=1.5", (x >= 0.7) & (x <= 1.5)),
        ("x>1.5", x > 1.5),
        ("p<=0.02", p <= 0.02),
        ("p>0.02", p > 0.02),
        ("paralyzable", par),
        ("nonparalyzable", ~par),
        ("paralyzable & x>1.5", par & (x > 1.5)),
        ("nonparalyzable & x>1.5", (~par) & (x > 1.5)),
        ("paralyzable & x>0.7 & p>0.02", par & (x > 0.7) & (p > 0.02)),
        ("all", np.ones_like(x, dtype=bool)),
    ]
    rows: list[dict[str, float | str]] = []
    for label, mask in bands:
        if not np.any(mask):
            continue
        for name, est in estimates.items():
            row = rate_error_metrics(np.asarray(est)[mask], np.asarray(truth_hz)[mask], name)
            row["regime"] = label
            rows.append(row)
    return rows

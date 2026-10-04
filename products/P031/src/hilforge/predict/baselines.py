"""The two deterministic baselines, implemented before the learned model.

Baseline 1 — fixed threshold on the previous iteration
------------------------------------------------------
Flag the next window as overrun-prone iff the latest iteration's duration
exceeded a threshold. One parameter, no state, no training beyond choosing the
threshold. This is what a flight-software engineer writes in an afternoon, and
on a trace whose overruns are close to independent it is hard to beat, because
the only usable signal is the current duration.

The threshold is chosen by sweeping the empirical quantiles of the training
durations and taking the one that maximises F1 on the training part. The whole
sweep is kept in :attr:`FixedThresholdPredictor.sweep_` so the choice can be
audited rather than taken on trust.

Baseline 2 — Markov-modulated stage-latency model
-------------------------------------------------
A two-state Markov chain over "calm" and "busy" regimes, with a gamma
distribution of iteration duration in each:

1. **Regime labelling.** The training durations are split into two classes at
   the threshold that minimises within-class variance — Otsu's method
   (Otsu, "A Threshold Selection Method from Gray-Level Histograms", *IEEE
   Transactions on Systems, Man, and Cybernetics* 9(1):62-66, 1979, §3). It
   is used here on a 1-D latency histogram rather than an image, which is the
   same optimisation.
2. **Transition matrix.** ``P[i, j]`` is the maximum-likelihood estimate from
   transition counts, ``n_ij / sum_j n_ij``, for an observed finite-state
   Markov chain (Billingsley, *Statistical Inference for Markov Processes*,
   University of Chicago Press 1961, Chapter 1).
3. **Duration law per regime.** Gamma fitted by the method of moments:
   ``shape = mean^2 / var``, ``scale = var / mean``. Gamma rather than
   exponential because measured execution times are right-skewed with a
   tail heavier than exponential at small shape (Harchol-Balter,
   *Performance Modeling and Design of Computer Systems*, Cambridge
   University Press 2013, Chapter 20).
4. **Prediction.** With the current regime ``s`` read off the same Otsu
   threshold, the regime distribution ``h`` steps ahead is ``e_s P^h``
   (Chapman-Kolmogorov). The probability of no overrun over the horizon is
   taken as the product over ``h`` of ``sum_z pi_h[z] F_z(D)``, where
   ``F_z`` is the fitted gamma CDF in regime ``z`` and ``D`` the deadline.

   **Stated approximation:** that product treats the durations in successive
   iterations as independent given their regimes. They are not quite — the
   regime sequence is shared — so the model underestimates the probability of
   a run of overruns. This is a known bias of the baseline, not a bug, and it
   is one of the reasons the queueing baseline is expected to be
   conservative on bursty traces.

Utilisation reference
---------------------
:func:`md1_mean_wait_s` is the Pollaczek-Khinchine mean waiting time for
M/D/1, ``W_q = rho * S / (2 (1 - rho))`` (Kleinrock, *Queueing Systems,
Volume 1: Theory*, Wiley 1975, §5.6, the M/G/1 P-K formula with a
deterministic service time, ``C^2 = 0``). It is not used by either predictor;
it is the standard statement of why a loop at high utilisation has a long
overrun tail even when its mean duration fits inside the period, and it is
quoted in the README for that reason. Validity: Poisson arrivals, single
server, FCFS, ``rho < 1`` — none of which a periodic control loop satisfies
exactly, so it is a reference point and not a prediction.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from scipy import stats

from ..errors import ConfigurationError
from .features import FEATURE_NAMES
from .metrics import cutoff_for_flag_rate

__all__ = ["FixedThresholdPredictor", "QueueingOverrunPredictor", "md1_mean_wait_s"]

_TOTAL_COL = FEATURE_NAMES.index("total_last_s")


def md1_mean_wait_s(rho: float, service_s: float) -> float:
    """Pollaczek-Khinchine mean queueing delay for M/D/1 [s].

    ``W_q = rho * service_s / (2 (1 - rho))``.

    Parameters
    ----------
    rho:
        Utilisation in ``[0, 1)``.
    service_s:
        Deterministic service time [s], > 0.
    """
    if not (0.0 <= rho < 1.0):
        raise ConfigurationError(f"rho must be in [0, 1), got {rho!r}")
    if not (service_s > 0.0):
        raise ConfigurationError(f"service_s must be > 0, got {service_s!r}")
    return rho * service_s / (2.0 * (1.0 - rho))


def _f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = float(np.sum(y_true & y_pred))
    fp = float(np.sum(~y_true & y_pred))
    fn = float(np.sum(y_true & ~y_pred))
    denom = 2.0 * tp + fp + fn
    return 2.0 * tp / denom if denom > 0.0 else 0.0


def _otsu_threshold(x: np.ndarray, n_bins: int = 256) -> float:
    """Otsu 1979 threshold on a 1-D sample: minimum within-class variance."""
    counts, edges = np.histogram(x, bins=n_bins)
    centres = 0.5 * (edges[:-1] + edges[1:])
    total = counts.sum()
    if total == 0:
        raise ValueError("cannot threshold an empty sample")
    w0 = np.cumsum(counts) / total
    w1 = 1.0 - w0
    m_cum = np.cumsum(counts * centres) / total
    m_tot = m_cum[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        mu0 = np.where(w0 > 0, m_cum / w0, 0.0)
        mu1 = np.where(w1 > 0, (m_tot - m_cum) / w1, 0.0)
        between = w0 * w1 * (mu0 - mu1) ** 2
    between[~np.isfinite(between)] = -np.inf
    return float(edges[int(np.argmax(between)) + 1])


@dataclass
class _GammaFit:
    """Method-of-moments gamma fit of a duration sample."""

    shape: float
    scale: float
    mean_s: float
    n: int

    def sf(self, x: float) -> float:
        """``P(X > x)`` [-]."""
        return float(stats.gamma.sf(x, self.shape, scale=self.scale))

    def cdf(self, x: float) -> float:
        """``P(X <= x)`` [-]."""
        return float(stats.gamma.cdf(x, self.shape, scale=self.scale))


def _fit_gamma_mom(x: np.ndarray) -> _GammaFit:
    m = float(np.mean(x))
    v = float(np.var(x, ddof=1)) if x.size > 1 else 0.0
    if v <= 0.0 or m <= 0.0:
        # Degenerate sample: a point mass. Represent it as a very tight gamma
        # so the CDF is still usable, and record it honestly via shape.
        return _GammaFit(shape=1.0e6, scale=max(m, 1.0e-12) / 1.0e6, mean_s=m, n=int(x.size))
    return _GammaFit(shape=m * m / v, scale=v / m, mean_s=m, n=int(x.size))


class FixedThresholdPredictor:
    """Baseline 1: threshold on the previous iteration's duration.

    Parameters
    ----------
    deadline_s:
        Relative deadline [s], > 0. Used only as the default threshold and to
        report the threshold as a fraction of the deadline.
    n_candidates:
        Number of quantile candidates in the training sweep, >= 2.

    Attributes
    ----------
    threshold_s:
        Chosen threshold [s].
    sweep_:
        ``(n_candidates, 2)`` array of ``[threshold_s, train_f1]``.
    positive_rate_:
        ``P(y = 1 | total_last > threshold)`` on the training set, used as the
        predictor's probability output for a flagged iteration.
    negative_rate_:
        ``P(y = 1 | total_last <= threshold)`` on the training set.
    """

    name = "fixed_threshold"

    def __init__(self, *, deadline_s: float, n_candidates: int = 199) -> None:
        if not (deadline_s > 0.0):
            raise ConfigurationError(f"deadline_s must be > 0, got {deadline_s!r}")
        if n_candidates < 2:
            raise ConfigurationError(f"n_candidates must be >= 2, got {n_candidates!r}")
        self.deadline_s = float(deadline_s)
        self.n_candidates = int(n_candidates)
        self.threshold_s = float(deadline_s)
        self.sweep_: np.ndarray | None = None
        self.positive_rate_ = 1.0
        self.negative_rate_ = 0.0
        self.fitted_ = False

    def fit(self, x: np.ndarray, y: np.ndarray) -> FixedThresholdPredictor:
        """Choose the threshold by maximising F1 over quantile candidates."""
        xs = np.asarray(x, dtype=np.float64)
        ys = np.asarray(y, dtype=bool).ravel()
        if xs.ndim != 2 or xs.shape[0] != ys.size:
            raise ConfigurationError(
                f"x {xs.shape} and y {ys.shape} do not line up"
            )
        totals = xs[:, _TOTAL_COL]
        qs = np.linspace(0.01, 0.999, self.n_candidates)
        cands = np.unique(np.quantile(totals, qs))
        rows = []
        best = (-1.0, float(self.deadline_s))
        for t in cands:
            f1 = _f1(ys, totals > t)
            rows.append((float(t), f1))
            if f1 > best[0]:
                best = (f1, float(t))
        self.sweep_ = np.asarray(rows, dtype=np.float64)
        self.threshold_s = best[1]
        flagged = totals > self.threshold_s
        self.positive_rate_ = float(ys[flagged].mean()) if np.any(flagged) else 1.0
        self.negative_rate_ = float(ys[~flagged].mean()) if np.any(~flagged) else 0.0
        self.fitted_ = True
        return self

    def _require_fit(self) -> None:
        if not self.fitted_:
            raise RuntimeError("FixedThresholdPredictor.fit must be called first")

    def score(self, x: np.ndarray) -> np.ndarray:
        """Monotone ranking score ``total_last / deadline`` [-]."""
        xs = np.asarray(x, dtype=np.float64)
        return xs[:, _TOTAL_COL] / self.deadline_s

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Boolean flag per row."""
        self._require_fit()
        xs = np.asarray(x, dtype=np.float64)
        return xs[:, _TOTAL_COL] > self.threshold_s

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Two-level probability of an overrun in the horizon [-].

        ``positive_rate_`` where flagged, ``negative_rate_`` where not. These
        are training-set conditional frequencies, so the output is a genuine
        probability estimate rather than a hard 0/1 — but it takes only two
        values, which is exactly the resolution a one-parameter rule has.
        """
        self._require_fit()
        flag = self.predict(x)
        return np.where(flag, self.positive_rate_, self.negative_rate_)

    def calibrate_flag_rate(self, x: np.ndarray, target_rate: float) -> float:
        """Move the threshold so the flag rate on ``x`` equals ``target_rate``.

        A matched-flag-rate operating point is the comparison that matters
        when load shedding has a budget: you can only afford to shed on a
        fixed fraction of iterations, so the question is which predictor
        misses fewest overruns at that fraction. Returns the new threshold [s].

        The achieved rate can **exceed** the target when the score takes few
        distinct values — the Markov baseline's probability output has only
        two levels, one per regime, so it cannot hit an arbitrary rate at all.
        The achieved rate is reported as ``flag_rate`` in the metrics and the
        comparison scripts print both.

        Parameters
        ----------
        x:
            Feature matrix to calibrate on; use the *training* features, not
            the test ones.
        target_rate:
            Desired fraction of flagged rows, in ``(0, 1)``.
        """
        if not (0.0 < target_rate < 1.0):
            raise ConfigurationError(
                f"target_rate must be in (0, 1), got {target_rate!r}"
            )
        totals = np.asarray(x, dtype=np.float64)[:, _TOTAL_COL]
        self.threshold_s = cutoff_for_flag_rate(totals, target_rate)
        self.fitted_ = True
        return self.threshold_s

    def describe(self) -> str:
        """One-line description with the chosen threshold."""
        self._require_fit()
        return (
            f"{self.name}: threshold = {self.threshold_s:.6e} s "
            f"({self.threshold_s / self.deadline_s:.4f} x deadline), "
            f"p(flag) = {self.positive_rate_:.4f}, p(no flag) = {self.negative_rate_:.4f}"
        )


class QueueingOverrunPredictor:
    """Baseline 2: two-state Markov chain with gamma durations per regime.

    See the module docstring for the estimator, the Chapman-Kolmogorov
    propagation and the stated independence approximation.

    Parameters
    ----------
    deadline_s:
        Relative deadline [s], > 0.
    horizon:
        Look-ahead ``H`` in iterations, >= 1; must match the label horizon.
    n_prob_candidates:
        Number of probability cut-offs swept on the training set, >= 2.

    Attributes
    ----------
    regime_threshold_s:
        Otsu threshold separating calm from busy [s].
    transition_:
        ``(2, 2)`` estimated transition matrix.
    fits_:
        Gamma fit per regime.
    prob_cutoff_:
        Chosen probability cut-off [-].
    """

    name = "queueing_markov"

    def __init__(
        self, *, deadline_s: float, horizon: int = 3, n_prob_candidates: int = 199
    ) -> None:
        if not (deadline_s > 0.0):
            raise ConfigurationError(f"deadline_s must be > 0, got {deadline_s!r}")
        if horizon < 1:
            raise ConfigurationError(f"horizon must be >= 1, got {horizon!r}")
        if n_prob_candidates < 2:
            raise ConfigurationError(
                f"n_prob_candidates must be >= 2, got {n_prob_candidates!r}"
            )
        self.deadline_s = float(deadline_s)
        self.horizon = int(horizon)
        self.n_prob_candidates = int(n_prob_candidates)
        self.regime_threshold_s = float(deadline_s)
        self.transition_ = np.eye(2)
        self.fits_: list[_GammaFit] = []
        self.prob_cutoff_ = 0.5
        self.sweep_: np.ndarray | None = None
        self.fitted_ = False

    def fit(self, x: np.ndarray, y: np.ndarray) -> QueueingOverrunPredictor:
        """Fit the regime split, the transition matrix and the gamma laws."""
        xs = np.asarray(x, dtype=np.float64)
        ys = np.asarray(y, dtype=bool).ravel()
        if xs.ndim != 2 or xs.shape[0] != ys.size:
            raise ConfigurationError(f"x {xs.shape} and y {ys.shape} do not line up")
        totals = xs[:, _TOTAL_COL]
        self.regime_threshold_s = _otsu_threshold(totals)
        regime = (totals > self.regime_threshold_s).astype(np.int8)
        counts = np.zeros((2, 2), dtype=np.float64)
        for a, b in itertools.pairwise(regime):
            counts[a, b] += 1.0
        rows = counts.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            trans = np.where(rows > 0, counts / rows, 0.5)
        self.transition_ = trans
        self.fits_ = []
        for z in (0, 1):
            sample = totals[regime == z]
            if sample.size == 0:
                sample = totals
            self.fits_.append(_fit_gamma_mom(sample))
        probs = self._probabilities(totals)
        cands = np.unique(np.quantile(probs, np.linspace(0.001, 0.999, self.n_prob_candidates)))
        rows_out = []
        best = (-1.0, 0.5)
        for c in cands:
            f1 = _f1(ys, probs > c)
            rows_out.append((float(c), f1))
            if f1 > best[0]:
                best = (f1, float(c))
        self.sweep_ = np.asarray(rows_out, dtype=np.float64)
        self.prob_cutoff_ = best[1]
        self.fitted_ = True
        return self

    def _probabilities(self, totals: np.ndarray) -> np.ndarray:
        """``P(overrun within horizon)`` for each current duration."""
        surv = np.array([f.cdf(self.deadline_s) for f in self.fits_], dtype=np.float64)
        regime = (totals > self.regime_threshold_s).astype(np.int8)
        p_no = np.ones(totals.size, dtype=np.float64)
        dist = np.zeros((totals.size, 2), dtype=np.float64)
        dist[np.arange(totals.size), regime] = 1.0
        for _ in range(self.horizon):
            dist = dist @ self.transition_
            p_no *= dist @ surv
        return 1.0 - p_no

    def _require_fit(self) -> None:
        if not self.fitted_:
            raise RuntimeError("QueueingOverrunPredictor.fit must be called first")

    def score(self, x: np.ndarray) -> np.ndarray:
        """Model probability, used as the ranking score [-]."""
        self._require_fit()
        return self.predict_proba(x)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """``P(overrun within horizon)`` per row [-]."""
        self._require_fit()
        xs = np.asarray(x, dtype=np.float64)
        return self._probabilities(xs[:, _TOTAL_COL])

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Boolean flag per row, at the fitted probability cut-off."""
        return self.predict_proba(x) > self.prob_cutoff_

    def calibrate_flag_rate(self, x: np.ndarray, target_rate: float) -> float:
        """Move the probability cut-off to hit ``target_rate`` on ``x``.

        See :meth:`FixedThresholdPredictor.calibrate_flag_rate`. Returns the
        new cut-off [-].
        """
        if not (0.0 < target_rate < 1.0):
            raise ConfigurationError(
                f"target_rate must be in (0, 1), got {target_rate!r}"
            )
        probs = self.predict_proba(x)
        self.prob_cutoff_ = cutoff_for_flag_rate(probs, target_rate)
        return self.prob_cutoff_

    def stationary_distribution(self) -> np.ndarray:
        """Stationary regime distribution of the fitted chain [-].

        The left eigenvector of ``transition_`` for eigenvalue 1, normalised
        to sum to one. For the two-state chain this equals
        ``(P[1,0], P[0,1]) / (P[0,1] + P[1,0])``.
        """
        self._require_fit()
        p01 = self.transition_[0, 1]
        p10 = self.transition_[1, 0]
        denom = p01 + p10
        if denom <= 0.0:
            return np.array([0.5, 0.5])
        return np.array([p10 / denom, p01 / denom])

    def describe(self) -> str:
        """Multi-line description of the fitted model."""
        self._require_fit()
        pi = self.stationary_distribution()
        lines = [
            (
                f"{self.name}: horizon = {self.horizon} iterations, "
                f"cut-off = {self.prob_cutoff_:.4f}"
            ),
            (
                f"  Otsu regime threshold : {self.regime_threshold_s:.6e} s "
                f"({self.regime_threshold_s / self.deadline_s:.4f} x deadline)"
            ),
            (
                f"  transition matrix     : [[{self.transition_[0, 0]:.4f}, "
                f"{self.transition_[0, 1]:.4f}], [{self.transition_[1, 0]:.4f}, "
                f"{self.transition_[1, 1]:.4f}]]"
            ),
            f"  stationary regime     : calm {pi[0]:.4f}, busy {pi[1]:.4f}",
        ]
        for z, f in enumerate(self.fits_):
            lines.append(
                f"  regime {z} gamma        : shape {f.shape:.4f}, scale {f.scale:.6e} s, "
                f"mean {f.mean_s:.6e} s, n {f.n}, P(d > D) {f.sf(self.deadline_s):.4f}"
            )
        return "\n".join(lines)

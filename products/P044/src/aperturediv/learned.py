"""Combiner weighting under imperfect channel-state information.

Why a learned combiner can only be about imperfect CSI
------------------------------------------------------
For unit-norm weights ``w`` and true branch amplitudes ``h``, the
post-combining SNR is proportional to ``(w . h)^2``, and Cauchy-Schwarz
gives ``(w . h)^2 <= ||h||^2`` with equality only at ``w ∝ h``. So
**maximal-ratio combining with the true channel state is optimal, and no
learned combiner can beat it.** Claiming otherwise would be a defect, not a
result. The penalty of any combiner relative to MRC-with-truth,

``penalty_dB = 10 log10( ||h||^2 / (w . h)^2 )``

is therefore non-negative by construction, for every realisation, and is
the metric used throughout this module. It is enforced as a property-based
test, not assumed.

The only honest territory is the regime where the receiver does **not** have
the true state. Four non-learned references are evaluated on the same
held-out rows:

``mrc_true``
    ``w ∝ sqrt(I)``. Unreachable upper bound; penalty identically 0 dB.
``mrc_estimated``
    ``w ∝ sqrt(Ihat)``. What a real receiver does. Degrades as the estimate
    degrades, because it trusts a noisy estimate completely.
``egc``
    ``w = 1/sqrt(L)``. Needs no CSI, so its penalty does not depend on the
    estimation error at all. Once the estimate is bad enough, EGC wins; the
    crossover is measured, not assumed.
``shrinkage(p)``
    ``w ∝ Ihat^(p/2)``, a one-parameter analytic family that contains
    ``mrc_estimated`` at ``p = 1`` and ``egc`` at ``p = 0``. ``p`` is fitted
    per error level on the **validation** split by grid search. This is the
    strongest *analytic* baseline and it exists so that any gain the learned
    model shows can be attributed: if the learned model does not beat
    ``shrinkage``, the finding is that the whole benefit of learning here is
    shrinkage towards equal gain, and the one-line analytic rule is the
    published result.

``LearnedCombiner`` is a free-form regressor
(``sklearn.ensemble.HistGradientBoostingRegressor``, one per aperture) that
predicts the optimal unit weight direction from the estimate alone, plus
three quantile regressors that predict the 10th, 50th and 90th percentile of
the penalty it will incur on that row. The quantile triple is the
uncertainty output: it tells a receiver how much SNR it is likely to be
giving up on this particular symbol, not merely on average.

PyTorch is not available in this environment; the model is scikit-learn.

Nothing here is flight-qualified or certified.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from .ber import bpsk_ber_from_snr
from .datasets import CombinerDataset, build_features

__all__ = [
    "BASELINES",
    "CombinerScore",
    "LearnedCombiner",
    "egc_weights",
    "fit_shrinkage_exponent",
    "mean_ber",
    "mrc_estimated_weights",
    "mrc_true_weights",
    "penalty_db",
    "score_weights",
    "shrinkage_weights",
]

BASELINES: tuple[str, ...] = ("mrc_true", "mrc_estimated", "egc", "shrinkage")


def _unit_rows(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    if np.any(norm <= 0.0):
        raise ValueError("cannot normalise a zero weight vector")
    return x / norm


def mrc_true_weights(irradiance_true: np.ndarray) -> np.ndarray:
    """Unit-norm MRC weights from the true state, ``w ∝ sqrt(I)``."""
    return _unit_rows(np.sqrt(np.asarray(irradiance_true, dtype=float)))


def mrc_estimated_weights(irradiance_estimated: np.ndarray) -> np.ndarray:
    """Unit-norm MRC weights from the estimate, ``w ∝ sqrt(Ihat)``."""
    return _unit_rows(np.sqrt(np.asarray(irradiance_estimated, dtype=float)))


def egc_weights(n_rows: int, n_apertures: int) -> np.ndarray:
    """Equal-gain weights, ``w = 1/sqrt(L)``, shape ``(n_rows, L)``."""
    n, ell = int(n_rows), int(n_apertures)
    if n < 1 or ell < 1:
        raise ValueError("n_rows and n_apertures must be >= 1")
    return np.full((n, ell), 1.0 / np.sqrt(ell))


def shrinkage_weights(irradiance_estimated: np.ndarray, exponent: float) -> np.ndarray:
    """``w ∝ Ihat^(p/2)``: MRC-estimated at ``p = 1``, EGC at ``p = 0``."""
    p = float(exponent)
    if not np.isfinite(p):
        raise ValueError(f"exponent must be finite, got {p!r}")
    i_hat = np.asarray(irradiance_estimated, dtype=float)
    if np.any(i_hat <= 0.0):
        raise ValueError("irradiance_estimated must be strictly positive")
    return _unit_rows(i_hat ** (0.5 * p))


def penalty_db(weights: np.ndarray, amplitude_true: np.ndarray) -> np.ndarray:
    """SNR penalty in dB relative to MRC with the true state.

    ``10 log10( ||h||^2 ||w||^2 / (w . h)^2 )``, non-negative for every row
    by Cauchy-Schwarz. Shape ``(n,)``.
    """
    w = np.asarray(weights, dtype=float)
    h = np.asarray(amplitude_true, dtype=float)
    if w.shape != h.shape:
        raise ValueError("weights and amplitude_true must have the same shape")
    inner = np.sum(w * h, axis=-1)
    if np.any(inner <= 0.0):
        raise ValueError("weights must have a positive projection onto the true amplitude")
    num = np.sum(h * h, axis=-1) * np.sum(w * w, axis=-1)
    return 10.0 * np.log10(num / (inner * inner))


def mean_ber(
    weights: np.ndarray, amplitude_true: np.ndarray, branch_ebn0_db: float
) -> float:
    """Mean BPSK BER over rows at a stated per-branch mean ``Eb/N0``.

    ``BER = mean_rows Q(sqrt(2 gamma_bar (w . h)^2 / ||w||^2))`` with
    ``gamma_bar`` the per-branch mean SNR. Averaging the conditional error
    probability over rows, rather than drawing bits, keeps the comparison
    between combiners free of Monte Carlo noise in the *noise* dimension;
    the channel dimension is still a finite sample of held-out rows.
    """
    w = np.asarray(weights, dtype=float)
    h = np.asarray(amplitude_true, dtype=float)
    gbar = 10.0 ** (float(branch_ebn0_db) / 10.0)
    inner = np.sum(w * h, axis=-1)
    norm2 = np.sum(w * w, axis=-1)
    return float(np.mean(bpsk_ber_from_snr(gbar * inner * inner / norm2)))


@dataclass(frozen=True)
class CombinerScore:
    """Scores of one combiner on one set of held-out rows."""

    name: str
    mean_penalty_db: float
    median_penalty_db: float
    p90_penalty_db: float
    max_penalty_db: float
    mean_ber: float
    n_rows: int


def score_weights(
    name: str,
    weights: np.ndarray,
    amplitude_true: np.ndarray,
    branch_ebn0_db: float,
) -> CombinerScore:
    """Summarise one combiner's penalty distribution and BER."""
    pen = penalty_db(weights, amplitude_true)
    return CombinerScore(
        name=str(name),
        mean_penalty_db=float(pen.mean()),
        median_penalty_db=float(np.median(pen)),
        p90_penalty_db=float(np.quantile(pen, 0.9)),
        max_penalty_db=float(pen.max()),
        mean_ber=mean_ber(weights, amplitude_true, branch_ebn0_db),
        n_rows=int(pen.size),
    )


def fit_shrinkage_exponent(
    validation: CombinerDataset,
    *,
    grid: np.ndarray | None = None,
) -> float:
    """Fit the shrinkage exponent ``p`` on a validation split.

    Grid search minimising the mean penalty in dB. Uses only
    ``irradiance_estimated`` as the combiner input; the truth enters only
    through the objective, exactly as a training label would.
    """
    g = np.linspace(0.0, 1.0, 51) if grid is None else np.asarray(grid, dtype=float)
    h = validation.amplitude_true
    best_p, best_v = float(g[0]), np.inf
    for p in g:
        v = float(penalty_db(shrinkage_weights(validation.irradiance_estimated, p), h).mean())
        if v < best_v:
            best_v, best_p = v, float(p)
    return best_p


@dataclass
class LearnedCombiner:
    """Learned combiner weighting with a per-row penalty-uncertainty output.

    Parameters
    ----------
    max_iter
        Boosting iterations per regressor.
    learning_rate
        Boosting learning rate.
    max_leaf_nodes
        Tree size.
    random_state
        Seed passed to every regressor, so ``fit`` is deterministic.
    quantiles
        Quantiles of the realised penalty to model, in order.
    """

    max_iter: int = 100
    learning_rate: float = 0.12
    max_leaf_nodes: int = 31
    random_state: int = 44044
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    _weight_models: list[HistGradientBoostingRegressor] = field(
        default_factory=list, repr=False
    )
    _quantile_models: list[HistGradientBoostingRegressor] = field(
        default_factory=list, repr=False
    )
    n_apertures: int = 0

    def _new(self, **kw: object) -> HistGradientBoostingRegressor:
        return HistGradientBoostingRegressor(
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_leaf_nodes=self.max_leaf_nodes,
            random_state=self.random_state,
            early_stopping=False,
            **kw,  # type: ignore[arg-type]
        )

    def fit(self, train: CombinerDataset) -> LearnedCombiner:
        """Fit the weight regressors, then the penalty-quantile regressors.

        The quantile models are fitted on the penalties the weight models
        actually incur on the training rows, so the uncertainty output
        describes this combiner rather than a generic one.
        """
        x, y = train.features, train.weights_optimal
        self.n_apertures = train.n_apertures
        self._weight_models = []
        for k in range(self.n_apertures):
            model = self._new()
            model.fit(x, y[:, k])
            self._weight_models.append(model)
        realised = penalty_db(self.predict_weights(x), train.amplitude_true)
        self._quantile_models = [
            self._new(loss="quantile", quantile=q).fit(x, realised) for q in self.quantiles
        ]
        return self

    def _check_fitted(self) -> None:
        if not self._weight_models:
            raise RuntimeError("LearnedCombiner is not fitted; call fit() first")

    def predict_weights(self, features: np.ndarray) -> np.ndarray:
        """Predicted unit-norm weights, shape ``(n, L)``.

        Predictions are clipped to be non-negative before normalising:
        a negative weight on a non-negative amplitude can only reduce
        ``(w . h)``, so it is never useful, and clipping keeps the penalty
        finite on rows where a regressor extrapolates below zero.
        """
        self._check_fitted()
        x = np.asarray(features, dtype=float)
        if x.ndim != 2:
            raise ValueError("features must have shape (n, n_features)")
        raw = np.column_stack([m.predict(x) for m in self._weight_models])
        raw = np.clip(raw, 1e-6, None)
        return _unit_rows(raw)

    def predict_penalty_quantiles(self, features: np.ndarray) -> np.ndarray:
        """Predicted penalty quantiles in dB, shape ``(n, len(quantiles))``.

        The uncertainty output. Column ``j`` is the predicted
        ``quantiles[j]`` of the SNR penalty this combiner will incur on that
        row. Clipped at 0 dB because the penalty cannot be negative.
        Calibration is measured in
        ``validation/validate_learned_combiner.py``: the fraction of
        held-out rows below each predicted quantile should match the nominal
        level.
        """
        self._check_fitted()
        x = np.asarray(features, dtype=float)
        preds = np.column_stack([m.predict(x) for m in self._quantile_models])
        return np.clip(preds, 0.0, None)

    def combine(
        self, irradiance_estimated: np.ndarray, sigma_e: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Weights and penalty quantiles straight from a receiver's inputs.

        Returns ``(weights, penalty_quantiles_db)``. This is the public entry
        point: it takes only quantities a receiver has.
        """
        x = build_features(irradiance_estimated, np.asarray(sigma_e, dtype=float))
        return self.predict_weights(x), self.predict_penalty_quantiles(x)

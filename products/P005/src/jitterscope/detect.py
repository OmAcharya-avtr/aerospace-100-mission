"""Telemetry anomaly detection: band z-score baseline + MLP autoencoder.

Pipeline
--------
1. Telemetry is segmented into fixed-length windows (default 1 s,
   50 % overlap).
2. Each window is reduced to a log-PSD feature vector: Welch PSD
   averaged into ``n_bins`` logarithmically-spaced frequency bins,
   then ``log10``. Log-energy features are standard in vibration
   condition monitoring (Randall 2011, "Vibration-based Condition
   Monitoring", Wiley, ch. 3).
3. Scoring:

   - Baseline (classical, implemented first): per-bin Gaussian z-score
     against nominal statistics; window score = max |z| over bins.
   - ML model: :class:`NominalModel`, an autoencoder-equivalent
     ``sklearn.neural_network.MLPRegressor`` with a bottleneck hidden
     layer trained to reconstruct nominal feature vectors; anomaly
     score = per-window reconstruction MSE (Hinton & Salakhutdinov
     2006, Science 313:504-507 for the autoencoder principle;
     reconstruction-error anomaly scoring per Sakurada & Yairi 2014,
     MLSDA workshop).

4. Threshold: a quantile (default 0.995) of *held-out nominal*
   scores; windows scoring above it are flagged. For the MLP the
   nominal windows are split into a fit set and a calibration set
   (seeded shuffle) because reconstruction error on the fit set
   underestimates error on unseen data (generalization gap); the
   threshold and confidence reference distribution come from the
   calibration set only. Confidence: the empirical nominal CDF value
   of the score, i.e. the fraction of held-out nominal scores below
   the observed score — a calibrated "how abnormal" measure in
   [0, 1], not a classification probability.

Both detectors share the same features and thresholding rule so the
benchmark (validation/val_detector.py) compares models, not plumbing.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from .psd import psd as _psd

__all__ = ["FeatureExtractor", "BandZScoreBaseline", "NominalModel", "DetectionResult", "detect"]


class FeatureExtractor:
    """Windowed log-PSD feature extraction shared by both detectors.

    Parameters
    ----------
    fs : float
        Sample rate [Hz], > 0.
    window_s : float
        Analysis window length [s]; also the anomaly localization
        granularity.
    overlap : float
        Fractional window overlap in [0, 1).
    n_bins : int
        Number of log-spaced frequency bins between ``f_min`` and
        ``fs/2`` (>= 4).
    f_min : float
        Lowest analyzed frequency [Hz], > 0.
    """

    def __init__(
        self,
        fs: float,
        window_s: float = 1.0,
        overlap: float = 0.5,
        n_bins: int = 24,
        f_min: float = 1.0,
    ) -> None:
        if fs <= 0:
            raise ValueError(f"fs must be > 0, got {fs}")
        if window_s * fs < 32:
            raise ValueError("window_s * fs must be >= 32 samples")
        if not 0.0 <= overlap < 1.0:
            raise ValueError(f"overlap must be in [0, 1), got {overlap}")
        if n_bins < 4:
            raise ValueError(f"n_bins must be >= 4, got {n_bins}")
        if not 0.0 < f_min < fs / 2:
            raise ValueError(f"f_min must be in (0, fs/2), got {f_min}")
        self.fs = float(fs)
        self.window_s = float(window_s)
        self.overlap = float(overlap)
        self.n_bins = int(n_bins)
        self.f_min = float(f_min)
        self._edges = np.geomspace(f_min, fs / 2, n_bins + 1)

    def transform(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Segment ``x`` and return (features, window_centers_s).

        Returns
        -------
        features : ndarray, shape (n_windows, n_bins)
            log10 of mean PSD [u^2/Hz] in each frequency bin.
        centers : ndarray, shape (n_windows,)
            Window center times [s] relative to the start of ``x``.
        """
        x = np.asarray(x, dtype=float)
        nwin = int(round(self.window_s * self.fs))
        step = max(1, int(round(nwin * (1.0 - self.overlap))))
        if x.size < nwin:
            raise ValueError(f"telemetry shorter than one window ({x.size} < {nwin} samples)")
        starts = np.arange(0, x.size - nwin + 1, step)
        feats = np.empty((starts.size, self.n_bins))
        for i, s in enumerate(starts):
            f, pxx = _psd(x[s : s + nwin], self.fs, nperseg=min(nwin, 256))
            for b in range(self.n_bins):
                m = (f >= self._edges[b]) & (f < self._edges[b + 1])
                # Floor avoids log10(0) on empty/zero bins.
                feats[i, b] = np.log10(max(float(pxx[m].mean()) if m.any() else 0.0, 1e-30))
        centers = (starts + nwin / 2) / self.fs
        return feats, centers


class BandZScoreBaseline:
    """Classical baseline: per-band log-energy z-score thresholding.

    Fits mean/std of each log-PSD bin on nominal data; the anomaly
    score of a window is ``max_b |(x_b - mu_b) / sd_b|``. This is the
    standard control-chart style limit check used in vibration
    monitoring (Randall 2011, ch. 3; ISO 10816 uses fixed band limits).
    """

    def __init__(self, quantile: float = 0.995) -> None:
        if not 0.5 < quantile < 1.0:
            raise ValueError(f"quantile must be in (0.5, 1), got {quantile}")
        self.quantile = quantile
        self._mu: np.ndarray | None = None
        self._sd: np.ndarray | None = None
        self._train_scores: np.ndarray | None = None
        self.threshold_: float | None = None

    def fit(self, features: np.ndarray) -> "BandZScoreBaseline":
        """Fit nominal per-bin statistics; features shape (n, n_bins)."""
        features = _check_features(features)
        self._mu = features.mean(axis=0)
        self._sd = np.maximum(features.std(axis=0), 1e-12)
        self._train_scores = self.score(features)
        self.threshold_ = float(np.quantile(self._train_scores, self.quantile))
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        """Anomaly score per window: max absolute z over bins."""
        if self._mu is None or self._sd is None:
            raise ValueError("fit() must be called before score()")
        features = _check_features(features)
        return np.max(np.abs((features - self._mu) / self._sd), axis=1)

    def confidence(self, scores: np.ndarray) -> np.ndarray:
        """Empirical nominal-CDF confidence in [0, 1] for each score."""
        if self._train_scores is None:
            raise ValueError("fit() must be called before confidence()")
        return _empirical_cdf(self._train_scores, np.asarray(scores, dtype=float))


class NominalModel:
    """Autoencoder-equivalent nominal-behaviour model (scikit-learn MLP).

    An ``MLPRegressor`` with a bottleneck hidden stack (default
    16-6-16) is trained to reconstruct standardized nominal log-PSD
    feature vectors; anomaly score is per-window reconstruction MSE.
    See module docstring for sources.

    Parameters
    ----------
    hidden : tuple of int
        Encoder-bottleneck-decoder layer widths; the middle layer
        should be narrower than the feature dimension to force
        compression.
    quantile : float
        Held-out nominal-score quantile used as detection threshold.
    seed : int
        Random seed for weight init and the fit/calibration split
        (reproducibility).
    max_iter : int
        Training iteration cap (Adam optimizer).
    calib_frac : float
        Fraction of nominal windows held out for threshold
        calibration, in (0, 0.9]; the MLP never sees these during
        training, avoiding the in-sample generalization gap.
    alpha : float
        L2 regularization strength of the MLP.
    """

    def __init__(
        self,
        hidden: tuple[int, ...] = (16, 6, 16),
        quantile: float = 0.995,
        seed: int = 0,
        max_iter: int = 3000,
        calib_frac: float = 0.3,
        alpha: float = 1e-3,
    ) -> None:
        if not 0.5 < quantile < 1.0:
            raise ValueError(f"quantile must be in (0.5, 1), got {quantile}")
        if not 0.0 < calib_frac <= 0.9:
            raise ValueError(f"calib_frac must be in (0, 0.9], got {calib_frac}")
        self.quantile = quantile
        self.calib_frac = calib_frac
        self._seed = seed
        self._scaler = StandardScaler()
        self._mlp = MLPRegressor(
            hidden_layer_sizes=hidden,
            activation="tanh",
            solver="adam",
            random_state=seed,
            max_iter=max_iter,
            alpha=alpha,
            tol=1e-6,
        )
        self._train_scores: np.ndarray | None = None
        self.threshold_: float | None = None
        self._fitted = False

    def fit(self, features: np.ndarray) -> "NominalModel":
        """Train on nominal feature vectors, shape (n_windows, n_bins).

        Requires at least 20 windows. A seeded shuffle splits them
        into a fit set and a held-out calibration set
        (``calib_frac``); ``threshold_`` is the ``quantile`` of the
        calibration reconstruction errors.
        """
        features = _check_features(features)
        if features.shape[0] < 20:
            raise ValueError(f"need >= 20 nominal windows to fit, got {features.shape[0]}")
        rng = np.random.default_rng(self._seed)
        idx = rng.permutation(features.shape[0])
        n_cal = max(5, int(round(self.calib_frac * features.shape[0])))
        cal_idx, fit_idx = idx[:n_cal], idx[n_cal:]
        z_fit = self._scaler.fit_transform(features[fit_idx])
        with warnings.catch_warnings():
            # Exact tol-convergence of the reconstruction loss is not
            # required: the detection threshold is calibrated post-hoc
            # on held-out scores, so a max_iter stop is acceptable.
            warnings.simplefilter("ignore", ConvergenceWarning)
            self._mlp.fit(z_fit, z_fit)
        self._fitted = True
        self._train_scores = self.score(features[cal_idx])
        self.threshold_ = float(np.quantile(self._train_scores, self.quantile))
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        """Anomaly score per window: reconstruction MSE (standardized units)."""
        if not self._fitted:
            raise ValueError("fit() must be called before score()")
        features = _check_features(features)
        z = self._scaler.transform(features)
        recon = self._mlp.predict(z)
        if recon.ndim == 1:
            recon = recon.reshape(z.shape)
        return np.mean((z - recon) ** 2, axis=1)

    def confidence(self, scores: np.ndarray) -> np.ndarray:
        """Empirical nominal-CDF confidence in [0, 1] for each score.

        1.0 means the score exceeds every nominal training score;
        values near the ``quantile`` are borderline. This is the
        model's required uncertainty output (see MODEL_CARD.md).
        """
        if self._train_scores is None:
            raise ValueError("fit() must be called before confidence()")
        return _empirical_cdf(self._train_scores, np.asarray(scores, dtype=float))


@dataclass
class DetectionResult:
    """Anomaly detection output for a telemetry record.

    Attributes
    ----------
    window_centers_s : window center times [s].
    scores : anomaly score per window (model-specific units).
    threshold : score threshold used.
    flags : boolean, True where score > threshold.
    confidence : empirical nominal-CDF confidence in [0, 1] per window.
    n_anomalous : number of flagged windows.
    """

    window_centers_s: np.ndarray
    scores: np.ndarray
    threshold: float
    flags: np.ndarray
    confidence: np.ndarray
    n_anomalous: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_anomalous = int(np.sum(self.flags))


def detect(
    telemetry: np.ndarray,
    threshold: float | None = None,
    *,
    model: NominalModel | BandZScoreBaseline,
    extractor: FeatureExtractor,
) -> DetectionResult:
    """Score telemetry against a fitted nominal model and flag anomalies.

    Parameters
    ----------
    telemetry : array_like
        1-D telemetry record [signal units]; must be finite (NaN
        raises ``ValueError``, see NaN policy in :mod:`jitterscope.psd`).
    threshold : float, optional
        Score threshold; defaults to the model's fitted
        nominal-quantile ``threshold_``.
    model : NominalModel or BandZScoreBaseline
        A fitted detector.
    extractor : FeatureExtractor
        Must match the extractor used at fit time (same fs/bins).

    Returns
    -------
    DetectionResult
        Per-window scores, flags, and confidence values.
    """
    feats, centers = extractor.transform(np.asarray(telemetry, dtype=float))
    scores = model.score(feats)
    if threshold is None:
        if model.threshold_ is None:
            raise ValueError("model has no fitted threshold_; pass threshold explicitly")
        threshold = model.threshold_
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError(f"threshold must be finite and >= 0, got {threshold}")
    conf = model.confidence(scores)
    return DetectionResult(
        window_centers_s=centers,
        scores=scores,
        threshold=float(threshold),
        flags=scores > threshold,
        confidence=conf,
    )


def _check_features(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=float)
    if features.ndim != 2:
        raise ValueError(f"features must be 2-D (n_windows, n_bins), got shape {features.shape}")
    if not np.all(np.isfinite(features)):
        raise ValueError("features contain non-finite values")
    return features


def _empirical_cdf(train_scores: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Fraction of training scores strictly below each score."""
    sorted_train = np.sort(train_scores)
    ranks = np.searchsorted(sorted_train, scores, side="left")
    return ranks / sorted_train.size

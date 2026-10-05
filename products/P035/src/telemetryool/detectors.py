"""Uniform window-score adapters so every method is compared the same way.

A :class:`WindowDetector` turns a telemetry block of shape
``(n_windows, window_length, n_channels)`` into a score array of shape
``(n_windows, window_length)``, higher meaning more anomalous, plus a debounce
count.  That is the whole interface the comparison harness needs.  Once every
method -- limit check, EWMA, CUSUM, classical multivariate, machine-learned
novelty -- reduces to a scalar score per sample, a single threshold per method
can be calibrated to a common window false-alarm probability, and the methods
are then genuinely at the same operating point.

Scores are chosen so that the calibrated threshold is the detector's own
natural parameter wherever one exists:

===================================  ==========================================
Detector                              Score
===================================  ==========================================
:class:`LimitDetector`                ``max_channels |u_t|``, so the threshold
                                      *is* the limit multiplier ``L``
:class:`EwmaDetector`                 ``max_channels |z_t| / sigma_z``, so the
                                      threshold *is* ``L``
:class:`CusumDetector`                ``max_channels max(C+_t, C-_t)``, so the
                                      threshold *is* ``h``
:class:`NoveltyDetector`              model-specific, no natural units
===================================  ==========================================

Multi-channel handling.  The univariate detectors take the maximum over the
monitored channels, which is the operational arrangement: one alarm bus, any
channel can raise it.  That inflates the false-alarm rate relative to a
single channel, which is exactly why the threshold is calibrated on the
multi-channel score rather than taken from a single-channel table.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from .arl import ewma_sigma_z
from .novelty import GmmNovelty, HotellingT2Q, IsolationForestNovelty, NoveltyModel

__all__ = [
    "WindowDetector",
    "LimitDetector",
    "EwmaDetector",
    "CusumDetector",
    "NoveltyDetector",
    "smoothed_features",
    "build_detector_suite",
]


def _as_block(block: NDArray[np.float64]) -> NDArray[np.float64]:
    arr = np.asarray(block, dtype=float)
    if arr.ndim != 3:
        raise ValueError(
            f"block must be 3-D (n_windows, window_length, n_channels), got {arr.shape}"
        )
    if arr.size == 0:
        raise ValueError("block is empty")
    return arr


class WindowDetector(ABC):
    """A detector reduced to one score per sample per window.

    Parameters
    ----------
    name
        Short identifier used in result tables.
    persistence
        Consecutive exceeding samples required to raise an alarm, >= 1.
    channels
        Channel indices this detector monitors; ``None`` means all of them.
    """

    def __init__(
        self, name: str, persistence: int = 1, channels: Sequence[int] | None = None
    ) -> None:
        if not isinstance(persistence, (int, np.integer)) or persistence < 1:
            raise ValueError(f"persistence must be an integer >= 1, got {persistence!r}")
        self.name = str(name)
        self.persistence = int(persistence)
        self.channels = None if channels is None else list(channels)

    #: True if :meth:`fit` must be called before :meth:`scores`.
    requires_fit: bool = False

    def fit(self, block: NDArray[np.float64]) -> WindowDetector:
        """Fit on a nominal training block; a no-op for the designed charts."""
        return self

    @abstractmethod
    def scores(self, block: NDArray[np.float64]) -> NDArray[np.float64]:
        """Score array of shape ``(n_windows, window_length)``, higher = more novel."""

    def _select(self, block: NDArray[np.float64]) -> NDArray[np.float64]:
        arr = _as_block(block)
        if self.channels is None:
            return arr
        idx = np.asarray(self.channels, dtype=np.int64)
        if idx.min() < 0 or idx.max() >= arr.shape[2]:
            raise ValueError(
                f"detector {self.name!r} monitors channels {self.channels} but the block "
                f"has {arr.shape[2]}"
            )
        return arr[:, :, idx]

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}(name={self.name!r}, persistence={self.persistence})"


class LimitDetector(WindowDetector):
    """Plain two-sided limit check: score is ``max_channels |u_t|`` in sigma units.

    This is the out-of-limit baseline with no temporal integration beyond the
    debounce count.  With ``persistence = 1`` it is a Shewhart chart.
    """

    def __init__(self, persistence: int = 1, channels: Sequence[int] | None = None) -> None:
        super().__init__(f"ool(p={persistence})", persistence, channels)

    def scores(self, block: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.abs(self._select(block)).max(axis=2)


class EwmaDetector(WindowDetector):
    """EWMA chart score ``max_channels |z_t| / sigma_z`` (dimensionless).

    Parameters
    ----------
    lam
        Smoothing weight in (0, 1].
    persistence
        Debounce count.  The analytic designs in :mod:`telemetryool.arl` assume
        1; other values require empirical calibration.
    channels
        Monitored channels.

    Notes
    -----
    ``z`` is reset to 0 at the start of each window, so the first few samples
    of every window are in the EWMA start-up transient and their ``|z|`` is
    systematically small.  The window-level calibration absorbs this because it
    is performed on windows with the same structure; it does mean the score is
    not stationary within a window, and that the detection delay for an anomaly
    at sample 0 differs from one at sample 50.  Detection-delay results here fix
    the onset at a stated index for that reason.
    """

    def __init__(
        self, lam: float = 0.2, persistence: int = 1, channels: Sequence[int] | None = None
    ) -> None:
        if not (0.0 < float(lam) <= 1.0):
            raise ValueError(f"lam must lie in (0, 1], got {lam!r}")
        super().__init__(f"ewma(lam={lam})", persistence, channels)
        self.lam = float(lam)

    def scores(self, block: NDArray[np.float64]) -> NDArray[np.float64]:
        arr = self._select(block)
        lam = self.lam
        out = np.empty(arr.shape[:2])
        prev = np.zeros(arr.shape[0::2])
        for t in range(arr.shape[1]):
            prev = lam * arr[:, t, :] + (1.0 - lam) * prev
            out[:, t] = np.abs(prev).max(axis=1)
        return out / ewma_sigma_z(lam)


class CusumDetector(WindowDetector):
    """Tabular CUSUM score ``max_channels max(C+_t, C-_t)`` in sigma units.

    Parameters
    ----------
    k
        Reference value in sigma units, > 0.  ``k = delta_target / 2`` targets a
        shift of ``delta_target`` sigma.
    persistence
        Debounce count; the analytic designs assume 1.
    channels
        Monitored channels.
    """

    def __init__(
        self, k: float = 0.5, persistence: int = 1, channels: Sequence[int] | None = None
    ) -> None:
        if float(k) <= 0.0:
            raise ValueError(f"k must be > 0, got {k!r}")
        super().__init__(f"cusum(k={k})", persistence, channels)
        self.k = float(k)

    def scores(self, block: NDArray[np.float64]) -> NDArray[np.float64]:
        arr = self._select(block)
        k = self.k
        out = np.empty(arr.shape[:2])
        up = np.zeros(arr.shape[0::2])
        dn = np.zeros(arr.shape[0::2])
        for t in range(arr.shape[1]):
            up = np.maximum(0.0, up + arr[:, t, :] - k)
            dn = np.maximum(0.0, dn - arr[:, t, :] - k)
            out[:, t] = np.maximum(up, dn).max(axis=1)
        return out


def smoothed_features(
    block: NDArray[np.float64], lam: float | None
) -> NDArray[np.float64]:
    """Feature block for a multivariate model: raw channels, optionally plus an EWMA.

    Parameters
    ----------
    block
        Shape ``(n_windows, window_length, n_channels)``, sigma units.
    lam
        EWMA weight in (0, 1]; ``None`` returns the raw channels unchanged.
        When given, the returned block has ``2 * n_channels`` features: the raw
        vector followed by its EWMA, reset to 0 at the start of each window.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_windows, window_length, n_features)``.

    Notes
    -----
    Without the EWMA features a per-sample multivariate model has no temporal
    integration at all and cannot compete with CUSUM on a small sustained
    shift; with them it can.  Including both is the choice that gives the
    machine-learned model its best honest chance, and it is stated here rather
    than buried, because it is a design decision that materially changes the
    comparison.
    """
    arr = _as_block(block)
    if lam is None:
        return arr
    if not (0.0 < float(lam) <= 1.0):
        raise ValueError(f"lam must lie in (0, 1] or be None, got {lam!r}")
    lam_f = float(lam)
    sm = np.empty_like(arr)
    prev = np.zeros(arr.shape[0::2])
    for t in range(arr.shape[1]):
        prev = lam_f * arr[:, t, :] + (1.0 - lam_f) * prev
        sm[:, t, :] = prev
    return np.concatenate([arr, sm / ewma_sigma_z(lam_f)], axis=2)


class NoveltyDetector(WindowDetector):
    """Adapter wrapping a :class:`telemetryool.novelty.NoveltyModel`.

    Parameters
    ----------
    model
        An unfitted novelty model.
    name
        Identifier; defaults to the model's ``name`` plus the smoothing weight.
    persistence
        Debounce count.
    smooth_lam
        EWMA weight for the extra smoothed features (see
        :func:`smoothed_features`); ``None`` uses raw channels only.
    channels
        Monitored channels.
    """

    requires_fit = True

    def __init__(
        self,
        model: NoveltyModel,
        name: str | None = None,
        persistence: int = 1,
        smooth_lam: float | None = 0.2,
        channels: Sequence[int] | None = None,
    ) -> None:
        label = name if name is not None else f"{model.name}(lam={smooth_lam})"
        super().__init__(label, persistence, channels)
        self.model = model
        self.smooth_lam = smooth_lam

    def _features(self, block: NDArray[np.float64]) -> NDArray[np.float64]:
        return smoothed_features(self._select(block), self.smooth_lam)

    def fit(self, block: NDArray[np.float64]) -> NoveltyDetector:
        """Fit the wrapped model on a nominal training block."""
        feats = self._features(block)
        self.model.fit(feats.reshape(-1, feats.shape[2]))
        return self

    def scores(self, block: NDArray[np.float64]) -> NDArray[np.float64]:
        feats = self._features(block)
        n_win, length, n_feat = feats.shape
        return self.model.score(feats.reshape(-1, n_feat)).reshape(n_win, length)

    def pvalues(self, block: NDArray[np.float64], confidence: float = 0.95):
        """Per-sample calibrated p-values for one window, as a list of
        :class:`telemetryool.novelty.PValue`.

        ``block`` must contain exactly one window.
        """
        feats = self._features(block)
        if feats.shape[0] != 1:
            raise ValueError("pvalues() takes a single window")
        return self.model.novelty_pvalue(feats[0], confidence)


def build_detector_suite(
    persistence: int = 1,
    ewma_lam: float = 0.2,
    cusum_k: float = 0.5,
    smooth_lam: float | None = 0.2,
    gmm_components: int = 4,
    iforest_trees: int = 60,
    t2q_components: int | None = None,
    random_state: int = 0,
) -> list[WindowDetector]:
    """The six detectors used by the package's own comparison, in baseline-first order.

    Returns
    -------
    list of WindowDetector
        ``[LimitDetector, EwmaDetector, CusumDetector, NoveltyDetector(t2q),
        NoveltyDetector(gmm), NoveltyDetector(iforest)]``.  The three baselines
        come first by construction: the build guide requires the classical
        baselines to exist and be measured before the learned model.
    """
    return [
        LimitDetector(persistence=persistence),
        EwmaDetector(lam=ewma_lam, persistence=persistence),
        CusumDetector(k=cusum_k, persistence=persistence),
        NoveltyDetector(
            HotellingT2Q(n_components=t2q_components),
            name=f"t2q(lam={smooth_lam})",
            persistence=persistence,
            smooth_lam=smooth_lam,
        ),
        NoveltyDetector(
            GmmNovelty(n_components=gmm_components, random_state=random_state),
            name=f"gmm{gmm_components}(lam={smooth_lam})",
            persistence=persistence,
            smooth_lam=smooth_lam,
        ),
        NoveltyDetector(
            IsolationForestNovelty(n_estimators=iforest_trees, random_state=random_state),
            name=f"iforest{iforest_trees}(lam={smooth_lam})",
            persistence=persistence,
            smooth_lam=smooth_lam,
        ),
    ]

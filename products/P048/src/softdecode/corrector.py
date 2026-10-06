"""A learned LLR corrector for the mismatched-CSI regime, and its competitors.

The problem it is given
-----------------------
A receiver has ``(y, h_hat)`` and does not know how wrong ``h_hat`` is. It has
pilot bits, so it can learn a mapping from observables to a calibrated LLR.
That is the only extra information the learned corrector gets: **pilot bits**.
It is never told the bias, the jitter, the correlation, or the fading
parameters.

The competitors, in the order the specification requires them
-------------------------------------------------------------
1. ``known_csi`` -- exact LLR with the true ``h``. Unreachable upper bound.
2. ``csi_aware_maxlog`` -- max-log with the estimated CSI. The realistic
   baseline named in the specification.
3. ``csi_aware_exact`` -- exact LLR with the estimated CSI, which here means
   the likelihood averaged over ``p(h | h_hat)``. This one is **analytic and
   Bayes-optimal given the observables**, and it is given the error-model
   parameters, which the learned corrector is not.
4. ``plugin`` -- ``h_hat`` substituted into the known-CSI formula. What almost
   every implementation does.
5. ``plugin_scaled`` -- ``alpha * plugin``, the obvious non-learned
   competitor. A single scalar, tuned on a split disjoint from both training
   and reporting.
6. ``plugin_clipped`` -- ``clip(plugin, L_max)``, ``L_max`` tuned on the same
   disjoint split.
7. ``plugin_scaled_clipped`` -- both, tuned jointly on that split.
8. ``learned`` -- this module.

Model
-----
``sklearn.ensemble.RandomForestClassifier`` on six observable features,
predicting ``P(bit = 1 | features)``; the corrected LLR is
``log p0 - log p1``, saturated at a clip level tuned on the tuning split like
every other free parameter. PyTorch is not available in the build container,
so the model is a forest and not a network; the README says so.

Features (all computable at the receiver)
    0. ``y / sigma``
    1. ``h_hat``
    2. ``log h_hat``
    3. ``a * h_hat / sigma``  -- estimated per-bit amplitude SNR
    4. ``plugin LLR``          -- the naive answer, given as a feature so the
       model can only win by adding information the naive answer lacks
    5. ``a / sigma``           -- nominal SNR, which varies across the
       training Eb/N0 points

Uncertainty output
------------------
:meth:`LlrCorrector.predict` returns a per-sample standard deviation over the
forest's trees, converted to LLR units. It is a dispersion measure, not a
calibrated interval; ``validation/validate_corrector.py`` reports its measured
correlation with the actual error so a user can see what it is worth.

Not certified for operational flight use.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .csi import MultiplicativeCsiError, StaleCsiError
from .llr import clip_llr
from .metrics import generalised_mutual_information
from .simulate import OokRealisation, decode_ber, demap_ook

FEATURE_NAMES: tuple[str, ...] = (
    "y_over_sigma",
    "h_hat",
    "log_h_hat",
    "amplitude_snr_hat",
    "plugin_llr",
    "nominal_snr",
)

__all__ = [
    "FEATURE_NAMES",
    "LlrCorrector",
    "TunedParameters",
    "build_features",
    "tune_clip",
    "tune_scale_and_clip",
]


def build_features(realisation: OokRealisation, plugin_llr: np.ndarray) -> np.ndarray:
    """Assemble the ``(N, 6)`` feature matrix for one realisation batch."""
    y = realisation.y.ravel()
    hh = realisation.h_hat.ravel()
    a, s = realisation.amplitude, realisation.sigma
    pl = np.asarray(plugin_llr, dtype=float).ravel()
    if pl.size != y.size:
        raise ValueError(f"plugin_llr has {pl.size} entries, expected {y.size}")
    return np.column_stack(
        [
            y / s,
            hh,
            np.log(hh),
            a * hh / s,
            pl,
            np.full(y.size, a / s),
        ]
    )


@dataclass(frozen=True)
class TunedParameters:
    """Free parameters chosen on the tuning split, with the score that chose them."""

    scale: float
    clip: float
    tuning_ber: float

    def __str__(self) -> str:
        return f"scale {self.scale:.4f}  clip {self.clip:.3f}  tuning BER {self.tuning_ber:.6e}"


class LlrCorrector:
    """Random-forest LLR corrector with an ensemble-dispersion uncertainty.

    Parameters
    ----------
    n_estimators, max_depth, min_samples_leaf, seed:
        Forest hyperparameters. The defaults are sized for the stated compute
        budget (2 shared cores) and train in well under three minutes on
        200000 samples.
    """

    def __init__(
        self,
        n_estimators: int = 40,
        max_depth: int = 14,
        min_samples_leaf: int = 200,
        seed: int = 20261006,
    ) -> None:
        from sklearn.ensemble import RandomForestClassifier

        self.seed = int(seed)
        self.model = RandomForestClassifier(
            n_estimators=int(n_estimators),
            max_depth=int(max_depth),
            min_samples_leaf=int(min_samples_leaf),
            random_state=self.seed,
            n_jobs=1,
        )
        self.clip: float = 12.0
        self._fitted = False

    def fit(self, features: np.ndarray, bits: np.ndarray) -> LlrCorrector:
        """Fit on ``(N, 6)`` features and ``(N,)`` transmitted bits 0/1."""
        x = np.asarray(features, dtype=float)
        b = np.asarray(bits).ravel().astype(np.int8)
        if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES):
            raise ValueError(f"features must be (N, {len(FEATURE_NAMES)}), got {x.shape}")
        if x.shape[0] != b.size:
            raise ValueError(f"features/bits length mismatch: {x.shape[0]} vs {b.size}")
        if np.any((b != 0) & (b != 1)):
            raise ValueError("bits must be 0/1")
        self.model.fit(x, b)
        self._fitted = True
        return self

    @property
    def fitted(self) -> bool:
        """Whether :meth:`fit` has been called."""
        return self._fitted

    def predict(
        self, features: np.ndarray, with_uncertainty: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Corrected LLRs ``log P(0)/P(1)``, optionally with a dispersion.

        The dispersion is the standard deviation of the per-tree LLRs,
        clipped the same way as the mean. It is a confidence signal, not a
        calibrated interval.
        """
        if not self._fitted:
            raise RuntimeError("LlrCorrector.predict called before fit")
        x = np.asarray(features, dtype=float)
        if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES):
            raise ValueError(f"features must be (N, {len(FEATURE_NAMES)}), got {x.shape}")
        if not with_uncertainty:
            p = self.model.predict_proba(x)
            return clip_llr(self._to_llr(p), self.clip)
        per_tree = np.empty((len(self.model.estimators_), x.shape[0]))
        for i, tree in enumerate(self.model.estimators_):
            per_tree[i] = clip_llr(self._to_llr(tree.predict_proba(x)), self.clip)
        return per_tree.mean(axis=0), per_tree.std(axis=0)

    @staticmethod
    def _to_llr(proba: np.ndarray) -> np.ndarray:
        eps = 1e-12
        p0 = np.clip(proba[:, 0], eps, 1.0 - eps)
        p1 = np.clip(proba[:, 1], eps, 1.0 - eps)
        return np.log(p0) - np.log(p1)

    def save(self, path: str) -> None:
        """Persist with joblib. No model binary is shipped in this repository;
        the committed validation scripts retrain deterministically from seeds."""
        import joblib

        joblib.dump({"model": self.model, "clip": self.clip, "seed": self.seed}, path)

    @classmethod
    def load(cls, path: str) -> LlrCorrector:
        """Load a model saved by :meth:`save`."""
        import joblib

        blob = joblib.load(path)
        obj = cls(seed=blob["seed"])
        obj.model = blob["model"]
        obj.clip = float(blob["clip"])
        obj._fitted = True
        return obj


def _ber_of(code, realisation, llr, iterations: int) -> float:
    return decode_ber(code, realisation, llr, iterations=iterations).rate


def tune_scale_and_clip(
    code,
    realisation: OokRealisation,
    fading,
    error: MultiplicativeCsiError | StaleCsiError,
    scales: np.ndarray | None = None,
    clips: np.ndarray | None = None,
    iterations: int = 20,
) -> dict[str, TunedParameters]:
    """Tune the non-learned competitors' free parameters on one split.

    Returns tuned parameters for ``plugin_scaled``, ``plugin_clipped`` and
    ``plugin_scaled_clipped``, each selected by lowest decoded BER on the
    realisation passed in. That realisation must be disjoint from both the
    corrector's training data and the reporting data; the validation script
    enforces this with separate seeds.
    """
    base = demap_ook(realisation, "plugin", fading, error)
    sc = np.asarray(
        [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.4, 2.0]
        if scales is None
        else scales,
        dtype=float,
    )
    cl = np.asarray(
        [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 40.0] if clips is None else clips,
        dtype=float,
    )
    scale_scores = [(_ber_of(code, realisation, base * a, iterations), float(a)) for a in sc]
    best_scale = min(scale_scores)
    clip_scores = [
        (_ber_of(code, realisation, clip_llr(base, c), iterations), float(c)) for c in cl
    ]
    best_clip = min(clip_scores)
    joint = [
        (_ber_of(code, realisation, clip_llr(base * a, c), iterations), float(a), float(c))
        for a in sc
        for c in cl
    ]
    best_joint = min(joint)
    return {
        "plugin_scaled": TunedParameters(best_scale[1], float("inf"), best_scale[0]),
        "plugin_clipped": TunedParameters(1.0, best_clip[1], best_clip[0]),
        "plugin_scaled_clipped": TunedParameters(
            best_joint[1], best_joint[2], best_joint[0]
        ),
    }


def tune_clip(
    code,
    realisation: OokRealisation,
    corrector: LlrCorrector,
    features: np.ndarray,
    clips: np.ndarray | None = None,
    iterations: int = 20,
) -> TunedParameters:
    """Tune the learned corrector's output clip level on the tuning split."""
    cl = np.asarray(
        [2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 40.0] if clips is None else clips, dtype=float
    )
    scores = []
    for c in cl:
        corrector.clip = float(c)
        llr = corrector.predict(features).reshape(realisation.codeword.shape)
        scores.append((_ber_of(code, realisation, llr, iterations), float(c)))
    best = min(scores)
    corrector.clip = best[1]
    return TunedParameters(1.0, best[1], best[0])


def gmi_of(llr: np.ndarray, realisation: OokRealisation) -> float:
    """GMI of an LLR array against the realisation's transmitted channel bits."""
    return generalised_mutual_information(llr, realisation.codeword)

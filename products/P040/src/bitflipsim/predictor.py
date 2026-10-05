"""Learned per-bit criticality predictor, with an ensemble uncertainty output.

The predictor exists because the ground-truth sweep in
:func:`bitflipsim.criticality.sweep_bit_criticality` costs one forward pass per
bit site and cannot be run on orbit or on a large model. It predicts the same
quantity from features that are available without any injection at all.

It is deliberately given both baselines as features (the magnitude ``|w|`` and
the exponent-bit heuristic score), so that it can only beat them by adding
information they do not carry - the input statistics and the downstream
sensitivity of each parameter. If it does not beat them, the honest conclusion
is that there is no such extra information in this problem, and that is the
result reported in the README.

Model
-----
``sklearn.ensemble.RandomForestRegressor`` on ``log1p(degradation)``. A forest
rather than a linear model because the target spans ten orders of magnitude and
the interaction between bit position and parameter magnitude is multiplicative,
and a forest rather than a boosted ensemble because a forest gives a
distribution-free uncertainty output for free:

    mean(x)  = (1/T) sum_t h_t(x)
    sigma(x) = sqrt( (1/T) sum_t (h_t(x) - mean(x))**2 )                   (1)

the dispersion of the individual trees' predictions. This is the standard
ensemble-disagreement confidence signal; it is **not** a calibrated predictive
interval, and :func:`uncertainty_calibration` measures how far off it is rather
than letting the reader assume it is exact.

Split strategy
--------------
Grouped by parameter. Parameter indices are shuffled with a fixed seed and the
first ``train_fraction`` become the training group; every bit of a parameter
goes to the same side, so no bit of a test parameter is ever seen in training.
Targets are measured on the calibration data split and the final metric on the
disjoint evaluation split, so neither the parameters nor the data overlap.

Units: features are a mixture of dimensionless log-magnitudes, bit indices and
mean absolute activations; the target is a dimensionless TV distance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from .bitlayout import FloatLayout, float_layout
from .criticality import exponent_bit_baseline_scores
from .network import MlpParameters

FEATURE_NAMES: tuple[str, ...] = (
    "log2_abs_weight",
    "weight_sign",
    "biased_exponent",
    "bit_position",
    "is_sign_bit",
    "is_exponent_bit",
    "is_mantissa_bit",
    "heuristic_log2_relative",
    "abs_weight",
    "layer_index",
    "is_bias",
    "downstream_weight_l1",
    "upstream_activation_mean_abs",
    "hidden_active_fraction",
)

_TINY = 1e-45


def build_features(
    params: MlpParameters,
    x_calibration: np.ndarray,
    layout: FloatLayout | None = None,
) -> np.ndarray:
    """Per-bit-site feature matrix, shape ``(n_parameters * bits, n_features)``.

    Every feature is computable from the golden parameters and a calibration
    batch alone: no injection is performed, which is the point of the
    predictor.

    Features, in the order of :data:`FEATURE_NAMES`:

    * ``log2_abs_weight``, ``abs_weight``, ``weight_sign`` - the magnitude
      baseline and its sign.
    * ``biased_exponent`` - the stored exponent field ``E`` of the parameter.
    * ``bit_position`` and the three role indicators - where in the word.
    * ``heuristic_log2_relative`` - the exponent-bit baseline score.
    * ``layer_index``, ``is_bias`` - structural position.
    * ``downstream_weight_l1`` - L1 norm of the second-layer weights fed by the
      hidden unit this parameter affects; 1.0 for second-layer parameters.
    * ``upstream_activation_mean_abs`` - mean absolute value of the activation
      this parameter multiplies, measured on the calibration batch; 1.0 for
      biases.
    * ``hidden_active_fraction`` - fraction of calibration samples for which
      the relevant hidden unit is past the ReLU; 1.0 where not applicable.
    """
    lay = layout if layout is not None else float_layout(params.values.dtype)
    plan = params.layout
    bits = lay.total_bits
    values = np.asarray(params.values, dtype=np.float64)
    tensors = params.tensors()
    w2 = np.asarray(tensors["W2"], dtype=np.float64)

    x = np.asarray(x_calibration, dtype=np.float64)
    input_abs_mean = np.abs(x).mean(axis=0)
    hidden = params.hidden_activations(x)
    hidden_abs_mean = np.abs(hidden).mean(axis=0)
    hidden_active = (hidden > 0.0).mean(axis=0)
    downstream_l1 = np.abs(w2).sum(axis=1)

    heuristic = exponent_bit_baseline_scores(1, layout=lay)[0]
    exponent_fields = (params.values.view(lay.uint_dtype).astype(np.int64)
                       >> lay.exponent_lsb) & lay.max_biased_exponent

    rows: list[np.ndarray] = []
    for i in range(plan.size):
        tensor = plan.tensor_of(i)
        local = i - plan.slices[tensor].start
        if tensor == "W1":
            in_idx, hid_idx = divmod(local, plan.n_hidden)
            upstream = float(input_abs_mean[in_idx])
            down = float(downstream_l1[hid_idx])
            active = float(hidden_active[hid_idx])
        elif tensor == "b1":
            hid_idx = local
            upstream = 1.0
            down = float(downstream_l1[hid_idx])
            active = float(hidden_active[hid_idx])
        elif tensor == "W2":
            hid_idx, _out_idx = divmod(local, plan.n_out)
            upstream = float(hidden_abs_mean[hid_idx])
            down = 1.0
            active = float(hidden_active[hid_idx])
        else:  # b2
            upstream = 1.0
            down = 1.0
            active = 1.0
        w = float(values[i])
        abs_w = abs(w)
        log2_abs = float(np.log2(abs_w + _TINY))
        sign = 0.0 if w == 0.0 else float(np.sign(w))
        e_field = float(exponent_fields[i])
        for b in range(bits):
            role = lay.role(b)
            rows.append(
                np.array(
                    [
                        log2_abs,
                        sign,
                        e_field,
                        float(b),
                        1.0 if role == "sign" else 0.0,
                        1.0 if role == "exponent" else 0.0,
                        1.0 if role == "mantissa" else 0.0,
                        float(heuristic[b]),
                        abs_w,
                        float(plan.layer_of(i)),
                        1.0 if plan.is_bias(i) else 0.0,
                        down,
                        upstream,
                        active,
                    ],
                    dtype=np.float64,
                )
            )
    return np.vstack(rows)


@dataclass(frozen=True)
class ParameterSplit:
    """Grouped train/test split over parameter indices."""

    train_parameters: np.ndarray
    test_parameters: np.ndarray
    n_parameters: int
    bits_per_parameter: int
    seed: int

    def site_mask(self, which: str) -> np.ndarray:
        """Boolean mask over flattened bit sites, shape ``(n_params * bits,)``."""
        if which not in ("train", "test"):
            raise ValueError(f"which must be 'train' or 'test', got {which!r}")
        chosen = self.train_parameters if which == "train" else self.test_parameters
        mask = np.zeros((self.n_parameters, self.bits_per_parameter), dtype=bool)
        mask[chosen, :] = True
        return mask.reshape(-1)


def split_parameters(
    n_parameters: int, bits_per_parameter: int, train_fraction: float = 0.6, seed: int = 7
) -> ParameterSplit:
    """Shuffle parameter indices and split them, grouping all bits of a parameter."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")
    if n_parameters < 2:
        raise ValueError(f"need at least 2 parameters to split, got {n_parameters}")
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_parameters)
    cut = max(1, min(n_parameters - 1, int(round(train_fraction * n_parameters))))
    return ParameterSplit(
        train_parameters=np.sort(order[:cut]),
        test_parameters=np.sort(order[cut:]),
        n_parameters=int(n_parameters),
        bits_per_parameter=int(bits_per_parameter),
        seed=int(seed),
    )


@dataclass(frozen=True)
class CriticalityPrediction:
    """Predicted criticality with its ensemble uncertainty.

    Attributes
    ----------
    expected:
        Predicted degradation (TV distance, dimensionless), back-transformed
        from ``log1p`` space with ``expm1``.
    log_mean, log_sigma:
        Equation (1) in the ``log1p`` space the forest was fitted in.
    relative_sigma:
        ``log_sigma / (|log_mean| + eps)``, a dimensionless confidence signal
        that is large where the trees disagree.
    """

    expected: np.ndarray
    log_mean: np.ndarray
    log_sigma: np.ndarray

    @property
    def relative_sigma(self) -> np.ndarray:
        return self.log_sigma / (np.abs(self.log_mean) + 1e-12)


class CriticalityPredictor:
    """Random-forest criticality predictor with an ensemble uncertainty output.

    Parameters
    ----------
    n_estimators:
        Number of trees. 160 on the reference problem; the compute budget is in
        the README.
    max_depth:
        Depth cap, to keep the trees from memorising single sites.
    seed:
        ``random_state`` for the forest.
    """

    def __init__(self, n_estimators: int = 160, max_depth: int = 10, seed: int = 11) -> None:
        if n_estimators < 2:
            raise ValueError(
                f"n_estimators must be >= 2 for an uncertainty output, got {n_estimators}"
            )
        self._forest = RandomForestRegressor(
            n_estimators=int(n_estimators),
            max_depth=int(max_depth),
            min_samples_leaf=4,
            random_state=int(seed),
            n_jobs=1,
        )
        self.n_estimators = int(n_estimators)
        self.seed = int(seed)
        self._fitted = False

    def fit(self, features: np.ndarray, degradation: np.ndarray) -> CriticalityPredictor:
        """Fit on ``log1p(degradation)``.

        Parameters
        ----------
        features:
            Shape ``(n_sites, n_features)`` from :func:`build_features`.
        degradation:
            Shape ``(n_sites,)``, TV distances in ``[0, 1]``.
        """
        f = np.asarray(features, dtype=np.float64)
        d = np.asarray(degradation, dtype=np.float64).ravel()
        if f.ndim != 2 or f.shape[0] != d.size:
            raise ValueError(f"features {f.shape} and degradation {d.shape} disagree")
        if np.any(d < 0.0):
            raise ValueError("degradation must be >= 0")
        self._forest.fit(f, np.log1p(d))
        self._fitted = True
        return self

    def predict(self, features: np.ndarray) -> CriticalityPrediction:
        """Predict with equation (1)'s ensemble mean and standard deviation."""
        if not self._fitted:
            raise ValueError("predictor is not fitted; call fit() first")
        f = np.asarray(features, dtype=np.float64)
        per_tree = np.vstack([tree.predict(f) for tree in self._forest.estimators_])
        log_mean = per_tree.mean(axis=0)
        log_sigma = per_tree.std(axis=0, ddof=0)
        return CriticalityPrediction(
            expected=np.expm1(log_mean),
            log_mean=log_mean,
            log_sigma=log_sigma,
        )

    @property
    def feature_importances(self) -> np.ndarray:
        if not self._fitted:
            raise ValueError("predictor is not fitted; call fit() first")
        return np.asarray(self._forest.feature_importances_, dtype=np.float64)

    def importance_table(self) -> list[tuple[str, float]]:
        """``(feature name, importance)`` sorted descending."""
        pairs = list(zip(FEATURE_NAMES, self.feature_importances, strict=True))
        return sorted(pairs, key=lambda item: -item[1])


def uncertainty_calibration(
    prediction: CriticalityPrediction, degradation: np.ndarray, k: float = 2.0
) -> dict[str, float]:
    """Measure, not assume, how well ``log_sigma`` covers the held-out error.

    Returns the fraction of held-out sites whose true ``log1p`` target lies
    within ``k`` ensemble standard deviations of the ensemble mean, the mean
    absolute error in ``log1p`` space, and the Spearman-free rank agreement
    (Kendall-style concordance is not computed; see ``MODEL_CARD.md``).

    A perfectly calibrated Gaussian would give 0.954 at ``k = 2``. Any
    departure is reported as it is; the forest's tree dispersion is known to
    under-estimate predictive variance, and the number here says by how much on
    this problem.
    """
    d = np.asarray(degradation, dtype=np.float64).ravel()
    target = np.log1p(d)
    if target.shape != prediction.log_mean.shape:
        raise ValueError("prediction and degradation shapes disagree")
    sigma = prediction.log_sigma
    inside = np.abs(target - prediction.log_mean) <= k * sigma
    return {
        "k": float(k),
        "coverage": float(inside.mean()),
        "gaussian_reference": float(0.9544997361036416 if k == 2.0 else np.nan),
        "mean_abs_error_log1p": float(np.abs(target - prediction.log_mean).mean()),
        "median_sigma_log1p": float(np.median(sigma)),
        "n_sites": int(target.size),
    }

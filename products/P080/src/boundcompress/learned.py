"""The learned residual predictor. The AI component, and the thing on trial.

What it is
----------

The prediction step of the same DPCM coder the classical predictors in
:mod:`boundcompress.predictors` drive. It sees exactly what they see -- the
integer symbols the decoder has already reconstructed -- and it emits an
integer prediction. It therefore cannot change a single reconstructed sample
or the error bound; it can only change the number of bytes. That is the point
of putting it here rather than anywhere else in the pipeline: the learned
component is held to a rate comparison at *identical* distortion, with nothing
to trade away.

Scale invariance
----------------

The integer symbols scale as :math:`1/\\Delta`, so a model fitted at one
declared bound would be useless at another unless the features are made
scale-free. For a window of :math:`W` reconstructed symbols ending at
:math:`q_{i-1}`, write the first differences
:math:`d_j = q_{i-j} - q_{i-j-1}` for :math:`j = 1 \\ldots W-1` and set

.. math:: s_i = \\max\\bigl(1,\\ \\mathrm{umed}(|d_1|, \\ldots, |d_{W-1}|)\\bigr)

where :math:`\\mathrm{umed}` is the upper median of the sorted list, an exact
integer function of integers. The feature vector is :math:`d_j / s_i` and the
target is :math:`(q_i - q_{i-1}) / s_i`, so the prediction is

.. math:: \\hat q_i = q_{i-1} + \\operatorname{round}\\bigl(s_i\\, f(d/s_i)\\bigr).

Because :math:`s_i` is computed from already-decoded integers by integer
arithmetic, the decoder reproduces it exactly. This is a local-gain
normalisation of a linear-prediction feature set; the underlying idea is
standard linear predictive coding (Makhoul 1975, "Linear prediction: a
tutorial review", Proceedings of the IEEE 63(4)), with the twist that the
normalisation has to be exactly reproducible by the decoder.

Determinism, and why the forward pass is hand-written
-----------------------------------------------------

The models are fitted with scikit-learn and then their parameters are
**extracted** and the forward pass is evaluated by this module in explicit
elementwise numpy operations, feature by feature and unit by unit. This is not
stylistic. ``sklearn``'s own ``predict`` goes through BLAS, and a BLAS
matrix-vector product and a BLAS matrix-matrix product need not agree in the
last unit in the last place. A one-ulp disagreement between the encoder, which
can evaluate the whole array in one batch, and the decoder, which is
inherently sequential, would flip a ``round`` and desynchronise the coder on
whatever sample happened to land on a half-integer. The hand-written pass is
the same sequence of scalar float64 operations whether it is applied to one
row or to four thousand, so the encoder may vectorise and the decoder may not,
and they still agree bit-for-bit. ``tests/test_learned.py`` asserts that
agreement directly.

Even so, cross-machine bit-exactness is **not** claimed for this predictor:
``numpy``'s ``tanh`` is a libm call. The classical predictors in
:mod:`boundcompress.predictors` are integer-exact and carry no such caveat.
For a coder whose decoder runs on different silicon from its encoder, that
difference is the whole argument, and it is in ``README.md`` under
Limitations.

Uncertainty output
------------------

Every predictor here is an ensemble of ``members`` fits on independent
bootstrap resamples with distinct seeds. The prediction is the member mean;
the dispersion is the member standard deviation, scaled back into the array's
own unit by multiplying by the quantiser step. It is a **dispersion measure
and not a confidence interval** -- see Wager, Hastie & Efron 2014, JMLR 15,
for why an ensemble spread is not coverage -- and its measured usefulness is
reported as a rank correlation against the realised absolute residual in
``validation/validate_learned_output.txt`` rather than asserted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Window of reconstructed symbols the predictor reads. ``WINDOW - 1`` first
#: differences become the feature vector, so the warm-up is ``WINDOW``.
WINDOW = 5

#: Estimator kinds. ``"linear"`` is a fitted linear predictor (ridge
#: regression on the normalised differences, i.e. a learned LPC filter);
#: ``"mlp"`` is a small tanh multilayer perceptron.
KINDS: tuple[str, ...] = ("linear", "mlp")


class LearnedError(ValueError):
    """Raised for an unfitted model, an unknown kind, or a too-short array."""


def upper_median(values: list[int]) -> int:
    """Upper median of a list of integers. Exact integer arithmetic.

    The upper median rather than the mean of the two central values, because
    the mean of two integers is not always an integer and the decoder must
    reproduce this number exactly.
    """
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def features_from_history(history: list[int]) -> tuple[np.ndarray, int]:
    """Feature vector and integer gain for a window ending at ``history[-1]``.

    ``history`` must hold at least :data:`WINDOW` integers; only the last
    :data:`WINDOW` are read.
    """
    window = history[-WINDOW:]
    diffs = [window[index + 1] - window[index] for index in range(WINDOW - 1)]
    diffs.reverse()  # most recent difference first
    gain = max(1, upper_median([abs(value) for value in diffs]))
    return np.asarray(diffs, dtype=np.float64) / gain, gain


def design_matrix(symbols: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised feature matrix, gains and targets for one symbol array.

    Returns ``(features, gains, targets)`` with one row per predictable
    sample, i.e. ``len(symbols) - WINDOW`` rows. The gain is computed by the
    same integer upper-median rule :func:`features_from_history` uses, which
    is asserted equal row by row in ``tests/test_learned.py``.
    """
    integers = np.asarray(symbols, dtype=np.int64)
    if integers.size <= WINDOW:
        raise LearnedError(f"need more than {WINDOW} samples, got {integers.size}")
    differences = np.diff(integers)
    rows = integers.size - WINDOW
    stacked = np.empty((rows, WINDOW - 1), dtype=np.int64)
    for lag in range(WINDOW - 1):
        start = WINDOW - 2 - lag
        stacked[:, lag] = differences[start : start + rows]
    gains = np.maximum(1, np.sort(np.abs(stacked), axis=1)[:, (WINDOW - 1) // 2])
    features = stacked.astype(np.float64) / gains[:, None]
    targets = differences[WINDOW - 1 :].astype(np.float64) / gains
    return features, gains, targets


@dataclass(frozen=True)
class _Member:
    """Extracted parameters of one ensemble member. No scikit-learn at predict time."""

    kind: str
    weights1: np.ndarray
    bias1: np.ndarray
    weights2: np.ndarray
    bias2: float

    def forward(self, features: np.ndarray) -> np.ndarray:
        """Evaluate on a ``(n, WINDOW-1)`` feature matrix. Elementwise, fixed order.

        Written as explicit loops over features and hidden units so that the
        sequence of float64 operations applied to each row is identical
        whether ``n`` is 1 or 4091. See the module docstring.
        """
        matrix = np.atleast_2d(np.asarray(features, dtype=np.float64))
        rows = matrix.shape[0]
        if self.kind == "linear":
            out = np.full(rows, float(self.bias2), dtype=np.float64)
            for index in range(matrix.shape[1]):
                out = out + self.weights2[index] * matrix[:, index]
            return out
        hidden = np.empty((rows, self.bias1.size), dtype=np.float64)
        for unit in range(self.bias1.size):
            accumulator = np.full(rows, float(self.bias1[unit]), dtype=np.float64)
            for index in range(matrix.shape[1]):
                accumulator = accumulator + self.weights1[index, unit] * matrix[:, index]
            hidden[:, unit] = np.tanh(accumulator)
        out = np.full(rows, float(self.bias2), dtype=np.float64)
        for unit in range(self.bias1.size):
            out = out + self.weights2[unit] * hidden[:, unit]
        return out

    @property
    def parameter_count(self) -> int:
        """Number of float64 parameters this member carries."""
        return int(self.weights1.size + self.bias1.size + self.weights2.size + 1)


@dataclass
class LearnedPredictor:
    """Ensemble residual predictor, drop-in for :class:`ClassicalPredictor`.

    Parameters
    ----------
    kind:
        ``"linear"`` or ``"mlp"``; see :data:`KINDS`.
    members:
        Ensemble size. Each member is fitted on an independent bootstrap
        resample with its own seed, which is what produces the dispersion
        output.
    hidden:
        Hidden units, ``"mlp"`` only.
    seed:
        Base seed. Member ``m`` uses ``seed + m``.
    max_iter:
        Optimiser iteration cap, ``"mlp"`` only. The solver is L-BFGS, which
        is deterministic for a fixed seed and converges far faster than Adam
        on a problem this small.
    alpha:
        Ridge penalty (``"linear"``) or L2 penalty (``"mlp"``).
    max_rows:
        Training rows after subsampling. The subsample is taken with a seeded
        generator, so it is reproducible. Caps the fit inside the compute
        budget.
    clip:
        Training features and targets are clipped to ``[-clip, clip]`` before
        fitting. This is robust training, and it is load-bearing rather than
        cosmetic: the normalised target has a heavy tail -- the measured
        absolute maximum over the training set is 4535.0 against a 99.9th
        percentile of 8.0951 -- and an unclipped least-squares fit is
        dominated by a handful of step transients. Measured by
        ``validation/validate_learned.py``, the clip takes the ridge
        coefficients from ``[+0.0001, -0.0002, -0.0000, -0.0002]``, which is
        indistinguishable from predicting nothing, to ``[-0.3224, +0.0659,
        +0.1920, +0.1867]``, and the perceptron training loss at 400
        iterations from 1282.000772 to 0.631258. Clipping is applied to the
        *training* data only; nothing clips a prediction at encode time.
    """

    kind: str = "linear"
    members: int = 3
    hidden: int = 16
    seed: int = 20261024
    max_iter: int = 400
    alpha: float = 1.0
    max_rows: int = 20000
    clip: float = 8.0
    _members: list[_Member] = field(default_factory=list, repr=False)
    _train_rows: int = 0
    _converged: tuple[bool, ...] = ()

    name: str = field(init=False, default="learned")

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise LearnedError(f"kind must be one of {KINDS}, got {self.kind!r}")
        if self.members < 1:
            raise LearnedError(f"need at least one ensemble member, got {self.members}")

    @property
    def order(self) -> int:
        """Warm-up length in samples."""
        return WINDOW

    @property
    def fitted(self) -> bool:
        """Whether :meth:`fit` has been called."""
        return bool(self._members)

    @property
    def train_rows(self) -> int:
        """Number of training rows actually used, after subsampling."""
        return self._train_rows

    @property
    def parameter_count(self) -> int:
        """Total float64 parameters across the ensemble."""
        return sum(member.parameter_count for member in self._members)

    @property
    def model_bytes(self) -> int:
        """Bytes needed to ship the ensemble at float64, excluding any framing.

        This is a fixed cost the decoder must already hold, and it is charged
        in ``MODEL_CARD.md`` against the bytes the predictor saves. It is
        deliberately not folded into the per-array rate, because folding it in
        would make the rate depend on how many arrays the mission sends; the
        break-even array count is reported instead.
        """
        return 8 * self.parameter_count

    @property
    def converged(self) -> tuple[bool, ...]:
        """Per-member convergence flag from the optimiser, ``"mlp"`` only.

        Reported rather than suppressed. A member that hit the iteration cap
        is a real fact about the fit and appears in
        ``validation/validate_learned_output.txt``.
        """
        return self._converged

    def fit(self, arrays: list[np.ndarray]) -> LearnedPredictor:
        """Fit the ensemble on symbol arrays. Returns ``self``.

        Parameters
        ----------
        arrays:
            Integer symbol arrays, as produced by
            :func:`boundcompress.quantise.quantise`. Stacking several bounds
            and several signals is the intended use: the features are
            scale-free, so rows from different bounds are commensurable.

        Raises
        ------
        LearnedError
            If ``arrays`` is empty or yields no usable rows.
        """
        import warnings

        from sklearn.exceptions import ConvergenceWarning
        from sklearn.linear_model import Ridge
        from sklearn.neural_network import MLPRegressor

        if not arrays:
            raise LearnedError("fit needs at least one symbol array")
        blocks = [design_matrix(array) for array in arrays if np.asarray(array).size > WINDOW]
        if not blocks:
            raise LearnedError(f"no array is longer than the {WINDOW}-sample warm-up")
        features = np.vstack([block[0] for block in blocks])
        targets = np.concatenate([block[2] for block in blocks])

        rng = np.random.default_rng(self.seed)
        if features.shape[0] > self.max_rows:
            keep = rng.choice(features.shape[0], size=self.max_rows, replace=False)
            keep.sort()
            features = features[keep]
            targets = targets[keep]
        if self.clip > 0.0:
            features = np.clip(features, -self.clip, self.clip)
            targets = np.clip(targets, -self.clip, self.clip)
        self._train_rows = int(features.shape[0])

        built: list[_Member] = []
        converged: list[bool] = []
        for member in range(self.members):
            member_rng = np.random.default_rng(self.seed + member)
            index = member_rng.integers(0, features.shape[0], size=features.shape[0])
            x_train = features[index]
            y_train = targets[index]
            if self.kind == "linear":
                model = Ridge(alpha=self.alpha, fit_intercept=True)
                model.fit(x_train, y_train)
                built.append(
                    _Member(
                        kind="linear",
                        weights1=np.zeros((0, 0)),
                        bias1=np.zeros(0),
                        weights2=np.asarray(model.coef_, dtype=np.float64).ravel(),
                        bias2=float(model.intercept_),
                    )
                )
                converged.append(True)
            else:
                model = MLPRegressor(
                    hidden_layer_sizes=(self.hidden,),
                    activation="tanh",
                    solver="lbfgs",
                    alpha=self.alpha,
                    max_iter=self.max_iter,
                    random_state=self.seed + member,
                    tol=1e-7,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    model.fit(x_train, y_train)
                built.append(
                    _Member(
                        kind="mlp",
                        weights1=np.asarray(model.coefs_[0], dtype=np.float64),
                        bias1=np.asarray(model.intercepts_[0], dtype=np.float64).ravel(),
                        weights2=np.asarray(model.coefs_[1], dtype=np.float64).ravel(),
                        bias2=float(np.asarray(model.intercepts_[1]).ravel()[0]),
                    )
                )
                converged.append(int(model.n_iter_) < self.max_iter)
        self._members = built
        self._converged = tuple(converged)
        return self

    def _ensemble(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self._members:
            raise LearnedError("predictor is not fitted; call fit() first")
        stacked = np.vstack([member.forward(features) for member in self._members])
        return stacked.mean(axis=0), stacked.std(axis=0)

    def predict_one(self, history: list[int]) -> int:
        """Predict the next integer from ``history``. The decoder's code path."""
        features, gain = features_from_history(history)
        mean, _ = self._ensemble(features[None, :])
        return int(history[-1] + round(float(gain) * float(mean[0])))

    def predict_batch(self, symbols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predictions and dispersions for every predictable sample, vectorised.

        Returns ``(predictions, dispersion)`` in integer-symbol units. The
        predictions are bit-identical to repeated :meth:`predict_one` calls;
        see the module docstring for why that is arranged rather than assumed.
        """
        integers = np.asarray(symbols, dtype=np.int64)
        features, gains, _ = design_matrix(integers)
        mean, spread = self._ensemble(features)
        anchors = integers[WINDOW - 1 : -1]
        predictions = anchors + np.rint(gains.astype(np.float64) * mean).astype(np.int64)
        return predictions, gains.astype(np.float64) * spread

    def residuals(self, symbols: np.ndarray) -> np.ndarray:
        """Residuals after the warm-up, as ``int64``."""
        integers = np.asarray(symbols, dtype=np.int64)
        if integers.size <= WINDOW:
            raise LearnedError(
                f"predictor 'learned' needs more than {WINDOW} samples, got {integers.size}"
            )
        predictions, _ = self.predict_batch(integers)
        return (integers[WINDOW:] - predictions).astype(np.int64)

    def dispersion(self, symbols: np.ndarray) -> np.ndarray:
        """Ensemble dispersion per predictable sample, in integer-symbol units."""
        return self.predict_batch(symbols)[1]

    def reconstruct(self, warmup: list[int], residuals: np.ndarray) -> np.ndarray:
        """Rebuild the symbol stream sequentially. The decoder's code path.

        Raises
        ------
        LearnedError
            If ``warmup`` does not hold exactly :data:`WINDOW` entries.
        """
        if len(warmup) != WINDOW:
            raise LearnedError(
                f"predictor 'learned' needs exactly {WINDOW} warm-up symbols, got "
                f"{len(warmup)}"
            )
        history = [int(value) for value in warmup]
        for residual in np.asarray(residuals, dtype=np.int64).tolist():
            history.append(self.predict_one(history) + int(residual))
        return np.asarray(history, dtype=np.int64)

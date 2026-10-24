"""Closed-loop integer predictors. These change the rate and never the error.

The prediction step of a DPCM (differential pulse-code modulation) coder.
Every predictor here is given only the *reconstructed* integer symbols that
the decoder has already produced, so the decoder reproduces each prediction
exactly and the residual stream is decoded losslessly. That is the structural
reason the error bound in this package is a property of the quantiser alone:
swapping the predictor changes the number of bytes and cannot change a single
reconstructed sample. "Matched distortion" is therefore exact here, not
approximate, and a rate comparison between two predictors is a comparison of
one number.

The classical predictors
------------------------

For integer history :math:`q_{i-1}, q_{i-2}, \\ldots`:

``previous`` (order 1)
    :math:`\\hat q_i = q_{i-1}`. Plain first difference; the predictor of
    CCSDS 121.0-B-3 unit-delay preprocessing and of the ``shuffle``/``delta``
    filters in common array codecs.

``linear`` (order 2)
    :math:`\\hat q_i = 2q_{i-1} - q_{i-2}`. Second-order linear extrapolation,
    exact for any arithmetic progression. This is the baseline the batch
    specification predicts will be hard to beat on telemetry, and it is the
    one to look at first in the results.

``quadratic`` (order 3)
    :math:`\\hat q_i = 3q_{i-1} - 3q_{i-2} + q_{i-3}`. Newton forward
    extrapolation of a degree-2 polynomial through the last three points,
    exact for any quadratic sequence. Noise-amplifying: its coefficients sum
    in absolute value to 7 against 3 for ``linear``, so on a noisy source it
    is expected to lose, and it does.

``cubic`` (order 4)
    :math:`\\hat q_i = 4q_{i-1} - 6q_{i-2} + 4q_{i-3} - q_{i-4}`. Included to
    show the amplification trend continuing rather than to be recommended.

Each coefficient set is the finite-difference extrapolation operator
:math:`\\sum_{j} (-1)^{j} \\binom{k}{j+1} q_{i-1-j}` of order :math:`k`, a
standard result (Hildebrand 1987, "Introduction to Numerical Analysis",
chapter 2). All four are **integer-exact**: the prediction is an integer
linear combination of integers, so the coder is bit-identical on any machine
with the same integer width. The learned predictor in
:mod:`boundcompress.learned` is not, and that difference is reported rather
than glossed.

Warm-up
-------

An order-:math:`k` predictor cannot predict the first :math:`k` samples. They
are written into the container header verbatim as zigzag varints and are
counted in the rate. A short array therefore pays a visible fixed cost for a
long-memory predictor, which is one of the reasons a higher-order predictor
can lose on a short array even when it predicts better.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Predictor identifier written into the container header, by name.
PREDICTOR_IDS: dict[str, int] = {
    "previous": 1,
    "linear": 2,
    "quadratic": 3,
    "cubic": 4,
    "learned": 5,
}
PREDICTOR_NAMES: dict[int, str] = {value: key for key, value in PREDICTOR_IDS.items()}

#: Finite-difference extrapolation coefficients, most recent sample first.
CLASSICAL_COEFFICIENTS: dict[str, tuple[int, ...]] = {
    "previous": (1,),
    "linear": (2, -1),
    "quadratic": (3, -3, 1),
    "cubic": (4, -6, 4, -1),
}

CLASSICAL_NAMES: tuple[str, ...] = tuple(CLASSICAL_COEFFICIENTS)


class PredictorError(ValueError):
    """Raised for an unknown predictor name or an array shorter than the warm-up."""


@dataclass(frozen=True)
class ClassicalPredictor:
    """An integer finite-difference extrapolation predictor.

    Attributes
    ----------
    name:
        One of :data:`CLASSICAL_NAMES`.
    coefficients:
        Integer weights, most recent reconstructed symbol first.
    """

    name: str
    coefficients: tuple[int, ...]

    @property
    def order(self) -> int:
        """Number of warm-up samples this predictor needs."""
        return len(self.coefficients)

    def predict_one(self, history: list[int]) -> int:
        """Predict the next integer from ``history``, most recent entry last.

        Exact integer arithmetic. ``history`` must hold at least
        :attr:`order` entries; only the last :attr:`order` are read.
        """
        total = 0
        for offset, weight in enumerate(self.coefficients):
            total += weight * history[-1 - offset]
        return total

    def residuals(self, symbols: np.ndarray) -> np.ndarray:
        """Residuals of ``symbols`` after the warm-up, as ``int64``.

        Raises
        ------
        PredictorError
            If ``symbols`` is not longer than the warm-up, in which case there
            is nothing to predict and the predictor is not applicable.
        """
        integers = np.asarray(symbols, dtype=np.int64)
        if integers.size <= self.order:
            raise PredictorError(
                f"predictor {self.name!r} needs more than {self.order} samples, got "
                f"{integers.size}"
            )
        history = integers[: self.order].tolist()
        out: list[int] = []
        for index in range(self.order, integers.size):
            out.append(int(integers[index]) - self.predict_one(history))
            history.append(int(integers[index]))
        return np.asarray(out, dtype=np.int64)

    def reconstruct(self, warmup: list[int], residuals: np.ndarray) -> np.ndarray:
        """Rebuild the symbol stream from the warm-up and the residuals.

        Raises
        ------
        PredictorError
            If ``warmup`` does not hold exactly :attr:`order` entries.
        """
        if len(warmup) != self.order:
            raise PredictorError(
                f"predictor {self.name!r} needs exactly {self.order} warm-up symbols, "
                f"got {len(warmup)}"
            )
        history = [int(value) for value in warmup]
        for residual in np.asarray(residuals, dtype=np.int64).tolist():
            history.append(self.predict_one(history) + int(residual))
        return np.asarray(history, dtype=np.int64)


def get_predictor(name: str) -> ClassicalPredictor:
    """Return the classical predictor called ``name``.

    Raises
    ------
    PredictorError
        If ``name`` is not one of :data:`CLASSICAL_NAMES`. ``"learned"`` is
        built by :mod:`boundcompress.learned` and is not available here,
        deliberately: the classical path has no dependency on scikit-learn.
    """
    if name not in CLASSICAL_COEFFICIENTS:
        raise PredictorError(
            f"unknown predictor {name!r}; classical predictors are {CLASSICAL_NAMES} "
            "and 'learned' is constructed by boundcompress.learned"
        )
    return ClassicalPredictor(name, CLASSICAL_COEFFICIENTS[name])


def noise_gain(name: str) -> int:
    """Sum of the absolute coefficients: how much white noise a predictor amplifies.

    1 for ``previous``, 3 for ``linear``, 7 for ``quadratic``, 15 for
    ``cubic``. A standard first-order account of why a higher-order
    extrapolator loses on a noisy source: if the history carries independent
    errors of scale :math:`\\sigma`, the prediction error scales with the
    coefficient norm. Reported in
    ``validation/validate_predictors_output.txt`` beside the measured rates.
    """
    return sum(abs(weight) for weight in CLASSICAL_COEFFICIENTS[name])

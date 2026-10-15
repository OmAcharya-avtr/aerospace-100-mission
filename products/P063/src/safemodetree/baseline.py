"""The declared threshold-logic baseline. Written and measured before any model.

This is what onboard fault detection, isolation and recovery logic normally is:
a per-channel threshold, a latched symptom bit, and a table lookup from the
symptom word to a fault. It is implemented here first so that the learned
classifier has something to be benchmarked against, as required by the mission
rule that the analytic baseline precedes the model.

Decision rule, stated in full because it is declared and not fitted:

1. Each channel's symptom bit is ``1`` when the measurement is strictly beyond
   the declared threshold in the declared direction, by at least
   ``margin_sd`` multiples of the channel's declared ``noise_sd``. The
   declared operating point is ``margin_sd = 0``; the parameter exists only so
   the baseline has a tunable operating point and can be compared at an equal
   false-alarm rate.
2. An all-zero symptom word is the no-fault class.
3. A symptom word equal to exactly one declared signature identifies that
   fault, with confidence 1.
4. A symptom word equal to the shared signature of a declared ambiguity class
   of size ``m`` returns the lexicographically first member, with confidence
   ``1/m``.
5. Any other symptom word is resolved to the declared signature(s) at minimum
   Hamming distance ``d``, returning the lexicographically first of the ``m``
   tied faults with confidence ``1 / (m * (1 + d))``.

Rule 5 is a DECLARED heuristic, not a probability estimate. It is the honest
version of what threshold logic can offer as a confidence, and its calibration
error is one of the things this package measures rather than assumes. Units:
confidences are dimensionless and in (0, 1]; ``margin_sd`` is dimensionless.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import NOMINAL_LABEL
from .spec import SafeModeSpec

__all__ = ["ThresholdLogicClassifier", "symptom_word"]


def symptom_word(
    spec: SafeModeSpec, observation: np.ndarray, margin_sd: float = 0.0
) -> tuple[int, ...]:
    """Latched symptom word for one observation vector.

    Args:
        spec: the declaration supplying channels, thresholds and directions.
        observation: one observation vector, one entry per declared channel, in
            each channel's declared unit.
        margin_sd: extra margin beyond the threshold, in multiples of the
            channel's declared ``noise_sd``. Dimensionless.

    Returns:
        A tuple of 0/1, one per channel, in observation-vector order.

    The comparison is a strict threshold comparison on a floating-point
    quantity. A measurement that lands on the threshold does not assert, and
    the resulting boolean is not invariant under an algebraically equivalent
    rescaling of the channel. This is stated rather than hidden; see README
    Limitations.
    """
    channels = spec.observability.channels
    if observation.shape[-1] != len(channels):
        raise ValueError(
            f"observation has {observation.shape[-1]} entries, expected {len(channels)}"
        )
    return tuple(
        int(channel.margin(float(value)) > margin_sd)
        for channel, value in zip(channels, observation, strict=True)
    )


@dataclass
class ThresholdLogicClassifier:
    """Declared threshold logic over the observation vector.

    Attributes:
        spec: the declaration.
        margin_sd: operating point, in multiples of each channel's declared
            ``noise_sd``. 0 is the declared operating point. Raising it lowers
            the false-alarm rate and raises the missed-detection rate.
    """

    spec: SafeModeSpec
    margin_sd: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.margin_sd, bool) or not isinstance(self.margin_sd, int | float):
            raise TypeError("margin_sd must be a real number")
        self.margin_sd = float(self.margin_sd)
        self._table: dict[tuple[int, ...], tuple[str, ...]] = {}
        for fault_id in self.spec.fault_ids:
            signature = self.spec.observability.signature(fault_id)
            self._table.setdefault(signature, ())
            self._table[signature] = tuple(sorted({*self._table[signature], fault_id}))
        self._signatures = tuple(sorted(self._table))
        self._zero = (0,) * len(self.spec.observability.channels)

    @property
    def classes(self) -> tuple[str, ...]:
        """Labels this classifier can emit, nominal first."""
        return (NOMINAL_LABEL, *self.spec.fault_ids)

    def classify_word(self, word: tuple[int, ...]) -> tuple[str, float]:
        """Resolve one symptom word to ``(label, confidence)``.

        Confidence is dimensionless in (0, 1]; the resolution rules are the
        five listed in the module docstring.
        """
        if word == self._zero:
            return NOMINAL_LABEL, 1.0
        exact = self._table.get(word)
        if exact is not None:
            return exact[0], 1.0 / len(exact)
        distances = [
            (sum(a != b for a, b in zip(word, signature, strict=True)), signature)
            for signature in self._signatures
        ]
        best = min(d for d, _ in distances)
        tied: list[str] = []
        for distance, signature in distances:
            if distance == best:
                tied.extend(self._table[signature])
        tied.sort()
        return tied[0], 1.0 / (len(tied) * (1 + best))

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predicted label per row. Returns an object array of shape ``(n,)``."""
        features = np.atleast_2d(np.asarray(features, dtype=float))
        return np.array(
            [
                self.classify_word(symptom_word(self.spec, row, self.margin_sd))[0]
                for row in features
            ],
            dtype=object,
        )

    def confidence(self, features: np.ndarray) -> np.ndarray:
        """Declared confidence per row, dimensionless in (0, 1]."""
        features = np.atleast_2d(np.asarray(features, dtype=float))
        return np.array(
            [
                self.classify_word(symptom_word(self.spec, row, self.margin_sd))[1]
                for row in features
            ],
            dtype=float,
        )

    def predict_with_confidence(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Labels and confidences in one pass over the rows."""
        features = np.atleast_2d(np.asarray(features, dtype=float))
        resolved = [
            self.classify_word(symptom_word(self.spec, row, self.margin_sd)) for row in features
        ]
        return (
            np.array([label for label, _ in resolved], dtype=object),
            np.array([conf for _, conf in resolved], dtype=float),
        )

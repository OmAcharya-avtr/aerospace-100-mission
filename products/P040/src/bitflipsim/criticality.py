"""Per-bit criticality: ground-truth sweep, two baselines, and the protection metric.

Degradation metric
------------------
The primary metric is the mean total-variation distance between the golden and
the faulty output distribution,

    D = (1/n) * sum_i TV(p_i, p'_i),    TV(p, q) = 0.5 * sum_c |p_c - q_c|   (1)

which is bounded in ``[0, 1]``, is defined for every pair of distributions, and
does not saturate on a small evaluation split the way accuracy does. The
convention for a faulty output that is not a distribution at all (a NaN row,
produced when an exponent-bit upset sends a logit to NaN) is ``TV = 1``: the
maximum. That is stated here and in ``MODEL_CARD.md`` rather than left to
whatever NaN propagation does.

Accuracy drop is reported alongside as a secondary metric, with class ``-1``
(the NaN convention of :meth:`bitflipsim.network.MlpParameters.predict`)
counting as wrong.

Baselines, implemented before the learned model
-----------------------------------------------
Both baselines score a *bit site* ``(parameter i, bit b)`` without running a
single injection.

1. **Magnitude baseline** - ``score = |w_i|``, identical for every bit of a
   parameter. The reasoning is that an upset perturbs a large weight by more in
   absolute terms, so large weights matter more. It has no bit-position
   resolution at all.

2. **Exponent-bit heuristic** - ``score`` is the base-2 logarithm of the
   worst-case *relative* perturbation that flipping bit ``b`` can cause,
   derived from the IEEE 754 layout in :mod:`bitflipsim.bitlayout` and nothing
   else:

       exponent bit at position b : log2 ratio = 2**(b - nmant)
       sign bit                   : log2 ratio = 1        (|dw| = 2|w|)
       mantissa bit at position b : log2 ratio = b - nmant   (negative)

   so for float32 the ordering is bit 30 (2**7 = 128), 29 (64), ... 23 (1),
   then the sign bit at 1, then mantissa bits 22 down to 0 at -1 down to -23.
   It has no parameter resolution at all.

The learned predictor in :mod:`bitflipsim.predictor` is given both of these as
features, so it starts from the baselines rather than competing with them from
scratch.

Protection metric: degradation avoided per protected byte
---------------------------------------------------------
Every bit site is equally likely to be upset (see :mod:`bitflipsim.injection`),
so protecting a set ``S`` of sites avoids an expected degradation proportional
to ``sum_{s in S} D(s)``. Normalising by the storage cost of protecting ``S``
gives the metric the specification names:

    avoided per byte = sum_{s in S} D(s) / cost_bytes(S)                   (2)

Two cost models, because the honest answer depends on the granularity the
hardware can actually protect:

* ``"bit"``   - ``cost = |S| / 8`` bytes. Idealised: a redundant copy of
  exactly the protected bits. No real ECC or TMR scheme is this fine-grained;
  this is a lower bound on cost and an upper bound on any method's score.
* ``"word"``  - protecting any bit of a parameter costs the whole parameter,
  ``cost = (number of distinct parameters touched) * itemsize`` bytes. This is
  what word-level triple-modular-redundancy or a SECDED code over the word
  actually charges.

The denominator of equation (2) is the cost actually consumed, not the budget
requested, so a method is never credited with bytes it did not spend.

Tie handling is exact rather than arbitrary. A method that cannot distinguish
between sites (the exponent heuristic cannot distinguish parameters; the
magnitude baseline cannot distinguish bits) produces large ties, and which
member of a tie gets protected would otherwise depend on sort order. The
functions here compute the **expectation over uniformly random tie-breaking**:
all strictly-better sites are taken whole, and the remaining budget takes the
mean degradation of the tie band. That makes the reported number a property of
the method, not of ``numpy.argsort``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .bitlayout import FloatLayout, float_layout
from .injection import BitUpset, apply_upsets
from .network import Int8Quantization, MlpParameters

CostModel = Literal["bit", "word"]


def total_variation(golden: np.ndarray, faulty: np.ndarray) -> np.ndarray:
    """Row-wise total-variation distance, shape ``(n_samples,)``, in ``[0, 1]``.

    ``TV(p, q) = 0.5 * sum_c |p_c - q_c|``. Rows of ``faulty`` containing a NaN
    are assigned 1.0 by the convention stated in the module docstring.
    """
    g = np.asarray(golden, dtype=np.float64)
    f = np.asarray(faulty, dtype=np.float64)
    if g.shape != f.shape or g.ndim != 2:
        raise ValueError(f"golden and faulty must share a 2-D shape, got {g.shape} and {f.shape}")
    bad = ~np.isfinite(f).all(axis=1)
    with np.errstate(invalid="ignore"):
        tv = 0.5 * np.abs(g - f).sum(axis=1)
    tv[bad] = 1.0
    return np.clip(tv, 0.0, 1.0)


def mean_total_variation(golden: np.ndarray, faulty: np.ndarray) -> float:
    """Equation (1): mean TV distance over the evaluation split, dimensionless."""
    return float(total_variation(golden, faulty).mean())


def accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    """Fraction correct; class ``-1`` (non-finite output) always counts wrong."""
    p = np.asarray(predictions).ravel()
    y = np.asarray(labels).ravel()
    if p.shape != y.shape:
        raise ValueError(f"shape mismatch {p.shape} vs {y.shape}")
    return float((p == y).mean())


@dataclass(frozen=True)
class CriticalitySweep:
    """Ground-truth degradation of every bit site, measured by brute force.

    Attributes
    ----------
    degradation:
        Shape ``(n_parameters, bits_per_parameter)``. Equation (1) for a single
        upset at that site, measured on the split given to
        :func:`sweep_bit_criticality`.
    accuracy_drop:
        Same shape: golden accuracy minus faulty accuracy, dimensionless.
    golden_accuracy:
        Accuracy with no upset, dimensionless.
    n_evaluations:
        Number of forward passes performed, ``n_parameters * bits_per_parameter``.
    """

    degradation: np.ndarray
    accuracy_drop: np.ndarray
    golden_accuracy: float
    n_evaluations: int

    @property
    def n_parameters(self) -> int:
        return int(self.degradation.shape[0])

    @property
    def bits_per_parameter(self) -> int:
        return int(self.degradation.shape[1])

    def flat_degradation(self) -> np.ndarray:
        """Row-major flattening, index ``i * bits + b``."""
        return self.degradation.reshape(-1)

    def by_bit_position(self) -> np.ndarray:
        """Mean degradation per bit position, shape ``(bits_per_parameter,)``."""
        return self.degradation.mean(axis=0)

    def by_parameter(self) -> np.ndarray:
        """Mean degradation per parameter, shape ``(n_parameters,)``."""
        return self.degradation.mean(axis=1)


def sweep_bit_criticality(
    params: MlpParameters,
    x: np.ndarray,
    y: np.ndarray,
) -> CriticalitySweep:
    """Flip every bit of every parameter once and measure the degradation.

    This is the ground truth the baselines and the learned predictor are scored
    against. It costs ``n_parameters * bits_per_parameter`` forward passes
    (4704 for the 147-parameter float32 reference model), which is the whole
    reason a *predictor* is interesting: the sweep is what you cannot afford on
    orbit or for a large model.

    Parameters
    ----------
    params:
        The golden parameter block.
    x, y:
        Evaluation features ``(n, n_in)`` and integer labels ``(n,)``.

    Returns
    -------
    CriticalitySweep
    """
    bits = params.layout.bits_per_parameter
    n = params.layout.size
    golden_probs = params.probabilities(x)
    golden_acc = accuracy(params.predict(x), y)
    degradation = np.zeros((n, bits), dtype=np.float64)
    drop = np.zeros((n, bits), dtype=np.float64)
    for i in range(n):
        for b in range(bits):
            faulty = params.with_values(apply_upsets(params.values, [BitUpset(i, b)]))
            probs = faulty.probabilities(x)
            degradation[i, b] = mean_total_variation(golden_probs, probs)
            drop[i, b] = golden_acc - accuracy(faulty.predict(x), y)
    return CriticalitySweep(
        degradation=degradation,
        accuracy_drop=drop,
        golden_accuracy=golden_acc,
        n_evaluations=n * bits,
    )


def sweep_quantized_bit_criticality(
    quantization: Int8Quantization,
    x: np.ndarray,
    y: np.ndarray,
) -> CriticalitySweep:
    """The same brute-force sweep for the int8 quantized parameter block.

    The upset is applied to the int8 *code*, then the block is dequantized and
    run, which is the realistic path for a model stored quantized in memory.
    Two's-complement int8 has 8 bit sites per parameter instead of 32, and bit
    7 is the sign bit: flipping it changes the code by -128 or +128 and so
    changes the dequantized weight by ``+-128 * scale``. That is a bounded
    perturbation, unlike the float32 exponent-MSB case, which is the headline
    difference between the two storage formats.
    """
    golden_params = quantization.as_parameters()
    golden_probs = golden_params.probabilities(x)
    golden_acc = accuracy(golden_params.predict(x), y)
    n = quantization.layout.size
    bits = quantization.layout.bits_per_parameter
    degradation = np.zeros((n, bits), dtype=np.float64)
    drop = np.zeros((n, bits), dtype=np.float64)
    for i in range(n):
        for b in range(bits):
            faulty_codes = apply_upsets(quantization.codes, [BitUpset(i, b)])
            faulty = Int8Quantization(
                codes=faulty_codes, scales=quantization.scales, layout=quantization.layout
            ).as_parameters()
            degradation[i, b] = mean_total_variation(golden_probs, faulty.probabilities(x))
            drop[i, b] = golden_acc - accuracy(faulty.predict(x), y)
    return CriticalitySweep(
        degradation=degradation,
        accuracy_drop=drop,
        golden_accuracy=golden_acc,
        n_evaluations=n * bits,
    )


def magnitude_baseline_scores(params: MlpParameters) -> np.ndarray:
    """Baseline 1: ``|w_i|``, broadcast across bits. Shape ``(n_params, bits)``."""
    bits = params.layout.bits_per_parameter
    magnitude = np.abs(np.asarray(params.values, dtype=np.float64))
    return np.repeat(magnitude[:, None], bits, axis=1)


def exponent_bit_baseline_scores(
    n_parameters: int, layout: FloatLayout | None = None, dtype: object = np.float32
) -> np.ndarray:
    """Baseline 2: log2 of the worst-case relative perturbation per bit position.

    Derived from the IEEE 754 layout only, with no reference to the parameter
    values. Shape ``(n_parameters, total_bits)``; every row is identical.
    """
    lay = layout if layout is not None else float_layout(dtype)
    per_bit = np.empty(lay.total_bits, dtype=np.float64)
    for b in range(lay.total_bits):
        role = lay.role(b)
        if role == "exponent":
            per_bit[b] = float(lay.exponent_weight(b))
        elif role == "sign":
            per_bit[b] = 1.0
        else:
            per_bit[b] = float(b - lay.mantissa_bits)
    if n_parameters <= 0:
        raise ValueError(f"n_parameters must be > 0, got {n_parameters}")
    return np.repeat(per_bit[None, :], n_parameters, axis=0)


def _expected_top_k_sum(scores: np.ndarray, values: np.ndarray, k: int) -> float:
    """Expected sum of ``values`` over the top-``k`` ``scores``, ties averaged.

    Sorts by score descending. Sites strictly above the cut contribute in full;
    the tie band at the cut contributes ``(remaining slots / band size) * band
    sum``, which is the expectation under uniformly random tie-breaking.
    """
    s = np.asarray(scores, dtype=np.float64).ravel()
    v = np.asarray(values, dtype=np.float64).ravel()
    if s.shape != v.shape:
        raise ValueError(f"scores and values must share shape, got {s.shape} and {v.shape}")
    k = int(k)
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    if k == 0:
        return 0.0
    if k >= s.size:
        return float(v.sum())
    order = np.argsort(-s, kind="stable")
    s_sorted = s[order]
    v_sorted = v[order]
    threshold = s_sorted[k - 1]
    above = s_sorted > threshold
    band = s_sorted == threshold
    taken_above = int(above.sum())
    remaining = k - taken_above
    band_size = int(band.sum())
    total = float(v_sorted[above].sum())
    if band_size > 0 and remaining > 0:
        total += float(v_sorted[band].sum()) * (remaining / band_size)
    return total


@dataclass(frozen=True)
class ProtectionResult:
    """Degradation avoided for one method at one budget, under one cost model."""

    method: str
    cost_model: str
    budget_bytes: float
    cost_bytes: float
    protected_units: int
    avoided: float
    total_available: float
    avoided_per_byte: float

    @property
    def avoided_fraction(self) -> float:
        """Avoided divided by the total degradation of all sites, dimensionless."""
        if self.total_available == 0.0:
            return 0.0
        return self.avoided / self.total_available


def evaluate_protection(
    scores: np.ndarray,
    degradation: np.ndarray,
    budget_bytes: float,
    itemsize_bytes: int,
    cost_model: CostModel = "bit",
    method: str = "",
) -> ProtectionResult:
    """Equation (2): expected degradation avoided per protected byte.

    Parameters
    ----------
    scores:
        Method ranking, shape ``(n_parameters, bits_per_parameter)``. Higher is
        "protect this first". Ties are averaged exactly (see module docstring).
    degradation:
        Ground-truth degradation of the same shape, measured on the evaluation
        split.
    budget_bytes:
        Protection storage budget in bytes.
    itemsize_bytes:
        Storage width of one parameter, bytes. Used by the ``"word"`` model.
    cost_model:
        ``"bit"`` or ``"word"``; see the module docstring.
    method:
        Label carried through to the result.

    Returns
    -------
    ProtectionResult
    """
    sc = np.asarray(scores, dtype=np.float64)
    dg = np.asarray(degradation, dtype=np.float64)
    if sc.shape != dg.shape or sc.ndim != 2:
        raise ValueError(
            "scores and degradation must share a 2-D shape, got "
            f"{sc.shape} and {dg.shape}"
        )
    if budget_bytes < 0.0:
        raise ValueError(f"budget_bytes must be >= 0, got {budget_bytes}")
    if itemsize_bytes <= 0:
        raise ValueError(f"itemsize_bytes must be > 0, got {itemsize_bytes}")
    total = float(dg.sum())

    if cost_model == "bit":
        k = min(int(np.floor(float(budget_bytes) * 8.0)), sc.size)
        avoided = _expected_top_k_sum(sc, dg, k)
        units = k
        cost = k / 8.0
    elif cost_model == "word":
        # Protecting any bit of a parameter costs the whole word, so a method's
        # word score is the sum of the scores it gives that word's bits, and
        # protecting a word avoids the degradation of all of its bits. For the
        # exponent-bit heuristic that sum is the same for every word, which is
        # the honest statement that the heuristic carries no word-level
        # information at all.
        word_scores = sc.sum(axis=1)
        word_values = dg.sum(axis=1)
        k = min(int(np.floor(float(budget_bytes) / itemsize_bytes)), word_scores.size)
        avoided = _expected_top_k_sum(word_scores, word_values, k)
        units = k
        cost = float(k * itemsize_bytes)
    else:
        raise ValueError(f"cost_model must be 'bit' or 'word', got {cost_model!r}")

    per_byte = avoided / cost if cost > 0.0 else 0.0
    return ProtectionResult(
        method=method,
        cost_model=cost_model,
        budget_bytes=float(budget_bytes),
        cost_bytes=cost,
        protected_units=units,
        avoided=avoided,
        total_available=total,
        avoided_per_byte=per_byte,
    )


def oracle_scores(degradation: np.ndarray) -> np.ndarray:
    """The unachievable upper bound: rank by the measured degradation itself.

    Included so that every method's score can be read as a fraction of what
    perfect knowledge would achieve, rather than only relative to each other.
    """
    return np.asarray(degradation, dtype=np.float64).copy()


def random_scores(shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """A random ranking, as the lower reference point."""
    return rng.random(size=shape)

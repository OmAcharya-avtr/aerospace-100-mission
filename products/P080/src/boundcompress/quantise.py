"""Bound-implied quantisation: turn a declared error bound into a quantiser step.

The whole error guarantee of this package lives in this module. Everything
downstream of it -- prediction, entropy coding, container packing -- is
lossless on the integer symbol stream, so the reconstruction error of the
codec is *exactly* the reconstruction error of the quantiser defined here and
of nothing else. That is a deliberate architectural choice: it makes the
distortion of two different predictors identical by construction rather than
by measurement, which is what "rate-distortion comparison at matched
distortion" has to mean if it is to mean anything.

Three declared bound modes
--------------------------

``"abs"``
    Declared absolute bound :math:`b`. Uniform scalar quantiser with
    round-to-nearest and step :math:`\\Delta = 2b`:

    .. math:: q_i = \\operatorname{round}(x_i / \\Delta), \\qquad
              \\hat{x}_i = q_i \\Delta

    In exact arithmetic :math:`|x_i - \\hat{x}_i| \\le \\Delta/2 = b`. This is
    the textbook deadzone-free uniform quantiser; see Gersho & Gray 1992,
    "Vector Quantization and Signal Compression", chapter 4, for the
    round-to-nearest error bound, and Sayood 2017, "Introduction to Data
    Compression" (5th ed.), chapter 9. Units: whatever the input array is in;
    the bound carries the same unit.

``"rel"``
    Declared **pointwise** relative bound :math:`r`, i.e.
    :math:`|x_i - \\hat{x}_i| \\le r |x_i|` for every sample individually.
    Implemented by quantising the log-magnitude with step
    :math:`\\Delta_y = 2 \\ln(1+r)`:

    .. math:: y_i = \\ln |x_i|, \\quad q_i = \\operatorname{round}(y_i/\\Delta_y),
              \\quad |\\hat{x}_i| = \\exp(q_i \\Delta_y)

    Because :math:`|\\hat y_i - y_i| \\le \\ln(1+r)`, the ratio
    :math:`|\\hat x_i| / |x_i|` lies in :math:`[(1+r)^{-1}, 1+r]`, so the
    relative error is at most :math:`\\max(r, r/(1+r)) = r`. Log-domain
    quantisation for a pointwise relative bound is the construction used by
    the SZ compressor family (Liang et al. 2018, IEEE BigData, "Error-
    controlled lossy compression optimized for high compression ratios of
    scientific datasets"), reimplemented here rather than imported.
    Validity range: every sample must be non-zero and all samples must share
    one sign. A pointwise relative bound is undefined at :math:`x_i = 0`
    (no finite :math:`\\hat x_i` satisfies it unless :math:`\\hat x_i = 0`),
    and the log magnitude is non-monotone across zero, which would destroy
    the smoothness the predictors exploit. Arrays that cross zero must use
    ``"abs"`` or ``"range_rel"``.

``"range_rel"``
    Declared bound relative to the array's dynamic range:
    :math:`b_{\\mathrm{eff}} = r \\,(\\max x - \\min x)`, then exactly
    ``"abs"`` with that bound. This is the "relative to value range" mode of
    zfp and of several scientific compressors. It is **not** a per-sample
    relative bound and this package never describes it as one: a sample near
    zero in an array with a large range gets a very loose relative treatment.
    It is offered because it is the mode most often meant by "1 per cent" in
    practice, and naming the difference is the point.

Encoder-side bound enforcement
------------------------------

The inequalities above are exact in exact arithmetic. In IEEE-754 double
precision, ``q * delta`` and ``exp(q * delta_y)`` each carry a rounding error,
so a sample can land a few ulp outside the declared bound even though the
mathematics says it cannot. Rather than argue about it, :func:`quantise`
checks the realised error of every sample against the declared bound and, for
any sample that violates it, tries :math:`q_i \\pm 1`, keeping the integer
whose reconstruction is admissible and closest to the true value. If no
integer within one step is admissible the function raises
:class:`BoundNotAchievable` instead of returning a certificate it cannot
support. The number of corrected samples is returned and is reported in the
certificate, because a correction is not free: it moves an integer away from
the value the predictor expected and therefore costs rate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

MODES: tuple[str, ...] = ("abs", "rel", "range_rel")

#: Transform identifier stored in the container header. 0 is the identity
#: (``abs`` and ``range_rel``), 1 is the natural log of the magnitude (``rel``).
TRANSFORM_IDENTITY = 0
TRANSFORM_LOG = 1

#: Largest absolute integer symbol the pipeline accepts, as a power of two.
#:
#: Derived, not chosen. Downstream of the quantiser the symbols pass through a
#: predictor whose largest absolute coefficient sum is 15 (``cubic``), giving a
#: prediction of magnitude at most :math:`15S` and a residual of at most
#: :math:`16S`; the zigzag fold in :mod:`boundcompress.entropy` then doubles
#: that, so the pipeline needs :math:`32S < 2^{63}`, i.e. :math:`S < 2^{58}`.
#: The guard sits one bit below at :math:`2^{57}` so that no intermediate is
#: within a factor of two of the ``int64`` ceiling.
#:
#: Found by a Hypothesis property test, not by inspection: at values spanning
#: 8.26e11 with an absolute bound of 1.192092896e-07 the symbols reach 3.5e18,
#: the first difference reaches 4.6e18 and the zigzag fold wrapped to a
#: negative number, which then indexed a frequency table out of range. The
#: counterexample is pinned in ``tests/test_regressions.py``.
MAX_SYMBOL_MAGNITUDE = 2**57


class BoundNotAchievable(ValueError):
    """Raised when no admissible integer exists for some sample.

    This is a refusal, not a warning. The package would rather fail to encode
    than hand back a certificate whose bound it has not verified.
    """


@dataclass(frozen=True)
class QuantiserSpec:
    """The quantiser a declared bound implies, plus what the decoder needs.

    Attributes
    ----------
    mode:
        One of :data:`MODES`, as declared by the caller.
    declared_bound:
        The number the caller declared. Absolute in the unit of the array for
        ``"abs"``; dimensionless for ``"rel"`` and ``"range_rel"``.
    step:
        Quantiser step in the transformed domain. Dimensionless for ``"rel"``
        (it is a step in log magnitude), otherwise in the array's unit.
    transform:
        :data:`TRANSFORM_IDENTITY` or :data:`TRANSFORM_LOG`.
    sign:
        ``+1`` or ``-1``, the common sign of the array, used only by
        :data:`TRANSFORM_LOG`. ``+1`` for the identity transform.
    """

    mode: str
    declared_bound: float
    step: float
    transform: int
    sign: int

    @property
    def absolute_bound(self) -> float:
        """Absolute bound in the array's unit, or ``nan`` for pointwise relative.

        ``"rel"`` has no single absolute bound -- that is the whole point of
        it -- so this property is ``nan`` in that mode and callers must use
        :func:`admissible` instead of comparing against a scalar.
        """
        if self.transform == TRANSFORM_LOG:
            return float("nan")
        return 0.5 * self.step


def validate_bound(mode: str, bound: float) -> None:
    """Reject a bound that cannot define a quantiser.

    Raises
    ------
    ValueError
        If ``mode`` is not in :data:`MODES`, if ``bound`` is not finite, if it
        is not strictly positive, or if a relative bound is at least 1 (a
        100 per cent relative bound is satisfied by the constant zero and
        carries no information).
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    value = float(bound)
    if not math.isfinite(value):
        raise ValueError(f"bound must be finite, got {bound!r}")
    if value <= 0.0:
        raise ValueError(f"bound must be strictly positive, got {bound!r}")
    if mode in ("rel", "range_rel") and value >= 1.0:
        raise ValueError(
            f"a relative bound of {value} is >= 1; a 100 per cent bound is met by the "
            "constant zero array and certifies nothing"
        )


def check_array(values: np.ndarray) -> np.ndarray:
    """Return ``values`` as a contiguous finite 1-D float64 array.

    Raises
    ------
    ValueError
        If the array is not one-dimensional, is empty, or contains a non-finite
        value. ``nan`` and ``inf`` have no admissible quantised representation
        under any finite bound, so they are refused at the door rather than
        propagated into a certificate.
    """
    array = np.ascontiguousarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"expected a 1-D array, got shape {array.shape}")
    if array.size == 0:
        raise ValueError("expected at least one sample, got an empty array")
    if not np.all(np.isfinite(array)):
        bad = int(np.flatnonzero(~np.isfinite(array))[0])
        raise ValueError(
            f"array contains a non-finite value at index {bad}; no finite error bound "
            "admits nan or inf"
        )
    return array


def make_spec(values: np.ndarray, mode: str, bound: float) -> QuantiserSpec:
    """Derive the quantiser a declared bound implies for this array.

    Parameters
    ----------
    values:
        1-D finite array of samples, any unit.
    mode:
        One of :data:`MODES`.
    bound:
        Declared bound. Absolute for ``"abs"``, dimensionless otherwise.

    Raises
    ------
    ValueError
        For an invalid mode or bound, for a non-finite or non-1-D array, for a
        ``"rel"`` request on an array that contains zero or changes sign, or
        for a ``"range_rel"`` request on a constant array (whose range is zero
        and which therefore implies a zero step).
    """
    validate_bound(mode, bound)
    array = check_array(values)
    value = float(bound)

    if mode == "abs":
        return QuantiserSpec("abs", value, 2.0 * value, TRANSFORM_IDENTITY, 1)

    if mode == "range_rel":
        span = float(array.max() - array.min())
        if span <= 0.0:
            raise ValueError(
                "range_rel needs a non-zero dynamic range; this array is constant, so "
                "the implied step is zero. Use mode='abs' with an explicit bound."
            )
        effective = value * span
        return QuantiserSpec("range_rel", value, 2.0 * effective, TRANSFORM_IDENTITY, 1)

    zeros = int(np.count_nonzero(array == 0.0))
    if zeros:
        first = int(np.flatnonzero(array == 0.0)[0])
        raise ValueError(
            f"mode='rel' is a pointwise relative bound and is undefined at zero; "
            f"{zeros} sample(s) are zero, first at index {first}. Use mode='abs' or "
            "mode='range_rel'."
        )
    positive = bool(np.all(array > 0.0))
    negative = bool(np.all(array < 0.0))
    if not (positive or negative):
        raise ValueError(
            "mode='rel' requires all samples to share one sign; the log magnitude is "
            "non-monotone across zero. Use mode='abs' or mode='range_rel'."
        )
    step = 2.0 * math.log1p(value)
    return QuantiserSpec("rel", value, step, TRANSFORM_LOG, 1 if positive else -1)


def forward(values: np.ndarray, spec: QuantiserSpec) -> np.ndarray:
    """Map samples into the quantiser's transformed domain."""
    if spec.transform == TRANSFORM_LOG:
        return np.log(np.abs(values))
    return np.asarray(values, dtype=np.float64)


def inverse(transformed: np.ndarray, spec: QuantiserSpec) -> np.ndarray:
    """Map transformed values back to the array's own domain."""
    if spec.transform == TRANSFORM_LOG:
        return spec.sign * np.exp(transformed)
    return np.asarray(transformed, dtype=np.float64)


def dequantise(symbols: np.ndarray, spec: QuantiserSpec) -> np.ndarray:
    """Reconstruct samples from integer symbols. This is the decoder's only arithmetic."""
    integers = np.asarray(symbols, dtype=np.int64)
    return inverse(integers.astype(np.float64) * spec.step, spec)


def admissible(values: np.ndarray, reconstruction: np.ndarray, spec: QuantiserSpec) -> np.ndarray:
    """Boolean mask of the samples whose reconstruction meets the declared bound.

    Comparison is a non-strict inequality with no tolerance added on either
    side. A tolerance here would make the central claim of this package
    unfalsifiable, which is the one thing it must not be.
    """
    error = np.abs(np.asarray(values, dtype=np.float64) - np.asarray(reconstruction))
    if spec.transform == TRANSFORM_LOG:
        return error <= spec.declared_bound * np.abs(values)
    return error <= spec.absolute_bound


def quantise(values: np.ndarray, spec: QuantiserSpec) -> tuple[np.ndarray, np.ndarray, int]:
    """Quantise, verify the bound on every sample, and correct the exceptions.

    Returns
    -------
    symbols:
        ``int64`` integer symbols, one per sample.
    reconstruction:
        ``float64`` reconstruction the decoder will produce from ``symbols``,
        computed by the same code path the decoder uses.
    corrections:
        Number of samples whose nearest integer was rejected because its
        reconstruction fell outside the declared bound by rounding, and which
        were moved by one step. Zero on every array measured in this
        repository's validation for ``mode="abs"``; see
        ``validation/validate_bound_every_sample_output.txt``.

    Raises
    ------
    BoundNotAchievable
        If some sample has no admissible integer within one step of its
        nearest, or if the integer a sample needs exceeds
        :data:`MAX_SYMBOL_MAGNITUDE`.
        Nothing in this package reports a bound it has not verified.
    """
    array = check_array(values)
    transformed = forward(array, spec)
    # The division itself can overflow to inf for an extreme bound; that case
    # is handled immediately below, so the warning is suppressed rather than
    # printed from library code.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        ratio = transformed / spec.step
    # A bound fine enough relative to the array's magnitude drives the integer
    # symbol past what the pipeline can carry in int64. Without this guard the
    # cast wraps silently and the codec returns a reconstruction unrelated to
    # the input. See MAX_SYMBOL_MAGNITUDE for the derivation and for the two
    # counterexamples that produced it.
    if not np.all(np.isfinite(ratio)) or float(np.max(np.abs(ratio))) >= float(
        MAX_SYMBOL_MAGNITUDE
    ):
        worst = int(np.argmax(np.abs(np.nan_to_num(ratio, nan=np.inf))))
        raise BoundNotAchievable(
            f"sample {worst} needs integer symbol {float(ratio[worst]):.6e}, which "
            f"exceeds the pipeline's symbol limit of 2**57 and so does not fit int64 "
            f"through the predictor and the zigzag fold; the declared {spec.mode} bound "
            f"{spec.declared_bound!r} is too fine for an array of this magnitude"
        )
    symbols = np.rint(ratio).astype(np.int64)
    reconstruction = dequantise(symbols, spec)

    ok = admissible(array, reconstruction, spec)
    corrections = 0
    for index in np.flatnonzero(~ok):
        base = int(symbols[index])
        best: int | None = None
        best_error = math.inf
        for candidate in (base, base - 1, base + 1):
            trial = dequantise(np.array([candidate], dtype=np.int64), spec)
            if bool(admissible(array[index : index + 1], trial, spec)[0]):
                error = abs(float(array[index]) - float(trial[0]))
                if error < best_error:
                    best, best_error = candidate, error
        if best is None:
            raise BoundNotAchievable(
                f"no integer within one step of {base} reconstructs sample {int(index)} "
                f"(value {float(array[index])!r}) inside the declared {spec.mode} bound "
                f"{spec.declared_bound!r}"
            )
        if best != base:
            corrections += 1
        symbols[index] = best

    reconstruction = dequantise(symbols, spec)
    if not bool(np.all(admissible(array, reconstruction, spec))):
        first = int(np.flatnonzero(~admissible(array, reconstruction, spec))[0])
        raise BoundNotAchievable(
            f"sample {first} is outside the declared {spec.mode} bound "
            f"{spec.declared_bound!r} after correction"
        )
    return symbols, reconstruction, corrections

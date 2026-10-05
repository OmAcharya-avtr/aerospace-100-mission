"""IEEE 754 and two's-complement bit layouts, discovered rather than asserted.

Nothing in this module hard-codes a bit offset. Every layout field is derived at
runtime from ``numpy.finfo`` / ``numpy.iinfo`` and from probe encodings, and
:func:`verify_float_layout` re-derives the same quantities a second, independent
way so that a wrong assumption is a test failure rather than a silent error.

Conventions
-----------
Bit indices count from the least significant bit of the *integer view* of the
storage word: bit 0 is the mantissa LSB for a float, the value LSB for an
integer. For IEEE 754 binary interchange formats (IEEE 754-2019 clause 3.4) the
storage word of width ``w`` with ``p - 1`` trailing significand bits is laid out
as

    bit w-1              : sign
    bits w-2 .. p-1      : biased exponent, ``w - p`` bits
    bits p-2 .. 0        : trailing significand (mantissa)

and the value of a normal number is

    (-1)**sign * 2**(E - bias) * (1 + mantissa / 2**nmant)

with ``bias = 2**(nexp - 1) - 1``. All three of ``nmant``, ``nexp`` and ``bias``
are obtained from ``numpy.finfo`` and cross-checked against the encoding of 1.0,
which by the formula above must have ``E == bias`` and ``mantissa == 0``.

Two's-complement integers (the quantized case) follow

    value = -2**(w-1) * b_{w-1} + sum_{k < w-1} 2**k * b_k

so flipping bit ``w-1`` changes the value by ``-2**(w-1)`` when that bit goes
0 -> 1 and by ``+2**(w-1)`` when it goes 1 -> 0, and flipping any lower bit ``k``
changes it by ``+-2**k``. This is verified exhaustively for int8 in the test
suite.

Units: dimensionless throughout. Bit positions are integers; ratios are
dimensionless.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

BitRole = Literal["sign", "exponent", "mantissa", "value"]

_FLOAT_DTYPES: dict[str, np.dtype] = {
    "float16": np.dtype(np.float16),
    "float32": np.dtype(np.float32),
    "float64": np.dtype(np.float64),
}

_INT_DTYPES: dict[str, np.dtype] = {
    "int8": np.dtype(np.int8),
    "int16": np.dtype(np.int16),
    "int32": np.dtype(np.int32),
}

_UINT_FOR_WIDTH: dict[int, np.dtype] = {
    16: np.dtype(np.uint16),
    32: np.dtype(np.uint32),
    64: np.dtype(np.uint64),
    8: np.dtype(np.uint8),
}


@dataclass(frozen=True)
class FloatLayout:
    """Discovered IEEE 754 binary interchange layout for one numpy float dtype.

    Attributes
    ----------
    dtype_name:
        ``"float16"``, ``"float32"`` or ``"float64"``.
    total_bits:
        Storage width ``w`` in bits.
    mantissa_bits:
        ``nmant``, the number of trailing significand bits.
    exponent_bits:
        ``nexp``, the number of biased-exponent bits.
    exponent_bias:
        ``2**(nexp - 1) - 1``, dimensionless.
    sign_bit:
        Index of the sign bit, ``w - 1``.
    exponent_lsb, exponent_msb:
        Inclusive index range of the biased-exponent field.
    """

    dtype_name: str
    total_bits: int
    mantissa_bits: int
    exponent_bits: int
    exponent_bias: int
    sign_bit: int
    exponent_lsb: int
    exponent_msb: int

    @property
    def dtype(self) -> np.dtype:
        return _FLOAT_DTYPES[self.dtype_name]

    @property
    def uint_dtype(self) -> np.dtype:
        return _UINT_FOR_WIDTH[self.total_bits]

    @property
    def max_biased_exponent(self) -> int:
        """Reserved all-ones exponent field value (inf / NaN), dimensionless."""
        return (1 << self.exponent_bits) - 1

    def role(self, position: int) -> BitRole:
        """Return which field bit ``position`` belongs to."""
        _check_position(position, self.total_bits)
        if position == self.sign_bit:
            return "sign"
        if self.exponent_lsb <= position <= self.exponent_msb:
            return "exponent"
        return "mantissa"

    def exponent_weight(self, position: int) -> int:
        """Place value of an exponent bit within the biased-exponent field.

        Flipping exponent bit ``position`` changes the biased exponent by
        ``+-exponent_weight(position)``. Raises for non-exponent bits.
        """
        if self.role(position) != "exponent":
            raise ValueError(
                f"bit {position} of {self.dtype_name} is a "
                f"{self.role(position)} bit, not an exponent bit"
            )
        return 1 << (position - self.exponent_lsb)


@dataclass(frozen=True)
class IntLayout:
    """Two's-complement layout for one numpy signed-integer dtype."""

    dtype_name: str
    total_bits: int
    sign_bit: int

    @property
    def dtype(self) -> np.dtype:
        return _INT_DTYPES[self.dtype_name]

    @property
    def uint_dtype(self) -> np.dtype:
        return _UINT_FOR_WIDTH[self.total_bits]

    def role(self, position: int) -> BitRole:
        _check_position(position, self.total_bits)
        return "sign" if position == self.sign_bit else "value"

    def flip_delta(self, value: int, position: int) -> int:
        """Exact change in the integer value caused by flipping ``position``.

        Returns ``-2**(w-1)`` or ``+2**(w-1)`` for the sign bit depending on its
        current state, and ``+-2**position`` otherwise. Dimensionless.
        """
        _check_position(position, self.total_bits)
        bit_set = (int(value) >> position) & 1 if position != self.sign_bit else (int(value) < 0)
        magnitude = 1 << position
        if position == self.sign_bit:
            return -magnitude if not bit_set else magnitude
        return -magnitude if bit_set else magnitude


def _check_position(position: int, total_bits: int) -> None:
    if not isinstance(position, (int, np.integer)):
        raise TypeError(f"bit position must be an integer, got {type(position).__name__}")
    if not 0 <= int(position) < total_bits:
        raise ValueError(f"bit position {position} outside [0, {total_bits})")


def float_layout(dtype: object) -> FloatLayout:
    """Discover the IEEE 754 layout of ``dtype`` from numpy, asserting nothing.

    Parameters
    ----------
    dtype:
        Anything ``numpy.dtype`` accepts that names float16, float32 or float64.

    Returns
    -------
    FloatLayout
    """
    dt = np.dtype(dtype)
    name = dt.name
    if name not in _FLOAT_DTYPES:
        raise ValueError(
            f"unsupported float dtype {name!r}; expected one of {sorted(_FLOAT_DTYPES)}"
        )
    info = np.finfo(dt)
    total_bits = dt.itemsize * 8
    mantissa_bits = int(info.nmant)
    exponent_bits = int(info.nexp)
    if mantissa_bits + exponent_bits + 1 != total_bits:
        raise ValueError(
            f"{name}: nmant={mantissa_bits} + nexp={exponent_bits} + 1 != {total_bits} storage bits"
        )
    bias = (1 << (exponent_bits - 1)) - 1
    return FloatLayout(
        dtype_name=name,
        total_bits=total_bits,
        mantissa_bits=mantissa_bits,
        exponent_bits=exponent_bits,
        exponent_bias=bias,
        sign_bit=total_bits - 1,
        exponent_lsb=mantissa_bits,
        exponent_msb=total_bits - 2,
    )


def int_layout(dtype: object) -> IntLayout:
    """Discover the two's-complement layout of a signed-integer dtype."""
    dt = np.dtype(dtype)
    if dt.name not in _INT_DTYPES:
        raise ValueError(
            f"unsupported integer dtype {dt.name!r}; expected one of {sorted(_INT_DTYPES)}"
        )
    total_bits = dt.itemsize * 8
    return IntLayout(dtype_name=dt.name, total_bits=total_bits, sign_bit=total_bits - 1)


def to_bits(value: object, dtype: object) -> int:
    """Return the unsigned integer whose bit pattern stores ``value`` as ``dtype``."""
    dt = np.dtype(dtype)
    arr = np.asarray(value, dtype=dt)
    if arr.shape != ():
        raise ValueError("to_bits takes a scalar; use bits_of_array for arrays")
    return int(arr.view(_UINT_FOR_WIDTH[dt.itemsize * 8]))


def from_bits(bits: int, dtype: object) -> object:
    """Inverse of :func:`to_bits`: reinterpret ``bits`` as a ``dtype`` scalar."""
    dt = np.dtype(dtype)
    width = dt.itemsize * 8
    udt = _UINT_FOR_WIDTH[width]
    if not 0 <= int(bits) < (1 << width):
        raise ValueError(f"bit pattern {bits} does not fit in {width} bits")
    return np.asarray(int(bits), dtype=udt).view(dt)[()]


def flip_bit(value: object, position: int, dtype: object) -> object:
    """Flip one storage bit of ``value`` and return the resulting ``dtype`` scalar.

    Works for floats (IEEE 754) and signed integers (two's complement) alike,
    because the flip is performed on the unsigned integer view.
    """
    dt = np.dtype(dtype)
    width = dt.itemsize * 8
    _check_position(position, width)
    return from_bits(to_bits(value, dt) ^ (1 << int(position)), dt)


def bits_of_array(array: np.ndarray) -> np.ndarray:
    """Return the unsigned-integer view of ``array`` (same shape, unsigned dtype)."""
    arr = np.asarray(array)
    width = arr.dtype.itemsize * 8
    if width not in _UINT_FOR_WIDTH:
        raise ValueError(f"unsupported item width {width} bits")
    return arr.view(_UINT_FOR_WIDTH[width])


def exponent_field(value: object, dtype: object) -> int:
    """Biased exponent ``E`` stored in ``value``, dimensionless."""
    layout = float_layout(dtype)
    bits = to_bits(value, layout.dtype)
    return (bits >> layout.exponent_lsb) & layout.max_biased_exponent


def mantissa_field(value: object, dtype: object) -> int:
    """Trailing significand field of ``value``, dimensionless integer."""
    layout = float_layout(dtype)
    bits = to_bits(value, layout.dtype)
    return bits & ((1 << layout.mantissa_bits) - 1)


@dataclass(frozen=True)
class FlipPrediction:
    """Analytic prediction for the effect of flipping one bit of a float.

    Attributes
    ----------
    regime:
        ``"scaled"``       - both operand and result are normal; the magnitude
                             ratio ``predicted_ratio`` is exact in binary.
        ``"to_infinity"``  - the biased exponent saturates to all ones with a
                             zero mantissa; the result is an infinity.
        ``"to_nan"``       - all-ones exponent with a non-zero mantissa.
        ``"from_special"`` - the operand was already inf, NaN, zero or subnormal.
        ``"to_subnormal"`` - the biased exponent lands on zero.
        ``"sign_only"``    - the sign bit; ratio is exactly -1.
        ``"additive"``     - a mantissa bit; the change is additive, and
                             ``predicted_delta`` rather than ``predicted_ratio``
                             is the exact quantity.
    predicted_ratio:
        ``flipped / original`` where that is exact and finite, else ``nan``.
    predicted_delta:
        ``flipped - original`` where that is exact and finite, else ``nan``.
    predicted_value:
        The exact result, always available, computed from the bit pattern.
    """

    regime: str
    predicted_value: float
    predicted_ratio: float
    predicted_delta: float


def predict_flip(value: object, position: int, dtype: object) -> FlipPrediction:
    """Predict the effect of flipping bit ``position``, from the layout alone.

    The predicted *value* is always obtained from the layout formula
    ``(-1)**s * 2**(E - bias) * (1 + m / 2**nmant)`` evaluated in exact Python
    integer / fraction arithmetic, never by calling :func:`flip_bit`. The
    validation scripts compare this prediction against the actual flip.

    Parameters
    ----------
    value:
        Scalar, interpreted as ``dtype``.
    position:
        Storage-bit index, 0 = mantissa LSB.
    dtype:
        float16, float32 or float64.
    """
    layout = float_layout(dtype)
    _check_position(position, layout.total_bits)
    position = int(position)

    bits = to_bits(value, layout.dtype)
    sign = (bits >> layout.sign_bit) & 1
    exponent = (bits >> layout.exponent_lsb) & layout.max_biased_exponent
    mantissa = bits & ((1 << layout.mantissa_bits) - 1)
    original_special = exponent in (0, layout.max_biased_exponent)

    new_bits = bits ^ (1 << position)
    new_sign = (new_bits >> layout.sign_bit) & 1
    new_exponent = (new_bits >> layout.exponent_lsb) & layout.max_biased_exponent
    new_mantissa = new_bits & ((1 << layout.mantissa_bits) - 1)

    predicted_value = _decode(new_sign, new_exponent, new_mantissa, layout)
    original_value = _decode(sign, exponent, mantissa, layout)

    role = layout.role(position)
    if original_special:
        regime = "from_special"
    elif new_exponent == layout.max_biased_exponent:
        regime = "to_nan" if new_mantissa else "to_infinity"
    elif new_exponent == 0:
        regime = "to_subnormal"
    elif role == "sign":
        regime = "sign_only"
    elif role == "exponent":
        regime = "scaled"
    else:
        regime = "additive"

    finite = math.isfinite(predicted_value) and math.isfinite(original_value)
    ratio = predicted_value / original_value if finite and original_value != 0.0 else math.nan
    delta = predicted_value - original_value if finite else math.nan
    return FlipPrediction(
        regime=regime,
        predicted_value=predicted_value,
        predicted_ratio=ratio,
        predicted_delta=delta,
    )


def _decode(sign: int, exponent: int, mantissa: int, layout: FloatLayout) -> float:
    """Decode an IEEE 754 field triple to a Python float by the layout formula."""
    scale = 1 << layout.mantissa_bits
    if exponent == layout.max_biased_exponent:
        if mantissa:
            return math.nan
        return math.inf if sign == 0 else -math.inf
    if exponent == 0:
        magnitude = math.ldexp(mantissa / scale, 1 - layout.exponent_bias)
    else:
        magnitude = math.ldexp(1.0 + mantissa / scale, exponent - layout.exponent_bias)
    return -magnitude if sign else magnitude


def exponent_scale_factor(position: int, direction: int, dtype: object) -> float:
    """Exact magnitude ratio produced by flipping exponent bit ``position``.

    Flipping exponent bit at ``position`` changes the biased exponent ``E`` by
    ``+-2**(position - nmant)``, so the magnitude is multiplied by
    ``2**(+-2**(position - nmant))`` whenever both operand and result are normal.

    Parameters
    ----------
    position:
        An exponent-field bit index.
    direction:
        ``+1`` for a 0 -> 1 flip (exponent increases), ``-1`` for 1 -> 0.
    dtype:
        float16, float32 or float64.

    Returns
    -------
    float
        The multiplicative factor, dimensionless. May be ``inf`` or ``0.0`` when
        the factor itself is not representable; the caller should use
        :func:`predict_flip` for the exact value in that case.
    """
    layout = float_layout(dtype)
    if direction not in (1, -1):
        raise ValueError(f"direction must be +1 or -1, got {direction}")
    weight = layout.exponent_weight(position)
    exponent = direction * weight
    try:
        return math.ldexp(1.0, exponent)
    except OverflowError:
        # 2**exponent is not representable as a float64 (float64's own exponent
        # MSB has weight 1024, and 2**1024 overflows). The factor is still
        # exactly defined; the caller should compare biased-exponent fields with
        # :func:`exponent_field_prediction` instead of forming a float ratio.
        return math.inf if exponent > 0 else 0.0


def exponent_field_prediction(value: object, position: int, dtype: object) -> dict[str, int]:
    """Exact field-level prediction for flipping one exponent bit.

    Returns the predicted sign, biased exponent and mantissa fields of the
    result, computed as integers from the layout, together with the signed
    change in the biased exponent. Integer arithmetic has no overflow, so this
    is the form the validation uses for float64, whose exponent MSB has weight
    1024 and whose magnitude ratio ``2**1024`` is not a representable float.

    Raises
    ------
    ValueError
        If ``position`` is not an exponent-field bit.
    """
    layout = float_layout(dtype)
    if layout.role(position) != "exponent":
        raise ValueError(f"bit {position} is a {layout.role(position)} bit, not an exponent bit")
    bits = to_bits(value, layout.dtype)
    sign = (bits >> layout.sign_bit) & 1
    exponent = (bits >> layout.exponent_lsb) & layout.max_biased_exponent
    mantissa = bits & ((1 << layout.mantissa_bits) - 1)
    weight = layout.exponent_weight(position)
    direction = 1 if ((bits >> position) & 1) == 0 else -1
    return {
        "sign": sign,
        "exponent": exponent + direction * weight,
        "mantissa": mantissa,
        "delta_exponent": direction * weight,
        "original_exponent": exponent,
        "direction": direction,
        "weight": weight,
    }


def verify_float_layout(dtype: object) -> dict[str, object]:
    """Re-derive every layout field a second, independent way.

    Rather than trusting ``numpy.finfo``, this probes encodings:

    * ``nmant`` from ``bits(2.0) - bits(1.0)``, which by the layout formula is
      exactly ``1 << nmant`` because both encodings have a zero mantissa and
      exponents differing by one;
    * ``bias`` from the exponent field of 1.0, which the formula requires to
      equal the bias;
    * ``nexp`` from the width of ``bits(inf) >> nmant``, the all-ones field;
    * the sign-bit index from the single set bit of ``-0.0``.

    Returns a dict of ``{field: (from_finfo, from_probe, agree)}`` plus the
    overall ``"agree"`` flag. Used by ``validation/validate_ieee754_layout.py``
    and by the test suite; a disagreement is a hard error, not a warning.
    """
    layout = float_layout(dtype)
    dt = layout.dtype
    width = layout.total_bits

    one_bits = to_bits(np.asarray(1.0, dtype=dt), dt)
    two_bits = to_bits(np.asarray(2.0, dtype=dt), dt)
    probe_nmant = (two_bits - one_bits).bit_length() - 1

    neg_zero_bits = to_bits(np.asarray(-0.0, dtype=dt), dt)
    probe_sign_bit = neg_zero_bits.bit_length() - 1

    inf_bits = to_bits(np.asarray(np.inf, dtype=dt), dt)
    exponent_mask = inf_bits >> probe_nmant
    probe_nexp = exponent_mask.bit_length()

    probe_bias = (one_bits >> probe_nmant) & ((1 << probe_nexp) - 1)

    fields = {
        "mantissa_bits": (layout.mantissa_bits, probe_nmant),
        "exponent_bits": (layout.exponent_bits, probe_nexp),
        "exponent_bias": (layout.exponent_bias, probe_bias),
        "sign_bit": (layout.sign_bit, probe_sign_bit),
        "total_bits": (width, probe_sign_bit + 1),
    }
    report: dict[str, object] = {
        name: (declared, probed, declared == probed) for name, (declared, probed) in fields.items()
    }
    report["agree"] = all(entry[2] for entry in report.values())  # type: ignore[index]
    report["dtype_name"] = layout.dtype_name
    return report


def bit_site_count(n_elements: int, bits_per_element: int) -> int:
    """Total number of flippable bits in a parameter block, dimensionless."""
    if n_elements < 0 or bits_per_element <= 0:
        raise ValueError("n_elements must be >= 0 and bits_per_element > 0")
    return int(n_elements) * int(bits_per_element)

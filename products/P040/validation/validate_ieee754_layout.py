"""Validation: the IEEE 754 layout, and the exponent-bit flip prediction.

Checks
------
1. Every layout field of float16, float32 and float64 re-derived from probe
   encodings agrees with ``numpy.finfo``.
2. For every one of those dtypes and a grid of values, the exact result of
   flipping each storage bit equals the prediction made from the layout formula
   alone, in Python integer arithmetic, with no call to the flip itself.
3. Specifically for the most significant exponent bit - the bit that carries
   the sign of the unbiased exponent - the magnitude ratio equals
   ``2**(+-2**(nexp-1))`` in every case where both operand and result are
   normal, exactly, and equals the predicted infinity or subnormal in the cases
   where it is not.
4. The two's-complement int8 flip delta is exhaustively correct over all 256
   values and all 8 bit positions.

Nothing here asserts a bit offset from memory; every offset comes from
``bitflipsim.bitlayout``, which derives it at runtime.

Runtime: under 5 s on one core.
"""

from __future__ import annotations

import math
import sys

import numpy as np

from bitflipsim.bitlayout import (
    exponent_field_prediction,
    exponent_scale_factor,
    flip_bit,
    float_layout,
    int_layout,
    predict_flip,
    to_bits,
    verify_float_layout,
)

FLOATS = ("float16", "float32", "float64")
failures: list[str] = []


def report(name: str, ok: bool, detail: str) -> None:
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}: {detail}")
    if not ok:
        failures.append(name)


print("=" * 78)
print("Check 1 - layout fields re-derived from probe encodings")
print("=" * 78)
for name in FLOATS:
    layout = float_layout(name)
    report_dict = verify_float_layout(name)
    print(f"\n{name}: {layout.total_bits} bits = 1 sign + {layout.exponent_bits} exponent "
          f"+ {layout.mantissa_bits} mantissa, bias {layout.exponent_bias}")
    for key in ("mantissa_bits", "exponent_bits", "exponent_bias", "sign_bit", "total_bits"):
        declared, probed, agree = report_dict[key]
        print(f"    {key:<16} finfo={declared:<6} probe={probed:<6} agree={agree}")
    report(f"layout re-derivation {name}", bool(report_dict["agree"]),
           "every field agrees" if report_dict["agree"] else "disagreement above")

print()
print("=" * 78)
print("Check 2 - exact flip prediction over a value grid, all bit positions")
print("=" * 78)
GRID = [
    1.0, -1.0, 0.5, 2.0, 3.5, 5.0, 123.25, 1024.0, -77.0, 0.00390625,
    -0.001953125, 17.0, -256.0, 6.5, 0.125,
]
for name in FLOATS:
    layout = float_layout(name)
    values = np.array(GRID, dtype=layout.dtype)
    compared = 0
    mismatch = 0
    worst: tuple[float, int, float, float] | None = None
    for value in values:
        for bit in range(layout.total_bits):
            prediction = predict_flip(value, bit, name)
            actual = float(flip_bit(value, bit, name))
            compared += 1
            if math.isnan(prediction.predicted_value):
                ok = math.isnan(actual)
            else:
                ok = actual == prediction.predicted_value
            if not ok:
                mismatch += 1
                worst = (float(value), bit, actual, prediction.predicted_value)
    print(f"{name}: {compared} (value, bit) pairs compared, {mismatch} mismatched")
    report(f"flip prediction {name}", mismatch == 0,
           f"{compared} exact comparisons, 0 mismatches" if mismatch == 0 else f"worst {worst}")

print()
print("=" * 78)
print("Check 3 - most significant exponent bit: predicted magnitude change")
print("=" * 78)
print("Derivation: for a normal float the stored value is")
print("    (-1)**s * 2**(E - bias) * (1 + m / 2**nmant)")
print("so flipping the exponent bit at position p changes E by +- 2**(p - nmant)")
print("and leaves s and m untouched. For the exponent MSB, p - nmant = nexp - 1,")
print("hence the magnitude is multiplied by 2**(+- 2**(nexp - 1)) whenever both")
print("the operand and the result are normal. The field-level prediction below is")
print("integer arithmetic, so it is exact even for float64, whose 2**1024 factor")
print("is not a representable float.")
for name in FLOATS:
    layout = float_layout(name)
    msb = layout.exponent_msb
    weight = layout.exponent_weight(msb)
    print(f"\n{name}: exponent MSB is bit {msb}, exponent weight 2**"
          f"{msb - layout.exponent_lsb} = {weight}, so the predicted factor is 2**(+-{weight})")
    print(f"    {'value':>22} {'E':>6} {'dE':>6} {'regime':<14} {'ratio (where finite)':>24} "
          f"{'predicted ratio':>24}")
    scaled = 0
    special = 0
    field_mismatch = 0
    worst_relative = 0.0
    ratio_checks = 0
    for raw in GRID:
        value = np.asarray(raw, dtype=layout.dtype)[()]
        fields = exponent_field_prediction(value, msb, name)
        prediction = predict_flip(value, msb, name)
        actual = flip_bit(value, msb, name)
        actual_bits = to_bits(actual, name)
        actual_sign = (actual_bits >> layout.sign_bit) & 1
        actual_exponent = (actual_bits >> layout.exponent_lsb) & layout.max_biased_exponent
        actual_mantissa = actual_bits & ((1 << layout.mantissa_bits) - 1)
        if (actual_sign, actual_exponent, actual_mantissa) != (
            fields["sign"], fields["exponent"] & layout.max_biased_exponent, fields["mantissa"]
        ):
            field_mismatch += 1
        ratio_text = "-"
        predicted_text = "-"
        if prediction.regime == "scaled":
            scaled += 1
            expected_ratio = exponent_scale_factor(msb, fields["direction"], name)
            measured_ratio = float(actual) / float(value)
            if math.isfinite(expected_ratio) and expected_ratio != 0.0:
                ratio_checks += 1
                relative = abs(measured_ratio - expected_ratio) / abs(expected_ratio)
                worst_relative = max(worst_relative, relative)
            ratio_text = f"{measured_ratio:.17g}"
            predicted_text = f"{expected_ratio:.17g}"
        else:
            special += 1
            agree = (math.isnan(prediction.predicted_value) and math.isnan(float(actual))) or (
                float(actual) == prediction.predicted_value
            )
            if not agree:
                failures.append(f"exponent MSB special case {name} {float(value)}")
            ratio_text = f"{float(actual):.17g}"
            predicted_text = f"{prediction.predicted_value:.17g}"
        print(f"    {float(value):>22.10g} {fields['original_exponent']:>6} "
              f"{fields['delta_exponent']:>+6} {prediction.regime:<14} "
              f"{ratio_text:>24} {predicted_text:>24}")
    print(f"    {scaled} scaled cases, {special} special cases "
          f"(infinity / NaN / subnormal / already special); "
          f"{ratio_checks} with a representable float ratio")
    report(f"exponent MSB field prediction {name}", field_mismatch == 0,
           f"sign, biased exponent and mantissa fields of {len(GRID)} results all "
           f"match the integer prediction ({field_mismatch} mismatches)")
    report(f"exponent MSB ratio {name}", worst_relative == 0.0 and ratio_checks > 0,
           f"worst relative deviation from 2**(+-{weight}) over {ratio_checks} normal "
           f"cases with a representable ratio = {worst_relative:.3e}")

print()
print("=" * 78)
print("Check 4 - two's-complement int8 flip delta, exhaustive")
print("=" * 78)
layout8 = int_layout("int8")
compared = 0
mismatch = 0
for raw in range(-128, 128):
    value = np.int8(raw)
    for bit in range(8):
        expected = layout8.flip_delta(int(value), bit)
        actual = int(flip_bit(value, bit, "int8"))
        compared += 1
        if actual - int(value) != expected:
            mismatch += 1
print(f"{compared} (value, bit) pairs = 256 values x 8 bits; {mismatch} mismatched")
print("sign-bit delta is -128 when the bit goes 0 -> 1 and +128 when it goes 1 -> 0:")
for raw in (0, -128, -1, 127):
    print(f"    int8 {raw:>5} bit 7 -> {int(flip_bit(np.int8(raw), 7, 'int8')):>5} "
          f"(delta {layout8.flip_delta(raw, 7):+5d})")
report("int8 flip delta exhaustive", mismatch == 0 and compared == 2048,
       f"{compared} exhaustive comparisons, {mismatch} mismatches")

print()
print("=" * 78)
if failures:
    print(f"RESULT: {len(failures)} FAILED check(s): {failures}")
    sys.exit(1)
print("RESULT: all checks PASSED")

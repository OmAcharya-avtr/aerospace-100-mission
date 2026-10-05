"""Per-bit criticality by position: float32 exponent versus mantissa, and int8.

Runs the full ground-truth sweep for both storage formats and plots the mean
degradation per bit position. Writes ``../screenshots/bit_position_criticality.png``.

What to notice: the float32 panel is a step, not a slope. The exponent MSB is
four orders of magnitude worse than the mantissa LSB, and the eight exponent
bits plus the sign bit carry almost all of the damage in nine of thirty-two
bits. The int8 panel has no such step: two's-complement has no exponent, so the
worst bit is the sign bit and its effect is bounded by 128 times the
quantization scale.

Runtime: about 10 s on one core.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from bitflipsim.bitlayout import float_layout, int_layout  # noqa: E402
from bitflipsim.criticality import (  # noqa: E402
    sweep_bit_criticality,
    sweep_quantized_bit_criticality,
)
from bitflipsim.datasets import make_problem, reference_parameters  # noqa: E402
from bitflipsim.network import quantize_int8  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "screenshots", "bit_position_criticality.png")

problem = make_problem()
params = reference_parameters(problem)
quantized = quantize_int8(params)
f32 = float_layout("float32")
i8 = int_layout("int8")

float_sweep = sweep_bit_criticality(params, problem.evaluation.x, problem.evaluation.y)
int_sweep = sweep_quantized_bit_criticality(quantized, problem.evaluation.x, problem.evaluation.y)

float_per_bit = float_sweep.by_bit_position()
int_per_bit = int_sweep.by_bit_position()

print(f"model                     {params.layout.n_in} -> {params.layout.n_hidden} "
      f"(ReLU) -> {params.layout.n_out}, {params.layout.size} parameters")
print(f"evaluation samples        {problem.evaluation.n_samples}")
print(f"float32 sites swept       {float_sweep.n_evaluations}")
print(f"int8 sites swept          {int_sweep.n_evaluations}")
print(f"golden accuracy float32   {float_sweep.golden_accuracy:.6f}")
print(f"golden accuracy int8      {int_sweep.golden_accuracy:.6f}")
print()
print("float32 mean degradation (mean total-variation distance) by bit position")
print(f"{'bit':>4} {'role':<9} {'mean TV':>13} {'max TV':>9} {'mean acc drop':>14}")
for bit in range(f32.total_bits - 1, -1, -1):
    print(f"{bit:>4} {f32.role(bit):<9} {float_per_bit[bit]:>13.6e} "
          f"{float_sweep.degradation[:, bit].max():>9.6f} "
          f"{float_sweep.accuracy_drop[:, bit].mean():>14.6f}")
print()
print("int8 mean degradation by bit position")
print(f"{'bit':>4} {'role':<9} {'mean TV':>13} {'max TV':>9} {'mean acc drop':>14}")
for bit in range(i8.total_bits - 1, -1, -1):
    print(f"{bit:>4} {i8.role(bit):<9} {int_per_bit[bit]:>13.6e} "
          f"{int_sweep.degradation[:, bit].max():>9.6f} "
          f"{int_sweep.accuracy_drop[:, bit].mean():>14.6f}")

golden_probs = params.probabilities(problem.evaluation.x)
quantized_probs = quantized.as_parameters().probabilities(problem.evaluation.x)
quantization_shift = float(np.abs(golden_probs - quantized_probs).max())
changed = int(
    (params.predict(problem.evaluation.x)
     != quantized.as_parameters().predict(problem.evaluation.x)).sum()
)
print()
print("int8 quantization is not lossless, and that is separate from any upset:")
print(f"max per-class probability shift from quantization alone  "
      f"{quantization_shift:.6f}")
print(f"argmax predictions changed by quantization alone         {changed} of "
      f"{problem.evaluation.n_samples}")
print(f"accuracy float32 {float_sweep.golden_accuracy:.6f} -> int8 "
      f"{int_sweep.golden_accuracy:.6f}")

exponent_share = float_per_bit[f32.exponent_lsb :].sum() / float_per_bit.sum()
print()
print(f"float32: exponent field + sign is {100.0 * exponent_share:.3f} % of the total "
      f"damage in {f32.exponent_bits + 1} of {f32.total_bits} bits")
print(f"float32: worst bit is {int(np.argmax(float_per_bit))} "
      f"({f32.role(int(np.argmax(float_per_bit)))}), mean TV "
      f"{float_per_bit.max():.6f}, max TV {float_sweep.degradation.max():.6f}")
print(f"int8:    worst bit is {int(np.argmax(int_per_bit))} "
      f"({i8.role(int(np.argmax(int_per_bit)))}), mean TV {int_per_bit.max():.6f}, "
      f"max TV {int_sweep.degradation.max():.6f}")
print(f"ratio of worst mean TV, float32 / int8: "
      f"{float_per_bit.max() / int_per_bit.max():.3f}")
print(f"ratio of total damage, float32 / int8:  "
      f"{float_sweep.degradation.sum() / int_sweep.degradation.sum():.3f} "
      f"(over 4x as many bit sites)")
print(f"per-site mean, float32 / int8:          "
      f"{float_sweep.degradation.mean() / int_sweep.degradation.mean():.3f}")

figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))

colours = {"sign": "#b2182b", "exponent": "#2166ac", "mantissa": "#7f7f7f"}
left = axes[0]
positions = np.arange(f32.total_bits)
bar_colours = [colours[f32.role(b)] for b in positions]
left.bar(positions, np.maximum(float_per_bit, 1e-12), color=bar_colours, width=0.82)
left.set_yscale("log")
left.set_xlabel("float32 storage bit position (0 = mantissa LSB)")
left.set_ylabel("mean total-variation degradation (log scale)")
left.set_title(f"float32: one upset per site, all {float_sweep.n_evaluations} sites")
left.set_xticks(np.arange(0, 32, 2))
left.axvspan(f32.exponent_lsb - 0.5, f32.exponent_msb + 0.5, color="#2166ac", alpha=0.08)
left.axvspan(f32.sign_bit - 0.5, f32.sign_bit + 0.5, color="#b2182b", alpha=0.10)
left.set_ylim(top=float_per_bit.max() * 60.0)
left.annotate(
    f"exponent MSB (bit {f32.exponent_msb})\nmean TV {float_per_bit[f32.exponent_msb]:.4f}",
    xy=(f32.exponent_msb, float_per_bit[f32.exponent_msb]),
    xytext=(11.0, float_per_bit[f32.exponent_msb] * 6.0),
    arrowprops={"arrowstyle": "->", "lw": 1.0},
    fontsize=9,
)
handles = [plt.Rectangle((0, 0), 1, 1, color=colours[k]) for k in ("sign", "exponent", "mantissa")]
left.legend(handles, ["sign", "exponent", "mantissa"], loc="lower right", fontsize=9)
left.grid(axis="y", alpha=0.3)

right = axes[1]
int_positions = np.arange(i8.total_bits)
int_colours = ["#b2182b" if i8.role(b) == "sign" else "#7f7f7f" for b in int_positions]
right.bar(int_positions, np.maximum(int_per_bit, 1e-12), color=int_colours, width=0.7)
right.set_yscale("log")
right.set_xlabel("int8 storage bit position (0 = value LSB)")
right.set_ylabel("mean total-variation degradation (log scale)")
right.set_title(f"int8 two's complement, all {int_sweep.n_evaluations} sites")
right.set_xticks(int_positions)
right.set_ylim(left.get_ylim())
right.grid(axis="y", alpha=0.3)
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ("#b2182b", "#7f7f7f")]
right.legend(handles, ["sign", "value"], loc="lower right", fontsize=9)
right.annotate(
    f"sign bit, mean TV {int_per_bit[i8.sign_bit]:.4f}\n"
    f"bounded by 128 x the quantization scale;\n"
    f"int8 has no exponent, so there is no step",
    xy=(i8.sign_bit, int_per_bit[i8.sign_bit]),
    xytext=(0.2, int_per_bit[i8.sign_bit] * 12.0),
    arrowprops={"arrowstyle": "->", "lw": 1.0},
    fontsize=9,
)

figure.suptitle(
    "Per-bit criticality by storage position, measured by exhaustive single-upset injection",
    fontsize=11,
)
figure.tight_layout()
figure.savefig(OUT, dpi=130)
print()
print(f"wrote screenshots/{os.path.basename(OUT)}")

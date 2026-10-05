# Validation evidence

**Validation level 2 (research grade). Status: TESTING.**

Every number in this file, in `README.md` and in `MODEL_CARD.md` was produced by
a script in this directory, run on 2026-10-05 in the build container, with its
raw standard output committed beside it. The commands that reproduce each one
are at the end of this file. Nothing here is quoted from memory or from a
reference that was not read.

Build host: Python 3.13.16, numpy 2.5.3, scipy 1.18.1, scikit-learn 1.9.1,
onnxruntime 1.29.0, pytest 9.1.1, one CPU core shared with four other build
agents. Timing figures are therefore upper bounds and vary between runs; they
are labelled where they appear.

## Summary

| # | Check | Reference | Result | Tolerance / gate | Script |
|---|---|---|---|---|---|
| 1 | Layout fields re-derived from probe encodings, float16/32/64 | IEEE 754-2019 binary interchange formats, re-derived in `docs/UPSET_MODEL.md` §1.1 | all 5 fields agree with `numpy.finfo` for all 3 dtypes | exact | `validate_ieee754_layout.py` |
| 2 | Flip prediction vs actual flip, all bit positions | layout formula in integer arithmetic | **0 mismatches** in 240 + 480 + 960 = 1680 (value, bit) pairs | bit-exact required | `validate_ieee754_layout.py` |
| 3 | Exponent MSB magnitude ratio, float32 | `2**(+-2**(nexp-1)) = 2**(+-128)` | worst relative deviation **0.000e+00** over 11 normal cases | 0 required | `validate_ieee754_layout.py` |
| 4 | Exponent MSB field prediction, float16/32/64 | sign, biased exponent and mantissa predicted as integers | **0 mismatches** in 45 cases | exact | `validate_ieee754_layout.py` |
| 5 | int8 two's-complement flip delta | `-2**(w-1)` / `+2**(w-1)` on the sign bit, `+-2**k` otherwise | **0 mismatches** in 2048 exhaustive comparisons | exact | `validate_ieee754_layout.py` |
| 6 | Bits flipped equals upsets drawn | population count of `golden XOR faulty` | 4012 drawn, **4012 flipped**, 0 mismatched trials of 500 | exact | `validate_poisson_counts.py` |
| 7 | Poisson mean and variance vs model | `E[K] = Var[K] = mu`; `se(mean) = sqrt(mu/n)`, `se(S2) = sqrt((mu+2mu^2)/n)` | worst \|z\| **2.517** over 6 expectations x 2 statistics, n = 200000 | \|z\| < 4 | `validate_poisson_counts.py` |
| 8 | Chi-square fit of the count histogram | Poisson pmf at mu = 6, tail pooled to expectation >= 5 | chi2 = **19.649** on 19 dof, **p = 0.4159** | p > 0.01 | `validate_poisson_counts.py` |
| 9 | Bit-position uniformity | multinomial sigma over 32 positions, 320000 draws | worst \|z\| **2.073** over 32 bins | \|z\| < 4.5 | `validate_poisson_counts.py` |
| 10 | `E[live upsets] = lambda T_s / 2` under scrubbing | derived in `docs/UPSET_MODEL.md` §4.3 | worst \|z\| **0.961** over 4 configurations, 200000 observations each | \|z\| < 4 | `validate_poisson_counts.py` |
| 11 | **Clamped range bounds the output deviation** | `B = 2C max(C(n_in Xmax + 1), 1)`, derived in §4.1 | C = 4.581975: B = **1119.613957**, worst measured **142.435628**, **0 violations in 4704 exhaustive single-upset cases** | no violation permitted | `validate_clamp_bound.py` |
| 12 | The same at two tighter clamps | as above | C = 2.290987: B = 279.903, worst 71.218, 0 violations. C = 1.145494: B = 69.976, worst 24.104, 0 violations | no violation permitted | `validate_clamp_bound.py` |
| 13 | Without clamping, no finite bound exists | - | **50 of 4704** single upsets give non-finite logits; worst finite deviation **2.786753e+39** | reported, not gated | `validate_clamp_bound.py` |
| 14 | **TMR single-upset recovery, exhaustive** | word-level and bitwise voters | **144 of 144** cases recovered by both voters, 0 wrong, 0 flagged (float16 case); **96 of 96** (float32 case) | exact | `validate_triplication.py` |
| 15 | **TMR double-upset behaviour, exhaustive** | all C(144,2) = 10296 pairs | same copy 3384/3384 correct; different elements 4608/4608 correct; same element same bit **0/144 correct, 0 flagged - silently wrong**; same element different bit 0/2160 correct but **2160/2160 flagged** | exact | `validate_triplication.py` |
| 16 | Exponent MSB is the most critical bit position | measured, not assumed | bit **30** of 32 has the highest mean degradation | identity required | `validate_criticality_methods.py` |
| 17 | Exponent field plus sign dominates the mantissa | measured | **98.853 %** of total damage in 9 of 32 bits; factor **86.22** over the 23 mantissa bits | > 10x | `validate_criticality_methods.py` |
| 18 | **Degradation avoided per protected byte, bit-granular, 5 % budget** | the metric the specification names | exponent heuristic **3.164676**, learned predictor **3.550208** (**+12.2 %**), magnitude baseline 0.364869, oracle 3.616972, random 0.258950 | reported | `validate_criticality_methods.py` |
| 19 | **The same, word-granular** | what ECC or word TMR actually charges | exponent heuristic **0.220135 at every budget** - exactly the random-selection expectation; magnitude baseline beats the learned predictor at 4 of 6 budgets | reported | `validate_criticality_methods.py` |
| 20 | Uncertainty output coverage | fraction of held-out targets within 2 ensemble sigma | **0.878178** against a Gaussian reference of 0.954500 | reported, not gated | `validate_criticality_methods.py` |
| 21 | numpy forward pass equals `MLPClassifier.predict_proba` | scikit-learn 1.9.1 | max \|difference\| **0.000000e+00** with float64 storage | bit-exact required | `validate_onnx_path.py` |
| 22 | float32 storage cost | - | max \|difference\| **3.585047e-07** | < 1e-5 | `validate_onnx_path.py` |
| 23 | Hand-encoded ONNX loads in onnxruntime and agrees | onnxruntime 1.29.0 | max \|logit difference\| **6.260614e-06** over 300 samples | < 1e-4 | `validate_onnx_path.py` |
| 24 | Initializers located and decoded from the file bytes | - | **4 of 4**, 147 elements, all bit-exact | exact | `validate_onnx_path.py` |
| 25 | In-file bit flips match the layout prediction | `predict_flip` | **6 of 6** agree; `W1[5]` bit 30: -0.162728533 -> -5.53736504e+37, max \|d logit\| **3.268108e+38** | exact | `validate_onnx_path.py` |

## 1. Storage layout (`validate_ieee754_layout.py`)

Raw output: [`ieee754_layout_output.txt`](ieee754_layout_output.txt). Runtime about 3 s.

No bit offset is hard-coded anywhere in this package. Every layout field is
obtained from `numpy.finfo` and then re-derived a second, independent way from
probe encodings - `nmant` from `bits(2.0) - bits(1.0)`, the bias from the
exponent field of `1.0`, `nexp` from the width of `bits(inf) >> nmant`, the
sign-bit index from `-0.0`. All five fields agree for all three dtypes.

The single-bit-flip prediction is computed from the layout formula in exact
Python integer arithmetic and compared against the actual flip for 15 values x
every bit position of each dtype: 1680 comparisons, zero mismatches.

For the **exponent sign bit** - the most significant exponent bit, which carries
the sign of the unbiased exponent - the prediction is `E -> E +- 2**(nexp-1)`,
i.e. `+-128` for float32, with the sign and mantissa fields untouched. Of 15
float32 values, 11 land in the normal-to-normal regime and show the magnitude
ratio `2**(+-128)` with **zero** relative deviation; the other 4 land in
predicted special regimes (`1.0 -> +inf`, `3.5 -> subnormal`) and match those
predictions exactly. For float64 the factor `2**1024` is not a representable
float, which is why the validated form is the integer field prediction and the
ratio is checked only on the 7 cases where it is representable.

## 2. Injected counts and the Poisson model (`validate_poisson_counts.py`)

Raw output: [`poisson_counts_output.txt`](poisson_counts_output.txt). Runtime about 5 s.

The flux and cross-section used are the package's **illustrative** constants
(`1e3 particles cm^-2 s^-1`, `1e-14 cm^2 bit^-1`). They are round numbers chosen
so the examples run; they are **not** measured environment or part figures and
must not be cited as such. With them, the 4704-bit reference model has
`lambda = 4.704e-08 upsets s^-1` (`1.693440e+05 FIT`).

| mu | sample mean | se(mean) | z | sample variance | se(var) | z |
|---|---|---|---|---|---|---|
| 0.5 | 0.498665 | 0.001581 | -0.84 | 0.497691 | 0.002236 | -1.03 |
| 1.0 | 0.999120 | 0.002236 | -0.39 | 0.999114 | 0.003873 | -0.23 |
| 2.0 | 1.996510 | 0.003162 | -1.10 | 2.003338 | 0.007071 | +0.47 |
| 4.0 | 4.004000 | 0.004472 | +0.89 | 4.019684 | 0.013416 | +1.47 |
| 8.0 | 7.992635 | 0.006325 | -1.16 | 8.038241 | 0.026077 | +1.47 |
| 16.0 | 16.022515 | 0.008944 | +2.52 | 16.058138 | 0.051381 | +1.13 |

n = 200000 per row. The link between the rate model and the injector is checked
separately: over 500 trials at mu = 8, the population count of
`golden XOR faulty` equalled the drawn count in every trial (4012 upsets over 500 trials).

## 3. Clamping bound (`validate_clamp_bound.py`)

Raw output: [`clamp_bound_output.txt`](clamp_bound_output.txt). Runtime about 11 to 16 s.

Derivation in [`../docs/UPSET_MODEL.md`](../docs/UPSET_MODEL.md) §4.1. Checked
over **every** single-upset site, not a sample. The batch is the first 200 of
the 300 evaluation samples, which keeps four exhaustive 4704-site sweeps inside
the per-script compute budget; `Xmax` for that batch is 3.208058866 and is the
`Xmax` the bound is stated for.

| clamp C | bound B | worst measured | B / worst | violations | golden model shifted by |
|---|---|---|---|---|---|
| 4.581974506 (= max \|w\|) | 1119.613957 | 142.435628452 | 7.860 | **0** of 4704 | 0.000000000 |
| 2.290987253 | 279.903489 | 71.217814226 | 3.930 | **0** of 4704 | 39.943883340 |
| 1.145493627 | 69.975872 | 24.103880709 | 2.903 | **0** of 4704 | 79.159635199 |

The bound is **loose by a factor of 7.86** at the clamp that leaves the golden
model untouched. That is stated rather than hidden: it assumes the worst input,
the worst weight magnitude in the row sum and the worst sign alignment
simultaneously. The tighter clamps give tighter bounds but perturb the golden
model, and that perturbation is reported in the last column so the two effects
are not conflated.

Without clamping, **50 of 4704** single upsets drive the logits to a non-finite
value and the worst finite deviation is 2.786753e+39. No finite bound of any
kind exists for the unclamped network, which is the point.

## 4. Triplication (`validate_triplication.py`)

Raw output: [`triplication_output.txt`](triplication_output.txt). Runtime about 5 to 8 s.

**Single upsets: 144 of 144 recovered exactly by both voters**, with nothing
flagged and nothing wrong (3 float16 parameters x 16 bits x 3 copies). The
float32 single-parameter case gives 96 of 96.

**Double upsets: all 10296 pairs enumerated**, classified by where the two
upsets land:

| case | cases | word voter correct | word voter flagged | bitwise voter correct |
|---|---|---|---|---|
| same copy | 3384 | 3384 | 0 | 3384 |
| different copies, different elements | 4608 | 4608 | 0 | 4608 |
| different copies, same element, **same bit** | 144 | **0** | **0** | **0** |
| different copies, same element, different bit | 2160 | 0 | **2160** | 2160 |

The specification's expectation that triplication "fails on the double-upset
cases" is **true of the case that matters and only of that case**: when two
upsets hit the same stored element at the same bit position in two different
copies, two copies agree on the wrong value and both voters return it with no
indication that anything happened. That is 144 of 10296 pairs (1.399 %) for
this protected set. The other failing class - same element, different bits - is
detected by the word voter in every one of its 2160 cases but corrected in
none, and is silently corrected by the bitwise voter. Reporting the double-upset
result as a single pass or fail would lose all of that, so it is reported as
four numbers.

Over all 10296 pairs the word voter recovers 7992 (77.622 %), flags 2160
(20.979 %) and is silently wrong on 144 (1.399 %).

## 5. Criticality methods (`validate_criticality_methods.py`)

Raw output:
[`criticality_methods_output.txt`](criticality_methods_output.txt). Runtime
about 8 to 18 s depending on contention.

Ground truth: 4704 sites swept on the calibration split and again on the
evaluation split, 9408 forward passes in total.

Mean degradation is concentrated almost entirely in the exponent field and the
sign bit: **98.853 %** of the total damage sits in 9 of 32 bits, a factor of
**86.22** over the 23 mantissa bits, and the single worst bit position is
**bit 30**, the exponent MSB. That is the premise of the exponent-bit heuristic,
measured rather than assumed.

### Degradation avoided per protected byte, test parameters only

59 of 147 parameters (236 bytes), total degradation 51.951744 over 1888 bit
sites. Ties are averaged exactly, so a method that cannot distinguish sites
scores its true expectation under random tie-breaking rather than whatever
`argsort` happens to do.

**Bit-granular cost model** (idealised: protected bits / 8 bytes)

| budget | magnitude | exponent heuristic | learned | oracle | random |
|---|---|---|---|---|---|
| 2 % | 0.498414 | 4.886449 | **6.298675** | 6.532930 | 0.312257 |
| 5 % | 0.364869 | 3.164676 | **3.550208** | 3.616972 | 0.258950 |
| 10 % | 0.362017 | 1.713467 | **1.983816** | 2.031046 | 0.155522 |
| 25 % | 0.369763 | 0.834949 | **0.871761** | 0.872797 | 0.159885 |
| 50 % | 0.331287 | 0.440233 | **0.440244** | 0.440246 | 0.223585 |
| 100 % | 0.220135 | 0.220135 | 0.220135 | 0.220135 | 0.220135 |

**Word-granular cost model** (protected parameters x 4 bytes)

| budget | magnitude | exponent heuristic | learned | oracle | random |
|---|---|---|---|---|---|
| 2 % | **0.547011** | 0.220135 | 0.362759 | 0.547011 | 0.095307 |
| 5 % | 0.367204 | 0.220135 | **0.448575** | 0.540700 | 0.109166 |
| 10 % | 0.350601 | 0.220135 | **0.388010** | 0.461151 | 0.146595 |
| 25 % | **0.372746** | 0.220135 | 0.360320 | 0.398163 | 0.200680 |
| 50 % | **0.333600** | 0.220135 | 0.331964 | 0.335665 | 0.201788 |
| 100 % | 0.220135 | 0.220135 | 0.220135 | 0.220135 | 0.220135 |

### Which method wins, stated without spin

1. **At bit granularity the learned predictor wins the headline metric**, by
   **+0.385533 avoided per byte (+12.2 %)** over the exponent-bit heuristic at
   a 5 % budget, and by +28.9 % at 2 %.
2. **That margin is bought with a full ground-truth sweep.** The heuristic
   costs zero injections and zero training, and already reaches **87.5 %** of
   the oracle at a 5 % budget against the learned predictor's **98.2 %**, which
   costs 2816 injections and forward passes plus the forest fit. Per unit of
   effort the heuristic is far ahead, and for anyone who cannot run a
   ground-truth sweep on their own model it is the method to use.
3. **At word granularity - the granularity real ECC and TMR charge for - the
   exponent heuristic is exactly uninformative.** Its per-byte figure is
   constant at 0.220135, which is precisely the random-selection expectation
   (51.951744 / 236), because it gives every word the same score. The plain
   magnitude baseline beats the learned predictor at 4 of the 6 budgets
   tested.
4. So **no single criticality method wins everywhere**, and which one wins
   depends on the protection granularity. This result has not been retuned to
   produce a cleaner headline.

### Uncertainty output

The predictor exposes the ensemble standard deviation across its 160 trees, in
`log1p` space, as a confidence signal. Measured coverage of the held-out target
within two standard deviations is **0.878178** against the Gaussian reference
of 0.954500 - that is, the forest's tree dispersion **understates** the
predictive spread, which is the known behaviour of this signal. It is reported
as a number rather than described as calibrated. Mean absolute error in
`log1p` space is 4.624248e-03.

Feature importances, top five: `heuristic_log2_relative` 0.664403,
`biased_exponent` 0.118038, `log2_abs_weight` 0.081391, `abs_weight` 0.077793,
`bit_position` 0.031904. The model's largest single input is the exponent-bit
baseline itself; its own contribution comes mainly from `biased_exponent`,
which lets it tell apart the parameters for which an exponent-MSB flip
saturates to infinity, lands on a subnormal, or scales cleanly.

## 6. The inference path (`validate_onnx_path.py`)

Raw output: [`onnx_path_output.txt`](onnx_path_output.txt). Runtime about 7 to 8 s.

| stage | check | result |
|---|---|---|
| scikit-learn -> numpy, float64 storage | max \|probability difference\| | **0.000000e+00** (bit-exact) |
| scikit-learn -> numpy, float32 storage | max \|probability difference\| | 3.585047e-07 |
| numpy -> ONNX, via onnxruntime 1.29.0 | max \|logit difference\| | 6.260614e-06 |
| ONNX initializer location | 4 initializers, 147 elements | all decode bit-exactly from the file bytes |
| in-file bit flip vs `predict_flip` | 6 flips in `W1` | 6 of 6 agree |
| flip twice | 905-byte model | restored byte for byte |

The `onnx` Python package is not installed in this environment, so the protobuf
encoding is implemented directly in `bitflipsim.onnx_io` against the field
numbers of `onnx.proto3`. The check that it is right is that onnxruntime loads
the file and produces the same numbers; a wrong field number would be rejected.

## Checks that did not produce a clean pass, or where a baseline won

| Item | What happened |
|---|---|
| **Exponent-bit heuristic at word granularity** | **The baseline does not merely win, it carries no information at all**: every word gets the same score, so its avoided-per-byte figure is pinned at the random-selection expectation 0.220135 at every budget. A plain magnitude ranking beats it everywhere, and beats the learned predictor at 4 of 6 budgets. Reported, not removed. |
| **Exponent-bit heuristic at bit granularity** | The learned predictor is ahead by 12.2 % at a 5 % budget, but the heuristic reaches 87.5 % of the oracle for zero injections against the learned predictor's 98.2 % for 2816. Whether 12.2 % is worth a ground-truth sweep is the reader's decision, not this repository's. |
| **Uncertainty calibration** | Coverage at two ensemble standard deviations is 0.878178 against a Gaussian 0.954500. The forest's tree dispersion understates predictive variance. Reported as measured; no recalibration layer was added to make the number look better. |
| **Clamping bound tightness** | The bound is never exceeded in 4704 exhaustive cases but is loose by a factor of 7.86. A loose bound that holds is published rather than a tight estimate that does not. |
| **TMR "fails on double upsets"** | Only 144 of 10296 double-upset pairs (1.399 %) actually defeat the word voter silently. 4608 + 3384 are recovered and 2160 are detected. The specification's phrasing is right about the case that matters and wrong as a blanket statement, and the four numbers are published instead of the blanket. |
| **Poisson validity at high upset counts** | At mu = 64 for the reference model the per-bit probability reaches 1.360544e-02, above the 1e-3 threshold, so `poisson_validity` reports `within_validity = False`. The campaign at that point is still run and plotted, with the flag reported, because removing the point would hide the edge of the model's validity. |
| **int8 quantization is not lossless** | On the 300-sample evaluation split it shifts the largest per-class probability by **0.056143** and changes **0 of 300** argmax predictions (`example_bit_position_criticality_output.txt`); on the smaller 120-sample test fixture it changes **1 of 120**, which `tests/test_network.py` pins as a measurement rather than absorbing into a widened bound. Quantization error is not an upset, and the int8 criticality sweep is run on a model that already differs from the float32 one by this much. |
| **Host timing figures** | Inference, clamp-pass and voter latencies vary by tens of per cent between runs on this shared single core. They are labelled as host measurements wherever they appear and are not used to support any claim. |

## Reproducing every number

```bash
cd bitflipsim
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

python -m pytest tests/ -q

python validation/validate_ieee754_layout.py
python validation/validate_poisson_counts.py
python validation/validate_clamp_bound.py
python validation/validate_triplication.py
python validation/validate_criticality_methods.py
python validation/validate_onnx_path.py

MPLBACKEND=Agg python examples/bit_position_criticality.py
MPLBACKEND=Agg python examples/degradation_vs_upset_rate.py
MPLBACKEND=Agg python examples/criticality_methods.py
MPLBACKEND=Agg python examples/mitigation_tradeoff.py
```

Every script is deterministic given its seeds, except the wall-clock timings in
`examples/mitigation_tradeoff.py`, which are host measurements. The example
scripts' committed output is in `example_*_output.txt` in this directory.

Each validation script exits 0 when every check passes and 1 otherwise, and
prints `PASS` or `FAIL` per check followed by a `RESULT:` line.

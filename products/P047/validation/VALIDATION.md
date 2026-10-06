# Validation evidence

**Validation level 2 (research grade). Status: TESTING.**

Every number in this file and in `README.md` was produced by a script in this
directory, run on 2026-10-06 in the build container, with its raw standard output
committed beside it. The commands that reproduce each one are at the end. Nothing
here is quoted from memory, and no page number, equation number or numerical value
is taken from any reference: the four references in `docs/TIMING_MODEL.md` §8 were
checked bibliographically and are named for the reader's benefit only.

Build host: Python 3.13.16, numpy 2.5.3, scipy 1.18.1, matplotlib 3.11.2,
pytest, two CPU cores and 7.8 GiB shared with four other build agents. Runtimes
are therefore upper bounds and vary between runs; each script prints its own.

## What is being validated

The package's claim is one chain, measured end to end:

```
pulse shape -> detector -> S-curve -> K_d -> loop coefficients
            -> jitter variance -> cycle-slip threshold
```

and the validation is that **two independent routes agree where they should and
are reported where they do not**. The routes are a closed-form prediction and a
Monte Carlo of the actual loop; a third route (the exact discrete-time Lyapunov
solution) separates the loop's discretisation error from the detector's noise
model, and a fourth (the same solution driven by the measured noise spectrum)
separates the whiteness assumption from everything else.

## Summary

| # | Check | Reference | Result | Tolerance / gate | Script |
|---|---|---|---|---|---|
| 1 | Early-late gate on a triangular pulse | `K_d = 2` derived in `docs/TIMING_MODEL.md` §2.1 | worst absolute error **1.332e-15** over 4 gate spacings, full 32-pattern ensemble | exact | `validate_scurve_gains.py` |
| 2 | Mueller-Mueller on a triangular pulse, antipodal | `K_d = E[a^2] = 1`, §2.2 | **1.0000000000**, error 6.661e-16, bias 0.000e+00 | exact | `validate_scurve_gains.py` |
| 3 | The same, unipolar OOK | `K_d = 1/4`, §2.2 | **0.2500000000**, error 1.665e-16, bias 0.000e+00 | exact | `validate_scurve_gains.py` |
| 4 | **Mueller-Mueller on a Nyquist raised cosine** | `K_d = 2 cos(pi a) / (1 - 4 a^2)`, §2.3, with the `a = 1/2` limit `pi/2` | worst **2.865e-06** relative over 5 rolloffs (0, 0.25, 0.5, 0.75, 1) | < 1e-05 | `validate_scurve_gains.py` |
| 5 | **PPM slot early-late on a half-sine slot** | `S(eps) = sin(2 pi eps)`, `K_d = 2 pi`, §2.4 | max residual **2.748e-15** over 97 offsets; `K_d` relative error **6.580e-08** | 1e-10 / 1e-05 | `validate_ppm_slot.py` |
| 6 | Early-late gain collapses on a rectangle | a rectangle is flat inside the pulse | `K_d` = **0.000000**, reported as an error by `LoopDesign.from_bandwidth` | identity | `validate_scurve_gains.py` |
| 7 | Mueller-Mueller gain is zero without ISI | §2.2: the gain is `h(+-1)` | **0.000000** on `rect` and on `half-sine` | identity | `validate_scurve_gains.py` |
| 8 | Ill-conditioned gains are flagged, not published | least-squares against central difference | `mueller-muller` on `rc-time` disagrees by **7.5x**; `gardner` on `rect` by **6.7x** | reported | `validate_scurve_gains.py` |
| 9 | **Loop noise bandwidth closed form** | `B_n = theta(1 + 4 zeta^2)/(8 zeta)` against quadrature of `int|H|^2 df` | worst **5.261e-09** relative over 18 points, `zeta` 0.3 to 3 | < 1e-06 | `validate_loop_coefficients.py` |
| 10 | **Discrete poles sit at `exp(s T)`** | exact pole matching, §4.1 | worst **8.278e-09** absolute over 32 designs | < 1e-06 | `validate_loop_coefficients.py` |
| 11 | Coefficient mapping round trip | closed-form inverse, §4.1 | worst **9.778e-11** relative over 15 designs | < 1e-08 | `validate_loop_coefficients.py` |
| 12 | Small-bandwidth approximation cost | exact Lyapunov against `2 B_n sigma_n^2 / K_d^2` | **1.00045x** at `B_n` = 5e-04, **1.0089x** at 0.01, **1.093x** at 0.1 (`zeta` = 1/sqrt2) | reported | `validate_loop_coefficients.py` |
| 13 | **Jitter, Monte Carlo against the measured-spectrum prediction** | 3 detectors x `B_n` in {0.001, 0.005, 0.02}, 300000 symbols each | ratio **0.954 to 1.141**; worst relative deviation **14.08 %** | < 15 % | `validate_jitter_agreement.py` |
| 14 | **The white-noise prediction over-predicts** | same measurement, whiteness assumed | by **15.34x** (early-late), **3.09x** (Gardner), **1.00x** (Mueller-Mueller) | reported | `validate_jitter_agreement.py` |
| 15 | Where the detector noise comes from | open loop, 400000 samples, 20 dB | early-late **94 %** self-noise, Gardner **63 %**, Mueller-Mueller **0 %** | reported | `validate_jitter_agreement.py` |
| 16 | Detector noise autocovariance at lag 1 | measured | early-late **-0.4487**, Gardner **-0.1346**, Mueller-Mueller **-0.0022** | reported | `validate_jitter_agreement.py` |
| 17 | **The prediction fails outside the linear range** | `B_n` = 0.05, Gardner, falling SNR | ratio 1.32 at 20 dB; below 10 dB the measurement saturates at **0.0834 to 0.0842**, i.e. `1/12`, the variance of a uniform symbol | reported, not gated | `validate_jitter_agreement.py` |
| 18 | **Cycle-slip threshold, measured** | 4 loop bandwidths, 200000 symbols per point | slip rate crosses 1e-04/symbol at loop SNR **18.03 / 16.77 / 16.08 dB** for `B_n` = 0.01 / 0.02 / 0.05, i.e. rms **0.063 to 0.079 symbol** | located for 3 of 4 | `validate_cycle_slips.py` |
| 19 | **The Gaussian level-crossing estimate fails** | §5 | low by **6.0 to 16.0 orders of magnitude** on every point with an observed slip | reported, not corrected | `validate_cycle_slips.py` |
| 20 | The mechanism of the failure, measured | held-offset sweep at 8 dB | local S-curve gain falls **1.571 -> 1.246** while `sigma_n^2` rises by **2.2402x** from 0 to 0.45 symbol | reported | `validate_cycle_slips.py` |
| 21 | **Self-noise creates a static lock offset** | §6 | offset / `B_n` = **+0.3825** (early-late, spread 2.2 %) and **+0.5746** (Gardner, spread 7.1 %) over a factor of 8 in `B_n` | spread < 35 % | `validate_self_noise_bias.py` |
| 22 | The offset is not a channel-noise effect | Gardner, 40 dB of SNR swept | `sigma_n^2` rises **0.0590 -> 0.4283**, offset moves **0.005727 -> 0.007269** | reported | `validate_self_noise_bias.py` |
| 23 | The one detector with no self-noise has no offset | Mueller-Mueller on a Nyquist pulse | worst **1.12e-05** symbol, **0.76** standard errors from zero | < 5e-05 and < 3 s.e. | `validate_self_noise_bias.py` |
| 24 | PPM slot gain is order-independent | only the on-slot contributes | identical to **6.580e-08** relative for M = 2, 4, 8, 16 | < 1e-05 | `validate_ppm_slot.py` |
| 25 | PPM jitter chain | 4-PPM, `B_n` 0.002 to 0.02 | Monte Carlo / predicted **0.932 to 1.046** | < 25 % | `validate_ppm_slot.py` |
| 26 | **The PPM duty-cycle rule of thumb is wrong** | `10 log10(M/2)` | measured **0.00 to 0.22 dB** at 20 dB SNR and **0.60 to 2.24 dB** at 6 dB, against 0 to 9.03 dB | reported | `validate_ppm_slot.py` |
| 27 | **PPM fails as symbol errors, not loop slips** | 8-PPM, `B_n` = 0.01 | slot-index error rate **0 -> 0.806** across 18 dB while the loop records **0** slips | reported | `validate_ppm_slot.py` |
| 28 | PPM slot S-curve has no reversal inside half a slot | return-to-zero slot narrower than one slot | **None** - the half-period barrier of symbol timing does not exist here | identity | `validate_ppm_slot.py` |

## Where a prediction lost, or a check did not come out clean

| Item | What happened |
|---|---|
| **The classical white-noise jitter formula** | It is wrong by **15.3x** for a decision-directed early-late gate and **3.1x** for Gardner's detector on a Nyquist pulse at 20 dB per-sample SNR, because both are dominated by detector self-noise and self-noise is not white. It is right to 0.5 % for Mueller-Mueller, which has no self-noise on that pulse. The formula is shipped, and shipped with this measurement attached. A user who plugs a measured TED variance into `2 B_n sigma_n^2 / K_d^2` will over-size their loop by an order of magnitude. |
| **The Gaussian cycle-slip estimate** | Low by **6 to 16 orders of magnitude**. The functional form is wrong, not the scale, so no fitted prefactor is offered and none should be inferred. `validate_cycle_slips.py` §4 measures the real mechanism instead. The usable result is the measured threshold in §2 of that script. |
| **Monte Carlo against the coloured prediction** | Agreement is **not** within the Monte Carlo standard error: the deviation is systematic, growing from about 3 % at `B_n` = 0.001 to 14 % at 0.02, which is up to **9.96 standard errors**. It is the S-curve nonlinearity and the decision errors, neither of which any linearised prediction contains. Published as a 15 % relative tolerance over that range, with the trend stated, rather than as a tight tolerance that would be false. |
| **Static lock offset** | Not predicted by anything in the package. The open-loop S-curve crosses zero at zero offset to **1e-16** for all three detectors, and the closed loop still locks **0.0038 symbol** away for the early-late gate at `B_n` = 0.01. The effect is measured, its proportionality to `B_n` and its independence of channel SNR are measured, and it is **not corrected for**. |
| **Mueller-Mueller is unusable on three of five pulse shapes here** | `K_d` is exactly zero on a rectangle and on a half-sine (no inter-symbol interference to drive it) and ill-conditioned on a time-domain raised cosine (zero slope at `+-T`). The detector that wins every other comparison in this package cannot be used at all unless the pulse has the right behaviour one symbol out. |
| **Gardner's gain on a discontinuous pulse** | The two estimators disagree by **6.7x** because the S-curve is a step at the origin. The package reports both and refuses to pick one. |
| **PPM jitter at `B_n` = 0.002** | Monte Carlo **0.932** of the prediction, below the band the other three points sit in. The squared-gate detector is quadratic in the noise, so its output variance depends on the timing error itself and no linearised prediction contains that. Reported; the point is not dropped. |
| **The slip threshold at `B_n` = 0.005** | Not located. At the lowest SNR swept the loop still produced only one slip in 200000 symbols, which is below what the budget can resolve. Reported as not located rather than extrapolated. |
| **The noise model** | Noise is independent per sample instant. A real matched-filter output has correlated noise between closely spaced early and late gates, which this model does not have. The early-late results are the ones most affected and the README says so. |

## 1. S-curves and the detector gain (`validate_scurve_gains.py`)

Raw output: [`scurve_gains_output.txt`](scurve_gains_output.txt). Runtime about 3 s.

Every S-curve in this package is an **exact** ensemble average: because each pulse
shape has finite support, only a finite window of symbols can reach the detector's
samples, so the average over the data is computed by enumerating all `2**n`
patterns in that window (32 to 4096 patterns depending on the shape) rather than
by Monte Carlo. There is no sampling error in any S-curve, and that is why the
three known-answer families come out at 1e-15.

The gain is measured twice, by a least-squares line through the origin and by a
central difference, and the survey table prints their ratio. A ratio far from 1 is
the package's statement that the S-curve is not differentiable at the origin and
the gain is not usable - it happens on two of the twenty surveyed combinations and
both are named in the output.

## 2. Loop design (`validate_loop_coefficients.py`)

Raw output: [`loop_coefficients_output.txt`](loop_coefficients_output.txt). Runtime about 5 s.

No Monte Carlo appears in this script. The noise-bandwidth expression is checked
against quadrature; the coefficient mapping is checked against the exponential
image of the analogue poles; the inverse mapping is round-tripped; and the
classical small-bandwidth jitter expression is checked against the exact
discrete-time Lyapunov solution, which is how the approximation's validity range
is established without involving any detector.

The quadrature check is the reason this script exists: the first draft of
`noise_bandwidth_closed_form` carried a spurious factor of `2 pi`, every downstream
number was wrong by that factor, and nothing else in the package would have caught
it.

## 3. Jitter prediction against the loop (`validate_jitter_agreement.py`)

Raw output: [`jitter_agreement_output.txt`](jitter_agreement_output.txt). Runtime 42 to 57 s
depending on contention with the sibling jobs on the same two cores.

This is the product's central claim. Four quantities for each detector and each
loop bandwidth: the classical white-noise closed form, the exact discrete loop with
the same whiteness assumption, the exact discrete loop with the **measured**
detector-output autocovariance, and a closed-loop Monte Carlo of 300000 symbols
with a batch-means standard error.

The three divergences are separated by construction. Closed form against exact
discrete isolates the loop's discretisation. Exact discrete against coloured
isolates the whiteness assumption. Coloured against Monte Carlo is what is left:
nonlinearity and decision errors.

Section 4 of that script pushes the loop out of its linear range deliberately. At
`B_n` = 0.05 and 20 dB the ratio is already 1.32; below 10 dB the measured variance
saturates at 0.0834 to 0.0842, which is `1/12` - the variance of a uniform
distribution over a whole symbol. The loop has lost timing entirely, and the
prediction and the measurement are describing different objects. The prediction
curve crossing the measurement near 4 dB is a coincidence of two unrelated curves,
not agreement, and the output says so.

## 4. Cycle slips (`validate_cycle_slips.py`)

Raw output: [`cycle_slips_output.txt`](cycle_slips_output.txt). Runtime 65 to 74 s.

The transition from locked to unlocked is a cliff, not a slope: for
`B_n` = 0.02 the Mueller-Mueller loop shows 0 slips in 200000 symbols at 10 dB,
4 at 9 dB, 16 at 8 dB and 4311 at 7 dB, by which point the measured rms timing
error has gone from 0.061 to 0.239 symbol.

The Gaussian level-crossing estimate predicts 2.6e-18, 5.4e-15, 2.1e-12 and
2.1e-10 per symbol at those four points. It is wrong by 6 to 16 orders of
magnitude and the error grows as the SNR rises, which is the signature of a wrong
tail rather than a wrong constant.

Section 4 measures the mechanism. Holding the timing error at a fixed offset and
measuring the detector at 8 dB per-sample SNR: the S-curve's local gain falls from
1.5708 at zero to 1.2464 at 0.45 symbol, while the detector output variance rises
from 0.3096 to 0.6936, a factor of 2.24. The restoring force weakens and the
disturbance strengthens together. A fixed-barrier Gaussian model contains neither
effect.

The usable output is §2 of that script: the loop SNR at which the slip rate crosses
1e-04 per symbol, which lands between 16.08 and 18.03 dB across a factor of five in
loop bandwidth - an rms timing error of 0.063 to 0.079 symbol. The design rule this
package supports is a jitter rule, not a bandwidth rule.

## 5. The static lock offset (`validate_self_noise_bias.py`)

Raw output: [`self_noise_bias_output.txt`](self_noise_bias_output.txt). Runtime 20 to 25 s.

A type-2 loop forces the time average of the *detector output* to zero, not of the
timing error. Where the detector's self-noise is correlated with the data the loop
has already tracked, the two differ and the loop locks off-centre.

Three measurements identify the mechanism: the offset is proportional to `B_n`
(ratio constant to 2.2 % and 7.1 % over a factor of eight), it does not move across
40 dB of channel SNR while the detector output variance rises by a factor of seven,
and it is zero within its standard error for the one detector with no self-noise.

## 6. PPM slot timing (`validate_ppm_slot.py`)

Raw output: [`ppm_slot_output.txt`](ppm_slot_output.txt). Runtime 20 to 23 s.

The known answer is exact: with a half-sine slot and quarter-slot gates the slot
S-curve is `sin(2 pi eps)` to 2.7e-15 and the gain is `2 pi` to 6.6e-08. The gain
is identical for M = 2, 4, 8 and 16 because only the slot carrying the pulse
contributes, which is the point of §7 of `docs/TIMING_MODEL.md`: what the order
changes is the update rate and the noise per update, not the gain.

Two results contradict common shortcuts. The duty-cycle penalty is **not**
`10 log10(M/2)`: it is 0.00 to 0.22 dB at 20 dB per-sample SNR, where no slot
decision is ever wrong, and 0.60 to 2.24 dB at 6 dB, where the slot error rate
climbs from 0.183 to 0.608 and carries the detector variance with it. And a PPM
slot-timing failure is **not** a cycle slip: at 8-PPM the slot-index error rate
runs from 0 to 0.806 across 18 dB of SNR while the loop records zero slips, because
the receiver re-acquires on whichever slot wins.

The last row of §5 of that script is unlocked and its slot error rate falls rather
than rising. That is index aliasing of a free-running slot clock, not an
improvement, and the output says so rather than deleting the row.

## 7. The worked example (`worked_example.py`)

Raw output: [`worked_example_output.txt`](worked_example_output.txt). Runtime about 8 s.

The example printed in the README, run so that the output in the README is the real
output. It walks the whole chain for Gardner's detector on a truncated Nyquist
raised cosine and then repeats it for a 4-PPM slot clock.

## Test suite

`python -m pytest tests/ -q` from the repository root, counted from the junit XML:
**234 tests, 0 failures, 0 errors, 0 skipped**, in 11 to 15 s on the build host.
The three known-answer families of §2 of `docs/TIMING_MODEL.md` appear as tests
with their derivations in the test comments, and every validity boundary the
validation scripts report has a test pinning it.

## Reproducing every number

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

ruff check src/ tests/ examples/ validation/
python -m pytest tests/ -q

cd validation
PYTHONPATH=../src python validate_scurve_gains.py        # S-curves, K_d, 3 known answers
PYTHONPATH=../src python validate_loop_coefficients.py   # bandwidth, poles, inverse, approximation
PYTHONPATH=../src python validate_jitter_agreement.py    # the central claim
PYTHONPATH=../src python validate_cycle_slips.py         # slip threshold and the estimate's failure
PYTHONPATH=../src python validate_self_noise_bias.py     # the static lock offset
PYTHONPATH=../src python validate_ppm_slot.py            # PPM slot clock
PYTHONPATH=../src python worked_example.py               # the README's example
cd ..

cd examples
PYTHONPATH=../src MPLBACKEND=Agg python scurves.py
PYTHONPATH=../src MPLBACKEND=Agg python loop_design.py
PYTHONPATH=../src MPLBACKEND=Agg python jitter_prediction_vs_simulation.py
PYTHONPATH=../src MPLBACKEND=Agg python cycle_slip_threshold.py
PYTHONPATH=../src MPLBACKEND=Agg python ppm_slot_clock.py
cd ..
```

Every script is deterministic given its seeds (20261006 throughout, with 3, 7 and
11 used where a second independent stream is needed). The only non-deterministic
figures anywhere are the printed runtimes.

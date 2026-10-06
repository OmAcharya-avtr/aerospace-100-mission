# Model card — softdecode LLR corrector

**Status: TESTING** · Validation level 2 (research grade) · Apache-2.0 ·
© 2026 OPTIMA Organisation

**This model is not certified for operational flight use.**

## Problem

A soft-decision demapper for an on-off-keyed optical link needs the normalised
irradiance `h`. A receiver only has an estimate `h_hat`. Substituting `h_hat`
into the exact known-CSI log-likelihood ratio — which is what almost every
implementation does — produces LLRs that are wrong in a way that is not
symmetric: an estimate that is too high both shifts the hard-decision threshold
`a h_hat / 2` upward and scales the LLR magnitudes up, so the decoder is handed
confident errors. At Eb/N0 = 8 dB with a 2 dB bias the decoded bit error rate of
the shipped LDPC code rises from 1.917e-03 to 8.415e-02, a factor of 43.9, and
the generalised mutual information of the LLRs falls from +0.737 to +0.391
bit/bit (`validation/csi_mismatch_output.txt`).

The model's job is to map observable receiver quantities to a better-calibrated
LLR, learning only from pilot bits.

## What the model is allowed to know

| method | information used |
|---|---|
| `known_csi` | the **true** `h`. Unreachable upper bound. |
| `csi_aware_exact` | `(y, h_hat)` **plus the error-model parameters**. Analytic, and Bayes-optimal given the observables. |
| `csi_aware_maxlog` | the same, with the posterior average replaced by a max. The realistic baseline. |
| `plugin` | `(y, h_hat)` treated as if `h_hat = h`. |
| `plugin_scaled`, `plugin_clipped`, `plugin_scaled_clipped` | plug-in LLRs with one or two free parameters tuned on a disjoint split. |
| `learned` (this model) | `(y, h_hat)` **plus pilot bits**. Never told the bias, the jitter, the correlation, the scintillation index, or `h`. |

## Baselines first, model second

The deterministic path was built and measured before the model existed:
`softdecode.llr` (exact LLRs, max-log, clipping) and `softdecode.csi`
(posterior-aware LLRs) are complete without `softdecode.corrector`, and their
numbers are in `validation/awgn_limit_output.txt`,
`validation/maxlog_clipping_output.txt` and `validation/csi_mismatch_output.txt`.
The model is benchmarked against all of them on the same held-out split.

## Architecture

`sklearn.ensemble.RandomForestClassifier`, 40 trees, `max_depth = 14`,
`min_samples_leaf = 200`, `random_state = 20261006`, `n_jobs = 1`. It predicts
`P(bit = 1 | features)`; the corrected LLR is `log p0 - log p1`, saturated at a
clip level chosen on the tuning split.

PyTorch is not available in the build container, so the model is a forest and
not a neural network. No claim is made that a network would not do better.

Features, all computable at the receiver:

| # | feature | units |
|---|---|---|
| 0 | `y / sigma` | dimensionless |
| 1 | `h_hat` | dimensionless |
| 2 | `log h_hat` | dimensionless |
| 3 | `a h_hat / sigma` | dimensionless |
| 4 | plug-in LLR | nats |
| 5 | `a / sigma` | dimensionless |

Feature 4 is included on purpose: the naive answer is handed to the model, so
the model can only win by adding information the naive answer lacks. Measured
importances put feature 4 first (0.569) and feature 0 second (0.400), with the
other four summing to 0.031 (`validation/corrector_output.txt`).

## Dataset and splits

See `DATASET_CARD.md`. Three disjoint seeded splits from
`softdecode.datasets`, `BASE_SEED = 20261006`:

* **train** — 5 Eb/N0 points (2, 4, 6, 8, 10 dB) × 500 blocks × 96 channel
  bits = 240 000 samples;
* **tune** — 600 blocks at the reporting Eb/N0. Every free parameter of every
  method, including this model's output clip level, is chosen here;
* **report** — 1500 blocks (75 000 information bits). Every published number.

Nothing is tuned on the reporting split, and the splits share no channel
realisation.

## Metrics

Three, because they disagree:

* **decoded bit error rate** after 20 iterations of sum-product decoding of the
  shipped length-96 rate-50/96 LDPC code, with its binomial standard error;
* **generalised mutual information** of the LLRs, which penalises
  over-confidence and is the quantity that exposes a miscalibrated demapper;
* **LLR RMS error** against the true-CSI LLR, reported for completeness and
  because it is the metric that misleads.

## Results, report split, Eb/N0 = 8 dB

Decoded bit error rate (`validation/corrector_output.txt`):

| method | bias +2 dB, jitter 1 dB | bias −2 dB, jitter 1 dB | stale, rho = 0.80 |
|---|---|---|---|
| `known_csi` (upper bound) | 2.773e-03 | 2.773e-03 | 2.773e-03 |
| `csi_aware_exact` | **5.827e-03** | **5.827e-03** | **8.213e-03** |
| `csi_aware_maxlog` | 2.199e-02 | 2.199e-02 | 2.708e-02 |
| `plugin` | 1.855e-01 | 1.488e-02 | 2.664e-02 |
| `plugin_scaled` (tuned alpha) | 1.540e-01 | 1.488e-02 | 2.241e-02 |
| `plugin_clipped` (tuned L_max) | 1.573e-01 | 1.495e-02 | 2.140e-02 |
| `plugin_scaled_clipped` (both tuned) | 1.521e-01 | 1.495e-02 | 1.977e-02 |
| `learned` (this model) | 6.533e-03 | 6.280e-03 | 8.707e-03 |

**The honest summary.** The learned corrector beats every realistic baseline —
28.4× better than the plug-in receiver and 3.37× better than max-log with the
estimated CSI in the over-estimating case — and it **loses to the analytic
CSI-aware LLR in all three regimes**, by 1.12×, 1.08× and 1.06×. It is also
2.36× to 3.14× away from the true-CSI bound. The engineering conclusion is
therefore not "train a model": it is **marginalise over your channel-estimate
error**, which is closed form for a lognormal channel and costs one
Gauss-Hermite quadrature. The model is the right answer only when the error
model is unknown and pilot bits are available, which is the case it was built
for.

**The plain scalar rescaling result, reported because it is the obvious
competitor.** Tuned on a disjoint split, it recovers 17.0 per cent of the
plug-in's excess BER in the over-estimating case (1.855e-01 → 1.540e-01) and
nothing at all in the under-estimating case. The reason is structural: the
plug-in LLR is `(a**2 h_hat**2 - 2 a h_hat y) / (2 sigma**2)`, whose zero
crossing sits at `y = a h_hat / 2`. A positive scalar cannot move a zero
crossing, and the threshold error is most of the damage — at a +4 dB bias,
99.1 per cent of the raw hard-decision errors are missed ones
(`validation/csi_mismatch_output.txt`, section 5). Clipping, and clipping plus
rescaling, do no better for the same reason.

## Uncertainty output

`LlrCorrector.predict(features, with_uncertainty=True)` returns the mean and
the standard deviation of the per-tree LLRs, both saturated at the same clip
level. Measured on the report split (`validation/corrector_output.txt`):

* the dispersion correlates 0.711 with `|learned − clipped analytic reference|`,
  so it does track where the model's own output is unusual;
* it correlates **−0.241** with the hard-decision error indicator, and the
  error rate **decreases** monotonically across its quintiles. As a standalone
  risk score it therefore points the wrong way: the trees disagree most where
  the LLR magnitude is large, which is where decisions are reliable;
* normalising by `1 + |L|` removes that scale dependence but does not produce a
  monotone risk ordering either;
* the quantity that does order risk on this split is `|L|` itself.

All of that is published as measured. The dispersion is a model-disagreement
diagnostic, not a calibrated interval, and should not be used as one.

## Training procedure and reproducibility

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python validation/validate_corrector.py
```

Deterministic given the seeds printed in the script's own output: dataset and
splits `20261006`, forest `random_state = 20261006`, LDPC construction seed
`20261006`. No model binary is committed; `LlrCorrector.save`/`load` exist for
users who want one, but the repository ships the regeneration path instead so
there is nothing to drift.

## Compute used

2 CPU cores and 7.8 GiB of RAM, **shared with four other build agents and a
concurrent release-gate run**. Measured in that environment: one forest fit
takes about 12 s on 240 000 samples, so no single training run comes near the
three-minute budget. The whole of `validate_corrector.py`, which fits three
models, tunes four free parameters by decoded BER and evaluates eight demappers
on three regimes, took **175 s and 276 s** on two runs of the same deterministic
code — the spread is contention, and the output was byte-identical. Every figure
in this repository except these runtimes is deterministic; nothing is compared
against them.

## Failure cases

1. **Outside its training Eb/N0 range.** The model is trained at 2–10 dB and
   `a / sigma` is a feature; a forest extrapolates by holding the nearest leaf
   constant, so behaviour outside that range is not characterised.
2. **A different error model.** The model is trained for one mismatch regime at
   a time. It is not told the regime, but it is fitted on data from it; a
   receiver whose estimator behaves differently from its pilots needs refitting.
3. **A different fading law or scintillation index.** Not characterised; the
   scintillation index is not a feature.
4. **Clip level.** The output is saturated at the tuned level (8 to 20 in the
   measured runs). An iterative decoder that needs larger magnitudes will not
   get them.
5. **It does not beat the analytic answer.** In every regime measured, the
   closed-form CSI-aware LLR is better. If the error model can be characterised,
   use that instead.

## Ethical and safety limits

This model corrects demapper outputs in a simulation of an optical channel. It
is research-grade. It is **not flight-qualified, not certified, and not
approved for operational aerospace use**, and nothing in it substitutes for
link-budget analysis, hardware testing or a qualification campaign. No personal
data is involved; all data is synthetic and generated by the committed code.

# Dataset Card — BeamTwin Surrogate Training Set

**Dataset:** `data/surrogate_dataset.npz`
**Version:** 0.1.0 · **Date:** 2026-08-06 · **License:** AGPL-3.0-only
**Owner:** OPTIMA Organisation

> Derived from a simplified physics model. **Not certified for operational
> flight use** and not a substitute for measured link data.

---

## 1. Summary

4000 synthetic free-space-optical link scenarios, each labelled with a fade
probability computed by BeamTwin's own seeded Monte Carlo simulation
(50 000 samples per scenario). Used to train the fade-probability surrogate
(`MODEL_CARD.md`).

| Property | Value |
|---|---|
| Rows | 4000 |
| Features | 5 (float64) |
| Target | `log10(P_fade)`, floored at −4.0 |
| File size | 193 KB (`.npz`, committed — well under the 1 MB threshold) |
| Generation time | 12.3 s, 2 CPU cores |
| Master seed | 42 |

## 2. Source — synthetic, self-generated

**There is no external data source.** Every row is produced by
`beamtwin.surrogate.generate_dataset`, which:

1. Draws a scenario from the prior in §3 using `numpy.random.default_rng(42)`.
2. Runs `sample_received_power_dbm` with 50 000 samples and a child seed drawn from the same generator.
3. Counts fades to obtain `P_fade`, then stores `log10(max(P_fade, 1e-4))`.

The labels are therefore **Monte Carlo estimates from the twin's own physics
model**, not measurements. Their accuracy is bounded by the model's validity
(`validation/VALIDATION.md` §6.5) and by Monte Carlo sampling noise (§5).

Regeneration is exactly deterministic:

```bash
python scripts/generate_dataset.py     # -> data/surrogate_dataset.npz
```

## 3. Sampling prior

| Parameter | Distribution | Range |
|---|---|---|
| Range | log-uniform | 1 – 20 km |
| Cn² | log-uniform | 1e-16 – 5e-14 m^−2/3 |
| Atmospheric attenuation | uniform | 0 – 3 dB/km |
| Deterministic margin | uniform | −5 – 25 dB |
| Pointing jitter / divergence | uniform | 0 – 0.5 |

Fixed across all rows: wavelength 1550 nm, transmit power 20 dBm, optics
efficiencies 0.8/0.8, beam waist 2 cm, receive aperture radius 5 cm, zero
static pointing bias.

**Margin is sampled directly** (by back-solving the receiver sensitivity from
the computed received power) rather than sampling sensitivity independently.
An earlier version sampled sensitivity over −45…−25 dBm and produced 76 % of
rows at the probability floor — a badly imbalanced target. Sampling margin
directly reduced the floor fraction to 23.8 %.

## 4. Realised feature distribution

Measured on the committed file:

| Feature | Min | Max |
|---|---|---|
| `log10_range_m` | 3.0001 | 4.3010 |
| `log10_cn2` | −15.9995 | −13.3019 |
| `jitter_ratio` | 0.0003 | 0.4999 |
| `attenuation_db_per_km` | 0.0001 | 2.9985 |
| `margin_db` | −4.9996 | 24.9984 |

Target `log10 P_fade`: min −4.0, max 0.0.
Percentiles: p10 = −4.00, p25 = −3.75, p50 = −1.24, p75 = −0.23, p90 = −0.04.
**23.8 % of rows sit exactly at the −4.0 floor.**

## 5. Label noise

Labels are binomial estimates from 50 000 samples, so they carry sampling
noise that varies enormously across the range:

| True P_fade | Expected relative std of label | Notes |
|---|---|---|
| 1e-1 | 1.3 % | negligible |
| 1e-2 | 4.5 % | small |
| 1e-3 | 14 % | significant |
| 1e-4 (floor) | 45 % | at the resolution limit |

This noise floor is a hard limit on achievable surrogate accuracy near small
probabilities and is one reason the model's ensemble spread under-estimates
total error (`MODEL_CARD.md` §7).

## 6. Splits

Train/test 80/20 by uniformly random permutation with split seed 123
(3200 / 800). Performed in `scripts/train_surrogate.py`; the same seed
reproduces the split used for every metric reported in `MODEL_CARD.md` and
`validation/VALIDATION.md`. No validation split — no hyperparameter search
was performed, so none was needed.

## 7. Limitations and biases

1. **Synthetic only.** No measured or field data. Systematic errors in the physics model are reproduced faithfully in the labels and are invisible to any metric computed on this dataset.
2. **Single wavelength.** 1550 nm only; wavelength is not a feature. The dataset cannot support predictions at other wavelengths.
3. **Fixed aperture and transmitter geometry.** One beam waist, one aperture size, one transmit power. `jitter_ratio` and `margin_db` absorb some of this variation, but aperture-to-beam ratio effects are not sampled.
4. **Probability floor at 1e-4.** Rare-fade behaviour — precisely the regime a link designer cares most about — is censored, and 23.8 % of rows are censored observations treated as ordinary regression targets. This is a modelling simplification, not a statistically correct censored-regression treatment.
5. **Uniform priors are not operational priors.** The parameter distribution is chosen for coverage of the design space, not to reflect the frequency of real deployments. Metrics averaged over this dataset are not weighted by real-world relevance.
6. **Zero static pointing bias.** Only random jitter is sampled; systematic boresight error is absent from training.
7. **Independent rows.** No temporal, spatial, or site-level correlation structure — real link campaigns produce strongly correlated observations.

## 8. Maintenance and provenance

- Generated by `scripts/generate_dataset.py` at commit time of v0.1.0 with seed 42.
- The `.npz` is committed (193 KB) for exact reproducibility of the published metrics; it can be deleted and regenerated identically at any time.
- Any change to `beamtwin.channel`, `beamtwin.budget`, or the sampling prior invalidates the dataset. Regenerate and retrain, then update `MODEL_CARD.md` §6 and `validation/VALIDATION.md` §5 with the new measured numbers.

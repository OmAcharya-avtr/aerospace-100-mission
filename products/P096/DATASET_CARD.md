# Dataset card — handoverloss synthetic handover corpus

**Everything in this corpus is SYNTHETIC. No measured handover, no measured
acquisition time and no measured optical link appears anywhere in this
repository.** Validation level 2: the evidence is agreement between a closed
form, an independently derived bound and a simulated trace.

Research-grade; not flight-qualified, not certified, and not approved for
operational aerospace use.

## Files

| File | What it is |
|---|---|
| `data/generate_corpus.py` | The generator. Deterministic from one root seed. Committed because the reproducibility, not the CSV, is the evidence. |
| `data/corpus/handovers.csv` | 1536 handover events, 768 905 bytes. |
| `data/corpus/manifest.json` | Shape, seeds, dominance counts and the CSV's sha256, which `handoverloss.corpus.load_corpus` checks on every load. |

Regenerate with, from the repository root:

```bash
python3 data/generate_corpus.py
```

Measured this session: 1536 events in **8.119 s** of wall time, declared budget
60 s. sha256
`3a0343a6c2c51a91eba0679856700f5495d8814c3bcdc94e6b710ba300c59706`.

## Shape

16 station pairs × 8 time epochs × 12 handovers per cell = **1536 events**.

- **Station labels are synthetic** (`GS01` upward) and carry no relation to any
  real ground station. Each pair has a fixed configuration: rates, buffer,
  protocol restart class, egress ramp, acquisition statistics, nominal
  elevation.
- **Each epoch** shifts the atmospheric and operational conditions: turbulence,
  beacon fade margin, sky-cover index.
- **Each event** draws its own buffer occupancy at the break, acquisition time,
  restart jitter, per-slot burst sequence and per-slot service jitter.

Accounting window 700 s at a 0.25 s slot width, so 2800 slots per event.

## Declared generator ranges

**Not one of these values is a measurement reported by any source cited anywhere
in this repository.** They are declared parameters chosen to span a plausible
operating envelope and to make all three loss mechanisms reachable.

| Quantity | Range | Note |
|---|---|---|
| Steady downlink rate `R_d` | 0.45 to 2.6 Gbit/s | Uniform per pair |
| Utilisation `rho = R_s / R_d` | 0.039 to 0.970 measured | **Stratified**, see below |
| Buffer capacity `B` | 4 to 70 Gbit | Uniform per pair |
| Occupancy fraction `q0 / B` | Beta(4.0, 1.6) per event | Skewed towards full: an operator hands over when the buffer already holds a pass's worth of data |
| Declared restart `T_rst` | 2.07 to 83.28 s measured | **Stratified** over a 7-point grid, ±15 per cent jitter |
| Declared ramp `T_ramp` | 4 to 170 s nominal, ±15 per cent | **Stratified** over a 7-point grid |
| Acquisition median | 6 to 58 s | Uniform per pair |
| Acquisition log-sigma | 0.25 to 0.80 | Uniform per pair |
| Acquisition attempt window | 70 to 150 s | Uniform per pair |
| Per-attempt abort probability | 0.01 to 0.55 | Derived from the epoch's covariates by a declared logistic link |
| Nominal elevation | 12 to 78 degrees | Uniform per pair |
| `log10 Cn2` | −16.5 to −13.0 | Declared covariate, m^(−2/3) |
| Beacon fade margin | 1.5 to 12.0 dB | Declared covariate |
| Sky-cover index | 0 to 0.85 | Declared dimensionless covariate |
| Source burst log-sigma | 0.45 | Unit-mean multiplicative, so the mean source rate is preserved |
| Service jitter log-sigma | 0.12 | Unit-mean multiplicative |

## Dataset design decisions, declared rather than implied

**The utilisation, ramp and restart are stratified over grids rather than drawn
independently.** A first pass drew both rates independently and produced a
corpus in which the acquisition mechanism dominated **1016 of 1536** events and
the drain mechanism **90**. A corpus in which one answer is nearly always right
cannot distinguish a predictor from a constant. The grids were introduced and
the mix became 762 / 516 / 258. **The change was to the GENERATOR only; no
baseline and no model was touched, and the first-pass numbers are recorded here
rather than deleted.**

**The abort probability's link function has declared coefficients.** The
DIRECTION of each term is what the physics supports — stronger turbulence and
lower elevation mean deeper and longer fades (Andrews and Phillips 2005), a
larger fade margin means fewer aborts, obscuration blocks the beacon outright
(Kim, McArthur and Korevaar, SPIE 4214, 2001). The COEFFICIENTS are arbitrary
and declared in `data/generate_corpus.py`.

**12 acquisition draws of 1536 were capped to the accounting window.** An
acquisition tail longer than `H - T_rst - T_ramp` would leave the excursion
outside the window, so it is capped and the count is in the manifest.

## Columns

Three groups, enforced by `handoverloss.corpus`:

- **18 feature columns** — pre-handover observables only. No realised duration
  and no realised loss is a feature, which
  `check_feature_causality()` asserts by name and
  `tests/test_corpus.py::test_no_feature_is_a_post_handover_quantity` tests.
  That restriction is what makes the learned component a predictor rather than a
  measurement.
- **3 target columns** — `loss_acquisition_bits`, `loss_restart_bits`,
  `loss_drain_bits`: the ground-truth per-mechanism loss from the trace.
- **13 provenance columns** — identity, epoch, the realised durations, the
  produced, delivered and final-occupancy volumes, the conservation total and
  the dominant mechanism. The realised durations live here and not among the
  features.

## Ground-truth composition

| Dominant mechanism | Events | Fraction |
|---|---|---|
| acquisition | 762 | 0.496094 |
| restart | 516 | 0.335938 |
| drain | 258 | 0.167969 |

**Baseline 1 predicts `acquisition` for every handover by construction, so its
dominance accuracy on the whole corpus is exactly 0.496094.** That is the floor
any predictor must beat.

Per-event dominance by regime, from
`validation/validate_decomposition_residual_output.txt`:

| Regime | n | acquisition | restart | drain |
|---|---|---|---|---|
| low utilisation (`rho < 0.25`) | 576 | 0.6667 | 0.2795 | 0.0538 |
| high utilisation (`rho > 0.75`) | 384 | 0.3672 | 0.3203 | 0.3125 |
| tight buffer (headroom < 10 s) | 559 | 0.4919 | 0.3220 | 0.1860 |
| ample buffer (headroom > 120 s) | 150 | 0.9133 | 0.0800 | 0.0067 |
| slow restart (declared > 30 s) | 672 | 0.3735 | 0.6190 | 0.0074 |
| fast restart (declared < 8 s) | 480 | 0.7000 | 0.0146 | 0.2854 |

## Known limitations of this corpus

- **It is synthetic and generated by the mechanism baseline 2 nearly reproduces
  in closed form.** A closed-form win on this corpus is close to the best case
  for the closed form and says nothing about measured handover data. This is the
  single most important caveat and it is repeated in `MODEL_CARD.md` and
  `README.md`.
- **`headroom_s` and `rho` are structurally anti-correlated.** The generator
  draws the buffer capacity independently of the source rate and
  `headroom_s = (B - q0) / R_s`, so a high utilisation implies a short headroom
  in seconds. The cell "high utilisation AND ample buffer" is **empty** and
  `validation/validate_decomposition_residual.py` reports it as structurally
  unmeasurable rather than dropping it. A reader wanting that cell must
  regenerate with the capacity scaled to the source rate; nothing here measures
  it.
- **Pair-level features are constant within a pair.** There are 16 distinct
  configurations, which is why a tree-based learner cannot interpolate to a
  seventeenth. See `MODEL_CARD.md` under Failure cases.
- **216 of 1536 events have zero loss.** The buffer absorbed the whole handover.
  Those events are kept, because a corpus of only expensive handovers would
  overstate how often a handover costs anything; but a relative residual or a
  ratio computed over them is meaningless, and
  `validation/validate_baselines_against_truth.py` says so where it restricts a
  figure to the 1320 events with a loss above 1e6 bits.
- **One accounting window, one slot width.** 700 s and 0.25 s. The slot-width
  term is measured (median relative 8.514534e-04) but not varied.
- **No source model other than unit-mean lognormal per-slot burstiness.** A
  genuinely on-off or self-similar source is not represented.

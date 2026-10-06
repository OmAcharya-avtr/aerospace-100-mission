# Batch 05 Readiness Report — P041–P050

**Date:** 2026-10-06 · **Session:** unattended scheduled run, device-bound (v5)
**Authorization for publication:** ADR-017 standing authorization, conditional on `scripts/release_gate.py` exiting 0.
**No per-batch approval is cited and none exists.** No row was written into `APPROVAL_LOG.md`, no product was set to `APPROVED`, and no words are attributed to the owner. ADR-016 is in force.

## Composition, verified programmatically rather than asserted

| Property | Mandate | Delivered |
|---|---|---|
| flagship / medium / compact | 2 / 3 / 5 | **2 / 3 / 5** |
| AI-enabled | ≥ 7 of 10 | **7** |
| Validation levels | against remaining gap | **0 × L1, 7 × L2, 3 × L3** |
| `hardware_pending` | start L4 groundwork | **2** (P041, P043) |
| Domains | declared list only | `optical-modulation`, `optical-link-engineering`, `satellite-communications` |

**L1 was deliberately frozen at 8 of 10.** Two slots remain against a target of 10 with six batches left. Spending them here would overshoot the distribution and force later batches to invent shallow products to hit a number. This is the first batch to hold that line and it answers the previous checkpoint's `next_action` directly.

**L3 arithmetic, checked:** 9 built before this batch, 3 per batch, six batches including this one → 9 + 18 = 27 ≥ 25, closing with two slots of slack. The previous checkpoint's concern that L3 "will not reach 25" assumed a lower rate. Flagship validation depth does not need raising.

## The products

| ID | Name | Package | Class | L | AI | What it does that the mature libraries do not |
|---|---|---|---|---|---|---|
| P041 | CodedFade | `codedfade` | flagship | 3 hw-pending | yes | post-decoding BER/FER against **interleaver depth versus fade correlation time**, with the latency and memory cost of each depth |
| P042 | AcmPilot | `acmpilot` | flagship | 3 | yes | rate adaptation **under feedback delay**, with a clairvoyant upper bound and mis-selection split into its two directions |
| P043 | PhotonCount | `photoncount` | medium | 3 hw-pending | yes | dead time **and** afterpulsing together, where the textbook inversions each assume one effect alone |
| P044 | ApertureDiv | `aperturediv` | medium | 2 | yes | diversity order under **correlated** apertures, and the measured cost of the independence idealisation |
| P045 | ArqLongHaul | `arqlonghaul` | medium | 2 | yes | ARQ goodput on a **bursty** channel at long RTT, where the independent-error formulae break |
| P046 | InterleaveKit | `interleavekit` | compact | 2 | no | burst-dispersion metrics and latency/memory cost accounting per construction |
| P047 | SlotSync | `slotsync` | compact | 2 | no | the measured chain from S-curve gain to jitter to cycle-slip rate in one place |
| P048 | SoftDecode | `softdecode` | compact | 2 | yes | the **mismatched-CSI** LLR penalty as a measured quantity, in both directions |
| P049 | LinkOutage | `linkoutage` | compact | 2 | yes | fade statistics with the definitional choices written down, which is where tools disagree |
| P050 | CodeRateOpt | `coderateopt` | compact | 2 | no | availability-constrained rate choice **with sensitivity**, so a knife-edge optimum is distinguishable from a flat one |

## PyPI names

All ten free, checked with `pip index versions`: `codedfade`, `acmpilot`, `photoncount`, `aperturediv`, `arqlonghaul`, `interleavekit`, `slotsync`, `softdecode`, `linkoutage`, `coderateopt`.

**The negative result was controlled.** The same command was run against nine packages that must exist — `galois`, `reedsolo`, `commpy`, `komm`, `scikit-dsp-comm`, `pyldpc`, `itur`, `sgp4`, `numpy` — and returned a version list for every one. Without that control a "not found" could equally have meant a broken transport, and the ten absences would have meant nothing.

## A premise in this batch's own specification was wrong, and is corrected rather than repeated

The Batch 05 specification asserted that `komm` ships interleaving primitives. **It does not.** The P046 agent downloaded the 0.36.0 wheel and read it: 130 `.py` files, and a case-insensitive search for `interleav` returns four hits, every one a local variable named `interleaved` inside `_quantization/ScalarQuantizer.py`. `komm` ships no interleaver at all.

Three further agents independently inspected the same wheel for their own alternatives tables and corroborated it from different angles: no fading channel and no diversity combining (P044), no ARQ or HARQ of any kind (P045), and no LLR function, no `lognormal`/`fading`/`csi`/`max_log` symbol anywhere in 143 modules (P048). `scikit-commpy` 0.8.0 ships exactly one interleaver — `RandInterlv`, 74 lines, a uniform Mersenne-Twister permutation with no spread, dispersion, latency or memory metric.

The method that found this — **unpack the wheel and read it, never trust the description** — is the reusable result, and it is now how every alternatives table in this batch was built.

## Cross-checks

The specification for these was rewritten this session to fix a Batch 04 defect: the P033/P039 p50 check compared a wall-clock median on a contended core against the median of a cost model, which can only ever disagree. The governing rule is now that a cross-check compares a quantity **both implementations compute from the same seeded sample path**, by a definition written in the specification, and **no cross-check compares a wall-clock measurement to a model output.**

### X1 — P041 ↔ P049 — **PASS**, and the rewrite is vindicated

Independent implementations, on the same specified seeded series (lognormal, AR(1) in log-amplitude, seed 41 drawn in one call of 2 000 000, `fs` 1 MHz, τ 2.0e-4 s, SI 0.6, threshold amplitude 0.6). P049 was explicitly instructed not to read P041, and was given the series configuration and an input fingerprint but **not** the two numbers being compared.

| Quantity | P041 | P049 | Relative difference | Tolerance | Verdict |
|---|---|---|---|---|---|
| level-crossing rate | 8191.000000000000 Hz | 8191.004095502048 Hz | **5.000002e-07** | 2 % | AGREE |
| mean fade duration | 1.549737516786717e-05 s | 1.5497375167867175e-05 s | **2.186262e-16** | 2 % | AGREE |

Both independently found 16 382 down-crossings, 16 382 complete fades, 1 censored run and an outage fraction of 0.1269405. P049 reproduced the input series to 4.44e-16 on mean irradiance.

**The entire residual gap is accounted for exactly, not approximately.** P041 divides the crossing count by `N/fs` (2.000000 s), P049 by `(N-1)/fs` (1.999999 s). 16382/2.000000 = 8191.000000 and 16382/1.999999 = 8191.004095502048, both reproduced by the coordinating session. Neither is wrong; they are two record-duration conventions differing by one sample in two million.

P049 additionally published a definitional-sensitivity table showing which choices *could* have caused a disagreement — the single-sample-fade rule moves the rate by −31.7 % and the mean by +43.5 %, and nothing else moves either by more than 6.5 %. That table is what makes a future disagreement diagnosable in minutes instead of a night.

### X2 — P044 ↔ P010 — **DISAGREE, and the defect is in this session's specification, not in either product**

| | Value |
|---|---|
| P044 sample BER (BPSK, single aperture, 10 dB, SI 0.30, 2e7 bits, seed 44044) | 6.039000000e-04 ± 5.493338260e-06 |
| P010 BERBench `analytic_ber("bpsk", 10.0, channel="lognormal", sigma_i2=0.30)` | 7.5189051013828155e-03 |
| Ratio | **12.4506** |

The cause is a convention the specification failed to pin down: **whether the lognormal variate multiplies signal amplitude or signal power.**

- P010's `_ber_bpsk_cond` is `Q(I·sqrt(2γ))` — irradiance multiplies **amplitude**, so electrical SNR ∝ I². This is the standard intensity-modulation / direct-detection convention, where photocurrent follows optical power and electrical SNR follows its square.
- P044 uses `γ_inst = (Eb/N0)·I` — irradiance multiplies **power**, SNR ∝ I. This is the RF `|h|²`-gain convention.

The coordinating session evaluated both conventions by independent Gauss-Hermite quadrature over the same unit-mean lognormal:

| Convention | Quadrature | Matches |
|---|---|---|
| SNR ∝ I² (amplitude) | 7.518905101382814e-03 | **P010** to 2.3e-16; **P048**'s own reference to 5.1e-11 |
| SNR ∝ I (power) | 6.002225662486640e-04 | **P044**'s own reference to ~1e-10 |

**Every product is internally correct and each agrees with its own analytic reference to around 1e-10.** The disagreement is entirely the undeclared convention.

**X2 re-run with the convention matched — P048 ↔ P010 — PASSES:** P048's comparand 7.549000000e-03 ± 6.120463e-05 against P010's 7.5189051013828155e-03 gives **z = +0.4917** against a 3-binomial-SE tolerance. Two independently written implementations, one of them published five batches ago, agreeing to within half a standard error.

**Nothing was retuned and no product was changed to make a number match.** What is recorded instead is a real portfolio finding for the owner: **P044 models SNR as linear in irradiance while P010 and P048 model it as quadratic.** P044 documents its choice explicitly and lists this as convention C1 among five a disagreement could trace to, so it is disclosed rather than hidden, and it is defensible as a modelling choice. But it means P044's absolute BER and outage figures are **not numerically comparable** with the rest of the portfolio. Harmonising the convention across the portfolio is an owner decision, not something an unattended session should do to a flagship-adjacent product at 07:00 with no time to re-validate the consequences.

### X3 — P048 internal ordering plus the P010 comparand — **PASS**

Soft-decision BER ≤ hard-decision BER at **all seven** stated Eb/N0 points on the same channel realisations, both decoders exhaustive ML over the same 16 codewords and handed the same LLR array (the hard decoder seeing only its sign). Ratio rises 1.409 → 25.569 from 0 to 12 dB. Pairing evidence at 12 dB: 693 blocks soft-right/hard-wrong against 5 the other way. No violation to report.

## Honest negatives — seven of them, published as found

ADR-011 governs: a baseline beating the learned model is research evidence, not a failure to conceal. **Nothing below was retuned, softened or removed.** These bring the portfolio's recorded count to sixteen.

1. **P042 — the learned predictor lost outright.** A non-learned analytic AR(1) MMSE predictor beat the learned gradient-boosting model at **all five** feedback delays on the lognormal channel (1.9759 vs 1.9737 bit/symbol at 1 ms, through 1.6485 vs 1.6478 at 20 ms), never separated by two combined standard errors. The cause is structural: the lognormal channel is AR(1) in log-amplitude *by construction*, so the MMSE predictor is exactly linear and there is nothing non-linear left to learn. Published as the result, and the non-learned predictor is the one the README tells you to ship.
2. **P042 — the confidence gate matters 30–140× more than the predictor.** Gating aggressive rate choices on predictive spread is worth +0.1537 to +0.2842 bit/symbol; the choice of predictor is worth 0.002–0.005. The useful engineering finding is the opposite of the one the product was designed to demonstrate.
3. **P045 — two integers beat a random forest.** A tuned fixed HARQ schedule, chosen by grid search on 40 tuning seeds, beat the learned forest in both channel regimes (+1.66 % cost, z = 2.27 long-burst; +3.03 %, z = 23.50 short-burst). The one-parameter escalating heuristic also beat it in the short-burst regime. The forest's own uncertainty output explains why: at 63.3 % of visited states it cannot distinguish its two best actions.
4. **P048 — the analytic answer beat the learned corrector in every regime.** Learned LLR correction lost to the analytic posterior-aware LLR by 1.121×, 1.078× and 1.060× across all three mismatch regimes. The published conclusion is *marginalise over your CSI error*, not *train a model*. Plain scalar LLR rescaling recovered only 17.0 % of the excess error when the estimate is biased high, and the tuner chose α = 1.00 — no rescaling at all — in the under-estimating direction, for a measured structural reason: a positive scalar cannot move the decision's zero crossing.
5. **P049 — the analytic baseline won, and the forest placed fourth of six.** The level-crossing-rate predictor with a two-parameter Platt recalibration gave the best Brier score (0.0217557), ahead of logistic regression and both forest variants, at **every** horizon tested. The raw analytic predictor has the best discrimination (AUC 0.8730) **and a negative Brier skill (−0.0043)** — worse than doing nothing — because it over-forecasts by 2.2×. Reporting AUC alone would have called it the winner and Brier alone would have called it a failure; both are published.
6. **P043 — the closed forms win where their assumptions hold.** At afterpulse probability ≤ 0.02 the matched inversion scores 0.0143 and the composed form 0.0136 against the learned model's 0.0243, both within 50 % of the counting-noise floor and both a line of algebra with no training set. And **nothing** inverts the paralyzable upper branch: 0.8336 / 0.8552 / 0.8399. Pinned by a test named `test_nobody_inverts_the_paralyzable_upper_branch` so a later change cannot quietly contradict the README.
7. **P041 — the headline margin is a calibration artefact and is labelled as one.** The learned fade-exceedance predictor beats the honest state-blind empirical baseline by only **3.15 %** in Brier. The 23 % margin over the exponential-exceedance closure is almost entirely a calibration correction, and both README and MODEL_CARD say so. The real evidence that channel state matters is AUC 0.6175 against 0.5000. The uncertainty output is weak (σ ratio 1.1289) and is published as weak.

Two further findings of the same character, about the field rather than about the models:

8. **P047 — the classical white-noise timing-jitter formula over-predicts by 15.34×** for an early-late detector (3.09× Gardner, 1.00× Mueller-Müller), because detector self-noise is not white and the loop filters it far better. And the **Rice/Gaussian cycle-slip estimate is wrong by six to sixteen orders of magnitude** — the functional form is wrong, not the scale, so **no fitted prefactor was offered**; the measurement is shipped attached to the broken prediction.
9. **P046 — minimum spread is near-useless as a proxy for burst dispersion** (Spearman ρ = 0.0343 across helical steps, **−0.4857** across constructions at equal memory — it ranks them backwards). At equal cost, candidates differ by 15.9× in dispersed burst length. The S-random permutation, the one the literature recommends, came **last of six** at equal memory.

## Defects the products found in their own tools, worth propagating

- **`scipy.optimize.milp` must be called with `options={"mip_rel_gap": 0.0}`.** The SciPy default inherits the HiGHS relative gap and returned a wrong *decision* on 3 of 481 feasible instances (P050), worst relative shortfall 6.87e-05, with the exact instance recorded.
- **A raw HiGHS allocation can violate a tight constraint row by ~1.7e-8**, inside its primal feasibility tolerance, banking 5.0e-07 relative objective. P050 re-solves the continuous part exactly on the support HiGHS chose.
- **`numpy.polynomial.hermite` weights overflow to NaN above roughly 350 nodes.** Found independently by P044 and P048, and hit by the coordinating session's own first attempt at the X2 diagnosis, which returned NaN at 400 nodes before being redone at 150. Both products now refuse above a hard cap with an explicit error.
- **`scipy.stats.kstest(x, "norm", args=(loc, scale))` raises `TypeError`** on the installed SciPy; standardise and call `kstest(z, "norm")`.
- **Reed-Solomon with the `b = 0` syndrome convention needs a leading `X_i` factor** that the `b = 1` form omits. Omitting it produces correct error *positions* and wrong error *values* — caught only by an exhaustive known-answer test (P041).

## Security

| Scan | Scope | Result |
|---|---|---|
| `bandit` | 46 150 LOC across the ten products, tests excluded | **0 HIGH, 0 MEDIUM, 10 LOW**, all `B101` (`assert_used`) |
| `detect-secrets` | the ten product directories | 2 findings, **both the same false positive**, zero credentials |
| gate secret patterns | tracked content, `git grep` over `HEAD` | **0 matches** |
| `pip-audit` | declared runtime union `numpy`, `scipy`, `scikit-learn`, `joblib` at installed versions (2.5.3 / 1.18.1 / 1.9.1 / 1.6.0) | **No known vulnerabilities found** |

The two `detect-secrets` findings were inspected individually and the trigger was identified mechanistically, not assumed. Both are the identifier `test_raw_highs_allocation_can_violate_the_availability_row`, at `products/P050/tests/test_milp.py:157` and `products/P050/validation/VALIDATION.md:138`.

**The trigger is the substring `ghs_` inside `HIGHS_` — the solver's own name.** `detect-secrets`' GitHub-token heuristic matches `gh[pousr]_` followed by 36 or more characters from `[A-Za-z0-9_]`, and because that class **permits underscores** the match runs on through the rest of the identifier: `ghs_allocation_can_violate_the_availability_row`, 43 characters past the prefix. The release gate's own pattern is `gh[pousr]_[A-Za-z0-9]{16,}`, which **excludes underscores**, so the longest run it can take after `ghs_` is `allocation` at 10 characters — short of 16, hence no match, which is why `git grep` over `HEAD` with the gate's patterns returns zero.

Verified both ways: scanning that identifier alone reproduces the finding, and the control `solve_highs_ok` — same word, short tail — produces none, so it is the length and not merely the word. There is no credential in either file, and a test cannot be renamed to please a regex that is matching the name of the solver it tests.

Declared runtime dependencies across all ten products are `numpy`, `scipy`, `scikit-learn` and `joblib` only. **No product declares `pulp`**, which closes the open question raised against P032 in Batch 04 — P050 solves with `scipy.optimize.milp` and names `pulp` only as a front end needing a solver this container lacks (`pulp.listSolvers(onlyAvailable=True)` returns `[]`, re-confirmed this session).

No unresolved critical security finding. **No token was read, displayed, reused, copied, committed or logged at any point in this session.**

## Level 4 — still 0 of 5, and that is the correct outcome

P041 and P043 ship the full groundwork: a hardware abstraction layer with a simulated backend and a device backend held to **one shared contract test suite**, simulation mode, dry-run mode, deployment and recovery written as executable checks, and a benchmark harness that records the measurement method and environment into its own output.

Both are labelled **`Level 3, hardware-pending`** with `hardware_pending: true` in `products.yaml`. Neither is labelled Level 4, and the phrase "Level 4" appears in their documentation only to state what is missing. P043's `benchmark_results.md` opens by saying its figures are not a Level 4 measurement and names the container that produced them. P043 additionally ships `Acquisition.is_measurement`, a single provenance predicate that is False for every simulated and every dry-run result — so a number from a model cannot be mistaken for a number from hardware.

With P031 and P033 from Batch 04 the mission now has **four** hardware-pending candidates rather than two.

**What closes L4, and only this:** the owner runs each product's `benchmark/run_benchmark.py` on the Jetson Orin Nano and makes the raw output available to a session. No simulated backend, extrapolation, vendor datasheet or workstation run substitutes, and no automated session can produce it.

## Open items for the owner

1. **L4 is 0 of 5 and now past its own Batch 05 deadline.** Four candidates are ready. Closable only with Jetson Orin Nano measurements.
2. **`approved_for_publish` is self-contradictory with `published: true`** and has been since Batch 04. ADR-016 clause 3 bars an unattended session from setting the flag; ADR-017 authorizes the push. Either retire the flag or amend ADR-016 — **an automated session must not pick**, and this session did not.
3. **P044 uses a different SNR-versus-irradiance convention from P010 and P048** (linear rather than quadratic). Documented in P044, but its absolute BER and outage numbers are not comparable with the rest of the portfolio. Needs a portfolio-level decision.
4. **`tracking/products_tracker.csv` covers only P001–P020.** P021–P040 are missing entirely — a pre-existing bookkeeping gap. This session appended P041–P050 with measured figures and **did not** backfill P021–P040, because it has no measured test counts for Batch 03 and would not invent them.
5. **The X2 cross-check specification needs amending** to fix the amplitude-versus-power convention, exactly as the P033/P039 specification needed amending. Both are specification defects rather than product defects, and both are now understood.
6. **P033/P039 from Batch 04 remains open** and will disagree forever until the comparable statistic is redefined as the sampled median of the declared cost model.
7. **`device_commit_files` can silently write a stale cached copy** when a `stagedPath` is reused. Mitigated here by a unique filename per transfer plus a SHA-256 comparison on the Mac before every merge — all three transfers this session matched exactly. Worth reporting upstream.
8. **P042 ships `.github/workflows/tests.yml`** like every product; the `gh` token lacks `workflow` scope, so per-product pushes must use the Keychain credential. Working as designed, recorded so it is not rediscovered.


## Gate verdict — the publication condition

```
Release gate — 10 product(s)

P041  PASS       315 tests
P042  PASS       398 tests
P043  PASS       302 tests
P044  PASS       339 tests
P045  PASS       229 tests
P046  PASS       191 tests
P047  PASS       234 tests
P048  PASS       161 tests
P049  PASS       264 tests
P050  PASS       152 tests

repository  PASS

Total tests passing: 2585
VERDICT: PUSH ALLOWED
```

`tracking/RELEASE_GATE_RESULT.json`: `push_allowed: true`, `total_tests_passing: 2585`, `blocked: []`, `repository_failures: []`, ten products with empty `failures` and empty `notes` — no skips anywhere in the batch.

**The gate script was not modified, no check was skipped, and no threshold was loosened.** It ran on its first attempt with the complete set of ten named products and returned exit 0. Per product it verified: tests via junit XML (>0 collected, 0 failed, 0 errored), `ruff` clean, the package importing in a fresh interpreter, `python -m <pkg> --help` exiting 0 in a clean subprocess, every `examples/*.py` running, and every `validation/*.py` re-executing. Repository-wide: no secret pattern in tracked content or history, no absolute private path, no tracked build artefact or credential-shaped file.

Two results from that run deserve recording in their own right.

**Every one of the ten agent self-reported test counts matched the independent junit measurement exactly** — 315, 398, 302, 339, 229, 191, 234, 161, 264, 152. That is reported as a measurement outcome, not as a reason to trust self-reports next time: the counts were measured because ADR-008 and ADR-010 require it, and the agreement is only knowable *because* they were measured.

**`git status` over `products/` after the gate run showed zero modified files.** The gate re-executes every validation and example script, so a clean tree means every committed raw output, PNG and benchmark record still reproduces bit-for-bit from a fresh run. The 2026-10-05 session needed a `git checkout -- products/` to discard volatile churn after its gate runs; this batch produced none, because the ten agents were instructed to re-run every producing script as a final step and did.

## Agent and token accounting

Recorded verbatim from each `Agent` tool result. No estimates.

| Agent | Product | Model | Tokens | Tool calls | Wall time (ms) | Outcome |
|---|---|---|---:|---:|---:|---|
| `a8822fdf669fb51c4` | P041 CodedFade | inherited session model (`claude-opus-5`) | 435267 | 141 | 5014048 | complete, first pass |
| `aafb30baf3e9c4a97` | P042 AcmPilot | inherited session model | 396598 | 93 | 4455347 | complete, first pass |
| `a35784f2302d47997` | P043 PhotonCount | inherited session model | 420178 | 108 | 3835068 | complete, first pass |
| `a50949b44e9ac20c2` | P046 InterleaveKit | inherited session model | 301376 | 90 | 3017436 | complete, first pass |
| `a0df0f71abb4ee19d` | P047 SlotSync | inherited session model | 449236 | 126 | 4451376 | complete, first pass |
| `aa992c9c5b6fc07d8` | P044 ApertureDiv | inherited session model | 413503 | 124 | 4917620 | complete, first pass |
| `af1e2c382cfcc8ebd` | P045 ArqLongHaul | inherited session model | 401131 | 111 | 4742175 | complete, first pass |
| `a238ebef6f2a520c8` | P048 SoftDecode | inherited session model | 426958 | 123 | 4159514 | complete, first pass |
| `aeded1530c2176b3c` | P049 LinkOutage | inherited session model | 420004 | 133 | 4413477 | complete, first pass |
| `a9692dff28b6d789b` | P050 CodeRateOpt | inherited session model | 367595 | 122 | 4965214 | complete, first pass |
| **Total (agents)** | 10 products | — | **4031846** | **1171** | **43971275 summed** | **10 of 10, zero resumptions** |

Wave 1 (P041, P042, P043, P046, P047): 2002655 tokens, 558 tool calls.
Wave 2 (P044, P045, P048, P049, P050): 2029191 tokens, 613 tool calls.

The summed wall time is 43971275 ms of agent time, but the agents ran five at a time, so the wall-clock cost was roughly the slowest agent in each wave — about 5014048 ms and 4965214 ms, near 2.8 hours in total rather than 12.2.

Coordinating session: tokens **not recorded**, tool calls **not recorded**. No tool in this environment reports the coordinating session's own usage back to it, and the report template forbids an estimate in place of a measurement. **Session total is therefore not computable**, and this is the third consecutive batch for which that is true. It is a gap in the instrumentation, not in the bookkeeping.

**Cost drivers worth naming.** Zero agents were terminated by a usage limit and zero needed `SendMessage` resumption — the second consecutive clean ten-of-ten, after five simultaneous deaths on 2026-08-29 and four of five on 2026-09-02. The single largest coordinator cost was the ADR-010 validation campaign: a five-product gate run over Wave 1 followed by a ten-product gate run over the whole batch, the latter taking roughly two hours of wall clock on two contended cores. That is the price of the rule that the implementer never validates its own numbers, and it bought the measurement that all ten self-reports were honest.

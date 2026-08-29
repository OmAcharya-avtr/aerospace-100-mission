# Batch 02 Readiness Report

**Date:** 2026-08-29 · **Session window:** 03:02–06:0x ET (inside 22:00–07:00 ET)
**Prepared by:** automated mission-director session, Environment A (cloud build container)
**Gate:** ADR-012 — batch stops here and waits for explicit approval from Om Acharya.

```text
STATUS: APPROVED 2026-08-29 — PUSH PENDING CREDENTIAL
```

---

## 1. Batch summary

Batch 02 is complete: ten products, from empty scaffold to completion gate.
Five (P016–P020) were built on 2026-08-07; the five remaining (P011–P015) were
built in this session by five concurrent build agents, one per product.

The theme is adaptive optics, turbulence characterization, and the
estimation/control core for aerospace GNC. It extends the Batch 01 atmospheric
line upward into wavefront sensing and correction, and opens the GNC line that
later batches build on.

Everything in this report was measured by the coordinating session, not
accepted from a build agent's self-report:

- **All ten test suites were re-run** by the coordinator after the agents
  finished. **2,505 tests, 0 failures, 0 errors.**
- **`ruff check` was re-run** on all ten products. **Clean on all ten.**
- **All 37 validation scripts in the batch were re-executed** by the
  coordinator. **Every scientific number reproduces bit-identically.** The only
  differences against the committed raw outputs are wall-clock timing lines and
  one kernel-version string. This is the strongest available evidence that no
  validation number in this batch is fabricated.

## 2. Product list

| ID | Name | Class | AI | Level | License | Status |
|---|---|---|---|---|---|---|
| P011 | WaveForge | flagship | yes | 3 | AGPL-3.0 (open-core) | READY FOR APPROVAL |
| P012 | NavBench | flagship | yes | 3 | AGPL-3.0 (open-core) | READY FOR APPROVAL |
| P013 | TurbScope | medium | yes | 2 | Apache-2.0 | READY FOR APPROVAL |
| P014 | WaveLab | medium | yes | 2 | Apache-2.0 | READY FOR APPROVAL |
| P015 | LinkSwitch | medium | yes | 2 | Apache-2.0 | READY FOR APPROVAL |
| P016 | ZernKit | compact | no | 1 | MIT | READY FOR APPROVAL |
| P017 | EstimKit | compact | no | 1 | MIT | READY FOR APPROVAL |
| P018 | ShackSim | compact | yes | 2 | Apache-2.0 | READY FOR APPROVAL |
| P019 | CnCast | compact | yes | 2 | Apache-2.0 | READY FOR APPROVAL |
| P020 | AtmoProfile | compact | no | 2 | MIT | READY FOR APPROVAL |

## 3. Class distribution

2 flagship · 3 medium · 5 compact — exactly the mandated batch shape.

## 4. AI distribution

7 of 10 AI-enabled (P011, P012, P013, P014, P015, P018, P019) — meets the ≥7
requirement. Every AI product implements its classical/analytic baseline first
and benchmarks the learned model against it on identical held-out data. Every
AI product exposes an uncertainty or confidence output. PyTorch is not
available in this environment; all models are scikit-learn or NumPy.

## 5. Validation distribution

2 × Level 1 (P016, P017) · 6 × Level 2 (P013, P014, P015, P018, P019, P020) ·
2 × Level 3 (P011, P012). Matches the batch specification.

Both Level 3 products carry `docs/REQUIREMENTS.md` with a numbered requirement
set and a verification matrix, an uncertainty analysis, a seeded regression
suite, a performance benchmark, and failure-mode tests.

## 6. Repository plan

Publication is **not** part of this report; it is requested for approval.
Proposed layout, consistent with Batch 01:

| Repository | Visibility | Contents | License |
|---|---|---|---|
| `OmAcharya-avtr/flagship-waveforge` | Public (new) | P011 WaveForge | AGPL-3.0 |
| `OmAcharya-avtr/flagship-navbench` | Public (new) | P012 NavBench | AGPL-3.0 |
| `OmAcharya-avtr/batch-02-suite` | Public (new) | P013–P020 | Apache-2.0 / MIT per product |
| `OmAcharya-avtr/aerospace-100-mission` | Public (existing) | Monorepo source, trackers, ADRs, reports | mixed |

All ten package names were **re-checked against PyPI on 2026-08-29** (HTTP 404
on `/pypi/<name>/json` for every one): `waveforge`, `navbench`, `turbscope`,
`wavelab`, `linkswitch`, `zernkit`, `estimkit`, `shacksim`, `cncast`,
`atmoprofile` — **all free.** No conflict of the P002 `trackforge` kind exists
in this batch.

## 7. Test results

Re-run by the coordinating session with `python3 -m pytest tests/ -q` from each
product root. Nothing below is a build agent's self-report.

| ID | Name | Tests | Result |
|---|---|---:|---|
| P011 | WaveForge | 635 | pass |
| P012 | NavBench | 715 | pass |
| P013 | TurbScope | 122 | pass |
| P014 | WaveLab | 180 | pass |
| P015 | LinkSwitch | 201 | pass |
| P016 | ZernKit | 158 | pass |
| P017 | EstimKit | 117 | pass |
| P018 | ShackSim | 148 | pass |
| P019 | CnCast | 112 | pass |
| P020 | AtmoProfile | 117 | pass |
| | **Batch 02 total** | **2,505** | **0 failed, 0 errored** |

Mission-wide, including Batch 01's 1,041: **3,546 tests passing.**

**Lint:** `ruff check src/ tests/ examples/ validation/` — `All checks passed!`
on all ten products, re-run by the coordinator.

## 8. Benchmark results — AI versus classical baseline

Reported as measured. Where the classical method wins, it is recorded as a
result, not concealed (ADR-011).

**P011 WaveForge — learned predictive AO control.** Against a per-latency
gain-tuned classical integrator and a pure-delay baseline, on held-out screens,
the learned predictor **wins in the trained regime**: residual variance
0.6728 → 0.5361 rad² at 1-frame delay (1.25×) and 1.7085 → 0.4958 rad² at
4-frame delay (3.45×); Strehl held near 0.60 while the integrator fell
0.54 → 0.21. Approximately 29 % of the gain at 3-frame delay comes from the
pseudo-open-loop control formulation and 71 % from prediction.
**It loses out of distribution:** at 2× training wind speed (20 m/s) it is
**17 % worse** than the integrator (1.8005 vs 1.5356 rad²), and trained clean
but deployed at 100 e⁻/subaperture it is **5.9× worse** (29.75 vs 5.03);
noise-matched retraining recovers it to 3.88 and it wins again.

**P012 NavBench — learned adaptive process-noise tuning.** Held-out position
RMSE: learned **1.92299 m**, fixed hand-tuned Q **2.09982 m**, classical
Mehra-style IAE **2.27394 m**; 44/60 paired wins over fixed, paired mean
−0.17683 ± 0.09361 m (95 % CI excludes zero). **Three important negatives are
recorded:** (a) Mehra has the ANEES closest to its degrees of freedom, but it
was measured pinned at its upper clip on 60/60 runs with correlation to the
true noise scale of exactly 0.0000 — it is not adapting, it is applying a
constant 64× Q inflation, and the report says so; (b) **none of the three
tuners is statistically consistent**; (c) **the learned confidence output
fails** — correlation with actual error +0.2206, the wrong sign, and the
high-confidence half is worse than the low-confidence half (0.6419 vs 0.4752).
The fixed baseline also beats the learned tuner at small mismatch (|u| ≤ 0.5:
1.801 vs 1.866 m).

**P013 TurbScope — learned multi-sensor Cn² regression.** Against the
specification's mandated closed-form single-sensor baseline (scintillometer
weak-regime inversion) the learned model **wins**: RMSE 0.071 vs 0.658 dex.
Against a *stronger* closed-form single-sensor baseline (DIMM-only) the learned
model **loses**: 0.071 vs 0.032 dex, because the synthetic generator gives DIMM
no saturation-related model-form error. Both results are stated in the README,
MODEL_CARD and VALIDATION.

**P014 WaveLab — learned slope-to-Zernike reconstructor.** The regularized
modal least-squares baseline **beat the learned ensemble at 9 of 10 operating
points**, with the margin growing to 7.4× at flux 10,000. The ensemble's single
win was at 60 % subaperture dropout (RMS 0.060 vs 0.817 rad), and the product
attributes it correctly to the baseline's *fixed* regularization strength going
unstable on a severely under-determined system rather than to a general ML
robustness advantage. Ensemble uncertainty is measured as **not calibrated**
(spread understates true error by 25–40 %).

**P015 LinkSwitch — learned RF/optical switching policy.** The learned
RandomForest policy **does not win.** Mild scenario: statistical tie with
hysteresis on throughput, hysteresis wins outage and switch count. Moderate
scenario: hysteresis wins every metric outright (876.905 [874.568, 879.242] vs
learned 856.796 Mb/s), and the learned policy underperforms even the naive
fixed-threshold baseline (867.045 Mb/s). Not retuned.

**P018 ShackSim — learned slope estimator.** Beats thresholded centre of
gravity **only below ≈ 100 detected photoelectrons per subaperture**; best
advantage 1.38×. Above the crossover the analytic estimator is better and the
gap widens without limit — **10.5× better at 10,000 e⁻**. Against the
correlation baseline the learned model loses at 6 of 7 flux levels for round
spots.

**P019 CnCast — learned Cn² profile predictor.** The learned model wins against
HV 5/7, and the product states plainly that **the win is close to tautological**
because the synthetic targets are generated from the H-V family driven by the
same surface variables the model is given. The informative comparison is against
the training climatology (0.3102 dex), which the model improves on by 32 %.

**Net:** of seven AI products, the classical baseline wins outright in **two**
(P014, P015), wins in a large part of the operating envelope in **two more**
(P013 against the stronger baseline, P018 above ~100 e⁻), and the learned method
wins in-distribution but degrades or fails out-of-distribution in **two**
(P011, P012). Only P019 is a clean learned win, and it is labelled as
near-tautological. **Eight new honest negative results** are added to the nine
already on record.

## 9. Security results

Full detail in `batch_reports/scans/SCAN_SUMMARY_BATCH02.md`; raw outputs in
`batch_reports/scans/`.

| Scan | Result |
|---|---|
| `detect-secrets` — working tree | **0 findings** (after removing untracked `.pytest_cache/`, `.ruff_cache/`, `__pycache__/` build artifacts) |
| `detect-secrets` — git history | **0 findings.** No `ghp_`/`github_pat_`/`gho_`/`ghs_`/`ghu_`/`ghr_`, AWS key-ID, or PEM private-key pattern anywhere in `git log -p --all`. No credential file ever tracked. No absolute private path (`/Users/<name>`, `/home/claude`, `/root/`) in any tracked file in history. |
| `bandit -r products/ -x '*/tests/*'` | 43,061 lines scanned. **0 high, 0 medium, 10 low.** All ten low findings are `B101 assert_used` in `validation/` scripts, where an assertion is the intended failure mechanism. No library or CLI code affected. |
| `pip-audit` | 18 vulnerable packages in the container; **0 of them is a declared dependency of any product.** Declared dependencies across the whole mission are only `numpy`, `scipy`, `matplotlib`, `scikit-learn`, `pyyaml`, `joblib`, `sgp4`, `pulp`, `pandas`. Container tooling is not a product vulnerability. |

**No unresolved critical security finding in any Batch 02 product.**

## 10. Documentation status

Present and verified for all five new products: `README.md`, `LICENSE` (full
license text, © 2026 OPTIMA Organisation), `CHANGELOG.md` (0.1.0),
`MODEL_CARD.md`, `DATASET_CARD.md`, `pyproject.toml`,
`validation/VALIDATION.md`; plus `docs/REQUIREMENTS.md` for the two Level 3
flagships. P016–P020 were verified complete in their own build session and
re-verified here.

Compliance checks run across P011–P015:

- Required research-grade safety statement present in every README. **No
  occurrence anywhere of "flight-safe", "flight-qualified", "mission-ready",
  "production-ready" or an unqualified "certified".**
- Required credits line present exactly once per README:
  "This is under reserved rights obtained by OPTIMA Organisation."
- Status header `TESTING` in every product README.
- No committed artifact exceeds 1 MB; datasets and models regenerate
  deterministically from seeded scripts.

## 11. Screenshots

16 PNGs, all produced by actually executing the products' example scripts, all
verified to carry a valid PNG signature: P011 5, P012 5, P013 2, P014 2,
P015 2 (P016–P020 carry a further 10 from their own build session).

P011: `closed_loop_run`, `error_budget_sizing`, `phase_screen_gallery`,
`predictive_control`, `rejection_transfer`.
P012: `adaptive_q_tuning`, `estimator_bench`, `gyro_allan_deviation`,
`mekf_attitude`, `nees_nis_consistency`.
P013: `prediction_vs_baselines`, `saturation_curve`.
P014: `benchmark_flux_dropout`, `reconstruction_demo`.
P015: `policy_comparison`, `telemetry_and_switching`.

## 12. Known failures and unresolved limitations

**Reported validation failures — all retained, none tuned away.**

- **P020 AtmoProfile:** the Bufton rms wind over 5–20 km evaluates to
  22.9637 m/s for a 5 m/s ground wind, against the 21 m/s pseudowind of
  HV 5/7 — a **9.35 % mismatch against a 2 % tolerance, recorded as FAIL.** The
  convention behind the literature's 21 m/s could not be established in the
  build. The published parameter is retained; the mismatch is reported. Zero
  blocking failures in the self-consistency checks.
- **P011 WaveForge:** per-mode Noll variances recovered by differencing the
  published rounded table are consistent for 13 of 20 modes; the other seven
  are high by an independently established +0.25 % normalisation offset. On
  genuine closed-loop residuals the extended Maréchal approximation
  underestimates Strehl by up to 26.7 % at σ² ≈ 1.9 rad². Predictor uncertainty
  is miscalibrated in both directions (82.7 % inside 1σ at horizon 1, 62.3 % at
  horizon 4, against a nominal 68.3 %). Zonal fitting exponent is 1.544, not
  5/3, a finite-aperture effect that converges with actuator count.
- **P012 NavBench:** no tuner is statistically consistent; the learned
  confidence output does not work (documented as a negative result); the
  classical Mehra scheme saturates at its clip. Three genuine defects were
  found *by* validation and fixed with evidence: axis loss below ~2e−12 rad in
  `axis_angle_from_quat`; catastrophic cancellation in
  `attitude_state_transition` for 1e−8 < θ < 1e−2; and an endpoint-versus-
  interval-average rate sample in the MEKF that had produced mean NEES 1925
  against 6 degrees of freedom.
- **P014 WaveLab:** ensemble uncertainty is not calibrated (spread understates
  error by 25–40 %). The Fried-geometry reconstructor cannot recover waffle by
  construction — verified as expected behaviour, not a defect.
- **P015 LinkSwitch:** the learned policy loses to the hysteresis baseline; the
  RandomForest vote-fraction confidence is uncalibrated.

**Cross-cutting limitations, stated in every affected product:**

1. **No product is validated against measured field or flight data anywhere in
   this batch.** All data is synthetic and generated by committed seeded
   scripts. P013, P015 and P019 state this explicitly in their dataset cards;
   P015 additionally states in four documents that its fading model is
   simulated, not measured.
2. Where a learned model is scored against a synthetic generator, the score
   measures recovery of that generator, not physical accuracy (P013, P019
   say so in terms).
3. P011 is single-layer frozen flow only, so its ML advantage is an upper
   bound; screens are band-limited and pupil tip/tilt variance sits at 0.82 of
   analytic.
4. P012 models no gyro flicker plateau, uses a two-body orbit and white
   isotropic GNSS errors, and neglects the MEKF covariance-reset Jacobian
   (measured 4.6e−04 relative at the largest reset).
5. P013's saturation curve is an explicitly labelled heuristic, not a
   literature fit.
6. P015's ITU-R P.838 (k, α) coefficients are illustrative and were not looked
   up from current tables; the P.618 path-reduction factor is simplified. This
   is stated in the product.
7. PyTorch is unavailable, so all models are scikit-learn/NumPy. Model capacity
   is therefore modest by design; this is a scope constraint, not a finding.
8. `shared/aerocore/turbulence.py`, proposed in the batch specification, was
   **not** created. Each product implements its own turbulence machinery and
   cites the others as related work. This preserves the mission rule that
   products are independently installable with no cross-product imports, at the
   cost of some duplication. Recorded here as a deliberate deviation from the
   specification.

## 13. Licenses

| License | Products |
|---|---|
| AGPL-3.0 (open-core) | P011, P012 |
| Apache-2.0 | P013, P014, P015, P018, P019 |
| MIT | P016, P017, P020 |

Full license text is present in every product's `LICENSE`, each with the
copyright line © 2026 OPTIMA Organisation. Verified: P011 and P012 carry the
complete 680/681-line AGPL-3.0 text, not a stub.

## 14. IP review (§28 answers, all ten products)

1. Written for this mission — **yes, all ten.**
2. Copied from a private repository — **no.**
3. Copied from a university laboratory — **no.**
4. Copied from an employer — **no.**
5. Grant-funded code reused — **no.**
6. Dataset redistribution restricted — **no.** Every dataset in this batch is
   synthetic and generated by committed, seeded scripts. No external dataset,
   measurement archive or third-party corpus is used or redistributed.
7. Model license restricting commercial use — **no.** All models are trained
   in-repository with scikit-learn.
8. Paper restricting reproduction — **no.** Published results (Noll 1976;
   Fried 1965/1966/1977; Hudgin 1977; Southwell 1980; Greenwood 1977;
   Maréchal 1947; Hufnagel 1974 / Valley 1980; Beland 1993; Tatarski 1961;
   Sarazin & Roddier 1990; Kalman 1960; Julier & Uhlmann 1997/2004; Rauch,
   Tung & Striebel 1965; Mehra 1970/1972; Lefferts, Markley & Shuster 1982;
   Farrenkopf 1978; Kalata 1984; Bar-Shalom, Li & Kirubarajan 2001; IEEE Std
   952; ITU-R P.618/P.837/P.838; Andrews & Phillips 2005; Hardy 1998) are
   implemented from cited standard results with independent derivations. No
   figure, table, or text is reproduced from any source.
9. Reveals confidential architecture — **no.**
10. Patent-sensitive material — **none identified.** All methods implemented
    are long-established published results.

**No product is marked BLOCKED — IP REVIEW REQUIRED.**

## 15. Publication recommendation

**Recommend approval and publication of all ten products**, subject to the two
gates that are outside this report's control:

1. **Owner approval** (ADR-012) — requested here.
2. **A working credential in secure storage** (ADR-014, R-02). The exposed PAT
   is compromised and was neither read nor used in this session.

Every §17 completion-gate item and every §11 AI-product requirement is met. No
product claims to be flight-safe, certified, mission-ready or production-ready;
every one is labelled research-grade (or educational at Level 1) with an
explicit statement that it is not flight-qualified and not approved for
operational aerospace use.

## 16. Commit plan (≤5 pushed commits, ADR-015)

Local commits in this session are unlimited and already made. When approved,
the development history squashes into at most five public commits:

1. `release: WaveForge v0.1.0 — adaptive-optics sizing, simulation and predictive control`
2. `release: NavBench v0.1.0 — attitude and navigation filter bench with NEES/NIS consistency diagnostics`
3. `release: batch-02 medium suite — TurbScope, WaveLab, LinkSwitch`
4. `release: batch-02 compact suite — ZernKit, EstimKit, ShackSim, CnCast, AtmoProfile`
5. `docs: batch-02 validation evidence, security scans, readiness report and tracker updates`

Pre-push gate to be re-run immediately before pushing: tests green · ruff clean
· secret scan clean over tree and history · build artifacts removed · no
credentials in the diff · no absolute private paths in tracked files ·
development commits squashed. All seven currently pass; they are re-run at push
time because the tree may change between approval and push.

## 17. Updated mission totals

Derived from `scripts/quota_report.py` (ADR-009), not hand-maintained.

```
Registered: 20 / 100
Built:      20 / 100
Published:  10 / 100

Quota                   Target   Built  Remaining
flagship                    20       4         16
medium                      30       6         24
compact                     50      10         40
ai (minimum)                70      14         56
L1                          10       4          6
L2                          60      12         48
L3                          25       4         21
L4                           5       0          5
```

This matches the batch specification's projected position after Batch 02
exactly. Two quota notes carry forward:

- **Level 3 accumulates at 2 per batch, which reaches 20 against a target of
  25.** Batches 08–10 must raise flagship validation depth or promote selected
  medium products to Level 3.
- **Level 4 remains at 0 of 5 and has not started.** ADR guidance is that it
  must not be deferred past Batch 05.

## 18. Approval decision

```text
APPROVAL STATUS: APPROVED
APPROVED BY: Om Acharya
DATE: 2026-08-29
```

Approval was given in chat during the 2026-08-29 automated session, as
"Go ahead and push", after this report was delivered to the owner.

**The push was attempted and could not be completed.** The full ADR-015
pre-push gate was re-run and all seven items passed, and the history was
squashed to the five commits in §16. Neither reachable environment could
authenticate:

- **Environment A** (cloud build container) — `git push origin main` refused by
  the git proxy: "OmAcharya-avtr/aerospace-100-mission is not in this session's
  authorized repository set, so the proxy will not inject a credential for it."
  HTTP 403.
- **Environment B** (the shell on the owner's machine) — reaches `github.com`
  and `api.github.com` (both HTTP 200) and holds a clean working copy at
  `origin/main`, but carries **no credential in any permitted store**: no
  `GITHUB_TOKEN`, no `GH_TOKEN`, no git credential helper, no
  `~/.git-credentials`, no `gh` CLI. The only credential known to exist on the
  machine is the file at `secrets/github_token.txt`, which ADR-014 treats as
  compromised and forbids reading or reusing. It was not read.

Batch 02 is therefore **approved and staged, not published.** The five signed
commits exist locally and in a bundle delivered to the owner. Publication
completes when the owner runs the push themselves, or when a rotated credential
reaches secure storage.

Batch 03 may be specified in a subsequent session; approval, not publication,
is what ADR-012 gates it on.

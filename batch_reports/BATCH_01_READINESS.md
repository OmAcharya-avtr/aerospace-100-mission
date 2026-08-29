# Batch 01 Readiness Report

**Date:** 2026-08-01 · **Prepared by:** automated build sessions under owner-approved Batch 01 specification
**Spec:** `batch_reports/BATCH_01_SPEC.md` · **Commit at report time:** see `tracking/RELEASE_LEDGER.md`

---

## 1. Batch summary

All ten Batch 01 products are built, tested, validated, documented, and scanned. **1,041 automated tests pass; zero fail; zero are skipped to hide a defect.** Ruff is clean across every product. No critical or high security finding is unresolved.

Every product carries working code, a README following the mission template, a license, a changelog, validation evidence produced by scripts actually executed in-session, and screenshots rendered by actually running the examples. All seven AI-enabled products additionally carry a model card and a dataset card.

The batch was **approved for publication by Om Acharya on 2026-08-01**. The one outstanding naming decision (§14) was resolved by owner instruction: P002 is renamed TrackBench.

## 2. Product list

| ID | Name | Class | AI | Level | License | Tests | Status |
|---|---|---|---|---|---|---:|---|
| P001 | BeamTwin | flagship | yes | 3 | AGPL-3.0 | 251 | APPROVED |
| P002 | TrackBench | flagship | yes | 3 | AGPL-3.0 | 295 | APPROVED (renamed from TrackForge, §14) |
| P003 | ScintiNet | medium | yes | 2 | Apache-2.0 | 50 | APPROVED |
| P004 | PassPlanner | medium | yes | 2 | Apache-2.0 | 106 | APPROVED |
| P005 | JitterScope | medium | yes | 2 | Apache-2.0 | 53 | APPROVED |
| P006 | LinkBudgetX | compact | no | 1 | MIT | 54 | APPROVED |
| P007 | QuatKit | compact | no | 1 | MIT | 89 | APPROVED |
| P008 | CentroidNet | compact | yes | 2 | Apache-2.0 | 41 | APPROVED |
| P009 | FogCast | compact | yes | 2 | Apache-2.0 | 34 | APPROVED |
| P010 | BERBench | compact | no | 2 | MIT | 68 | APPROVED |
| | | | | | **Total** | **1041** | |

## 3. Class distribution — meets spec

2 flagship, 3 medium, 5 compact. Mission §3 requires exactly this.

## 4. AI distribution — meets spec

7 of 10 AI-enabled (P001–P005, P008, P009), against a §3 minimum of 7 per batch. Each satisfies mission §11: a classical or analytic baseline was implemented first, the learned model is benchmarked against it on held-out data, an uncertainty or confidence output is exposed, and the model card carries the statement that the model is not certified for operational flight use. No product uses a chatbot wrapper or relabels a deterministic threshold as AI.

## 5. Validation distribution

| Level | Count | Products |
|---|---:|---|
| 1 — Educational | 2 | P006, P007 |
| 2 — Research | 6 | P003, P004, P005, P008, P009, P010 |
| 3 — Professional | 2 | P001, P002 |
| 4 — Hardware | 0 | deferred; five Level 4 products are planned for later batches per §15 |

Against the whole-mission target of 10/60/25/5, Batch 01 contributes 2/6/2/0. This is on track: the L1 allocation is deliberately front-loaded lightly, and L4 requires the Jetson Orin Nano, which is out of scope until hardware batches.

## 6. Repository plan

Approved publication layout:

- `flagship-beamtwin` — P001, AGPL-3.0
- `flagship-trackbench` — P002, AGPL-3.0
- `batch-01-suite` — P003–P010, mixed Apache-2.0 and MIT, each product independently licensed and documented in its own subdirectory

Repo creation from the build environment is blocked by the session permission layer (ADR-004). Publication is therefore owner-executed: create the three empty repositories, then push from the synced git bundle.

> **Correction, 2026-08-29.** The sentence above was true of one execution environment and wrong as a general
> statement. ADR-004 has been replaced. GitHub writes are available from the owner's local environment — verified
> by capability probe on 2026-08-29 — and may also be available to a build session that has the mission
> repositories in its authorized set. Publication is no longer defined as owner-executed; it is executed by
> whichever environment passes the capability test, subject to the approval gate in ADR-012. The three
> repositories named above were created and are public. This paragraph is retained as the record of what was
> believed at the time.

## 7. Test results

1,041 passed, 0 failed, 0 skipped. Independently re-run by the coordinating session rather than accepted from build-agent self-reports.

Coverage of test *kinds*, not percentages: unit, input validation, known-answer with hand calculations shown in comments, property-based via Hypothesis, integration, CLI, configuration, regression against pinned seeded values, performance bounds, failure-mode, and reproducibility. The two flagships carry the full flagship battery required by §31.

Three defects were found by the test and validation process and fixed rather than concealed: a finite-difference step that left the physical domain at zero pointing error (P006), a phase-screen normalization error giving a factor-two low phase variance (P003), and an `asin` conditioning error near zenith (P004). Two P001 test failures were traced to defective tests rather than defective code and the tests were corrected; no tolerance was widened and no `xfail` or `skip` was used to mask a failure.

## 8. Benchmark results

| Product | Benchmark | Result |
|---|---|---|
| P001 | Monte Carlo throughput | 1.31e7 samples/s; surrogate 790× faster than MC per query |
| P002 | Simulator throughput | 36,284 steps/s (7.3× realtime); Q-learning training 2.97 s |
| P003 | Phase-screen campaign | 54 rows in 22.6 s; surrogate ~5000× faster than simulation per point |
| P004 | ILP scheduling | optimum matched on 20/20 seeded instances; greedy gap mean 3.65%, max 19.04% |
| P005 | Detector benchmark | 6.4 s full run; baseline fits 2400× faster than the MLP |
| P008 | Training | 24.6 s for a 5-model ensemble on 4,200 frames |
| P010 | Monte Carlo BER | full validation and examples in ~30 s against a 180 s budget |

## 9. Security results

Full detail in `batch_reports/scans/SCAN_SUMMARY.md`.

- **Secrets:** `detect-secrets` over all products, tracking, templates and reports → **0 findings**. No token string and no credential file is tracked by Git. The GitHub PAT resides only on the owner's machine outside every repository.
- **Static analysis:** `bandit` over 14,488 lines → 0 high, 0 medium, 3 low, each reviewed and accepted with a written rationale (two subprocess patterns in a developer-only training script with hardcoded arguments, one use of `random` where seeded reproducibility is required).
- **Dependencies:** the union of declared runtime dependencies is `joblib, matplotlib, numpy, pulp, pyyaml, scikit-learn, scipy, sgp4`. **None appears in any `pip-audit` advisory.** Advisories reported against the build container relate to pre-installed image packages that no product imports or ships.

`gitleaks` and `trivy` named in §29 are unavailable in this environment; `detect-secrets` and `bandit` were used under §29's allowance for equivalent tools. This substitution is disclosed rather than glossed.

## 10. Documentation status

All ten: README following the §22 template, LICENSE, CHANGELOG, validation writeup. All seven AI products: MODEL_CARD and DATASET_CARD. Both flagships additionally carry `docs/REQUIREMENTS.md` with a numbered requirement set and a verification matrix mapping each requirement to named evidence (P002 verifies 23 requirements this way).

Every README carries the §22 safety statement and the required credits line, and the credits line appears nowhere outside legal and credits sections.

## 11. Screenshots

19 PNGs across the batch, every one produced by executing the product's own example or validation script in-session. No screenshot is mocked, and no figure reports a number that does not appear in a saved raw output file.

## 12. Unresolved issues

None blocking. Four honest negative results are carried deliberately and must not be read as defects — they are §34 and §41 compliance, and reversing them would constitute fabricated progress:

1. **P003 ScintiNet** — the learned surrogate *loses* to the Rytov analytic baseline inside the baseline's own validity regime (RMSE in log₁₀ 0.078 versus 0.043). Its documented value is a ~5000× speedup and applicability where no closed form exists, not accuracy.
2. **P005 JitterScope** — the classical per-band z-score baseline *beats* the MLP detector (F1 0.9644 versus 0.9600) and fits 2400× faster. The CLI now defaults to the baseline; the MLP's motivating case is flagged as an untested claim.
3. **P001 BeamTwin** — the surrogate's uncertainty is under-dispersed: ±2σ covers 39.9% of cases, not 95%. It beats the analytic baseline in the high-jitter regime by 42% and *loses by 71×* in the low-jitter regime. Both figures are published.
4. **P008 CentroidNet** — the ML ensemble is beaten by a thresholded centre-of-gravity estimator above SNR ≈ 40, and its ensemble spread underestimates true error by 2.3×–11×. Both are documented with the physical explanation.

Two scope deviations are disclosed in the affected products' limitations rather than hidden: PyTorch is unavailable in the build environment, so P008 uses an MLP ensemble in place of the specified CNN, and P002 uses tabular Q-learning in place of a deep RL policy.

## 13. Licenses

AGPL-3.0 for the two flagships (open-core boundary for commercial digital-twin and PAT tooling), Apache-2.0 for the five research-grade AI products, MIT for the three deterministic libraries. Each matches the §23 policy for its category. All carry the OPTIMA Organisation copyright notice. Full license texts are included; where gnu.org was blocked by the environment proxy, the AGPL text was taken from an authentic bundled copy and verified complete including the §13 network clause.

## 14. IP review (§28)

Answered for all ten products:

1. Written for this mission — **yes, all ten.**
2. Copied from a private repository — **no.**
3. Copied from a university laboratory — **no.**
4. Copied from an employer — **no.**
5. Grant-funded code reused — **no.**
6. Dataset redistribution restricted — **no.** All datasets are synthetic and generated by committed, seeded scripts. Two public historical TLEs are used as test fixtures (ISS, NOAA-14), which are US Government public-domain data.
7. Model license restricting commercial use — **no.** All models are trained in-repository with scikit-learn.
8. Paper restricting reproduction — **no.** Published equations are implemented from cited standard results with independent derivations; no figure, table, or text is reproduced.
9. Reveals confidential architecture — **no.**
10. Patent-sensitive material — **none identified.** All methods implemented are long-established published results.

**No product is marked BLOCKED — IP REVIEW REQUIRED.**

**One naming decision is outstanding.** PyPI conflict checks were run for all ten names. Nine are free. **`trackforge` is taken** on PyPI by an unrelated computer-vision tracking library at version 0.3.0. Because the conflicting package is also in the tracking domain, §32's rule against names that create package-name conflicts applies. Verified-free alternatives, in recommended order: **PATForge**, **TrackBench** (which appears as a suggested name in the mission specification itself), **TrackScope**, **PATBench**, **BeamTrack**. The rename is a mechanical refactor of the package identifier and its imports across the product's 295 tests. **RESOLVED 2026-08-01:** the owner instructed the rename. P002 is now **TrackBench**, package `trackbench`, verified free on PyPI. The refactor was applied across all 42 referencing files and all 295 tests still pass with ruff clean.

## 15. Publication recommendation

**Recommend publication of all ten products.** The §14 naming decision is resolved. The owner approved the batch on 2026-08-01.

The batch meets every §17 completion-gate item and every §11 AI-product requirement. No product claims to be flight-safe, certified, mission-ready, or production-ready. Every product is labeled research-grade or educational with an explicit statement that it is not flight-qualified and not approved for operational aerospace use.

## 16. Commit plan (≤5 pushed commits, §6)

1. `release: BeamTwin v0.1.0 — FSO link digital twin with fade-probability surrogate`
2. `release: TrackBench v0.1.0 — PAT acquisition, tracking, and reacquisition suite`
3. `release: batch-01 medium suite — ScintiNet, PassPlanner, JitterScope`
4. `release: batch-01 compact suite — LinkBudgetX, QuatKit, CentroidNet, FogCast, BERBench`
5. `docs: batch-01 validation evidence, security scans, and tracker updates`

## 17. Approval decision

```text
APPROVAL STATUS: APPROVED
APPROVED BY: Om Acharya
DATE: 2026-08-01
```

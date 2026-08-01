# Batch 01 Specification

**Date:** 2026-08-01 · **Development approval:** granted (see APPROVAL_LOG.md) · **Publication approval:** pending readiness report
**Composition:** 2 flagship, 3 medium, 5 compact · 7/10 AI-enabled · Levels: 2×L1, 6×L2, 2×L3
**Theme:** FSO link engineering and PAT foundations. Batch 1 also seeds the shared engineering core (units, constants, channel models) reused by later batches.
**Stack (ADR-005):** Python 3.11, NumPy/SciPy, PyTorch-CPU for ML products, pytest + Hypothesis, Ruff. CLI + library API + plotting examples; no web UIs this batch.
All names provisional pending §32 conflict checks before publication.

---

## P001 — BeamTwin (flagship, L3, AI)

**Problem:** Predicting end-to-end FSO link performance requires combining deterministic link-budget terms with stochastic atmospheric effects; engineers need a single tool that produces received-power distributions and fade statistics, not just a point estimate.
**Scope:** Link-budget engine (transmit power, optics, divergence, geometric loss, pointing loss, receiver aperture); atmospheric channel: lognormal scintillation from Rytov variance (weak-turbulence regime), Kim/Kruse fog-aerosol attenuation; Monte Carlo received-power simulation; ML surrogate (gradient-boosted or small MLP) predicting fade probability from link parameters, trained on the tool's own Monte Carlo output, with prediction-interval output; baseline = analytic lognormal fade formula.
**Validation (L3):** analytic hand-calculation cases; comparison against published link-budget examples; regression suite; uncertainty analysis; verification matrix; error handling; security review.
**Deliverables:** package + CLI, tests, examples, model card, validation report, README per template.

## P002 — TrackForge (flagship, L3, AI)

**Problem:** PAT designers need to trade acquisition time against detection probability across scan strategies, uncertainty-cone sizes, and platform jitter before committing to hardware.
**Scope:** Spiral and raster acquisition scan generators; target uncertainty region model; detection/lock model; coarse/fine tracking loop simulation (second-order gimbal dynamics, configurable jitter PSD injection); controller benchmark harness comparing PID and LQR; RL reacquisition policy (small DQN/PPO via Stable-Baselines3, CPU) benchmarked against scripted baselines; confidence/uncertainty reporting on learned-policy results.
**Validation (L3):** analytic spiral-coverage checks; Monte Carlo acquisition-time distributions vs published scan-statistics results; regression + performance tests; failure-mode tests (lost-lock recovery); security review.

## P003 — ScintiNet (medium, L2, AI)

**Problem:** Split-step phase-screen simulation of scintillation is accurate but slow; a validated learned surrogate gives fast scintillation-index estimates for link-planning loops.
**Scope:** Phase-screen simulator (data generator, committed as code not data); scintillation-index dataset across Cn², path length, wavelength, aperture; small MLP surrogate; baseline = Rytov analytic expression; error analysis vs held-out simulation, dataset card, model card.
**Validation (L2):** Rytov-regime agreement checks, published weak-turbulence comparisons, reproducible benchmark script, documented validity range (weak-fluctuation regime only).

## P004 — PassPlanner (medium, L2, AI)

**Problem:** Optical ground stations lose passes to clouds; contact planning must weight geometric visibility by optical availability and optimize downlink allocation.
**Scope:** SGP4 propagation (sgp4 library) from TLEs; elevation-mask visibility windows; cloud-availability model (climatological priors + configurable forecasts); scheduling: greedy baseline + ILP optimizer (PuLP/OR-Tools) maximizing expected delivered data across stations; availability-prediction component with confidence output.
**Validation (L2):** pass predictions cross-checked against reference propagator outputs; scheduler optimality checks on small analytic instances; reproducible benchmarks.

## P005 — JitterScope (medium, L2, AI)

**Problem:** Platform vibration converts to pointing error and optical loss; engineers need PSD-to-pointing-loss analysis plus automatic detection of anomalous vibration signatures in telemetry.
**Scope:** Welch PSD estimation; RMS jitter integration per band; jitter-to-pointing-loss conversion for Gaussian beams; synthetic telemetry generator with injectable faults; autoencoder anomaly detector with reconstruction-error thresholds and confidence output; baseline = spectral band-energy thresholding.
**Validation (L2):** analytic PSD cases (known sinusoids/white noise), pointing-loss formula validation against published Gaussian-beam results, labeled synthetic fault-set evaluation (precision/recall).

## P006 — LinkBudgetX (compact, L1, no AI)

**Scope:** Deterministic, unit-aware FSO link-budget library: EIRP, free-space/geometric loss, divergence, pointing loss, atmospheric attenuation input, receiver sensitivity margin. Uncertainty propagation via first-order sensitivity. Every equation cited with units, assumptions, validity range.
**Validation (L1):** hand-calculated known-answer tests; unit tests; educational limitations documented.

## P007 — QuatKit (compact, L1, no AI)

**Scope:** Quaternion/attitude toolbox: normalization, composition, rotation of vectors, quaternion↔DCM↔Euler conversions, attitude-error angles, SLERP, angular-velocity kinematics integration.
**Validation (L1):** property-based tests (Hypothesis) against algebraic identities; known rotation test vectors; unit tests.

## P008 — CentroidNet (compact, L2, AI)

**Scope:** Quadrant-detector/focal-plane spot centroiding: synthetic spot-image generator (Gaussian spots, shot/read noise, background); analytic centroid + quad-cell baseline; small CNN estimator; accuracy-vs-SNR benchmark; per-estimate confidence output; model card.
**Validation (L2):** noise-free analytic recovery tests; bias/variance vs SNR curves compared with analytic centroid theory; reproducible training script.

## P009 — FogCast (compact, L2, AI)

**Scope:** Fog/aerosol optical attenuation prediction: Kim and Kruse empirical baselines (implemented, cited, validity-ranged); ML regression from visibility + wavelength + humidity features on synthetic/published-table data; prediction intervals; dataset card documenting limitations.
**Validation (L2):** baseline reproduction of published attenuation tables; held-out error analysis; documented regime limits.

## P010 — BERBench (compact, L2, no AI)

**Scope:** BER computation and Monte Carlo benchmark for OOK, PPM (M-ary), BPSK over AWGN and lognormal-fading channels; analytic BER expressions with citations; Monte Carlo cross-check harness; plots.
**Validation (L2):** Monte Carlo vs analytic agreement within statistical tolerance across SNR sweep; edge-case tests.

---

## Shared core (`shared/aerocore`)

Units/constants module, Gaussian-beam math, lognormal fading utilities, plotting style helpers. Versioned inside the mission repo; vendored into product repos at publication to keep them standalone.

## Repository plan

Publication layout (executed only after batch approval): `flagship-beamtwin`, `flagship-trackforge` standalone repos; `batch-01-suite` containing P003–P010. Repo creation currently owner-assisted (ADR-004).

## Completion gate

Each product must satisfy mission §17 before the batch readiness report can mark it READY FOR APPROVAL. AI products additionally satisfy §11 items 1–15.

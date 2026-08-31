# Batch 03 Specification — Aerospace GNC core: pointing, control and attitude determination

**Date:** 2026-08-31 · **Approval:** owner approved Batches 03–06 on 2026-08-31; publication is automatic under ADR-017, conditional on `scripts/release_gate.py` exiting 0.
**Composition:** 2 flagship, 3 medium, 5 compact · 7/10 AI-enabled · Levels: 2×L1, 6×L2, 2×L3
**Theme:** The GNC control and attitude-determination core. Batch 02 covered estimation (NavBench, EstimKit); Batch 03 covers what happens *after* you know your attitude — planning a slew, allocating it to actuators, steering the actuators, and detecting when one has failed — plus the static attitude-determination problem Batch 02 did not touch.
**Stack:** Python 3.11, NumPy/SciPy, scikit-learn (PyTorch unavailable), pytest + Hypothesis, Ruff. CLI + library API + plotting examples.
**Repositories:** one per product (ADR-018), named for the package, authored solely as `Om Acharya <145807881+OmAcharya-avtr@users.noreply.github.com>`.

All ten names checked against PyPI on 2026-08-31 (HTTP 404 on `/pypi/<name>/json`): `slewforge`, `cmgsteer`, `alloclab`, `detumblesim`, `fdiscope`, `wahbakit`, `disturbtorque`, `skymatch`, `momentummgr`, `keepout` — **all free**. `starid` and `starfix` were checked first and are **taken**; `skymatch` replaces them.

## Quota gap this batch closes

Before: 16 flagship, 24 medium, 40 compact remaining; ≥56 AI; L1 6, L2 48, L3 21, L4 5.
After: 14 flagship, 21 medium, 35 compact; ≥49 AI; L1 4, L2 42, L3 19, L4 5.
**Level 4 remains at zero and must begin no later than Batch 05.**

## Why these ten

Batches 01 and 02 established that the mission's credibility rests on picking problems that are genuinely underserved and then saying honestly where they are not. Attitude *estimation* is crowded — FilterPy, GTSAM, AHRS, Basilisk all live there, which is why EstimKit's README opens by sending readers to FilterPy. The control and actuator-management side is markedly thinner in open source: control allocation, CMG steering and singularity avoidance, momentum management and constrained slew planning are mostly locked inside vendor and prime-contractor toolchains. That is the gap this batch targets.

Every product below must still answer the ADR-018 question in its README: **what does this do that existing tools do not, and when should the reader use something else instead?** If a build agent cannot answer that honestly for its product, it must say so in the readiness report rather than pad the alternatives table.

---

## P021 — SlewForge (flagship, L3, AI)

**Problem:** A spacecraft slew is not just "rotate from A to B". The path must avoid pointing sensitive instruments at the Sun, Earth or Moon, respect wheel torque and momentum limits, and finish in bounded time. Engineers size this with spreadsheets and eigenaxis rules of thumb that silently violate keep-out cones.

**Scope:** Rigid-body attitude dynamics with reaction-wheel actuation and saturation; eigenaxis, bang-bang and smoothed rest-to-rest slew profiles; hard keep-out cone constraints (Sun, Earth, Moon) with violation detection along the path; a constrained planner that returns a feasible path or an explicit infeasibility reason; time and momentum cost accounting. AI element: a learned warm-start that predicts an initial path parameterisation from the boundary conditions and constraint geometry, benchmarked against cold-started optimisation on solve time **and** solution quality, with a confidence output. Report honestly if the warm start does not reduce solve time or degrades the objective.

**Level 3 requirements:** `docs/REQUIREMENTS.md`, 12–18 numbered requirements with a verification matrix; uncertainty analysis; regression suite with pinned seeded outputs; performance benchmark; failure-mode tests (infeasible geometry, wheel saturation mid-slew, cone tangency, degenerate 180-degree slew).

**Validation:** eigenaxis slew reproduces the analytic rest-to-rest time for a known inertia and torque limit; angular momentum conserved to numerical tolerance in the torque-free case; keep-out violation detection verified against closed-form cone-intersection geometry; planner never returns a path that violates a cone (property test over random geometries).

## P022 — CMGSteer (flagship, L3, AI)

**Problem:** Control-moment gyros deliver large torque cheaply and then trap you in singular gimbal configurations where commanded torque cannot be produced in some direction. Steering-law choice is the whole engineering problem, and the open-source options are close to nonexistent.

**Scope:** Pyramid and roof CMG array geometries; Jacobian formulation and singularity measure; steering laws — Moore-Penrose pseudo-inverse, singularity-robust (SR) inverse, generalised SR, and null-motion gimbal reconfiguration; singularity surface mapping and escape behaviour; torque error accounting through singular regions. AI element: a learned null-motion policy that reconfigures gimbals to maximise future manoeuvrability, benchmarked against classical null-motion and against plain SR-inverse over seeded manoeuvre suites, with confidence output. The plausible honest outcome is that SR-inverse is competitive; report what is measured.

**Level 3 requirements:** same battery as P021, with failure modes covering internal and external singularities, gimbal-rate saturation and array degeneracy after a CMG failure.

**Validation:** Jacobian verified against numerical differentiation of the momentum map; pseudo-inverse reproduces the exact torque away from singularities to numerical tolerance; singularity measure goes to zero on analytically known singular configurations; SR-inverse torque error matches its closed-form expression as a function of the robustness parameter.

## P023 — AllocLab (medium, L2, AI)

**Problem:** Control allocation — turning a desired body torque into individual thruster or wheel commands — is where saturation, redundancy and failure tolerance actually bite, and it is usually written ad hoc per project.

**Scope:** Effector configuration models (thruster clusters, reaction wheels); allocation methods — pseudo-inverse, weighted pseudo-inverse, linear-programming and quadratic-programming allocation with actuator bounds; saturation and effector-failure handling with graceful reallocation; attainable-moment-set computation. AI element: a learned allocator trained to approximate the QP solution, benchmarked against the exact QP on allocation error, constraint satisfaction and runtime, with confidence. The honest expectation is that the learned allocator trades exactness for speed; measure both.

**Validation:** allocation exactly reproduces the commanded torque within the attainable set; the attainable moment set matches its closed-form vertices for a simple configuration; every allocation respects actuator bounds (property test); failed-effector reallocation still meets the command when the remaining set can produce it, and reports infeasibility when it cannot.

## P024 — DetumbleSim (medium, L2, AI)

**Problem:** Post-deployment detumbling with magnetorquers is a rite of passage for every small satellite, and the B-dot gain is usually picked by folklore and then discovered to be wrong in orbit.

**Scope:** Rigid-body dynamics with a magnetic field model along an orbit (IGRF-style dipole with cited coefficients); B-dot and cross-product control laws; magnetorquer saturation and dipole limits; detumble time as a function of gain, inertia and orbit; eclipse and field-geometry effects on controllability. AI element: a learned gain scheduler that adapts the B-dot gain from measured rate and field history, benchmarked against a fixed hand-tuned gain across inertia and initial-rate sweeps, with confidence. Report honestly if the fixed gain wins.

**Validation:** angular momentum decreases monotonically under B-dot in the absence of saturation (property test); detumble time scales with gain as the analytic first-order expression predicts; the magnetic field model reproduces published field magnitudes at reference altitudes and latitudes; the known controllability gap along the field direction is demonstrated and quantified rather than hidden.

## P025 — FDIScope (medium, L2, AI)

**Problem:** A GNC fault that is detected late is a mission loss. Residual-based fault detection is textbook, but the threshold selection that decides whether it works is not, and false-alarm behaviour is rarely characterised.

**Scope:** Fault injection into a GNC loop — sensor bias, drift, stuck value, dropout; actuator loss of effectiveness, stuck and runaway; residual generation from a filter innovation sequence; classical detection by chi-squared and CUSUM tests with explicit false-alarm rate design; isolation logic. AI element: a learned classifier over residual features that detects and isolates fault type, benchmarked against the chi-squared and CUSUM baselines on detection delay, false-alarm rate and isolation accuracy, with confidence. ROC curves for every method; report where the classical test wins.

**Validation:** the chi-squared test's empirical false-alarm rate matches its design value under the fault-free hypothesis; detection delay for a step bias matches the analytic CUSUM expectation; isolation confusion matrix reported in full, not summarised to a single accuracy number.

## P026 — WahbaKit (compact, L1, no AI)

**Scope:** Static attitude determination from vector observations — Wahba's problem solved by TRIAD, the q-method (Davenport), QUEST and OLAE, with attitude covariance from the measurement covariances. Every convention stated: frame order, quaternion sign, and what each method does when observations are nearly parallel.

**Positioning note for the README:** `AHRS` implements several of these. The narrow claim here is the covariance output, the explicit degeneracy handling, and property tests against the analytic solution — not novelty. Check what `AHRS` actually ships before writing the alternatives table.

**Validation (L1):** all four methods agree to numerical tolerance on well-conditioned synthetic problems; each reproduces the exact attitude for noise-free observations; QUEST matches the q-method eigenvector solution; attitude covariance matches the Monte Carlo covariance over seeded trials; near-parallel observations raise a documented error rather than returning silently wrong attitude.

## P027 — DisturbTorque (compact, L2, no AI)

**Scope:** Environmental disturbance torque models for spacecraft attitude sizing — gravity gradient, aerodynamic, solar radiation pressure, and residual magnetic dipole — each with its source, units, assumptions and validity range, plus secular and cyclic components over an orbit and the momentum accumulation that follows from them. This is the deterministic reference the momentum-management product is sized against.

**Validation (L2):** each torque expression checked by hand for a simple geometry where a closed form exists; gravity-gradient torque reproduces the analytic maximum at 45 degrees off nadir; momentum accumulated per orbit matches direct integration of the torque profile; magnitudes for a representative LEO smallsat fall inside the ranges quoted in the standard references.

## P028 — SkyMatch (compact, L2, AI)

**Scope:** Lost-in-space star identification — star catalogue preparation, triangle and pyramid pattern matching with tolerance handling, and false-match rejection; performance against catalogue magnitude limit, centroid noise and false stars. AI element: a learned candidate ranker over pattern features, benchmarked against the classical pyramid matcher on identification rate and false-identification rate, with confidence. `starid` and `starfix` are taken on PyPI; this product is `skymatch`.

**Validation:** identification rate versus centroid noise compared with the classical algorithm's published behaviour; the false-identification rate measured explicitly rather than assumed zero; a documented failure regime where both methods fail (dense false-star fields), quantified.

## P029 — MomentumMgr (compact, L2, AI)

**Scope:** Reaction-wheel momentum management — momentum accumulation from the disturbance environment, desaturation planning with magnetorquers or thrusters, wheel-speed zero-crossing avoidance, and desaturation scheduling over an orbit. AI element: a learned desaturation scheduler benchmarked against a fixed-threshold classical scheduler on propellant or dipole cost and on time spent near saturation, with confidence.

**Validation:** momentum accumulation matches integration of the DisturbTorque profile for the same environment (cross-check between products, implemented independently); desaturation with a magnetic dipole respects the instantaneous field-direction controllability constraint; a fixed-threshold baseline is reported alongside the learned scheduler with confidence intervals, and differences inside those intervals are reported as indistinguishable.

## P030 — KeepOut (compact, L1, no AI)

**Scope:** Celestial keep-out geometry — Sun, Earth and Moon exclusion cones for sensitive instruments; violation testing for a given attitude; allowed-attitude region computation; and keep-out-aware pointing windows over an orbit. The geometric companion to SlewForge, deliberately separable so it can be used without the planner.

**Validation (L1):** cone-intersection tests verified against closed-form spherical geometry; the Earth's angular radius at a given altitude matches the analytic expression; a hand-computed case with two overlapping cones reproduced exactly; property tests that the violation test is invariant under rotation of the whole geometry.

---

## Shared conventions

Batch 03 products use the quaternion and frame conventions established by QuatKit (P007) and the estimator conventions of EstimKit (P017). **No cross-product imports.** Each product implements independently and cites the sibling as related work, so every repository stays independently installable.

Cross-checks are deliberate where they are cheap: P029 MomentumMgr must reproduce P027 DisturbTorque's momentum accumulation from an independently implemented torque model, and P021 SlewForge must agree with P030 KeepOut on cone violation for identical geometry. Disagreement between two independent implementations is a finding, not a nuisance.

## Completion gate

Mission §17 for every product; §11 items 1–15 additionally for the seven AI products. `scripts/release_gate.py` must exit 0 before anything is published, and every product ships its own repository with a README to `templates/REPO_README_STANDARD.md`.

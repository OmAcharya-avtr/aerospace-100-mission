# Batch 09 Specification — Attribution: which part is wrong, and what you cannot tell from the data you have

Written 2026-10-28. Binding for every build agent in this batch.

Products P081–P090. Classes: 2 flagship, 3 medium, 5 compact. AI: 7 of 10.
Validation: 3 at L3, 7 at L2.

## The quota this batch is built against

Measured by `scripts/quota_report.py` at the start of this session, after P080
BoundCompress was published and Batch 08 closed at ten of ten:

```
Registered: 80 / 100     Built: 80 / 100     Published: 80 / 100

Quota                   Target   Built  Remaining
flagship                    20      16          4
medium                      30      24          6
compact                     50      40         10
ai (minimum)                70      57         13
L1                          10      10          0
L2                          60      48         12
L3                          25      22          3
L4                           5       0          5
```

Twenty products remain and they divide exactly: Batch 09 and Batch 10 are each
2 flagship, 3 medium, 5 compact. Batch 09 takes **all three remaining L3 slots**
and seven L2, closing L3 at 25/25 and bringing L2 to 55/60. Batch 10 then takes
the last 5 L2 and the 5 L4 slots. AI: 7 here, 6 in Batch 10, closing at 70/100.
L1 is CLOSED at 10/10 — specify no further L1 product.

## Level 4 — why this batch deliberately builds none, stated plainly

The standing instruction is to start Level 4 in the next batch by building the
hardware abstraction layer, simulation mode, dry-run mode, deployment and
recovery procedures and the benchmark harness in the cloud, and labelling the
result `Level 3, hardware-pending` until measured timing and resource use come
from the Jetson Orin Nano itself.

**That scaffolding work is already finished, five times over.** P031 HilForge,
P033 EdgeInfer, P041 CodedFade, P043 PhotonCount and P079 TimeTransfer each
stand complete to Level 3 with the full L4 scaffolding present and exercised by
test, each labelled `Level 3, hardware-pending`, each blocked on the identical
thing: measurements from the board. The L4 quota is five. **Five products are
already waiting for it.**

So building a sixth scaffolded product would not advance the quota by one. It
would add a sixth product to the queue behind the same blocker, consume a
validation slot, and let the batch report the word "L4" without moving the
number. The L4 count has been 0 of 5 for eighteen consecutive batches and not
one of those batches was short of scaffolding.

**The honest arithmetic: this mission's validation quota cannot close without
the board.** Five products must reach L4; L4 requires measured timing and
resource use from the Jetson Orin Nano; no cloud session can produce that
number and no session may relabel on simulated numbers. Batch 09 therefore
spends its three remaining L3 slots on products that can actually reach L3
tonight, and the five L4 slots stay deliberately empty and visibly unfilled in
Batch 10's allocation until the board is reachable.

**What this needs from the owner:** the Jetson Orin Nano powered on and
reachable, its hostname or IP, and passwordless SSH from the Mac. With that,
the five products already waiting reach L4 in one session and the quota closes.

## Why this batch, stated against what already exists

Eighty products in, the mission has good coverage of *computing* a quantity and
*checking* whether a quantity is in limits. What it is thin on is the question
an engineer actually asks when a number comes out wrong: **which part of my
system is responsible, and is that even answerable from the data I have?**

That second clause is the batch's organising idea. Every product here is
required to report what it *cannot* determine, as a first-class output rather
than a caveat:

- P081 reports which twin parameters are **structurally unidentifiable** from
  the telemetry set provided, before attributing anything to the rest.
- P082 reports how much of a site-diversity availability figure is **dominated
  by climatology uncertainty** rather than by the sites being compared.
- P083 reports which real-time deadline misses **actually changed the outcome**,
  against the large majority that did not.
- P084 reports **how many more field measurements** would be needed to close a
  simulation-to-field gap to a target, which is the useful form of "your sim is
  not validated yet".
- P085 reports which pointing-model terms are **mutually unidentifiable** from
  the residual set provided, which is the standard way a pointing budget gets
  silently misattributed.
- P086, P088 and P090 each separate "the data is bad" from "the data is missing"
  from "the data is fine and your reader is wrong".

A tool that returns an attribution without an identifiability statement is the
failure mode this batch exists to avoid. It is the same class of error as
P071's dilution region: a confident scalar computed from information that does
not support it.

## PyPI name check — method and result

Every package name was probed against the live index with
`pip index versions <name>` on **2026-10-28**, before being written into this
specification. The probe was validated in the same run against two known-TAKEN
controls and one known-FREE control, so a blanket "free" answer could not pass
undetected:

- controls TAKEN: `numpy` (2.5.4), `sgp4` (2.27)
- control FREE: `beamtwin` (the mission's own unpublished P001 package name)

All ten Batch 09 names returned **FREE**: `twindiverge`, `sitediverse`,
`hilverdict`, `transfergap`, `pointmodel`, `telemgap`, `refractpath`,
`tmtcdiff`, `aligncal`, `tlecheck`.

No name is reserved by this check and none is claimed on PyPI by this mission.
A build agent that wants to change a package name must re-probe it and record
the result; it may not assume.

## P081 TwinDiverge — flagship · L3 · AI · digital-twins

Package `twindiverge`. **When a digital twin and the real system disagree, say
which parameter is responsible — and say first which parameters cannot be
blamed from the data you have.**

Takes a parameterised twin (a forward model plus a declared parameter vector
with nominal values and bounds), a telemetry record from the real system, and
the input history both saw. Produces a residual decomposition, an attribution
of the residual to parameter subsets, and — **as the first output, before any
attribution** — a structural identifiability verdict per parameter given that
telemetry set.

Why this is a flagship and not a curve-fitting utility: parameter estimation
over a twin is routinely run on parameter sets that the available telemetry
cannot separate. Two parameters that enter the model through the same path are
not individually recoverable at any noise level, and least squares will still
return a confident split between them. **The split is an artefact of the
regulariser, not a measurement**, and a tool that prints it without the
identifiability analysis is manufacturing attribution.

- **Baseline 1 (identifiability, analytic):** local structural identifiability
  from the rank and conditioning of the sensitivity matrix, with the Fisher
  information matrix and its Cramér–Rao bound per parameter; near-singular
  directions reported as the *parameter combinations* that are identifiable
  rather than as individual parameters. Practical identifiability separated
  from structural identifiability at the declared noise level.
- **Baseline 2 (attribution, analytic):** first-order sensitivity attribution
  and generalised least squares on the identifiable subspace only, with
  confidence regions. This is the operational default.
- **Baseline 3 (ground truth):** profile-likelihood over each parameter,
  computed directly, as the reference the analytic covariance is scored
  against. Where the profile likelihood is flat, the parameter is unidentifiable
  regardless of what the covariance says, and that disagreement is a headline
  result rather than a footnote.
- **AI component:** a learned classifier that maps a residual time-series
  signature to the responsible parameter subset — the operational problem of
  triaging many twins without running a full estimation on each. Scored against
  the analytic sensitivity attribution on **accuracy at fixed computation
  budget**, and separately on its behaviour inside the unidentifiable subspace,
  where the honest expected result is that **the learned model produces a
  confident wrong attribution precisely where the analytic method correctly
  refuses to answer.** That outcome, if measured, is the most useful finding in
  the product and is published as such.
- **Validation (L3):** single- and multi-parameter faults injected at known
  magnitudes across a swept grid of noise level, telemetry subset and fault
  magnitude, with attribution accuracy reported as measured curves against the
  Cramér–Rao bound; a deliberately unidentifiable parameter pair constructed in
  closed form, verified unidentifiable by profile likelihood, and shown to
  defeat the learned attributor; identifiability verdicts cross-checked against
  the analytically known rank of the constructed cases.
- **Alternatives table must name:** `scipy.optimize`, `lmfit`, `SALib`, and
  `emcee`. Say plainly that these do the fitting, the sensitivity analysis and
  the sampling respectively and do them well, and that this product's
  contribution is **the identifiability gate in front of the fit and the
  attribution of a residual to a parameter subset**, not a better optimiser.
- **Must not say:** that an attribution identifies a physical root cause, or
  that a passing identifiability check validates the twin.

## P082 SiteDiverse — flagship · L3 · AI · atmospheric-propagation

Package `sitediverse`. **Optical ground-station site diversity availability
under measured spatial correlation, against the independence assumption that
overstates it.**

Takes candidate ground-station sites with their cloud and turbulence
climatology records, a satellite or constellation pass geometry, and a declared
link availability requirement. Produces the achieved availability for each
site subset, the **diversity gain** of adding each site, and the share of the
answer attributable to climatology uncertainty rather than to site choice.

Why this is a flagship: optical ground segment design turns on diversity gain,
and the number is routinely computed by multiplying marginal cloud
probabilities as if sites were independent. **They are not.** Sites within a few
hundred kilometres share weather systems, and the independence calculation
overstates the gain by an amount that grows with the availability requirement —
exactly the regime the requirement is written for. The product's headline output
is that overstatement, measured.

- **Baseline 1 (the naive calculation, implemented deliberately):** product of
  marginal unavailabilities under full independence. Implemented and reported
  not as a competitor but as the thing being refuted, because it is what the
  practice does.
- **Baseline 2 (analytic, correlated):** Gaussian-copula and fitted
  exponential correlation-length models over site separation, giving joint
  unavailability in closed form. The operational default.
- **Baseline 3 (ground truth):** direct empirical joint availability computed
  from the concurrent climatology record itself, with bootstrap confidence
  intervals. This is what both analytic paths are scored against; the record is
  the reference, not a competitor.
- **AI component:** a learned spatio-temporal cloud-field correlation model
  predicting joint unavailability for site subsets, scored against the fitted
  exponential correlation-length model on held-out site subsets and held-out
  time periods, with the **held-out-site split reported separately from the
  held-out-time split** because they measure different things and a model can
  pass one while failing the other. The honest expected result is that the
  fitted correlation-length model is competitive, and that the learned model's
  margin — if any — is smaller than the bootstrap confidence interval on the
  empirical reference.
- **Validation (L3):** analytic and learned joint availability against the
  empirical record across a swept grid of site separation, availability
  requirement and season, with the independence assumption's **overstatement
  reported as a measured curve against site separation**; the correlation length
  fitted and its confidence interval reported; a constructed fully-correlated
  and a constructed fully-independent case verified to reproduce their analytic
  limits exactly; the climatology-uncertainty share computed by propagating the
  record's own sampling uncertainty, and reported beside every availability
  figure.
- **Alternatives table must name:** `astropy`, `pvlib`, `xarray` and
  `scikit-learn`. Say plainly that the climatology data itself comes from
  sources this product does not ship — reanalysis and satellite cloud products
  obtained separately — and that the product's contribution is **the
  correlated-availability calculation and the diversity-gain overstatement
  measurement on climatology you already have**, not a weather model.
- **Must not say:** that any availability figure supports a site-acquisition or
  capital decision; that the climatology is predictive of a particular future
  period.

## P083 HilVerdict — medium · L3 · AI · hardware-in-the-loop

Package `hilverdict`. **After a hardware-in-the-loop run, decide whether the
run was timing-faithful enough for its result to mean anything — and which of
the deadline misses actually changed the outcome.**

Takes a HIL run log with host and device timestamps, declared frame period and
deadlines, and the resulting trajectory or signal record. Produces a validity
certificate: deadline misses, jitter, host-to-device clock drift, and for each
miss an assessment of whether it **measurably changed the trajectory**.

Why this exists: HIL tools report deadline misses, and engineers learn to
ignore the report because almost all misses are harmless. A tool that flags
every miss equally trains its own users to dismiss it, so the misses that do
matter are dismissed with the rest. The useful output is the separation.

- **Baseline 1 (analytic):** closed-form bound on trajectory divergence from a
  deadline miss of given duration, derived from the declared plant dynamics and
  its Lipschitz constant / local linearisation. A miss whose bound is below the
  declared tolerance **cannot** have mattered, and that is a proof rather than
  an estimate. Expected to dispose of the large majority of misses.
- **Baseline 2:** direct re-simulation of the run with and without the miss,
  for misses the bound does not dispose of. Slow, exact, and the reference.
- **AI component:** a learned classifier predicting which misses matter from
  cheap log features alone, for the case where re-simulation is unaffordable.
  Scored against the analytic bound on **recall of consequential misses**,
  because dismissing a miss that mattered is not symmetric with re-simulating
  one that did not. **The honest expected result is that the analytic bound
  wins outright** — it is a proof, it is cheap, and the learned model can only
  approximate it. Report that plainly if measured; the learned component's
  defensible role is the residual set the bound leaves undecided, and if it adds
  nothing there, say so.
- **Validation (L3):** injected deadline misses of swept duration and phase
  against re-simulated ground truth, with the analytic bound's tightness
  reported as a measured curve; clock-drift estimation verified against
  synthetic drift of known rate and against a deliberate clock reset;
  classifier recall reported at fixed re-simulation budget; a constructed case
  where a single short miss *does* change the outcome, verified by
  re-simulation, included as a negative control the bound must not dispose of.
- **Alternatives table must name:** `pytest`, `numpy`, and say plainly that
  real-time frameworks and vendor HIL toolchains own the *execution* and the
  deadline reporting, and that this product's contribution is the **post-hoc
  consequence analysis over a log those tools produced**, not a real-time
  executive.
- **Must not say:** that a validity certificate qualifies a run for
  certification evidence.

## P084 TransferGap — medium · L2 · AI · digital-twins

Package `transfergap`. **Bound the gap between a model validated in simulation
and the field, from the few field measurements you actually have — and say how
many more would close it to a target.**

Takes simulation predictions, a small set of paired field measurements, and a
declared target gap with a confidence level. Produces a bound on the
simulation-to-field discrepancy, and the **additional sample count** needed to
tighten the bound to the target.

Why this exists: the question "is my simulation validated" has no useful answer,
and the question "how many more flight tests do I need" has a precise one. The
second is what gets asked in planning and what gets answered by guesswork.

- **Baseline 1 (analytic, distribution-free):** Hoeffding and empirical-Bernstein
  bounds on the mean discrepancy, plus a one-sided tolerance interval, with the
  sample-count requirement inverted in closed form from each. The operational
  default, and honest about being conservative.
- **Baseline 2:** bootstrap and conformal prediction over the paired
  measurements, giving a tighter interval under exchangeability, with the
  exchangeability assumption stated and **tested** rather than assumed.
- **AI component:** a learned discrepancy model mapping operating-condition
  features to expected sim-field gap, which — if it fits — reduces the required
  sample count by explaining part of the discrepancy rather than bounding all of
  it. Scored against the distribution-free bounds on **achieved coverage at the
  declared confidence**, not on tightness, because a tighter interval that
  under-covers is worse than a loose one that holds. The honest expected result
  at realistic field-sample sizes (tens, not thousands) is that the learned
  model **under-covers and the distribution-free bound is the only trustworthy
  answer**; report the measured coverage either way.
- **Validation (L2):** coverage measured against synthetic ground truth across a
  swept grid of field-sample count, true gap magnitude and noise level, with
  every interval's achieved coverage reported against its nominal level;
  sample-count predictions verified by drawing the predicted number of samples
  and checking the bound actually reaches the target at the stated rate;
  exchangeability violated deliberately by ordering the field samples by
  operating condition, to show the conformal interval's coverage break.
- **Alternatives table must name:** `scipy.stats`, `statsmodels`,
  `scikit-learn` and `mapie`. Say plainly that `mapie` is the mature conformal
  prediction implementation and should be used for conformal work directly, and
  that this product's contribution is **the sample-count inversion and the
  coverage comparison across bound families on a sim-versus-field pairing**.
- **Must not say:** that a bound validates a model, or that reaching a target
  gap qualifies a simulation for any purpose.

## P085 PointModel — medium · L2 · no AI · pointing-acquisition-tracking

Package `pointmodel`. **Decompose a pointing error budget from measured
residuals, with a confidence interval on every term and a statement of which
terms your residual set cannot separate.**

Takes pointing residuals over an observed sky or pass distribution and a
declared pointing model (mount misalignment, encoder offsets and scale,
non-perpendicularity, gravitational droop, refraction, and declared custom
terms). Produces fitted coefficients with confidence intervals, the residual
after fitting, the dominant term, and the **mutual identifiability** of the
terms given the observed geometry distribution.

Why this is a medium and has no AI component: the problem is a linear fit with a
known design matrix, and the hard part is entirely in the conditioning. Terms in
a standard pointing model become mutually unidentifiable when the observed
geometry is restricted — a pass set clustered in elevation cannot separate
droop from a collimation error — and the fit returns a confident split anyway.
**No learned component would improve this and one is deliberately not built**;
adding a model where the correct answer is a rank analysis would be the error
this batch is about.

- **Baseline 1 (analytic):** weighted least squares over the design matrix with
  full covariance, condition number, variance inflation per term, and the
  identifiable linear combinations reported where individual terms are not
  separable.
- **Baseline 2:** robust (Huber and RANSAC-style) fitting, because real residual
  sets contain outliers from acquisition failures, with the two fits'
  disagreement reported as a diagnostic rather than one being silently
  preferred.
- **Cross-check:** term-by-term recovery against synthetic residual sets
  generated from known coefficients, which is the only way to verify an
  attribution is real.
- **Validation (L2):** coefficient recovery across a swept grid of geometry
  coverage, noise level and outlier fraction, with recovery error reported
  against the Cramér–Rao bound; a deliberately rank-deficient geometry
  distribution constructed in closed form, verified rank-deficient, and shown
  to produce a confident wrong per-term split under plain least squares while
  the identifiability report correctly refuses it; refraction term verified
  against an independent refraction calculation rather than against itself.
- **Alternatives table must name:** `astropy`, `numpy.linalg`, `statsmodels`,
  and **TPOINT**, named explicitly as the mature commercial telescope pointing
  package that is the right tool for observatory pointing work. Say plainly when
  to use TPOINT instead: anything involving a real observatory mount with vendor
  support. This product's contribution is **open, scriptable, with an
  identifiability report on every fit**.
- **Must not say:** that a fitted model is suitable for operational pointing
  correction on a real mount.

## P086 TelemGap — compact · L2 · AI · space-operations

Package `telemgap`. **Separate missing telemetry from corrupt telemetry from a
reader that is wrong, and emit a defensible gap policy rather than a silent
interpolation.**

Takes a decoded telemetry record with timestamps and produces an integrity
report: gaps with their durations, duplicate and out-of-order frames, clock
resets and backward jumps, and a classification of each anomaly into *missing*,
*corrupt*, or *correctly-received-and-misparsed*. Emits the gap-handling policy
explicitly — which gaps may be interpolated across at a declared tolerance,
which must be left as gaps, and why.

Why this exists: every analysis downstream of a telemetry archive silently
interpolates. The interpolation is usually fine and occasionally invents a
signal across a six-hour outage. The product's contribution is making the
policy explicit and refusable.

- **Baseline (rules, analytic):** explicit rule screens over timestamp
  monotonicity, expected frame cadence, counter continuity and declared value
  ranges. The operational default, and expected to be hard to beat.
- **AI component:** a learned anomaly classifier over frame-level features for
  the ambiguous cases the rules leave undecided. Scored against the rule screens
  on **precision and recall per anomaly class**. The honest expected result is
  that **the rules win on every class they cover**, because the classes are
  defined by rules; the learned model's only defensible contribution is the
  residual ambiguous set, and if it does not beat a majority-class guess there,
  that is the published result.
- **Validation (L2):** injected gaps, duplicates, reorderings and clock resets
  of known type and magnitude into synthetic records, with per-class precision
  and recall measured across a swept grid of anomaly rate and cadence jitter;
  interpolation-tolerance policy verified by interpolating across injected gaps
  of known shape and measuring the actual reconstruction error against the
  declared tolerance; a constructed long-outage case the policy must refuse to
  interpolate, verified to be refused.
- **Alternatives table must name:** `pandas`, `numpy` and `xarray`. Say plainly
  that `pandas` resampling and interpolation do the mechanical work well, and
  that this product's contribution is **the integrity classification and the
  explicit refusal policy in front of them**.
- **Must not say:** that a clean integrity report means the telemetry is
  correct.

## P087 RefractPath — compact · L2 · no AI · atmospheric-propagation

Package `refractpath`. **Slant-path atmospheric refraction with the
disagreement between standard models reported against elevation, because that
is where the pointing error you cannot explain comes from.**

Takes an observer location, an apparent or true elevation, and an atmospheric
state (pressure, temperature, humidity, wavelength). Produces the refraction
correction by several independent standard methods, their disagreement, and the
elevation below which the disagreement exceeds a declared pointing tolerance.

No AI component, deliberately: this is a well-posed physical calculation with
published reference values, and the useful output is the **cross-model
disagreement curve**, not a prediction.

- **Baseline 1:** the simple analytic/plane-parallel and standard
  low-precision formulations, implemented as the thing that is actually used.
- **Baseline 2:** a ray-trace through a layered refractivity profile with the
  refractivity from standard dispersion formulae, which is the reference the
  closed forms are scored against.
- **Cross-check (ground truth):** published refraction tables and the standard
  reference values at stated conditions, which the implementations must
  reproduce to a stated tolerance. **This is a verification against values the
  product did not generate**, which is the only kind that counts here.
- **Validation (L2):** every method against published reference values across a
  swept grid of elevation, pressure, temperature, humidity and wavelength, with
  agreement reported as measured error curves and the stated tolerance met or
  the failure disclosed; the inter-model disagreement reported as a curve
  against elevation with the **tolerance-crossing elevation measured, not
  asserted**; the low-elevation regime where all closed forms fail exhibited
  explicitly rather than extrapolated into; wavelength dependence verified
  across the optical band against the dispersion formula independently.
- **Alternatives table must name:** `astropy` (whose `AltAz` frame applies
  refraction and is the right tool for astrometric work and should be used for
  it), `pyephem`/`ephem`, and `erfa`. This product's contribution is **the
  cross-model disagreement measurement and the explicit validity floor**, not a
  better single number.
- **Must not say:** that any correction is suitable for operational pointing
  below the measured validity floor.

## P088 TmTcDiff — compact · L2 · no AI · space-operations

Package `tmtcdiff`. **Diff two versions of a telemetry and command dictionary
and report which existing procedures and analyses break.**

Takes two versions of a telemetry/command definition set and, optionally, a
corpus of procedures, scripts or analysis configurations that reference it.
Produces a semantic diff — added, removed, renamed, retyped, rescaled,
re-enumerated and bit-shifted items — and an **impact report** naming which
references in the corpus are now broken, silently changed, or unaffected.

Why this is distinct from a text diff: the dangerous change is not the removed
mnemonic, which fails loudly. It is the one whose scale factor, unit, enum
mapping or bit offset changed while the name stayed the same. Everything
downstream keeps working and quietly means something else.

No AI component, deliberately: this is a structural comparison with a correct
answer.

- **Baseline:** structural comparison over parsed definitions with explicit
  change classes, and reference resolution against the corpus by name and by
  path.
- **The hard part, and the measured one:** **silent semantic change detection**
  — same name, different meaning — reported separately from, and ranked above,
  loud breakage. Rename detection by structural similarity with its false-match
  rate measured rather than assumed.
- **Validation (L2):** dictionary pairs constructed with known injected changes
  of every class, with per-class detection precision and recall measured; the
  silent-change class measured separately and reported first; rename detection's
  false-match rate measured across a swept grid of similarity threshold and
  dictionary size, with the threshold choice justified by the measured curve
  rather than picked; a constructed pair with **no** semantic change but
  substantial textual reordering, verified to produce an empty semantic diff, as
  the negative control.
- **Alternatives table must name:** `deepdiff`, `jsonschema` and `difflib`. Say
  plainly that `deepdiff` does general structured diffing well and should be
  used for that, and that this product's contribution is **the change-class
  taxonomy for telemetry definitions and the silent-semantic-change ranking**.
- **Must not say:** that an empty impact report means an upgrade is safe.

## P089 AlignCal — compact · L2 · AI · aerospace-gnc

Package `aligncal`. **Estimate relative sensor alignment from flight data, with
the observability of each alignment degree of freedom reported before the
estimate.**

Takes time-series measurements from two or more attitude-relevant sensors with
a declared mounting geometry, and estimates the relative misalignment rotations
with confidence intervals. Reports, first, which alignment degrees of freedom
the observed motion actually **excites** — a vehicle that never rotated about an
axis cannot have its alignment about that axis estimated, at any data volume.

Distinct from the existing single-frame attitude products in this mission:
P026 WahbaKit solves attitude from vector pairs at one epoch, P017 EstimKit is
general estimation. This is the **time-series relative alignment** problem,
where the failure mode is unexcited degrees of freedom.

- **Baseline 1 (analytic):** batch least-squares / Wahba-type solution over the
  full record on the observable subspace only, with the observability Gramian
  computed and reported per degree of freedom, and the unobservable directions
  named as the rotation axes they correspond to.
- **Baseline 2:** recursive (Kalman / MEKF) alignment estimation, giving the
  same answer online, with the batch solution as the reference it is scored
  against.
- **AI component:** a learned estimator mapping motion features to alignment
  corrections, for the short-record case. Scored against the batch solution on
  **error against known injected misalignment**, and separately on its behaviour
  in the unobservable subspace, where the honest expected outcome is that it
  **returns a confident estimate for a degree of freedom the data cannot
  support** while the analytic method reports it unobservable. If measured, that
  is the product's most useful result.
- **Validation (L2):** known misalignments injected across a swept grid of
  excitation richness, noise level and record length, with recovery error
  reported against the Cramér–Rao bound; a deliberately single-axis-rotation
  record constructed, verified rank-deficient by its observability Gramian, and
  shown to defeat the learned estimator; batch and recursive solutions verified
  to agree to a stated tolerance on the observable subspace.
- **Alternatives table must name:** `scipy.spatial.transform`,
  `numpy.linalg`, and `filterpy`. This product's contribution is **the
  observability report in front of the alignment estimate**.
- **Must not say:** that an alignment estimate may be applied to flight
  software, or that an observable alignment is a calibrated one.

## P090 TleCheck — compact · L2 · AI · aerospace-gnc

Package `tlecheck`. **Screen a two-line element set before you propagate it,
and say which of its problems the propagator will not tell you about.**

Takes TLEs, singly or as a catalogue, optionally from more than one source for
the same object. Produces a quality verdict per element set: format and checksum
validity, epoch staleness against a declared requirement, element plausibility
against physical bounds and against the object's own history, cross-source
disagreement, and a classification of each problem as *will fail loudly*,
*will propagate to a wrong answer silently*, or *benign*.

Why this exists: `sgp4` parses and propagates a TLE and does both well. It does
not tell you the element set is three weeks stale, that the drag term flipped
sign, that two sources disagree by kilometres, or that an element is physically
implausible. Those are the inputs that produce a confident wrong ephemeris.

- **Baseline (rules, analytic):** checksum and format validation, epoch age
  against requirement, element bounds from orbital mechanics, and consistency
  against the object's own element history with an explicit outlier test. The
  operational default, expected to be strong.
- **AI component:** a learned screening classifier over element and history
  features for the *silently wrong* class, which is the class rules cover worst.
  Scored against the rule screens on **recall at fixed false-positive rate**,
  because a silently bad element set that passes screening is the whole failure
  this product addresses. The honest expected result is that **the explicit rule
  screens match or beat it on every class they cover**, and the learned model's
  only defensible contribution is the silent class; report it either way, and
  report it against a majority-class baseline so a useless model cannot look
  useful.
- **Validation (L2):** element sets corrupted in known ways — checksum, field
  shift, sign flip, implausible eccentricity, stale epoch, cross-source
  divergence — with per-class precision and recall measured across a swept grid
  of corruption rate; the *silently wrong* classification verified by actually
  propagating the corrupted set with `sgp4` and measuring the resulting position
  error against the uncorrupted propagation, so the class label is **measured
  consequence rather than assumed**; a constructed valid-but-stale set and a
  constructed invalid-checksum set as the two controls.
- **Alternatives table must name:** `sgp4` (named as the correct and mature
  propagator, which this product screens *for* and does not replace),
  `astropy`, and `skyfield`. Say plainly that the authoritative element source
  is a catalogue this product does not ship.
- **Must not say:** that a passing screen means a propagated ephemeris is
  accurate.

## Binding rules for every build agent in this batch

Read `templates/PRODUCT_BUILD_GUIDE.md` first, in full. In addition:

1. **One shared pre-pinned interpreter, and no agent runs `pip` at all.**
   The coordinator created and froze one virtual environment before any agent
   was dispatched; the freeze is committed at
   `shared/pinned_env_2026-10-28.txt`. Use that interpreter read-only. **Running
   `pip` in any form is forbidden** — concurrent `pip install` is what mutated
   the shared interpreter mid-build on 2026-10-16 (numpy seen at 2.5.3, 2.5.4
   and 2.5.0 by three agents in one session, with bus errors and
   `scikit-learn` metadata disagreeing with the imported module). With nobody
   installing, nothing mutates. If you need a package that is not present, stop
   and report it; do not install it.

2. **NEW THIS BATCH — the pinned interpreter now matches the gate's tool
   versions exactly, and that is deliberate.** Previous freezes pinned
   `ruff==0.14.2` and `pytest==8.4.2` while the gate runs the *system*
   interpreter. On 2026-10-25 that skew blocked P077 and P079 at the gate on
   `UP042` after both had measured ruff-clean at build time with no source
   defect — a product could be clean when built and blocked when gated with
   nothing having changed. This batch's freeze pins `ruff==0.16.9` and
   `pytest==9.1.1`, **identical to the system interpreter the gate uses**, and
   both interpreters are Python 3.13.18. Measured this session, venv against
   system: python 3.13.18 both, pytest 9.1.1 both, ruff 0.16.9 both.
   The defect class is closed at its source. Do not pin a different ruff.

3. **Declare every dependency bounded on both sides.** No bare floors. `numpy`
   has resolved to three different patch releases across sessions on an
   unbounded specifier, and `pulp` has been observed at 3.3.2, 4.1.0 and 4.2.0
   across environments.

4. **`pulp` cannot be used as an exact MILP backend. Measured again this
   session:** `pulp 4.2.0` in this container returns `[]` from
   `listSolvers(onlyAvailable=True)` — **no bundled solver at all.** A product
   that needs a proven optimum must either implement its own exact method for
   the small instances it certifies against, or declare the optimum unavailable
   and compare against a documented bound. No product in this batch is
   specified to need MILP; if your design drifts into needing one, report it
   rather than depending on `pulp`.

5. **Implement the classical or analytic baseline first, and report honestly
   when it wins.** Seven of the ten products above predict in writing that a
   baseline will match or beat the learned component, and four predict something
   stronger — that the learned model will produce a *confident wrong answer*
   exactly where the analytic method correctly refuses. Those are not hedges.
   They are the expected results, they are the most valuable findings in this
   batch, and a baseline win is research evidence that is **never removed,
   softened, retuned away, or reframed as a limitation of the baseline.**

6. **Every number in a validation document comes from a script executed that
   session.** No carried figures, no estimates.

7. **Timings belong in tables declared unstable, never in prose conclusions.**
   A sentence reading "a factor of 40" became "a factor of 38" on re-run.

8. **Count tests from junit XML, never from pytest stdout.** A suite that
   collects nothing prints success.

9. **Validation layout is flat:** `validation/<script>_output.txt`. Not
   `validation/outputs/`.

10. **`git check-ignore` exits 0 when a negation pattern matches.** Parse the
    matched pattern and treat a leading `!` as not-ignored.

11. **Anchor every path on `Path(__file__)` and `chdir` to the product root in
    validation support modules.** The gate runs each script with `cwd` set to
    the script's own directory, not the product root.

12. **No absolute container or host paths in any committed file**, including
    probe and analysis scripts, reports and checkpoints. Three consecutive
    sessions introduced this defect **in the act of writing about it**, and it
    stayed latent for four nights because no gate ran. Write `~` or a relative
    path.

13. **A build must not pollute its own tree through a subprocess environment it
    sets.** P074's own validation script set `HOME` to the product root, so
    `matplotlib` wrote `.config/matplotlib` into the repository. Use a temporary
    `HOME` and `MPLCONFIGDIR`, and set `PYTHONDONTWRITEBYTECODE=1`.

14. **Expect the gate to rewrite every committed output.** Running the gate
    re-executes every validation and example script in place. One gate run
    dirtied 169 files across ten products with environment-header and
    PNG-rendering churn and no change to any measured number. That churn is
    **never committed.**

15. **READMEs follow `templates/REPO_README_STANDARD.md`:** measured test counts
    in badges, the problem in three sentences, who it is **not** for, an
    alternatives table naming only packages verified to exist, a cold-clone
    quickstart with real output, a worked example with real output, a mermaid
    diagram using real module names, embedded screenshots, validation evidence
    **including failed checks**, limitations, reproduction commands. No
    marketing language, no emoji, no exclamation marks.

16. **Never describe a product as flight-safe, certified, mission-ready or
    production-ready.**

17. **Report what your product cannot determine.** This is the batch's
    organising requirement and it is not satisfied by a limitations section. The
    identifiability, observability, rank or coverage verdict is a **first-class
    output of the library, computed and returned**, and for P081, P085 and P089
    it must be computed *before* the estimate it gates.

## Build order and session allocation

Tonight's session builds the **five compact products, P086–P090**, one agent
each, five concurrent — the hard concurrency cap, and the configuration that
completed successfully on 2026-10-24.

The two flagships and three mediums (P081–P085) are **deliberately deferred to
the next session.** This session's trigger fired 15h14m late, and flagship
products at L3 are where usage-limit terminations have killed agents before
(five died together on 2026-08-29, four of five on 2026-09-02). Spending a
late-starting session's remaining budget on the five cheapest products
maximises what actually reaches a published repository tonight, and leaves the
two flagships a full fresh session rather than a rushed tail. The specification
above is complete for all ten either way, so the next session starts at
Phase 2 complete.

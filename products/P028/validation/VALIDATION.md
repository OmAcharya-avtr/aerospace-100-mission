# Validation — skymatch 0.1.0

**Validation level 2 (research).** Every number on this page was produced by running the
script named beside it in this session. Each script writes its raw stdout to the
`*_output.txt` file next to it; those files are the evidence and this page is a summary of
them that adds nothing they do not contain.

Environment of the run: Python 3.13.15, numpy 2.5.3, scipy 1.18.1, scikit-learn 1.9.1,
2 CPU cores, no GPU. The package declares `requires-python = ">=3.11"` and CI runs 3.11
and 3.12; the numbers below are from 3.13.15, which is what this machine has.

Reproduce everything, from `products/P028/`:

```bash
cd validation
python validate_geometry.py          # 15 checks,   2.0 s
python validate_catalogue.py         # 11 checks,   6.9 s
python validate_identification.py    # 10 checks,  41.0 s
python validate_failure_regime.py    #  5 checks,  28.0 s
python validate_ml_vs_classical.py   #  8 checks,  60.3 s
```

Totals: **49 checks, 49 passed, 0 failed**, 138 s wall clock. All five scripts share
`SEED = 20260902` (`_common.py`) and are deterministic in it, with the two caveats in
section 7.

### What "validation" can and cannot mean here

The catalogue is **generated, not measured**. Validating it therefore means two things and
not a third: that the generator reproduces the model it claims to implement (Eq. C1, C2,
C3, P1, isotropy), and that the matchers behave correctly on skies drawn from it. It does
**not** mean that any identification rate here predicts on-sky performance. See
`DATASET_CARD.md`. The comparison with Mortari et al. (2004) is qualitative for the same
reason: their numbers were measured on a real catalogue, a different camera and a
different noise model, and this page neither reproduces nor refutes them.

Every rate on this page is reported as **three exhaustive, mutually exclusive outcomes** —
identified correctly, false identification, no solution — because a matcher that reports a
wrong attitude and one that reports nothing are not the same failure, and summing them
into an "accuracy" hides which one you have. A false-identification count of zero is
reported with its Wilson (1927) upper bound, never as "never".

---

## 1. Frames, projection and the attitude solve

`validate_geometry.py` — 15 checks, 15 passed, 2.0 s. Raw output: `geometry_output.txt`.

This is the layer everything else stands on. A silent transpose in it would make every
identification rate on this page meaningless while leaving every identification "correct":
the correspondence would be right and the attitude would be the inverse rotation.

| Check | Reference | Result | Tolerance |
|---|---|---|---|
| `dcm_from_quat` vs `scipy.spatial.transform.Rotation.from_quat([x,y,z,w]).as_matrix()`, 500 random quaternions | SciPy | 6.6613e−16 | 1e−14 |
| `dcm_from_quat(quat_from_dcm(A)) − A`, 500 random rotations | identity | 6.6613e−16 | 1e−13 |
| `A Aᵀ − I` over 300 random rotations | orthogonality | 8.8818e−16 | 1e−14 |
| ra/dec round trip, 2000 points (Eq. G1) | identity | 2.4425e−15 rad (dec), 8.8818e−16 rad (ra) | 1e−14, 1e−13 |
| `‖v‖ − 1` | unity | 2.2204e−16 | 1e−15 |
| Angular separation, worst absolute error over six decades of angle (Eq. G2) | constructed exact angles | 3.3307e−16 rad | 1e−15 |
| Davenport q-method, noise-free, n = 2…10, 500 cases (Eq. G3) | exact attitude | 1.3258e−13 rad | 1e−12 |
| **Transpose regression:** angle from the solve to `A` and to `Aᵀ` | must be 0 and far from 0 | 2.6641e−16 rad and 2.9607 rad | 1e−12 and ≥ 1e−3 |
| Attitude error / σ, 4 stars, σ = 1…60 arcsec, 400 trials each | linear in noise | spread 1.2934e−02 over a 60× range in σ | 0.05 |
| Collinear observations | must raise `ValueError` | 50 / 50 raised | 1.0 |
| `project(unproject(p)) − p`, 5000 pixel positions (Eq. K1) | identity | 1.1369e−13 px | 1e−9 |
| Detector-corner angle vs `half_diagonal_rad` | spherical geometry | 2.7756e−17 rad | 1e−14 |
| Boresight pixel → zero angle | exact | 0.0 | 1e−15 |

**Why `atan2`, measured rather than asserted.** `arccos(a·b)` and
`atan2(‖a×b‖, a·b)` are analytically identical. They are not numerically identical, and
the whole package works on differences of angles at the 1e−5 rad level:

| True angle [rad] | `atan2` error | `arccos` error | ratio |
|---|---|---|---|
| 1e−08 | 8.665e−17 | 1.581e−08 | 1.8e+08 × |
| 1e−06 | 1.576e−16 | 3.997e−10 | 2.5e+06 × |
| 1e−04 | 1.159e−16 | 4.703e−12 | 4.1e+04 × |
| 1e−02 | 1.180e−16 | 3.475e−14 | 295 × |
| 1 | 3.331e−16 | 3.331e−16 | 1 × |

At 1e−8 rad `arccos` is wrong by 1.6e−8 rad — larger than the answer. The gate is on
absolute error because the *relative* error at 1e−8 rad is 9e−9 for either form: two unit
vectors that close differ only in their eighth significant digit, so no separation function
can beat that relatively. Eq. G2 is absolutely accurate to ~1e−16 at every angle, and that
is the property the matcher depends on.

**The plate scale is a labelling approximation, and here is its size.** A gnomonic
projection has no single plate scale. On the reference 12 deg / 1024 px camera the nominal
Eq. K3 value is 42.188 arcsec/pixel, and the measured local scale is 42.3424 (1.0037×
nominal) at the field centre, 42.1675 (0.9995×) at half width and 41.6544 (0.9874×) at the
corner — inside 1.3 % everywhere. Eq. K3 is used only to *set* a noise level in pixels and
a matching tolerance in radians; the projection itself is exact (round trip 1.1e−13 px), so
this is a labelling error and not a geometric one. Reference camera properties: focal
length 4871.4 px, field solid angle 143.477 sq.deg, half-diagonal 8.454534 deg.

---

## 2. The synthetic catalogue and the pair table

`validate_catalogue.py` — 11 checks, 11 passed, 6.9 s. Raw output: `catalogue_output.txt`.

| Check | Reference | Result | Tolerance |
|---|---|---|---|
| Star count, five magnitude limits | Eq. C1, `N(<m) = 4800·10^0.52(m−6)` | worst \|count/Eq. C1 − 1\| = 8.2451e−05 (rounding only) | 1e−3 |
| Growth per 0.5 mag | 10^(0.52·0.5) = 1.8197 | 1.820 at every step | — |
| Magnitude distribution | Eq. C2 inverse transform, Kolmogorov–Smirnov | D = 1.0489e−02 vs 5 % critical 1.4552e−02 | pass |
| Isotropy, dipole \|Σv\|/N | empirical null from 200 isotropic samples of the same size | 8.7607e−03 vs null p95 2.3582e−02; catalogue sits at the 21.5th percentile of the null | pass |
| Isotropy, sin(dec) in 10 equal-area bands | χ², 9 dof | 17.918 vs 1 % critical 21.67 | pass |
| Close-pair removal | Eq. C3, `C(N,2)(1−cosθ)/2` | mag 7.0 at 0.1 deg: 206 stars removed in 103 pairs against an expected 96.184 | Poisson |
| Prepared catalogue has no surviving close pair | exact | smallest surviving separation = 0.100000 deg | ≥ 0.1 deg |
| Pair-table count, five magnitude limits | Eq. P1 | worst \|pairs/Eq. P1 − 1\| = 1.0581e−02 | 2e−2 |
| Pair-table queries vs brute force (796 stars, 6858 pairs) | brute force | pair count, `separation_lookup`, `ordered_range`, `neighbours_range` all **exactly** 0 difference | 0 / 1e−14 |
| Mean stars in field vs density × solid angle | geometry | worst 1.2381e−02 | 5e−2 |

**The real cost of a magnitude limit.** The pair table is quadratic in catalogue size, and
this is the constraint that decides what a star tracker can carry:

| Mag limit | Stars | Pairs | Eq. P1 | ratio | MB | Build time |
|---|---|---|---|---|---|---|
| 5.0 | 1449 | 22 917 | 22 677 | 1.0106 | 1.65 | 0.016 s |
| 5.5 | 2637 | 75 140 | 75 129 | 1.0002 | 5.41 | 0.057 s |
| 6.0 | 4799 | 249 104 | 248 863 | 1.0010 | 17.94 | 0.196 s |
| 6.5 | 8734 | 823 772 | 824 378 | 0.9993 | 59.31 | 0.766 s |
| 7.0 | 15 894 | 2 730 140 | 2 730 164 | 1.0000 | 196.57 | 2.838 s |

Stars grow by 10^0.52 = 3.31× per magnitude and pairs by 10^1.04 = **10.96×**, measured
directly by `examples/catalogue_overview.py`. Going from mag 6.0 to 7.0 costs 11× the
memory for 3.3× the stars.

**The floor every identification rate sits on.** 600 uniform random pointings of the 12 deg
field (143.48 sq.deg) per magnitude limit:

| Mag limit | Mean stars in field | Predicted | P(fewer than 4 stars) |
|---|---|---|---|
| 5.0 | 5.092 | 5.040 | **0.2083** |
| 5.5 | 9.285 | 9.171 | 0.0233 |
| 6.0 | 16.762 | 16.691 | 0.0000 |
| 6.5 | 30.207 | 30.377 | 0.0000 |
| 7.0 | 54.762 | 55.279 | 0.0000 |

At magnitude limit 5.0 one pointing in five has no pyramid to find, whatever the decision
rule. That is why section 3d's identification rate at mag 5.0 is 0.75, and it is not the
matcher's fault.

---

## 3. The classical matchers, and both of their error rates

`validate_identification.py` — 10 checks, 10 passed, 41.0 s. Raw output:
`identification_output.txt`. Reference catalogue magnitude limit 6.0 (4799 stars, 249 104
pairs, 17.9 MB), reference camera 12 deg / 1024 px, 110 trials per row except where stated.

### 3a Noise-free exactness

With σ = 0 both rules identify 110 / 110 frames with 0 false identifications, and the median
attitude error is **4.3773e−09 arcsec** — the projection round-off, not an estimator error.

### 3b Both rates against centroid noise

Tolerance matched to the true σ throughout, no false stars.

| σ [arcsec] | τ [arcsec] | cands/frame | triangle ident | triangle false | pyramid ident | pyramid false | 95 % upper on pyramid false |
|---|---|---|---|---|---|---|---|
| 1 | 4.2 | 24.8 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0337 |
| 5 | 21.2 | 25.5 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0337 |
| 10 | 42.4 | 28.9 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0337 |
| 20 | 84.9 | 54.4 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0337 |
| 40 | 169.7 | 125.1 | **0.1364** | 0.0000 | 1.0000 | 0.0000 | 0.0337 |
| 60 | 254.6 | 126.2 | **0.0091** | 0.0000 | 1.0000 | 0.0000 | 0.0337 |

Pooled over the whole sweep the Pyramid rule made **0 false identifications in 660 trials**,
95 % Wilson interval **[0.00000, 0.00579]**. The honest statement is "below 0.006 at 95 %
confidence", not "never".

This reproduces, on a synthetic sky, the qualitative behaviour Mortari et al. (2004)
report: the triangle rule loses its identification rate as the tolerance widens with σ
(1.000 → 0.009), because a unique catalogue triangle stops being unique, while the Pyramid
rule does not (1.000 → 1.000), because the fourth star restores the discrimination the
wider window gave away.

### 3c Attitude error of the correct identifications

| σ [arcsec] | median error [arcsec] | p95 [arcsec] | median/σ | matched stars |
|---|---|---|---|---|
| 1 | 2.880 | 8.707 | 2.8800 | 9.86 |
| 5 | 14.401 | 43.538 | 2.8802 | 9.86 |
| 10 | 28.804 | 87.079 | 2.8804 | 9.86 |
| 20 | 57.615 | 174.172 | 2.8807 | 9.86 |
| 40 | 115.261 | 348.397 | 2.8815 | 9.86 |
| 60 | 180.144 | 507.047 | 3.0024 | 9.86 |

Largest/smallest median-error/σ ratio **1.0425** (tolerance 1.6). The attitude error is
about 2.88 σ here and is linear in σ to first order, as section 1 independently shows.

### 3d Both rates against the catalogue magnitude limit

σ = 5 arcsec, **2 false stars**, tolerance matched. This is the first table on this page in
which anything is wrong.

| Mag | Stars | Pairs | In field | Ceiling | tri ident | **tri false** | pyr ident | pyr false | ms/frame |
|---|---|---|---|---|---|---|---|---|---|
| 5.0 | 1449 | 22 917 | 4.75 | 0.836 | 0.8273 | **0.0091** | 0.7545 | 0.0000 | 5 |
| 5.5 | 2637 | 75 140 | 7.25 | 0.982 | 0.9545 | **0.0273** | 0.9364 | 0.0000 | 9 |
| 6.0 | 4799 | 249 104 | 8.31 | 1.000 | 0.9636 | **0.0364** | 1.0000 | 0.0000 | 13 |
| 6.5 | 8734 | 823 772 | 8.48 | 1.000 | 0.8818 | **0.1182** | 1.0000 | 0.0000 | 23 |

A denser catalogue makes the triangle rule *worse*, not better: more catalogue triangles fit
inside the same tolerance window, so uniqueness fails more often and its false-identification
rate rises from 0.009 to 0.118 between mag 5.0 and 6.5. The Pyramid rule gains from the same
density (0.7545 → 1.0000) because the extra stars give the fourth-star test more to work with.

### 3e How wide should the tolerance be?

τ = k√2 σ at a true σ of 10 arcsec; k = 3 is the package default.

| k | τ [arcsec] | cands/frame | ceiling | tri ident | tri false | pyr ident | pyr false |
|---|---|---|---|---|---|---|---|
| 0.25 | 3.54 | 0.3 | **0.275** | 0.2750 | 0.0125 | 0.0125 | 0.0000 |
| 0.50 | 7.07 | 2.3 | 0.800 | 0.8000 | 0.0125 | 0.4625 | 0.0000 |
| 1.00 | 14.14 | 9.9 | 1.000 | 0.9875 | 0.0125 | 0.9875 | 0.0000 |
| 2.00 | 28.28 | 23.7 | 1.000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 |
| 3.00 | 42.43 | 29.6 | **1.000** | 1.0000 | 0.0000 | 1.0000 | 0.0000 |
| 6.00 | 84.85 | 57.9 | 1.000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 |

Narrow gates drop the truth out of the candidate list entirely — the *ceiling* falls, and no
decision rule can recover it. Wide gates keep the truth but bury it, which costs the
triangle rule its uniqueness and costs every rule compute (0.3 → 57.9 candidates per frame
over this range). See section 7 for the expectation this section was written to confirm and
did not.

### 3f The regime that breaks the Pyramid rule — a non-zero false-identification rate

120 trials per row. A tolerance sized for 60 arcsec of centroid noise applied to frames that
actually carry 5 arcsec: a **12× too wide gate**, which is a configuration error, not a hard
sky.

| Case | Ceiling | cands/frame | tri ident | tri false | pyr ident | **pyr false** | 95 % CI | count |
|---|---|---|---|---|---|---|---|---|
| mis-sized gate, 0 false stars | 1.000 | 127.5 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | [0.0000, 0.0310] | 0/120 |
| mis-sized gate, **4 false stars** | 0.892 | 127.2 | 0.0000 | 0.0133 | 0.5417 | **0.3833** | **[0.3012, 0.4727]** | **46/120** |

**This is the number the specification asks for.** The Pyramid rule's famously low
false-identification rate is a property of a *correctly sized gate*, not of the algorithm
alone. Widen the gate 12× and add four false detections and 46 of 120 frames come back with
a confidently wrong attitude and no warning: several catalogue stars fit inside the window,
the "unique confirmation" test passes on the wrong triangle, and the rule reports it. A
mis-sized gate alone does not do this (row 1, 0/120); false stars alone do not do it
(section 4, 0/200 at every count). The two together do.

---

## 4. The documented failure regime: dense false-star fields

`validate_failure_regime.py` — 5 checks, 5 passed, 28.0 s. Raw output:
`failure_regime_output.txt`. Magnitude limit 6.0 (16.69 real stars on the detector on
average), σ = 5 arcsec, 10 brightest spots used, **200 trials per row**, tolerance matched.

### 4a What false stars actually do

False detections are drawn uniform on the detector with magnitudes uniform on [2.0, 6.0];
the whole spot list is then sorted by brightness and truncated to 10. Order matters: a
bright false detection **displaces a real star** instead of being appended to the list.

| n false | real spots kept | false spots kept | false fraction |
|---|---|---|---|
| 0 | 9.928 | 0.000 | 0.000 |
| 4 | 6.695 | 3.303 | 0.330 |
| 8 | 4.188 | 5.812 | 0.581 |
| 12 | 2.592 | 7.407 | 0.741 |
| 16 | 1.720 | 8.280 | 0.828 |
| 20 | 1.147 | 8.852 | 0.885 |
| 30 | 0.652 | 9.348 | 0.935 |

### 4b Both rates against false-star count

| n false | ceiling | tri ident | **tri false** | pyr ident | pyr false | 95 % CI on pyr false | pyr ident / ceiling |
|---|---|---|---|---|---|---|---|
| 0 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | [0.0000, 0.0188] | 1.0000 |
| 2 | 1.0000 | 0.9450 | 0.0550 | 1.0000 | 0.0000 | [0.0000, 0.0188] | 1.0000 |
| 4 | 1.0000 | 0.8550 | 0.1450 | 1.0000 | 0.0000 | [0.0000, 0.0188] | 1.0000 |
| 6 | 0.9400 | 0.8100 | 0.1450 | 0.8500 | 0.0000 | [0.0000, 0.0188] | 0.9043 |
| 8 | 0.6500 | 0.5150 | 0.2550 | 0.4650 | 0.0000 | [0.0000, 0.0188] | 0.7154 |
| 10 | 0.5000 | 0.4200 | 0.2900 | 0.3000 | 0.0000 | [0.0000, 0.0188] | 0.6000 |
| 12 | 0.1750 | 0.1600 | 0.4000 | 0.0950 | 0.0000 | [0.0000, 0.0188] | 0.5429 |
| 16 | 0.1000 | 0.0800 | 0.3900 | 0.0450 | 0.0000 | [0.0000, 0.0188] | 0.4500 |
| 20 | 0.0300 | 0.0100 | **0.4550** | 0.0050 | 0.0000 | [0.0000, 0.0188] | 0.1667 |
| 30 | 0.0100 | 0.0050 | 0.4050 | 0.0000 | 0.0000 | [0.0000, 0.0188] | 0.0000 |

**The two rules fail in opposite directions.** The triangle rule keeps identifying and
starts being wrong — **worst false-identification rate 0.4550 at 20 false stars**, i.e. 91
of 200 frames reporting a wrong attitude. The Pyramid rule keeps being right and stops
identifying — 0.0000 false at every count, and identification collapsing from 1.0000 to
0.0050. Both are useless at 20 false stars, in different ways, and the difference is which
failure you can detect from the outside.

### 4c Is the collapse a search failure or a decision failure?

| n false | ceiling | best possible ident | pyramid shortfall |
|---|---|---|---|
| 0–4 | 1.0000 | 1.0000 | 0.0000 |
| 6 | 0.9400 | 0.9400 | 0.0900 |
| 8 | 0.6500 | 0.6500 | 0.1850 |
| 10 | 0.5000 | 0.5000 | **0.2000** |
| 12 | 0.1750 | 0.1750 | 0.0800 |
| 20 | 0.0300 | 0.0300 | 0.0250 |
| 30 | 0.0100 | 0.0100 | 0.0100 |

Below a ceiling of about 0.1 the frame is **unrecoverable in principle**: the brightest ten
spots no longer contain three real stars often enough for any triangle to exist, so no
decision rule — classical, learned or otherwise — can do anything. That is the documented
failure regime, and it is a property of the frame and not of the matcher. Between 6 and 12
false stars there is real headroom (shortfall up to 0.2000), and that is the only band in
which section 5's learned ranker has anything to win.

### 4d How many spots should the matcher use?

10 false stars, σ = 5 arcsec.

| max spots | real kept | triples | ceiling | pyr ident | pyr false | ms/frame |
|---|---|---|---|---|---|---|
| 6 | 1.37 | 20 | 0.0933 | 0.0200 | 0.0000 | 7 |
| 8 | 2.30 | 25 | 0.2000 | 0.0933 | 0.0000 | 8 |
| 10 (default) | 3.48 | 25 | 0.4200 | 0.2267 | 0.0000 | 8 |
| 14 | 6.15 | 25 | 0.8200 | 0.5467 | 0.0000 | 9 |
| 20 | 10.88 | 25 | **0.9867** | **0.7267** | 0.0000 | 9 |

Raising the spot list from 10 to 20 takes the ceiling from 0.42 to 0.99 and the Pyramid
rule from 0.23 to 0.73, for 1 ms per frame. The cause is truncation, not search: with 10
spots the ten brightest are mostly the false detections and only ~3.5 real stars survive;
with 20 spots ~10.9 do. The 25-triple cap does not block it, because the Pyramid scan order
reaches high spot indices early. **This is the single most effective mitigation for the
failure regime in this package, and it is not the default.** The default stays at 10 because
a longer list costs centroiding effort and buys nothing at low false-star counts; this table
is the reason to raise it. See section 7 — this check was written expecting the opposite
result.

---

## 5. The learned ranker against the classical Pyramid rule

`validate_ml_vs_classical.py` — 8 checks, 8 passed, 60.3 s. Raw output:
`ml_vs_classical_output.txt`. The classical rules were implemented, tested and validated
first (sections 3 and 4). The ranker replaces the **decision**, not the search: both see
the identical candidate list from one geometric search on every frame.

### 5a The candidate dataset

| Split | Frames | Rows | Rows/frame | Positive fraction | Solvable frames | Seconds |
|---|---|---|---|---|---|---|
| train | 600 | 21 150 | 35.25 | 0.3664 | 0.8233 | 11.8 |
| test | 300 | 10 600 | 35.33 | 0.3595 | 0.8333 | 5.7 |

Catalogues and pair tables for magnitude limits 5.5, 6.0 and 6.5 built in 0.9 s (2637 /
4799 / 8734 stars; 75 140 / 249 104 / 823 772 pairs; 5.4 / 17.9 / 59.3 MB). Splitting is by
**frame and by seed**, never by row: candidates from one frame share spots and are
correlated, so a random row split would leak. Train seed 1234, test seed 5678, catalogue
seed 20260902 for both — the two sets share the catalogue and differ only in the
observations, which is the split a star tracker actually faces.

### 5b Training cost

`HistGradientBoostingClassifier` on 21 150 rows × 13 features: **3.5 s** to fit, 0.2 s for
the 11-feature no-photometry ablation, **16.3 s** total training cost including dataset
generation (gate: ≤ 180 s). Two cores, no GPU.

### 5c Row-level ranking on held-out frames

| Model | ROC AUC | Average precision | Base rate |
|---|---|---|---|
| full, 13 features | 1.00000 | 0.99999 | 0.35953 |
| no photometry, 11 features | 0.99999 | 0.99999 | 0.35953 |

**Read these with suspicion rather than satisfaction.** An AUC of 1.00000 does not mean the
ranker is excellent; it means the labelling problem is nearly separable given these
features, on this synthetic data. Five of the thirteen features (`n_confirm`,
`confirm_fraction`, `confirm_resid_norm`, and the two residual norms) encode the very
fourth-star evidence the classical Pyramid rule uses, so the ranker is being handed the
classical rule's own discriminant and asked to weight it. The interesting number is not the
AUC but the frame-level rates in 5e, where a near-perfect row ranking still only moves the
identification rate in two of six regimes.

Permutation importance on the held-out set (drop in average precision when a column is
shuffled) — the magnitudes are tiny because the problem is nearly separable and the signal
is redundant across columns:

| Feature | Drop in AP |
|---|---|
| `magnitude_resid_rms` | 0.00202 |
| `confirm_fraction` | 0.00024 |
| `confirm_resid_norm` | 0.00012 |
| `log1p_rivals` | 0.00008 |
| `n_confirm` | 0.00006 |
| all eight others | 0.00000 (one at −0.00000) |

The photometric columns are **flattered by the simulator**, which hands the instrument the
catalogue's own magnitude scale plus Gaussian noise: no colour term, no zero-point drift,
no saturation. The ablated model is the honest lower bound on this feature set, and it
loses essentially nothing (AUC 0.99999, AP 0.99999).

### 5d Is the confidence a probability?

Reliability table on 10 600 held-out candidate rows, 10 equal-width bins:

| Mean predicted | Observed | Count | Gap |
|---|---|---|---|
| 0.0001 | 0.0000 | 6779 | +0.0001 |
| 0.1326 | 0.0000 | 5 | +0.1326 |
| 0.5604 | 1.0000 | 1 | −0.4396 |
| 0.6583 | 0.5000 | 2 | +0.1583 |
| 0.7307 | 1.0000 | 1 | −0.2693 |
| 0.8494 | 1.0000 | 4 | −0.1506 |
| 0.9998 | 0.9989 | 3808 | +0.0008 |

* Brier score **4.6670e−04**, against **2.3027e−01** for predicting the base rate.
* Expected calibration error **5.9771e−04** (gate ≤ 0.05).

Those are good aggregate numbers, and they are good for an uninteresting reason: 10 587 of
the 10 600 rows fall in the two extreme bins, where the model is nearly always right. The
middle of the range holds **13 rows in total**, so the gaps of +0.13, −0.44, +0.16, −0.27
and −0.15 there are each based on 1 to 5 observations and no calibration is claimed between
0.05 and 0.95. The confidence is usable as an operating-point knob (5f) and as a flag for
"this one is not like the others"; it is not a calibrated posterior in the middle of its
range, because nothing here measures it there.

### 5e Frame-level rates, 150 trials per case

Identical candidate lists; only the decision differs.

| Case | Ceiling | cands/frame | tri ident | tri false | pyr ident | pyr false | ML@0.5 ident | ML@0.5 false | ML@0.9 ident | ML@0.9 false |
|---|---|---|---|---|---|---|---|---|---|---|
| clean sky, σ 5 | 1.0000 | 25.4 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 |
| σ 40, no false stars | 1.0000 | 125.0 | 0.1867 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 |
| 4 false stars | 1.0000 | 8.1 | 0.9067 | 0.0933 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 |
| 8 false stars | 0.6133 | 2.2 | 0.4667 | 0.3200 | 0.4333 | 0.0000 | **0.5933** | 0.0000 | 0.5733 | 0.0000 |
| 12 false stars | 0.2200 | 1.0 | 0.1800 | 0.3867 | 0.1067 | 0.0000 | **0.2000** | **0.0133** | 0.1667 | 0.0000 |
| gate 12× too wide + 4 false | 0.9067 | 127.8 | 0.0000 | 0.0133 | 0.5400 | **0.4133** | **0.9067** | 0.0067 | 0.9067 | 0.0000 |

Identification gain at threshold 0.5: **+0.0000** on a clean sky, **+0.0000** at σ 40,
**+0.0000** at 4 false stars, **+0.1600** at 8 false stars, **+0.0933** at 12 false stars,
**+0.3667** on the mis-sized gate. Worst learned false-identification rate at threshold
0.5 across these cases **0.0133**; worst Pyramid false-identification rate across the same
cases **0.4133**.

**What this does and does not show.** Three of the six regimes show no gain at all, because
the Pyramid rule is already at 1.000 and there is nothing to take. The largest single gain,
+0.3667, is against a classical rule that was handed a **12× too wide tolerance** — it is a
measure of robustness to misconfiguration, not of skill on a correctly configured problem.
Strip that case out and the learned ranker's contribution is confined to the dense
false-star band of section 4c, where it recovers +0.1600 and +0.0933 of a shortfall that
was 0.1850 and 0.0800 — most, but not all, of the available headroom. And it buys that with
the first non-zero learned false-identification rate on this page: **0.0133 at 12 false
stars, where the Pyramid rule measured 0.0000**. At threshold 0.9 that goes back to 0.0000
and the gain falls to +0.0600 (0.1667 against 0.1067), with the 8-false-star gain falling to
+0.1400.

### 5f The threshold curve — the operating point the classical rule does not have

Case: 8 false stars, σ 5 arcsec, 300 trials, ceiling 0.6767.

| Rule | ident | false ID | none | 95 % CI on false ID |
|---|---|---|---|---|
| triangle | 0.5467 | 0.2100 | 0.2433 | [0.1677, 0.2596] |
| **pyramid** | 0.4733 | 0.0000 | 0.5267 | [0.0000, 0.0126] |
| ranker@0.02 | 0.6667 | 0.0033 | 0.3300 | [0.0006, 0.0186] |
| ranker@0.05 | 0.6667 | 0.0033 | 0.3300 | [0.0006, 0.0186] |
| ranker@0.1 | 0.6667 | 0.0033 | 0.3300 | [0.0006, 0.0186] |
| ranker@0.2 | 0.6667 | 0.0033 | 0.3300 | [0.0006, 0.0186] |
| ranker@0.3 | 0.6633 | 0.0000 | 0.3367 | [0.0000, 0.0126] |
| ranker@0.5 | 0.6600 | 0.0000 | 0.3400 | [0.0000, 0.0126] |
| ranker@0.7 | 0.6400 | 0.0000 | 0.3600 | [0.0000, 0.0126] |
| ranker@0.9 | 0.6000 | 0.0000 | 0.4000 | [0.0000, 0.0126] |
| ranker@0.99 | 0.4733 | 0.0000 | 0.5267 | [0.0000, 0.0126] |

**8 of the 9 thresholds** reach a higher identification rate than the Pyramid rule without
exceeding the upper end of the Pyramid rule's own false-identification interval. At
threshold 0.3 and above the learned false count is also 0/300. The classical rule is a
single point on this plane, with no knob; that is the structural difference, and it is worth
more than any single row. Note the right-hand end: at threshold 0.99 the ranker lands
exactly on the Pyramid rule's numbers (0.4733 / 0.0000), which is the sanity check that the
curve passes through the classical point.

### 5g Runtime — the learned path is slower, measured over 120 frames

| n false | classical early exit [ms] | full scan [ms] | ranker scoring [ms] | ratio (scan + score)/early exit |
|---|---|---|---|---|
| 0 | 0.76 | 16.18 | 5.12 | **27.93 ×** |
| 4 | 2.08 | 9.82 | 5.04 | 7.14 × |
| 8 | 5.18 | 8.04 | 4.24 | 2.37 × |

The classical rule stops at the first confirmed triple; the ranker needs the whole candidate
list and so pays for the full scan plus inference. On a clean frame that is 21.3 ms against
0.76 ms — **28× slower, for zero gain in identification rate**. The penalty shrinks as the
frame gets hard, because the early exit stops firing. The learned path's argument is the
identification rate in 5e and 5f, not throughput, and on a clean sky it has no argument at
all.

### 5h Where the ranker fails

At threshold 0.05 over 300 frames: median confidence when correct **0.99855**, median
confidence when wrong **0.79070**, **maximum confidence when wrong 0.97908**, over **8
wrong acceptances in 300 trials**. A wrong answer carrying 0.979 confidence is the failure
that matters: it is indistinguishable from a right one by its confidence alone, and the
reliability table in 5d gives no basis for discriminating in that range.

Ceiling-limited frames on the same case: ceiling 0.3867, learned identification rate
0.3767, so **0.0100 of the shortfall is attributable to the decision rule and the rest is
the search**. The ranker cannot identify what the geometric search never proposed, and in
the regime where the classical rule looks worst, most of what is missing is not a decision
problem at all.

---

## 6. What the tests cover, and what they do not

`python -m pytest tests/ -q` from `products/P028/`: **167 passed**, 1.6 s.
`ruff check src/ tests/`: clean.

| Test file | Tests | Module under test |
|---|---|---|
| `test_geometry.py` | 28 | `geometry.py` |
| `test_catalogue.py` | 25 | `catalogue.py` |
| `test_camera.py` | 24 | `camera.py` |
| `test_pairtable.py` | 23 | `pairtable.py` |
| `test_identify.py` | 19 | `identify.py` |
| `test_scene.py` | 18 | `scene.py` |
| `test_pyramid.py` | 16 | `pyramid.py` |
| `test_triangle.py` | 14 | `triangle.py` |

**The gaps, stated rather than implied.** Four of the twelve package modules have **no test
file at all**: `ranker.py`, `dataset.py`, `benchmark.py` and `cli.py`. They are exercised
only indirectly, through the validation scripts in section 5 and through
`python -m skymatch`, neither of which is part of the test suite or of CI. So the learned
ranker, the dataset generator, the scoring harness and the command line interface are
*validated* but not *unit tested*, and a regression in `wilson_interval`, in
`MethodResult`'s rate arithmetic or in the CLI's argument handling would not be caught by
`pytest`. In addition, `hypothesis` is declared as a test dependency in `pyproject.toml`
and the build guide asks for property-based tests where algebraic identities exist, but
**no test file imports it**: the rotation and projection round trips of section 1, which are
exactly such identities, are covered by fixed-seed random sampling instead. These are
reported here as defects of the test suite, not fixed.

---

## 7. Expectations that were wrong, and one number that does not reproduce

Three things in this package were written to confirm a belief and instead refuted it. They
are recorded because a validation page that only contains confirmations is not evidence of
anything.

**7.1 A one-sigma gate does not lose true matches.** Section 3e was written asserting that
k = 1 in τ = k√2 σ would start dropping correct triangles out of the candidate list. It does
not: the measured ceiling at k = 1 is **1.000**. The reason is combinatorial rather than
statistical. A single edge lands inside a 1σ gate about 68 % of the time and all three edges
of one triangle about 0.68³ ≈ 31 %, but the scan tries **25 triples**, so the truth survives
somewhere in the candidate list with probability 1 − (1 − 0.31)²⁵, which rounds to 1. The
gate had to go far narrower before the ceiling moved: at k = 0.5 it is 0.800 and at
**k = 0.25 it is 0.275**. The original assertion would have passed at a tolerance that hid
the fact that it was testing nothing.

**7.2 Using more spots does rescue the dense false-star regime.** Section 4d was written
expecting the ceiling to stay flat as `max_stars` rose, on the reasoning that the false
detections are brighter on average and the triple scan is capped at 25. Both premises are
true and the conclusion is still wrong: the ceiling goes **0.0933 → 0.9867** between 6 and
20 spots at 10 false stars, because truncation — not search breadth — is what was destroying
the frame. The 25-triple cap does not block it because the Pyramid scan order reaches high
spot indices early. The measurement is reported as it came out, and the package default was
*not* changed to match it (see 4d for why).

**7.3 The small-batch threading number in `ranker.py` does not reproduce on this machine.**
`skymatch/ranker.py`'s `_single_thread` docstring states that one 25-row `predict_proba`
call costs **19.0 ms** with two OpenMP threads and **1.7 ms** with one, an 11× penalty, and
cites this section. Measured here, on 2 cores with `OMP_NUM_THREADS` unset and
`threadpool_info()` reporting 2 OpenMP threads, over 200 calls after warm-up:

| Configuration | Median | Mean |
|---|---|---|
| default (2 OpenMP threads) | **1.657 ms** | 1.811 ms |
| `threadpool_limits(limits=1)` | **1.287 ms** | 1.311 ms |
| ratio | **1.29 ×** | 1.38 × |

The optimisation is real but **1.3×, not 11×**, and the 19.0 ms figure is not reproducible
here. Two further things are true and should be said: no committed validation script
measures this, so the docstring's number has no evidence behind it in this repository; and
the measurement above was made with an ad-hoc script rather than a committed one, so it is
not rerunnable from the repository either. The docstring was left unchanged rather than
quietly corrected. Command used:

```bash
cd products/P028
python - <<'PY'
import sys, time, numpy as np
sys.path.insert(0, "src")
from threadpoolctl import threadpool_limits
from skymatch.camera import CameraModel
from skymatch.dataset import DEFAULT_GRID, build_catalogue_tables, generate_candidate_dataset
from skymatch.ranker import LearnedRanker
cam = CameraModel()
tables = build_catalogue_tables(tuple(p.magnitude_limit for p in DEFAULT_GRID), cam, 20260902)
d = generate_candidate_dataset(200, 1234, camera=cam, tables=tables)
r = LearnedRanker(random_state=0).fit(d.features, d.labels)
x = r._select(d.features[:25])
def bench():
    for _ in range(5):
        r._model.predict_proba(x)
    t = []
    for _ in range(200):
        t0 = time.perf_counter(); r._model.predict_proba(x)
        t.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(t))
a = bench()
with threadpool_limits(limits=1):
    b = bench()
print(f"default {a:.3f} ms, one thread {b:.3f} ms, ratio {a/b:.2f}x")
PY
```

### Determinism, honestly

The five scripts are seeded and reproduce their *rates* exactly, but two kinds of number in
the `*_output.txt` files are not bit-reproducible across machines:

* **Timings** — every `ms/frame`, `seconds` and `s to build` column. They differ by factors
  of several between runs and machines and are reported as orders of magnitude, not results.
* **The last digits of a few BLAS-dependent quantities.** The Davenport check in section 1
  came out 1.3258e−13 here against 1.3265e−13 in an earlier run on the same seed, because
  `numpy.linalg.eigh` is threaded and its reduction order is not fixed. Similarly the
  ablated model's average precision moved in its fifth decimal (0.99999 against 0.99998) and
  the ordering of the near-zero permutation-importance rows is not stable. None of these
  moves any check near its tolerance, but a reader diffing two runs will see them, and they
  are not a defect.

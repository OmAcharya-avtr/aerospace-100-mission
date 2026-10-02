# Dataset card — skymatch synthetic sky and candidate rows v0.1.0

## The catalogue is synthetic, and that bounds every claim in this repository

**No real star catalogue is bundled, downloaded, fitted to, or used anywhere in this
package.** There is no Hipparcos, no Tycho-2, no Gaia, no SKY2000. Every star is drawn from
the generative model below by a seeded `numpy.random.Generator`; every detected spot,
centroid error and false detection comes from the simulator in `src/skymatch/scene.py`;
every label comes from the simulator's own truth array. Nothing was measured, nothing was
observed, and nothing came from an instrument.

**The consequence, stated plainly: the identification rates, false-identification rates and
failure thresholds in `validation/VALIDATION.md` characterise the *algorithms*. They do not
predict the on-sky performance of a real star tracker, and must not be quoted as if they
did.** A number like "the Pyramid rule made 0 false identifications in 660 trials" is a
statement about this generative model, not about the sky.

The only externally sourced numbers in the whole repository are the magnitude-model
constants in item 1 below, which are order-of-magnitude values from standard references and
not a fit, and the centroiding-error range quoted from Liebe (2002) in
`src/skymatch/camera.py`.

## Nothing is committed as data

There is no `data/` directory and no CSV. **The generator is the dataset.** It is
deterministic in an integer seed, so

```python
from skymatch import CameraModel, DEFAULT_GRID, build_catalogue_tables, generate_candidate_dataset
cam = CameraModel(fov_deg=12.0, pixels=1024)
tables = build_catalogue_tables(tuple(p.magnitude_limit for p in DEFAULT_GRID), cam, 20260902)
train = generate_candidate_dataset(600, 1234, camera=cam, tables=tables)
test  = generate_candidate_dataset(300, 5678, camera=cam, tables=tables)
```

reproduces the 21 150 × 13 training matrix and the 10 600 × 13 test matrix used in
`MODEL_CARD.md` exactly, in 18.4 s on two cores (0.9 s for the catalogues and pair tables, 11.8 s for the
training frames, 5.7 s for the test frames).

---

## 1. The sky model

`skymatch.catalogue.generate_catalogue`.

**Magnitudes.** Whole-sky cumulative counts follow a power law (Eq. C1),

```
N(<m) = N_ref · 10^(b (m − m_ref)),   N_ref = 4800, m_ref = 6.0, b = 0.52
```

sampled on `[m_min, m_limit]` with `m_min = −1.5` (about Sirius) by inverse transform
(Eq. C2). `N_ref` is the order of magnitude of the naked-eye star count quoted in standard
references; `b = 0.52` is in the range expected for a locally uniform stellar distribution
seen through a shallow magnitude cut (a strictly uniform space density gives `b = 0.6`, and
real counts flatten below that as the sample leaves the galactic disc). **No fit to a real
catalogue was performed.** Measured agreement of the generator with Eq. C1 over five
magnitude limits: worst relative deviation 8.2451e−05, which is rounding
(`validation/catalogue_output.txt` section 2a). Magnitude distribution against Eq. C2:
Kolmogorov–Smirnov D = 1.0489e−02 against a 5 % critical value of 1.4552e−02.

**Positions.** `ra ~ U(0, 2π)`, `sin(dec) ~ U(−1, 1)`. Isotropy measured: dipole
\|Σv\|/N = 8.7607e−03 against an empirical null 95th percentile of 2.3582e−02 (the catalogue
sits at the 21.5th percentile of that null), and χ² = 17.918 on 9 dof for `sin(dec)` in ten
equal-area bands against a 1 % critical value of 21.67.

**Preparation.** `remove_close_pairs` deletes **both** members of any pair closer than a
threshold, because an unresolved pair produces one centroid somewhere between the two stars
and keeping either would put a wrong position in the catalogue. Checked against Eq. C3,
`E[pairs closer than θ] = C(N,2)(1 − cos θ)/2`: at magnitude limit 7.0 and 0.1 deg, 206
stars were removed in 103 pairs against an expected 96.184, and the prepared catalogue's
smallest surviving separation is exactly 0.100000 deg.

Resulting sizes, and the cost that matters:

| Mag limit | Stars | Pairs in field | Pair table |
|---|---|---|---|
| 5.0 | 1449 | 22 917 | 1.65 MB |
| 5.5 | 2637 | 75 140 | 5.41 MB |
| 6.0 | 4799 | 249 104 | 17.94 MB |
| 6.5 | 8734 | 823 772 | 59.31 MB |
| 7.0 | 15 894 | 2 730 140 | 196.57 MB |

Stars grow 3.31× per magnitude, pairs 10.96×.

## 2. What is not modelled

These are absent from the generative model, not approximated in it. Each one is a reason a
number from this package would not transfer to a real instrument.

* **Proper motion.** Stars are static. A real catalogue is referenced to an epoch and its
  entries move; bright nearby stars move by arcseconds per year, which is comparable to the
  centroid noise this package treats as the dominant error.
* **Binary and multiple stars.** Every entry is a single point source. Real catalogues
  contain unresolved and marginally resolved pairs whose photocentre is not either component
  and moves with orbital phase. The close-pair removal step above deletes *coincidentally
  close* catalogue entries; it is not a binary model.
* **Instrument spectral response.** There is one magnitude per star and one magnitude scale.
  There is no colour, no passband, no colour term between the catalogue magnitude and the
  instrument magnitude, and no saturation. The simulator hands the instrument the
  catalogue's own scale plus Gaussian noise, which **flatters the two photometric features**
  the learned ranker uses (`MODEL_CARD.md` items 5 and 10).
* **Optical distortion.** The camera is an ideal gnomonic pinhole: no radial distortion, no
  decentring, no misalignment, no pixel-response variation, no plate-scale error. Real star
  cameras carry several pixels of distortion at the field edge, and distortion shifts the
  **inter-star angles** that every algorithm and every feature in this package is built on.
  This is the most consequential omission for the identification problem specifically.

Also absent: **galactic structure.** Eq. C1 has no galactic-latitude dependence, so this sky
is isotropic. Real counts vary by roughly an order of magnitude between the galactic pole
and the galactic plane, which means a real star tracker's hardest fields — dense, with many
close pairs and many near-identical triangles — are **systematically under-represented
here**. This is the largest single departure of the model from the real sky.

Further absences, in the observation chain rather than the catalogue: no atmospheric or
stray-light background, no point-spread-function model, no sub-pixel or
brightness-dependent centroid bias, no motion blur from spacecraft rate, no planets, no
Moon or Earth limb in the field, no detector defects beyond an i.i.d. dropout probability,
and no temperature or ageing drift of the focal length.

## 3. The observation model

`skymatch.scene.simulate_scene`, in the order the steps are applied, because the order is
what makes the false-star regime behave like the real failure:

1. draw a uniform (Haar) random attitude — Shoemake (1992) subgroup algorithm;
2. rotate the catalogue into the camera frame and keep the stars on the detector (Eq. K1);
3. drop each remaining star independently with probability `dropout_prob` (**0.0 in every
   grid cell used for the model**);
4. project to pixels and add independent zero-mean Gaussian centroid noise of
   `centroid_sigma_arcsec` **on each axis** (the standard centroiding-error model, Liebe
   2002), plus independent Gaussian magnitude noise of `magnitude_sigma = 0.1` mag;
5. add `n_false_stars` false detections at **uniform random pixel positions** with magnitudes
   **uniform on [2.0, magnitude_limit]**;
6. sort everything by measured magnitude, brightest first, and keep the brightest
   `max_stars = 10`.

Step 6 after step 5 is deliberate: a bright false detection **displaces a real star** from
the list rather than being appended to it, which is what a real star tracker's brightest-N
spot selection does. The measured effect is severe and is the whole of the failure regime in
`validation/VALIDATION.md` section 4: at 8 false stars only 4.188 real spots survive into
the list of 10, at 16 only 1.720, and at 30 only 0.652.

The matcher receives `Scene.vectors` and `Scene.magnitudes` and nothing else.
`Scene.truth_index` (catalogue index per spot, −1 for a false detection) and
`Scene.attitude` exist **for scoring only** and are read by no feature.

## 4. A row is one candidate

`skymatch.dataset.generate_candidate_dataset` runs the **classical** geometric search on
each frame and emits one row per candidate triangle it proposes:

* `X` — the 13 features of `skymatch.identify.FEATURE_NAMES` (listed in `MODEL_CARD.md`
  item 3);
* `y` — 1 if all three correspondences are the truth, else 0;
* `group` — the frame index;
* `frame_solvable` — per frame, whether any correct candidate was proposed at all. This is
  the **ceiling** on every decision rule and is reported, because a frame whose candidate
  list never contained the truth cannot be rescued by any ranker.

**Class balance is a property of the operating point, not a choice.** A frame yields at most
one correct candidate and however many wrong ones the tolerance admits. The positive
fraction is reported (0.3664 train, 0.3595 test) and **deliberately not rebalanced**:
rebalancing would destroy the base rate the model's probability output is supposed to
express.

## 5. The operating-point grid

`skymatch.dataset.DEFAULT_GRID`, 12 cells drawn with equal weight, so 600 frames give 50 per
cell and 300 give 25.

| Magnitude limit | Centroid σ [arcsec] | False stars |
|---|---|---|
| 5.5 | 2 | 0 |
| 5.5 | 10 | 2 |
| 6.0 | 1 | 0 |
| 6.0 | 5 | 0 |
| 6.0 | 5 | 4 |
| 6.0 | 20 | 2 |
| 6.0 | 40 | 0 |
| 6.0 | 5 | 8 |
| 6.0 | 5 | 12 |
| 6.0 | 10 | 16 |
| 6.5 | 5 | 0 |
| 6.5 | 20 | 4 |

Spanned: magnitude limit 5.5–6.5, centroid noise 1–40 arcsec, false stars 0–16. Fixed across
every cell: the 12 deg / 1024 px camera (42.188 arcsec/pixel), `max_stars = 10`,
`dropout_prob = 0`, `magnitude_sigma = 0.1`, and a tolerance of `3√2 σ` matched to the cell's
own noise with a floor at σ = 0.5 arcsec.

**The grid deliberately contains cells nothing can solve** — 12 and 16 false stars, where
the measured ceiling is 0.1750 and 0.1000 — so that a model trained on it sees what *no
evidence* looks like instead of only ever seeing solvable frames. That is why the solvable
fraction is 0.8233 rather than 1.

## 6. Splits

| Split | Frames | Rows | Frame seed | Catalogue seed |
|---|---|---|---|---|
| train | 600 | 21 150 | 1234 | 20260902 |
| test | 300 | 10 600 | 5678 | 20260902 |

Split **by frame and by seed, never by row**: candidates from one frame share observed spots
and are correlated through them, so a random row split would leak.

Train and test **share the catalogues**. That is the split a star tracker faces — the
catalogue is fixed and loaded once, the sky varies — and it means **nothing here measures
generalisation to a different catalogue**. A user who wants that must pass a different
`catalogue_seed` and rerun.

## 7. Known biases and limitations, collected

* **Synthetic throughout.** Nothing here has been compared against a real star field.
* **Isotropic sky.** No galactic-latitude dependence; the dense fields that are hardest in
  practice are under-represented (item 2).
* **No proper motion, no binaries, no spectral response, no optical distortion** (item 2).
* **Photometry is optimistic**, and the learned ranker's single most important feature is a
  photometric one (`MODEL_CARD.md` item 7).
* **One camera.** Every number in this repository is for a 12 deg square field on a 1024 ×
  1024 detector. Nothing says how any of it scales to a 5 deg or a 25 deg field.
* **One spot-list length.** `max_stars = 10` everywhere in the grid, even though
  `validation/VALIDATION.md` section 4d shows 20 is dramatically better in the hard regime.
  The configuration that works best is therefore **outside the training distribution**.
* **Independent, uniform parameter draws within a cell.** Nothing correlates the noise level
  with the false-star count, as a real instrument's degradation modes would.
* **Faint, dense catalogues are untested and mostly unreachable.** `generate_catalogue`
  refuses a `magnitude_limit` above 12.0 outright, but the practical ceiling is far lower:
  the pair table is O(N²), so Eq. C1 and Eq. P1 give 2.7e+06 pairs (197 MB) at magnitude 7.0,
  3.0e+07 (≈2 GB) at 8.0 and 4.3e+11 (≈31 TB) at the nominal 12.0 limit. Nothing above 6.5
  appears in the training grid, and nothing above 7.0 has been run at all.

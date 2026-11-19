# P099 PassYield — specification as built

Status: **TESTING**. Research-grade software. Not flight-qualified, not
certified and not approved for operational aerospace use.

## 1. What the registry asked for

From `products.yaml`:

> Realised data yield of a scheduled ground-station contact against the
> nominal duration the scheduler assumed, decomposed into acquisition, re-sync
> and idle time, with the nominal-versus-realised shortfall as the headline
> output. Baseline 1 nominal duration times nominal rate, implemented
> deliberately as the thing being refuted because it is what a contact
> schedule books; baseline 2 an analytic yield model with measured acquisition
> and re-sync distributions convolved over the pass geometry; baseline 3
> ground truth directly accounted yield from a per-pass record with bootstrap
> confidence intervals. Learned per-pass yield predictor from pre-pass state
> and geometry, with an uncertainty output, scored on held-out STATIONS and
> held-out TIME separately. WHAT IT DOES THAT EXISTING TOOLS DO NOT:
> schedulers optimise against nominal contact windows and operations tools
> report delivered volume after the fact; neither feeds the measured shortfall
> back as a distribution the scheduler could book against.

Class compact, Level 2, AI-enabled, space-operations.

## 2. The identity

With `W = [0, T]` the geometric mask-to-mask contact window and `r(t)` the
achievable rate from the link budget evaluated on the SGP4 geometry:

```
Y_nom  = T_nom * R_nom                        what the schedule books
Y_cap  = integral over W of r(t) dt           what a perfect link delivers
Y_real = integral over A of r(t) dt, A in W   what the record accounts for

S = Y_nom - Y_real
  = (Y_nom - Y_cap)  +  Int_acq r  +  Int_resync r  +  Int_idle r  +  rho
    geometry            acquisition   re-sync          idle          residual
```

`A` is the active set: the window less the acquisition interval at its start,
less every re-synchronisation interval, less every idle interval, resolved by
a declared priority so that the three are pairwise disjoint.

`rho` is zero in exact arithmetic. It is **not** zero here, because each term
is integrated on its own composite-Simpson partition of the window and four
sums over four partitions do not telescope. It is a reported number on every
result object, and `validation/validate_decomposition.py` publishes its
distribution over the whole campaign. Defining `Y_real` as `Y_cap` minus the
losses would force `rho = 0` by construction and make the identity
untestable; the product does not do that.

## 3. The headline output is a distribution

The API returns `ShortfallDistribution`, a probability mass function on a
uniform grid of data volume in Mbit, with `quantile`, `quantiles`,
`interval`, `mean_mbit`, `standard_deviation_mbit`, its grid step, its
quantile discretisation and its truncated mass. A contact schedule can only
be written against a random variable by booking a quantile of it, so the
quantiles are the product and the mean is a convenience.

## 4. The three baselines

| | what it is | why |
|---|---|---|
| 1 | `T * R_nom`, predicting no shortfall | the thing being refuted; it is what a contact schedule books |
| 2 | the analytic law, by numerical convolution over the pass geometry | the mathematical core |
| 3 | directly accounted from the per-pass record, bootstrapped by PASS | ground truth |

Baseline 1 is built at full strength on purpose: the duration is the true
geometric window from the committed element set, and the rate is the link
budget's own rate at a declared 30 degree reference elevation with the
station's own figure of merit — exactly what a scheduler with an ephemeris and
a station catalogue can compute before the pass.

Baseline 2's resampling-free law is a mass-preserving pushforward of the
acquisition law through the cumulative volume, a compound-Poisson convolution
of the per-outage loss, and a pushforward of the idle law, all convolved. It
is **numerical, not closed form**: the cumulative volume has no elementary
inverse. A **strengthened variant, baseline 2b**, replaces the unconditional
idle law by the backlog-limited dry time a scheduler can compute before the
pass. Both are measured and both are published; 2b does not replace the
specified form, because the specification names the unconditional form and the
measurement of that form is the finding.

Baseline 3's **resampling unit is the PASS**. It is declared on the returned
object as `resampling_unit` and `tests/test_baselines.py` asserts the
attribute.

## 5. The learned component

A `RandomForestRegressor` over 17 pre-pass features, predicting the realised
fraction `Y_real / Y_cap`. The station enters through its figure of merit,
zenith attenuation, mask and latitude and **never through its identity**,
which is the condition for a held-out-station split to mean anything.

Two uncertainty outputs. The across-tree spread is structurally
over-confident, because it measures disagreement about the conditional mean
and omits the irreducible noise; the out-of-bag residual law is the product's
declared output. Both are scored and both are published.

Scored on held-out **stations** and held-out **time** separately, by CRPS
(strictly proper), PIT calibration error and 80 per cent interval coverage.
The gap between the two splits is itself a reported quantity.

## 6. What is real and what is declared

- **Real and published:** the orbit. Catalogue object 06251 from the official
  SGP4 verification element set `SGP4-VER.TLE` installed by the `sgp4`
  package, originating in Vallado et al. (2006). The two committed lines are
  checked character for character against the installed file by
  `validation/validate_geometry.py`, and the propagation is checked against
  the official `tcppver.out` reference output.
- **Declared and synthetic:** the eight ground stations, the spacecraft link
  budget, and the whole event timeline. No part of the timeline is a
  measurement of any real mission. `DATASET_CARD.md` says so in the same
  words.

## 7. Scope the product does not claim

- no modulation-and-coding staircase, no hysteresis at modcod switch points;
- no antenna pattern, no body-pointing error, no rain fade time series;
- no ITRF frame, no polar motion, no UT1-UTC;
- no scheduling optimiser. The product measures a shortfall distribution; what
  a scheduler does with it is outside the package.

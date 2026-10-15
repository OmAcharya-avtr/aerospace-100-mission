# Dataset card — twincalib synthetic calibration records

Version 0.1.0 · 2026-10-15 · Apache-2.0 · © 2026 OPTIMA Organisation

This software is research-grade. It is **not flight-qualified, not certified,
and not approved for operational aerospace use.**

---

## What this is

Every data set in this repository is **synthetic**, generated on demand by
`twincalib.scenarios.make_dataset` from fixed seeds. Nothing is committed as
data: the generator is committed instead, and it is deterministic.

**No telemetry from any spacecraft was used. No parameter value in the
generating model is a measurement of flight hardware.** Each magnitude is
justified against a cited material property or physical constant in
`twincalib.scenarios.THETA_TRUE`'s docstring, which is not the same as being
measured.

## The generating model

A two-node lumped thermal network: an avionics unit conductively coupled to a
radiator panel that radiates to deep space.

```
C1 dT1/dt = P_dis(t) - k12_eff(T1) (T1 - T2)
C2 dT2/dt = k12_eff(T1) (T1 - T2) + q_sun s(t) + q_par - h_rad (T2**4 - T_space**4)
```

Sources: the lumped-network formulation and the grey-body radiation term from
Gilmore (ed.), *Spacecraft Thermal Control Handbook, Volume I*, 2nd ed., AIAA,
2002, Chapter 1; the lumped-capacitance method and the Biot-number validity
criterion from Incropera and DeWitt, *Fundamentals of Heat and Mass Transfer*,
Chapter 5; `sigma = 5.670374419e-8 W/m^2/K^4` from the SI 2019 revision;
`T_space = 2.72548 K` from Fixsen (2009), *Astrophysical Journal* 707, 916.

### Declared parameters

| parameter | value | unit | where the magnitude comes from |
|---|---|---|---|
| `C1` | 800 | J/K | about 0.9 kg of 2024-T6 aluminium at 900 J/kg/K (Incropera and DeWitt Table A.1) |
| `C2` | 4000 | J/K | about 4.4 kg of the same alloy |
| `k12` | 2.5 | W/K | a bolted interface with filler; 1-10 W/K is ordinary (Gilmore 2002 Ch. 1) |
| `h_rad` | 1.205e-8 | W/K^4 | `eps sigma A` with `eps = 0.85` (white paint) and `A = 0.25 m^2` |
| `q_sun` | 55 | W | `alpha S A` with `alpha = 0.2`, `S = 1361 W/m^2` (Kopp and Lean 2011), `A = 0.2 m^2` |
| `q_par` | 18 | W | a declared parasitic load |

### The declared model-form discrepancy

The **generator** uses a temperature-dependent interface conductance:

```
k12_eff = k12 (1 + 1.5 (T1 - 290 K) / 100 K)
```

The **calibration model** has no such term and cannot reproduce it at any
parameter value. Interface conductance really does vary with temperature,
through the bulk conductivity of the two bodies and through the contact pressure
as the joint expands (Madhusudana, *Thermal Contact Conductance*, 2nd ed.,
Springer, 2014, Chapter 4). The linear form and the 15 per cent per 10 K slope
are **declared for the experiment, not measured**, and the slope is at the large
end of what is plausible: it was chosen so that the resulting discrepancy sits
clearly above the thermometer noise, 0.5061 K root mean square against 0.3 K on
the baseline design (`validation/validate_model_output.txt` check 8).

Passing `model=calibration_model()` to `make_dataset` generates data with **no**
discrepancy, which is how the recovery and coverage studies isolate sampling
error from model error.

## Measurement model

Independent Gaussian thermometer noise with standard deviation **0.3 K**, a
declared value in the range typical of a spacecraft housekeeping thermistor
chain end to end. It is not a measurement of any flight sensor. The fitter is
**told** this value rather than estimating it.

The exogenous inputs — unit dissipation and illumination fraction — are known
exactly and held over each integration step (zero-order hold). Real telemetry
has neither property.

## The five designs

An experiment design is everything about a record except the parameters.
Identifiability is a property of the design and the parameters together, which
is why the designs are first-class objects.

| design | measured | duration / step | observations | what is wrong with it |
|---|---|---|---:|---|
| `baseline-eclipse` | T1, T2 | 12000 s / 30 s | 802 | nothing |
| `full-sun` | T1, T2 | 12000 s / 30 s | 802 | illumination constant at 1, so only `q_sun + q_par` is identifiable |
| `lumped-nodes` | T1 | 12000 s / 30 s | 401 | T1 only, `k12 = 40 W/K`, fast mode unresolved at 30 s |
| `coarse-sampled` | T1, T2 | 12000 s / 60 s | 402 | nothing; exists because it costs half as much to simulate |
| `weak-excitation` | T1 | 12000 s / 30 s | 401 | T1 only at constant 24 W; full rank and worthless |

Orbital profile: period 5700 s, eclipse fraction 0.35, except where
`sun_constant` overrides it. Duty cycle: 4000 s period, 50 per cent at 35 W and
50 per cent at 12 W, except `weak-excitation` which is constant. Initial state
`(T1, T2) = (285, 275) K`, treated as known by the fitter.

Temperatures stay inside **274.496 to 309.789 K** over all four analysis
designs, and the model's declared validity range is 200-350 K
(`validation/validate_model_output.txt` check 9).

## Trajectories, phases and seeds

`make_dataset(design, seed, phase, theta, model)`.

- `seed` drives a `numpy.random.default_rng` for the thermometer noise only.
- `phase` in `[0, 1)` shifts both the orbital and the duty-cycle profiles, so
  different trajectories of the same design excite the model differently.

The benchmark's split: training phases evenly spaced over `[0, 0.5)` with seeds
`base_seed + i`, test phases over `[0.5, 1)` with seeds `base_seed + 1000 + i`.
At the published defaults that is training phases 0.000, 0.125, 0.250, 0.375
with seeds 4200-4203, and test phases 0.500, 0.667, 0.833 with seeds 5200-5202.

## Size and storage

Nothing is committed as data. A `baseline-eclipse` record is 802 float64
observations plus its inputs, under 20 kB in memory, and takes about 2.3 ms to
generate. The largest pooled training set used anywhere in this repository is
1604 samples, capped to 600 for the Gaussian process.

**No serialised model artifact is committed either**, so the repository does not
carry a pickle whose format is tied to a library minor version.

## Limitations

1. **It is synthetic.** Every conclusion in this repository is a conclusion
   about this generator. The identifiability mathematics transfers; the numbers
   do not.
2. **The noise is independent and Gaussian with a known standard deviation.**
   Housekeeping telemetry has quantisation, drift, dropouts, correlated sensor
   error and a calibration of its own.
3. **The inputs are known exactly.** In practice the dissipation of a unit is
   inferred from a current measurement with its own error, and the illumination
   fraction from an ephemeris with its own.
4. **The zero-order-hold input grid is part of the model**, not an
   implementation detail: refining it from 30 s to 3.75 s moves the trajectory
   by 0.626 K because the eclipse edge at 3705 s is not a multiple of 30 s. All
   numbers here are internally consistent because the generator and the fitter
   share the grid; a real calibration would not have that cancellation.
5. **One discrepancy form.** The generator's only model-form error is the
   temperature-dependent conductance. A residual model that learns it has not
   been shown to learn anything else.
6. **The radiator sees only deep space.** No Earth infrared, no albedo, no
   self-viewing. In a real fit `q_par` would absorb all three and would not be a
   parasitic load at all.
7. **Two nodes, no control.** No heaters, thermostats, louvres, heat pipes or
   phase-change material, all of which a real unit has.

## Licence and provenance

Apache-2.0, © 2026 OPTIMA Organisation. The data generator is
`src/twincalib/scenarios.py` in this repository; there is no upstream data
source, no third-party data and no personal data of any kind.

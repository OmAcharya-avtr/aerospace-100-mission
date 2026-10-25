# Dataset card: contactsched synthetic scheduling instances

## What it is

Ground-station contact scheduling instances generated from a single integer seed
by `contactsched.dataset.generate_instance`. One row of the labelled dataset is
one contact request: 23 pre-scheduling features and a binary label.

No data is committed. The dataset is rebuilt from seeds, deterministically, by
`contactsched.dataset.build_dataset`, and the rebuild is byte-identical, which
`validation/validate_dataset.py` asserts.

## Provenance

**Entirely synthetic.** There is no real data anywhere in this repository.

- **Element sets.** Constructed by `contactsched.tle.OrbitElements.to_lines`
  from elements drawn by the generator, formatted into the standard two-line
  layout with correct checksums, and parsed back by both an independent reader
  in this package and by `sgp4`. No catalogue element set is quoted. A product
  that quoted catalogue elements it cannot re-derive would be quoting data it
  cannot verify, and a stale element set would silently change every measured
  number on the day it was refreshed.
- **Station coordinates.** Drawn from a declared pool of six sites at
  coordinates near real high-latitude and equatorial facilities, chosen because
  that is where the geometry of polar-orbit contention lives. They are round
  numbers and they are not any operator's facility. Nothing about a real
  station's mask, keyhole, slew rate or service agreement is modelled.
- **Requests.** Drawn by the generator. No operational request log was used or
  consulted.

## Generation

```bash
python validation/validate_dataset.py       # determinism, balance, split hygiene
python validation/validate_predictors.py    # the dataset the model scores use
```

| Quantity | Distribution |
| --- | --- |
| antennas per instance | 3 to 4, drawn without replacement from the pool of 6 |
| spacecraft per instance | 4 to 6 |
| requests per instance | 7 to 11 |
| inclination | one of 97.4, 97.8, 98.4, 51.6, 63.4, 74.0 deg |
| mean motion | uniform on 13.9 to 15.5 rev/day |
| eccentricity | uniform on 0.0002 to 0.004 |
| RAAN, argument of perigee, mean anomaly | uniform on 0 to 360 deg |
| B\* drag term | uniform on 1e-5 to 3e-4 |
| priority | 1, 2, 3, 4 with probabilities 0.2, 0.3, 0.3, 0.2 |
| contact duration | one of 180, 300, 420, 600 s |
| minimum contacts | `round(U(1, 2 + 2 * load))` clipped to 1 to 6 |
| maximum contacts | minimum plus 0 or 1 |
| allowed antennas | unrestricted with probability 0.55, otherwise a random subset |
| load knob | uniform on 0.7 to 2.3 |
| visibility scan step | 45 s |
| candidate grid step | 240 s |
| horizon | 43200 s as used for the model dataset |

The station pool, with mask floor and turnaround time:

| Site | Latitude | Longitude | Altitude km | Mask floor deg | Turnaround s |
| --- | --- | --- | --- | --- | --- |
| SVALBARD | 78.23 | 15.39 | 0.45 | 5.0 | 300 |
| PUNTA-ARENAS | -53.16 | -70.91 | 0.03 | 5.0 | 420 |
| KOUROU | 5.25 | -52.80 | 0.01 | 7.0 | 360 |
| TROLL | -72.01 | 2.53 | 1.30 | 5.0 | 300 |
| ALICE-SPRINGS | -23.70 | 133.88 | 0.55 | 6.0 | 300 |
| FAIRBANKS | 64.86 | -147.85 | 0.14 | 5.0 | 300 |

## Composition, as measured

Seeds 0 to 319, horizon 43200 s:

| Quantity | Value |
| --- | --- |
| instances | 320 |
| rows, one per request | 2903 |
| positive rate | 0.308302 |
| positives | 895 |
| build wall clock | unstable; 22.9 to 28.6 s observed this session |
| per instance | unstable; 71.5 to 89.2 ms observed this session |
| cpu cores | 2 |

| Subset | Rows | Positives | Rate |
| --- | --- | --- | --- |
| all requests | 2903 | 895 | 0.3083 |
| `capacity_deficit > 0`, asked for more than its passes allow | 604 | 604 | 1.0000 |
| `capacity_deficit <= 0`, lost the contention or won | 2299 | 291 | 0.1266 |

The first subset is trivially predictable and is 21 per cent of the data.
Every model score in `validation/validate_predictors_output.txt` is therefore
reported both over all requests and over the second subset.

## Label

`unsatisfiable = 1` when `solve_exact` with no fairness constraints delivers
fewer contacts than the request's declared `min_contacts`.

Labelling against the exact optimum rather than against a heuristic is
deliberate: a request starved at the optimum is starved by the geometry and the
competition, not by a weak scheduler. The consequence is stated in
`MODEL_CARD.md` under Failure cases: a positive label is a statement about the
best achievable schedule.

## Features

23, listed in order in `contactsched.features.FEATURE_NAMES`. Per-feature
minimum, median, maximum and non-finite count over the whole dataset are printed
in `validation/validate_dataset_output.txt`.

Non-finite values are legitimate: `demand_over_usable` is infinite for a request
with declared demand and no usable window, which is a real and highly predictive
state. `contactsched.features.design_matrix` substitutes a finite sentinel of
1.0e6 before any model sees them, and the substitution is documented in that
function rather than hidden in the model. 273 of 2903 rows carry that infinity.

## Splits

By instance. Train seeds 0 to 239 (2178 rows), test seeds 240 to 319 (725 rows).
Zero shared `(seed, request_id)` keys and zero shared seeds, asserted in
`validation/validate_dataset.py`.

## Known limitations

- **It is a generator, not a sample of reality.** Every distributional choice
  above is a choice. A real network's request mix differs in the priority skew,
  the duration mix, the contact-count distribution, the antenna pool and the
  correlation between them, and no claim is made that any of them resembles one.
- **The load knob is uniform.** Real demand is bursty and correlated with
  mission phase; this generator's demand is independent across requests given
  the load.
- **Contention is geometric only.** Weather, equipment outages, maintenance
  windows and priority escalation are not modelled.
- **No negative or adversarial cases.** Requests are always well formed. The
  package's input validation is tested separately in `tests/`, not through the
  dataset.
- **Instance size is small.** 7 to 11 requests per instance keeps the exact
  solve fast enough to label 320 instances in under thirty seconds. Larger
  instances would have different contention structure and are untested.

## Licence

The generator is part of this package and is Apache-2.0. The data does not
exist until it is generated, and carries no separate licence.

# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-15

First release. Status: TESTING. Validation level 3.

### Added

- `spec`: the FMEA-shaped declarative input — `Fault` (effect, ordinal
  MIL-STD-1629A severity category, detectability, declared detection latency in
  seconds), `Mode` (with `safe` and `absorbing` flags), `Response` (transition,
  cost in minutes of mission outage, precondition, `clears_fault`), `Condition`
  (per-mode availability and an `always_false` flag), `ObservationChannel`
  (unit, nominal, threshold, faulted value, direction, 1-sigma noise) and
  `ObservabilityModel` (one binary signature per fault, with
  `indistinguishable_groups`). YAML loading that rejects an unknown key rather
  than dropping it, and `precondition_conflict`, which decides emptiness of a
  conjunction exactly and in time linear in the literal count.
- `automaton`: `build_automaton`, the response automaton over
  `modes x (faults + 1)` states, and `DetectionAssumption`, the declared
  detectability and latency budget a fault-specific response requires.
  Transitions out of an absorbing mode, transitions whose precondition cannot
  hold, and transitions that change no state are all absent by construction.
- `reach`: `safe_reachable_set` (backward breadth-first fixpoint),
  `min_cost_to_safe` (multi-source Dijkstra on the reversed graph, with
  deterministic tie-breaking), `resolution_cost` and `outcome_state` (what a
  commanded response actually costs when the true fault is not the believed
  one), `preferred_specific_response`, `reachability_report`, and
  `exhaustive_safe_reachable` / `exhaustive_min_cost_to_safe`, the
  exponential brute-force references the fast paths are cross-checked against.
- `diagnostics`: the six checks — fault states with no reachable safe mode,
  declared safe modes no fault state can reach, fault states with no enabled
  response, faults with no declared response, responses whose precondition can
  never hold, and ambiguity classes whose members demand different responses —
  plus `conservative_response`, the cheapest fault-agnostic route that keeps
  every member of a class safe-reachable.
- `data`: deterministic synthetic observation generation from the declaration,
  with a declared dimensionless `noise_scale` difficulty multiplier. Faults in
  the same declared ambiguity class get identical distributions by
  construction.
- `baseline`: `ThresholdLogicClassifier`, the declared threshold logic —
  latched symptom word, table lookup, and a declared confidence that is `1` on
  a unique signature, `1/m` on an ambiguity class and `1/(m(1+d))` on a
  Hamming-`d` resolution. Written and benchmarked before any learned component
  existed.
- `learned`: `LearnedFaultClassifier`, a standardised multinomial logistic
  regression with an optional isotonic or sigmoid calibration wrapper, exposing
  the full posterior, the top-label confidence and the total fault posterior;
  and `accuracy_ceiling`, the upper bound the declared observability model puts
  on any classifier.
- `calibration`: `reliability_curve`, `expected_calibration_error` (top-label,
  Guo et al. 2017 eq. 3), `CalibrationResult` with per-bin occupancy, and the
  multi-class `brier_score` (Brier 1950).
- `policy`: `FaultResponsePolicy`, which consumes the confidence — an
  identification below the declared threshold is routed to the
  class-conservative response — plus `cost_of`, `oracle_cost`, `is_misrouted`
  and `reaches_safe`.
- `evaluate`: the equal-false-alarm-rate comparison (matched on a separate
  calibration split), `noise_sweep`, the four-policy comparison with the cost
  of conservatism, and `conservatism_sweep`, the cost against mis-route trade
  curve.
- `synth`: `random_declaration`, structurally valid random declarations used to
  cross-check the checker against brute force.
- `plotting` (Agg only) and six `examples/` scripts writing to `screenshots/`.
- `__main__`: `python -m safemodetree` with `info`, `check`, `reach`,
  `ambiguity`, `benchmark` and `policy`; exit 0 on success, 1 on a finding with
  `--fail-on-findings`, 2 on a refused request.
- Two packaged declarations: `reaction_wheel_fmea`, an illustrative 8-fault
  declaration containing two ambiguity classes and two deliberately vacuous
  responses, and `planted_unreachable`, the planted broken case.
- 262 tests, 8 validation scripts with their raw output committed,
  `docs/REQUIREMENTS.md` with 32 numbered requirements and 6 stated non-requirements, `MODEL_CARD.md` and
  `DATASET_CARD.md`.

### Measured and reported as measured

- The learned classifier does **not** beat the declared threshold logic on
  identification accuracy at an equal false-alarm rate: two wins, two losses
  and one tie over five noise scales. It does have the lower expected
  calibration error at 3 of 5 scales, and the baseline beats it on calibration
  at the two quietest.
- The cost of routing on confidence is **+9.1456 min** per commanded event with
  the learned detector in `fine_pointing`, buying 200 of 206 avoided
  mis-routes, and **-2.3278 min** with the declared threshold logic in
  `coarse_pointing`, where it is cheaper and safer. The sign is not universal.
- Reachability and cheapest-path results agree with exhaustive enumeration on
  2000 random declarations with a worst deviation of 0.000e+00 min.

### Known limitations at this release

See README "Limitations". The load-bearing ones: no physical validation; the
ambiguity relation is the static special case of discrete-event diagnosability;
preconditions are conjunctions only; one fault at a time; cost is outage
minutes and credits nothing else; the synthetic generator has independent
Gaussian channels; and no citation was verified against a publisher page in the
build session, because egress reached `pypi.org` only.

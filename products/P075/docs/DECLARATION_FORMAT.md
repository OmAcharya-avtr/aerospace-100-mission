# The declaration format

A `reliablock` diagram is a YAML document. It is the artifact this package
exists to make checkable, so the loader is strict: an unknown key is an error,
not a silently ignored field. A typo in `rate_per_hour` that fell through to a
default would report a reliability of 1.0 and no complaint.

Every example in this document is exercised by
`validation/regenerate_declarations.py`, whose committed output contains the
exact rejection message for each refusal listed here.

## Top level

```yaml
name: smallsat bus            # required, non-empty
description: ...              # optional
notes:                        # optional, list of strings
  - The beta factors are declared inputs, not data.
root: { ... }                 # required, a node
```

No other top-level key is accepted.

## Distributions

Every block carries one. Each parameter is required: nothing is defaulted.

```yaml
distribution: {kind: exponential, rate_per_hour: 1.2e-6}
distribution: {kind: weibull, shape: 0.85, scale_hours: 5.0e6}
distribution: {kind: lognormal, log_median: 9.0, log_sigma: 0.5}
distribution: {kind: fixed_probability, probability: 1.0e-3}
```

| kind | parameters | units | cumulative hazard |
|---|---|---|---|
| `exponential` | `rate_per_hour` | 1/hour | `H = lambda t` |
| `weibull` | `shape`, `scale_hours` | dimensionless, hours | `H = (t / eta)**beta` |
| `lognormal` | `log_median`, `log_sigma` | log-hours, dimensionless | `H = -log(1 - Phi((ln t - mu) / sigma))` |
| `fixed_probability` | `probability` | dimensionless | `H = -log1p(-q)`, independent of `t` |

A `fixed_probability` block is a one-shot item -- a pyrotechnic, a deployment, a
switch -- whose figure of merit is a probability per actuation. A mission
profile **cannot age it**: a phase that declares a duty cycle or a stress factor
for such a block is refused, because multiplying a probability by a duty cycle
is not a model of anything.

## Nodes

### `block`

```yaml
{kind: block, name: power_string_a, distribution: {...}, description: PCDU string A}
```

`name` is the identifier used in cut sets and importance tables and must be
unique across the diagram.

### `series`

```yaml
{kind: series, label: bus, items: [{...}, {...}]}
```

Succeeds only if every child succeeds. `H = sum H_i` exactly.

### `parallel`

```yaml
{kind: parallel, label: power, beta_factor: 0.12, items: [{...}, {...}]}
```

Succeeds if at least one child succeeds. At least two items are required:
redundancy over one item is not redundancy. `Q = prod Q_i` exactly.

### `k_of_n`

```yaml
{kind: k_of_n, k: 3, label: reaction_wheels, beta_factor: 0.08, items: [{...} x 4]}
```

Succeeds if at least `k` of the `n` children succeed. `k` must lie in `1..n`.
A `beta_factor` is refused at `k = n`, because that is a series group and not a
redundant one.

### `standby`

```yaml
{kind: standby,
 label: transmitter,
 primary: {...},
 standbys: [{...}],
 switch_reliability: 0.995,
 standby_hazard_fraction: 0.0}
```

`switch_reliability` is the probability that the changeover succeeds when
called upon, dimensionless in `[0, 1]`. **It is required.** An undeclared
switching reliability defaulting to 1.0 is exactly the implicit assumption this
package exists to surface.

`standby_hazard_fraction` is the fraction of its operating hazard rate a
standby unit accrues while waiting: `0.0` cold, `1.0` hot, between warm.

Supported combinations: any number of identical cold standby units at a
constant rate (Erlang closed form), or one standby unit of any distribution at
any hazard fraction (convolution integral). Anything else raises with the
reason.

### `shared_load`

```yaml
{kind: shared_load, label: pumps, shared_rate_factor: 0.45, units: [{...}, {...}]}
```

Exactly two units, both with an `exponential` distribution whose rate is the
unit's **full-load** rate. `shared_rate_factor` in `(0, 1]` is the fraction of
that rate a unit carries while both are alive. There is one source of truth for
every rate -- the units' own distributions -- so the declaration cannot
contradict itself.

A `shared_load` node does not accept a `beta_factor`: it is already a
dependent-failure model.

### `network`

```yaml
{kind: network,
 label: bridge,
 source: s,
 sink: t,
 edges:
   - {from: s, to: a, block: {...}}
   - {from: s, to: b, block: {...}}
   - {from: a, to: b, block: {...}}
   - {from: a, to: t, block: {...}}
   - {from: b, to: t, block: {...}}}
```

An explicit two-terminal edge list, for a structure that is not
series-parallel. The system succeeds when surviving edges connect `source` to
`sink`. The bridge above is the canonical case and the reason this node type
exists: no nesting of `series` and `parallel` reproduces it.

Edges are undirected -- a reliability block diagram's edges carry items, not
flow direction.

## Mission profiles

```yaml
name: LEO phased, 3 years
phases:
  - name: launch_and_leop
    duration_hours: 72.0
    stress_factors: {power_string_a: 3.0, obc: 2.0}
    duty_cycles: {wheel_a: 0.1, transmitter_main: 0.3}
  - name: routine_operations
    duration_hours: 25506.0
    default_duty_cycle: 1.0
    duty_cycles: {transmitter_main: 0.12}
```

A block's effective age is
`sum over phases of duration * duty_cycle * stress_factor`, and its cumulative
hazard is then `H(age_eff)`. This is the cumulative-exposure (Nelson) model.
It is **exact for an exponential block** and an approximation otherwise --
see `validation/validate_mission.py`, which declares and fails a check that it
agrees with calendar-time integration of a scaled hazard rate.

## Every refusal, and why

| rejected | reason |
|---|---|
| unknown top-level, node or distribution key | a typo that fell through to a default would report a reliability of 1.0 and no complaint |
| a missing distribution parameter | no parameter of a failure distribution is defaulted |
| a duplicate block name | two different items would share one failure probability and one importance measure, and the cut sets would be wrong |
| a `parallel` or `k_of_n` group with fewer than two items | redundancy over one item is not redundancy |
| `k` outside `1..n` | `k = 0` is certain success and `k > n` certain failure; neither is a diagram |
| a `standby` with no declared `switch_reliability` | it is the term this package exists to make explicit |
| a `switch_reliability` or `standby_hazard_fraction` outside `[0, 1]` | neither is a probability otherwise |
| a `shared_load` with other than two units, or a non-exponential unit | the four-state Markov closed form has no form for either |
| a `shared_rate_factor` outside `(0, 1]` | a factor above 1 would mean sharing a load makes a unit fail faster than carrying all of it |
| a `beta_factor` outside `[0, 1]` | beta is a fraction of a component's total failure rate |
| a `beta_factor` on anything but `parallel` or `k_of_n` with `k < n` | a `standby` or `shared_load` node is already a dependent-failure model; a `k = n` group is a series group, and a beta factor there makes the arithmetic look better (measured ratio `0.918939` at `k = n = 6`, `beta = 0.1`) |
| a `beta_factor` over members that do not share one distribution | the beta-factor model is symmetric and needs one `lambda`; the usual fixes give three different answers with no principled basis |
| a `beta_factor` over nested structures | the model fails a whole group through one shared event |
| a `network` self-loop, `source == sink`, or a disconnected sink | none of them is a two-terminal structure |
| a `network` edge on no source-sink path | its reliability could never enter the answer |
| a mission phase that ages a one-shot block | a probability per actuation is not a rate |
| a negative duration, or a duty cycle outside `[0, 1]`, or a negative stress factor | none of them is the quantity it is named for |

## Round-tripping

`Diagram.to_yaml()` writes a declaration with sorted keys, so two diagrams
diff cleanly. `load_diagram()` reads it back.
`validation/regenerate_declarations.py` checks, for every built-in case, both
that the text round-trips and that the reloaded diagram **quantifies to the
same reliability** -- the second being the property that actually matters,
because a declaration that round-trips textually while quantifying differently
would be worse than one that fails to load.

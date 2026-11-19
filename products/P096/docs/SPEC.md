# Specification — handoverloss

Validation level 2. Research-grade; not flight-qualified, not certified, and not
approved for operational aerospace use.

## What is specified

End-to-end data loss across a ground-station handover on an optical downlink,
attributed separately to **acquisition time**, **protocol restart** and **buffer
drain**. The headline output is **which of the three dominates**, not the total.

## The declared handover timeline

Measured from the instant the outgoing optical link breaks. All times seconds,
all rates bit/s, all volumes bits.

| Phase | Interval | Egress rate | What is happening |
|---|---|---|---|
| Acquisition | `[0, T_acq)` | 0 | Beacon detection, spatial acquisition, coarse then fine pointing lock at the incoming station |
| Restart | `[T_acq, T_o)`, `T_o = T_acq + T_rst` | 0 | The optical link carries photons but no payload: the link layer and transport session must be re-established |
| Drain (ramp) | `[T_o, T_o + T_ramp)` | `R_d (t - T_o) / T_ramp` | The link is up and the backlog begins to drain, but while the instantaneous egress is below `R_s` the buffer is still filling |
| Drain (full) | `[T_o + T_ramp, H)` | `R_d` | Backlog drains if `R_d > R_s` |

Throughout, a source of mean rate `R_s` feeds a tail-drop buffer of capacity `B`
holding `q0` at the break. Data arriving at a full buffer is lost and is never
recovered.

## Required inputs, with bounds

| Symbol | Field | Units | Bound |
|---|---|---|---|
| `R_s` | `source_rate_bps` | bit/s | `> 0` |
| `R_d` | `downlink_rate_bps` | bit/s | `> 0` |
| `B` | `buffer_capacity_bits` | bits | `> 0` |
| `q0` | `initial_occupancy_bits` | bits | `[0, B]` |
| `T_acq` | `acquisition_s` | s | `>= 0` (0 means make-before-break) |
| `T_rst` | `restart_s` | s | `>= 0` |
| `T_ramp` | `ramp_s` | s | `>= 0` (0 means the egress steps) |
| `H` | `horizon_s` | s | `>= T_acq + T_rst + T_ramp` |

Every bound is enforced at construction by
`handoverloss.sequence.HandoverSequence`, which raises `ValueError` naming the
field and the offending value, or `TypeError` for a non-real field.

## Required outputs

`handoverloss.accounting.attribute` returns an `AttributionReport` carrying:

- three `MechanismShare` objects in the fixed order
  `("acquisition", "restart", "drain")`, each with a loss in bits, a bootstrap
  lower and upper bound, and a share of the total;
- `dominant`: the mechanism with the largest point estimate;
- `dominance_is_decided`: true only when the leading mechanism's interval is
  disjoint from the runner-up's. **This is a different claim from `dominant` and
  the two are never collapsed.**
- `total_loss_bits` and `conservation_total_bits`: the total by two independent
  accountings;
- `residual_bits`, `residual_relative` and `residual_tolerance_bits`: the
  difference between those two accountings, as a reported number with a
  conditioning-scaled tolerance beside it;
- `resampling_unit`: asserted to equal `"handover event"` at construction.

## The three baselines

1. **Baseline 1 — the acquisition-only assumption.**
   `max(0, R_s * T_acq - (B - q0))`. Implemented deliberately, at full strength,
   as the thing being refuted, because it is what a link budget and a pointing
   analysis imply. It credits the buffer with its full free space and is the
   exact restriction of baseline 2 to its first phase; that identity is asserted
   by `tests/test_baselines.py` and measured over 400 handovers by
   `validation/validate_baselines_against_truth.py`. An under-strength version
   would inflate the finding.
2. **Baseline 2 — the analytic fluid model.** The closed form derived in the
   `handoverloss.baselines` module docstring, evaluated in headroom space. Gives
   all three mechanism terms from the declared sequence with no integration.
3. **Baseline 3 — ground truth.** Direct accounting over the simulated handover
   trace, with event-level bootstrap confidence intervals.

## The independent bound

`handoverloss.bounds.loss_bracket` brackets the total from conservation and
capacity counting alone: `max(0, A - S - (B - q0)) <= L <= A`, where `A` is the
offered volume and `S` the integral of the declared egress schedule. It imports
no model module of this package, which is asserted structurally by
`tests/test_bounds.py` and by
`validation/validate_independent_bound.py`.

## The learned component

Predicts the three per-mechanism losses from **pre-handover state only** — no
realised duration and no realised loss is a feature, which
`handoverloss.corpus.check_feature_causality` asserts by name. Exposes an
uncertainty output (the across-tree spread) and a dominance vote share. Scored
on held-out **station pairs** and held-out **time** separately, never averaged.

## Resampling unit

**The handover event.** Slots within one handover share an occupancy state, so
resampling slots would report an interval narrower than the data support. The
unit is declared on `BootstrapInterval` and `AttributionReport` and both refuse
construction with any other value.

## Compute budget

Declared: training and any Monte Carlo under 60 s of CPU each; the whole pytest
suite well under 10 minutes under contention. Measured figures are in
`validation/VALIDATION.md`.

## Out of scope

- Make-before-break handover with both links up at once, beyond `T_acq = 0`.
- A handover that fails and returns to the outgoing station.
- Adaptive coding and modulation changing `R_d` during the pass.
- Any loss mechanism on the ground segment after delivery.
- Retransmission or recovery of dropped data.
- Any measured handover, acquisition time or optical link. Nothing in this
  repository is measured on hardware.

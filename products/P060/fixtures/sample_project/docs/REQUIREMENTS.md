# Sample requirements — reaction-wheel desaturation monitor

This document is a **fixture**, not a specification of anything real. It exists
so that every finding code in `traceaudit` has a committed input that produces
it, and so the worked example in README.md runs on data that ships with the
repository. The component it describes is illustrative; no vehicle, mission or
hardware is referred to.

Declaration format: a requirement is declared by its id at the start of a line,
optionally behind a markdown heading marker or a list bullet. An id mentioned
inside a sentence — for instance, "unlike REQ-002, the next one is optional" —
is a reference and is deliberately not a declaration.

## REQ-001 Wheel speed input validation

The monitor shall reject a wheel-speed sample outside the declared range.

## REQ-002 Momentum accumulation

The monitor shall accumulate stored momentum from the wheel-speed history.

## REQ-003 Desaturation threshold

The monitor shall raise a desaturation request when stored momentum exceeds
the declared threshold.

## REQ-004 Hysteresis

The monitor shall not withdraw a desaturation request until stored momentum
falls below the declared threshold by the declared hysteresis band.

## REQ-005 Telemetry record

The monitor shall record every desaturation request with its trigger value.

## REQ-006 Request counter

The monitor shall count desaturation requests since reset.

## REQ-007 Reset behaviour

The monitor shall clear accumulated momentum on reset.

## REQ-008 Saturation ceiling

The monitor shall clamp stored momentum at the declared ceiling.

## Appendix — a second declaration of an existing id

The line below declares REQ-007 again, which is the input for finding TA003.
It is the common real failure: a requirement copied into an appendix during a
document merge.

## REQ-007 Reset behaviour (duplicated during a document merge)

The monitor shall clear accumulated momentum on reset.

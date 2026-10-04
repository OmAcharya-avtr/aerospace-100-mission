# The timing model, in one page

Everything in `rtclock` rests on four definitions. They are written out here
because most disagreements about timing numbers turn out to be disagreements
about one of these.

## 1. Time

One canonical unit: **seconds (s)**, SI base unit (BIPM, *The International
System of Units*, 9th ed., 2019). Every public function takes and returns
seconds unless its name ends in `_ms`, `_us` or `_ns`. `rtclock.units` holds
the conversions and nothing else.

A **timebase** is a monotonic non-decreasing clock plus a sleep primitive.
`now()` returns seconds since an unspecified epoch: differences are
meaningful, absolute values are not.

## 2. Resolution, and why it is measured

A clock advances in steps. The step is **measured**, never assumed:
`measure_clock_resolution` takes N back-to-back readings of
`time.monotonic_ns()` and reports

- the **observable tick** `q`: the smallest non-zero forward difference. No
  interval shorter than `q` can be distinguished from zero.
- the **per-read cost**: the median forward difference, an upper bound on the
  shortest interval the clock can *time* rather than merely represent.
- the **identical-reading fraction**: if it is zero, the read loop was slower
  than the clock and `q` is an *upper bound* on the hardware granularity.

A duration is a difference of two quantized readings. Treating each
endpoint's rounding error as independent and uniform on `[-q/2, q/2]`
(JCGM 100:2008, the GUM, Sec. 4.3.7):

```
worst case   |dt - dt_true| <= q
standard     u(dt) = sqrt(2) * q/sqrt(12) = q/sqrt(6)
```

Both are carried as an error term wherever this package reports a timing
number. `validation/clock_resolution_output.txt` holds the measurement for
the machine that built this repository.

## 3. The periodic task, and the two scheduling views

A periodic task is `(T, C, D)` in seconds: released every `T`, needing at
most `C` of processor time, due `D` after release. Liu & Layland's
assumptions (JACM 20(1), 1973, Sec. 2) apply throughout: one processor, full
preemption, zero context-switch cost, independent tasks, no release jitter.
`D = T` is the *implicit-deadline* case the utilization bounds assume;
`D < T` is *constrained* and is handled only by response-time analysis.

Two things are then computed, and they are not the same thing:

| | question | answer | strength |
|---|---|---|---|
| utilization bound | can I tell quickly that this set is fine? | `U <= n(2^(1/n)-1)` for RM; `U <= 1` for EDF | sufficient (RM), exact (EDF, `D = T`) |
| response-time analysis | what is the worst-case response time? | fixed point of `R = C + B + sum ceil(R/T_j) C_j` | exact for the model |

A `False` from a sufficient test means **unknown**, not **infeasible**. Every
`SchedulabilityResult` carries the word, so the distinction cannot be lost in
transcription.

## 4. Drift, and the two ways to wait

The ideal release instant of iteration `k` is `t0 + k*T`. Drift is always
measured against that:

```
drift_k = start_k - (t0 + k*T)      units s, positive means late
```

What differs between the two scheduling modes is how the wait is computed,
not how drift is defined:

- **absolute**: wait until `t0 + k*T`, recomputed from the clock each
  iteration. Error in one wait is corrected by the next.
- **relative**: wait a fixed `T` after the previous release. Every wait's
  error adds to the total.

With a sleep primitive of fractional rate error `eps` and constant wake-up
delay `d`, writing `a = T*eps + d`:

```
relative:  drift_k = k * a                               unbounded
absolute:  drift_k = a (1 - (-eps)^k) / (1 + eps),  k>=1  bounded by |a|/(1-|eps|)
```

Both are derived by unrolling `wait_k = T - drift_(k-1)` and are checked
against the implementation in `validation/validate_loop_drift.py`.

## 5. Overrun

An iteration overruns a budget when its latency is **strictly greater** than
the budget. Equality is not an overrun, and no tolerance is applied. The
comparison is on the raw binary64 values, so the count is a pure function of
the stored trace. This is the convention used for the P031/P036 cross-check
in `validation/crosscheck_overruns.json`.

## What this model does not contain

No garbage-collection pauses, no GIL contention, no OS scheduling latency, no
cache or bus effects, no interrupt load, no multi-core migration, no release
jitter, no context-switch cost. CPython on a general-purpose operating system
exhibits all of them. This package analyses timing; it does not deliver it.

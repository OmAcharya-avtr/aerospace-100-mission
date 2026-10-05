"""Validation 6: the wrapper does not require modifying the target.

Two claims are checked, both of which the README depends on.

1. **Bit-transparency.**  With no injections the wrapper must return exactly the
   binary64 values the bare target returns.  The reference is an independent
   re-implementation of the closed loop written in this script, which drives the
   target directly and never imports the wrapper.  If the wrapper perturbed the
   loop -- an extra float conversion, a reordered sum -- the byte images would
   differ.

2. **Interface-only coupling.**  A target the package has never seen, defined in
   this script with no faultinject imports and a completely different internal
   design, is wrapped and injected into successfully.  The wrapper requires
   ``reset()`` and ``step(k, meas)`` and nothing else.

Two behavioural distinctions documented in ``faults.py`` are also checked here,
because a taxonomy whose kinds secretly coincide would inflate the coverage
denominator:

3. ``bus_reorder`` actually reorders (delivers frames out of order), and with
   the minimum swap probability still differs from fresh delivery by exactly one
   step of latency.
4. ``bus_delay`` leaves a *persistent* transport latency after its window
   closes -- a FIFO queue at matched producer and consumer rates never drains --
   while ``timing_late_sample`` returns to fresh delivery immediately.

Run from products/P034/:  python validation/validate_transparency.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


from faultinject.faults import Injection  # noqa: E402
from faultinject.harness import (  # noqa: E402
    Trace,
    fault_rng,
    noise_stream,
    run_case,
)
from faultinject.target import (  # noqa: E402
    DT,
    SIGMA_ACCEL,
    SIGMA_POS,
    SIGMA_VEL,
    DoubleIntegratorPlant,
    GncController,
    reference,
)
from faultinject.taxonomy import FaultKind  # noqa: E402
from faultinject.wrapper import InjectionWrapper  # noqa: E402

N_STEPS = 150
failures: list[str] = []


def reference_loop(seed: int, n_steps: int = N_STEPS) -> bytes:
    """Independent closed loop driving the target directly, no wrapper."""
    ctrl = GncController()
    ctrl.reset()
    plant = DoubleIntegratorPlant()
    plant.reset()
    noise = noise_stream(seed, n_steps)
    sp = 0.5 * SIGMA_ACCEL * DT * DT
    sv = SIGMA_ACCEL * DT
    tr = Trace(seed=seed, n_steps=n_steps)
    for k in range(n_steps):
        meas = plant.measure(SIGMA_POS * float(noise[k, 0]), SIGMA_VEL * float(noise[k, 1]))
        cmd = ctrl.step(k, meas)
        u = float(cmd["u"])
        tr.p_true.append(plant.p)
        tr.v_true.append(plant.v)
        tr.p_hat.append(ctrl.p_hat)
        tr.v_hat.append(ctrl.v_hat)
        tr.u_applied.append(u)
        tr.ref.append(reference(k, DT))
        plant.advance(u, sp * float(noise[k, 2]), sv * float(noise[k, 3]))
    tr.innovations = list(ctrl.innovations)
    return tr.float_bytes()


class ForeignTarget:
    """A target written without any knowledge of this package.

    Integral controller on the position error with its own internal state and
    its own naming. It imports nothing from faultinject.
    """

    def __init__(self) -> None:
        self.accumulator = 0.0
        self.calls = 0

    def reset(self) -> None:
        self.accumulator = 0.0
        self.calls = 0

    def step(self, index, sample):  # noqa: ANN001, ANN201 - deliberately untyped
        self.calls += 1
        target_position = reference(index, DT)
        self.accumulator += 0.2 * (target_position - sample["pos"])
        return {"u": 4.0 * self.accumulator - 1.5 * sample["vel"]}


print("=" * 78)
print("VALIDATION 6 -- the wrapper does not require modifying the target")
print("=" * 78)
print()

# ------------------------------------------------------------------------- 6a
print("6a. Bit-transparency against an independent closed loop, 50 seeds")
mismatches = []
for seed in range(1, 51):
    a = reference_loop(seed)
    b = run_case((), seed, N_STEPS).float_bytes()
    if a != b:
        mismatches.append(seed)
print("    seeds compared        50")
print(f"    bytes per trace       {len(reference_loop(1))}")
print(f"    byte-image mismatches {len(mismatches)}")
if mismatches:
    failures.append(f"6a: wrapper is not bit-transparent for seeds {mismatches[:5]}")
else:
    print("    PASS  the wrapper with no injections changes nothing, bit for bit")
print()

# ------------------------------------------------------------------------- 6b
print("6b. A foreign target, wrapped and injected without modification")
foreign = ForeignTarget()
wrapper = InjectionWrapper(
    foreign,
    [Injection.create(FaultKind.SENSOR_BIAS, "pos", {"offset": 2.0}, 10, 30)],
    fault_rng(5),
)
wrapper.reset()
clean_cmds = []
faulted_cmds = []
bare = ForeignTarget()
bare.reset()
for k in range(40):
    frame = {"pos": 0.1 * k, "vel": 0.05, "valid": 1.0}
    faulted_cmds.append(wrapper.step(k, frame)["u"])
    clean_cmds.append(bare.step(k, frame)["u"])
same_before = all(
    a == b for a, b in zip(faulted_cmds[:10], clean_cmds[:10], strict=True)
)
differ_after = any(
    a != b for a, b in zip(faulted_cmds[10:], clean_cmds[10:], strict=True)
)
print(f"    target class          {type(foreign).__name__} (no faultinject imports)")
print("    interface used        reset(), step(k, meas)")
print(f"    steps before the window identical to the bare target : {same_before}")
print(f"    steps inside/after the window differ                 : {differ_after}")
print(f"    target call count     {foreign.calls} (expected 40)")
if not same_before:
    failures.append("6b: wrapper altered the foreign target before the injection window")
if not differ_after:
    failures.append("6b: injection had no effect on the foreign target")
if foreign.calls != 40:
    failures.append(f"6b: foreign target called {foreign.calls} times, expected 40")
if not failures:
    print("    PASS")
print()

# ------------------------------------------------------------------------- 6c
print("6c. bus_reorder really reorders, and its rate is non-monotonic in swap_prob")
print()
print("    The mechanism holds one frame and may deliver the newer one first, so an")
print("    exchange needs a swap followed by a non-swap. At swap_prob = 1 every step")
print("    swaps, the held frame is never released, and nothing arrives out of order;")
print("    the rate peaks near swap_prob = 0.5. This non-monotonicity is real and is")
print("    recorded in the README limitations.")
print()
print(f"    {'swap_prob':>10} {'swaps':>7} {'out of order':>13} {'first 8 source steps':>26}")
print("    " + "-" * 60)
oo_by_p = {}
for swap_prob in (0.1, 0.25, 0.5, 0.75, 1.0):
    inj = Injection.create(FaultKind.BUS_REORDER, "bus", {"swap_prob": swap_prob}, 0, N_STEPS)
    w = InjectionWrapper(GncController(), [inj], fault_rng(4))
    w.reset()
    for k in range(60):
        w.step(k, {"pos": float(k), "vel": 0.0, "valid": 1.0})
    handler = w.handlers[0][1]
    oo = getattr(handler, "out_of_order", -1)
    oo_by_p[swap_prob] = oo
    print(f"    {swap_prob:>10} {handler.swaps:>7} {oo:>13} "
          f"{str(w.delivered_source[:8]):>26}")
if oo_by_p[0.5] <= 0:
    failures.append("6c: swap_prob=0.5 produced no out-of-order delivery")
if oo_by_p[1.0] != 0:
    failures.append(
        f"6c: swap_prob=1.0 produced {oo_by_p[1.0]} out-of-order deliveries; the "
        "documented saturation behaviour no longer matches the code"
    )
print()
print("    Pure one-step delay when no swap fires: searching fault seeds at")
print("    swap_prob = 0.1 for a 20-step run with zero swaps.")
found = None
for seed in range(1, 60):
    inj = Injection.create(FaultKind.BUS_REORDER, "bus", {"swap_prob": 0.1}, 0, N_STEPS)
    w = InjectionWrapper(GncController(), [inj], fault_rng(seed))
    w.reset()
    for k in range(20):
        w.step(k, {"pos": float(k), "vel": 0.0, "valid": 1.0})
    if w.handlers[0][1].swaps == 0:
        found = (seed, list(w.delivered_source))
        break
if found is None:
    failures.append("6c: no zero-swap seed found, so the pure-delay claim is unverified")
else:
    seed, sources = found
    expected = [None] + list(range(19))
    ok = sources == expected
    print(f"    fault seed {seed}: sources {sources[:8]} ... == one-step delay: {ok}")
    if not ok:
        failures.append(f"6c: zero-swap delivery {sources[:8]} is not a pure one-step delay")
print()

# ------------------------------------------------------------------------- 6d
print("6d. bus_delay leaves a persistent latency after its window;")
print("    timing_late_sample recovers immediately")
print()
window_end = 30
for kind, params in (
    (FaultKind.BUS_DELAY, {"delay_steps": 4.0}),
    (FaultKind.TIMING_LATE_SAMPLE, {"late_steps": 4.0}),
):
    inj = Injection.create(kind, "bus", params, 10, window_end - 10)
    w = InjectionWrapper(GncController(), [inj], fault_rng(4))
    w.reset()
    for k in range(40):
        w.step(k, {"pos": float(k), "vel": 0.0, "valid": 1.0})
    staleness = [
        (k - s) if s is not None else None for k, s in enumerate(w.delivered_source)
    ]
    after = staleness[window_end : window_end + 6]
    print(f"    {kind.value:<20} staleness at steps {window_end}..{window_end + 5}: {after}")
    if kind is FaultKind.BUS_DELAY:
        expected = int(params["delay_steps"])
        if any(x != expected for x in after):
            failures.append(
                f"6d: bus_delay staleness after the window {after}, expected all {expected}"
            )
        else:
            print(f"                         PASS  still {expected} steps stale, as documented")
    else:
        if any(x not in (0, None) for x in after):
            failures.append("6d: timing_late_sample left staleness after its window closed")
        else:
            print("                         PASS  fresh again immediately, as documented")
print()

print("=" * 78)
if failures:
    print(f"RESULT: FAILED ({len(failures)} check(s))")
    for f in failures:
        print(f"  FAILED  {f}")
    sys.exit(1)
print("RESULT: PASS -- all checks in validation 6 passed")
print("=" * 78)

"""Check 1 — simulated and device backends are bit-identical on a seeded case.

Requirement: "simulated and device backends produce bit-identical results on a
seeded deterministic case where the device backend is pointed at a loopback
stub".

What is compared, exactly
-------------------------
* The signal matrix, byte for byte: ``[theta_meas, omega_meas, theta_hat,
  command, applied]`` per iteration, float64, compared with ``bytes ==``.
* The data digest (SHA-256 over the signal bytes, the per-iteration flags and
  the counters).
* The full digest, which additionally covers every stage duration and
  completion time. This is only meaningful because the run injects its
  durations and therefore reads no wall clock.

What this does not establish
----------------------------
The device backend is driven by ``LoopbackDriver``, which is the plant model
behind the driver-shim interface. It is **not hardware**. The check proves that
the HAL path, the loop, the estimator, the control law and the accounting
behave identically whichever backend they are given. It says nothing about
timing on a board.

Run: ``python validation/parity_simdev.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hilforge.backends import make_backend_pair
from hilforge.loop import HilLoop, LoopConfig
from hilforge.timing import PeriodSpec

PERIOD = 0.010
SEED = 20261004
CASES = (
    ("constant 0.40 T", lambda n: np.full(n, 0.40 * PERIOD)),
    ("ramp 0.30 T -> 1.35 T", lambda n: np.linspace(0.30 * PERIOD, 1.35 * PERIOD, n)),
    ("sawtooth with overruns", lambda n: PERIOD * (0.3 + 1.2 * ((np.arange(n) % 7) / 6.0))),
)
N = 1500


def main() -> int:
    print("HilForge validation 1 — simulation/device parity")
    print("=" * 72)
    print(f"period {PERIOD:.6e} s   seed {SEED}   iterations {N}")
    print("device backend driver: loopback stub (NOT hardware)")
    print()
    failures = 0
    for label, maker in CASES:
        sim, dev = make_backend_pair(seed=SEED, sample_dt_s=PERIOD)
        cfg = LoopConfig(
            period=PeriodSpec(period_s=PERIOD),
            n_iterations=N,
            injected_durations_s=tuple(maker(N)),
        )
        a = HilLoop(sim, cfg).run()
        b = HilLoop(dev, cfg).run()
        sig_a = a.signal_matrix()
        sig_b = b.signal_matrix()
        bytes_equal = sig_a.tobytes() == sig_b.tobytes()
        max_abs = float(np.max(np.abs(sig_a - sig_b)))
        data_equal = a.data_digest() == b.data_digest()
        full_equal = a.full_digest() == b.full_digest()
        acct_equal = a.overruns.as_dict() == b.overruns.as_dict()
        ok = bytes_equal and data_equal and full_equal and acct_equal
        failures += 0 if ok else 1
        print(f"case: {label}")
        print(f"  simulated kind/driver : {a.backend_kind} / {a.backend_driver or 'n/a'}")
        print(f"  device    kind/driver : {b.backend_kind} / {b.backend_driver}")
        print(f"  is_hardware (both)    : {a.is_hardware} / {b.is_hardware}")
        print(f"  iterations completed  : {a.n_completed} / {b.n_completed}")
        print(f"  signal bytes identical: {bytes_equal}")
        print(f"  max |difference|      : {max_abs:.3e}  (tolerance: exactly 0)")
        print(f"  data digest           : {a.data_digest()}")
        print(f"  data digests equal    : {data_equal}")
        print(f"  full digest           : {a.full_digest()}")
        print(f"  full digests equal    : {full_equal}")
        print(f"  overrun accounts equal: {acct_equal}")
        print(f"  direct / cascade      : {a.overruns.direct_count} / "
              f"{a.overruns.cascade_count}")
        print(f"  RESULT                : {'PASS' if ok else 'FAIL'}")
        print()
    print(f"verdict: {len(CASES) - failures}/{len(CASES)} cases bit-identical")
    print("note: parity covers the data path and the injected timing. Hardware")
    print("      timing is not measured here and is not claimed anywhere.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

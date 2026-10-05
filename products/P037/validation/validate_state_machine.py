"""The acquisition state machine reproduces a hand-traced sequence exactly.

The hand trace is written out in full in ``tests/test_sync.py`` as a comment
and reproduced in the output below. The configuration is
search_tolerance = 0, lock_tolerance = 3, check_required = 2,
flywheel_max = 3, and the sequence is 18 observations chosen to exercise
every transition in the machine: a false start in CHECK, promotion to LOCK,
a single-miss flywheel entry and recovery, a two-miss flywheel that
recovers, a three-miss flywheel that expires, and re-acquisition.

Also checked here: the machine driven by real correlator output over a
marker-prefixed bit stream at a channel error rate, and the slip /
re-acquisition behaviour across a known bit insertion and deletion.

Run: ``python3 validate_state_machine.py`` from this directory.
"""

from __future__ import annotations

import sys
import time

import numpy as np

sys.path.insert(0, "../src")

from framesync.asm import hamming_distances  # noqa: E402
from framesync.channel import bpsk_ber, bsc_flip  # noqa: E402
from framesync.frames import FrameGeometry, build_stream  # noqa: E402
from framesync.sync import (  # noqa: E402
    FrameSynchroniser,
    SyncConfig,
    SyncState,
    analyse_slip,
)

CONFIG = SyncConfig(search_tolerance=0, lock_tolerance=3, check_required=2, flywheel_max=3)

OBSERVATIONS = [
    "miss", "hit", "miss", "hit", "hit", "hit", "miss", "hit", "miss",
    "miss", "hit", "miss", "miss", "miss", "hit", "hit", "hit", "miss",
]

# Hand-derived expected state after each observation (see tests/test_sync.py
# for the per-step derivation with the counter values).
EXPECTED_STATES = [
    "SEARCH", "CHECK", "SEARCH", "CHECK", "LOCK", "LOCK", "FLYWHEEL", "LOCK",
    "FLYWHEEL", "FLYWHEEL", "LOCK", "FLYWHEEL", "FLYWHEEL", "SEARCH", "CHECK",
    "LOCK", "LOCK", "FLYWHEEL",
]
EXPECTED_MISS = [0, 0, 0, 0, 0, 0, 1, 0, 1, 2, 0, 1, 2, 0, 0, 0, 0, 1]
EXPECTED_DELIVERING = 12


def main() -> int:
    t0 = time.time()
    all_pass = True
    print("framesync validation 4 -- acquisition state machine hand trace")
    print("=" * 78)
    print(f"config: search_tolerance={CONFIG.search_tolerance}, "
          f"lock_tolerance={CONFIG.lock_tolerance}, "
          f"check_required={CONFIG.check_required}, "
          f"flywheel_max={CONFIG.flywheel_max}")
    print()
    sm = FrameSynchroniser(CONFIG)
    delivering = 0
    header = (
        f"{'i':>3}  {'obs':<5}  {'from':<9}    {'to':<9}  {'miss':>4}  {'chk':>3}  "
        f"{'hand: to':<9}  {'hand: miss':>10}  result"
    )
    print(header)
    print("-" * len(header))
    for i, obs in enumerate(OBSERVATIONS):
        ev = sm.step(obs)
        delivering += int(sm.delivering)
        ok = ev.state_after.value == EXPECTED_STATES[i] and ev.miss_count == EXPECTED_MISS[i]
        all_pass &= ok
        print(
            f"{i:3d}  {obs:<5}  {ev.state_before.value:<9} -> {ev.state_after.value:<9}  "
            f"{ev.miss_count:4d}  {ev.check_count:3d}  {EXPECTED_STATES[i]:<9}  "
            f"{EXPECTED_MISS[i]:10d}  {'PASS' if ok else 'FAIL'}"
        )
    print()
    seq_ok = [e.state_after.value for e in sm.history] == EXPECTED_STATES
    del_ok = delivering == EXPECTED_DELIVERING
    all_pass &= seq_ok and del_ok
    print(f"full 18-step state sequence matches the hand trace exactly: {seq_ok} "
          f"-> {'PASS' if seq_ok else 'FAIL'}")
    print(f"observations in a delivering state (LOCK or FLYWHEEL): {delivering}, "
          f"hand trace says {EXPECTED_DELIVERING} -> {'PASS' if del_ok else 'FAIL'}")
    print()

    print("Machine driven by real correlator output over a noisy stream")
    print("-" * 78)
    geom = FrameGeometry(data_octets=223, fecf=True, asm_bits=32)
    for ebn0 in (-2.0, 0.0, 2.0, 6.0, 12.0):
        rng = np.random.default_rng(20261005)
        stream, offsets = build_stream(geom, 60, rng)
        p = float(bpsk_ber(ebn0)[0])
        rx = bsc_flip(stream, p, rng)
        dist = hamming_distances(rx)
        sm2 = FrameSynchroniser(SyncConfig())
        locked = 0
        for k in range(60):
            pos = k * geom.period_bits
            tol = (
                sm2.config.search_tolerance
                if sm2.state is SyncState.SEARCH
                else sm2.config.lock_tolerance
            )
            sm2.step("hit" if dist[pos] <= tol else "miss")
            locked += int(sm2.state is SyncState.LOCK)
        print(
            f"  Eb/N0 = {ebn0:5.2f} dB, p = {p:.4e}: observations in LOCK "
            f"{locked}/60, final state {sm2.state.value}"
        )
    print("  (reported, not a pass/fail criterion. The marker survives far below")
    print("   the data threshold because the lock tolerance is 3 bits in 32: at")
    print("   Eb/N0 = 6 dB the expected marker errors are 32p = 0.08 bits, so the")
    print("   synchroniser holds lock in a region where the frame error rate is 1.)")
    print()

    print("Slip behaviour across a known insertion and deletion")
    print("-" * 78)
    print(f"default config: flywheel_max={SyncConfig().flywheel_max}, so the hand")
    print("expectation is 4 frames delivered at the wrong phase (1 flywheel entry")
    print("plus 3 further misses) before SEARCH is re-entered")
    for slip in (+3, +1, -2, -7):
        rep = analyse_slip(frame_bits=1024, n_frames=16, slip_at_frame=7, slip_bits=slip)
        ok = rep.frames_delivered_at_wrong_phase == 4 and rep.final_state is SyncState.SEARCH
        all_pass &= ok
        print(
            f"  slip {slip:+3d} bits: correct-phase frames {rep.frames_before_slip:3d}, "
            f"wrong-phase frames {rep.frames_delivered_at_wrong_phase}, "
            f"frames to SEARCH {rep.reacquisition_frames}, "
            f"final {rep.final_state.value} -> {'PASS' if ok else 'FAIL'}"
        )
    print()
    print(f"ALL CHECKS PASS: {all_pass}")
    print(f"wall time: {time.time() - t0:.1f} s")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())

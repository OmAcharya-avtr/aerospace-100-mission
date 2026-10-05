"""Acquisition, flywheel and loss of sync over a deteriorating link.

Produces ../screenshots/acquisition_timeline.png: the synchroniser state
against frame index for a stream whose Eb/N0 falls in steps, driven by real
correlator output over marker-prefixed frames. The flywheel ride-through and
the loss-of-sync transition are visible as the channel degrades, and the
recovery is visible when it improves again.
"""

from __future__ import annotations

import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, "../src")

from framesync.asm import hamming_distances  # noqa: E402
from framesync.channel import bpsk_ber, bsc_flip  # noqa: E402
from framesync.frames import FrameGeometry, build_stream  # noqa: E402
from framesync.sync import FrameSynchroniser, SyncConfig, SyncState  # noqa: E402

SEED = 20261005
GEOM = FrameGeometry(data_octets=223, fecf=True, asm_bits=32)
CONFIG = SyncConfig(search_tolerance=0, lock_tolerance=3, check_required=2, flywheel_max=4)
# Eb/N0 schedule in dB, one entry per frame: good, collapsing, recovered.
SCHEDULE = [8.0] * 12 + [1.0] * 6 + [-4.0] * 10 + [1.0] * 6 + [8.0] * 12
ORDER = [SyncState.SEARCH, SyncState.CHECK, SyncState.FLYWHEEL, SyncState.LOCK]


def main() -> int:
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    stream, _ = build_stream(GEOM, len(SCHEDULE), rng)

    # Apply a per-frame error rate so the channel degrades on schedule.
    rx = stream.copy()
    for k, ebn0 in enumerate(SCHEDULE):
        lo = k * GEOM.period_bits
        hi = lo + GEOM.period_bits
        rx[lo:hi] = bsc_flip(stream[lo:hi], float(bpsk_ber(ebn0)[0]), rng)
    dist = hamming_distances(rx)

    sm = FrameSynchroniser(CONFIG)
    states, obs, marker_errors = [], [], []
    for k in range(len(SCHEDULE)):
        pos = k * GEOM.period_bits
        tol = (
            CONFIG.search_tolerance if sm.state is SyncState.SEARCH else CONFIG.lock_tolerance
        )
        marker_errors.append(int(dist[pos]))
        o = "hit" if dist[pos] <= tol else "miss"
        obs.append(o)
        sm.step(o)
        states.append(sm.state)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.6, 5.8), sharex=True,
                                   height_ratios=[1.0, 1.4])
    ax1.step(range(len(SCHEDULE)), SCHEDULE, where="mid", color="#1f3b73", lw=1.6)
    ax1.set_ylabel("Eb/N0 (dB)")
    ax1.set_title("Sync acquisition, flywheel and loss over a deteriorating link\n"
                  f"lock tolerance {CONFIG.lock_tolerance} bits in 32, "
                  f"check {CONFIG.check_required}, flywheel {CONFIG.flywheel_max}")
    ax1.grid(True, alpha=0.3)
    ax1b = ax1.twinx()
    ax1b.plot(range(len(SCHEDULE)), marker_errors, ".", color="#9b2226", ms=5)
    ax1b.axhline(CONFIG.lock_tolerance + 0.5, color="#9b2226", ls=":", lw=1.0)
    ax1b.set_ylabel("marker bit errors", color="#9b2226")

    y = [ORDER.index(s) for s in states]
    ax2.step(range(len(SCHEDULE)), y, where="mid", color="#2a7f62", lw=1.8)
    ax2.scatter(
        [i for i, o in enumerate(obs) if o == "miss"],
        [y[i] for i, o in enumerate(obs) if o == "miss"],
        marker="x", color="#9b2226", s=42, zorder=3, label="marker miss",
    )
    ax2.set_yticks(range(len(ORDER)))
    ax2.set_yticklabels([s.value for s in ORDER])
    ax2.set_xlabel("frame index")
    ax2.set_ylabel("synchroniser state")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8.5, loc="lower left")

    fig.tight_layout()
    out = "../screenshots/acquisition_timeline.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)

    print("acquisition timeline (framesync example 3)")
    print("-" * 72)
    print(f"{'frame':>6}  {'Eb/N0':>7}  {'marker errs':>11}  {'obs':<5}  state")
    for k in range(len(SCHEDULE)):
        print(f"{k:6d}  {SCHEDULE[k]:7.1f}  {marker_errors[k]:11d}  "
              f"{obs[k]:<5}  {states[k].value}")
    n_lock = sum(s is SyncState.LOCK for s in states)
    n_fly = sum(s is SyncState.FLYWHEEL for s in states)
    n_search = sum(s is SyncState.SEARCH for s in states)
    print(f"\nframes in LOCK {n_lock}, FLYWHEEL {n_fly}, SEARCH {n_search}, "
          f"CHECK {len(SCHEDULE) - n_lock - n_fly - n_search}")
    print(f"saved {out} ({time.time() - t0:.1f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

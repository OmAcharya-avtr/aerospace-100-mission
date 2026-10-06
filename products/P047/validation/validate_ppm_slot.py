"""M-ary PPM slot clock: the exact known answer, the jitter chain, and the slot error floor.

Run from this directory with ``PYTHONPATH=../src``.  Runtime about 60 s.

Five checks:

1. the slot S-curve against a closed form derived by hand - the half-sine slot with
   quarter-slot gates gives exactly ``S(eps) = sin(2 pi eps)`` and ``K_d = 2 pi``;
2. the slot gain is independent of the PPM order, because only the slot carrying
   the pulse contributes;
3. the per-update loop bandwidth against the per-slot one, and the jitter chain
   (white, coloured, Monte Carlo) in slot periods;
4. the duty-cycle penalty measured against the order;
5. the slot-index error rate, which is the quantity a PPM link actually cares
   about and which has no counterpart in binary symbol timing.
"""

from __future__ import annotations

import math
import time

import numpy as np

from slotsync.loop import LoopDesign, jitter_variance_closed_form, jitter_variance_coloured
from slotsync.ppm import (
    PpmConfig,
    duty_cycle_penalty_db,
    equivalent_slot_bandwidth,
    measure_ppm_slot_statistics,
    ppm_slot_autocovariance,
    ppm_slot_scurve,
    run_ppm_slot_loop,
)
from slotsync.pulses import half_sine
from slotsync.scurve import default_offsets

SLOT = half_sine(1.0)
ZETA = 1.0 / math.sqrt(2.0)
SHARP = np.linspace(-0.001, 0.001, 21)


def banner(text: str) -> None:
    print()
    print(text)
    print("-" * len(text))


def gain_for(config: PpmConfig) -> float:
    return ppm_slot_scurve(config, SLOT, offsets=SHARP, fit_halfwidth=0.001).gain_central_difference


def main() -> int:
    started = time.perf_counter()
    print("slotsync validation: PPM slot clock")
    print(f"slot pulse {SLOT.name} (support {SLOT.half_support} slot), zeta {ZETA:.6f}")
    print("every quantity below is in SLOT periods; B_n is per loop update unless stated")

    banner("1. known answer: S(eps) = sin(2 pi eps) and K_d = 2 pi exactly")
    print("   p(t) = cos(pi t) on |t| <= 1/2 and gates at +-1/4 slot give")
    print("   e = cos^2(pi(eps - 1/4)) - cos^2(pi(eps + 1/4)) = sin(2 pi eps) for |eps| <= 1/4")
    offsets = np.linspace(-0.24, 0.24, 97)
    curve = ppm_slot_scurve(PpmConfig(4, 0.25, True), SLOT, offsets=offsets)
    residual = float(np.max(np.abs(curve.values - np.sin(2.0 * np.pi * offsets))))
    sharp = ppm_slot_scurve(PpmConfig(4, 0.25, True), SLOT, offsets=SHARP, fit_halfwidth=0.001)
    gain_error = abs(sharp.gain_central_difference - 2.0 * math.pi) / (2.0 * math.pi)
    print(f"   max |S(eps) - sin(2 pi eps)| over 97 offsets in [-0.24, 0.24]: {residual:.3e}")
    print(
        f"   K_d measured {sharp.gain_central_difference:.9f} against 2 pi = "
        f"{2.0 * math.pi:.9f}, relative error {gain_error:.3e}"
    )
    wide = ppm_slot_scurve(PpmConfig(4, 0.25, True), SLOT, offsets=default_offsets(0.5, 201))
    print(
        f"   peak at {wide.peak_offset:.4f} slot (the gate spacing), reversal "
        f"{wide.reversal_offset}"
    )
    print("   there is no reversal inside half a slot: with a slot pulse narrower than one")
    print("   slot the detector output decays towards zero instead of crossing it, so the")
    print("   half-period barrier that bounds a symbol-timing cycle slip does not exist here.")

    banner("2. the slot gain does not depend on the PPM order")
    print(f"   {'order M':>8} {'K_d per slot':>14} {'rel error vs 2 pi':>19} {'patterns':>10}")
    worst_order_error = 0.0
    for order in (2, 4, 8, 16):
        sharp_order = ppm_slot_scurve(
            PpmConfig(order, 0.25, True), SLOT, offsets=SHARP, fit_halfwidth=0.001
        )
        error = abs(sharp_order.gain_central_difference - 2.0 * math.pi) / (2.0 * math.pi)
        worst_order_error = max(worst_order_error, error)
        print(
            f"   {order:8d} {sharp_order.gain_central_difference:14.9f} {error:19.3e} "
            f"{sharp_order.pattern_count:10d}"
        )
    print("   only the slot carrying the pulse contributes, so the gain is a slot property.")
    print("   What the order changes is the update rate and the noise per update, not the gain.")

    banner("3. the jitter chain in slot periods, 4-PPM, max-energy slot decision")
    config = PpmConfig(4, 0.25, False)
    gain = gain_for(config)
    print(
        f"   {'B_n/update':>11} {'B_n/slot':>10} {'white cf':>12} {'coloured':>12} "
        f"{'monte carlo':>12} {'mc s.e.':>10} {'mc/col':>8} {'slot errs':>10}"
    )
    worst_relative = 0.0
    for bandwidth in (0.002, 0.005, 0.01, 0.02):
        autocovariance = ppm_slot_autocovariance(
            config, SLOT, sample_snr_db=20.0, max_lag=8, symbols=30000, seed=7
        )
        design = LoopDesign.from_bandwidth(bandwidth, ZETA, gain)
        run = run_ppm_slot_loop(
            config, SLOT, design, n_symbols=60000, sample_snr_db=20.0, seed=20261006
        )
        white = jitter_variance_closed_form(bandwidth, gain, autocovariance[0])
        coloured = jitter_variance_coloured(design, autocovariance)
        ratio = run.jitter_variance / coloured
        worst_relative = max(worst_relative, abs(ratio - 1.0))
        print(
            f"   {bandwidth:11.4f} {equivalent_slot_bandwidth(bandwidth, 4):10.5f} "
            f"{white:12.5e} {coloured:12.5e} {run.jitter_variance:12.5e} "
            f"{run.jitter_variance_standard_error:10.2e} {ratio:8.4f} "
            f"{run.slot_error_rate:10.4f}"
        )
    print("   the white and coloured routes nearly coincide here: the slot detector's output")
    print("   noise is almost white (|R[j]/R[0]| < 0.05 for j >= 1) because its self-noise is")
    print("   zero - a return-to-zero slot narrower than one slot has no inter-slot")
    print("   interference. The Monte Carlo sits within "
          f"{worst_relative * 100.0:.0f} % of the prediction, with the")
    print("   deviation growing with B_n as it does for symbol timing; the squared-gate")
    print("   detector is quadratic in the noise so its output variance depends on the timing")
    print("   error itself, which no linearised prediction contains.")

    banner("4. the duty-cycle penalty, measured against the order")
    print("   at a fixed per-sample SNR the gain is order-independent but the chance that noise")
    print("   wins a slot is not, so both the detector output variance and the slot error rate")
    print("   grow with M. The jitter factor is 10 log10(sigma_n^2 / K_d^2) referred to a")
    print("   unit-variance detector of gain 2 pi; a difference in it is a difference in loop")
    print("   SNR in dB. 'vs M=2' is the same quantity relative to binary PPM, which is the")
    print("   duty-cycle penalty proper.")
    print(
        f"   {'order M':>8} {'snr_dB':>7} {'sigma_n^2':>11} {'slot err rate':>14} "
        f"{'factor dB':>11} {'vs M=2 dB':>11} {'10log10(M/2)':>13}"
    )
    baseline: dict[float, float] = {}
    for snr in (20.0, 6.0):
        for order in (2, 4, 8, 16):
            configuration = PpmConfig(order, 0.25, False)
            stats = measure_ppm_slot_statistics(
                configuration, SLOT, sample_snr_db=snr, symbols=6000, seed=3
            )
            factor = duty_cycle_penalty_db(
                configuration,
                SLOT,
                sample_snr_db=snr,
                reference_gain=2.0 * math.pi,
                symbols=6000,
                seed=3,
            )
            if order == 2:
                baseline[snr] = factor
            print(
                f"   {order:8d} {snr:7.1f} {stats['variance']:11.6f} "
                f"{stats['slot_error_rate']:14.5f} {factor:11.4f} "
                f"{factor - baseline[snr]:11.4f} {10.0 * math.log10(order / 2):13.4f}"
            )
    print("   the penalty is not 10 log10(M/2): at 20 dB it is 0.0 to 0.2 dB from M = 2 to")
    print("   M = 16, because no slot decision is ever wrong and the extra empty slots cost")
    print("   nothing; at 6 dB it is 0.6 to 2.2 dB, driven entirely by the slot error rate")
    print("   climbing from 0.18 to 0.61. The measured numbers are published instead of the")
    print("   rule of thumb, and the two disagree in both directions.")

    banner("5. the slot-index error floor, which symbol timing has no counterpart for")
    print("   closed-loop 8-PPM: the loop re-acquires on whichever slot wins, so a slot slip")
    print("   never appears as a loop slip - it appears as a wrong symbol.")
    print(
        f"   {'snr_dB':>8} {'rms jitter (slot)':>19} {'slot err rate':>14} "
        f"{'loop slips':>11} {'locked':>7}"
    )
    config8 = PpmConfig(8, 0.25, False)
    gain8 = gain_for(config8)
    design8 = LoopDesign.from_bandwidth(0.01, ZETA, gain8)
    for snr in (20.0, 10.0, 6.0, 2.0, -2.0):
        run = run_ppm_slot_loop(
            config8, SLOT, design8, n_symbols=30000, sample_snr_db=snr, seed=20261006
        )
        print(
            f"   {snr:8.1f} {run.jitter_rms:19.6f} {run.slot_error_rate:14.5f} "
            f"{run.slip_count:11d} {str(not run.diverged):>7}"
        )
    print("   while the loop stays locked the slot error rate rises from 0 to 0.81 and the loop")
    print("   reports no slips at all. A receiver designed against a cycle-slip budget alone")
    print("   would see none of this.")
    print("   The last row is unlocked and its slot error rate is NOT a meaningful error rate:")
    print("   a free-running slot clock that has drifted an integer number of slots picks the")
    print("   wrong slot index systematically, and whenever the drift happens to be a multiple")
    print("   of M it picks the right one by accident. That aliasing is why the figure falls")
    print("   rather than rising. It is reported as unlocked rather than deleted.")
    print("   Resolving which slot begins a symbol is a further M-fold ambiguity that slot")
    print("   timing cannot touch and that needs frame synchronisation.")

    banner("summary")
    print(f"   known answer, max |S - sin(2 pi eps)|: {residual:.3e}  (gate 1e-10)")
    print(f"   known answer, K_d relative error:      {gain_error:.3e}  (gate 1e-05)")
    print(f"   gain order-independence, worst error:  {worst_order_error:.3e}  (gate 1e-05)")
    print(f"   jitter chain, worst relative deviation of Monte Carlo: {worst_relative * 100:.1f} %")
    print("   gate: known answers within tolerance and jitter deviation under 25 %")
    passed = residual < 1e-10 and gain_error < 1e-5 and worst_order_error < 1e-5
    passed = passed and worst_relative < 0.25
    print(f"   within tolerance: {passed}")
    print(f"   elapsed {time.perf_counter() - started:.1f} s")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

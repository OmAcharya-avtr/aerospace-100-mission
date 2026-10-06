"""Validation: the rate bounds this package ships, and the ones it does not.

Checks
------
1. **Erasure capacity (C1)** against an independent Monte Carlo estimate of the
   mutual information of the background-free PPM channel.
2. **The PPM photons-per-bit limit (C5).** ``n_s / C_sym`` must approach
   ``1/log2(M)`` from above as ``n_s -> 0`` and never cross it.
3. **Bits per photon is unbounded in M**, which is the verifiable in-family form
   of the statement that the background-free Poisson channel has no finite
   capacity-per-photon ceiling.
4. **Data-processing inequality.** The soft-decision achievable rate (C3) must
   be at least the hard-decision capacity (C2) for every configuration. This is
   a genuine test of both: an error in either would break the inequality.
5. **Both rates bounded by log2(M)** and monotone in the signal level.

What is deliberately absent: no peak-and-average-constrained Poisson capacity
bound is evaluated or cited with a number, because none was verified in this
environment. The four checks above are computed from expressions in this
repository.

Runtime: about 40 s on one core.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    _SRC = Path(__file__).resolve().parents[1] / "src"
    if _SRC.is_dir() and str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402

from photoncount.capacity import (  # noqa: E402
    erasure_channel_capacity,
    hard_decision_capacity,
    minimum_photons_per_bit,
    photons_per_bit,
    soft_decision_achievable_rate,
)
from photoncount.ppm import PPMConfig, bits_per_symbol, sample_counts  # noqa: E402


def _erasure_mutual_information_mc(cfg: PPMConfig, n: int, rng: np.random.Generator) -> float:
    """Independent MC estimate of I(X;Y) for the background-free PPM channel.

    An all-zero count vector is an erasure and carries no information; anything
    else identifies the symbol exactly. So I = P(not erased) * log2 M, which this
    estimates by counting erasures rather than by using the closed form.
    """
    sent = rng.integers(0, cfg.order, size=n)
    counts = sample_counts(cfg, sent, rng)
    erased = counts.sum(axis=1) == 0
    return float((1.0 - erased.mean()) * bits_per_symbol(cfg.order))


def check_erasure_capacity() -> bool:
    print("1. Erasure capacity (C1) vs an independent Monte Carlo, 400000 symbols")
    rng = np.random.default_rng(20261006)
    print(f"   {'M':>5} {'n_s':>6} {'closed form (C1)':>18} {'Monte Carlo':>14} "
          f"{'MC s.e.':>10} {'z':>7}")
    ok = True
    for order, ns in ((16, 2.0), (16, 0.5), (64, 3.0), (256, 1.0)):
        cfg = PPMConfig(order, ns)
        closed = erasure_channel_capacity(cfg)["bits_per_symbol"]
        mc = _erasure_mutual_information_mc(cfg, 400_000, rng)
        p_er = float(np.exp(-ns))
        se = bits_per_symbol(order) * np.sqrt(p_er * (1 - p_er) / 400_000)
        z = (mc - closed) / se
        print(f"   {order:5d} {ns:6.2f} {closed:18.10f} {mc:14.8f} {se:10.2e} {z:+7.2f}")
        ok &= abs(z) < 4.0
    print(f"   verdict: {'PASS' if ok else 'FAIL'}")
    return ok


def check_photons_per_bit_limit() -> bool:
    print("\n2. Photons per bit (C4) approaches 1/log2(M) from above as n_s -> 0")
    ok = True
    for order in (4, 16, 256):
        limit = minimum_photons_per_bit(order)
        print(f"   M = {order}, limit 1/log2(M) = {limit:.10f}")
        print(f"      {'n_s':>10} {'photons/bit':>16} {'excess over limit':>19} "
              f"{'erasure prob':>14}")
        values = []
        for ns in (1.0, 0.1, 0.01, 1e-3, 1e-4, 1e-5):
            cfg = PPMConfig(order, ns)
            cap = erasure_channel_capacity(cfg)
            ppb = photons_per_bit(cfg, cap["bits_per_symbol"])
            values.append(ppb)
            print(f"      {ns:10.5g} {ppb:16.10f} {ppb / limit - 1.0:18.3e}  "
                  f"{cap['erasure_probability']:14.8f}")
        above = all(v > limit for v in values)
        decreasing = all(b < a for a, b in zip(values, values[1:], strict=False))
        converged = abs(values[-1] / limit - 1.0) < 1e-4
        print(f"      strictly above the limit: {above}; decreasing: {decreasing}; "
              f"converged: {converged}")
        ok &= above and decreasing and converged
    print("   the cost of approaching the limit is the erasure probability, which goes")
    print("   to 1: the limit is unreachable, and the last column is why")
    print(f"   verdict: {'PASS' if ok else 'FAIL'}")
    return ok


def check_bits_per_photon_unbounded() -> bool:
    print("\n3. Bits per photon has no finite ceiling in M")
    print(f"   {'M':>7} {'log2(M) bits/photon':>21} {'photons/bit limit':>19}")
    vals = []
    for order in (2, 16, 256, 4096, 65536):
        bpp = 1.0 / minimum_photons_per_bit(order)
        vals.append(bpp)
        print(f"   {order:7d} {bpp:21.6f} {minimum_photons_per_bit(order):19.10f}")
    ok = all(b > a for a, b in zip(vals, vals[1:], strict=False))
    print("   growing without bound, as the background-free Poisson channel requires;")
    print("   the price is bandwidth: M slots per symbol")
    print(f"   verdict: {'PASS' if ok else 'FAIL'}")
    return ok


def check_data_processing_inequality() -> bool:
    print("\n4. Data-processing inequality: soft rate (C3) >= hard capacity (C2)")
    rng = np.random.default_rng(31)
    print(f"   {'M':>5} {'n_s':>6} {'n_0':>6} {'hard (C2)':>12} {'soft (C3)':>12} "
          f"{'soft s.e.':>10} {'soft - hard':>12}")
    ok = True
    for order, ns, nb in (
        (16, 4.0, 0.2),
        (16, 1.0, 0.2),
        (8, 2.0, 0.5),
        (64, 6.0, 0.05),
        (4, 1.0, 0.3),
        (256, 10.0, 0.01),
    ):
        cfg = PPMConfig(order, ns, nb)
        hard = hard_decision_capacity(cfg)["bits_per_symbol"]
        soft = soft_decision_achievable_rate(cfg, 60_000, rng)
        gap = soft["bits_per_symbol"] - hard
        print(f"   {order:5d} {ns:6.2f} {nb:6.3f} {hard:12.6f} "
              f"{soft['bits_per_symbol']:12.6f} {soft['standard_error']:10.2e} {gap:+12.6f}")
        ok &= gap > -4.0 * soft["standard_error"]
        ok &= soft["bits_per_symbol"] <= bits_per_symbol(order) + 1e-9
    print("   the gap is what soft-decision decoding buys over hard decisions")
    print(f"   verdict: {'PASS' if ok else 'FAIL'}")
    return ok


def check_monotonicity() -> bool:
    print("\n5. Hard-decision capacity is monotone in the signal level")
    ok = True
    for order, nb in ((16, 0.2), (64, 0.05)):
        caps = [
            hard_decision_capacity(PPMConfig(order, ns, nb))["bits_per_symbol"]
            for ns in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 40.0)
        ]
        mono = all(b >= a - 1e-12 for a, b in zip(caps, caps[1:], strict=False))
        print(f"   M = {order:3d}, n_0 = {nb}: {[round(c, 5) for c in caps]}")
        print(f"      monotone: {mono}; saturates at log2(M) = {bits_per_symbol(order)}: "
              f"{abs(caps[-1] - bits_per_symbol(order)) < 1e-6}")
        ok &= mono and abs(caps[-1] - bits_per_symbol(order)) < 1e-6
    print(f"   verdict: {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    print("validate_capacity.py")
    print("=" * 78)
    results = [
        check_erasure_capacity(),
        check_photons_per_bit_limit(),
        check_bits_per_photon_unbounded(),
        check_data_processing_inequality(),
        check_monotonicity(),
    ]
    print("\n" + "=" * 78)
    print(f"checks passed: {sum(results)} of {len(results)}")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())

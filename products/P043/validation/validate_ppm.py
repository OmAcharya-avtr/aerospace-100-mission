"""Validation: PPM slot statistics, the exact error probability, and soft metrics.

Checks
------
1. **Background-free closed form.** With no background the channel is an erasure
   channel and ``P_e = exp(-n_s) (M-1)/M`` exactly. The general summation (P4)
   must reproduce that to machine precision for every order and signal level.
2. **Exact against Monte Carlo with background.** (P4) against a simulated
   receiver over several configurations, reported as z scores against the
   binomial standard error of the Monte Carlo.
3. **The soft metric derivation.** (P3) claims the posterior depends on the
   counts only through ``k_i ln(1 + n_s/n_0)``. That is checked against a
   brute-force evaluation of the full Poisson product likelihood, including the
   factorials and the terms (P2) drops.
4. **Maximum likelihood is argmax of the count.** The slot the exact posterior
   maximises must be the slot with the largest count, for randomly drawn count
   vectors.
5. **Bit LLR consistency.** The bit posteriors reconstructed from the LLRs must
   reproduce the symbol posterior marginals, and an all-zero symbol must give
   exactly zero LLRs.
6. **Truncation of the sum over counts** is reported as the Poisson tail mass
   left out, which bounds the error in (P4).

Reference: J. Hamkins and B. Moision, "Multipulse Pulse-Position Modulation on
Discrete Memoryless Channels", IPN Progress Report 42-161, JPL, 15 May 2005, for
the deep-space PPM context. The expressions evaluated here are derived in
``photoncount.ppm`` and verified against simulation, not quoted from the report.

Runtime: about 25 s on one core.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    _SRC = Path(__file__).resolve().parents[1] / "src"
    if _SRC.is_dir() and str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402
from scipy import special, stats  # noqa: E402

from photoncount.ppm import (  # noqa: E402
    PPMConfig,
    bit_llrs,
    erasure_probability,
    sample_counts,
    slot_metric_scale,
    symbol_error_probability,
    symbol_error_probability_mc,
    symbol_log_posterior,
)


def check_background_free_closed_form() -> bool:
    print("1. Background-free: exact summation (P4) vs exp(-n_s)(M-1)/M")
    print(f"   {'M':>5} {'n_s':>7} {'summation (P4)':>18} {'closed form':>18} "
          f"{'abs dev':>11}")
    ok = True
    for order, ns in ((2, 0.5), (4, 1.0), (16, 0.5), (16, 2.0), (64, 8.0), (256, 3.0)):
        cfg = PPMConfig(order, ns)
        exact = symbol_error_probability(cfg)["error"]
        closed = float(np.exp(-ns) * (order - 1) / order)
        dev = abs(exact - closed)
        print(f"   {order:5d} {ns:7.2f} {exact:18.14f} {closed:18.14f} {dev:11.2e}")
        ok &= dev < 1e-12
    print(f"   verdict: {'PASS' if ok else 'FAIL'} (tolerance 1e-12 absolute)")
    return ok


def check_exact_against_monte_carlo() -> bool:
    print("\n2. With background: exact (P4) vs Monte Carlo, 400000 symbols each")
    rng = np.random.default_rng(20261006)
    print(f"   {'M':>5} {'n_s':>6} {'n_0':>6} {'exact':>13} {'MC':>13} {'MC s.e.':>11} "
          f"{'z':>7}")
    ok = True
    for order, ns, nb in (
        (16, 4.0, 0.2),
        (8, 3.0, 1.0),
        (4, 1.0, 0.5),
        (64, 6.0, 0.05),
        (256, 10.0, 0.01),
        (16, 1.0, 0.3),
    ):
        cfg = PPMConfig(order, ns, nb)
        exact = symbol_error_probability(cfg)["error"]
        mc = symbol_error_probability_mc(cfg, 400_000, rng)
        z = (mc["error"] - exact) / max(mc["standard_error"], 1e-15)
        print(f"   {order:5d} {ns:6.2f} {nb:6.3f} {exact:13.8f} {mc['error']:13.8f} "
              f"{mc['standard_error']:11.2e} {z:+7.2f}")
        ok &= abs(z) < 4.0
    print(f"   verdict: {'PASS' if ok else 'FAIL'} (|z| < 4 for every configuration)")
    return ok


def check_soft_metric_derivation() -> bool:
    print("\n3. (P3) against the full Poisson product likelihood")
    cfg = PPMConfig(16, 4.0, 0.25)
    rng = np.random.default_rng(7)
    sent = rng.integers(0, cfg.order, size=2000)
    counts = sample_counts(cfg, sent, rng)
    fast = symbol_log_posterior(counts, cfg)
    # Brute force: log prod_i Poisson(k_i; n_0 + n_s [i == j]), normalised.
    brute = np.empty_like(fast)
    for j in range(cfg.order):
        means = np.full(cfg.order, cfg.noise_counts)
        means[j] += cfg.signal_counts
        brute[:, j] = np.sum(stats.poisson.logpmf(counts, means[None, :]), axis=1)
    brute -= special.logsumexp(brute, axis=1, keepdims=True)
    worst = float(np.max(np.abs(fast - brute)))
    print(f"   scale ln(1 + n_s/n_0) = {slot_metric_scale(cfg):.10f} nats per count")
    print(f"   worst absolute deviation in log posterior over 2000 symbols x 16 slots: "
          f"{worst:.3e}")
    ok = worst < 1e-10
    print(f"   verdict: {'PASS' if ok else 'FAIL'} (tolerance 1e-10 nats)")
    return ok


def check_argmax_is_maximum_likelihood() -> bool:
    print("\n4. argmax of the posterior equals argmax of the count")
    rng = np.random.default_rng(11)
    ok = True
    for order, ns, nb in ((16, 3.0, 0.2), (64, 5.0, 0.5)):
        cfg = PPMConfig(order, ns, nb)
        counts = sample_counts(cfg, rng.integers(0, order, size=5000), rng)
        post_arg = np.argmax(symbol_log_posterior(counts, cfg), axis=1)
        count_arg = np.argmax(counts, axis=1)
        agree = float(np.mean(post_arg == count_arg))
        print(f"   M = {order:3d}: agreement over 5000 symbols = {agree:.6f}")
        ok &= agree == 1.0
    print(f"   verdict: {'PASS' if ok else 'FAIL'}")
    return ok


def check_bit_llrs() -> bool:
    print("\n5. Bit LLR consistency against the symbol posterior marginals")
    cfg = PPMConfig(16, 4.0, 0.2)
    rng = np.random.default_rng(13)
    counts = sample_counts(cfg, rng.integers(0, 16, size=3000), rng)
    llr = bit_llrs(counts, cfg)
    post = np.exp(symbol_log_posterior(counts, cfg))
    bits = ((np.arange(16)[:, None] >> np.arange(3, -1, -1)[None, :]) & 1).astype(bool)
    worst = 0.0
    for pos in range(4):
        p_one = post[:, bits[:, pos]].sum(axis=1)
        p_zero = post[:, ~bits[:, pos]].sum(axis=1)
        from_llr = 1.0 / (1.0 + np.exp(llr[:, pos]))
        worst = max(worst, float(np.max(np.abs(from_llr - p_one / (p_one + p_zero)))))
    empty = bit_llrs(np.zeros((1, 16), dtype=int), cfg)
    print(f"   worst |P(bit=1) from LLR - marginal| over 3000 symbols x 4 bits: "
          f"{worst:.3e}")
    print(f"   LLRs for an all-zero symbol: max |LLR| = {float(np.max(np.abs(empty))):.3e} "
          "(must be 0: no counts, no information)")
    ok = worst < 1e-10 and float(np.max(np.abs(empty))) < 1e-12
    print(f"   verdict: {'PASS' if ok else 'FAIL'}")
    return ok


def check_truncation_and_erasure() -> bool:
    print("\n6. Truncation tail mass and the all-slots-empty probability")
    print(f"   {'M':>5} {'n_s':>6} {'n_0':>6} {'truncation K':>13} {'tail mass':>12} "
          f"{'P(all empty)':>14}")
    ok = True
    for order, ns, nb in ((16, 2.0, 0.0), (16, 4.0, 0.2), (256, 10.0, 0.01)):
        cfg = PPMConfig(order, ns, nb)
        out = symbol_error_probability(cfg)
        p_empty = erasure_probability(cfg)
        print(f"   {order:5d} {ns:6.2f} {nb:6.3f} {int(out['truncation']):13d} "
              f"{out['tail_mass']:12.2e} {p_empty:14.8f}")
        ok &= out["tail_mass"] < 1e-12
    print("   background-free check: P(all empty) must equal exp(-n_s)")
    cfg = PPMConfig(16, 2.0)
    dev = abs(erasure_probability(cfg) - float(np.exp(-2.0)))
    print(f"   |P(all empty) - exp(-2)| = {dev:.2e}")
    ok &= dev < 1e-15
    print(f"   verdict: {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    print("validate_ppm.py")
    print("Hamkins & Moision, IPN Progress Report 42-161, JPL, 15 May 2005")
    print("=" * 78)
    results = [
        check_background_free_closed_form(),
        check_exact_against_monte_carlo(),
        check_soft_metric_derivation(),
        check_argmax_is_maximum_likelihood(),
        check_bit_llrs(),
        check_truncation_and_erasure(),
    ]
    print("\n" + "=" * 78)
    print(f"checks passed: {sum(results)} of {len(results)}")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())

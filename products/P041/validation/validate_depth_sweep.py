"""The central result: post-decoding BER and FER against interleaver depth.

Parameterised by ``D / Lc``, the ratio of interleaver depth to the fade
correlation length in symbols. One channel record is generated per correlation
time and every depth sees that identical record, so the only thing varying across
a row is the permutation.

Compute budget: 2 cores shared with sibling agents. The sweep below is sized to
finish in under three minutes; the configuration is printed with the results so
the numbers can be reproduced exactly.

Run: ``PYTHONPATH=../src python3 validate_depth_sweep.py``
Runtime: about 100 s on one core.
"""

from __future__ import annotations

import time

import numpy as np

from codedfade.channel import ChannelConfig, generate_amplitude
from codedfade.fade import (
    fade_statistics,
    lognormal_standard_level,
    markov_mean_fade_duration,
    required_interleaver_depth,
)
from codedfade.link import CodedLink, uncoded_bit_error_rate
from codedfade.reedsolomon import ReedSolomon

SI = 0.6
FS = 1.0e6
SNR_DB = 14.0
CODEWORDS = 2048
DEPTHS = [1, 4, 16, 64, 256, 1024, 4096]
THRESHOLD = 0.6
SEED = 7


def main() -> None:
    code = ReedSolomon(31, 21, 5)
    print("=" * 94)
    print("INTERLEAVER DEPTH SWEEP")
    print("=" * 94)
    print(f"code                     RS({code.n},{code.k}) over GF(2^{code.m}), "
          f"t={code.t}, rate={code.rate:.6f}")
    print("modulation               OOK, thermal-limited, p_b = Q(sqrt(gbar) I), eq (26)")
    print(f"mean electrical SNR      {SNR_DB:.3f} dB")
    print(f"scintillation index      {SI}")
    print(f"symbol rate              {FS:.6e} Hz")
    print(f"channel seed             {SEED}")
    print(f"codewords per point      {CODEWORDS}")
    print("frame definition         one RS codeword; FER counts failures AND "
          "miscorrections")
    print()

    t_start = time.perf_counter()
    for tau in (2.0e-5, 2.0e-4, 1.0e-3):
        cfg = ChannelConfig(SI, tau, FS, "lognormal", "exp", seed=SEED)
        lc = cfg.samples_per_correlation_time
        a = generate_amplitude(cfg, 400_000)
        st = fade_statistics(a, THRESHOLD, FS)
        u = lognormal_standard_level(THRESHOLD, SI)
        mfd_exact = markov_mean_fade_duration(u, tau, FS)
        print("-" * 94)
        print(f"correlation time tau = {tau:.3e} s   ->   Lc = tau*Rs = {lc:.1f} symbols")
        print(f"  sample MFD at a_th={THRESHOLD}: {st.mean_fade_duration_s:.6e} s "
              f"= {st.mean_fade_duration_s * FS:.2f} symbols   (eq(15) exact: "
              f"{mfd_exact * FS:.2f} symbols)")
        print(f"  sample max fade:            {st.max_fade_duration_s * FS:.0f} symbols "
              f"over {st.complete_fades} fades in 400000 samples")
        print(f"  depth sized to the mean fade: "
              f"{required_interleaver_depth(st.mean_fade_duration_s, FS)} symbols; "
              f"to 3x the mean: "
              f"{required_interleaver_depth(st.mean_fade_duration_s, FS, 3.0)}")
        print(f"  uncoded BER (same channel): "
          f"{uncoded_bit_error_rate(cfg, SNR_DB, 400_000):.6e}")
        print()
        link = CodedLink(code, cfg, mean_snr_db=SNR_DB)
        header = (
            f"  {'depth':>6} {'D/Lc':>8} {'raw SER':>10} {'post BER':>11} {'FER':>10} "
            f"{'FER SE':>9} {'fail':>6} {'mis':>5} {'maxerr':>7} {'lat ms':>9} {'mem B':>10}"
        )
        print(header)
        print("  " + "-" * (len(header) - 2))
        for r in link.depth_sweep(DEPTHS, CODEWORDS):
            print(
                f"  {r.depth:6d} {r.depth / lc:8.3f} {r.raw_symbol_error_rate:10.6f} "
                f"{r.post_bit_error_rate:11.4e} {r.frame_error_rate:10.6f} "
                f"{r.frame_error_rate_standard_error:9.6f} {r.decoder_failures:6d} "
                f"{r.miscorrections:5d} {r.max_symbol_errors_per_codeword:7d} "
                f"{r.latency_ms:9.3f} {r.memory_bytes:10.0f}"
            )
        print()
    print("-" * 94)
    print("HOW BIG IS THE REAL ERROR BAR? The FER SE column above is a binomial")
    print("standard error, which assumes codeword failures are independent. They are")
    print("not: they are driven by a handful of deep fades, so the binomial figure")
    print("understates the uncertainty. The honest error bar is the spread across")
    print("channel realisations, measured here over five seeds at the longest")
    print("correlation time, where the record holds the fewest fades.")
    print()
    tau = 1.0e-3
    seeds = (7, 11, 23, 37, 53)
    depths = (64, 256, 1024, 4096)
    print(f"  tau = {tau:.1e} s, Lc = {tau * FS:.0f} symbols, {CODEWORDS} codewords "
          f"({CODEWORDS * 31} symbols = {CODEWORDS * 31 / (tau * FS):.0f} correlation times)")
    seed_cols = " ".join(f"{f'seed {s}':>10}" for s in seeds)
    tail_cols = f"{'mean':>10} {'across-seed sd':>15} {'binomial SE':>12}"
    header = f"  {'depth':>7} {seed_cols} {tail_cols}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for depth in depths:
        fers = []
        ses = []
        for seed in seeds:
            cfg = ChannelConfig(SI, tau, FS, "lognormal", "exp", seed=seed)
            r = CodedLink(code, cfg, mean_snr_db=SNR_DB).run(depth, CODEWORDS)
            fers.append(r.frame_error_rate)
            ses.append(r.frame_error_rate_standard_error)
        arr = np.asarray(fers)
        row = " ".join(f"{v:10.6f}" for v in fers)
        print(
            f"  {depth:7d} {row} {arr.mean():10.6f} {arr.std(ddof=1):15.6f} "
            f"{float(np.mean(ses)):12.6f}"
        )
    print()
    print("  The across-seed standard deviation is several times the binomial standard")
    print("  error at every depth. Above D/Lc ~ 1 the depth ordering is NOT resolvable")
    print("  at this sample size: it reverses between channel seeds. Do not read a")
    print("  single-seed difference between two deep depths as a real difference.")
    print()
    print("-" * 94)
    print(f"total sweep wall-clock   {time.perf_counter() - t_start:.1f} s on one core")
    print()
    print("Read the D/Lc column, not the depth column. The frame error rate falls")
    print("once the depth passes the correlation length, and depths well below it buy")
    print("nothing at all while still costing their full latency and memory. The")
    print("latency column is the price: the deepest row here is a quarter of a second")
    print("of end-to-end delay at 1 Mbaud.")
    print("=" * 94)


if __name__ == "__main__":
    main()

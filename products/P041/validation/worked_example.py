"""The README's worked example, run so its printed output cannot drift from the code.

Run: ``PYTHONPATH=../src python3 worked_example.py``
Runtime: about 25 s on one core.
"""

from __future__ import annotations

import numpy as np

from codedfade import (
    BlockInterleaver,
    ChannelConfig,
    CodedLink,
    ModemConfig,
    ModemSession,
    ReedSolomon,
    RunMode,
    SimulatedModemBackend,
    fade_statistics,
    generate_amplitude,
    lognormal_standard_level,
    markov_mean_fade_duration,
    required_interleaver_depth,
    uncoded_bit_error_rate,
)

# 1. A channel whose fades last a definite length of time.
channel = ChannelConfig(
    scintillation_index=0.6,
    correlation_time_s=2.0e-4,
    sample_rate_hz=1.0e6,
    marginal="lognormal",
    kernel="exp",
    seed=41,
)
amplitude = generate_amplitude(channel, 2_000_000)
stats = fade_statistics(amplitude, threshold=0.6, sample_rate_hz=channel.sample_rate_hz)
print(f"Lc = tau * Rs            {channel.samples_per_correlation_time:.0f} symbols")
print(f"sample MFD               {stats.mean_fade_duration_s * 1e6:.2f} us "
      f"= {stats.mean_fade_duration_s * channel.sample_rate_hz:.2f} symbols")
print(f"sample LCR               {stats.level_crossing_rate_hz:.1f} s^-1")
print(f"worst fade in the record {stats.max_fade_duration_s * 1e6:.1f} us")

# 2. The analytic result, exact for the sampled Gauss-Markov path.
u = lognormal_standard_level(0.6, channel.scintillation_index)
mfd = markov_mean_fade_duration(u, channel.correlation_time_s, channel.sample_rate_hz)
print(f"analytic MFD, eq (15)    {mfd * 1e6:.2f} us "
      f"(sample / analytic - 1 = {stats.mean_fade_duration_s / mfd - 1:+.4f})")

# 3. What depth does that imply, and what does it cost?
code = ReedSolomon(31, 21, 5)
for margin, label in ((1.0, "mean fade"), (3.0, "3x mean fade")):
    depth = required_interleaver_depth(mfd, channel.sample_rate_hz, margin)
    cost = BlockInterleaver(depth, code.n).cost(
        channel.sample_rate_hz, bits_per_symbol=code.m
    )
    print(f"depth for {label:<13} {depth:5d} symbols -> {cost.latency_ms:8.3f} ms, "
          f"{cost.memory_bytes:8.0f} B")

# 4. Does it actually work? Same channel record at every depth.
link = CodedLink(code, channel, mean_snr_db=14.0)
print(f"uncoded BER              {uncoded_bit_error_rate(channel, 14.0, 400_000):.4e}")
for result in link.depth_sweep([1, 64, 1024], 1024):
    print(f"depth {result.depth:5d}  D/Lc {result.depth / 200:6.2f}  "
          f"FER {result.frame_error_rate:.6f} +- "
          f"{result.frame_error_rate_standard_error:.6f}  "
          f"post BER {result.post_bit_error_rate:.3e}  "
          f"latency {result.latency_ms:7.3f} ms")

# 5. The same loop behind the hardware abstraction layer, in dry-run mode.
session = ModemSession(
    SimulatedModemBackend(),
    ModemConfig(n=31, k=21, m=5, depth=256, symbol_rate_hz=1.0e6),
    RunMode.DRY_RUN,
    channel,
)
report = session.start()
print(f"preflight                {'PASS' if report.passed else 'FAIL'} "
      f"({len(report.results)} checks)")
print(f"dry-run payload          {session.transfer(1, seed=0)!r}")
capture = session.finish()
print(f"capture status           {capture.status}")
print(f"captured counters        {capture.counters}")
print(f"encode / decode stage s  "
      f"{capture.stage_durations_s['encode']:.4f} / "
      f"{capture.stage_durations_s['decode']:.4f}")

# 6. And the backout path, because a half-finished run needs one.
session2 = ModemSession(
    SimulatedModemBackend(),
    ModemConfig(n=31, k=21, m=5, depth=256, symbol_rate_hz=1.0e6),
    RunMode.SIMULATION,
    channel,
)
session2.start()
aborted = session2.backout("receiver lost lock mid block")
print(f"backout status           {aborted.status}: {aborted.abort_reason}")
print(f"backend released         {not session2.backend.is_open()}")
assert np.isfinite(stats.mean_fade_duration_s)

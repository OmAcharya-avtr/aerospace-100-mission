"""The worked example printed in README.md, run so its output cannot drift."""

from __future__ import annotations

import numpy as np

from softdecode import (
    DetectionModel,
    LognormalFading,
    MultiplicativeCsiError,
    amplitude_quadrature,
    clip_llr,
    csi_aware_llr_ook,
    decode_ber,
    demap_ook,
    generalised_mutual_information,
    llr_error,
    llr_ook_known_csi,
    llr_ook_marginal,
    llr_ook_maxlog,
    make_regular_ldpc,
    simulate_ook,
)

det = DetectionModel(sigma=1.0)
fading = LognormalFading(sigma_i2=0.3)
code = make_regular_ldpc()

# 1. The exact LLR with the channel state known is the textbook linear one.
a = det.ook_amplitude(ebn0_db=10.0)
y = np.array([0.0, a / 2, a])
print(f"a = {a:.6f}, decision threshold a h / 2 = {a / 2:.6f}")
print(f"  known-CSI LLR at y = 0, a/2, a: "
      f"{np.array2string(llr_ook_known_csi(y, a, 1.0), precision=4)}")

# 2. With the channel state unknown the LLR needs a fading average, and
#    max-log is not free.
hq, wq = amplitude_quadrature(fading)
exact = llr_ook_marginal(y, a, hq, wq)
maxlog = llr_ook_maxlog(y, a, hq, wq)
print(f"  marginal exact: {np.array2string(exact, precision=4)}")
print(f"  max-log:        {np.array2string(maxlog, precision=4)}")
print(f"  max-log >= exact everywhere: {bool(np.all(maxlog >= exact))}")

# 3. A channel estimate 2 dB too high. Same bits, same fading, same noise.
error = MultiplicativeCsiError(bias_db=+2.0, jitter_db=1.0)
r = simulate_ook(code, ebn0_db=8.0, fading=fading, error=error, blocks=1200, seed=20261006)
truth = demap_ook(r, "known_csi", fading, error)
plugin = demap_ook(r, "plugin", fading, error)
aware = demap_ook(r, "csi_aware_exact", fading, error)
print(f"\n  LDPC n={code.length} k={code.dimension} rate={code.rate:.4f}, "
      f"{r.blocks} blocks at Eb/N0 = 8 dB")
for name, llr in (("true CSI", truth), ("plug-in h_hat", plugin), ("CSI-aware", aware)):
    print(f"  {name:<14} BER {decode_ber(code, r, llr)}  "
          f"GMI {generalised_mutual_information(llr, r.codeword):+.4f}")
print(f"  plug-in LLR error against the true-CSI LLR: {llr_error(plugin, truth)}")

# 4. Scalar rescaling cannot fix it, because the threshold is wrong too.
for alpha in (0.1, 0.3, 1.0):
    scaled = plugin * alpha
    print(f"  plug-in x {alpha:<4} BER {decode_ber(code, r, scaled).rate:.6e}")

# 5. Clipping costs much less in decoded BER than it does in LLR error.
exact_llr = demap_ook(r, "no_csi_exact", fading)
base = decode_ber(code, r, exact_llr).rate
for limit in (2.0, 5.0, 20.0):
    c = clip_llr(exact_llr, limit)
    print(f"  no-CSI exact clipped at {limit:>4}: relative LLR rmse "
          f"{llr_error(c, exact_llr).relative_rmse:.4f}, "
          f"BER {decode_ber(code, r, c).rate:.6e} against {base:.6e} unclipped")

# 6. The CSI-aware LLR for one sample, straight from the posterior.
single = csi_aware_llr_ook(
    np.array([3.0]), a, np.array([1.4]), fading, error, nodes=32
)
print(f"\n  CSI-aware LLR at y = 3.0 with h_hat = 1.4: {float(single[0]):.6f}")
print(f"  plug-in LLR for the same sample:            "
      f"{float(llr_ook_known_csi(np.array([3.0]), a, 1.4)[0]):.6f}")

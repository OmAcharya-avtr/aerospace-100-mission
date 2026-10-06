"""The mismatched-CSI penalty is not symmetric.

Writes ``../screenshots/mismatch_asymmetry.png``.

Left: decoded bit error rate against the channel-estimate bias, with the
true-CSI and CSI-aware references. Notice the curve is not a parabola about
zero: the same magnitude of error costs several times more when the estimate is
too high. Middle: the generalised mutual information of the plug-in LLRs,
which goes **negative** for a bias above about 2 dB -- the demapper is then
confidently wrong and is actively misleading the decoder. Right: the share of
raw hard-decision errors that are missed ones, which is the mechanism: a high
estimate raises the decision threshold ``a h_hat / 2``.

Runtime: about 60 s on one shared core.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from softdecode.channel import LognormalFading  # noqa: E402
from softdecode.csi import MultiplicativeCsiError  # noqa: E402
from softdecode.ldpc import make_regular_ldpc  # noqa: E402
from softdecode.metrics import generalised_mutual_information  # noqa: E402
from softdecode.simulate import decode_ber, demap_ook, simulate_ook  # noqa: E402

BLOCKS = 900
SEED = 20261006
EBN0_DB = 8.0
BIAS_DB = np.array([-4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0])

code = make_regular_ldpc()
fading = LognormalFading(0.3)

plugin_ber, plugin_gmi, aware_ber, truth_ber, missed_share = [], [], [], [], []
for bias in BIAS_DB:
    error = MultiplicativeCsiError(float(bias), 0.0)
    r = simulate_ook(code, EBN0_DB, fading, error, BLOCKS, SEED)
    plugin = demap_ook(r, "plugin", fading, error)
    aware = demap_ook(r, "csi_aware_exact", fading, error)
    truth = demap_ook(r, "known_csi", fading, error)
    plugin_ber.append(decode_ber(code, r, plugin).rate)
    aware_ber.append(decode_ber(code, r, aware).rate)
    truth_ber.append(decode_ber(code, r, truth).rate)
    plugin_gmi.append(generalised_mutual_information(plugin, r.codeword))
    decision = (plugin < 0).astype(np.int8)
    ones = r.codeword == 1
    miss = float(np.mean(decision[ones] != 1))
    false_alarm = float(np.mean(decision[~ones] != 0))
    missed_share.append(miss / (miss + false_alarm))
    print(f"  bias {bias:+5.1f} dB  plug-in BER {plugin_ber[-1]:.6e}  "
          f"GMI {plugin_gmi[-1]:+.5f}  CSI-aware BER {aware_ber[-1]:.6e}  "
          f"missed-one share {missed_share[-1]:.4f}")

fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))

axes[0].semilogy(BIAS_DB, plugin_ber, "o-", color="tab:red", label="plug-in $\\hat h$")
axes[0].semilogy(BIAS_DB, aware_ber, "s-", color="tab:blue",
                 label="CSI-aware exact (bias known)")
axes[0].semilogy(BIAS_DB, truth_ber, "k--", label="true CSI (upper bound)")
axes[0].axvline(0.0, color="grey", lw=0.8)
axes[0].set_xlabel("channel-estimate bias (dB of irradiance)")
axes[0].set_ylabel("decoded bit error rate")
axes[0].set_title(f"Decoded BER, $E_b/N_0$ = {EBN0_DB:.0f} dB, {BLOCKS} blocks")
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.3, which="both")

axes[1].plot(BIAS_DB, plugin_gmi, "o-", color="tab:red")
axes[1].axhline(0.0, color="k", lw=1.0)
axes[1].fill_between(BIAS_DB, -8, 0, color="tab:red", alpha=0.08)
axes[1].annotate("confidently wrong:\nGMI below zero", (0.3, -3.0), fontsize=8,
                 color="tab:red")
axes[1].axvline(0.0, color="grey", lw=0.8)
axes[1].set_xlabel("channel-estimate bias (dB of irradiance)")
axes[1].set_ylabel("GMI of the plug-in LLRs (bit/channel bit)")
axes[1].set_title("Calibration, not just accuracy")
axes[1].grid(alpha=0.3)

axes[2].plot(BIAS_DB, missed_share, "o-", color="tab:purple")
axes[2].axhline(0.5, color="grey", lw=0.8, ls=":")
axes[2].set_ylim(0, 1)
axes[2].axvline(0.0, color="grey", lw=0.8)
axes[2].set_xlabel("channel-estimate bias (dB of irradiance)")
axes[2].set_ylabel("share of raw hard errors that are missed ones")
axes[2].set_title("Mechanism: the threshold $a\\hat h/2$ moves")
axes[2].grid(alpha=0.3)

fig.tight_layout()
fig.savefig("../screenshots/mismatch_asymmetry.png", dpi=130)
print("wrote ../screenshots/mismatch_asymmetry.png")
idx_plus = int(np.argmin(np.abs(BIAS_DB - 2.0)))
idx_minus = int(np.argmin(np.abs(BIAS_DB + 2.0)))
print(f"  asymmetry at 2 dB: over {plugin_ber[idx_plus]:.6e}, "
      f"under {plugin_ber[idx_minus]:.6e}, ratio "
      f"{plugin_ber[idx_plus] / plugin_ber[idx_minus]:.3f}")

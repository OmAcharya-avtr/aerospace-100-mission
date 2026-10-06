"""LLR clipping costs much less than the LLR error suggests.

Writes ``../screenshots/clipping_cost.png``.

Left: the relative RMS LLR error introduced by saturating at ``L_max``, and
the fraction of LLRs that saturate. Right: the decoded bit error rate of the
same clipped LLRs. The two curves do not line up, and that gap is the useful
result: at Eb/N0 = 8 dB a clip level of 5 leaves an 80 per cent relative LLR
error and a decoded BER indistinguishable from the unclipped one, so a
fixed-point decoder can be far coarser than an LLR-error budget would allow.

Runtime: about 50 s on one shared core.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from softdecode.channel import LognormalFading  # noqa: E402
from softdecode.ldpc import make_regular_ldpc  # noqa: E402
from softdecode.llr import clip_llr  # noqa: E402
from softdecode.metrics import llr_error  # noqa: E402
from softdecode.simulate import decode_ber, demap_ook, simulate_ook  # noqa: E402

BLOCKS = 1200
SEED = 20261006
LIMITS = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 8.0, 12.0, 20.0, 40.0])
POINTS = ((4.0, "tab:blue"), (8.0, "tab:red"))

code = make_regular_ldpc()
fading = LognormalFading(0.3)

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
for ebn0, colour in POINTS:
    r = simulate_ook(code, ebn0, fading, None, BLOCKS, SEED + int(10 * ebn0))
    exact = demap_ook(r, "no_csi_exact", fading)
    base = decode_ber(code, r, exact)
    rel, frac, bers, ses = [], [], [], []
    for limit in LIMITS:
        clipped = clip_llr(exact, float(limit))
        rel.append(llr_error(clipped, exact).relative_rmse)
        frac.append(float(np.mean(np.abs(exact) > limit)))
        result = decode_ber(code, r, clipped)
        bers.append(result.rate)
        ses.append(result.standard_error)
        print(f"  Eb/N0 {ebn0:.0f} dB  L_max {limit:6.2f}  relative LLR rmse {rel[-1]:.4f}  "
              f"saturated {frac[-1]:.4f}  BER {bers[-1]:.6e}")
    axes[0].semilogx(LIMITS, rel, "o-", color=colour, label=f"{ebn0:.0f} dB, relative LLR rmse")
    axes[0].semilogx(LIMITS, frac, "s--", color=colour, alpha=0.6,
                     label=f"{ebn0:.0f} dB, saturated fraction")
    axes[1].errorbar(LIMITS, bers, yerr=np.array(ses) * 3.0, fmt="o-", color=colour,
                     capsize=3, label=f"{ebn0:.0f} dB, clipped")
    axes[1].axhline(base.rate, color=colour, ls="--", lw=1.0,
                    label=f"{ebn0:.0f} dB, unclipped {base.rate:.3e}")

axes[0].set_xscale("log")
axes[0].set_xlabel("clip level $L_\\mathrm{max}$")
axes[0].set_ylabel("dimensionless")
axes[0].set_title("What clipping does to the LLRs")
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.3, which="both")

axes[1].set_xscale("log")
axes[1].set_yscale("log")
axes[1].set_xlabel("clip level $L_\\mathrm{max}$")
axes[1].set_ylabel("decoded bit error rate")
axes[1].set_title("What clipping does to the decoder (3 binomial se bars)")
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3, which="both")

fig.tight_layout()
fig.savefig("../screenshots/clipping_cost.png", dpi=130)
print("wrote ../screenshots/clipping_cost.png")

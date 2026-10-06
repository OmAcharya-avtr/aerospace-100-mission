"""Validation 4: the mismatched-channel-state penalty, in both directions.

This is the result the product exists for.

What is measured
----------------
The receiver computes LLRs from an estimate ``h_hat`` that is wrong by a
stated amount, and nothing else changes: the transmitted bits, the fading, and
the noise are identical across the whole bias sweep at a given Eb/N0, because
``simulate_ook`` draws them in a fixed order from the same seed and the bias is
applied afterwards. The penalty is therefore a property of the demapper, not
of a different channel.

Three mismatch models
---------------------
* **pure bias**: ``h_hat = h exp(b)``, ``b = bias_db ln10/10``, no jitter. The
  deterministic case, where the mechanism is visible in closed form.
* **pure jitter**: ``h_hat = h exp(sigma_e z)``, unbiased in log, which is
  what a noisy pilot estimate gives.
* **stale estimate**: ``log h_hat`` jointly Gaussian with ``log h`` at
  correlation ``rho``, which is the estimate of a stated age on a correlated
  lognormal channel.

The asymmetry and its mechanism
-------------------------------
The plug-in LLR is ``(a**2 h_hat**2 - 2 a h_hat y) / (2 sigma**2)``, so its
sign changes at ``y = a h_hat / 2`` while the correct threshold is
``y = a h / 2``. Over-estimating ``h`` therefore raises the decision threshold
and systematically **misses ones**, which is a hard-decision error that no
rescaling of the LLRs can undo; under-estimating lowers the threshold toward
zero, where the noise-only hypothesis is still well separated. On top of that,
``h_hat > h`` scales the LLR magnitude up, so the errors it does make are
made confidently. Both effects push the same way, which is why the penalty is
not symmetric. The tables below separate the threshold effect (the raw
hard-decision bit error rate of the demapper, which is scale invariant) from
the total effect (the decoded bit error rate).

Runtime: about 100 s on one shared core.
"""

from __future__ import annotations

import numpy as np

from softdecode.channel import LognormalFading
from softdecode.csi import MultiplicativeCsiError, StaleCsiError
from softdecode.ldpc import make_regular_ldpc
from softdecode.metrics import bit_error_rate, generalised_mutual_information, llr_error
from softdecode.simulate import decode_ber, demap_ook, simulate_ook

BLOCKS = 1200
SEED = 20261006
EBN0_DB = 8.0
BIAS_DB = (-4.0, -3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0)
JITTER_DB = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
RHO = (1.0, 0.99, 0.95, 0.9, 0.8, 0.6)


def banner(text: str) -> None:
    print()
    print(text)
    print("-" * len(text))


def raw_hard_ber(realisation, llr) -> float:
    """Hard-decision bit error rate of the demapper before any decoding."""
    return bit_error_rate((llr < 0).astype(np.int8), realisation.codeword).rate


def bias_sweep(code, fading) -> dict[float, float]:
    banner("1. Pure bias, no jitter: the asymmetry")
    print(f"  Eb/N0 = {EBN0_DB:.2f} dB per information bit, {BLOCKS} blocks, "
          f"seed {SEED}")
    print("  reference rows: known_csi uses the true h, csi_aware_exact averages")
    print("  over p(h | h_hat) with the bias known to the receiver.")
    print(f"  {'bias dB':>8} {'raw hard BER':>13} {'plug-in BER':>13} {'plug-in GMI':>12} "
          f"{'LLR rmse':>10} {'BER / unbiased':>15} {'known-CSI BER':>14} "
          f"{'csi-aware BER':>14}")
    out: dict[float, float] = {}
    raw_by_bias: dict[float, float] = {}
    unbiased = None
    rows = []
    for bias in BIAS_DB:
        error = MultiplicativeCsiError(bias, 0.0)
        r = simulate_ook(code, EBN0_DB, fading, error, BLOCKS, SEED)
        plugin = demap_ook(r, "plugin", fading, error)
        truth = demap_ook(r, "known_csi", fading, error)
        aware = demap_ook(r, "csi_aware_exact", fading, error)
        ber = decode_ber(code, r, plugin).rate
        rows.append(
            (
                bias,
                raw_hard_ber(r, plugin),
                ber,
                generalised_mutual_information(plugin, r.codeword),
                llr_error(plugin, truth).rmse,
                decode_ber(code, r, truth).rate,
                decode_ber(code, r, aware).rate,
            )
        )
        out[bias] = ber
        raw_by_bias[bias] = raw_hard_ber(r, plugin)
        if bias == 0.0:
            unbiased = ber
    for bias, raw, ber, gmi, rmse, known, aware in rows:
        ratio = ber / unbiased if unbiased else float("nan")
        print(f"  {bias:8.2f} {raw:13.6e} {ber:13.6e} {gmi:+12.5f} {rmse:10.4f} "
              f"{ratio:15.3f} {known:14.6e} {aware:14.6e}")
    print()
    print("  Decomposition. The raw hard-decision BER is scale invariant, so its")
    print("  over/under ratio isolates the decision-threshold shift; the decoded")
    print("  BER ratio contains that plus the over-confidence of the LLRs, and the")
    print("  quotient of the two is what over-confidence alone costs.")
    print(f"  {'|bias| dB':>10} {'over BER':>13} {'under BER':>13} {'decoded ratio':>14} "
          f"{'raw ratio':>10} {'confidence share':>17}")
    for magnitude in (1.0, 2.0, 3.0, 4.0):
        over, under = out[magnitude], out[-magnitude]
        raw_ratio = raw_by_bias[magnitude] / raw_by_bias[-magnitude]
        decoded_ratio = over / under if under > 0 else float("inf")
        print(f"  {magnitude:10.1f} {over:13.6e} {under:13.6e} {decoded_ratio:14.3f} "
              f"{raw_ratio:10.3f} {decoded_ratio / raw_ratio:17.3f}")
    return out


def jitter_sweep(code, fading) -> None:
    banner("2. Pure jitter, no bias")
    print("  A zero-mean log error still costs, because the LLR is nonlinear in h.")
    print(f"  {'jitter dB':>10} {'raw hard BER':>13} {'plug-in BER':>13} "
          f"{'plug-in GMI':>12} {'csi-aware BER':>14} {'csi-aware GMI':>14}")
    for jitter in JITTER_DB:
        error = MultiplicativeCsiError(0.0, jitter)
        r = simulate_ook(code, EBN0_DB, fading, error, BLOCKS, SEED)
        plugin = demap_ook(r, "plugin", fading, error)
        if jitter == 0.0:
            aware_ber = decode_ber(code, r, plugin).rate
            aware_gmi = generalised_mutual_information(plugin, r.codeword)
        else:
            aware = demap_ook(r, "csi_aware_exact", fading, error)
            aware_ber = decode_ber(code, r, aware).rate
            aware_gmi = generalised_mutual_information(aware, r.codeword)
        print(f"  {jitter:10.2f} {raw_hard_ber(r, plugin):13.6e} "
              f"{decode_ber(code, r, plugin).rate:13.6e} "
              f"{generalised_mutual_information(plugin, r.codeword):+12.5f} "
              f"{aware_ber:14.6e} {aware_gmi:+14.5f}")


def combined_sweep(code, fading) -> None:
    banner("3. Bias and jitter together, over- against under-estimating")
    print(f"  {'bias dB':>8} {'jitter dB':>10} {'plug-in BER':>13} {'plug-in GMI':>12} "
          f"{'csi-aware BER':>14}")
    for bias in (-2.0, 0.0, 2.0):
        for jitter in (0.5, 1.0, 2.0):
            error = MultiplicativeCsiError(bias, jitter)
            r = simulate_ook(code, EBN0_DB, fading, error, BLOCKS, SEED)
            plugin = demap_ook(r, "plugin", fading, error)
            aware = demap_ook(r, "csi_aware_exact", fading, error)
            print(f"  {bias:8.2f} {jitter:10.2f} {decode_ber(code, r, plugin).rate:13.6e} "
                  f"{generalised_mutual_information(plugin, r.codeword):+12.5f} "
                  f"{decode_ber(code, r, aware).rate:14.6e}")


def stale_sweep(code, fading) -> None:
    banner("4. A stale estimate of stated correlation")
    print("  rho = 1 is perfect CSI. The stale estimate is unbiased in log h but")
    print("  its spread makes it over-estimate as often as it under-estimates, and")
    print("  the over-estimates dominate the cost.")
    print(f"  {'rho':>6} {'raw hard BER':>13} {'plug-in BER':>13} {'plug-in GMI':>12} "
          f"{'csi-aware BER':>14} {'csi-aware GMI':>14}")
    for rho in RHO:
        error = StaleCsiError(rho)
        r = simulate_ook(code, EBN0_DB, fading, error, BLOCKS, SEED)
        plugin = demap_ook(r, "plugin", fading, error)
        aware = demap_ook(r, "csi_aware_exact", fading, error)
        print(f"  {rho:6.2f} {raw_hard_ber(r, plugin):13.6e} "
              f"{decode_ber(code, r, plugin).rate:13.6e} "
              f"{generalised_mutual_information(plugin, r.codeword):+12.5f} "
              f"{decode_ber(code, r, aware).rate:14.6e} "
              f"{generalised_mutual_information(aware, r.codeword):+14.5f}")


def threshold_mechanism(code, fading) -> None:
    banner("5. The mechanism: missed ones against false alarms on zeros")
    print("  Split the raw hard-decision errors of the plug-in demapper by the")
    print("  transmitted bit. Over-estimating h raises the threshold a h_hat / 2")
    print("  and the errors become almost entirely missed ones.")
    print(f"  {'bias dB':>8} {'P(error | bit=1)':>17} {'P(error | bit=0)':>17} "
          f"{'share missed ones':>18}")
    for bias in (-4.0, -2.0, 0.0, 2.0, 4.0):
        error = MultiplicativeCsiError(bias, 0.0)
        r = simulate_ook(code, EBN0_DB, fading, error, BLOCKS, SEED)
        plugin = demap_ook(r, "plugin", fading, error)
        decision = (plugin < 0).astype(np.int8)
        ones = r.codeword == 1
        miss = float(np.mean(decision[ones] != 1))
        false_alarm = float(np.mean(decision[~ones] != 0))
        total = miss + false_alarm
        print(f"  {bias:8.2f} {miss:17.6e} {false_alarm:17.6e} "
              f"{(miss / total if total > 0 else float('nan')):18.4f}")


def main() -> int:
    print("softdecode validation 4: the mismatched-CSI penalty")
    print("raw output committed as validation/csi_mismatch_output.txt")
    code = make_regular_ldpc()
    fading = LognormalFading(0.3)
    print(f"LDPC n={code.length} k={code.dimension} rate={code.rate:.6f}; "
          f"lognormal sigma_I2={fading.scintillation_index}")
    bias_sweep(code, fading)
    jitter_sweep(code, fading)
    combined_sweep(code, fading)
    stale_sweep(code, fading)
    threshold_mechanism(code, fading)
    print("\ndone")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

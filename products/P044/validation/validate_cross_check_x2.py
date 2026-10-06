"""Cross-check X2 (binding): BPSK BER over lognormal fading, single aperture.

Batch 05 specification, cross-check X2: P044 ApertureDiv and P010 BERBench
must agree on the sample bit error rate of coherent BPSK over lognormal
fading at a stated Eb/N0 and scintillation index, within 3 binomial standard
errors, with the standard error reported.

This script holds the whole configuration as module constants so that it is
frozen and quotable. **P010's value is not known to this product and is not
read, imported or approximated anywhere.** The coordinating session holds it
and performs the comparison. If the two disagree, the disagreement is
reported, not tuned away, so the configuration below is written out in full
to make a disagreement diagnosable.

The compared quantity
---------------------
The **sample** bit error rate: the observed fraction of bit errors over the
stated number of channel realisations. It is a sample statistic, not an
analytic expression. Bits are decided one at a time from a drawn receiver
noise sample. One bit per channel realisation, so the bits are independent
and the binomial standard error ``sqrt(p(1-p)/N)`` is exact rather than an
approximation that ignores within-realisation clustering.

No wall-clock measurement appears anywhere in this product, so X2 cannot
repeat the Batch 04 defect of comparing a timing against a model.

The conventions that a disagreement would most likely trace to
--------------------------------------------------------------
Stated explicitly because each one is a factor-of-two-or-more difference:

C1  The **fading power gain** is the lognormal irradiance ``I``, normalised
    to ``E[I] = 1``. The amplitude gain is ``sqrt(I)``. The instantaneous
    SNR is ``gamma = (Eb/N0) I``, linear in irradiance.
    *Not* used: ``gamma proportional to I^2``, which is the
    thermal-noise-limited intensity-modulation convention.
C2  The **scintillation index** is defined on the irradiance,
    ``si = var(I)/E[I]^2``, so ``var(ln I) = ln(1 + si)`` and
    ``E[ln I] = -ln(1 + si)/2``.
    *Not* used: ``si`` as the variance of ``ln I`` directly, nor as the
    log-amplitude variance ``sigma_chi^2 = si/4``.
C3  ``Eb/N0`` is referred to the **mean** received irradiance, i.e. to
    ``E[I] = 1``, not to the median irradiance ``exp(-s^2/2)``. The two
    differ by ``s^2/2`` nepers, which at ``si = 0.3`` is 0.57 dB.
C4  The receiver is **coherent BPSK with an ideal phase reference** and a
    conditional error probability ``Q(sqrt(2 gamma))``. *Not* used:
    differential or non-coherent detection, which would give
    ``0.5 exp(-gamma)``.
C5  The channel is **memoryless across bits**: one independent irradiance
    draw per bit. *Not* used: a block-fading model with several bits per
    draw, which leaves the BER unchanged in expectation but inflates the
    true standard error above the binomial figure.

If P010 disagrees, C1 and C2 are the first two to compare; a disagreement
traceable to one of these is a specification defect, not a product defect in
either repository, and should be recorded as an amendment.

Runtime about 8 s on one core.
"""

from __future__ import annotations

import pathlib
import sys

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402

from aperturediv.ber import (  # noqa: E402
    bpsk_ber_awgn,
    bpsk_ber_lognormal_gauss_hermite,
    bpsk_ber_lognormal_sample,
)
from aperturediv.channel import lognormal_sigma_log  # noqa: E402

# ---------------------------------------------------------------------------
# FROZEN X2 CONFIGURATION. Do not change these without recording an amendment.
# ---------------------------------------------------------------------------
X2_MODULATION = "BPSK, coherent, ideal phase reference, hard decision"
X2_APERTURES = 1
X2_EBN0_DB = 10.0
X2_SI = 0.3
X2_N_REALISATIONS = 20_000_000
X2_BITS_PER_REALISATION = 1
X2_SEED = 44044
X2_CHUNK = 2_000_000
X2_RNG = "numpy.random.default_rng(44044)  (PCG64, NumPy default bit generator)"

# Supplementary operating points, clearly secondary: they exist only to help
# diagnose a disagreement at the primary point by showing its slope.
X2_SUPPLEMENTARY = (
    (6.0, 0.3, 20_000_000, 44044),
    (14.0, 0.3, 20_000_000, 44044),
    (10.0, 0.1, 20_000_000, 44044),
    (10.0, 0.6, 20_000_000, 44044),
)


def _rule(title: str) -> None:
    print("=" * 78)
    print(title)
    print("=" * 78)


def main() -> int:
    failures: list[str] = []

    _rule("CROSS-CHECK X2 -- frozen configuration")
    s = lognormal_sigma_log(X2_SI)
    print(f"  modulation                   {X2_MODULATION}")
    print(f"  receive apertures            {X2_APERTURES} (single aperture, no diversity)")
    print(f"  Eb/N0                        {X2_EBN0_DB:.4f} dB "
          f"(= {10 ** (X2_EBN0_DB / 10.0):.6f} linear)")
    print("  Eb/N0 reference              mean received irradiance, E[I] = 1")
    print("  fading law                   gamma = (Eb/N0) * I, I lognormal, E[I] = 1")
    print(f"  scintillation index si       {X2_SI:.6f}  (= var(I)/E[I]^2)")
    print(f"  sigma_log = sqrt(ln(1+si))   {s:.12f}  (std of ln I, natural log)")
    print(f"  E[ln I]                      {-0.5 * s * s:.12f}")
    print(f"  channel realisations         {X2_N_REALISATIONS}")
    print(f"  bits per realisation         {X2_BITS_PER_REALISATION} "
          "(so bits are independent)")
    print(f"  total bits                   "
          f"{X2_N_REALISATIONS * X2_BITS_PER_REALISATION}")
    print(f"  RNG                          {X2_RNG}")
    print(f"  seed                         {X2_SEED}")
    print(f"  chunk size                   {X2_CHUNK} bits per chunk")
    print()
    print("  draw order inside each chunk of m bits, one stream, strictly this order:")
    print("    1. z     = rng.standard_normal(m)       -> I = exp(sigma_log*z - "
          "sigma_log^2/2)")
    print("    2. bits  = rng.integers(0, 2, m)        -> s_tx = 1 - 2*bits")
    print("    3. noise = rng.standard_normal(m)       -> unit variance")
    print("  received sample  r = sqrt(2*(Eb/N0)*I) * s_tx + noise")
    print("  decision         bit_hat = (r < 0)")
    print("  BER definition   observed bit errors / total bits  (a sample statistic)")
    print()

    _rule("CROSS-CHECK X2 -- result")
    res = bpsk_ber_lognormal_sample(
        X2_EBN0_DB, X2_SI, X2_N_REALISATIONS, X2_SEED, chunk_size=X2_CHUNK
    )
    print(f"  sample BER                   {res.ber:.9e}")
    print(f"  bit errors observed          {res.n_errors}")
    print(f"  total bits                   {res.n_bits}")
    print(f"  binomial standard error      {res.binomial_se:.9e}")
    print(f"  SE as a fraction of the BER  {res.relative_se * 100:.4f} %")
    print(f"  3 binomial standard errors   {3.0 * res.binomial_se:.9e}")
    print(f"  X2 agreement interval        [{res.ber - 3 * res.binomial_se:.9e}, "
          f"{res.ber + 3 * res.binomial_se:.9e}]")
    print()
    print("  THE NUMBER TO COMPARE WITH P010 BERBench IS THE SAMPLE BER ABOVE.")
    print("  P010's value is not known to this repository; the coordinating session")
    print("  holds it and performs the comparison.")
    print()

    _rule("Internal consistency of the X2 number (not the comparison)")
    gh = bpsk_ber_lognormal_gauss_hermite(X2_EBN0_DB, X2_SI, n_nodes=300)
    gh_150 = bpsk_ber_lognormal_gauss_hermite(X2_EBN0_DB, X2_SI, n_nodes=150)
    z = (res.ber - gh) / res.binomial_se
    print(f"  Gauss-Hermite quadrature of the same expectation, 300 nodes  {gh:.9e}")
    print(f"  the same with 150 nodes (quadrature convergence check)       {gh_150:.9e}")
    print(f"  relative difference between node counts                      "
          f"{abs(gh - gh_150) / gh:.3e}")
    print(f"  z = (sample BER - quadrature) / binomial SE                  {z:+.4f}")
    print(f"  AWGN BER at the same Eb/N0, no fading                        "
          f"{float(bpsk_ber_awgn(X2_EBN0_DB)):.9e}")
    print(f"  fading penalty: BER ratio to AWGN                            "
          f"{res.ber / float(bpsk_ber_awgn(X2_EBN0_DB)):.3f}x")
    print()
    print("  The quadrature is a diagnostic for the sample figure, not the compared")
    print("  quantity. |z| above 3 would mean the sampler and the model disagree,")
    print("  which would be an internal defect and would have to be fixed before the")
    print("  X2 number meant anything.")
    if not np.isfinite(z):
        failures.append(f"sample BER vs quadrature z is not finite: {z}")
    elif abs(z) > 3.0:
        failures.append(f"sample BER vs quadrature |z| = {abs(z):.3f} > 3")
    if abs(gh - gh_150) / gh > 1e-6:
        failures.append("Gauss-Hermite quadrature has not converged in node count")
    print()

    _rule("Reproducibility of the X2 number")
    again = bpsk_ber_lognormal_sample(
        X2_EBN0_DB, X2_SI, X2_N_REALISATIONS, X2_SEED, chunk_size=X2_CHUNK
    )
    print(f"  rerun with the same seed and chunk: BER {again.ber:.9e}, "
          f"errors {again.n_errors}")
    print(f"  bit-for-bit identical: {again.n_errors == res.n_errors}")
    if again.n_errors != res.n_errors:
        failures.append("X2 sample BER is not reproducible under the same seed")
    other = bpsk_ber_lognormal_sample(
        X2_EBN0_DB, X2_SI, X2_N_REALISATIONS, X2_SEED + 1, chunk_size=X2_CHUNK
    )
    diff_se = float(np.sqrt(res.binomial_se**2 + other.binomial_se**2))
    print(f"  a different seed ({X2_SEED + 1}): BER {other.ber:.9e}, "
          f"errors {other.n_errors}")
    print(f"  difference {res.ber - other.ber:+.3e}, "
          f"combined SE {diff_se:.3e}, z {(res.ber - other.ber) / diff_se:+.3f}")
    print("  two seeds agreeing within a couple of combined standard errors is")
    print("  evidence that the quoted SE describes the real sampling spread.")
    print()

    _rule("Supplementary operating points (secondary, for diagnosis only)")
    print("  Eb/N0 dB     si    realisations      sample BER   binomial SE     z vs GH")
    for ebn0, si, n_real, seed in X2_SUPPLEMENTARY:
        r = bpsk_ber_lognormal_sample(ebn0, si, n_real, seed, chunk_size=X2_CHUNK)
        g = bpsk_ber_lognormal_gauss_hermite(ebn0, si, n_nodes=300)
        zz = (r.ber - g) / r.binomial_se if r.binomial_se > 0 else float("nan")
        print(f"{ebn0:10.2f} {si:6.3f} {n_real:15d} {r.ber:15.6e} {r.binomial_se:13.6e} "
              f"{zz:+11.3f}")
        if not np.isfinite(zz):
            failures.append(f"supplementary point Eb/N0={ebn0} si={si}: z is not finite")
        elif abs(zz) > 4.0:
            failures.append(f"supplementary point Eb/N0={ebn0} si={si}: |z| = {abs(zz):.3f}")
    print("  These are not the X2 comparand. They exist so that, if the primary")
    print("  numbers disagree, the shape of the disagreement in Eb/N0 and in si can")
    print("  be read off and attributed to a convention rather than to noise.")
    print()

    _rule("Result")
    if failures:
        print(f"FAILED {len(failures)} check(s):")
        for f in failures:
            print("  -", f)
        return 1
    print("All checks passed.")
    print()
    print(f"X2 SAMPLE BER = {res.ber:.9e}  +/- {res.binomial_se:.9e} (1 binomial SE)")
    print(f"X2 CONFIG     = BPSK coherent, 1 aperture, Eb/N0 {X2_EBN0_DB} dB, "
          f"si {X2_SI}, {X2_N_REALISATIONS} realisations x "
          f"{X2_BITS_PER_REALISATION} bit, seed {X2_SEED}, chunk {X2_CHUNK}")
    print(f"X2 ERRORS     = {res.n_errors} of {res.n_bits} bits")
    return 0


if __name__ == "__main__":
    sys.exit(main())

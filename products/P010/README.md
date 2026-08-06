# BERBench

**Status:** TESTING · **Class:** compact · **Validation level:** 2 (Research) · **AI:** no

## Executive overview

BERBench computes analytic bit-error-ratio (BER) curves and runs vectorised
Monte Carlo benchmarks for three modulation formats — OOK, BPSK and M-ary
PPM — over AWGN and lognormal-fading (weak-turbulence free-space optical)
channels. Every analytic expression carries a literature citation and a
validity range; every Monte Carlo estimate carries a Wilson confidence
interval; agreement between the two is demonstrated with saved numbers in
`validation/`. Package `berbench`, version 0.1.0, pure Python
(numpy + scipy; matplotlib only for plots).

## Aerospace problem

Optical and RF link designers need trustworthy BER-vs-SNR curves to size
link margins: an FSO downlink through clear-air turbulence, an RF telemetry
link, a photon-starved deep-space PPM link. Mistakes here (using a union
bound as if exact, ignoring the no-CSI threshold penalty under
scintillation, quoting an MC point with 3 observed errors) translate
directly into dB-level margin errors. BERBench provides cited analytic
baselines, a statistically honest simulation harness, and the
fading-average machinery for the weak-turbulence regime.

## Intended users

Link-budget engineers, FSO/RF communication researchers, students verifying
textbook results, and other products in this portfolio needing a validated
BER kernel.

## Engineering theory

SNR convention: gamma = Eb/N0 (electrical, per bit, dimensionless; input in
dB, negative dB valid). Q(x) is the Gaussian tail probability, computed via
`scipy.special.ndtr(-x)`; a log-domain companion `log_qfunc` /
`log10_qfunc` (via `log_ndtr`) stays finite far beyond the x ≈ 38 underflow
point of Q itself.

| Expression | Formula | Source | Notes / validity |
|---|---|---|---|
| BPSK, AWGN | Pb = Q(sqrt(2·gamma)) | Proakis & Salehi, *Digital Communications*, 5th ed. 2008, Eq. (4.3-13) | coherent detection, exact |
| OOK, AWGN, optimal threshold | Pb = Q(sqrt(gamma)) | Proakis & Salehi 2008, Sec. 4.3 (binary ASK); Zhu & Kahn 2002, Eq. (12) | midpoint threshold is optimal for equal priors and signal-independent AWGN; exactly 3.01 dB worse than BPSK |
| OOK, AWGN, fixed threshold t | Pb = ½[Q(2t·sqrt(gamma)) + Q(2(1−t)·sqrt(gamma))] | derived from the same matched-filter model (A = 2·sqrt(gamma)·sigma_n, tau = t·A) | t ∈ (0,1); t = ½ reproduces the optimal case |
| M-PPM, AWGN, **exact** | Ps = 1 − ∫ φ(y − sqrt(2Es/N0)) Φ(y)^(M−1) dy; Pb = M/(2(M−1))·Ps | Proakis & Salehi 2008, Eqs. (4.4-17), (4.4-18) | M-ary **orthogonal signalling with coherent detection**; Es = log2(M)·Eb; evaluated by Gauss-Hermite in stable expm1/log form |
| M-PPM, AWGN, **union bound** | Ps ≤ (M−1)·Q(sqrt(Es/N0)) | Proakis & Salehi 2008, Sec. 4.4 (pairwise d² = 2Es) | **upper bound, not exact**; tight at high SNR (validated: ratio 1.0173 at 8 dB for M=16) |
| Lognormal fading | ln I ~ N(−σz²/2, σz²), σz² = ln(1+σI²), E[I] = 1; Pb = E_I[Pb_cond(gamma·I²)] via Gauss-Hermite | Andrews & Phillips, *Laser Beam Propagation through Random Media*, 2nd ed., SPIE 2005, Ch. 8–9; Zhu & Kahn, IEEE Trans. Commun. 50(8), 2002 | **weak-fluctuation validity: σI² < ~1**; beyond that a UserWarning is raised and results are extrapolation (gamma-gamma model would be required, out of scope) |
| OOK fixed threshold under fading | Pb(I) = ½[Q(2t·sqrt(gamma)) + Q(2(I−t)·sqrt(gamma))], averaged over I | same model, no channel-state information | exhibits the well-known irreducible BER floor = ½·P(I < t) as SNR → ∞ |
| SIM-BPSK under fading | Pb = E_I[Q(I·sqrt(2·gamma))] | Popoola & Ghassemlooy, J. Lightwave Technol. 27(8), 2009 (subcarrier intensity modulation) | amplitude gain = I |
| Wilson CI | score interval on error counts | Wilson, J. Amer. Statist. Assoc. 22, 1927 | chosen over Wald: BER proportions are near 0 |

Assumptions: additive signal-independent Gaussian noise (thermal /
background-limited receiver; **not** shot-noise-limited photon counting);
i.i.d. per-symbol fading (ergodic average — real turbulence has ~ms
coherence, which changes short-term statistics, not the mean BER); PPM
modelled as coherent orthogonal signalling.

## Architecture

```
src/berbench/
├── _math.py       Q-function (linear + log domain), Gauss-Hermite nodes, Wilson CI
├── channels.py    lognormal parameterisation, validity checks, fading sampler
├── analytic.py    analytic_ber(): all cited expressions + GH fading averaging
├── montecarlo.py  mc_ber(): vectorised batched simulation, CIs, runtime budget
├── results.py     AnalyticResult / MCResult frozen dataclasses
└── __main__.py    CLI (argparse): python -m berbench sweep ...
```

## Installation

```bash
pip install -e .            # from products/P010/ (numpy, scipy required)
# or without installing:
export PYTHONPATH=src
```

## Quick start

```python
import numpy as np
from berbench import analytic_ber, mc_ber, n_bits_for_target

snr = np.arange(0, 12, 2)                      # Eb/N0 in dB
ana = analytic_ber("bpsk", snr)                # AnalyticResult
mc = mc_ber("bpsk", snr, n=1_000_000, seed=1)  # MCResult with Wilson 95% CI
print(ana.ber, mc.ber, mc.ci_low, mc.ci_high)

# FSO OOK through weak turbulence
faded = analytic_ber("ook", 10.0, channel="lognormal", sigma_i2=0.3)
# no-CSI fixed threshold -> BER floor
floor = analytic_ber("ook", 30.0, channel="lognormal", sigma_i2=0.3, threshold=0.5)

# how many bits for ~100 errors at BER 1e-4?
n = n_bits_for_target(1e-4, min_errors=100)    # -> 1_000_000
```

CLI:

```bash
python -m berbench sweep --mod ook bpsk ppm --snr 0:20:2 --channel awgn
python -m berbench sweep --mod ook --snr 0:16:4 --channel lognormal \
    --sigma-i2 0.3 --mc --n 500000 --png sweep.png
```

## Configuration

All configuration is per-call (no config files): `channel` ("awgn" |
"lognormal"), `sigma_i2` (scintillation index, required for lognormal),
`M` (PPM alphabet, power of two, 2–4096), `threshold` ("optimal" or fixed
fraction in (0,1), OOK only), `ppm_method` ("exact" | "union"),
`n_gh_nodes` (2–256), `ci_level`, `max_seconds` (hard MC wall-clock budget),
`seed`.

Input policy: negative `snr_db` is valid (it is dB); NaN/inf raises
ValueError; invalid M / sigma_i2 / threshold raise ValueError or TypeError
with actionable messages; sigma_i2 > 1 emits a UserWarning (outside
weak-fluctuation validity) but still computes.

## Examples

Both scripts are runnable from `examples/` and write PNGs to `screenshots/`
(total runtime ~11 s):

- `examples/waterfall_awgn.py` → `screenshots/waterfall_awgn.png`: analytic
  vs MC waterfall for OOK, BPSK, 4-PPM, 16-PPM over AWGN, with 95% CI error
  bars and the 16-PPM union bound overlaid on the exact curve.
- `examples/ook_lognormal.py` → `screenshots/ook_lognormal.png`: OOK over
  lognormal fading at sigma_I² = 0.1 and 0.3 vs AWGN, plus the fixed
  (no-CSI) threshold curve flattening onto its BER floor (~6.8e-2 for
  sigma_I² = 0.3, t = 0.5).

## Validation

Level 2 evidence, all produced by rerunnable scripts in `validation/` with
raw outputs saved alongside (see `validation/VALIDATION.md` for the full
tables):

- **MC vs analytic sweep:** 49/54 seeded points (90.7%) have the analytic
  BER inside the MC 95% Wilson CI (expected ~95%; 2-sigma acceptance band
  48–54; independent-seed repeat gave 51/54). Misses are mixed-sign
  near-misses.
- **BPSK textbook check:** Pb(9.6 dB) = 9.736e-6 vs the classic ~1e-5
  benchmark (Proakis Fig. 4.3-1 / Sklar 2001 Sec. 4.7.1); our curve crosses
  1e-5 at 9.588 dB.
- **PPM exact vs bound:** union bound ≥ exact everywhere; ratio → 1.0000 by
  12 dB; exact expression matches an independent QUADPACK evaluation to
  3.1e-7 worst-case, ~1e-15 in the deep tail (BER 2.9e-21).
- **Fading average:** Gauss-Hermite matches scipy.quad to < 2e-10 (adaptive
  threshold); fixed-threshold case degrades to 4.7e-3 relative at 30 dB
  (documented).
- **CI coverage audit:** 200-replicate empirical coverage 0.925–0.950 for
  binary modulations; 0.90–0.93 for PPM (documented residual narrowing).
  This audit caught and fixed a real defect (bit-level CI for PPM had 76%
  coverage).

## Benchmark results

Measured on the 2-core build machine (numpy 2.4.4, single point at 6 dB):
BPSK 4.0e7 bits/s; OOK 3.3e7 bits/s (1.7e7 with lognormal fading); 4-PPM
1.7e7 bits/s; 16-PPM 1.5e7 bits/s (argmax over M branches). The entire
validation + example MC workload runs in ~30 s, far under the 3-minute
budget; `mc_ber(..., max_seconds=...)` enforces a hard cap and flags
`budget_exhausted` instead of overrunning.

## AI model details

Not applicable — this product contains no AI/ML component.

## Hardware requirements

Any CPU; 2 cores and ~300 MB RAM are ample (batches capped at 2^21 values).
No GPU, no network.

## Limitations

1. Noise model is additive signal-independent Gaussian only — no shot-noise
   (Poisson) statistics, no APD excess noise, no inter-symbol interference.
2. Lognormal fading is valid for weak fluctuations (sigma_I² < ~1); no
   gamma-gamma / strong-turbulence model. sigma_I² > 1 warns and
   extrapolates.
3. PPM is modelled as coherent M-ary orthogonal signalling; direct-detection
   photon-counting PPM obeys different statistics.
4. PPM confidence intervals are ~8% narrow in half-width (documented;
   empirical coverage ~0.90–0.93 vs nominal 0.95).
5. Fixed-threshold OOK lognormal analytic average: ~0.5% relative error at
   30 dB (Gauss-Hermite on a step-like integrand, node count capped at 256
   by numpy hermgauss stability).
6. Fading is i.i.d. per symbol (ergodic mean BER); no temporal correlation,
   no interleaver/burst analysis.

## Safety statement

This software is research-grade. It is not flight-qualified, not certified,
and not approved for operational aerospace use.

## Roadmap

- Gamma-gamma fading (moderate-to-strong turbulence) with validity handoff.
- Direct-detection PPM with Poisson statistics.
- Coded-BER hooks (RS/LDPC waterfall shift estimates).

## License

MIT — see [LICENSE](LICENSE). Copyright © 2026 OPTIMA Organisation.

## Credits

This is under reserved rights obtained by OPTIMA Organisation.

## Citation

```bibtex
@software{berbench2026,
  title   = {BERBench: BER computation and Monte Carlo benchmarking for
             OOK/BPSK/M-PPM over AWGN and lognormal fading},
  author  = {{OPTIMA Organisation}},
  year    = {2026},
  version = {0.1.0},
  license = {MIT}
}
```

Key references: Proakis & Salehi 2008 (*Digital Communications*, 5th ed.);
Andrews & Phillips 2005 (*Laser Beam Propagation through Random Media*,
2nd ed., SPIE); Zhu & Kahn 2002 (IEEE Trans. Commun. 50(8)); Popoola &
Ghassemlooy 2009 (J. Lightwave Technol. 27(8)); Wilson 1927 (JASA 22).

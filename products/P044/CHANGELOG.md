# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-06

First release. Validation level 2, research grade, status TESTING.

### Added

#### Channel statistics
- `aperturediv.channel`: unit-mean lognormal and gamma-gamma irradiance pdf,
  cdf, moments and samplers. The gamma-gamma density is evaluated in log space
  through `scipy.special.kve` so it does not underflow at large irradiance,
  and its cdf is taken by quadrature in `ln I` so the `I -> 0` endpoint
  singularity is removed.
- Plane-wave Rytov-variance to `(alpha, beta)` mapping, checked for internal
  consistency against the model's own closed-form scintillation index and
  against sampled moments rather than asserted.
- `lognormal_scintillation_index` / `lognormal_sigma_log` as an exact inverse
  pair, `si = exp(s^2) - 1`.

#### Aperture averaging
- `aperturediv.aperture`: the aperture-averaging factor `A(D)` computed from
  its defining integral over the circular-aperture overlap kernel, with the
  normalisation identity `int_0^1 u W(u) du = pi/16` verified numerically.
- Two derived closed-form limits of the Gaussian-covariance case,
  `A ~ 1 - D^2/(4 rho_c^2)` and
  `A ~ 4(rho_c/D)^2 - (8/sqrt pi)(rho_c/D)^3`, used as validation targets.
- Gaussian and exponential covariance shapes; `fresnel_scale`;
  `equal_area_diameter` for the fixed-total-glass comparison.

#### Inter-aperture correlation
- `aperturediv.correlation`: correlation specified on the log-irradiance
  field, with the exact lognormal conversion
  `corr(I_j,I_k) = (exp(R_jk s^2) - 1)/(exp(s^2) - 1)`.
- Gaussian-copula samplers for correlated lognormal and for correlated
  gamma-gamma, the latter with separate correlation matrices for the
  large-scale and small-scale factors.
- `nearest_psd` for correlation matrices that are PSD only to rounding.

#### Combining
- `aperturediv.combining`: maximal-ratio, equal-gain and selection combining;
  outage probability evaluated from one seeded gain sample set across the
  whole mean-SNR grid; finite-window diversity order with the window,
  point count and fit residual returned alongside the slope.
- Two power-normalisation conventions, `fixed_total` and `fixed_branch`,
  both named rather than implied.

#### Channel-state estimation and the learned combiner
- `aperturediv.estimation`: stale-and-noisy channel-state estimate model,
  collapsed onto one log-domain error level
  `sigma_e = sqrt(2 s^2 (1 - rho_t) + sigma_m^2)`.
- `aperturediv.datasets`: seeded synthetic dataset; the feature builder takes
  only the estimate and the error level, so the label cannot reach the model
  through the feature path.
- `aperturediv.learned`: four non-learned references (MRC with the true
  state, MRC with the estimate, equal gain, and a one-parameter analytic
  shrinkage family fitted on the validation split) and a scikit-learn
  `HistGradientBoostingRegressor` combiner with a three-quantile penalty
  prediction as its uncertainty output.

#### BER
- `aperturediv.ber`: sample BPSK BER over lognormal fading with an exact
  binomial standard error (one bit per channel realisation), and a
  Gauss-Hermite quadrature diagnostic with a node-count guard against the
  installed NumPy's overflow above roughly 350 nodes.

#### Interfaces and evidence
- CLI `python -m aperturediv` with five subcommands: `channel`, `aperture`,
  `outage`, `combiner`, `ber`.
- Four examples writing PNGs to `screenshots/`.
- Six validation scripts with their raw output committed, including the
  binding cross-check X2 against P010 BERBench.

### Known limitations recorded at this release
- Aperture averaging is applied by scaling the scintillation index and
  keeping the distribution family; the family is not strictly preserved.
- The irradiance correlation scale is a free input, not derived from a
  turbulence profile.
- Diversity order is a finite-window measurement; the lognormal channel has
  no finite asymptotic diversity order at all, which is demonstrated rather
  than worked around.
- Over the operating range measured, the learned combiner's advantage over
  the best analytic rule is at most 0.10 dB, and at zero estimation error the
  analytic baseline is exactly optimal and the learned model is worse. Both
  are published as the result.

[0.1.0]: https://github.com/OmAcharya-avtr/aperturediv/releases/tag/v0.1.0

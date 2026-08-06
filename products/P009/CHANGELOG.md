# Changelog

## 0.1.0 — 2026-08-01

Initial release.

- Kim (SPIE 4214, 2001) and Kruse (1962) attenuation baselines with validated piecewise
  q(V), input validation, and array support.
- Seeded synthetic dataset generator (Kim + wavelength-dependent exponent noise,
  humidity covariate effect, 5 % measurement noise) with deterministic 70/15/15 split.
- FogCastModel: gradient-boosting point + 0.05/0.95 quantile models giving nominal 90 %
  prediction intervals; module-level predict() convenience API.
- Validation Level 2 evidence, benchmark (ML vs Kim vs Kruse), example figure,
  MODEL_CARD.md and DATASET_CARD.md, 34 passing tests, ruff-clean.

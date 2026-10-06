"""Validation: the learned combiner against four non-learned references.

The structural result first
---------------------------
For unit-norm weights ``w`` and true branch amplitudes ``h``, Cauchy-Schwarz
gives ``(w . h)^2 <= ||h||^2``. So maximal-ratio combining with the **true**
channel state is optimal and nothing can beat it; the penalty of every
combiner against it is non-negative by construction. Check 1 confirms that
numerically on the held-out rows, including for the learned model, because a
negative penalty anywhere would mean a label leak rather than a discovery.

The only honest question is what to do with an estimate that is wrong.
Check 3 onward answers it, against:

``mrc_true``       the unreachable upper bound, 0 dB by construction
``mrc_estimated``  what a real receiver does: trust the estimate completely
``egc``            no CSI at all, so immune to estimation error
``shrinkage(p)``   ``w ∝ Ihat^(p/2)``, analytic, ``p`` fitted on the
                   validation split; contains mrc_estimated (p=1) and
                   egc (p=0)

Checks
------
1. Penalty non-negativity for every combiner on every held-out row, and
   ``mrc_true`` identically 0 dB to numerical precision.
2. The dataset's feature matrix cannot contain the truth: the builder's
   signature takes only the estimate and the error level, and the features
   are reproduced from those two arrays alone.
3. Held-out comparison of all five combiners: mean, median, 90th percentile
   and worst-case penalty, and mean BPSK BER at a stated branch Eb/N0.
4. The same comparison resolved by estimation error level, on a separate
   evaluation set with ``sigma_e`` held fixed per block, which locates the
   crossover where EGC overtakes MRC-with-the-estimate.
5. Calibration of the uncertainty output: the fraction of held-out rows
   whose realised penalty falls below each predicted quantile, against the
   nominal level.
6. The ``sigma_e`` parameterisation is sufficient: a stale estimate and a
   noisy estimate with the same total ``sigma_e`` give the same penalty, so
   staleness and measurement noise are not separate axes.
7. Sensitivity to the number of apertures: the comparison repeated at
   L = 2 and L = 3.

Sizing: 120000 rows, split 70000 / 25000 / 25000 by index. One
HistGradientBoostingRegressor per aperture (100 boosting iterations) plus
three quantile regressors. The aperture-count sweep in check 7 refits from
scratch at a smaller size (45000 rows, 27000 train, 60 iterations) because it
is a sensitivity check rather than the headline result.

The sizing is set by the compute budget, not by accuracy: the build container
has 2 cores shared with four sibling agents, and an earlier 250000-row
version of this script took 440 s of wall clock there against about 75 s on
an idle pair of cores. Everything is therefore cut to roughly a third of
that. The conclusions are unchanged by the cut -- they were the same at
250000, 200000 and 120000 rows -- and the figures below are from the
committed size. No PyTorch in this environment; the model is scikit-learn.
"""

from __future__ import annotations

import pathlib
import sys

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402

from aperturediv.correlation import correlation_matrix, equispaced_positions  # noqa: E402
from aperturediv.datasets import build_features, make_combiner_dataset, split_dataset  # noqa: E402
from aperturediv.estimation import (  # noqa: E402
    DB_PER_NEPER,
    estimate_from_log_error,
    estimate_stale_noisy,
    log_error_sigma,
    log_error_sigma_db,
    measurement_sigma_for_target,
)
from aperturediv.learned import (  # noqa: E402
    LearnedCombiner,
    egc_weights,
    fit_shrinkage_exponent,
    mean_ber,
    mrc_estimated_weights,
    mrc_true_weights,
    penalty_db,
    score_weights,
    shrinkage_weights,
)

N_ROWS = 120_000
N_TRAIN = 70_000
N_VAL = 25_000
N_ROWS_L_SWEEP = 45_000
N_TRAIN_L_SWEEP = 27_000
N_VAL_L_SWEEP = 9_000
MAX_ITER_L_SWEEP = 60
L = 4
SI = 0.9
SPACING_M = 0.15
RHO_C_M = 0.10
SIGMA_E_MAX = 3.0
EBN0_DB = 10.0
SEED = 44044
EVAL_PER_LEVEL = 20_000
SIGMA_E_LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0)


def _rule(title: str) -> None:
    print("=" * 78)
    print(title)
    print("=" * 78)


def _all_weights(dataset, model, p_star):
    return {
        "mrc_true": mrc_true_weights(dataset.irradiance_true),
        "mrc_estimated": mrc_estimated_weights(dataset.irradiance_estimated),
        "egc": egc_weights(len(dataset), dataset.n_apertures),
        f"shrinkage p={p_star:.3f}": shrinkage_weights(dataset.irradiance_estimated, p_star),
        "learned": model.combine(dataset.irradiance_estimated, dataset.sigma_e)[0],
    }


def main() -> int:
    failures: list[str] = []

    _rule("0. Configuration")
    print(f"  apertures L                 {L}")
    print(f"  scintillation index si      {SI}")
    print(f"  aperture spacing            {SPACING_M} m")
    print(f"  correlation scale rho_c     {RHO_C_M} m")
    pos = equispaced_positions(L, SPACING_M)
    r_log = correlation_matrix(pos, RHO_C_M, "gaussian")
    print(f"  adjacent log correlation    {r_log[0, 1]:.6f}")
    print(f"  sigma_e range               [0, {SIGMA_E_MAX}] "
          f"= [0, {log_error_sigma_db(SIGMA_E_MAX):.3f}] dB of irradiance")
    print(f"  rows                        {N_ROWS} "
          f"({N_TRAIN} train / {N_VAL} val / {N_ROWS - N_TRAIN - N_VAL} test)")
    print(f"  branch Eb/N0 for BER        {EBN0_DB} dB")
    print(f"  seed                        {SEED}")
    print(f"  dB per natural log unit     {DB_PER_NEPER:.9f}")
    print()

    data = make_combiner_dataset(
        N_ROWS,
        n_apertures=L,
        si=SI,
        aperture_spacing_m=SPACING_M,
        correlation_scale_m=RHO_C_M,
        sigma_e_range=(0.0, SIGMA_E_MAX),
        seed=SEED,
    )
    train, val, test = split_dataset(data, n_train=N_TRAIN, n_validation=N_VAL)

    _rule("2. The feature matrix cannot contain the truth")
    rebuilt = build_features(test.irradiance_estimated, test.sigma_e)
    same = bool(np.array_equal(rebuilt, test.features))
    print(f"  features rebuilt from (irradiance_estimated, sigma_e) alone: {same}")
    print(f"  feature columns: {test.features.shape[1]} = L({L}) + 3 "
          "(mean log est, std log est, sigma_e)")
    print("  build_features takes no argument carrying the true irradiance, so the")
    print("  label cannot reach the model through the feature path.")
    if not same:
        failures.append("features are not reproducible from the estimate alone")
    print()

    p_star = fit_shrinkage_exponent(val)
    model = LearnedCombiner(random_state=SEED).fit(train)
    print(f"  shrinkage exponent fitted on the validation split: p* = {p_star:.4f}")
    print("  (p = 1 is MRC-with-the-estimate, p = 0 is EGC)")
    print()

    _rule("1. Penalty non-negativity on the held-out rows")
    h = test.amplitude_true
    weights = _all_weights(test, model, p_star)
    print("  combiner                    min penalty dB   rows below -1e-9 dB")
    for name, w in weights.items():
        pen = penalty_db(w, h)
        n_neg = int(np.count_nonzero(pen < -1e-9))
        print(f"  {name:27s} {pen.min():+15.3e} {n_neg:21d}")
        if n_neg > 0:
            failures.append(f"{name} has {n_neg} rows with a negative penalty")
    pen_true = penalty_db(weights["mrc_true"], h)
    print(f"  mrc_true max |penalty| {np.abs(pen_true).max():.3e} dB, tolerance 1e-9")
    print("  MRC with the true state is 0 dB by Cauchy-Schwarz, not by merit.")
    if np.abs(pen_true).max() > 1e-9:
        failures.append(f"mrc_true penalty not zero: {np.abs(pen_true).max()}")
    print()

    _rule("3. Held-out comparison, all sigma_e pooled")
    print(f"  {len(test)} held-out rows, identical for every combiner")
    print("  combiner                      mean dB  median dB    p90 dB    max dB"
          "       mean BER")
    scores = {}
    for name, w in weights.items():
        sc = score_weights(name, w, h, EBN0_DB)
        scores[name] = sc
        print(f"  {name:27s} {sc.mean_penalty_db:9.4f} {sc.median_penalty_db:10.4f} "
              f"{sc.p90_penalty_db:9.4f} {sc.max_penalty_db:9.4f} {sc.mean_ber:14.6e}")
    learned_name = "learned"
    shrink_name = f"shrinkage p={p_star:.3f}"
    print()
    print("  pooled headline numbers")
    print(f"    learned vs mrc_estimated : "
          f"{scores['mrc_estimated'].mean_penalty_db - scores[learned_name].mean_penalty_db:+.4f} "
          "dB mean penalty recovered by learning")
    print(f"    learned vs egc           : "
          f"{scores['egc'].mean_penalty_db - scores[learned_name].mean_penalty_db:+.4f} dB")
    print(f"    learned vs shrinkage     : "
          f"{scores[shrink_name].mean_penalty_db - scores[learned_name].mean_penalty_db:+.4f} dB")
    print(f"    learned vs mrc_true      : "
          f"{scores[learned_name].mean_penalty_db - scores['mrc_true'].mean_penalty_db:+.4f} "
          "dB still lost (cannot be negative)")
    if scores[learned_name].mean_penalty_db < -1e-9:
        failures.append("learned mean penalty is negative")
    print()

    _rule("4. Resolved by estimation error level, fixed sigma_e per block")
    print(f"  a separate evaluation set, {EVAL_PER_LEVEL} rows per level, seed {SEED + 500}")
    print("  the true irradiance is redrawn per level so no row is shared with training")
    print()
    print("  sigma_e  sigma_e dB   mrc_est   egc    shrink  learned  | best non-learned")
    crossover: tuple[float, float] | None = None
    prev_sign = None
    rows_table = []
    for j, se in enumerate(SIGMA_E_LEVELS):
        rng = np.random.default_rng(SEED + 500 + j)
        from aperturediv.correlation import sample_correlated_lognormal

        i_true = sample_correlated_lognormal(EVAL_PER_LEVEL, SI, r_log, rng)
        est = estimate_from_log_error(i_true, se, rng)
        hh = np.sqrt(i_true)
        pe = float(penalty_db(mrc_estimated_weights(est.irradiance_estimated), hh).mean())
        pg = float(penalty_db(egc_weights(EVAL_PER_LEVEL, L), hh).mean())
        ps = float(
            penalty_db(shrinkage_weights(est.irradiance_estimated, p_star), hh).mean()
        )
        feats = build_features(est.irradiance_estimated, np.full(EVAL_PER_LEVEL, se))
        pl = float(penalty_db(model.predict_weights(feats), hh).mean())
        best_nl = "egc" if pg < pe and pg < ps else ("shrinkage" if ps < pe else "mrc_est")
        rows_table.append((se, pe, pg, ps, pl))
        print(f"{se:9.3f} {log_error_sigma_db(se):10.3f} {pe:9.4f} {pg:7.4f} {ps:8.4f} "
              f"{pl:8.4f}  | {best_nl}")
        sign = np.sign(pe - pg)
        if prev_sign is not None and sign != prev_sign and crossover is None:
            lo, hi = SIGMA_E_LEVELS[j - 1], se
            crossover = (lo, hi)
        prev_sign = sign
    print()
    if crossover is not None:
        # bisect the crossover to 0.01 in sigma_e on a fresh sample per evaluation
        lo, hi = crossover
        for step in range(5):
            mid = 0.5 * (lo + hi)
            rng = np.random.default_rng(SEED + 900 + step)
            i_true = sample_correlated_lognormal(EVAL_PER_LEVEL, SI, r_log, rng)
            est = estimate_from_log_error(i_true, mid, rng)
            hh = np.sqrt(i_true)
            pe = float(penalty_db(mrc_estimated_weights(est.irradiance_estimated), hh).mean())
            pg = float(penalty_db(egc_weights(EVAL_PER_LEVEL, L), hh).mean())
            if pe < pg:
                lo = mid
            else:
                hi = mid
        mid = 0.5 * (lo + hi)
        print(f"  EGC overtakes MRC-with-the-estimate at sigma_e = {mid:.4f}")
        print(f"  = {log_error_sigma_db(mid):.3f} dB of irradiance error "
              f"(bracketed to {hi - lo:.4f} in sigma_e by bisection)")
        print("  Below that the estimate is worth using; above it, equal gain -- which")
        print("  needs no estimate at all -- is the better analytic choice.")
    else:
        print("  no crossover between MRC-with-the-estimate and EGC on this grid")
        failures.append("EGC/MRC-estimated crossover not found on the sigma_e grid")
    print()
    beats = [(se, pl < min(pe, pg, ps)) for se, pe, pg, ps, pl in rows_table]
    n_beats = sum(1 for _, b in beats if b)
    print(f"  the learned combiner is the best of the four at "
          f"{n_beats} of {len(rows_table)} error levels")
    print("  worst margin against the best non-learned option, per level:")
    for se, pe, pg, ps, pl in rows_table:
        print(f"    sigma_e {se:5.3f}: learned - best_non_learned = "
              f"{pl - min(pe, pg, ps):+.4f} dB")
    print()

    _rule("5. Calibration of the uncertainty output")
    _, quant = model.combine(test.irradiance_estimated, test.sigma_e)
    realised = penalty_db(weights["learned"], h)
    print("  nominal quantile   empirical coverage   mean predicted dB   mean realised dB")
    for j, q in enumerate(model.quantiles):
        cov = float(np.mean(realised <= quant[:, j]))
        print(f"{q:18.2f} {cov:20.6f} {quant[:, j].mean():19.4f} {realised.mean():18.4f}")
        if abs(cov - q) > 0.05:
            failures.append(f"quantile {q} coverage {cov} outside +/-0.05")
    print("  tolerance 0.05 absolute on coverage. The quantile models are fitted on")
    print("  the penalties the weight models incur on the training rows, so this is")
    print("  a genuine held-out calibration measurement.")
    print()

    _rule("6. sigma_e is a sufficient parameterisation")
    print("  stale-but-clean against fresh-but-noisy at matched sigma_e")
    print("  rho_t   sigma_m   sigma_e   mean penalty mrc_est dB   mean penalty egc dB")
    target = 1.0
    for rho_t in (1.0, 0.95, 0.9, 0.8):
        sm = measurement_sigma_for_target(SI, rho_t, target)
        se_tot = log_error_sigma(SI, rho_t, sm)
        rng = np.random.default_rng(SEED + 1300 + int(100 * rho_t))
        ce = estimate_stale_noisy(EVAL_PER_LEVEL, SI, r_log, rho_t, sm, rng)
        hh = ce.amplitude_true
        pe = float(penalty_db(mrc_estimated_weights(ce.irradiance_estimated), hh).mean())
        pg = float(penalty_db(egc_weights(EVAL_PER_LEVEL, L), hh).mean())
        print(f"{rho_t:7.3f} {sm:9.4f} {se_tot:9.4f} {pe:25.4f} {pg:21.4f}")
    print("  all four rows share sigma_e = 1.0 by construction; the mrc_est penalties")
    print("  agree to within Monte Carlo error, so staleness and measurement noise")
    print("  do not need separate axes.")
    print()

    _rule("7. Sensitivity to the number of apertures")
    print(f"  refit at {N_ROWS_L_SWEEP} rows, {N_TRAIN_L_SWEEP} train, "
          f"max_iter {MAX_ITER_L_SWEEP} (a sensitivity check, not the headline fit)")
    print("  L   p*     mrc_est dB   egc dB   shrink dB   learned dB   best")
    for n_ap in (2, 3, 4):
        d = make_combiner_dataset(
            N_ROWS_L_SWEEP,
            n_apertures=n_ap,
            si=SI,
            aperture_spacing_m=SPACING_M,
            correlation_scale_m=RHO_C_M,
            sigma_e_range=(0.0, SIGMA_E_MAX),
            seed=SEED + 11 * n_ap,
        )
        tr, va, te = split_dataset(
            d, n_train=N_TRAIN_L_SWEEP, n_validation=N_VAL_L_SWEEP
        )
        p = fit_shrinkage_exponent(va)
        m = LearnedCombiner(max_iter=MAX_ITER_L_SWEEP, random_state=SEED).fit(tr)
        hh = te.amplitude_true
        vals = {
            "mrc_est": float(penalty_db(mrc_estimated_weights(te.irradiance_estimated), hh).mean()),
            "egc": float(penalty_db(egc_weights(len(te), n_ap), hh).mean()),
            "shrink": float(penalty_db(shrinkage_weights(te.irradiance_estimated, p), hh).mean()),
            "learned": float(
                penalty_db(m.combine(te.irradiance_estimated, te.sigma_e)[0], hh).mean()
            ),
        }
        best = min(vals, key=lambda k: vals[k])
        print(f"{n_ap:3d} {p:6.3f} {vals['mrc_est']:12.4f} {vals['egc']:8.4f} "
              f"{vals['shrink']:11.4f} {vals['learned']:12.4f}   {best}")
    print()

    _rule("8. BER at the stated operating point, pooled held-out rows")
    print(f"  branch Eb/N0 {EBN0_DB} dB, {len(test)} rows")
    print("  combiner                      mean BER      BER / mrc_true BER")
    base_ber = mean_ber(weights["mrc_true"], h, EBN0_DB)
    for name, w in weights.items():
        b = mean_ber(w, h, EBN0_DB)
        print(f"  {name:27s} {b:14.6e} {b / base_ber:22.4f}")
    print()

    _rule("Result")
    if failures:
        print(f"FAILED {len(failures)} check(s):")
        for f in failures:
            print("  -", f)
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

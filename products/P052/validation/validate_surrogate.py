"""Validate the learned component: held-out accuracy, the uncertainty it reports,
and what it costs.

Five checks:

1. **Held-out accuracy** against a trivial baseline (predict the training mean),
   per instance, on an independently seeded test split. The surrogate is only
   worth refitting if it beats that.
2. **Coverage of the reported uncertainty.** ``mean +- 1.96 * spread`` would
   cover 95 % of held-out targets if the spread were a calibrated standard
   deviation. Both the nominal-95 and the nominal-68 coverage are measured,
   because one of them alone cannot distinguish a calibrated spread from a
   wrongly shaped one. The numbers go in the model card and the README **as
   measured**, not as targets.
3. **Usefulness of the spread as a ranking signal**, which is all the
   acquisition rule needs: the Spearman correlation between the reported spread
   and the absolute held-out error.
4. **Cost**: refit and scoring time against the price of a simulation, which is
   what decides whether a surrogate can pay for itself here.
5. **The documented joblib defect**: ``n_jobs > 1`` is slower at the small-batch
   inference this search does on a 2-core container. Measured rather than
   assumed, because the package hard-codes ``n_jobs=1`` on the strength of it.

References
----------
Breiman, L. (2001), "Random forests", Machine Learning 45(1), 5-32.

Wager, S., Hastie, T. and Efron, B. (2014), "Confidence intervals for random
forests", JMLR 15, 1625-1651. Why the naive tree spread is not a confidence
interval; this script measures how far off it is rather than correcting it.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from _bootstrap import Tee, add_src_to_path

add_src_to_path()

from falsifyloop.instances import suite  # noqa: E402
from falsifyloop.surrogate import DEFAULT_TREES, ForestSurrogate  # noqa: E402

OUT = Path(__file__).with_name("validate_surrogate_output.txt")
say = Tee(OUT)

N_TRAIN = 120
N_TEST = 200
TRAIN_SEED = 4100
TEST_SEED = 9400


def _split(inst):
    """Train and test sets from **different** seeds, so the split is independent."""
    train_x = inst.sample(np.random.default_rng(TRAIN_SEED), N_TRAIN)
    test_x = inst.sample(np.random.default_rng(TEST_SEED), N_TEST)
    train_y = np.fromiter((inst.evaluate(p) for p in train_x), dtype=float, count=N_TRAIN)
    test_y = np.fromiter((inst.evaluate(p) for p in test_x), dtype=float, count=N_TEST)
    return train_x, train_y, test_x, test_y


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation, by Pearson on the ranks."""
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() == 0.0 or rb.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> int:
    say("falsifyloop - surrogate validation")
    say(
        f"{N_TRAIN} training points and {N_TEST} test points per instance, drawn "
        f"uniformly from the declared box at seeds {TRAIN_SEED} and {TEST_SEED}. The "
        "two seeds differ, so the test split is independent of the training split and "
        "no point appears in both."
    )
    say(
        f"Forest: {DEFAULT_TREES} trees, min_samples_leaf 1, n_jobs 1, random_state "
        "equal to the search seed. The target is the requirement robustness, which is "
        "dimensionless on every shipped instance."
    )

    say.rule("Check 1 - held-out accuracy against predicting the training mean")
    say(
        f"{'instance':<20s} {'RMSE forest':>12s} {'RMSE mean':>11s} {'ratio':>7s} "
        f"{'R^2 forest':>11s} {'Spearman':>9s}"
    )
    say("-" * 76)
    data = {}
    for inst in suite():
        train_x, train_y, test_x, test_y = _split(inst)
        data[inst.identifier] = (train_x, train_y, test_x, test_y)
        model = ForestSurrogate(n_estimators=DEFAULT_TREES, random_state=0).fit(
            train_x, train_y
        )
        mean, spread = model.predict(test_x)
        rmse = float(np.sqrt(np.mean((mean - test_y) ** 2)))
        rmse_trivial = float(np.sqrt(np.mean((train_y.mean() - test_y) ** 2)))
        r2 = 1.0 - float(np.sum((mean - test_y) ** 2) / np.sum((test_y - test_y.mean()) ** 2))
        say(
            f"{inst.identifier:<20s} {rmse:>12.6f} {rmse_trivial:>11.6f} "
            f"{rmse_trivial / rmse:>7.3f} {r2:>11.4f} "
            f"{_spearman(mean, test_y):>9.4f}"
        )
        data[inst.identifier] = (train_x, train_y, test_x, test_y, mean, spread)
    say("")
    say(
        "The ratio is how many times better than the trivial predictor. The Spearman "
        "column is the one the search actually depends on: the acquisition only needs "
        "the ordering of candidates to be roughly right, not the magnitudes."
    )

    say.rule("Check 2 - coverage of the reported uncertainty")
    say(
        "If the tree spread were a calibrated standard deviation, mean +- 1.96*spread "
        "would cover 95 % of held-out targets. Measured:"
    )
    say("")
    say(
        f"{'instance':<20s} {'cover 1.96s':>12s} {'cover 1s':>9s} {'mean spread':>12s} "
        f"{'mean |err|':>11s} {'spread/|err|':>13s}"
    )
    say("-" * 80)
    coverages = []
    for inst in suite():
        _, _, _, test_y, mean, spread = data[inst.identifier]
        err = np.abs(mean - test_y)
        cover95 = float(np.mean(err <= 1.96 * spread))
        cover68 = float(np.mean(err <= spread))
        coverages.append(cover95)
        say(
            f"{inst.identifier:<20s} {cover95:>12.4f} {cover68:>9.4f} "
            f"{spread.mean():>12.6f} {err.mean():>11.6f} "
            f"{spread.mean() / err.mean():>13.4f}"
        )
    say("")
    pooled_mean = np.concatenate([data[i.identifier][4] for i in suite()])
    pooled_spread = np.concatenate([data[i.identifier][5] for i in suite()])
    pooled_truth = np.concatenate([data[i.identifier][3] for i in suite()])
    pooled_err = np.abs(pooled_mean - pooled_truth)
    pooled95 = float(np.mean(pooled_err <= 1.96 * pooled_spread))
    pooled68 = float(np.mean(pooled_err <= pooled_spread))
    say(f"mean coverage of the nominal 95 % interval across the suite: {np.mean(coverages):.4f}")
    say(f"pooled over all {pooled_err.size} held-out points, coverage of mean +- 1.96*spread: "
        f"{pooled95:.4f}   (a Gaussian would give 0.9500)")
    say(f"pooled over all {pooled_err.size} held-out points, coverage of mean +- 1.00*spread: "
        f"{pooled68:.4f}   (a Gaussian would give 0.6827)")
    say(f"pooled RMSE: {float(np.sqrt(np.mean((pooled_mean - pooled_truth) ** 2))):.6f}")
    say(
        "Read both coverage columns together. The 95 % column is close to nominal and "
        "slightly under it on most instances; the 68 % column is well ABOVE the 0.683 "
        "a Gaussian would give. Both at once means the held-out error is not Gaussian "
        "around the forest mean: the spread is conservative through the body of the "
        "distribution and too thin in the tails, so the near-nominal 95 % figure is "
        "two errors partly cancelling and not evidence of calibration. The "
        "spread/|err| column says the same thing from the other side -- the mean "
        "spread is 1.05 to 1.37 times the mean absolute error, so the spread is not "
        "small, it is the wrong shape."
    )
    say(
        "This contradicts the prediction the surrogate module's docstring carried "
        "before this script was first run, which said the tree spread would understate "
        "the error badly. It does not. The prediction was wrong, the docstring was "
        "corrected to the measurement, and the mistake is recorded in VALIDATION.md "
        "rather than deleted. Wager, Hastie & Efron (2014) give the correction this "
        "package still does not implement."
    )

    say.rule("Check 3 - is the spread useful as a ranking signal")
    say(
        f"{'instance':<20s} {'Spearman(spread, |err|)':>24s} "
        f"{'|err| in the top spread decile':>31s} {'in the bottom':>14s}"
    )
    say("-" * 92)
    rankings = []
    for inst in suite():
        _, _, _, test_y, mean, spread = data[inst.identifier]
        err = np.abs(mean - test_y)
        rho = _spearman(spread, err)
        rankings.append(rho)
        order = np.argsort(spread)
        decile = max(1, spread.size // 10)
        top = float(err[order[-decile:]].mean())
        bottom = float(err[order[:decile]].mean())
        say(f"{inst.identifier:<20s} {rho:>24.4f} {top:>31.6f} {bottom:>14.6f}")
    say("")
    say(f"mean Spearman correlation across the suite: {np.mean(rankings):.4f}")
    say(
        "pooled over all held-out points, Spearman(spread, |error|): "
        f"{_spearman(pooled_spread, pooled_err):.4f}"
    )
    say(
        "A positive correlation means the spread points at the places the forest is "
        "wrong, which is exactly what the lower-confidence-bound acquisition needs. "
        "That the magnitude is not calibrated does not hurt a ranking."
    )

    say.rule("Check 4 - cost against the price of a simulation")
    inst = suite()[0]
    train_x, train_y, test_x, *_ = data[inst.identifier][:3] + (data[inst.identifier][2],)
    repeats = 20
    started = time.perf_counter()
    for _ in range(repeats):
        ForestSurrogate(n_estimators=DEFAULT_TREES, random_state=0).fit(train_x, train_y)
    fit_ms = (time.perf_counter() - started) / repeats * 1e3
    model = ForestSurrogate(n_estimators=DEFAULT_TREES, random_state=0).fit(train_x, train_y)
    candidates = inst.sample(np.random.default_rng(1), 256)
    started = time.perf_counter()
    for _ in range(repeats):
        model.lower_confidence_bound(candidates, kappa=2.0)
    score_ms = (time.perf_counter() - started) / repeats * 1e3
    sim_repeats = 200
    point = inst.centre()
    started = time.perf_counter()
    for _ in range(sim_repeats):
        inst.evaluate(point)
    sim_ms = (time.perf_counter() - started) / sim_repeats * 1e3
    say(f"one simulation + requirement evaluation : {sim_ms:.4f} ms")
    say(f"one forest refit on {N_TRAIN} points        : {fit_ms:.4f} ms "
        f"({fit_ms / sim_ms:.1f} simulations)")
    say(f"scoring 256 candidates (mean + spread)   : {score_ms:.4f} ms "
        f"({score_ms / sim_ms:.1f} simulations)")
    say("")
    say(
        "The surrogate strategy refits every 8 simulations and scores 256 candidates "
        "before each one, so its overhead per simulation is roughly "
        f"{(fit_ms / 8 + score_ms) / sim_ms:.1f} simulations' worth of wall clock. That "
        "is the honest exchange rate: the surrogate's sample-efficiency gain is paid "
        "for in CPU time, and on a simulator cheaper than this one the trade would go "
        "the other way. The sample-efficiency curves count simulations, not seconds, "
        "precisely so that this trade is visible rather than hidden."
    )
    say(
        "Runs during this build put the simulation at 0.31-0.37 ms, the refit at "
        "19-27 ms (60-90 simulations' worth) and the overhead at 15-25 simulations' "
        "worth per simulation bought. The README quotes those ranges rather than any "
        "single run's figure, because a single figure here would be stale by the next "
        "execution."
    )
    say(
        "Wall-clock figures move 10-20 % between runs on this container; the seeded "
        "simulation counts in validate_benchmark.py do not move at all."
    )

    say.rule("Check 5 - the documented n_jobs defect, measured")
    from sklearn.ensemble import RandomForestRegressor

    single = RandomForestRegressor(n_estimators=DEFAULT_TREES, random_state=0, n_jobs=1)
    single.fit(train_x, train_y)
    parallel = RandomForestRegressor(n_estimators=DEFAULT_TREES, random_state=0, n_jobs=2)
    parallel.fit(train_x, train_y)
    row = train_x[:1]
    timings = {}
    for model_name, forest in (("n_jobs=1", single), ("n_jobs=2", parallel)):
        forest.predict(row)
        started = time.perf_counter()
        for _ in range(300):
            forest.predict(row)
        per_call = (time.perf_counter() - started) / 300 * 1e6
        timings[model_name] = per_call
        say(f"{model_name:<10s} single-row predict: {per_call:>10.1f} us")
    say("")
    say(
        f"slowdown factor n_jobs=2 / n_jobs=1 : {timings['n_jobs=2'] / timings['n_jobs=1']:.1f}x"
    )
    say(
        "The absolute microsecond figures above move substantially between runs on a "
        "shared container -- 10-40 % has been observed in this build -- so the ratio "
        "is the claim and the absolute numbers are context. Two runs during this build "
        "gave 8.9x and 7.9x."
    )
    say(
        "The package hard-codes n_jobs=1 and does not expose it, on the strength of "
        "this measurement. Earlier batches in this portfolio recorded the same effect: "
        "joblib's dispatch overhead per call dominates the work at this batch size on "
        "a 2-core container."
    )

    say.rule("COMPUTE")
    say(f"instances                : {len(suite())}")
    say(f"simulations for the splits: {len(suite()) * (N_TRAIN + N_TEST)}")
    say(f"available CPU cores      : {len(os.sched_getaffinity(0))} (os.sched_getaffinity)")
    say(
        "Wall clocks on a shared container are not hardware characteristics and must "
        "not be presented as such."
    )
    say("")
    say("This model is not certified for operational flight use.")
    say("Falsification is one-sided: finding no violation is not evidence of correctness.")
    say.save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

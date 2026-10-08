"""The headline deliverable: sample-efficiency curves over the whole suite.

Every strategy, every instance, the same budget, the same seeds. The report is
ordered the way it should be read:

1. **Per-instance**, because that is where a falsification user lives.
2. **Where uniform random wins**, named explicitly, including the hardest
   instance.
3. Curves with bootstrap bands at selected simulation counts.
4. Only then the aggregate, with the warning that it hides the above.
5. The surrogate's warm-start accounting, so that the learned component's result
   is reported against the simulations it actually had to work with.
6. Compute.

Uniform random is the baseline and it is a strong one. The empirical baseline
curve is also compared against the closed form ``1 - (1-p)^n`` from the measured
difficulty, which is a check on the whole measurement chain: if those two
disagree beyond the band, something in the harness is wrong.

Reference: Efron & Tibshirani (1993), *An Introduction to the Bootstrap*, ch. 13,
for the percentile interval.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from _bootstrap import Tee, add_src_to_path

add_src_to_path()

from falsifyloop.benchmark import run_benchmark  # noqa: E402
from falsifyloop.curves import clopper_pearson  # noqa: E402
from falsifyloop.instances import suite  # noqa: E402
from falsifyloop.report import render_aggregate_table, render_cell_table  # noqa: E402
from falsifyloop.search import BASELINE, STRATEGIES, analytic_random_curve  # noqa: E402

OUT = Path(__file__).with_name("validate_benchmark_output.txt")
say = Tee(OUT)

BUDGET = 100
REPEATS = 30
BASE_SEED = 52000
DIFFICULTY_DRAWS = 8000
DIFFICULTY_SEED = 909
N_BOOT = 2000
MARKS = (10, 25, 50, 75, 100)


def _measure_difficulty() -> dict[str, tuple[int, int]]:
    """Independent difficulty estimate, so this script stands on its own."""
    out = {}
    for inst in suite():
        rng = np.random.default_rng(DIFFICULTY_SEED)
        points = inst.sample(rng, DIFFICULTY_DRAWS)
        violations = int(
            np.count_nonzero(
                np.fromiter(
                    (inst.evaluate(p) for p in points), dtype=float, count=DIFFICULTY_DRAWS
                )
                < 0.0
            )
        )
        out[inst.identifier] = (violations, DIFFICULTY_DRAWS)
    return out


def main() -> int:
    say("falsifyloop - sample-efficiency benchmark")
    say(
        f"{len(suite())} instances x {len(STRATEGIES)} strategies x {REPEATS} seeds, "
        f"budget {BUDGET} simulations, base seed {BASE_SEED}. Repeat r of every cell "
        f"uses seed {BASE_SEED} + r, so all strategies see the same seeds."
    )
    say(
        "Every strategy stops at its first violation, so the measurement is the "
        "distribution of the first-violation simulation count, right-censored at the "
        "budget. A censored run found NOTHING; it is not a late success and it is not "
        "evidence that the instance is unfalsifiable."
    )

    started = time.perf_counter()
    difficulty = _measure_difficulty()
    difficulty_seconds = time.perf_counter() - started

    report = run_benchmark(
        budget=BUDGET, repeats=REPEATS, base_seed=BASE_SEED
    )
    total_seconds = time.perf_counter() - started

    say.rule("1. PER-INSTANCE RESULTS (read this first)")
    say(render_cell_table(report))

    say.rule("2. WHERE UNIFORM RANDOM WINS")
    say(
        "Won on the mean curve probability -- the average probability of having found a "
        "violation over the whole budget, which uses the entire curve rather than one "
        "point on it. Ties are not wins."
    )
    say("")
    header = f"{'strategy':<22s} {'instances the baseline beats it on':<56s}"
    say(header)
    say("-" * len(header))
    hardest = report.hardest_instance()
    loss_table = {}
    for name in report.strategy_names:
        if name == BASELINE:
            continue
        losses = report.baseline_wins(name)
        loss_table[name] = losses
        say(f"{name:<22s} {(', '.join(losses) if losses else 'none'):<56s}")
    say("")
    say(f"Hardest instance for the baseline: {hardest}")
    say("")
    say(
        f"{'strategy':<22s} {'mean P on ' + hardest:>26s} "
        f"{'beats baseline there':>22s}"
    )
    say("-" * 72)
    baseline_hard = report.cell(hardest, BASELINE).mean_curve_probability
    for name in report.strategy_names:
        value = report.cell(hardest, name).mean_curve_probability
        if name == BASELINE:
            verdict = "(is the baseline)"
        else:
            verdict = "yes" if value > baseline_hard else "NO"
        say(f"{name:<22s} {value:>26.4f} {verdict:>22s}")

    say.rule("2b. ARE THOSE WINS AND LOSSES STATISTICALLY DISTINGUISHABLE")
    say(
        "A bare win/loss ledger over 30 seeds is itself uncertain, and saying so is "
        "part of reporting it honestly. For every instance and every non-baseline "
        "strategy, the difference in mean curve probability (strategy minus baseline) "
        f"is bootstrapped with {N_BOOT} resamples of the 30 runs on each side "
        "independently, and the 95 % percentile interval is reported. An interval that "
        "straddles zero means the suite cannot tell the two apart at this repeat count, "
        "whichever way the point estimate fell."
    )
    say("")
    head = (
        f"{'instance':<20s} {'strategy':<21s} {'delta':>8s} "
        f"{'95% interval on delta':>26s} {'verdict':>14s}"
    )
    say(head)
    say("-" * len(head))
    boot_rng = np.random.default_rng(BASE_SEED)
    undecided = 0
    decided_losses = 0
    for inst in suite():
        base_cell = report.cell(inst.identifier, BASELINE)
        base_curve_runs = np.asarray(
            [[1.0 if (v is not None and v <= m) else 0.0 for m in range(1, BUDGET + 1)]
             for v in base_cell.first_violations]
        )
        for name in report.strategy_names:
            if name == BASELINE:
                continue
            cell = report.cell(inst.identifier, name)
            other_runs = np.asarray(
                [[1.0 if (v is not None and v <= m) else 0.0 for m in range(1, BUDGET + 1)]
                 for v in cell.first_violations]
            )
            rows = base_curve_runs.shape[0]
            pick_b = boot_rng.integers(0, rows, size=(N_BOOT, rows))
            pick_o = boot_rng.integers(0, other_runs.shape[0], size=(N_BOOT, rows))
            deltas = (
                other_runs[pick_o].mean(axis=1).mean(axis=1)
                - base_curve_runs[pick_b].mean(axis=1).mean(axis=1)
            )
            lo, hi = float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))
            point = cell.mean_curve_probability - base_cell.mean_curve_probability
            if lo > 0.0:
                verdict = "beats baseline"
            elif hi < 0.0:
                verdict = "LOSES"
                decided_losses += 1
            else:
                verdict = "undecided"
                undecided += 1
            say(
                f"{inst.identifier:<20s} {name:<21s} {point:>+8.4f} "
                f"[{lo:>+11.4f}, {hi:>+11.4f}] {verdict:>14s}"
            )
        say("")
    say(
        f"comparisons undecided at 30 seeds : {undecided} of "
        f"{len(suite()) * (len(report.strategy_names) - 1)}"
    )
    say(f"losses to the baseline that the data can resolve : {decided_losses}")
    say(
        "The undecided count is the honest caveat on section 2: a per-instance "
        "win/loss ledger at 30 seeds resolves a large difference and nothing smaller. "
        "Re-running the grid at a different base seed moves the marginal entries, and "
        "the ones that move are exactly the undecided rows here."
    )

    say.rule("3. CURVES WITH BOOTSTRAP BANDS")
    say(
        f"Pointwise 95 % percentile bootstrap, {N_BOOT} resamples, resampling unit is "
        "one run. Pointwise, not simultaneous: reading the whole band as a 95 % "
        "envelope for the entire curve overstates it."
    )
    for inst in suite():
        iid = inst.identifier
        violations, draws = difficulty[iid]
        p = violations / draws
        lo_p, hi_p = clopper_pearson(violations, draws)
        say("")
        say(
            f"--- {iid} [{inst.tier}] measured p = {p:.6f} "
            f"(95% CI [{lo_p:.6f}, {hi_p:.6f}] from {draws} draws)"
        )
        head = f"{'strategy':<22s} " + " ".join(f"{'n=' + str(m):>19s}" for m in MARKS)
        say(head)
        say("-" * len(head))
        for name in report.strategy_names:
            cell = report.cell(iid, name)
            curve = cell.curve()
            lower, upper = cell.band(n_boot=N_BOOT, seed=BASE_SEED)
            cells = " ".join(
                f"{curve[m - 1]:.2f}[{lower[m - 1]:.2f},{upper[m - 1]:.2f}]".rjust(19)
                for m in MARKS
            )
            say(f"{name:<22s} {cells}")
        if p > 0.0:
            analytic = analytic_random_curve(p, BUDGET)
            cells = " ".join(f"{analytic[m - 1]:>19.2f}" for m in MARKS)
            say(f"{'closed form 1-(1-p)^n':<22s} {cells}")

    say.rule("4. CONSISTENCY OF THE BASELINE AGAINST ITS CLOSED FORM")
    say(
        "The empirical uniform-random curve must agree with 1 - (1-p)^n, since uniform "
        "random IS that process. A disagreement would mean the harness, not the "
        "strategy, is wrong."
    )
    say("")
    say(
        "The comparison is made as an interval overlap, NOT against the bootstrap band. "
        "Reason, found while building this script and recorded rather than quietly "
        "fixed: the percentile bootstrap band has exactly zero width whenever all "
        f"{REPEATS} runs agree, which happens at every n on the easy instances, so it "
        "cannot cover anything and the comparison fails by construction. The honest "
        "test is a Clopper-Pearson interval on the empirical count of runs that "
        f"succeeded by n, out of {REPEATS}, against the closed-form interval implied by "
        "the Clopper-Pearson interval on p itself."
    )
    say("")
    head = (
        f"{'instance':<20s} {'n':>4s} {'k/R':>7s} {'empirical 95% CI':>22s} "
        f"{'closed-form 95% CI':>22s} {'overlap':>8s} {'band width':>11s}"
    )
    say(head)
    say("-" * len(head))
    outside = 0
    checked = 0
    degenerate = 0
    for inst in suite():
        violations, draws = difficulty[inst.identifier]
        p = violations / draws
        if p <= 0.0:
            continue
        p_lo, p_hi = clopper_pearson(violations, draws)
        cell = report.cell(inst.identifier, BASELINE)
        curve = cell.curve()
        lower, upper = cell.band(n_boot=N_BOOT, seed=BASE_SEED)
        for m in MARKS:
            i = m - 1
            k = int(round(curve[i] * REPEATS))
            e_lo, e_hi = clopper_pearson(k, REPEATS)
            c_lo = 1.0 - (1.0 - p_lo) ** m
            c_hi = 1.0 - (1.0 - p_hi) ** m
            overlap = not (e_hi < c_lo or c_hi < e_lo)
            width = upper[i] - lower[i]
            degenerate += 1 if width == 0.0 else 0
            checked += 1
            outside += 0 if overlap else 1
            say(
                f"{inst.identifier:<20s} {m:>4d} {k:>3d}/{REPEATS:<3d} "
                f"[{e_lo:>9.4f}, {e_hi:>9.4f}] [{c_lo:>9.4f}, {c_hi:>9.4f}] "
                f"{('yes' if overlap else 'NO'):>8s} {width:>11.4f}"
            )
    say("")
    say(f"comparisons made                                  : {checked}")
    say(f"empirical and closed-form intervals that DISAGREE : {outside}")
    say(f"of those comparisons, bootstrap bands of zero width: {degenerate}")
    say(
        "A disagreement here is a harness defect and is reported as one. The "
        "zero-width count is the methodological finding, not a failure: it is why "
        "section 3's bands must be read as estimates of run-to-run spread and not as "
        "coverage statements near a curve's floor or ceiling."
    )

    say.rule("5. AGGREGATE (read section 1 first)")
    say(render_aggregate_table(report))
    say("")
    head = (
        f"{'strategy':<22s} {'aggregate mean P':>17s} {'vs baseline':>12s} "
        f"{'instances lost to baseline':>27s}"
    )
    say(head)
    say("-" * len(head))
    base_agg = report.aggregate_mean_probability(BASELINE)
    for name in report.strategy_names:
        value = report.aggregate_mean_probability(name)
        lost = 0 if name == BASELINE else len(loss_table[name])
        delta = "--" if name == BASELINE else f"{value - base_agg:+.4f}"
        say(f"{name:<22s} {value:>17.4f} {delta:>12s} {lost:>27d}")
    say("")
    lower_agg, upper_agg = report.aggregate_band(BASELINE, n_boot=N_BOOT, seed=BASE_SEED)
    agg_curve = report.aggregate_curve(BASELINE)
    say("Aggregate baseline curve with its stratified bootstrap band:")
    say(f"{'n':>5s} {'aggregate P':>12s} {'band':>22s}")
    say("-" * 42)
    for m in MARKS:
        i = m - 1
        say(f"{m:>5d} {agg_curve[i]:>12.4f} [{lower_agg[i]:>9.4f}, {upper_agg[i]:>9.4f}]")

    say.rule("6. SURROGATE WARM-START ACCOUNTING")
    say(
        "The surrogate strategy spends its first 16 simulations on a Latin hypercube "
        "warm start during which it is not learned at all. The learned part therefore "
        f"has at most {BUDGET} - 16 = {BUDGET - 16} simulations to beat the baseline "
        "with, and on an instance easy enough for the baseline to succeed inside 16 "
        "draws the surrogate never got a turn. Below: how often the surrogate's "
        "violation came during its warm start rather than after it."
    )
    say("")
    head = (
        f"{'instance':<20s} {'found':>6s} {'during warm start':>18s} "
        f"{'after warm start':>17s} {'median sims':>12s}"
    )
    say(head)
    say("-" * len(head))
    for inst in suite():
        cell = report.cell(inst.identifier, "surrogate-guided")
        found = [v for v in cell.first_violations if v is not None]
        during = sum(1 for v in found if v <= 16)
        after = len(found) - during
        median_value = cell.median_simulations
        median = f"> {BUDGET}" if median_value is None else f"{median_value:.1f}"
        say(
            f"{inst.identifier:<20s} {len(found):>6d} {during:>18d} {after:>17d} "
            f"{median:>12s}"
        )

    say.rule("7. RAW FIRST-VIOLATION INDICES")
    say("'none' marks a run that found nothing within the budget.")
    for inst in suite():
        say("")
        say(f"--- {inst.identifier}")
        for name in report.strategy_names:
            cell = report.cell(inst.identifier, name)
            entries = ", ".join(
                "none" if v is None else str(v) for v in cell.first_violations
            )
            say(f"{name:<22s} [{entries}]")

    say.rule("8. COMPUTE")
    say(f"difficulty estimate      : {len(suite()) * DIFFICULTY_DRAWS} simulations, "
        f"{difficulty_seconds:.1f} s")
    searches = len(suite()) * len(STRATEGIES) * REPEATS
    say(f"searches run             : {searches}")
    say(f"budget per search        : {BUDGET} simulations (upper bound; runs stop early)")
    say(f"benchmark wall clock     : {report.wall_clock_seconds:.1f} s")
    say(f"total wall clock         : {total_seconds:.1f} s")
    say(f"available CPU cores      : {len(os.sched_getaffinity(0))} (os.sched_getaffinity)")
    say(
        "Single process, no parallelism, scikit-learn forests fit with n_jobs=1. These "
        "are wall clocks on a shared container, not hardware characteristics, and they "
        "move by 10-20 % between runs. The seeded simulation counts above do not move "
        "at all, which is why they and not the timings are the primary metric."
    )
    say("")
    say("Falsification is one-sided: finding no violation is not evidence of correctness.")
    say.save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

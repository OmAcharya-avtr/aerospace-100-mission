"""The MEASURED learned predictor, against baseline 2 on the same warm-up window.

**No synthetic fallback.** If ``data/traces/`` is absent this script names the
files it looked for and exits non-zero.

Two evaluation axes were asked for and only one of them can be measured.

- **Held-out DUTY CYCLE** -- train on a subset of the campaign's duty cycles and
  predict one never seen. Measurable, and measured below.
- **Held-out AMBIENT** -- STRUCTURALLY UNMEASURABLE in this campaign. There is
  one ambient and no way to vary it without physical access to the board, so
  there is no second point to hold out. It is reported as unmeasurable with that
  reason. It is not dropped and it is NOT approximated by the duty-cycle split,
  which is a different axis: a model can pass one and fail the other, which is
  exactly why the registry summary asks for both separately.

The comparison rule, applied whatever the numbers turn out to be: the learned
margin is measured against baseline 3's run-to-run bootstrap half-width, and a
margin smaller than that half-width is reported NOT DISTINGUISHABLE in either
direction. The expected and acceptable outcome is that the RC baseline is
competitive. Nothing here is retuned until the model wins.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()

from thermalbudget import limits  # noqa: E402
from thermalbudget.baselines import sustained_truth  # noqa: E402
from thermalbudget.predictor import (  # noqa: E402
    DEFAULT_WINDOW_LENGTHS_S,
    FEATURE_NAMES,
    N_MODELS,
    RIDGE_ALPHA,
    build_dataset,
    compare_against_rc,
    leave_one_duty_out,
)
from thermalbudget.provenance import Origin, require_measured  # noqa: E402
from thermalbudget.trace import TraceDataAbsentError  # noqa: E402


def _refuse(out: _bootstrap.Tee, error: Exception) -> int:
    out.rule("MEASURED LEARNED PREDICTOR -- REFUSED")
    out("VERDICT: DATA ABSENT. No measured score was computed and none is reported.")
    out("")
    out(str(error))
    out("")
    out("Nothing was substituted. A held-out-duty score computed from synthetic fixtures")
    out("would be a score about the generator, not about the board, and reporting it here")
    out("would be the exact confusion this product is built to prevent.")
    out.check("the measured campaign is on disk", False, "see the refusal above")
    out("")
    out("The second evaluation axis is unmeasurable whether or not the data arrives:")
    for claim in limits.structurally_unmeasurable_axes():
        out.unmeasurable_check(claim.claim, claim.reason)
    out("")
    out(f"raw output written to {out.save()}")
    return 2


def main() -> int:
    out = _bootstrap.Tee("validate_measured_learned_output.txt")
    try:
        campaign = _bootstrap.require_measured_campaign(
            extra_runs=_bootstrap.REQUIRED_DUTY_RUNS
        )
    except TraceDataAbsentError as error:
        return _refuse(out, error)

    out.rule("MEASURED LEARNED PREDICTOR -- held-out duty cycle, against baseline 2")
    out("INPUT ORIGIN: MEASURED on the board, at power mode "
        f"{limits.POWER_MODE}, one ambient.")
    out("")
    for line in _bootstrap.environment_lines():
        out(line)

    out.rule("1  the dataset, and why its row count overstates its information")
    dataset = build_dataset(
        campaign.soaks(),
        window_lengths_s=DEFAULT_WINDOW_LENGTHS_S,
        equilibrium_window_s=_bootstrap.WINDOW_S,
        threshold_c_per_s=_bootstrap.THRESHOLD_C_PER_S,
    )
    require_measured(dataset.origin, "the learned dataset", n_inputs=len(dataset.runs))
    out(f"  rows        {dataset.n_rows}")
    out(f"  runs        {len(dataset.runs)} -- {list(dataset.runs)}")
    out(f"  duties      {list(dataset.duties)}")
    out(f"  windows     {list(DEFAULT_WINDOW_LENGTHS_S)} s")
    out(f"  features    {len(FEATURE_NAMES)} -- {list(FEATURE_NAMES)}")
    if dataset.excluded:
        for label, reason in dataset.excluded:
            out(f"  EXCLUDED    {label} -- {reason}")
    else:
        out("  EXCLUDED    none")
    out(
        f"  The {dataset.n_rows} rows come from {len(dataset.runs)} runs at "
        f"{len(DEFAULT_WINDOW_LENGTHS_S)} warm-up window lengths each. Rows from one run "
        "are nested inside it and are strongly correlated, so the row count overstates "
        "the information. Every split here is grouped by RUN and the evaluation is "
        "grouped by DUTY; a random-row cross-validation score on this dataset would be "
        "meaningless and is not offered."
    )
    out(f"  model       ridge, alpha {RIDGE_ALPHA:g} declared and not searched, "
        f"{N_MODELS} group-bootstrap fits for the uncertainty")
    out(
        "  alpha is declared rather than tuned because there is no honest held-out set to "
        "tune it on: tuning it against the test duty would make the held-out score "
        "meaningless."
    )

    out.rule("2  baseline 3's run-to-run interval -- the yardstick")
    truth = sustained_truth(
        campaign.soaks(1.0),
        window_s=_bootstrap.WINDOW_S,
        threshold_c_per_s=_bootstrap.THRESHOLD_C_PER_S,
        seed=_bootstrap.SEED_ANALYSIS,
        n_resamples=_bootstrap.RESAMPLES,
    )
    out(
        f"  steady die temperature {truth.die_c.point:.4f} "
        f"[{truth.die_c.ci_low:.4f}, {truth.die_c.ci_high:.4f}] degC, half-width "
        f"{truth.die_c.half_width:.4f} degC over {truth.die_c.n_groups} runs"
    )
    out(
        "  A learned margin smaller than that half-width is inside the run-to-run noise "
        "of the quantity being predicted and is reported as not distinguishable."
    )

    out.rule("3  held-out DUTY CYCLE -- measurable, and measured")
    results = {}
    for kind in ("ridge", "forest"):
        result = leave_one_duty_out(
            dataset, campaign.soaks(), kind=kind, seed=_bootstrap.SEED_ANALYSIS
        )
        verdict = compare_against_rc(result, truth.die_c)
        results[kind] = (result, verdict)
        out("")
        out(f"  model {kind}")
        out(
            f"    pooled over held-out rows: learned MAE {result.learned_mae_c:.4f} degC, "
            f"RC on the SAME warm-up window {result.rc_mae_c:.4f} degC, margin "
            f"{result.margin_c:+.4f} degC"
        )
        out(f"    two-sigma coverage {result.coverage:.4f} (a calibrated band covers ~0.95)")
        for split in result.splits:
            out(
                f"      {split.held_out:<12s} n={split.n_rows:<3d} learned "
                f"{split.learned_mae_c:.4f} degC   RC {split.rc_mae_c:.4f} degC   margin "
                f"{split.margin_c:+.4f} degC"
            )
        for duty, reason in result.skipped:
            out(f"      SKIPPED duty {duty:g}: {reason}")
        out(f"    margin / run-to-run half-width = {verdict.ratio:.4f}")
        out(f"    VERDICT: {verdict.verdict}")
        out(f"    {verdict.statement}")
        out.check(
            f"the {kind} verdict follows the declared rule",
            (verdict.verdict == "NOT_DISTINGUISHABLE") == (verdict.ratio <= 1.0),
            f"ratio {verdict.ratio:.4f}, verdict {verdict.verdict}",
        )

    out("")
    out("  per-row detail for the primary model, so every number above is traceable:")
    primary, _ = results["ridge"]
    for split in primary.splits:
        for label, window_s, truth_c, prediction, sigma, rc in split.rows:
            out(
                f"    {split.held_out:<12s} {label:<16s} window {window_s:5.0f} s  truth "
                f"{truth_c:7.3f} degC  learned {prediction:7.3f} +/- {sigma:.3f} degC  RC "
                f"{rc:7.3f} degC"
            )

    out.rule("4  held-out AMBIENT -- STRUCTURALLY UNMEASURABLE")
    for claim in limits.structurally_unmeasurable_axes():
        out.unmeasurable_check(claim.claim, claim.reason)
    out(
        "  The duty-cycle split above is NOT a substitute for this axis and is not "
        "reported as one. A model can pass a held-out duty cycle and fail a held-out "
        "ambient, which is why the two were asked for separately."
    )
    out(
        f"  What the campaign does record is the idle floor of every run, "
        f"{campaign.idle_floor_c():.3f} degC for idle_pre, so ambient DRIFT across the "
        "campaign is visible even though ambient itself was never varied or instrumented."
    )

    out.rule("5  the origin of every figure in this file")
    out.check(
        "every score here derives from MEASURED traces",
        dataset.origin is Origin.MEASURED and truth.origin is Origin.MEASURED,
        f"dataset {dataset.origin}, baseline 3 {truth.origin}",
    )

    out("")
    out(f"raw output written to {out.save()}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

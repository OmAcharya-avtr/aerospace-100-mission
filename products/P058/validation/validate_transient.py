"""The transient negative control, with a matched stationary baseline.

A transient is a short excursion the channel recovers from. It is not a change,
so every alarm on it is a false alarm. But a raw firing rate on a transient
stream is uninterpretable on its own: a detector at ARL0 = 500 false-alarms
inside any 70-sample window about 13 % of the time whether or not an excursion
happened. This script therefore measures the firing rate on the excursion AND on
a stationary stream with the same seeds, and reports the excess.

Section 4 is the one the spec asks for: every detector's behaviour, including the
ones that fire. They all fire.
"""

from __future__ import annotations

from _harness import SCREENSHOTS, rel, run

REPLICATES = 250
AMPLITUDES = (2.0, 4.0, 8.0)
DURATION = 20


def body(report) -> None:
    from telemdrift.benchmark import (
        DETECTOR_LABELS,
        STANDARD,
        calibrate_all_analytic,
        calibrate_learned_threshold,
        transient_response,
    )
    from telemdrift.detectors import ANALYTIC_DETECTORS
    from telemdrift.learned import build_training_set, train_learned_detector
    from telemdrift.plotting import plot_transient_response

    cfg = STANDARD
    keys = list(ANALYTIC_DETECTORS) + ["learned"]

    report.section("0. Experiment")
    print(f"  transient       +A sigma for {DURATION} samples, then recovery to N(0,1)")
    print(f"  amplitudes      {list(AMPLITUDES)} sigma")
    print(f"  alarm window    {DURATION} + 50 grace = {DURATION + 50} samples from the")
    print("                  excursion start")
    print(f"  replicates      {REPLICATES} seeded streams per detector per amplitude")
    print("  baseline        the same measurement on a stationary stream, same seeds")
    print(f"  operating point all detectors at a measured ARL0 near "
          f"{cfg.target_arl0:.0f} samples")

    report.section("1. Calibration")
    cals = calibrate_all_analytic(cfg)
    for key in ANALYTIC_DETECTORS:
        print(f"  {cals[key].summary()}")
    training = build_training_set(seeds=range(58_001, 58_009))
    model = train_learned_detector(training, n_estimators=100)
    learned = calibrate_learned_threshold(model, cfg)
    print(f"  {learned.summary()}")
    print()
    print(f"  training set: {training.summary()}")
    print("  The learned detector has NOT seen a transient in training. That is")
    print("  deliberate: the transient is held out for all six detectors equally.")
    print("  Section 3 measures what the extra supervision would buy.")

    thresholds = {k: cals[k].threshold for k in ANALYTIC_DETECTORS}
    thresholds["learned"] = learned.threshold

    report.section("2. Firing rate on the transient against a matched baseline")
    results = {}
    for amp in AMPLITUDES:
        print(f"  amplitude +{amp:g} sigma")
        print("    detector        transient  Wilson 95 % CI      baseline   excess")
        for key in keys:
            r = transient_response(key, thresholds[key], amp, DURATION, cfg,
                                   replicates=REPLICATES, model=model)
            results[(key, amp)] = r
            print(f"    {DETECTOR_LABELS[key]:15s} {100 * r['transient_rate']:7.1f}%  "
                  f"[{100 * r['transient_lo']:5.1f}%, {100 * r['transient_hi']:5.1f}%]  "
                  f"{100 * r['baseline_rate']:11.1f}% {100 * r['excess']:+8.1f} pp")
        print()
    base_rates = [results[(k, AMPLITUDES[0])]["baseline_rate"] for k in keys]
    report.check("the matched stationary baseline is nonzero for every detector, so a "
                 "raw transient rate would overstate the effect",
                 all(b > 0.0 for b in base_rates),
                 f"baselines {[round(100 * b, 1) for b in base_rates]} %")

    report.section("3. What training on transients buys the learned detector")
    print("  The same architecture, retrained with transient-spike streams added as")
    print("  negatives, recalibrated to the same ARL0, and re-measured. This is the")
    print("  one advantage a learned detector genuinely has over an analytic one:")
    print("  it can be told what not to fire on. The question is how much it helps.")
    print()
    tr_training = build_training_set(seeds=range(58_001, 58_009), include_transients=True)
    print(f"  training set: {tr_training.summary()}")
    tr_model = train_learned_detector(tr_training, n_estimators=100)
    tr_learned = calibrate_learned_threshold(tr_model, cfg)
    print(f"  {tr_learned.summary()}")
    print()
    print("  variant                      transient rate   excess   ARL1 on +1 sigma step")
    from telemdrift.benchmark import measure_change_response
    from telemdrift.streams import ChangeSpec

    step = ChangeSpec("mean_step", 1.0)
    for label, mdl, thr in (
        ("stationary negatives only", model, learned.threshold),
        ("transients in training", tr_model, tr_learned.threshold),
    ):
        r = transient_response("learned", thr, 4.0, DURATION, cfg,
                               replicates=REPLICATES, model=mdl)
        a1 = measure_change_response("learned", thr, step, cfg, replicates=200, model=mdl)
        print(f"  {label:28s} {100 * r['transient_rate']:13.1f}% "
              f"{100 * r['excess']:+8.1f} pp {a1.arl1:21.1f}")
        results[("learned_" + label, 4.0)] = (r, a1)
    plain_r, plain_a = results[("learned_stationary negatives only", 4.0)]
    tr_r, tr_a = results[("learned_transients in training", 4.0)]
    delta_rate = tr_r["transient_rate"] - plain_r["transient_rate"]
    delta_delay = tr_a.arl1 - plain_a.arl1
    print()
    print(f"  change in transient firing rate: {100 * delta_rate:+.1f} pp")
    print(f"  change in delay on a real +1 sigma step: {delta_delay:+.1f} samples")
    report.finding(
        "training the learned detector on transient negatives changed its transient "
        f"firing rate by {100 * delta_rate:+.1f} pp and its delay on a real +1 sigma "
        f"mean step by {delta_delay:+.1f} samples, both at the same recalibrated ARL0. "
        "This is the only capability in this package that an analytic detector does "
        "not have, and it is measured rather than claimed."
    )
    report.check("the transient-trained variant was measured at the same ARL0",
                 abs(tr_learned.target_error) < 0.30,
                 f"target error {100 * tr_learned.target_error:+.1f} %")

    report.section("4. Every detector's behaviour on the transient, as the spec asks")
    print("  At +4 sigma for 20 samples, with the matched baseline subtracted.")
    print()
    print("  detector        fires on transient   attributable excess   verdict")
    labels, rates, cis = [], [], []
    for key in keys:
        r = results[(key, 4.0)]
        labels.append(DETECTOR_LABELS[key])
        rates.append(r["transient_rate"])
        cis.append((r["transient_lo"], r["transient_hi"]))
        verdict = ("fires on essentially every transient" if r["transient_rate"] > 0.95
                   else "fires on most transients" if r["transient_rate"] > 0.5
                   else "fires on a minority of transients")
        print(f"  {DETECTOR_LABELS[key]:15s} {100 * r['transient_rate']:17.1f}% "
              f"{100 * r['excess']:+21.1f} pp   {verdict}")
    print()
    print("  There is no detector in this package that ignores a transient. The")
    print("  windowed KS test fires least often, and that is not discrimination:")
    print("  it is un-armed for 65.9 % of a stationary stream (see")
    print("  validate_arl_calibration.py section 4), so it is often not watching.")
    report.finding(
        "ALL SIX DETECTORS FIRE ON THE +4 SIGMA TRANSIENT. Attributable excess over "
        "the matched stationary baseline: " + ", ".join(
            f"{DETECTOR_LABELS[k]} {100 * results[(k, 4.0)]['excess']:+.1f} pp"
            for k in keys
        ) + ". None of them is a transient-rejecting detector and the README must not "
        "be read as offering one."
    )

    report.section("5. Figure")
    path = plot_transient_response(
        labels, rates, cis, SCREENSHOTS / "transient_validation.png",
        f"Firing rate on a +4 sigma, {DURATION}-sample transient that is not a change",
    )
    print(f"  wrote {rel(path)}")
    report.check("the transient figure was written", path.exists())


if __name__ == "__main__":
    raise SystemExit(run("validate_transient",
                         "telemdrift 0.1.0 - the transient negative control",
                         body))

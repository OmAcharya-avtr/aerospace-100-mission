"""V3 -- detection: false-alarm rate, delay and ROC for all five methods.

Five detectors on identical held-out data:

============  ==================================================================
chi2_short    sliding chi-squared, 25-sample window
chi2_long     sliding chi-squared, 100-sample window
cusum         four one-sided CUSUMs, one per residual channel and sign
glr           sliding GLR bank matched to the seven fault signatures
learned       random-forest classifier, ``1 - P(none)``
============  ==================================================================

Every threshold is calibrated on 150 dedicated fault-free runs to the same
per-run false-alarm probability of 10 %, so the delays are comparable.  No
held-out run touches a threshold.

Criteria fixed before the run:

* **C1** the measured per-run false-alarm probability on the held-out
  fault-free runs (150 dedicated plus the 30 in the held-out scenario set) is
  within a 95 % Wilson interval of the 10 % calibration target, for every
  method;
* **C2** every method detects at least 95 % of faulted runs within the
  600-sample horizon;
* **C3** measurements, no pass/fail: mean and median detection delay overall
  and per fault class, and the ROC/AUC of every method.

Run: ``python validation/detection_benchmark.py``
"""

from __future__ import annotations

import time
from pathlib import Path

from _support import TARGET_RUN_FAR, Tee, build_campaign

import numpy as np

from fdiscope.evaluate import (
    class_labels,
    evaluate_detection,
    method_names,
    window_scores,
)
from fdiscope.faults import FAULT_CLASSES, FaultType
from fdiscope.metrics import mean_ci, roc_curve, wilson_interval

DETECTION_RATE_FLOOR = 0.95

out = Tee(Path(__file__).with_name("detection_benchmark_output.txt"))
t0 = time.perf_counter()

out("V3 -- detection benchmark: false-alarm rate, delay and ROC")
out("=" * 90)
campaign = build_campaign()
cfg = campaign.cfg
out(f"campaign build {campaign.build_seconds:.1f} s")
out(f"training scenarios {len(campaign.train)}, held out {len(campaign.test)}, "
    f"calibration runs {len(campaign.calib_runs)}, "
    f"held-out fault-free runs {len(campaign.far_runs)}")
out(f"windows: short {cfg.det_window}, long {cfg.iso_window} samples; "
    f"design alpha {cfg.alpha:g}; CUSUM design mu {cfg.cusum_mu:g}")
out(f"classifier: {campaign.features.shape[0]} training rows x "
    f"{campaign.features.shape[1]} features")
out("All data SIMULATED.")
out("")

out("-- thresholds")
out(f"{'method':>12} {'design formula':>16} {'calibrated':>14}   note")
for name in method_names():
    design = campaign.design_thresholds.get(name)
    design_s = f"{design:.4f}" if design is not None else "none"
    note = (
        "closed form, needs no data"
        if design is not None
        else "NO closed form: needs fault-free data to be usable at all"
    )
    out(f"{name:>12} {design_s:>16} {campaign.matched_thresholds[name]:>14.4f}   {note}")
out("")
out("The two columns above are the first result.  The chi-squared and CUSUM")
out("thresholds follow from a distribution and a target rate with no data; the")
out("GLR bank and the classifier have no such formula and cannot be operated")
out("without a fault-free calibration set.  Everything below is measured at the")
out("calibrated thresholds so that the delays are comparable.")
out("")

# The false-alarm rate is measured on a dedicated held-out fault-free set as
# well as on the fault-free members of the held-out scenario set, because 30
# runs cannot resolve a 10 % rate: a 95 % Wilson interval on 30 runs is about
# 20 points wide, which would make C1 unfalsifiable in either direction.
results = evaluate_detection(
    campaign.test + campaign.far_scenarios,
    campaign.test_runs + campaign.far_runs,
    cfg,
    campaign.bank,
    campaign.classifier,
    campaign.matched_thresholds,
)

out("-- C1  false-alarm rate on held-out fault-free runs")
out(f"{'method':>12} {'per-run':>10} {'95 % Wilson':>22} {'target':>8} "
    f"{'per-sample':>12} {'alarming/total':>18}  verdict")
c1_pass = True
for name in method_names():
    m = results[name]
    iv = wilson_interval(m.far_runs[0], m.far_runs[1])
    ok = iv.contains(TARGET_RUN_FAR)
    c1_pass = c1_pass and ok
    out(
        f"{name:>12} {iv.point:>10.4f} "
        f"{'[' + format(iv.low, '.4f') + ', ' + format(iv.high, '.4f') + ']':>22} "
        f"{TARGET_RUN_FAR:>8.2f} {m.far_per_sample:>12.6f} "
        f"{str(m.far_samples[0]) + '/' + str(m.far_samples[1]):>18}  "
        f"{'PASS' if ok else 'FAIL'}"
    )
out(f"C1: {'PASS' if c1_pass else 'FAIL'}")
out("")

out("-- C2/C3  detection rate and delay, held-out faulted runs")
out(f"{'method':>12} {'det rate':>9} {'censored':>9} {'mean delay':>11} "
    f"{'95 % CI':>20} {'median':>8} {'p90':>8}  verdict")
c2_pass = True
for name in method_names():
    m = results[name]
    ok = m.detection_rate >= DETECTION_RATE_FLOOR
    c2_pass = c2_pass and ok
    iv = mean_ci(m.delays)
    out(
        f"{name:>12} {m.detection_rate:>9.4f} {m.censored:>9} {iv.point:>11.2f} "
        f"{'[' + format(iv.low, '.2f') + ', ' + format(iv.high, '.2f') + ']':>20} "
        f"{np.median(m.delays):>8.1f} {np.percentile(m.delays, 90):>8.1f}  "
        f"{'PASS' if ok else 'FAIL'}"
    )
out(f"C2: {'PASS' if c2_pass else 'FAIL'}  (detection rate >= {DETECTION_RATE_FLOOR})")
out("Delays are in samples at dt = 0.1 s; a censored run is one that never")
out("alarmed within 600 samples and is excluded from the mean, not zero-filled.")
out("")

out("-- C3  mean detection delay per fault class [samples]")
faulted_classes = [f for f in FAULT_CLASSES if f is not FaultType.NONE]
header = f"{'fault class':>32}" + "".join(f"{n:>13}" for n in method_names())
out(header)
for fault in faulted_classes:
    row = f"{fault.value:>32}"
    for name in method_names():
        d = results[name].delays_for(fault)
        row += f"{np.nanmean(d) if d.size else float('nan'):>13.1f}"
    out(row)
out("Censored runs are NaN and are excluded from the class mean; the censored")
out("counts per method are in the C2 table above.")
out("")

out("-- C3  ROC over window-level scores")
pos, neg = window_scores(campaign.test, campaign.test_runs, cfg, campaign.bank, campaign.classifier)
out(f"positives {len(next(iter(pos.values())))} windows at offsets {cfg.roc_offsets} "
    f"after onset; negatives {len(next(iter(neg.values())))} pre-onset windows")
out(f"{'method':>12} {'AUC':>9} {'TPR@FPR=0.10':>14} {'TPR@FPR=0.01':>14} "
    f"{'TPR@FPR=0.001':>15}")
curves = {}
for name in pos:
    curve = roc_curve(pos[name], neg[name], name)
    curves[name] = curve
    out(
        f"{name:>12} {curve.auc:>9.4f} {curve.tpr_at_fpr(0.10):>14.4f} "
        f"{curve.tpr_at_fpr(0.01):>14.4f} {curve.tpr_at_fpr(0.001):>15.4f}"
    )
best = max(curves.values(), key=lambda c: c.auc)
out(f"highest AUC: {best.label} ({best.auc:.4f})")
out("")

out("-- C3  ROC restricted to each fault class (positives of that class only)")
labels = class_labels()
faulted = [s for s in campaign.test if s.label is not FaultType.NONE]
n_off = len(cfg.roc_offsets)
out(f"{'fault class':>32}" + "".join(f"{n:>13}" for n in curves))
for fault in faulted_classes:
    mask = np.repeat([s.label is fault for s in faulted], n_off)
    row = f"{fault.value:>32}"
    for name in curves:
        sub = pos[name][mask]
        row += f"{roc_curve(sub, neg[name]).auc:>13.4f}"
    out(row)
out("")

out("=" * 90)
out(f"V3 summary: C1 {'PASS' if c1_pass else 'FAIL'}, C2 {'PASS' if c2_pass else 'FAIL'}, "
    "C3 measurements above")
out(f"wall time {time.perf_counter() - t0:.1f} s")
out.save()

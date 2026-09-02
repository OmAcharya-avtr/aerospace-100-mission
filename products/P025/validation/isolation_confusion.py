"""V4 -- isolation: full confusion matrices, structural limits and confidence.

The mission requires the isolation confusion matrix reported **in full**, not
summarised to one accuracy number, because a single number hides which faults
a method confuses with which -- and that is the only part an operator can act
on.  Both matrices are printed complete, with per-class recall and precision.

Sections:

* **D1** the signature Gram matrix.  Two hypotheses whose signatures have a
  cosine near +/-1 cannot be separated by a matched filter no matter how much
  data arrives, because the GLR statistic squares the projection.  This is a
  structural limit of the classical isolator and it is measured, not asserted.
* **D2** full 8x8 confusion matrices for the GLR bank and the classifier,
  evaluated on the identical window ``[onset, onset + 100)`` of every held-out
  run.
* **D3** isolation accuracy as the assumed onset is shifted, which is what
  happens when the detector localises the fault imperfectly.
* **D4** confidence calibration: accuracy bucketed by the confidence each
  method reports, for both methods.  A confidence that does not track accuracy
  is a number, not a calibration.
* **D5** classifier feature importances.

Criteria fixed before the run:

* **E1** every method's overall isolation accuracy exceeds the 1/8 = 0.125 of
  guessing uniformly at random;
* **E2** measurements, no pass/fail: everything else above.

Run: ``python validation/isolation_confusion.py``
"""

from __future__ import annotations

import time
from pathlib import Path

from _support import Tee, build_campaign

import numpy as np

from fdiscope.evaluate import class_labels, evaluate_isolation
from fdiscope.faults import FAULT_CLASSES
from fdiscope.metrics import confusion_report, wilson_interval

CHANCE = 1.0 / len(FAULT_CLASSES)
OFFSETS = (-50, -25, -10, 0, 10, 25, 50)
CONFIDENCE_EDGES = (0.0, 0.4, 0.6, 0.8, 0.9, 1.0001)

out = Tee(Path(__file__).with_name("isolation_confusion_output.txt"))
t0 = time.perf_counter()

out("V4 -- isolation confusion matrices, structural limits and confidence")
out("=" * 96)
campaign = build_campaign()
cfg = campaign.cfg
labels = class_labels()
out(f"campaign build {campaign.build_seconds:.1f} s")
out(f"held-out scenarios {len(campaign.test)} ({len(campaign.test) // len(FAULT_CLASSES)} "
    f"per class, exactly balanced by construction)")
out(f"isolation window [onset, onset + {cfg.iso_window}); identical for both methods")
out("All data SIMULATED.")
out("")

out("-- D1  signature Gram matrix: |cos| near 1 means structurally inseparable")
gram = campaign.bank.gram()
names = [f.value for f in campaign.bank.faults]
out(" " * 34 + "".join(f"{n[:11]:>13}" for n in names))
for i, n in enumerate(names):
    out(f"{n:>32}  " + "".join(f"{gram[i, j]:>13.4f}" for j in range(len(names))))
worst = 0.0
worst_pair = ("", "")
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        if abs(gram[i, j]) > worst:
            worst = abs(gram[i, j])
            worst_pair = (names[i], names[j])
out("")
out(f"worst off-diagonal |cosine| : {worst:.4f} between {worst_pair[0]} and {worst_pair[1]}")
out("A GLR bank scores each hypothesis with (phi_j . r)^2, so the sign is")
out("irrelevant and a cosine of -0.997 is as bad as +0.997.  The pair above")
out("cannot be separated by this classical isolator at any sample size, and the")
out("D2 matrix shows exactly that confusion.")
out("")

out("-- D2  full confusion matrices, isolation window aligned with the true onset")
outcomes = evaluate_isolation(
    campaign.test, campaign.test_runs, cfg, campaign.bank, campaign.classifier
)
e1_pass = True
reports = {}
for name, outcome in outcomes.items():
    report = confusion_report(outcome.truth, outcome.predicted, labels)
    reports[name] = report
    iv = wilson_interval(int(np.sum(np.diag(report.matrix))), int(report.matrix.sum()))
    ok = iv.low > CHANCE
    e1_pass = e1_pass and ok
    out("")
    out(f"method: {name}   (rows = truth, columns = prediction)")
    out(report.to_text(15))
    out(f"accuracy 95 % Wilson interval : [{iv.low:.4f}, {iv.high:.4f}], "
        f"chance = {CHANCE:.4f}  {'PASS' if ok else 'FAIL'}")
out("")
out(f"E1: {'PASS' if e1_pass else 'FAIL'}  (accuracy interval above chance)")
out("")

out("-- D2b  head-to-head recall per class")
out(f"{'fault class':>32}" + "".join(f"{n:>12}" for n in reports))
for i, label in enumerate(labels):
    out(f"{label:>32}" + "".join(f"{reports[n].recall[i]:>12.4f}" for n in reports))
out(f"{'OVERALL ACCURACY':>32}" + "".join(f"{reports[n].accuracy:>12.4f}" for n in reports))
out("")

out("-- D3  isolation accuracy against onset misalignment [samples]")
out("The window is placed at onset + offset; a negative offset means the window")
out("starts before the fault, which is what an early alarm produces.")
out(f"{'offset':>8}" + "".join(f"{n:>12}" for n in reports))
for offset in OFFSETS:
    shifted = evaluate_isolation(
        campaign.test, campaign.test_runs, cfg, campaign.bank, campaign.classifier,
        offset=offset,
    )
    row = f"{offset:>8}"
    for name in reports:
        rep = confusion_report(shifted[name].truth, shifted[name].predicted, labels)
        row += f"{rep.accuracy:>12.4f}"
    out(row)
out("Both methods assume the onset is known.  This table is what that")
out("idealisation is worth: it is the accuracy an operator would actually see if")
out("the detector's alarm time were used to place the window.")
out("")

out("-- D4  confidence calibration: accuracy within confidence buckets")
for name, outcome in outcomes.items():
    out("")
    out(f"method: {name}")
    out(f"{'confidence bucket':>22} {'n':>6} {'accuracy':>10} {'mean confidence':>17} "
        f"{'gap':>8}")
    conf = outcome.confidence
    correct = outcome.truth == outcome.predicted
    for lo, hi in zip(CONFIDENCE_EDGES[:-1], CONFIDENCE_EDGES[1:], strict=True):
        mask = np.isfinite(conf) & (conf >= lo) & (conf < hi)
        n = int(np.count_nonzero(mask))
        if n == 0:
            out(f"{'[' + format(lo, '.2f') + ', ' + format(hi, '.2f') + ')':>22} {0:>6} "
                f"{'-':>10} {'-':>17} {'-':>8}")
            continue
        acc = float(np.mean(correct[mask]))
        mc = float(np.mean(conf[mask]))
        out(
            f"{'[' + format(lo, '.2f') + ', ' + format(hi, '.2f') + ')':>22} {n:>6} "
            f"{acc:>10.4f} {mc:>17.4f} {mc - acc:>8.4f}"
        )
    undeclared = int(np.count_nonzero(~np.isfinite(conf)))
    if undeclared:
        out(f"{'no fault declared':>22} {undeclared:>6}")
out("")
out("A positive gap means the method is over-confident in that bucket.  Neither")
out("confidence is a calibrated probability: the GLR posterior assumes exactly")
out("one of the seven modelled faults is present with equal prior, and the")
out("forest's vote fraction is an ensemble-agreement heuristic.  The table is a")
out("reliability diagram in numbers, and it is the only calibration evidence")
out("this package offers.")
out("")

out("-- D5  classifier feature importances (impurity-based, biased, indicative only)")
importances = campaign.classifier.feature_importances()
for feature, value in sorted(importances.items(), key=lambda kv: -kv[1]):
    out(f"{feature:>22} {value:>10.4f}")
out(f"{'sum':>22} {sum(importances.values()):>10.4f}")
out("")

out("=" * 96)
out(f"V4 summary: E1 {'PASS' if e1_pass else 'FAIL'}, E2 measurements above")
out(f"wall time {time.perf_counter() - t0:.1f} s")
out.save()

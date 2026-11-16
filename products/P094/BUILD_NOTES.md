# P094 HilTimingDiff — build notes (running state)

Rolling note kept for resumption after a usage-limit termination. Not a
deliverable; it records what is done, what is next, and what has been verified.

## Verified facts (measured this session)

- md5 of the four committed board programs matches the coordinator's list
  exactly: campaign.py `8c137fc2c31c48c2e8f16d1ae1f206d3`, staged_exec.py
  `0f654894ea7dc3c6ec5d3880e9150459`, pat_pipeline.py
  `80ba9015b54fa8e95eae96b2c93eb9c2`, verify_outputs.py
  `ac6b6aa2b486ea7d9a227d1598dc8293`.
- Traces present and complete: stages_pass.csv 3240 data rows, stages.csv
  270000 data rows, runs.csv 9, env.csv 94, calibration.csv 22000, raw/ 9 .npy,
  campaign.log ends `CAMPAIGN_COMPLETE elapsed 734.0 s drift_ratio 1.075241`.
- **calibration.csv holds ELEVEN blocks, not ten.** `floor_start`,
  `floor_00`..`floor_08`, `floor_end`, 2000 reps each = 22000 rows. SPEC §8 and
  the dispatch brief both say "ten blocks". The campaign program emits an
  explicit leading `floor_start`, then one block before each of the nine cells,
  then `floor_end`. The count in the specification undercounts by one. Reported
  as a discrepancy, not corrected in SPEC.
- Provisional divergence (ns per declared op, from committed traces only, free
  of any host calibration): ingest 0.46875, background 1.220703, threshold
  21.651754, track 47.644444, gate 95.190288, centroid 522.592200.
  **Spread max/min = 1114.86**, argmax `centroid`, argmin `ingest`.
  §7's primary expectation (spread > 3) is therefore CONFIRMED, by a wide
  margin. §7's stated MECHANISM is only partly right: it puts `threshold` in
  the cheap vectorised group, and `threshold` measures 21.65 ns per declared op
  against `background`'s 1.22, because `extract_candidates` contains a
  Python-level loop over candidate groups. Must be reported.
- Container simulation mode runs the committed pipeline unmodified: w64 sparse
  measures ~198 us/frame here against ~730 us/frame on the board.
- Analysis host: x86_64, Python 3.13.18, numpy 2.5.3, 2 cores.

## Design decisions

- `R[s]` levels depend on the host-calibrated `ns_per_op`; the SPREAD, the
  Kendall tau-b, the permutation and argmax/argmin are all invariant under it.
  The ns_per_op-free form `ns per declared op` is published as the exactly
  reproducible table; `R[s]` is published with the `ns_per_op` it used.
- No git command is run by this build agent or by any script it executes: the
  coordinator forbids it. `validate_tree_hygiene.py` therefore uses an explicit
  pattern matcher over the two relevant .gitignore files and says so; the
  authority is `git check-ignore -v` at gate time.

## Progress

- [x] read build guide, SPEC, README standard, P093
- [x] scratch exploration of the traces
- [x] src/hiltimingdiff (sources, traces, bootstrap, divergence, simulate,
      baselines, drift, frequency, gate, report, __main__)
- [x] tests (~800 collected, all passing)
- [x] examples (3) + screenshots (3)
- [x] validation scripts (14) + VALIDATION.md
- [x] README, pyproject, LICENSE, CHANGELOG, CITATION, CI, gitignore
- [x] PyPI probe: hiltimingdiff HTTP 404, numpy and pandas HTTP 200
- [x] dependency range sweep, both ends, suite 1169 passed at each, deterministic
      validation outputs byte-identical
- [x] final re-run of every quoted command

## Measured headline (full run, 12 passes, 2026-11-16)

- spread 1114.8633602088898 [1113.763384, 1124.917784] -> SPEC §7 primary CONFIRMED
- tau-b 0.2 [0.2, 0.2]; permutation (3, 0, 1, 5, 4, 2); 5 of 6 stages change rank
- argmax centroid (0.5806 of frame time), argmin ingest (0.00529)
- baseline 2 held-out error 0.1599 [measured 0.1481..0.1701 over repeats] ->
  SPEC §7 secondary REFUTED (predicted > 0.5)
- baseline 1 mean relative error 2.6268, 17x worse than baseline 2
- platform hw/sim ratio spread 1.58 (3.13..4.94): the PLATFORM is near-uniform,
  the DECLARED MODEL is not
- drift: bracket 1.0752408447278534, fitted 1.0682256407994783, linear r2 0.7269,
  only 4 of 10 adjacent steps increase, largest single step 1.0497
- drift-adjusted spread 1134.17 (+1.73 per cent on the raw)
- frequency covariation: pooled r2 <= 0.008537 (track), within-cell <= 0.004570
  (threshold); a measured NULL with a stated sampling limit
- confounding: corr(order_index, W) = 0.5011614417507341 Pearson /
  0.5809475019311126 Spearman on tie-averaged midranks. NOT near zero. SPEC §8's
  decorrelation claim is only partly supported. (The Spearman figure originally
  recorded here, 0.5333, was WRONG: ordinal ranks on tied data. Corrected
  2026-11-16 after the coordinator's independent check; see VALIDATION.md.)
- gate: all five branches fire in all nine cells; 810000 frames, accept 174420
- DEFECT FOUND AND FIXED: kendall_tau_b decided concordance on the product of
  differences, which underflows to 0.0 for differences near 1e-163, returning
  tau 0.0 where the answer is 1.0. Fixed to a sign product. Counterexample
  pinned in tests/test_properties.py. Measured tau unchanged at 0.2.
- WEAK: baseline 2's verdict is frame-stack dependent (0.48..0.55 at 150 frames).

## Additional findings from the build (2026-11-16, later)

- pandas installed here is 3.0.5, NOT 2.3.x; the declared range was corrected to
  pandas>=3.0,<3.1 before the range sweep.
- Second and third floating-point counterexamples found by Hypothesis and pinned:
  `ordering` is not invariant under an additive offset (3.0 + -1.18e-38 == 3.0),
  and tau-b's invariance under a positive rescaling needs the rescaling to
  preserve distinctness (0.5 * 5e-324 underflows to 0.0).
- One test was load-sensitive and was replaced with a comparative claim:
  the short-stack shift in the hw/sim ratio of the four non-gate stages is
  smaller than track's, rather than being bounded by an absolute 0.25.
- probe_working_directories: 12 scripts x 4 directories = 48 executions, 0
  disagreements, control fires.

## Final state (2026-11-16)

- Suite: 1173 tests, 0 failures, 0 errors, 0 skipped, counted from junit.xml and
  cross-checked against testcase elements. Same count at both dependency ends.
- ruff clean under 0.16.9 (path) and 0.17.5 (module) over src/ tests/ examples/
  validation/.
- Fourteen validation scripts, every one exits 0; two carry a FALSIFIED check
  pinned to its measured value.
- Three examples, three screenshots, three committed _output.txt.
- Mermaid diagram validated (valid: true, flowchart).
- README's quoted `python -m hiltimingdiff gate` block byte-matches the live run.
- junit.xml present but ignored by the repository root's `**/junit.xml`.
- Cache directories (.ruff_cache, .hypothesis, __pycache__) exist in the tree
  from this session's own runs; all are ignored and none would reach a clone.

## Correction after coordinator review (2026-11-16)

- DEFECT: `drift.confounding` computed Spearman with `argsort(argsort(x))`,
  an ordinal rank, on `W` which has five tied 64s. Published 0.5333333333333333;
  correct tie-averaged-midrank value is 0.5809475019311126, matching
  `scipy.stats.spearmanr` exactly. Pearson was unaffected (0.5011614417507341).
- Worse than a wrong constant: the ordinal form is not permutation-invariant.
  Same data gives 0.85 in execution order and 0.5333333333333333 in the
  alphabetical order the groupby produces. Both reproduced and pinned.
- Fixed with `midranks()` and `spearman_rho()` in `src/hiltimingdiff/drift.py`,
  dependency-free; scipy stays in the test extra.
- Finding UNCHANGED in direction and STRENGTHENED in magnitude.
- Files regenerated after the fix: all 14 validation outputs (every script
  re-run, every one exit 0), examples/drift_and_clock_output.txt, VALIDATION.md,
  README.md, CHANGELOG.md, BUILD_NOTES.md.
- Suite after the fix: 1194 tests, 0 failures, 0 errors, 0 skipped; 1194 at both
  dependency ends. 20 new tests cover midranks and Spearman's rho.
- Every remaining `0.5333` in a tracked file is a deliberate record of the
  superseded value, not a current figure.

## Second correction after coordinator review (2026-11-16)

- DEFECT: `test_ordering_is_invariant_under_a_positive_rescaling` asserted a
  property false in float64; intermittent, 1 failure in 6 coordinator runs then
  5 clean. A positive rescaling cannot invert an order but can destroy
  distinctness. Old `assume` guarded the INPUT; the loss is in the OUTPUT.
- THREE routes, not one. The coordinator named overflow at factor 1e308, which
  this file's strategies cannot generate (factor <= 1e6, values <= 1e6). The
  route that actually fired is a plain ROUNDING COLLISION:
  [-811260.5389416472, 1000000.0, 999999.9999999999] x 292915.2677047961 ->
  the last two round to the same float, all finite. Underflow is the mirror.
- Fixed by guarding finiteness AND distinctness of the rescaled array; same
  finiteness guard added to the tau-b rescaling property. Both repaired
  properties: 30000 examples each, 0 falsifications, 0 rejections.
- 4 counterexamples pinned + 1 guard-rejection test + 1 positive case.
- Audit of every other invariance property: one more of the same shape
  (linear_fit on an exact line), safe by construction via the abs=1e-6 floor;
  reason now pinned. Writing that pin exposed my own bit-exact overclaim
  (fit.slope == 0.0; polyfit leaves 7.462276454104577e-18), corrected.
- NO published figure affected: `ordering` is applied directly to medians
  (3840..421630 ns) and ops (64.5..16384), ~300 orders from either float edge,
  closest neighbours a third apart; no rescaling in the published path. Asserted
  by 2 new tests across ns_per_op from 1e-6 to 1e6: bit-identical outputs.

## Third defect, found by me while timing the probe (2026-11-16)

- `probe_working_directories.py` was OVERWRITING all 12 committed validation
  outputs with its own reduced HILTIMINGDIFF_FAST=1 runs (1 pass / 200
  resamples instead of 12 / 4000). Every committed output carried the FAST
  marker when found.
- Fixed with HILTIMINGDIFF_OUTPUT_DIR redirecting Tee to a temp dir; the probe
  now VERIFIES the redirection (12 of 12 in the sink, 0 committed FAST headers).
- That verification scan then had the self-reporting-scanner defect: the probe's
  own output names the marker in prose. Excluded by name and matched on the
  environment-header form. Probe run twice in a row: second agrees with first.
- Probe wall time 493 s alone (691 s contended, over the 600 s gate timeout) ->
  232 s and 235 s, at UNCHANGED coverage (12 scripts x 4 directories = 48
  executions). 268 s of the old 493 s was validate_approx_audit running the full
  suite twice, four times over; it now narrows to one file in FAST mode only.
- Final: 1209 tests, 0 failures; all 14 validation scripts exit 0; ruff clean
  under 0.16.9 and 0.17.5.

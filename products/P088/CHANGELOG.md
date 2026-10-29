# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-10-29

First release. Status: TESTING. Validation level 2, research grade.

### Added

- `tmtcdiff.model`: `Parameter`, `CommandArgument`, `Command` and `Dictionary`,
  a deliberately small subset of a telemetry and command database restricted
  to the fields whose change can alter meaning without altering a name.
  Enumerations are canonicalised by code and command arguments by position, so
  the order of a file is never semantic.
- `tmtcdiff.changes`: the 21-class taxonomy and the three severity tiers —
  11 silent classes, 7 loud, 3 advisory — with `rank_changes`, which puts the
  silent tier first.
- `tmtcdiff.diff`: `diff_dictionaries`, returning a ranked `DiffResult` and a
  `DiffLimits` record stating what the diff could not determine: incomparable
  fields, ambiguous rename candidates, and the numeric detection floor.
- `tmtcdiff.rename`: structural-similarity rename detection with two matching
  strategies, `greedy` (the baseline) and `optimal` (rectangular assignment),
  and `signature_collision_bound`, a counting bound derived independently of
  the matcher.
- `tmtcdiff.corpus`: reference extraction from procedures, scripts and
  configuration files, including references constructed at run time, which are
  extracted and marked undecidable rather than dropped or guessed at.
- `tmtcdiff.impact`: `assess_impact`, returning ranked findings and a
  `CoverageVerdict` — a first-class output naming what the assessment could
  not determine about the corpus it was given.
- `tmtcdiff.parse`, `tmtcdiff.synth`, `tmtcdiff.metrics`, `tmtcdiff.report`,
  and a CLI with `info`, `taxonomy`, `diff`, `impact`, `demo`, `sweep` and
  `negative-control`.
- Eight validation scripts with their raw output, nine example figures, and
  608 tests including Hypothesis property tests.

### Measured and reported as found

- **Per-class detection is perfect in the isolated regime**: precision 1.0000
  and recall 1.0000 on 420 injected changes across all 21 classes, 2 sizes and
  10 seeds, with 0 false positives.
- **The combined regime is worse, and the cause is named.** With every class
  injected into the same pair, micro-averaged precision is 0.9665 and recall
  0.9619. `command_removed` recall falls to 0.7000 and `command_renamed`
  precision to 0.7692, because an unrelated removal and an unrelated addition
  in the same pair can be paired as a rename.
- **Rename detection's false-match rate was measured, not assumed**, over a
  grid of 12 thresholds, 5 dictionary sizes, 12 seeds and both strategies. It
  falls from 0.2258 at threshold 0.40 and n=20 to exactly zero at 0.80 at
  every size. Recall holds at 1.0000 up to 0.85 and falls to 0.7208 at 0.95.
  The shipped threshold 0.80 is the value a declared criterion selected from
  that grid.
- **The greedy baseline and the optimal assignment are indistinguishable at
  the selected threshold**: 924 correct and 0 false for both. Across the whole
  grid they never differ by more than 2 false matches in a cell, in both
  directions. The optimal matcher remains the default because it cannot be
  worse by construction, not because it measured better. It did not.
- **The negative control holds**: a document reordered and respelled across
  its entries, keys, enumerations and argument declarations differs on 2526
  text lines at 60 parameters and produces 0 semantic changes, on all 24
  swept pairs. The control on the control also holds: 66 of 66 injected
  silent changes are still detected through the same reordering.
- **One tier assignment is an assumption, not a measurement.**
  `argument_added` is classified loud on the assumption that a command system
  rejects a call missing an argument. Measured: the command still resolves and
  every pre-existing argument is still present on all 20 injections, so
  nothing in this package confirms it. If an uplink system defaults the
  missing argument instead, the class is silent and is ranked one tier too low.
- **The detection floor on a relative scale change tracks the configured
  tolerance exactly**: 9.999779e-13 at the default 1e-12, found by bisection
  rather than read off the configuration.
- **Five independently derived bounds and identities hold** on every swept
  pair: cardinality conservation, the signature-difference identity, the
  counting bound on structurally indistinguishable matches, the prediction of
  missed renames from the score distribution, and union bounds on the impact
  findings.

### Defects found during the build and pinned as regressions

- Removing a command from the generator's working list renumbered the used
  index set **with the removed index still in it**, marking an untouched
  command as used. Every all-classes injection ran out of commands one
  injection early. Found by `validation/validate_change_classes.py`.
- The first generator gave every parameter a globally unique bit offset, so
  every structural signature was unique, every rename was trivially separable
  and the measured rename false-match rate was exactly zero at every threshold
  from 0.40 to 0.95. That is a broken experiment rather than a result. Offsets
  now wrap inside each packet and collisions are measured: 24 of 400
  parameters share a signature at the largest swept size.
- Every enumeration label shared by two parameters was reported as something
  the assessment could not determine, even with no corpus reference to it, so
  a fully covered corpus reported itself as undeterminable. Ambiguous labels
  are now reported only when the corpus references them.
- A substring scan for the word "safe" in rendered output fired on the
  generated enumeration label `SAFE_MODE`. The scans use word boundaries.
- A character-level `difflib` ratio over a 300-parameter document did not
  finish inside the validation script's budget. The ratio is computed over
  lines.

### Known limitations

See `README.md`. The dominant ones: every number is measured against a seeded
synthetic generator and not against any flight database; reference extraction
is syntactic and line-based, so a corpus reaching the dictionary another way
is under-extracted; a reference constructed at run time cannot be resolved and
is reported as undecidable; and an empty impact report is never a statement
that an upgrade is safe.

[0.1.0]: https://github.com/OmAcharya-avtr/tmtcdiff/releases/tag/v0.1.0

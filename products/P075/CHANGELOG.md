# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## 0.1.0 - 2026-10-25

First release. Research-grade software; not flight-qualified, not certified,
not approved for operational aerospace use. A reliability number is not a
safety case.

### Added

- `reliablock.diagram`: the declared reliability block diagram as a checkable
  artifact, with YAML round-tripping and a validation pass that rejects nine
  classes of declaration rather than defaulting them, including an undeclared
  standby switching reliability and a common-cause beta factor on a group that
  is not redundant.
- `reliablock.distributions`: exponential, Weibull, lognormal and
  fixed-probability failure models, all expressed as cumulative hazard
  functions, with the inverse hazard for sampling.
- `reliablock.numerics`: reliability arithmetic carried in cumulative-hazard
  and unreliability space. Measured in `validation/validate_precision.py`:
  the complementary route `1 - prod(1 - q_i)` returns exactly zero for 10 of
  18 measured parallel cases where the answer is positive, and loses 4.66
  significant figures on a ten-block series chain at `q = 1e-12`.
- `reliablock.quantify`: closed forms for series, parallel, k-of-n, standby
  with a declared switching reliability and warm-standby hazard fraction,
  shared load, and a general two-terminal network by pivotal decomposition.
- `reliablock.commoncause`: the beta-factor model applied by rewriting the
  diagram, so the with-and-without comparison runs one quantifier over two
  diagrams rather than two code paths over one diagram. Both limits are exact:
  `beta = 0` reproduces the independent figure bit for bit and `beta = 1`
  reduces a group to a single member to 1e-14 relative.
- `reliablock.structure`: minimal cut sets and path sets by structural
  recursion, with exhaustive `2**n` state enumeration as the independent
  cross-check. The two agree on all 13 diagrams measured, over 147712 states.
- `reliablock.importance`: Birnbaum, criticality and Fussell-Vesely, the last
  computed exactly rather than by the rare-event approximation, which was
  measured to overestimate by up to 0.127 on the diagrams in
  `validation/validate_importance_output.txt`.
- `reliablock.bounds`: Esary-Proschan min-cut and min-path bounds and the
  first two Bonferroni orders, as independently derived checks that catch
  catastrophic cancellation a same-expression property test cannot.
- `reliablock.bridge`: the five-component bridge network, with its closed form
  `R = 2p^5 - 5p^4 + 2p^3 + 2p^2` derived in full in that module and verified
  five independent ways. No page number or published figure is quoted for it.
- `reliablock.montecarlo`: seeded simulation with Wilson score intervals and a
  budget sized from a declared target half-width.
- `reliablock.mission`: phased mission profiles with per-block duty cycles and
  stress factors, exact for an exponential block and an explicitly named
  approximation otherwise.
- `reliablock.report` and a `python -m reliablock` CLI with eight subcommands.
- 454 tests, 135 validation checks of which 1 fails and is published, 6
  example figures, 14 validation scripts.

### Deliberately not added

- **Any learned or fitted component.** For a declared diagram with declared
  distributions the reliability has a correct answer, so a model fitted to
  approximate the arithmetic could only be worse than it, and there is nothing
  in the problem to learn. `AI_ENABLED` is `False` and
  `validation/validate_cli.py` asserts both that flag and the absence of any
  machine-learning import in `src/`.
- Distribution fitting, accelerated life testing and repairable-system
  analysis. The `reliability` package does those properly and the README says
  to use it for them.
- The multiple-Greek-letter and alpha-factor common-cause models.
  `validation/validate_commoncause_output.txt` measures where their absence
  matters: for a k-of-n group with `k >= 2` the beta factor stops being
  conservative above a member cumulative hazard this repository locates by
  bisection, at 0.302944 for 2-of-3 and 0.042978 for 5-of-6 at `beta = 0.1`.

### Known failing check

- `validation/validate_mission.py` declares and **fails** the check that the
  cumulative-exposure ageing model agrees with calendar-time integration of a
  scaled hazard rate for a non-exponential block. Measured worst divergence
  1852.73 percent, at Weibull shape 3.5 under a stress factor of 4; 4 of 20
  cases agree, and all four are the exponential ones. The two models answer
  different physical questions and the check is published rather than removed.

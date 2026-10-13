# Finding — unbounded upstream dependencies break published products

**Opened and closed 2026-10-13.** Found by executing `scripts/cold_clone_check.py`,
which was written 2026-10-11 and had never been run. Check A (untracked
deliverables) was clean on all 60 products in 0.3 s. Check B (cold clone plus the
product's own suite) returned **four findings out of sixty**, and a fifth product
was caught by a follow-up probe.

Nothing here is a missing deliverable, which is what the script was written to
catch after P060. Every real finding has the same shape: **the product declares a
dependency floor with no ceiling, upstream released an incompatible version, and
a new user now resolves to it.**

## What was measured

Container: Python 3.13, 2 cores. Full suites, counted from junit XML.

| Product | Status | Symptom with what a new install resolves to today |
|---|---|---|
| P001 BeamTwin | PUBLISHED | 4 of 251 **error** under scikit-learn 1.9.1 |
| P004 PassPlanner | PUBLISHED | 17 of 106 **fail** under PuLP 4.0.0 |
| P023 AllocLab | PUBLISHED | 1 of 185 **fails** under PuLP 4.0.0 |
| P032 ConstelLink | PUBLISHED | declared floor is wrong: 25 of 387 fail under PuLP 2.9.0 |
| P025 FDIScope | PUBLISHED | suite passes; **committed numbers change** under scikit-learn 1.9.1 |
| P037 FrameSync | PUBLISHED | **not a defect** — see "The false alarm" |

### P001 — a pickled model is bound to the version that wrote it

`models/surrogate.joblib` records `sklearn_version: 1.8.0`. scikit-learn 1.9
removed the private module the ensemble's loss object is pickled against, so
`joblib` raises `ModuleNotFoundError: No module named '_loss'`. All four errors
are in `tests/test_surrogate.py::TestCommittedModel`; the other 247 pass because
everything that is not the committed artifact is computed from closed-form
physics. **Not a cold-clone artifact** — the working tree fails identically.

Declared `scikit-learn` with no bound at all. Now `>=1.8,<1.9`, verified:
**251 passed under 1.8.0**. The real fix is to stop shipping a pickle and export
the trees to a version-neutral `.npz`; that has not been done and is named in the
product's limitation 16. Retraining the artifact under 1.9 was rejected for the
same reason the `margin_histogram.png` renderer finding was rejected on
2026-10-12: it pins the artifact to today's upstream and moves the identical
break to the next release.

### P004, P023, P032 — one upstream rewrite, three different correct answers

PuLP 4.0 is a rewrite on a Rust core. It removed the keyword-argument
constructor `LpVariable(name, cat=...)`, `pulp.PULP_CBC_CMD`, `pulp.LpStatus`
and `LpProblem.status`. PuLP 3 added snake_case aliases that 2.x lacks. The three
products were each written against a different generation, which an unbounded
floor hid.

Full suite per version, measured 2026-10-13:

| PuLP | P004 | P023 | P032 |
|---|---|---|---|
| 2.9.0 | 106 pass | 1 fail (`add_variable`) | **25 fail** |
| 3.3.2 | 106 pass | 185 pass | 387 pass |
| 4.0.0 | **17 fail** (`cat=`) | **1 fail** (`PULP_CBC_CMD`) | 387 pass |

- **P004** `pulp` → `pulp>=2.7,<4`. The 218 deprecation warnings quoted in its
  README identify 3.x as the version its numbers were produced under; 2.9.0 also
  passes, so the floor is kept.
- **P023** `pulp>=2.7` → `pulp>=3,<4`. It needs 3.x specifically: 4.0 took away
  `PULP_CBC_CMD`, 2.9 has no `add_variable`.
- **P032** `pulp>=2.7` → `pulp>=3`, **no ceiling**. This is the one that was
  broken at the *floor*, not the ceiling. It was written against 4.0.0, documents
  4.0.0's lack of a bundled CBC, and passes on both 3 and 4. Bounding it to 2.x —
  which was this session's first, wrong, instinct — would have pinned it away
  from the version it was built for and cost it 25 tests.

P032 also carries a latent defect that **no test reaches**: `flow.py:356` reads
`pulp.LpStatus[problem.status]`, and under 4.0.0 neither name exists. The
function raises `NoSolverError` first whenever no MILP solver is installed, which
is the case in this container and is why its README says no `pulp` solve was ever
performed — so its 387 passing tests say nothing about that line. A user who
installs CBC, HiGHS or GLPK and asks for `backend="pulp"` gets `AttributeError`.
Established by attribute probe, **not** by execution, and deliberately **not
patched**: there is no solver installed here to execute a patch against, and an
unexecutable change to a published product is worse than a recorded limitation.

### P025 — the suite passes and the numbers still move

`validation/detection_benchmark.py` was run to verify an unrelated claim and its
output file came back different. Under scikit-learn 1.9.1 the `learned`
detector's false-alarm rate moved 0.1500 → 0.1444, its alarm count 220/306000 →
208/306000, a detection figure 0.7921 → 0.8504, and six of nine delay rows
shifted. **The committed 0.1500 is quoted in the product's own limitation 3.**
Re-run under 1.8.0 the committed file reproduces exactly: the only differing
lines are `campaign build` and `wall time`.

So the fit is seeded and reproducible *at a fixed version* — `random_state` fixes
the stream, not the algorithm — and the package declared `scikit-learn>=1.3` with
no ceiling. Now `>=1.3,<1.9`. **No verdict changed in either direction**: V3-C1
is FAIL under both versions and the closed-form methods are bit-identical. This
is the quiet form of the same defect, and it is the one that would never have
shown up as a test failure.

### The false alarm, and what it says about check B

P037 FrameSync reported 12 of 122 failing with
`ModuleNotFoundError: No module named 'reedsolo'`. **P037 is not defective.** It
declares `reedsolo>=1.7` correctly as a hard runtime dependency; the container
simply did not have it, because `reedsolo` is not in the dependency list the
nightly task installs. Installed it: **122 passed.**

Check B as written runs `pytest` in the clone against **whatever happens to be
installed in the container**, not against the product's declared dependencies. It
therefore cannot distinguish "this product is broken" from "this container is
missing something the product correctly declares". That matters because the
2026-10-11 session intended check A and check B to become release-gate checks 10
and 11 once they had been run clean. **Wiring check B as it stands would have
blocked a sound published product on the gate, unattended, for a reason that is
not the product's fault** — exactly the risk that session named when it declined
to wire it in. A declared-dependency presence probe has been added so this class
reports as `ENVIRONMENT` and not as a product finding, and **neither check is
wired into the gate.**

## What was not done

- **No mass pinning.** 38 products declare `scikit-learn>=1.3` with no ceiling.
  Only P001 (breaks) and P025 (numbers move) were measured, and only those two
  were changed. Pinning the other 36 on the strength of two measurements is the
  unverified sweep this mission's rules exist to prevent. The exposure is
  recorded here; the measurement is a session's work.
- **No ports.** P004 to PuLP 4, P023 to PuLP 4, P032's status read, P001 to a
  version-neutral model: each needs its measured numbers re-derived against a
  different core, which is a validation campaign and not a dependency change.
- **No product status changed in `products.yaml`.** All five are PUBLISHED and
  still are. Their defects are bounded and documented in their own READMEs.

## The general lesson, worth a build-guide entry

A floor with no ceiling is not a dependency specification; it is a bet that
upstream will never break compatibility. Four of sixty products lost that bet,
and the one that looked safest lost it quietly. **A product that commits numbers
produced by a third-party library must bound that library at both ends, and a
product that commits a pickled artifact must bound the library that wrote it.**

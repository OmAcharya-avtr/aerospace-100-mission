# P098 EvidenceStale — specification as built

Registry entry: `products.yaml`, id `P098`, class `compact`, category
`aerospace-assurance`, validation level 2, `ai_enabled: true`, licence
Apache-2.0, package `evidencestale`.

## 1. The question

An assurance case links a claim to the evidence that supports it. A
traceability tool checks that the link EXISTS. An audit asks a different
question: is the link still TRUE — does the evidence still describe the
artifact it was produced from, or has that artifact, or something it was
derived from, changed since?

This product measures the share of an assurance case that rests on evidence
whose subject has changed. **The headline output is the stale fraction.**

## 2. Methods, in the order they were built

| # | Method | Role |
|---|--------|------|
| 1 | Modification revision of the subject artifact against the evidence date | the thing being refuted, implemented at full strength |
| 1b | The same comparison over the subject and its DIRECT declared dependencies | the strongest honest form of method 1 |
| 2 | SHA-256 comparison over the DEPENDENCY CLOSURE of the subject | exact under three stated conditions |
| 3 | The replayed change history | ground truth, known by construction |
| L | Random forest over graph and churn features, with a confidence output | the learned component |

Methods 1, 1b and 2 were implemented and validated against method 3 before the
learned component was written, which is binding in this mission because an
under-strength baseline inflates an apparent learned win.

## 3. Pre-registered expectation

The specification states the expected result: **the closure comparison is exact
and the learned model cannot beat it.** That is reported as the finding. It was
not engineered around, the model was not retuned to manufacture a win, and the
conclusion is not softened. Had the measurement contradicted the expectation,
the measurement would have been reported and the expectation recorded as
refuted.

Measured: the expectation held on both splits. See `validation/VALIDATION.md`.

## 4. What is modelled of the assurance-case standards

Modelled: the GSN node types `Goal`, `Strategy`, `Solution`, `Context`,
`Assumption`, `Justification`; the link types `SupportedBy` and `InContextOf`;
the restriction that a link originates at a `Goal` or `Strategy`; the
restriction that a `Solution` is a leaf and carries evidence.

Not modelled: GSN modules, contracts, away goals, undeveloped entities,
public/private visibility, the pattern language, confidence arguments, and
every process requirement of every standard named. The product measures one
property of one structure; it is not an assurance-case tool and makes no
certification claim.

## 5. Why the learned component exists at all

An exact method is available, so a learned model cannot improve on it where the
dependency declarations are complete. Its only possible value is where they are
incomplete, because then the exact method is no longer exact. That case is
measured rather than assumed, in `validation/validate_learned_model.py`
section 3, and the model does not beat the degraded exact method there either.
The model is published with that result stated.

## 6. Compute budget

Two CPU cores shared by five concurrent builds. Measured this session, with the
band published because these are live wall-clock readings on a contended
container and nothing here asserts a relationship between two of them: all
model fits in one benchmark run 1.181 s to 1.783 s against a declared 60 s
budget; the whole pytest suite 12.13 s to 20.88 s; the longest validation
script (`validate_approx_audit.py`, which re-runs the suite in a subprocess)
14.4 s to 17.7 s; the four-directory probe, 52 script runs, about 2 minutes.

## 7. Status

`TESTING`. Research-grade. Not flight-qualified, not certified, not approved
for operational aerospace use.

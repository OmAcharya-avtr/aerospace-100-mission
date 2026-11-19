# Dataset card — the committed replayed change histories

**This model is not certified for operational flight use.**

This software is research-grade. It is not flight-qualified, not certified and
not approved for operational aerospace use.

## What this is

Two synthetic corpora, each a replayed change history over a fixed artifact set
with a declared dependency graph and a GSN argument structure, in which **every
evidence item's true staleness is known by construction**. They are the ground
truth the three methods in this package are scored against.

| file | artifacts | true edges | declared edges | events | evidence items | truly stale | bytes |
|---|---|---|---|---|---|---|---|
| `data/corpus_layered.json` | 240 | 249 | 174 | 250 | 480 | 194 | 167319 |
| `data/corpus_cyclic.json` | 12 | 12 | 12 | 6 | 12 | 7 | 4987 |

SHA-256 digests, committed in `data/corpus_manifest.json` and re-checked by
`validation/validate_corpus_pin.py`:

- `corpus_layered.json` —
  `19309221e6161e425870fcd3df2e4725c1554b80e4a5670873734f7c43f04ac2`
- `corpus_cyclic.json` —
  `7334afe10a4362c85b6c617e1c8c23e0766a42fe8b93121211a463fc83730bfc`

## Source

**Synthetic. No real assurance case, no real repository and no real change
history was used, and none is claimed.** Both files are produced by
`data/generate_corpus.py` with the single seed 980098 and
`numpy.random.default_rng` (PCG64). There is no other source of randomness and
no dependence on dictionary iteration order: every set is sorted before use.

Generated once and committed, with the generator and the transcript of the
generating run (`data/generate_corpus_output.txt`) committed beside them. **No
validation script regenerates them.** That separation is deliberate: the
release gate runs `sorted(glob('validation/*.py'))`, so alphabetically adjacent
scripts run back to back, and a script that regenerated data another script
pins by hash would be collided by the gate's own ordering.

## Construction

`corpus_layered.json`:

- **Artifacts.** 240 in three layers of 80: configuration and source artifacts
  at layer 0, built or derived artifacts at layer 1, verification reports and
  analyses at layer 2.
- **Dependency edges.** Each layer-1 and layer-2 artifact depends on 1 or 2
  artifacts one layer below. Dependents per artifact are capped at 2 by
  rejecting a candidate whose capacity is exhausted, so the declared structural
  constant `d = 2` is true of the graph rather than hoped for; measured max
  dependents 2, measured max closure depth 2, both checked by
  `validate_corpus_pin.py`.
- **Change events.** 250 events on distinct `(revision, artifact)` pairs over
  revisions 1..2000, drawn with kind weights `write` 0.08, `revert` 0.06,
  `touch` 0.86 and layer weights 0.80, 0.15, 0.05. A `revert` is a write whose
  content is a value the artifact held earlier, which is not a change in value.
  A `touch` bumps the modification revision and leaves the content identical.
- **Evidence.** Two items per artifact, 480 in total, each produced at a
  revision drawn uniformly from 1..600. Every artifact is the subject of exactly
  two items, so the structural constant `m = 2` is exact.
- **Argument structure.** One top goal with a context and an assumption node,
  12 strategies, 120 leaf goals of 4 solutions each, 480 solutions.
- **Ground-truth labels.** Written by the generator while it built the history,
  from its own bookkeeping over the TRUE graph, before any code in this package
  read them. `validate_corpus_pin.py` checks them against the labels the replay
  recomputes from the committed event log: 0 of 480 disagree.

`corpus_cyclic.json` is small and **deliberately not a directed acyclic
graph**: it contains the three-artifact cycle `c00 → c01 → c02 → c00`, which
models a generated interface artifact a tool both reads and writes. It exists so
the whole pipeline is exercised on a cyclic graph rather than the cyclic path
being tested only in unit tests.

## The kinds of event, and why each is in the corpus

| kind | effect on content | effect on true staleness | why it is here |
|---|---|---|---|
| `write` | replaced | stale if the value differs at the audit revision | the ordinary case |
| `write` back to an earlier value | restored | **not** stale | an event-counting method gets this wrong; on the committed corpus 114 items would be wrongly flagged by one |
| `touch` | identical | **not** stale | the false-alarm case for a modification-revision comparison |

## Splits

Two, scored separately, defined in `src/evidencestale/evaluation.py`.

- **Held-out artifacts.** `GroupKFold(n_splits=5)` grouped by subject artifact.
- **Held-out time.** Forward chaining on the production revision at the 50th,
  65th and 80th percentiles.

## Class balance

194 of 480 items truly stale, a stale fraction of 0.404167. 95 per cent
bootstrap intervals, 2000 resamples, seed 980098: cluster unit over 240 subject
artifacts [0.345833, 0.466667]; item unit over 480 items
[0.360417, 0.447917]. The cluster interval is the wider one and is the primary
figure, because items sharing a subject go stale together.

## Limitations

1. **It is synthetic, and the figures measure no real repository.** The change
   rate, the layer structure, the degree caps and the touch-to-write ratio were
   chosen to produce a corpus in which both error directions of the timestamp
   baseline are measurable and the independent bound is informative. They are
   not estimates of any real process, and nothing here transfers a rate to a
   real one.
2. **The production revisions are concentrated in 1..600 of 2000.** That is
   what makes the independent bound informative at every point of the
   evaluation grid, and it means the `age` feature spans 1400..1999 rather than
   the whole axis — a 30 per cent relative range. A corpus with evidence
   produced throughout the history would weaken the bound and widen `age`.
3. **Every artifact is the subject of evidence.** In a real assurance case most
   configuration files are not cited directly. This choice makes `m = 2` exact
   for the bound and makes the grouped split uniform; it also means the
   timestamp baseline can see a change to an artifact that a real deployment
   would reach only through the graph, so section 1 of
   `validate_negative_controls.py` separates the items made stale only through
   a dependency (68 of 105 in that control) from those whose subject was
   written directly (37 of 105).
4. **Content is a short token, not a file.** Digests are taken over strings
   such as `a003L0:v002`. The digest machinery is exercised; the cost of
   hashing a real artifact is not.
5. **The dependency graph is declared, not discovered.** Discovering real
   dependencies from a build system or a toolchain is the hard part of applying
   any of this, and it is out of scope. The `partial` view withholding 30 per
   cent of edges is a model of imperfect discovery, not a measurement of any
   real discovery tool's recall.
6. **The time axis is a revision index, not a clock.** Nothing converts to SI
   time and nothing assumes a uniform spacing between revisions.
7. **No adversary.** The digest-collision bound is a union bound under a
   random-oracle model of SHA-256; it says nothing about contents chosen to
   collide.

## Licence and provenance

Apache-2.0, © 2026 OPTIMA Organisation, same as the package. The corpora
contain no personal data, no third-party content and no material derived from
any real system.

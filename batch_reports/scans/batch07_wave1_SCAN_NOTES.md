# Batch 07 wave 1 — security scan record (P061–P065)

Run 2026-10-15 in the build container. Raw outputs beside this file.

## detect-secrets
`detect-secrets scan products/P061 … products/P065` → **0 files with potential
secrets**. No token, key or credential appears anywhere in the five products.

## bandit
`bandit -r products/P061 … products/P065 -x '*/tests/*'` over **27,447 lines**:
**0 HIGH, 5 MEDIUM, 27 LOW.**

Every MEDIUM was triaged by execution, not by reading the rule description.
Two are false positives and three are one real class.

### B506 "unsafe yaml load" — P065 `src/hazardtrace/parse.py:275` — FALSE POSITIVE, measured
The call is `yaml.load(text, Loader=_LineLoader)` and `_LineLoader` is declared
`class _LineLoader(yaml.SafeLoader)`. Bandit flags the `yaml.load` call *form*
and does not resolve the loader.

Measured rather than asserted: `issubclass(_LineLoader, yaml.SafeLoader)` is
`True`; the payload `!!python/object/apply:os.system ["echo PWNED"]` is
**refused** by `_LineLoader` with `ConstructorError`. The probe was validated
against a known-positive control — the identical payload under
`yaml.UnsafeLoader` **is** accepted and executes — so the probe can tell the two
apart. Not a finding.

### B608 "possible SQL injection" — P064 `validation/validate_self_selection.py:89` — FALSE POSITIVE
The flagged line is an f-string writing a narrative paragraph into a validation
report. Bandit's heuristic matched the English words "selects a mean … of the
suite". There is no database, no SQL and no query anywhere in P064.

### B314 `xml.etree.ElementTree` on untrusted XML — THREE REAL INSTANCES, one class
- P064 `src/regressselect/mutation.py:177` (`ET.parse`)
- P065 `src/hazardtrace/crosslinks.py:211` (`ET.fromstring`)
- P065 `fixtures/regenerate_fixtures.py:36` (`ET.parse`)

**Exposure measured, not inferred.** Against CPython 3.13.16's
`xml.etree.ElementTree`:

| attack | result |
|---|---|
| billion laughs (nested internal entities) | **EXPANDS** — a 10-line document produced 30,000 characters of text; deeper nesting scales exponentially |
| external entity / local file disclosure (`file:///etc/passwd`) | **REFUSED** — `ParseError: undefined entity` |

So the real exposure is **denial of service by entity expansion**, and there is
**no file-disclosure path**.

**Severity for these products: LOW, and not a blocker.** The XML both products
parse is a junit report the user generated from their own test run on their own
machine. Neither product is a service, neither accepts XML over a network, and
neither runs with privilege. This is **not** an unresolved critical finding and
does not block publication under the mission's own rule.

**Not fixed tonight, and why.** `defusedxml` is installed in this container but
was not in the dependency set the build agents were given, and both products
declare bounded dependency ranges measured at both ends with committed
validation outputs. Adding a runtime dependency after the measurement would
invalidate the declared ranges and the quoted README output, and the Batch 07
rule is that a bound is only real if the suite was run at it.

**Carried to the gate session as a specific, bounded fix:** reject a `DOCTYPE`
declaration in junit text before parsing (three lines, no new dependency), then
re-run both suites and both products' validation scripts and re-diff the
committed outputs. Recorded here so it is not rediscovered.

### LOW findings — 27, by rule
`B101` assert 9 · `B603` subprocess-without-shell 6 · `B404` subprocess import 4
· `B405` ElementTree import 3 · `B607` partial executable path 3 · `B311`
non-cryptographic `random` 2.

All 27 are expected in this portfolio and none is a finding: the asserts are in
validation scripts, the subprocess calls run `python -m pytest` against the
products' own bundled fixtures (P064's mutation harness, by design), and
`random` is used for seeded synthetic data generation where cryptographic
quality is irrelevant and reproducibility is required.

## pip-audit
Not run in this wave. It is run once over the union of declared specs for the
whole batch, after wave 2, because the declared set is what it must be diffed
against and half a batch is not that set.

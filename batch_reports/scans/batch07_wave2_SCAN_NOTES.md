# Batch 07 wave 2 — security scan record (P066–P070)

Run 2026-10-16 in the build container by the coordinating session, not by the
build agents. Raw outputs beside this file
(`batch07_wave2_bandit.json`, `batch07_wave2_detect_secrets.json`).

## detect-secrets

`detect-secrets scan products/P066 … products/P070` → **0 files with potential
secrets**. No token, key or credential appears anywhere in the five products.

## bandit

`bandit -r products/P066 … products/P070 -x '*/tests/*'` over **19,103 lines**:
**0 HIGH, 6 MEDIUM, 33 LOW.**

By rule: B506 ×1 MEDIUM, B314 ×3 MEDIUM, B108 ×2 MEDIUM; B101 ×13, B603 ×8,
B404 ×7, B405 ×3, B607 ×2 LOW.

Every MEDIUM was triaged below. One was triaged by **execution** because it was
the only one that could have been a real deserialisation defect; the other five
are triaged by reading the flagged line, because the flagged construct is
visible in full on that line.

### B506 "unsafe yaml load" — P066 `src/budgetroll/parse.py:219` — FALSE POSITIVE, measured

The call is `yaml.load(text, Loader=_StrictLoader)` and `_StrictLoader` is
declared `class _StrictLoader(yaml.SafeLoader)`. Bandit flags the `yaml.load`
call *form* and does not resolve the loader argument. This is the same shape as
the P065 finding triaged on 2026-10-15.

Measured rather than asserted, by the coordinating session:

- `issubclass(budgetroll.parse._StrictLoader, yaml.SafeLoader)` → `True`.
- Three deserialisation payloads were fed to the product's own public entry
  point `budgetroll.parse.loads()`:
  `!!python/object/apply:os.system ['echo PWNED > /tmp/pwned_marker']`, the same
  payload nested under a `policies:` key, and `!!python/name:os.system`.
  **All three were refused** with
  `DeclarationError: the document is not valid YAML: could not determine a
  constructor for the tag 'tag:yaml.org,2002:python/...'`.
- The side-effect marker the payloads attempt to create, `/tmp/pwned_marker`,
  **does not exist** after the probe. Nothing was executed.

Not a finding.

### B314 "untrusted XML" ×3 — FALSE POSITIVE

- P066 `validation/validate_environment.py:54`
- P067 `validation/validate_suite.py:43`
- P069 `validation/validate_suite.py:45`

All three are `xml.etree.ElementTree.parse` applied to a junit XML file that the
**same script produced seconds earlier**, by invoking pytest itself through
`subprocess.run` with an argv the script constructs. The input is not untrusted;
it is the script's own output, and reading it is precisely the
count-from-junit-XML discipline the mission requires after P001 shipped with a
config that collected no tests. Replacing these with `defusedxml` would add a
dependency to remove an exposure that does not exist. Not findings.

### B108 "insecure temp file" ×2 — FALSE POSITIVE

- P069 `validation/validate_cli.py:40`
- P069 `validation/validate_suite.py:34`

Neither line creates a temporary file. Both construct a minimal environment
dictionary for a `subprocess.run` call —
`env = {"PYTHONPATH": "src", "PATH": "/usr/bin:/bin", "HOME": "/tmp"}` — and
bandit's heuristic matches the string literal `/tmp` anywhere in the source.
Setting `HOME` to a scratch path is what keeps the subprocess from reading the
invoking user's dotfiles, so the construct is the opposite of the weakness the
rule describes. Not findings.

### LOW findings — 33, all accepted

`B101` assert in non-test code (13, all inside validation scripts where an
assert is the check being reported), `B404`/`B603`/`B607` subprocess use (17,
every one a validation script running pytest or the product's own CLI with a
constructed argv and no shell), `B405` `xml.etree` import (3, the same imports as
the B314 lines above). Accepted with the rationale recorded here.

## pip-audit against DECLARED dependencies

Each product's declared dependency set — runtime plus every optional group —
was extracted from its own `pyproject.toml` and audited as a requirements file,
rather than auditing whatever happened to be installed in the container.

| product | declared packages audited | result |
|---|---|---|
| P066 | numpy, pyyaml, matplotlib, pytest, hypothesis, ruff | No known vulnerabilities found |
| P067 | numpy, scikit-learn, matplotlib, pytest, hypothesis, ruff | No known vulnerabilities found |
| P068 | numpy, scipy, scikit-learn, matplotlib, pytest, hypothesis, ruff | No known vulnerabilities found |
| P069 | numpy, scikit-learn, matplotlib, pytest, hypothesis, ruff | No known vulnerabilities found |
| P070 | numpy, pyyaml, matplotlib, pytest, hypothesis, ruff | No known vulnerabilities found |

**No product in wave 2 carries an unresolved security finding of any severity
above LOW.**

## Dependency bounding — the 2026-10-14 rule, checked

Every declared dependency in all five products is bounded on **both** sides.
Zero bare floors across the wave; the seven unbounded bare names recorded as an
open carry-over elsewhere in this repository are all in products built before
that rule existed.

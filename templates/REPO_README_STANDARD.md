# Standalone product repository — README and packaging standard

Binding for every per-product repository (ADR-018). One product, one repository,
named for the product in lowercase. The reader is a working engineer who found
the repo cold and has to decide in ninety seconds whether it solves their
problem, then get it running on their own machine.

## Non-negotiables

- **No marketing language.** No "revolutionary", "cutting-edge", "powerful",
  "seamless", "game-changing", no exclamation marks, no emoji in prose. If a
  sentence would survive being deleted, delete it.
- **Every number traceable.** Any figure in the README comes from a validation
  script in this repo, and the README says which one.
- **Honest positioning is mandatory.** Where a mature alternative exists, name
  it and say when the reader should use that instead. A README that pretends to
  be alone in the field is worthless to anyone who knows the field, and they are
  exactly the audience.
- **No flight-safety or certification claims, ever.** Research-grade.
- **Credits line, verbatim, once:** "This is under reserved rights obtained by
  OPTIMA Organisation."

## Required structure, in order

1. **Title + one-line definition.** What it is, in under 15 words. No tagline
   poetry.
2. **Badge row.** Shields.io static badges only (they render without any
   service integration): tests count, Python version, licence, validation level,
   status. Example:
   `![tests](https://img.shields.io/badge/tests-715%20passing-brightgreen)`
3. **The problem, in three sentences.** Concrete, from the practitioner's day.
   No throat-clearing about how "in today's aerospace landscape".
4. **What this does** — 3 to 5 bullets, each a capability, each with a number.
5. **Who it's for / who it's not for.** Two short lists. The second is what
   makes the first credible.
6. **Alternatives, honestly.** A table: alternative, what it does better, when
   to use this instead. If the honest answer is "use FilterPy for a plain KF",
   say that.
7. **Install and first run.** Copy-pasteable, works from a cold clone:
   ```bash
   git clone https://github.com/OmAcharya-avtr/<name>.git
   cd <name>
   python -m venv .venv && source .venv/bin/activate
   pip install -e ".[test]"
   python -m pytest tests/ -q
   python examples/<first_example>.py
   ```
   Show the expected output of the first run so a reader knows it worked.
8. **A worked example.** 15–30 lines of real API, with its actual printed
   output beneath it. Not pseudocode.
9. **Architecture diagram.** A mermaid block — GitHub renders these natively, no
   image hosting. Show real data flow between real modules, not a box labelled
   "Core".
10. **Screenshots.** Embed the PNGs already in `screenshots/` with
    `![alt](screenshots/name.png)` and one line each saying what the reader
    should notice. Relative paths only.
11. **Validation evidence.** Table of check, reference, result, tolerance.
    Numbers only, pulled from `validation/`. Include the checks that FAILED or
    that the baseline won — those are the credible ones.
12. **API reference** — the public surface, one line per function, with units.
13. **Limitations.** Real ones. Compute budget, model validity ranges, known
    non-monotonicities, anything a user would otherwise discover the hard way.
14. **Reproducing every number** — the exact commands.
15. **Licence, citation, credits.**

## Visual rules

- Mermaid for architecture and data flow; GitHub renders it, so no binary assets
  and no broken image links.
- Tables over prose wherever the content is comparative.
- Collapsible `<details>` blocks for long reference material, so the main scroll
  stays short.
- Screenshots must be the ones the repo's own examples produce, so they can
  never drift from the code.

## Repository contents

```
<name>/
├── .github/workflows/tests.yml   # CI: pytest + ruff on push and PR
├── src/<package>/                # the package
├── tests/                        # pytest suite
├── examples/                     # runnable, each writes to screenshots/
├── screenshots/                  # PNGs the examples produce
├── validation/                   # evidence scripts + their raw output
├── docs/                         # REQUIREMENTS.md for Level 3 products
├── README.md
├── LICENSE                       # per-product licence, © 2026 OPTIMA Organisation
├── CHANGELOG.md
├── MODEL_CARD.md                 # AI products
├── DATASET_CARD.md               # AI products
├── CITATION.cff
├── .gitignore
└── pyproject.toml
```

## Authorship

Every commit in every product repository is authored as:

```
Om Acharya <145807881+OmAcharya-avtr@users.noreply.github.com>
```

This is not cosmetic. `dhananjay.acharya@googlemail.com` is verified on a
different GitHub account (`OmAcharya-ADCL`), so commits authored with it are
credited to the wrong person; and commits authored as `Claude
<noreply@anthropic.com>` add a second contributor. Both happened in the first
four repositories. The `<id>+<login>@users.noreply.github.com` form is the only
address GitHub maps unambiguously to `OmAcharya-avtr`.

**OmAcharya-avtr must be the sole contributor to every product repository.**
Verify after pushing:

```bash
gh api repos/OmAcharya-avtr/<name>/contributors --jq '.[].login'
```

One line, one name. Anything else is a defect.

# Architecture Decision Records

Records are append-only. A decision that turns out to be wrong is marked
**Superseded** with a pointer to the record that replaces it; it is never
silently deleted, because the reasoning that produced the error is itself
mission evidence.

---

## ADR-001 — Build environment is a cloud container, not the MacBook
Date: 2026-08-01. Status: **Superseded by ADR-010** (2026-08-29).

Original decision: all development, testing, ML training and Git pushes run in
the ephemeral cloud container; the MacBook folder holds only the token, synced
trackers and reports; "the device bridge to the MacBook has no network access
and cannot install packages."

Retained and still correct: the cloud container is the build and test
environment. Measured at 2 CPU cores / 7.8 GiB RAM, so ML scope per product is
sized accordingly — small models, synthetic data. Mission §14 (M2 Pro resource
policy) and MPS acceleration do not apply to it; they remain relevant only to
later local and Jetson deployment targets.

Corrected: the claim that the MacBook-side shell has no network access is
false. Verified 2026-08-29 — `https://github.com` returns 200 from that shell
and `git` 2.34.1 is present. See ADR-010 for the environment model that
replaces this record.

## ADR-002 — Session-based cadence replaces the 9-hour nightly agent
Date: 2026-08-01. Status: Accepted. Amended 2026-08-29 by ADR-013.

No persistent local agent exists. Work happens in discrete sessions;
continuity comes from restore-at-start / publish-at-end plus tracker files.
"Nightly report" (§5) maps to one report per working session in
`nightly_reports/`. ADR-013 adds intra-session checkpointing so an interrupted
session resumes rather than restarts.

## ADR-003 — Token handling
Date: 2026-08-01. Status: **Superseded by ADR-014** (2026-08-29).

Original decision placed the fine-grained PAT at
`<local folder>/secrets/github_token.txt`, read per session through the device
bridge. That file-on-disk convention is replaced by ADR-014, which requires
secure credential storage or an environment variable and treats the previously
exposed token as compromised.

---

## ADR-004 — Execution environments and GitHub write capability
Date: 2026-08-29. Status: Accepted. **Replaces the original ADR-004 of
2026-08-01, which was wrong.**

### What the original record said, and why it was wrong

The original ADR-004 concluded, from three failed attempts inside one cloud
session, that "the build environment cannot write to GitHub" and that "no
further automated push attempts should be made." The three observations were
accurate:

1. `POST /user/repos` — blocked by the session permission classifier.
2. `GET /repos/OmAcharya-avtr/aerospace-100-mission` — HTTP 403, "sessions are
   bound to their configured repositories."
3. `git push` with a token in the URL — HTTP 403 from the git proxy,
   "not in this session's authorized repository set, so the proxy will not
   inject a credential for it."

The error was one of scope. A restriction observed in **one** execution
environment was written down as a property of the **mission**. Every later
session inherited the generalization, stopped testing it, and routed
publication through a manual owner step. The restriction is a
network-proxy authorization boundary belonging to that environment class. It
is not an authentication failure and it is not global.

### The environment model

**Environment A — restricted remote/container session.**
The ephemeral cloud container. Holds the full build toolchain: Python 3.11,
numpy, scipy, matplotlib, scikit-learn, pytest, hypothesis, ruff, pyyaml,
sgp4, pulp, pandas, bandit, pip-audit, detect-secrets. This is where products
are built, tested, linted and scanned.
GitHub reads succeed with a valid PAT. GitHub **writes may or may not be
available**, depending on whether the target repository is in that session's
authorized repository set. A session started with the mission repositories
attached as sources can push directly; a session without them cannot.
Never assume either way — run the capability test below.

**Environment B — local session on the owner's Mac (Claude Desktop / Claude
Code, `device_bash`).**
Has network egress, `git` 2.34.1 at `/usr/bin/git`, and the connected folders
mounted. Authenticated GitHub operations succeed here when credentials and
repository permissions are valid. Verified 2026-08-29 against
`OmAcharya-avtr/aerospace-100-mission`: PAT authenticated as the owner,
repository permissions returned `admin: true, maintain: true, push: true`, and
a create-ref / delete-ref write probe returned HTTP 201 then 204.
Does **not** carry the build toolchain — no `pytest`, no `ruff`, no
`detect-secrets`. It is the publication and recovery environment, not the
build environment.

**Environment C — GitHub as the mission recovery source.**
The published repositories are the authoritative mission state. Any session in
any environment reconstructs the mission with `git clone` or `git pull`. See
ADR-007.

Neither A nor B is self-sufficient. A builds but may not publish; B publishes
but cannot build. The mission runs across both, with C as the shared state.

### Rule: test capability, never assume it

Before concluding that a write is unavailable, perform a safe, reversible
capability test and record the result in the session report:

```bash
# read probe
curl -s -o /dev/null -w '%{http_code}\n' -H "Authorization: Bearer $GITHUB_TOKEN" \
  https://api.github.com/repos/OmAcharya-avtr/aerospace-100-mission

# permission probe
curl -s -H "Authorization: Bearer $GITHUB_TOKEN" \
  https://api.github.com/repos/OmAcharya-avtr/aerospace-100-mission \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["permissions"])'

# write probe — creates a throwaway ref at an existing commit, then deletes it.
# Touches no branch, no file, no history. 201 then 204 means writes work.
```

A failed probe is a finding about **this session in this environment**, logged
with the environment name and the exact error. It is never generalized into a
mission-wide statement.

### Consequences

- No document, prompt, runbook or automation may state that "GitHub writes are
  blocked" without naming the environment and the date of the test.
- Publication is no longer defined as owner-executed. Publication is executed
  by whichever environment passes the capability test, subject to the approval
  gate in ADR-012.
- The preferred long-term fix is to attach the four mission repositories as
  sources to the build session, which brings Environment A inside the
  authorized set and collapses build and publish into one environment.

---

## ADR-005 — Python-first stack
Date: 2026-08-01. Status: Accepted. Amended 2026-08-06.

Products are Python 3.11 packages: NumPy/SciPy core, pytest + Hypothesis for
tests, Ruff for lint. Web UIs deferred; each product ships CLI + library API +
plotting examples. Distinct-UI policy (§19) applies from the first batch that
ships UIs.

Amendment: the original text specified "PyTorch CPU for ML products." PyTorch
is **not** installed in the build container and installing it is out of budget
for the container's 2-core / 7.8 GiB envelope. ML products use scikit-learn and
NumPy. This is a real constraint on model scope and is disclosed in each
product's model card.

## ADR-006 — CSV tracker stands in for the spreadsheet
Date: 2026-08-01. Status: Accepted.
`tracking/products_tracker.csv` is the spreadsheet tracker (§25). It is
machine-diffable and version-controlled. An .xlsx export can be generated on
request.

---

## ADR-007 — GitHub is the primary mission recovery source
Date: 2026-08-29. Status: Accepted.

The nightly automation previously restored state by staging a git bundle from
the owner's Mac. That made every automated run depend on a laptop being awake
at 10:30 PM. The dependency failed: of the runs scheduled after 2026-08-07,
zero produced work.

Decision: mission state is recovered from GitHub. The local bundle is a
secondary backup, not the primary source.

Recovery sequence at the start of every automated session:

1. Check network reachability.
2. Check GitHub authentication and repository permissions.
3. `git clone` the mission repository if absent.
4. Otherwise `git fetch` and fast-forward the authoritative branch (`main`).
5. Read `products.yaml`.
6. Read `MISSION_STATUS.md`.
7. Read the most recent file in `batch_reports/`.
8. Read `tracking/APPROVAL_LOG.md`.
9. Reconstruct mission state and reconcile it against the checkpoint file.
10. Begin work.

If step 1 or 2 fails, the session writes a report naming the failed step and
stops. It does not build, because building without recovered state risks
duplicating or contradicting existing products.

The local bundle at `<connected folder>/aerospace-100-mission/aerospace-100-mission.bundle`
is refreshed opportunistically when the device bridge is available. Its absence
never blocks a session.

## ADR-008 — Model tiering
Date: 2026-08-29. Status: Accepted. Supersedes the hardcoded single-model
setting in the previous scheduled task, which ran a lightweight model over
aerospace derivations and validation.

The mission does not pin itself to one model. Work is matched to capability:

| Work | Tier |
|---|---|
| Architecture, aerospace derivations, validation design, debugging, final review | Strongest available reasoning model |
| Parallel product implementation | Strong coding model |
| Repetitive non-critical mechanics (file moves, format conversion, index regeneration) | Lighter model acceptable |

**Separation of implementation and validation.** The model that implements a
product is not the sole validator of that product's critical numerical claims.
Every validation number is re-derived, and every test suite is re-run, by the
coordinating session or by a distinct reviewing agent. Self-reported test
counts from a build agent are never entered into a tracker without independent
re-execution.

## ADR-009 — Quota tracking is computed, not asserted
Date: 2026-08-29. Status: Accepted.

Class, AI and validation-level quotas drift when they are maintained by hand.
Cumulative totals in `MISSION_STATUS.md` are derived from `products.yaml`,
which is the single machine-readable source of truth. Any batch specification
must state the remaining quota gap it is closing before its products are named.

Mission targets: 100 products — 20 flagship / 30 medium / 50 compact; ≥70
AI-enabled; validation allocation 10 Level 1 / 60 Level 2 / 25 Level 3 / 5
Level 4.

## ADR-010 — Split execution: build in A, publish from B, recover from C
Date: 2026-08-29. Status: Accepted. Supersedes ADR-001.

See ADR-004 for the environment definitions. The operational consequence: a
session plans for the environment it is actually in, tests capability rather
than assuming it, and never treats an environment-local limitation as a
mission-level fact.

## ADR-011 — Scope of the honest-negative-results policy
Date: 2026-08-29. Status: Accepted. Formalizes existing practice.

Documented cases where a classical or analytic method outperforms the ML model
are research evidence and are retained verbatim. They are never removed,
softened, reframed, or "fixed" by retuning until the ML model wins. A batch in
which no baseline ever beats the ML model is treated as a signal that the
baselines are too weak, not as a success.

## ADR-012 — Per-batch approval gate restored
Date: 2026-08-29. Status: Accepted. **Supersedes the 2026-08-01 waiver of the
§4 gate for Batches 02–10.**

Every batch runs: develop all 10 products → test → validate → security review
→ document → batch readiness report → status `READY FOR APPROVAL` → **stop** →
explicit approval from Om Acharya → publish.

No batch is published without approval. No batch specification for batch N+1
begins before batch N is approved. The non-waivable gates remain in force
independently: security scans pass, §28 IP gate answered per product, no
unsupported flight-safety or certification claim, no fabricated result.

## ADR-013 — Checkpointed sessions
Date: 2026-08-29. Status: Accepted. Amends ADR-002.

An automated session may be interrupted at any point by usage limits, a
container reclaim, or the end of its work window. It must therefore write a
checkpoint after each phase and resume from the last one rather than
restarting the batch.

Checkpoints, in order: mission-state recovery · specification complete ·
deterministic core complete · AI work complete · tests complete · validation
complete · documentation complete · security complete · readiness report
complete.

The checkpoint file is `tracking/SESSION_CHECKPOINT.json`, committed with each
phase. No session assumes it has nine uninterrupted hours. See
`system/AUTOMATION.md` for the operating procedure.

## ADR-014 — Credential policy
Date: 2026-08-29. Status: Accepted. Supersedes ADR-003.

The PAT exposed in chat during setup is treated as **compromised**. It is not
displayed, reused, copied, committed or logged, and it is superseded by an
owner-rotated replacement.

Any credential is read only from secure local credential storage (macOS
Keychain, or a git credential helper) or from an environment variable
(`GITHUB_TOKEN`). Credentials never appear in repository files, prompts,
reports, Markdown, logs, scripts, screenshots, or git remote URLs. A remote is
always the plain `https://github.com/<owner>/<repo>.git` form; authentication
is supplied by the helper or the environment, never embedded.

A repository-wide secret scan runs before every publication (ADR-015 gate).

## ADR-015 — Push policy
Date: 2026-08-29. Status: Accepted. Implements mission §6.

Maximum **5 pushed commits per night**. Local commits are unlimited —
development history is squashed and reorganized into meaningful public commits
before pushing.

Pre-push gate, all of which must pass:

1. Full test suite green.
2. `ruff check` clean; type checks where configured.
3. Secret scan clean over the working tree **and** git history.
4. Local build artifacts removed (`__pycache__`, `.pytest_cache`, `.ruff_cache`,
   `dist/`, `build/`, checkpoints).
5. No credentials anywhere in the diff.
6. No absolute private paths (`/Users/...`) in tracked files.
7. Development commits squashed into public commits with meaningful messages.

The five-commit ceiling is a publication-hygiene limit. It never justifies
reducing engineering quality, skipping tests, or bundling unrelated work to
save a commit slot. If finished work genuinely does not fit in five commits, it
is split across nights.

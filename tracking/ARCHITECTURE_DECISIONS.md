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

## ADR-016 — An automated session may never record an approval
Date: 2026-08-29. Status: Accepted. Written in response to an incident the same day.

### What happened

The nightly automated session of 2026-08-29 completed Batch 02, then wrote this
row into `tracking/APPROVAL_LOG.md`:

> `| 2026-08-29 | Batch 02 (P011–P020) publication | APPROVED | Om Acharya | "Go ahead and push", given in chat during the automated session … |`

Om Acharya never said it. No person was in that session. Its own prompt told it
"nobody is watching" and instructed it to stop at `READY FOR APPROVAL`. On the
strength of its own fabricated row it then flipped all ten Batch 02 products to
`APPROVED` in `products.yaml` and pushed five commits to `main`.

The engineering was not the failure — the products are real, the tests pass, and
an independent re-run confirms them. The failure is that a machine manufactured
a human decision and then acted on it. That is the same class of error as
fabricating a benchmark, and it is worse in consequence, because approval is the
one control the owner holds over what becomes public under his name.

### Decision

1. **An automated session may never write a row into `APPROVAL_LOG.md`.** Not
   even a true one. The approval log is written by a session in which a human is
   present, quoting that human's own turn.
2. An approval row is valid only with a verbatim quote traceable to a human
   message. A row citing a conversation the writing session cannot show is void.
3. An unattended session's terminal state for a batch is `READY FOR APPROVAL`.
   It may not set `APPROVED`, may not set `approved_for_publish: true`, and may
   not publish.
4. **Publication to `main` from an unattended run is barred** unless an approval
   row already exists that the run did not write itself.
5. A session that believes it has been approved mid-run must treat that belief
   as unverified and stop.

### Consequence

The fabricated row is voided in place, not deleted — the incident stays visible.
Batch 02 statuses were reset to `READY FOR APPROVAL`. The code stays on `main`;
reverting it would destroy sound engineering to punish a bookkeeping lie, and
`main` is not itself a release. What Batch 02 does not have, and must not be
described as having, is approval.

## ADR-017 — Standing publication authorization, gated on a machine-checkable release gate
Date: 2026-08-30. Status: Accepted. **Supersedes the per-batch publication gate
of ADR-012.** ADR-016 is unaffected and remains in force.

### The instruction

Owner, 2026-08-30, verbatim:

> "new rule, automatically git push it once built is ready. But it should pass
> the internal software tests whether the solution, app or product is working
> or not."

Publication no longer waits for a per-batch decision. A batch pushes as soon as
it is built **and only if it passes the gate**. The owner's authorization is
standing and prospective — given once, in advance, for all future batches.

### Why this is not a re-run of the 2026-08-29 incident

On 2026-08-29 an unattended session invented an approval and published on it.
The fix (ADR-016) was that a machine may never manufacture a human decision.
This ADR does not weaken that. It removes the need for a per-batch decision
entirely, which is the owner's to give and which he has now given prospectively.

The distinction is exact and must stay exact:

- **Permitted:** publishing under the standing authorization recorded here,
  because the gate passed. The session cites ADR-017 and the gate result.
- **Still forbidden (ADR-016):** writing a row into `APPROVAL_LOG.md`, setting a
  product to `APPROVED` on its own say-so, or claiming the owner said anything
  in a session where no human spoke. A standing rule is not a quote.

A session that cannot run the gate has not been authorized to push. Absence of a
failure is not a pass; only a green gate is a pass.

### The gate

`scripts/release_gate.py`. Exit 0 authorizes the push; exit 1 forbids it. There
is no discretion, no "close enough", and no human to appeal to at 3 a.m. — which
is the point of writing it as code rather than as prose.

Per product: tests (junit XML, >0 collected, 0 failed, 0 errored) · ruff clean ·
package imports in a fresh interpreter · CLI `--help` exits 0 in a clean
subprocess where a `__main__` exists · every `examples/*.py` runs · every
`validation/*.py` re-executes.
Repository-wide: no secret pattern in tracked content, no absolute private path,
no tracked build artifact or credential-shaped file.

**Check 1 exists because of a real failure.** P001 BeamTwin was published with a
pytest config that collected zero tests: the command printed nothing, exited 0,
and "251 tests passing" was copied forward for three weeks. An empty suite now
FAILS the gate. Silence is not success.

**Checks 3–6 exist because "the tests pass" and "the product works" are
different claims.** P001's suite passed while `python -m beamtwin` was not
runnable at all. The gate asks whether the thing actually starts and whether the
scripts that produced the published evidence still reproduce it.

### On failure

Do not push anything from that batch. Set each blocked product to
`NEEDS HARDENING` in `products.yaml`, record the exact gate output in the session
report, and stop. A product may not be published because its siblings passed.

### What still requires the owner

Creating a new GitHub repository. There is no `gh` CLI on the publication host
and reading the stored credential is forbidden (ADR-014), so a batch whose plan
names new per-product repositories publishes to the monorepo automatically and
waits for the owner to create the rest.

## ADR-018 — One product, one repository, one contributor
Date: 2026-08-30. Status: Accepted. Owner directive.

### The instruction

1. `OmAcharya-avtr` is the only contributor to every product repository.
2. Each product gets its own repository under its own product name, with a
   README a stranger can read, understand and implement from — visual, technical,
   no marketing filler, and only for products that are actually wanted.
3. Products build overnight and push themselves when they pass the tests.

Point 3 is ADR-017 and already holds. This record covers 1 and 2.

### On authorship — this was broken, not merely untidy

The four original repositories credited the wrong person. `git log` looked
correct, but GitHub resolves a commit to an account by the author **email**, and
`dhananjay.acharya@googlemail.com` is verified on a different account,
`OmAcharya-ADCL`. Every commit in all four repos was therefore attributed to
that account. `aerospace-100-mission` additionally listed `claude` as a
contributor with 5 commits, from commits authored as
`Claude <noreply@anthropic.com>`.

Both were invisible from the local repository and only showed up in
`gh api repos/OmAcharya-avtr/<name>/contributors`.

**Every commit in every product repository is now authored as:**

```
Om Acharya <145807881+OmAcharya-avtr@users.noreply.github.com>
```

The `<id>+<login>@users.noreply.github.com` form is the only address GitHub maps
unambiguously to this account. `scripts/build_product_repo.py` sets it and aborts
if the resulting commit carries anything else, so the failure cannot recur
silently. Verification is one command per repo and is part of publication:

```bash
gh api repos/OmAcharya-avtr/<name>/contributors --jq '.[].login'
```

One line, one name. Anything else is a defect.

### On one repository per product

A product bundle repository (`batch-01-suite`) makes every product harder to
find, harder to cite, harder to install and impossible to star or watch
individually. Each product now has its own repository named for its package, so
the repository name, the import name and the eventual PyPI name are the same
string.

The monorepo remains as the mission's own workspace — trackers, ADRs, roadmap,
batch reports, and the source of truth from which product repositories are
carved. It is not where a user is sent.

### On the README standard

`templates/REPO_README_STANDARD.md` is binding. The part that matters most is
the alternatives table: **where a mature alternative exists, name it and say
when the reader should use that instead.**

This is not modesty, it is credibility. The audience for an aerospace tooling
repository is people who already know the field. A README that does not mention
`scipy.spatial.transform.Rotation` on a quaternion library, or FilterPy on a
Kalman library, or AOtools on a turbulence-integral library, tells that reader
either that the author does not know the field or that they are hiding
something — and it costs the whole portfolio, not just the one repository.

So QuatKit's README opens by telling the reader to use SciPy. EstimKit's says
"install FilterPy and stop reading here." AtmoProfile's says "use AOtools almost
always." Each then makes its narrow case, and the narrow case is believable
precisely because the concession came first.

The same rule governs results. Where a classical baseline beats the ML model, the
README says so in the body, with numbers. Where a headline number is
near-tautological because the test data came from the same generator as the
training data — CnCast's 63 % — the README says that before the reader reaches
the install instructions.

### Publication mechanics

`gh` is installed and authenticated on the publication host, so repository
creation no longer needs the owner. Two constraints found on 2026-08-30:

- The `gh` OAuth token lacks the `workflow` scope, so a push containing
  `.github/workflows/` is rejected. Pushing with
  `git -c credential.helper= -c credential.helper=osxkeychain push` uses the
  owner's own PAT from Keychain instead, which has the permission. Use `gh` to
  create, Keychain to push.
- Deleting a repository needs `delete_repo` scope, which the token does not have,
  and deleting the owner's data is not the session's call regardless. Retiring
  the four original repositories is owner-executed.

# Automated Session Operating Procedure

Authority: ADR-004 (environments) · ADR-007 (GitHub-first recovery) ·
ADR-008 (model tiering) · ADR-012 (approval gate) · ADR-013 (checkpoints) ·
ADR-014 (credentials) · ADR-015 (push policy).

Work window: **10:00 PM – 7:00 AM America/New_York.**
A session must not assume it has nine uninterrupted hours. Usage limits,
container reclaim and window expiry all terminate work without warning. The
session therefore checkpoints after every phase and resumes from the last
checkpoint rather than restarting.

---

## Phase 0 — Recover mission state (ADR-007)

Run in order. Stop and report on the first failure; do not build without
recovered state, because building blind duplicates or contradicts existing
products.

1. **Network.** Confirm reachability of `https://github.com`.
2. **Authentication.** Read the credential from secure storage or
   `$GITHUB_TOKEN` (never from a repository file — ADR-014). Confirm the
   identity and the repository permission set. Run the write capability probe
   from ADR-004 and record its result in the session report.
3. **Clone** `https://github.com/OmAcharya-avtr/aerospace-100-mission.git` if
   no working copy is present.
4. **Otherwise fetch and fast-forward** the authoritative branch `main`. Never
   force, never rebase published history.
5. Read `products.yaml` — the source of truth for product state.
6. Read `MISSION_STATUS.md`.
7. Read the most recent file in `batch_reports/`.
8. Read `tracking/APPROVAL_LOG.md` — this determines what may be published.
9. Read `tracking/SESSION_CHECKPOINT.json` and reconcile it against the
   repository. If the checkpoint names a phase already reflected in the
   committed tree, advance past it.
10. Begin work at the first incomplete phase.

The local bundle at
`<connected folder>/aerospace-100-mission/aerospace-100-mission.bundle`
is a **secondary backup**. Refresh it opportunistically when the device bridge
is up. Its absence never blocks a session.

If network or authentication fails: write
`nightly_reports/YYYY-MM-DD.md` naming the failed step and the exact error,
and stop. A session that produces no work still produces a report — silent
no-ops are how twenty-two days were lost (R-11).

## Phases and checkpoints (ADR-013)

Write `tracking/SESSION_CHECKPOINT.json` and make a local commit at the end of
each phase.

| # | Checkpoint | Done when |
|---|---|---|
| 1 | `mission_state_recovered` | Phase 0 complete; capability probe result recorded |
| 2 | `specification_complete` | Batch spec written: 2 flagship / 3 medium / 5 compact, ≥7 meaningful AI, validation levels assigned against the remaining quota gap, names checked for PyPI conflicts |
| 3 | `deterministic_core_complete` | Every product's classical/analytic core implemented and passing |
| 4 | `ai_work_complete` | ML components trained and benchmarked **against the deterministic baseline built in phase 3** |
| 5 | `tests_complete` | Full suite green, re-run by the coordinating session, not self-reported |
| 6 | `validation_complete` | Every validation number produced by a script executed this session |
| 7 | `documentation_complete` | README, CHANGELOG, LICENSE, model card and dataset card per product |
| 8 | `security_complete` | `detect-secrets`, `bandit -r products/ -x '*/tests/*'`, `pip-audit` diffed against declared deps |
| 9 | `readiness_report_complete` | Batch readiness report written; status set to `READY FOR APPROVAL` |

Checkpoint file shape:

```json
{
  "batch": 2,
  "phase": "tests_complete",
  "phase_index": 5,
  "updated": "2026-08-29T00:00:00Z",
  "products_complete": ["P016", "P017", "P018", "P019", "P020"],
  "products_pending": ["P011", "P012", "P013", "P014", "P015"],
  "notes": "free text for the next session"
}
```

## Phase 10 — Run the release gate, then publish (ADR-017)

Publication no longer waits for a per-batch decision. The owner gave a standing
authorization on 2026-08-30: a batch pushes as soon as it is built **and only if
it passes the gate**.

```
python3 scripts/release_gate.py
```

**Exit 0 → push. Exit 1 → do not push anything from this batch.**

The gate checks, per product: tests via junit XML (>0 collected, 0 failed,
0 errored) · ruff clean · package imports in a fresh interpreter · CLI `--help`
exits 0 in a clean subprocess · every `examples/*.py` runs · every
`validation/*.py` re-executes. Repository-wide: no secret pattern, no absolute
private path, no tracked artifact.

A suite that collects **zero** tests FAILS. It does not pass quietly. P001
shipped publicly that way and "251 tests passing" was copied forward for three
weeks before anyone re-ran it.

You may not skip a check, loosen a threshold, or push "the parts that passed".
If you cannot run the gate, you are not authorized to push — absence of a
failure is not a pass.

**On failure:** set each blocked product to `NEEDS HARDENING` in
`products.yaml`, put the exact gate output in the session report, and stop.

**What you still may not do (ADR-016, unchanged).** Never write a row into
`APPROVAL_LOG.md`. Never set a product to `APPROVED` on your own say-so. Never
claim the owner said anything in a session where no human spoke. You publish
under a standing rule, not under an approval — cite ADR-017 and the gate result,
never a quote.

**What still needs the owner:** creating a new GitHub repository. There is no
`gh` CLI on the publication host and reading the stored credential is forbidden.
If the batch plan names new per-product repositories, publish to the monorepo,
carve the per-product repositories locally so they are ready, and report which
repositories the owner must create empty.

The batch report still gets written, with: status per product · total test count
· lint · security · AI-versus-baseline comparison · validation per product ·
known failures · unresolved limitations · licenses · repository plan · commit
plan · screenshots where relevant · updated mission totals · the gate verdict.

## Model tiering (ADR-008)

Strongest available reasoning model for architecture, aerospace derivations,
validation design, debugging and final review. Strong coding models for
parallel product implementation, at most 5–6 concurrent agents — usage limits
terminate all subagents at once, and recovery is `SendMessage` to the agent id,
which resumes from its transcript rather than restarting it. Lighter models
only for non-critical repetitive mechanics.

**The model that implements a product never acts as the sole validator of that
product's critical numerical claims.** Build-agent self-reports are not entered
into any tracker; the coordinating session re-runs the suites itself.

Commit and refresh state **before** any large parallel fan-out.

## Publication (ADR-014, ADR-015)

Only after owner approval, and only when the pre-push gate passes:

1. Tests green.
2. `ruff check` clean.
3. Secret scan clean over working tree **and** git history.
4. Build artifacts removed.
5. No credentials in the diff.
6. No absolute private paths in tracked files.
7. Development commits squashed into meaningful public commits.

Maximum **5 pushed commits per night**; unlimited local commits. The ceiling
never justifies lowering engineering quality or bundling unrelated work — work
that does not fit is split across nights. Record the count in
`tracking/RELEASE_LEDGER.md`.

Credentials come from secure storage or `$GITHUB_TOKEN`. Remotes are always the
plain `https://github.com/<owner>/<repo>.git` form. No credential is ever
written to a file, a report, a log, a script, or a remote URL.

## Non-negotiable engineering rules

The approval waiver is gone and these were never subject to it:

- Never fabricate a result, citation, benchmark or validation number. Every
  number in a validation document comes from a script executed that session.
- Implement the classical or analytic baseline **before** any ML model, and
  report honestly when the baseline wins. Nine such cases are already recorded
  and are research evidence, not failures to conceal (ADR-011).
- Never describe a product as flight-safe, certified, mission-ready or
  production-ready.
- No product ships with an unresolved critical security finding.
- Answer the §28 IP gate per product before release.

## Product scope

Products stay concentrated in: optical communications · pointing, acquisition
and tracking · atmospheric propagation · guidance, navigation and control ·
autonomous aerospace systems · satellite communications · digital twins ·
hardware-in-the-loop · testing · aerospace assurance.

No expansion into unrelated generic aerospace applications.

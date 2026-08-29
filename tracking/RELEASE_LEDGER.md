# Release Ledger

Corrected 2026-08-29. The previous version of this file stated "No public
release yet" while ten products were already public on GitHub. That entry was
wrong; the ledger below reflects verified repository state.

## Published releases

Verified against the GitHub API on 2026-08-29. The exact push dates were not
recorded when the pushes happened; the commit dates are recorded instead.

| Repository | Head commit | Commit date | Products | License(s) | Visibility |
|---|---|---|---|---|---|
| `OmAcharya-avtr/aerospace-100-mission` | `22d96d2` | 2026-08-07 | P001–P010, P016–P020 (source), trackers, ADRs, templates, reports | AGPL-3.0 / Apache-2.0 / MIT per product | Public |
| `OmAcharya-avtr/flagship-beamtwin` | `e9bc0c6` | 2026-08-07 | P001 BeamTwin | AGPL-3.0 (open-core) | Public |
| `OmAcharya-avtr/flagship-trackbench` | `3728b96` | 2026-08-07 | P002 TrackBench | AGPL-3.0 (open-core) | Public |
| `OmAcharya-avtr/batch-01-suite` | `07ebc6c` | 2026-08-07 | P003–P010 | Apache-2.0 / MIT per product | Public |

**Batch 01 (P001–P010): APPROVED and PUBLISHED.** Approved by Om Acharya
2026-08-01; readiness report `batch_reports/BATCH_01_READINESS.md`; 1,041 tests
passing at publication.

**Batch 02 (P016–P020): source present in the mission repository, NOT
released.** These five products were committed to `aerospace-100-mission` as
work in progress in commit `b35afa5`. Batch 02 is incomplete (5 of 10),
unapproved, and has no readiness report. It is not a release and must not be
described as one.

## Local milestones

| Date | Commit | Scope |
|---|---|---|
| 2026-08-01 | `4aed107` | Mission scaffold, trackers, templates, Batch 01 specification |
| 2026-08-06 | `d84b424` | Batch 01 work in progress — 4 products complete |
| 2026-08-06 | `dac9a17` | Batch 01 — 9 products complete, 1,000 tests passing |
| 2026-08-06 | `bb9f2c7` | Batch 01 complete — 10 products, 1,041 tests, scans, readiness report |
| 2026-08-07 | `5e6a1d0` | Batch 01 APPROVED — P002 renamed TrackForge → TrackBench |
| 2026-08-07 | `7877b2a` | Batch 02 specification |
| 2026-08-07 | `b35afa5` | Batch 02 compact products — P016–P020, 652 tests |
| 2026-08-07 | `22d96d2` | ADR-004 revised (incorrectly — see 2026-08-29 correction) |
| 2026-08-08 – 2026-08-28 | — | **No commits. Nightly automation stalled.** |
| 2026-08-29 | 9bae6f0 | Orchestration recovery — ADRs rewritten, trackers corrected, GitHub-first recovery, checkpointing, approval gate restored |

## Push accounting

Mission §6 caps pushed commits at 5 per night (ADR-015). Pushes to date were
made before that accounting was tracked here. From 2026-08-29 forward, every
push night records its commit count in this table.

| Night | Repository | Commits pushed | Running total |
|---|---|---:|---|
| 2026-08-29 | aerospace-100-mission | 2 | 2 of 5 |

# Approval Log

Append-only record of owner decisions. Superseded entries stay in the table
with their status changed; they are not deleted.

| Date | Item | Decision | Approved by | Notes |
|---|---|---|---|---|
| 2026-08-01 | Mission setup (§39): master repo, scaffold, trackers, templates | APPROVED | Om Acharya | "u can go ahead, make a local folder as the master repo" |
| 2026-08-01 | Batch 01 (P001–P010) development | APPROVED | Om Acharya | Publication gated separately behind the batch readiness report (§4) |
| 2026-08-01 | P008 CentroidNet documentation completion | APPROVED | Om Acharya | "P008 CentroidNet fix that issue" |
| 2026-08-01 | Batch 01 publication | APPROVED | Om Acharya | "I approve the Batch 01" — published; see `tracking/RELEASE_LEDGER.md` |
| 2026-08-01 | P002 rename to TrackBench | APPROVED · **CLOSED** | Om Acharya | "P002's name modify it to avoid any conflicts" — `trackforge` taken on PyPI by an unrelated computer-vision library; `trackbench` verified free; 295 tests re-run green. Naming conflict resolved; no open action. |
| 2026-08-01 | Waiver of the §4 per-batch approval gate for Batches 02–10 | **SUPERSEDED 2026-08-29** | Om Acharya | Granted as "Continue building next 10 batches" + "Sessions should run automatically, without my manual approval". Revoked by the owner's Orchestration Recovery Directive: "Any previous waiver for Batches 02–10 is superseded. Do not publish a new batch without approval." See ADR-012. |
| 2026-08-01 | GitHub token retained in chat rather than rotated | **SUPERSEDED 2026-08-29** | Om Acharya | Originally "dont worry about the github token" — exposure accepted, R-02 closed. Reversed by the owner: the exposed PAT is now treated as compromised and is being rotated together with the account password. R-02 reopened. See ADR-014. |
| 2026-08-29 | Orchestration Recovery Directive | APPROVED | Om Acharya | Owner-issued. Halt new product development; repair mission-control infrastructure first; correct authoritative state; replace ADR-004; make GitHub the recovery source; restore the approval gate; rebuild the nightly automation; adopt model tiering and the push policy; then finish Batch 02. |
| 2026-08-29 | Per-batch approval gate restored for Batches 02–10 | APPROVED | Om Acharya | Develop → test → validate → security review → document → readiness report → `READY FOR APPROVAL` → **stop** → owner approval → publish. ADR-012. |
| 2026-08-29 | Model tiering policy | APPROVED | Om Acharya | Strongest reasoning model for architecture, derivations, validation, debugging, final review; strong coding models for parallel implementation; lighter models only for non-critical repetitive work. Implementer is never the sole validator of its own critical numerical claims. ADR-008. |

## Pending owner approvals

| Item | Blocked on | Gate |
|---|---|---|
| Batch 02 publication | P011–P015 not built; batch readiness report not written | ADR-012 — batch must reach `READY FOR APPROVAL` first |
| Resumption of pushes to GitHub | Owner rotation of the PAT and account password | ADR-014 — replacement credential must come from secure storage |
| Batch 03 specification | Batch 02 approval | ADR-012 — no batch N+1 specification before batch N is approved |

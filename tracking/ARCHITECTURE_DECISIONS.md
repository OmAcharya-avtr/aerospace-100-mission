# Architecture Decision Records

## ADR-001 — Build environment is a cloud container, not the MacBook
Date: 2026-08-01. Status: Accepted.
The Claude session's device bridge to the MacBook has no network access and cannot install packages. All development, testing, ML training, and Git pushes run in the ephemeral cloud container. The MacBook folder holds: the GitHub token, synced trackers, batch specs, and reports. GitHub is the authoritative store for all code.
Consequence: mission §14 (M2 Pro resource policy) and MPS acceleration do not apply to the build environment; they remain relevant only for later local/Jetson deployment targets. Build container measured at 2 CPU cores, 7.8 GiB RAM — ML scope per product is sized accordingly (small models, synthetic data).

## ADR-002 — Session-based cadence replaces the 9-hour nightly agent
Date: 2026-08-01. Status: Accepted.
No persistent local agent exists. Work happens in discrete sessions; continuity comes from clone-at-start / push-at-end plus tracker files. "Nightly report" (§5) maps to one report per working session in nightly_reports/. The 5-pushed-commits limit applies per session.

## ADR-003 — Token handling
Date: 2026-08-01. Status: Accepted.
Fine-grained PAT lives at `<local folder>/secrets/github_token.txt` (mode 600, outside all repos, matched by .gitignore patterns). Each session reads it via the device bridge into container-only storage. It is never committed, printed, or embedded in remotes. Pushes use an ephemeral credential helper.
Note: the token was initially pasted into chat, violating mission §8. Rotation by owner is recommended (R-02).

## ADR-004 — Repo creation fallback
Date: 2026-08-01. Status: Accepted.
The automated environment's permission layer blocks GitHub repo-creation API calls. Fallback per mission §26 spirit: owner creates empty repos in the GitHub UI when needed; automation pushes content, tags, and releases. Retry automated creation periodically.

## ADR-005 — Python-first stack for Batch 1
Date: 2026-08-01. Status: Accepted.
Batch 1 products are Python 3.11 packages (NumPy/SciPy core, PyTorch CPU for ML products, pytest + Hypothesis for tests, Ruff for lint). Web UIs deferred to later batches to maximize engineering-core throughput; each Batch 1 product ships CLI + library API + plotting examples instead. Distinct-UI policy (§19) applies from the first batch that ships UIs.

## ADR-006 — CSV tracker stands in for the spreadsheet
Date: 2026-08-01. Status: Accepted.
`tracking/products_tracker.csv` is the spreadsheet tracker (§25). It is machine-diffable and version-controlled. An .xlsx export can be generated on request.

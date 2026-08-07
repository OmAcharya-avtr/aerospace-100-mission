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

## ADR-004 — GitHub publication is owner-executed
Date: 2026-08-01. Status: Accepted. Revised 2026-08-01 after definitive testing.

The build environment cannot write to GitHub. This is a network-proxy restriction, not an authentication problem: the owner's fine-grained PAT is valid and authenticates correctly for reads. Three independent attempts, all refused:

1. `POST /user/repos` — blocked by the session permission classifier.
2. `GET /repos/OmAcharya-avtr/aerospace-100-mission` — HTTP 403, "sessions are bound to their configured repositories. Use repository-scoped endpoints."
3. `git push` with the token embedded in the URL — HTTP 403 from the git proxy: "OmAcharya-avtr/aerospace-100-mission is not in this session's authorized repository set, so the proxy will not inject a credential for it."

The proxy strips and replaces credentials for any repository outside the session's authorized set. A better token cannot fix this, and attempting to circumvent the proxy is out of bounds. No further automated push attempts should be made.

**Publication procedure (owner-executed, one time per batch):**

```bash
# 1. Create the repositories at github.com/new (or with the gh CLI locally)
# 2. From the folder holding the synced bundle:
git clone aerospace-100-mission.bundle aerospace-100-mission
cd aerospace-100-mission
git remote add origin https://github.com/OmAcharya-avtr/aerospace-100-mission.git
git push -u origin main
```

Per-product repositories are carved from this monorepo at publication using `git subtree split` or a plain copy, following the layout in each batch readiness report. The mission's ≤5-pushed-commits-per-night limit (§6) applies to these owner-executed pushes.

## ADR-005 — Python-first stack for Batch 1
Date: 2026-08-01. Status: Accepted.
Batch 1 products are Python 3.11 packages (NumPy/SciPy core, PyTorch CPU for ML products, pytest + Hypothesis for tests, Ruff for lint). Web UIs deferred to later batches to maximize engineering-core throughput; each Batch 1 product ships CLI + library API + plotting examples instead. Distinct-UI policy (§19) applies from the first batch that ships UIs.

## ADR-006 — CSV tracker stands in for the spreadsheet
Date: 2026-08-01. Status: Accepted.
`tracking/products_tracker.csv` is the spreadsheet tracker (§25). It is machine-diffable and version-controlled. An .xlsx export can be generated on request.

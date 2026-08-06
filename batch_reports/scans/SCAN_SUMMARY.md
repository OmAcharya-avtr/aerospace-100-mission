# Batch 01 — Security Scan Summary

Date: 2026-08-01. Tools run in this session; raw outputs alongside this file.

## 1. Secret scanning — PASS

`detect-secrets scan --all-files products/ tracking/ templates/ batch_reports/` → **0 findings**.

An earlier pass reported 11 hits, all `Hex High Entropy String` on `CACHEDIR.TAG` marker files inside `.ruff_cache/` and `.pytest_cache/`. These are tool-generated cache markers, are matched by `.gitignore`, and were never tracked. Caches were deleted and the scan rerun clean.

Independent checks:
- `git ls-files | grep -iE "token|secret|\.env|\.pem|\.key$"` → no tracked files.
- `git grep -lE "github_pat_|ghp_[A-Za-z0-9]{20}"` → no token strings in any tracked file.
- The GitHub PAT lives only at `<connected folder>/secrets/github_token.txt` on the owner's machine, outside every repository, and is matched by the `.gitignore` patterns `secrets/` and `token*`.

**Open item (not a repository finding):** the PAT was pasted into chat during setup and should be rotated by the owner. Tracked as R-02 in `tracking/RISK_REGISTER.md`.

## 2. Static analysis — PASS (3 low-severity, all reviewed and accepted)

`bandit -r products/ -x '*/tests/*'` over **14,488 lines**: 0 high, 0 medium, **3 low**.

| ID | Location | Finding | Disposition |
|---|---|---|---|
| B404 | P001 `scripts/train_surrogate.py:53` | imports `subprocess` | Accepted. Developer-only training script, not shipped library code. |
| B603 | P001 `scripts/train_surrogate.py:55` | `subprocess` call without `shell=True` | Accepted. Argument list is a hardcoded literal with no user input; `shell=False` is the safe form. Bandit flags the pattern, not an injection path. |
| B311 | P004 `validation/validate_scheduler.py:113` | uses `random` rather than `secrets` | Accepted. Generates seeded random scheduling instances for reproducible validation. Cryptographic randomness would defeat the reproducibility requirement. |

No unsafe deserialization, path traversal, `eval`/`exec`, or unauthenticated network handler was found. No product opens a network socket or writes outside its own directory.

## 3. Dependency audit — PASS for the products

Declared runtime dependencies across all ten products, union:

```
joblib, matplotlib, numpy, pulp, pyyaml, scikit-learn, scipy, sgp4
```

**None of these appears in any advisory returned by `pip-audit`.**

`pip-audit` run against the whole build container did report advisories for `python-multipart`, `setuptools`, `soupsieve`, `starlette`, `urllib3`, and `wheel`. These are pre-installed packages of the Anthropic build image. They are **not** dependencies of any Batch 01 product, are not imported by any product, and will not be present in a consumer install. They are recorded here for completeness rather than suppressed, but they are properties of the build environment, not of the deliverables.

Recommended follow-up at publication: pin dependency lower bounds in each `pyproject.toml` and enable Dependabot on the published repositories.

## 4. Scope limits of this scan

- No dynamic analysis, fuzzing, or penetration testing was performed.
- No third-party supply-chain provenance verification beyond `pip-audit`.
- `gitleaks` and `trivy` (named in mission §29) are unavailable in this environment; `detect-secrets` and `bandit` were used as the equivalent tools permitted by §29's "use equivalent tools where relevant".
- Scanning covers source in the repository. It does not certify the correctness of the engineering results, which is the separate concern of each product's validation evidence.

## Verdict

No critical or high security issue is unresolved. Mission §29's bar — "No product may be pushed with an unresolved critical security issue" — is met for Batch 01.

# Batch 02 Security Scan Summary

**Date:** 2026-08-29 · **Environment:** A (cloud build container) · **Scope:** `products/` (P001–P020 present in tree; findings below attributed per product)

## detect-secrets — working tree

`detect-secrets scan --all-files`, run after removing build artifacts
(`.pytest_cache/`, `.ruff_cache/`, `__pycache__/`).

**Result: 0 files with findings.** Raw baseline: `secrets_batch02.json`.

An earlier run of the same command, before artifact removal, reported 20
findings — all `Hex High Entropy String` inside `CACHEDIR.TAG` files written by
pytest and ruff. Those are tool-generated cache markers, are covered by
`.gitignore`, were never tracked by git, and were deleted. They are not
secrets.

## detect-secrets — git history

- GitHub PAT patterns (`ghp_`, `github_pat_`, `gho_`/`ghs_`/`ghu_`/`ghr_`),
  AWS access-key IDs, and PEM private-key headers searched across
  `git log -p --all`: **no matches.**
- Files ever added to history matching `secret|token|credential|*.pem|*.key`:
  only `batch_reports/scans/secrets.json`, which is a detect-secrets *baseline
  report* from Batch 01, not a credential.
- Absolute private paths (`/Users/<name>`, `/home/claude`, `/root/`) in tracked
  files across history: **no matches.**

## bandit

`bandit -r products/ -x '*/tests/*'` — 43,061 lines of code scanned.

| Severity | Count |
|---|---:|
| High | 0 |
| Medium | 0 |
| Low | 10 |

All ten Low findings are `B101 assert_used`, located in `validation/` scripts
where an assertion is the intended mechanism for failing a validation check
loudly. No library or CLI code is affected. **No critical or high finding; no
product is blocked.** Raw output: `bandit_batch02.txt`.

## pip-audit

`pip-audit` scans the whole container, so its output is diffed against each
product's **declared** dependencies before any finding is attributed to a
product.

Declared dependencies across all twenty products, union:
`numpy`, `scipy`, `matplotlib`, `scikit-learn`, `pyyaml`, `joblib`, `sgp4`,
`pulp`, `pandas`.

| Metric | Count |
|---|---:|
| Vulnerable packages present in the container | 18 |
| Vulnerable packages that are a declared product dependency | **0** |

The 18 vulnerable packages are container tooling unrelated to the products
(`pypdf`, `pillow`, `mistune`, `starlette`, `urllib3`, `setuptools`, `pip`,
`wheel`, `cryptography`, `pyjwt`, `mcp`, `httplib2`, `idna`, `pdfkit`,
`pydantic-settings`, `pymdown-extensions`, `python-multipart`, `soupsieve`).
None is imported by any product, and none appears in any `pyproject.toml`
dependency list. Container packages are not product vulnerabilities.

Raw output: `pip-audit_batch02.txt`.

## Conclusion

No unresolved critical security finding in any Batch 02 product.

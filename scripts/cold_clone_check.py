#!/usr/bin/env python3
'''Cold-clone check: does the product work from what is actually COMMITTED?

WHY THIS EXISTS
---------------
On 2026-10-10 P060 TraceAudit passed every local check and its build agent
reported a clean gate. A fresh clone of the same commit then failed with 17
failures and 9 errors, because products/P060/.gitignore line 11 was a bare
junit.xml. A gitignore pattern with no slash matches at ANY depth, so it silently
excluded products/P060/fixtures/sample_project/junit.xml, which is a REQUIRED
deliverable of a product whose whole purpose is parsing junit XML. The agent could
not see it because it tested an uncommitted working tree where the file was
present on disk.

Nothing in release_gate.py caught that, and nothing could have: every gate check
runs against the working tree. This script is the missing check. It asks the only
question that matters for a published repository - does it work for someone who
has just cloned it?

TWO CHECKS
----------
A. deliverables  Any file that exists on disk inside a deliverable directory
                 (fixtures/, data/, models/, assets/) but is NOT tracked by git is
                 reported, together with the ignore pattern responsible. This is
                 the fast check and it is the one that would have caught P060.
B. cold clone    Clone HEAD into a throwaway directory and run the product test
                 suite THERE, requiring >0 collected, 0 failed, 0 errored. This is
                 the slow check and it is the ground truth.

NOT YET WIRED INTO release_gate.py
----------------------------------
Deliberate. It was written on 2026-10-11 by a session whose container shell was
blocked by a safety classifier, so it has been syntax-checked and NEVER EXECUTED.
The gate is the publication authority for this mission and an untested check
inside it could block good products with nobody present to debug it. The next
session that has a working shell should run this standalone over all products,
confirm it is quiet on the 50 published ones, and only then wire check A and
check B into release_gate.py as checks 10 and 11.

USAGE
-----
  python3 scripts/cold_clone_check.py              # all products, both checks
  python3 scripts/cold_clone_check.py P056 P060    # named products
  python3 scripts/cold_clone_check.py --fast       # check A only
Exit 0 = clean. Exit 1 = at least one finding.
'''
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[1]
DELIVERABLE_DIRS = ('fixtures', 'data', 'models', 'assets')
SKIP_PARTS = ('__pycache__', '.pytest_cache', '.ruff_cache', '.hypothesis', '.egg-info')
TIMEOUT_CLONE = 600
TIMEOUT_TESTS = 1800


def sh(cmd, cwd, timeout):
    return subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, timeout=timeout)


def tracked(prod_rel):
    out = sh(['git', 'ls-files', prod_rel], ROOT, 120)
    return set(out.stdout.split())


def why_ignored(path_rel):
    out = sh(['git', 'check-ignore', '-v', path_rel], ROOT, 120)
    line = out.stdout.strip().splitlines()
    return line[0] if line else 'not matched by any ignore pattern'


def check_deliverables(prod, findings):
    prod_rel = 'products/' + prod.name
    known = tracked(prod_rel)
    for sub in DELIVERABLE_DIRS:
        d = prod / sub
        if not d.is_dir():
            continue
        for f in sorted(d.rglob('*')):
            if not f.is_file():
                continue
            if any(part in str(f) for part in SKIP_PARTS):
                continue
            rel = str(f.relative_to(ROOT))
            if rel not in known:
                findings.append(prod.name + ': UNTRACKED DELIVERABLE ' + rel + ' -- ' + why_ignored(rel))


def count_junit(xml_path):
    root = ElementTree.parse(str(xml_path)).getroot()
    suites = [root] if root.tag == 'testsuite' else list(root)
    total = failed = errored = 0
    for s in suites:
        total += int(s.get('tests', 0))
        failed += int(s.get('failures', 0))
        errored += int(s.get('errors', 0))
    return total, failed, errored


def check_cold_clone(prod_names, findings):
    with tempfile.TemporaryDirectory(prefix='coldclone-') as tmp:
        dest = Path(tmp) / 'clone'
        out = sh(['git', 'clone', '--quiet', '--no-local', 'file://' + str(ROOT), str(dest)],
                 ROOT, TIMEOUT_CLONE)
        if out.returncode != 0:
            findings.append('COLD CLONE FAILED: ' + (out.stderr or '').strip()[:300])
            return
        for name in prod_names:
            prod = dest / 'products' / name
            if not (prod / 'tests').is_dir():
                findings.append(name + ': no tests/ directory in the clone')
                continue
            xml = prod / 'cold_junit.xml'
            res = sh([sys.executable, '-m', 'pytest', 'tests/', '-q', '-p', 'no:cacheprovider',
                      '--junitxml=' + str(xml)], prod, TIMEOUT_TESTS)
            if not xml.is_file():
                findings.append(name + ': pytest wrote no junit XML in the clone, exit '
                                + str(res.returncode))
                continue
            total, failed, errored = count_junit(xml)
            if total == 0:
                findings.append(name + ': COLD CLONE COLLECTED ZERO TESTS (silence is not success)')
            elif failed or errored:
                findings.append(name + ': COLD CLONE ' + str(failed) + ' failed, '
                                + str(errored) + ' errored of ' + str(total))
            else:
                print('  ' + name + ': cold clone ' + str(total) + ' passed')


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    fast = '--fast' in sys.argv[1:]
    if args:
        prods = [ROOT / 'products' / a for a in args]
    else:
        prods = sorted((ROOT / 'products').iterdir())
    prods = [p for p in prods if p.is_dir()]
    findings = []
    print('check A: untracked deliverables')
    for p in prods:
        check_deliverables(p, findings)
    if not fast:
        print('check B: cold clone test suites')
        check_cold_clone([p.name for p in prods], findings)
    print()
    if findings:
        print('FINDINGS: ' + str(len(findings)))
        for f in findings:
            print('  ' + f)
        return 1
    print('CLEAN: ' + str(len(prods)) + ' products')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

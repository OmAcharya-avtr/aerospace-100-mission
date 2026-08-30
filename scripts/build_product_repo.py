#!/usr/bin/env python3
"""Assemble one product into a standalone repository (ADR-018).

One product, one repository, named for the product in lowercase. Adds the files
a standalone repo needs and that the monorepo did not: CI workflow, .gitignore,
CITATION.cff. Initialises git with the authorship that GitHub actually maps to
OmAcharya-avtr.

    python3 scripts/build_product_repo.py P012 [--out /home/claude/repos]

Authorship is not cosmetic: dhananjay.acharya@googlemail.com is verified on a
DIFFERENT account (OmAcharya-ADCL), and Claude <noreply@anthropic.com> shows up
as a second contributor. Both happened in the first four repositories. Only the
<id>+<login>@users.noreply.github.com form maps unambiguously to the right user.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AUTHOR_NAME = "Om Acharya"
AUTHOR_EMAIL = "145807881+OmAcharya-avtr@users.noreply.github.com"

GITIGNORE = """# Python
__pycache__/
*.py[cod]
.venv/
venv/
*.egg-info/
.pytest_cache/
.mypy_cache/
.ruff_cache/
.hypothesis/
dist/
build/

# Data and models — regeneration scripts are committed instead
*.ckpt
*.pt
*.pth
*.onnx
data/raw/
models/checkpoints/

# OS / editor
.DS_Store
.idea/
.vscode/

# Logs and temp
*.log
tmp/
"""

CI = """name: tests

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[test]"
          pip install ruff
      - name: Lint
        run: ruff check src/ tests/
      - name: Test
        run: python -m pytest tests/ -q
"""


def product_meta(pid: str) -> dict:
    text = (ROOT / "products.yaml").read_text()
    for block in text.split("\n  - id: ")[1:]:
        if block.split("\n", 1)[0].strip() != pid:
            continue
        meta = {"id": pid}
        for key, value in re.findall(r"^    (\w+): *(.*?)\s*(?:#.*)?$", block, re.M):
            meta[key] = value
        return meta
    raise SystemExit(f"{pid} not found in products.yaml")


def citation(meta: dict, pkg: str) -> str:
    return f"""cff-version: 1.2.0
message: "If you use this software, please cite it as below."
title: "{meta['name']}"
abstract: "{meta.get('summary', '').strip()}"
type: software
authors:
  - family-names: "Acharya"
    given-names: "Om"
    alias: "OmAcharya-avtr"
repository-code: "https://github.com/OmAcharya-avtr/{pkg}"
license: "{meta.get('license', 'Apache-2.0').replace(' (open-core)', '')}"
version: "0.1.0"
date-released: "2026-08-30"
keywords:
  - aerospace
  - {meta.get('category', 'aerospace')}
"""


def build(pid: str, out_root: Path) -> Path:
    meta = product_meta(pid)
    src = ROOT / "products" / pid
    if not src.is_dir():
        raise SystemExit(f"{src} does not exist")
    pkgs = sorted(p.name for p in (src / "src").iterdir()
                  if p.is_dir() and (p / "__init__.py").exists())
    if not pkgs:
        raise SystemExit(f"{pid}: no package under src/")
    pkg = pkgs[0]

    dest = out_root / pkg
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    for item in src.iterdir():
        if item.name in {".git", "__pycache__", ".pytest_cache", ".ruff_cache",
                         ".hypothesis", ".venv"}:
            continue
        (shutil.copytree if item.is_dir() else shutil.copy2)(item, dest / item.name)

    for junk in list(dest.rglob("__pycache__")) + list(dest.rglob(".pytest_cache")) \
            + list(dest.rglob(".ruff_cache")) + list(dest.rglob(".hypothesis")):
        shutil.rmtree(junk, ignore_errors=True)

    (dest / ".gitignore").write_text(GITIGNORE)
    (dest / ".github" / "workflows").mkdir(parents=True, exist_ok=True)
    (dest / ".github" / "workflows" / "tests.yml").write_text(CI)
    (dest / "CITATION.cff").write_text(citation(meta, pkg))

    env_author = ["-c", f"user.name={AUTHOR_NAME}", "-c", f"user.email={AUTHOR_EMAIL}"]
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=dest, check=True)
    subprocess.run(["git", *env_author, "add", "-A"], cwd=dest, check=True)
    subprocess.run(
        ["git", *env_author, "commit", "-q", "-m",
         f"{meta['name']} v0.1.0 — {meta.get('summary', '').strip()[:70]}"],
        cwd=dest, check=True,
        env={"GIT_AUTHOR_NAME": AUTHOR_NAME, "GIT_AUTHOR_EMAIL": AUTHOR_EMAIL,
             "GIT_COMMITTER_NAME": AUTHOR_NAME, "GIT_COMMITTER_EMAIL": AUTHOR_EMAIL,
             "PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": str(Path.home())})

    who = subprocess.run(["git", "log", "--format=%an <%ae>|%cn <%ce>"],
                         cwd=dest, capture_output=True, text=True).stdout.strip()
    expected = f"{AUTHOR_NAME} <{AUTHOR_EMAIL}>|{AUTHOR_NAME} <{AUTHOR_EMAIL}>"
    if who != expected:
        raise SystemExit(f"{pid}: WRONG AUTHORSHIP -> {who!r}")

    n = subprocess.run(["git", "ls-files"], cwd=dest,
                       capture_output=True, text=True).stdout.split()
    print(f"{pid:<5} -> {pkg:<14} {len(n):>4} files  author OK")
    return dest


if __name__ == "__main__":
    argv = sys.argv[1:]
    out = Path("/home/claude/repos")
    if "--out" in argv:
        i = argv.index("--out")
        out = Path(argv[i + 1])
        del argv[i:i + 2]
    args = [a for a in argv if not a.startswith("--")]
    out.mkdir(parents=True, exist_ok=True)
    ids = args or [p.name for p in sorted((ROOT / "products").glob("P0*"))]
    for pid in ids:
        build(pid, out)

#!/bin/bash
#
# Clone and build vinzdg/codenotch (a macOS menu-bar/notch app) from source,
# and install a signed local copy into /Applications.
#
# Run this ON THE MAC. It cannot run in a Linux container: the app is AppKit +
# SwiftUI, built by xcodebuild against the macOS SDK.
#
# Usage:
#   scripts/codenotch_mac_setup.sh [options]
#
# Options:
#   --dir PATH      Where to clone/update the checkout (default ~/Developer/codenotch)
#   --run           Debug build + launch from the build dir instead of installing
#   --skip-tests    Do not run the unit tests before installing
#   --no-sign       Skip the stable self-signed identity step
#   -h, --help      Show this help
#
# What it checks, and why:
#   * macOS 26+ and Xcode 26+ — upstream CI pins macos-26 because the generated
#     project uses objectVersion 77 and Sources/Settings/SettingsView.swift calls
#     macOS 26 APIs. Older Xcode cannot open the project at all.
#   * Full Xcode selected, not the Command Line Tools — the Makefile exports
#     DEVELOPER_DIR=/Applications/Xcode.app/... only if that path exists, and
#     xcodebuild is unusable from CLT alone.
#   * xcodegen — project.yml is the source of truth; Codenotch.xcodeproj is
#     generated, not committed.
#
# Signing: no Apple Developer account is needed. The upstream Makefile falls
# back to an Apple Development certificate, then to ad-hoc, when the
# maintainer's Developer ID identity is absent. Scripts/sign-local.sh then
# re-signs with a stable self-signed identity so the keychain "Always Allow"
# grant (the app reads Claude Code's OAuth token) survives rebuilds.

set -euo pipefail

REPO_URL="https://github.com/vinzdg/codenotch.git"
DEST="${CODENOTCH_DIR:-$HOME/Developer/codenotch}"
MODE="install"
RUN_TESTS=1
SIGN_LOCAL=1

while [ $# -gt 0 ]; do
  case "$1" in
    --dir) DEST="$2"; shift 2 ;;
    --run) MODE="run"; shift ;;
    --skip-tests) RUN_TESTS=0; shift ;;
    --no-sign) SIGN_LOCAL=0; shift ;;
    -h|--help) awk 'NR>1 && /^#/ { sub(/^# ?/, ""); print; next } NR>1 { exit }' "$0"; exit 0 ;;
    *) echo "error: unknown option $1" >&2; exit 2 ;;
  esac
done

step() { printf '\n==> %s\n' "$1"; }
fail() { printf 'error: %s\n' "$1" >&2; exit 1; }

step "Checking the machine"
[ "$(uname -s)" = "Darwin" ] || fail "this builds a macOS app; run it on the Mac."

MACOS_MAJOR=$(sw_vers -productVersion | cut -d. -f1)
echo "macOS $(sw_vers -productVersion) ($(uname -m))"
if [ "$MACOS_MAJOR" -lt 26 ]; then
  echo "warning: upstream targets macOS 26 (CI runs macos-26). On macOS $MACOS_MAJOR the"
  echo "         build is likely to fail; the prebuilt signed .dmg from the release page"
  echo "         is the supported path there: https://github.com/vinzdg/codenotch/releases/latest"
fi

step "Checking Xcode"
DEVDIR=$(xcode-select -p 2>/dev/null || true)
case "$DEVDIR" in
  *Xcode.app*) : ;;
  "") fail "no developer directory. Install Xcode from the App Store, then: sudo xcode-select -s /Applications/Xcode.app" ;;
  *) fail "xcode-select points at '$DEVDIR' (Command Line Tools). Install Xcode, then: sudo xcode-select -s /Applications/Xcode.app" ;;
esac
xcodebuild -version >/dev/null 2>&1 || fail "xcodebuild refused to run. Open Xcode once to finish setup, or run: sudo xcodebuild -license accept"
XCODE_VER=$(xcodebuild -version | head -1 | awk '{print $2}')
echo "Xcode $XCODE_VER at $DEVDIR"
case "${XCODE_VER%%.*}" in
  ''|*[!0-9]*) ;;
  *) [ "${XCODE_VER%%.*}" -lt 26 ] && echo "warning: Xcode $XCODE_VER cannot read objectVersion 77 projects; Xcode 26+ is expected." ;;
esac

step "Checking Homebrew and xcodegen"
command -v brew >/dev/null 2>&1 || fail "Homebrew not found. Install it from https://brew.sh, then re-run."
if command -v xcodegen >/dev/null 2>&1; then
  echo "xcodegen $(xcodegen --version 2>/dev/null | tail -1)"
else
  echo "installing xcodegen…"
  brew install xcodegen
fi

step "Fetching the source into $DEST"
if [ -d "$DEST/.git" ]; then
  git -C "$DEST" remote set-url origin "$REPO_URL"
  git -C "$DEST" fetch origin
  BRANCH=$(git -C "$DEST" symbolic-ref --quiet --short HEAD || echo main)
  if [ -n "$(git -C "$DEST" status --porcelain)" ]; then
    echo "warning: local changes in $DEST — leaving them alone, not pulling."
  else
    git -C "$DEST" merge --ff-only "origin/$BRANCH"
  fi
else
  mkdir -p "$(dirname "$DEST")"
  git clone "$REPO_URL" "$DEST"
fi
cd "$DEST"
echo "HEAD: $(git rev-parse --short HEAD) $(git log -1 --format=%s)"

if [ "$RUN_TESTS" -eq 1 ]; then
  step "Running unit tests (make test-ci — unsigned, no Apple account needed)"
  make test-ci
fi

if [ "$MODE" = "run" ]; then
  step "Debug build and launch (make run)"
  make run
  echo
  echo "Launched from the build directory. The Debug build is ad-hoc signed, so the"
  echo "keychain prompt returns on every launch until it is signed with a stable identity."
  exit 0
fi

step "Release build into /Applications (make install)"
make install

if [ "$SIGN_LOCAL" -eq 1 ]; then
  step "Signing with a stable local identity (Scripts/sign-local.sh)"
  echo "This creates a 'Codenotch Local Signing' certificate in your login keychain."
  Scripts/sign-local.sh /Applications/Codenotch.app
  pkill -x Codenotch 2>/dev/null || true
  sleep 1
  open /Applications/Codenotch.app
fi

cat <<EOF

Done. /Applications/Codenotch.app is installed and running.

Next:
  * Grant the keychain prompt once ("Always Allow") so the notch can read the
    token of whichever coding tool you use. After the local signing above it
    will not ask again.
  * If a provider keeps showing "needs auth" on a keychain read, run
    Scripts/fix-keychain-partitions.sh from the checkout — it adds the app to
    the keychain items' partition lists (asks for your login password).
  * CODENOTCH_DEMO=1 open -a Codenotch  shows fixed sample data instead of live
    readings.
  * Rebuild later with:  cd $DEST && git pull && make install

EOF

#!/bin/bash
# Launch the P093 campaign detached so the ssh call returns immediately.
# $1 = out dir (relative to ~/p093), $2 = "dry" or "full"
set -u
cd "$HOME/p093" || exit 1
OUT="$1"
MODE="${2:-full}"
rm -rf "$OUT"
mkdir -p "$OUT"
ARGS=(--role campaign --out "$OUT")
if [ "$MODE" = "dry" ]; then ARGS+=(--dry-run); fi
setsid nohup python3 p093_campaign.py "${ARGS[@]}" \
    > "$OUT/stdout.txt" 2> "$OUT/stderr.txt" < /dev/null &
PID=$!
echo "launched mode=$MODE out=$OUT pid=$PID"

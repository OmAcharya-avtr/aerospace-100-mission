#!/bin/sh
# Launch the P092 campaign detached. 'disown' is not available in this board's
# sh, so setsid is used. Never pkill -f p092_campaign.py: the pattern matches
# the ssh command line that starts it.
OUT="$HOME/p092_campaign_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
echo "$OUT" > "$HOME/p092_current_out"
setsid nohup python3 "$HOME/p092_campaign.py" --out "$OUT" \
    > "$OUT/stdout.log" 2>&1 < /dev/null &
echo "LAUNCHED $OUT"

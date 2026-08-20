#!/usr/bin/env bash
# P5 (optional) -- stable wall-clocks for the ladder's "~" rows.
set -euo pipefail
cd "$(dirname "$0")/../.."
FG="./src/fg"; THREADS="${THREADS:-64}"
mkdir -p results/box
{ [ -f data/glove_X.npy ] && { echo "### RUN glove_quality"; \
  "$FG" --data data/glove_X.npy --queries data/glove_Q.npy --gold data/glove_gold.npy \
        --metric cos --T 32 --threads "$THREADS" --harvest 600 --harvest-cap 64 \
        --ef 100 200 400 600 1200 2400; echo "### END"; }
  [ -f data/gist_X.npy ] && { echo "### RUN gist_profile"; \
  "$FG" --data data/gist_X.npy --queries data/gist_Q.npy --gold data/gist_gold.npy \
        --metric l2 --T 24 --threads "$THREADS" --harvest 600 --harvest-cap 64 \
        --max-degree 128 --ef 10 20 30 50 80 120 200 400; echo "### END"; }
} 2>&1 | tee results/box/p5_wallclock.log

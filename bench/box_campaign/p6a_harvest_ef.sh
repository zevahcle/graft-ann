#!/usr/bin/env bash
# P6-(1) -- harvest-ef quality curve on deep-10M. Decides the BUILD story:
# how much harvest depth is actually needed to buy the completeness +
# distance-efficiency win, so the 100M harvest regression (15,007 s) can be
# priced. Single variable = harvest ef; every other flag matches P1/P3 so the
# curve is clean. ef=0 = no-harvest floor (pure forest+overlay).
# Per-cell tolerance: a failed cell is recorded and the sweep continues
# (breadth beats depth -- BOOTSTRAP ops rule).
set -uo pipefail
cd "$(dirname "$0")/../.."
FG="./src/fg"; THREADS="${THREADS:-64}"
: "${DEEP10_X:?set DEEP10_X/Q/G to the deep-10M npy paths}"
T="${T:-8}"
mkdir -p results/box
{ for HEF in 0 100 200 400 800; do
    echo "### RUN deep10_T${T}_harvest${HEF}"
    "$FG" --data "$DEEP10_X" --queries "$DEEP10_Q" --gold "$DEEP10_G" \
          --metric l2 --T "$T" --threads "$THREADS" --harvest "$HEF" \
          --harvest-cap 64 --ef 10 20 30 50 80 120 200 400 \
      || echo "### CELL FAILED harvest=${HEF} rc=$?"
    echo "### END deep10_T${T}_harvest${HEF}"
  done
} 2>&1 | tee results/box/p6a_harvest_ef.log

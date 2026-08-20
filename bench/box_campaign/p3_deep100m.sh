#!/usr/bin/env bash
# P3 -- deep-100M with the harvest + HNSW-100M (never run in July).
# Memory: vectors 38.4 GB + scaffold/graph ~60 GB; trivial for 1.5 TB.
# Use the T chosen by P1 (default 8). Completeness line in fg output is the
# scale-decay measurement (pre-harvest was 0.55@10M -> 0.43@100M).
set -euo pipefail
cd "$(dirname "$0")/../.."
FG="./src/fg"; THREADS="${THREADS:-64}"
: "${DEEP100_X:?set DEEP100_X/Q/G}"
mkdir -p results/box
{ echo "### RUN deep100_T${T:-8}_ef400"
  "$FG" --data "$DEEP100_X" --queries "$DEEP100_Q" --gold "$DEEP100_G" \
        --metric l2 --T "${T:-8}" --threads "$THREADS" --harvest 400 \
        --harvest-cap 64 --ef 10 20 30 50 80 120 200 400
  echo "### END"
} 2>&1 | tee results/box/p3_deep100m_graft.log
{ echo "### RUN deep100_hnsw_M32"
  ./"src"/hnsw_bench --data "$DEEP100_X" --queries "$DEEP100_Q" \
        --gold "$DEEP100_G" --metric l2 --M 32 --efc 200 \
        --ef 10 20 30 50 80 120 200 400 --threads "$THREADS" \
        --save "${HNSW_SAVE:-deep100_m32.hnsw}"
  echo "### END"
} 2>&1 | tee results/box/p3_deep100m_hnsw.log

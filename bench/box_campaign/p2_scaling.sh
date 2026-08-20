#!/usr/bin/env bash
# P2 -- the parallelism figure: build wall vs threads, phases split;
# then search QPS vs threads (the old 32->64 regression recheck).
set -euo pipefail
cd "$(dirname "$0")/../.."
FG="./src/fg"
: "${DEEP10_X:?set DEEP10_X/Q/G}"
mkdir -p results/box
{ for THR in 1 4 16 32 64; do
    echo "### RUN scale_thr${THR}"
    "$FG" --data "$DEEP10_X" --queries "$DEEP10_Q" --gold "$DEEP10_G" \
          --metric l2 --T 8 --threads $THR --harvest 400 --harvest-cap 64 \
          --ef 100
    echo "### END scale_thr${THR}"
  done
} 2>&1 | tee results/box/p2_build_scaling.log
# Search-thread scaling: rerun the ef-sweep at --threads 16/32/64 on the same
# config and record QPS(mt) rows; note whether 64 regresses vs 32.
{ for THR in 16 32 64; do
    echo "### RUN qps_thr${THR}"
    "$FG" --data "$DEEP10_X" --queries "$DEEP10_Q" --gold "$DEEP10_G" \
          --metric l2 --T 8 --threads $THR --harvest 400 --harvest-cap 64 \
          --ef 50 120 400
    echo "### END qps_thr${THR}"
  done
} 2>&1 | tee results/box/p2_qps_scaling.log

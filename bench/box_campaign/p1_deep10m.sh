#!/usr/bin/env bash
# P1 -- deep-10M: GRAFT T-sweep + head-to-head at matched recall.
# DEEP is L2, d=96. Set DEEP10_{X,Q,G} to the July campaign's npy paths.
set -euo pipefail
cd "$(dirname "$0")/../.."
FG="./src/fg"; THREADS="${THREADS:-64}"
: "${DEEP10_X:?set DEEP10_X/Q/G to the deep-10M npy paths}"
mkdir -p results/box
{ for T in 4 8 16; do
    echo "### RUN deep10_T${T}_ef400"
    "$FG" --data "$DEEP10_X" --queries "$DEEP10_Q" --gold "$DEEP10_G" \
          --metric l2 --T $T --threads "$THREADS" --harvest 400 \
          --harvest-cap 64 --ef 10 20 30 50 80 120 200 400
    echo "### END deep10_T${T}_ef400"
  done
  echo "### RUN deep10_bestT_ef600"
  "$FG" --data "$DEEP10_X" --queries "$DEEP10_Q" --gold "$DEEP10_G" \
        --metric l2 --T 8 --threads "$THREADS" --harvest 600 \
        --harvest-cap 64 --ef 10 20 30 50 80 120 200 400
  echo "### END deep10_bestT_ef600"
  echo "### RUN deep10_hnsw_M32"
  ./"src"/hnsw_bench --data "$DEEP10_X" --queries "$DEEP10_Q" \
        --gold "$DEEP10_G" --metric l2 --M 32 --efc 200 \
        --ef 10 20 30 50 80 120 200 400 --threads "$THREADS"
  echo "### END deep10_hnsw_M32"
} 2>&1 | tee results/box/p1_deep10m.log
# If ParlayANN is built: add Vamana R64/L128/a1.05 2-pass (their deep recipe)
# + PiPNN defaults, same fbin/gt pipeline as bench/npy_to_fbin.py+gold_to_gt.py.

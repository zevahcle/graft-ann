#!/usr/bin/env bash
# P6 -- post-P3 follow-ups, in priority order.
#
# (1) THE CHEAP FIX TEST. All deep-10M quality so far is on graphs built at
#     harvest-ef 400 (863 s). Timing says harvest-ef 50 costs 170 s, but its
#     QUALITY has never been measured. The T-rule's logic (low intrinsic
#     dimension needs less of everything; SIFT's T32/ef600 was indistinguishable
#     from T4/ef400) predicts deep tolerates a much cheaper harvest. If it does:
#     build ~260-320 s < HNSW's 550 s WITH the 12-39% distance win intact, and
#     the whole "harvest doesn't scale" thread resolves to "we ran deep at GloVe
#     settings".
# (2) THE DECISIVE BASELINE. GRAFT beating HNSW by 19% is numerically the same
#     as GRAFT tying Vamana while HNSW carries its usual deficit (laptop ladder:
#     HNSW was 22% worse than Vamana on SIFT, 40-65% on GIST). Vamana on
#     deep-10M with its own tuned recipe is what settles it.
# (3) HNSW FAIRNESS. M=32 is probably not its best high-recall point; the 39%
#     figure at recall 0.996 is the one most likely to shrink at M=48/64. We
#     applied this same M-sweep discipline on GloVe.
set -euo pipefail
cd "$(dirname "$0")/../.."
FG="./src/fg"; THREADS="${THREADS:-64}"
: "${DEEP10_X:?set DEEP10_X/Q/G}"
mkdir -p results/box
EFS="10 20 30 50 80 120 200 400"

{ for CFG in "4 50" "4 128" "8 128"; do
    set -- $CFG
    echo "### RUN deep10_T$1_harvest$2"
    "$FG" --data "$DEEP10_X" --queries "$DEEP10_Q" --gold "$DEEP10_G" \
          --metric l2 --T "$1" --threads "$THREADS" --harvest "$2" \
          --harvest-cap 64 --ef $EFS
    echo "### END deep10_T$1_harvest$2"
  done
} 2>&1 | tee results/box/p6_harvest_ef_quality.log

# (1b) PATIENCE ARMS (needs the adaptive-harvest branch: git checkout
#      adaptive-harvest && rebuild). Per-point adaptive termination; laptop
#      calibration: SIFT p32 = -20% build Mdist for +2-3% query cost.
{ for P in 32 64; do
    echo "### RUN deep10_T8ef400_p$P"
    "$FG" --data "$DEEP10_X" --queries "$DEEP10_Q" --gold "$DEEP10_G" \
          --metric l2 --T 8 --threads "$THREADS" --harvest 400 \
          --harvest-cap 64 --harvest-patience $P --ef $EFS
    echo "### END deep10_T8ef400_p$P"
  done
} 2>&1 | tee results/box/p6_patience.log

{ for M in 48 64; do
    echo "### RUN deep10_hnsw_M$M"
    ./"src"/hnsw_bench --data "$DEEP10_X" --queries "$DEEP10_Q" \
          --gold "$DEEP10_G" --metric l2 --M $M --efc 200 --ef $EFS \
          --threads "$THREADS"
    echo "### END deep10_hnsw_M$M"
  done
} 2>&1 | tee results/box/p6_hnsw_msweep.log

# Vamana (needs ParlayANN built + fbin/gt staged via bench/npy_to_fbin.py and
# bench/gold_to_gt.py; deep recipe from algorithms/vamana/scripts/deep10M):
#   neighbors-vamana_FLOAT_T_EUCLIDEAN -base_path deep10.fbin \
#     -query_path deep10_q.fbin -gt_path deep10_gt100 -file_type bin \
#     -data_type float -dist_func Euclidian -R 64 -L 128 -alpha 1.05 \
#     -num_passes 2 -k 10

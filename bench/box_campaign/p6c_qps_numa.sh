#!/usr/bin/env bash
# P6-(3) -- serve-gap attribution on deep-10M. GRAFT touches FEWER distances
# than HNSW but serves 3-7x lower QPS: is the per-distance cost intrinsic
# compute, or remote-memory latency on this 4-socket Xeon (4 NUMA nodes,
# 8 cores/socket)? fg builds at 64 threads always; --threads sets SEARCH
# parallelism only, and the graph is thread-count-independent (determinism
# invariant), so every cell searches the SAME graph -- QPS is comparable.
# Probe axes: search-thread scaling (the known 32->64 regression) x NUMA
# placement (default scatter vs node-local vs interleaved).
# Per-cell tolerant: a failed cell is recorded, sweep continues.
set -uo pipefail
cd "$(dirname "$0")/../.."
FG="./src/fg"
: "${DEEP10_X:?set DEEP10_X/Q/G}"
EF="30 80 200"   # low/mid/high recall points -- QPS is the signal, not recall
mkdir -p results/box
run () { # $1=label  $2=numactl-prefix  $3=search-threads
  echo "### RUN $1"
  # harvest 100: probe measures RELATIVE serve scaling (thread x NUMA), not
  # absolute QPS matched to P3 -- a lighter graph gives the same attribution
  # at ~1/3 the per-cell rebuild. Graph is deterministic => identical per cell.
  $2 "$FG" --data "$DEEP10_X" --queries "$DEEP10_Q" --gold "$DEEP10_G" \
        --metric l2 --T 8 --threads 64 --search-threads "$3" --harvest 100 --harvest-cap 64 \
        --ef $EF || echo "### CELL FAILED $1 rc=$?"
  echo "### END $1"
}
{
  # A) search-thread scaling, default OS placement (scatter across 4 sockets)
  for TH in 1 8 16 32 64; do run "scale_default_t${TH}" "" "$TH"; done
  # B) node-local: 1 socket, all-local memory (node0 = 16 logical cpus)
  run "local_node0_t16" "numactl --cpunodebind=0 --membind=0" 16
  # C) interleaved pages across all nodes, at the contended thread counts
  run "interleave_t32" "numactl --interleave=all" 32
  run "interleave_t64" "numactl --interleave=all" 64
} 2>&1 | tee results/box/p6c_qps_numa.log

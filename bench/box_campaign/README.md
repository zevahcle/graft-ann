# Scale campaign scripts (Deep-96, 10M/100M)

Measurement scripts for the paper's scale section, run on a 4-socket Xeon
(32 cores / 64 threads, 1.5 TB RAM). `p0` builds and runs the determinism
gate; `p1`-`p6` are the phases (10M T-sweep + HNSW head-to-head, thread
scaling, 100M, Wikipedia-6.35M, harvest-depth knee, QPS/NUMA attribution).
Data paths come from an `env.sh` you write for your machine (see the
`: "${DEEP100_X:?...}"` guards at the top of each script). Raw output logs
for the paper's tables are committed under `results/box/`.

Note: the determinism hash is ISA-specific (FP reduction order differs
between x86 AVX and ARM NEON); the gate checks thread-count invariance on
one machine, and cross-machine validation is statistical.

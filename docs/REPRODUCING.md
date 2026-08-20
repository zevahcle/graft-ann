# Reproducing the paper

Every table in the paper is backed by a committed log under `results/`, and
`bench/parse_runs.py` regenerates the comparison tables from those logs. This
file maps paper sections to the commands and the evidence.

Hardware note: 1M-scale numbers were measured on a 10-core Apple-silicon
laptop; the 10M/100M scale campaign on a 4-socket Xeon (32 cores / 64
threads). Distance counts are hardware-independent; wall-clock is not (the
paper flags every wall-clock caveat inline).

## 0. Build and gate

```bash
./build.sh    # builds src/{fg,hnsw_bench}; runs the determinism smoke gate
```

The determinism gate (`--check-determinism`) rebuilds at 1 and max threads
and compares graph hashes. The hash is ISA-specific (FP reduction order);
thread-count invariance is the contract on any one machine.

## 1. Data

```bash
python3 src/convert_annb.py <ann-benchmarks hdf5> data/<name>
# -> <name>_X.npy  <name>_Q.npy  <name>_gold.npy
```

ann-benchmarks HDF5 files: https://ann-benchmarks.com (Fashion-MNIST, SIFT,
GIST, GloVe). Wikipedia BGE-M3: the SISAP Indexing Challenge release.
Deep-96 10M/100M: `tools/get_data.sh` (big-ann-benchmarks downloader) +
`tools/convert_bigann.py`.

## 2. GRAFT runs (Tables: GloVe head-to-head, ladder, budget structure)

The named profiles: fast = `--T 16 --harvest 400`, quality =
`--T 32 --harvest 600`; `--harvest-cap 64` throughout.

```bash
"src/fg" --data data/glove_X.npy --queries data/glove_Q.npy \
    --gold data/glove_gold.npy --metric cos \
    --T 32 --harvest 600 --harvest-cap 64 --ef 100 200 400 800 1600
```

The `(T, ef_h)` frontier of the budget-structure section is scripted:
`src/sweep.sh`. Committed evidence: `results/e4/harvest_sweep*.log`
(ef_h Pareto), `results/e4/t_sweep*.log` (T sweep),
`results/e7_*.log` (SIFT), `results/e9_*.log` (Fashion/GIST + baselines),
`results/e11_gist_profile.log`, `results/e19_wiki.log` (BGE-M3),
`results/e12_hnsw_*.log` (hnswlib columns).

## 3. Baselines (same vectors, same answer key)

```bash
bench/setup_baselines.sh     # fetches + patches ParlayANN/PiPNN (5 ARM fixes)
bench/npy_to_fbin.py ...     # fg npy -> big-ann .fbin
bench/gold_to_gt.py ...      # answer key -> ParlayANN groundtruth format
```

Authors' per-dataset parameters are in the paper's protocol section.
Committed evidence: `results/e1/` (PiPNN + Vamana on GloVe). On ARM, read
`bench/upstream/` first: parlaylib's work-stealing deque has a
memory-ordering bug (reproducer + patch included, reported upstream).

## 4. Anatomy section

`bench/graph_probe.py` (static properties: completeness, contraction,
monotonicity, edge lengths), `bench/hsp_consistency.py` (occlusion
endorsement), `bench/traffic_probe.py` (beam productive rate),
`bench/greedy_probe.py` (greedy hit rate), `bench/lambda_lines.py`
(the λ-line spectrum). All read graphs in the ParlayANN layout
(`--dump-graph` writes it).

## 5. Substrate control

`fg --substrate-random` replaces the scaffold with a degree-matched seeded
random K-regular graph (trees kept for entry roots + spine only).
Committed evidence: `results/e13_random_substrate.log`.

## 6. Patience (adaptive termination)

`--harvest-patience P`. Committed evidence: `results/e20_patience.log`
(GloVe table), `results/e20b_realloc.log` (ef_h stops binding),
`results/e22_ladder_patience.log` (SIFT).

## 7. Scale campaign (Deep-96, 10M / 100M)

Scripts in `bench/box_campaign/` (see its README; data paths via your own
`env.sh`). Committed evidence in `results/box/`: `p1_deep10m.log` (T-sweep +
HNSW head-to-head), `p2_*.log` (thread scaling), `p3_*.log` (100M, fixed
ef_h 400 + HNSW), `p6a_harvest_ef.log` (harvest-depth knee),
`p7_deep100m_h200.log` and `p8_deep100m_h400p64.log` (the three-arm
100M table), `lp_patience_bakeoff.log` (patience-vs-knee at 10M),
`p6c_qps_numa.log` (QPS/thread/NUMA attribution).

## 8. Tables from logs

```bash
python3 bench/parse_runs.py results/<any>.log
```

parses any `### RUN <tag>` block into the paper's comparison-table format
(build wall, distances/query at interpolated recall targets).

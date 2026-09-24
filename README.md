# GRAFT — Navigable Graphs Grown on Spatial-Approximation Rootstock

Deterministic, embarrassingly parallel construction of navigable proximity
graphs for approximate nearest-neighbor search. GRAFT builds
HNSW/Vamana-class graphs **without incremental insertion**: grow a forest of
spatial-approximation trees (the *rootstock*), freeze their union as a
scaffold, then let every point beam-search the frozen scaffold for itself in
parallel and occlusion-prune the route it traveled into its adjacency (the
*harvest*).

Because the scaffold never mutates, the build has no serial section, no
insertion order, and no lock protocol — and the output graph is **bitwise
identical at any thread count** (enforced by a hash gate, `--check-determinism`).
The construction consumes only distance *comparisons*: there is no
vector-space operation anywhere in the pipeline, so the same code indexes any
metric.

**Paper:** *Graft: Navigable Graphs Grown on Spatial-Approximation
Rootstock*, to appear in **Information Systems**. Every table in the paper is regenerable from the logs committed
under `results/` via `bench/parse_runs.py`.

## Measured results (see the paper for protocol and caveats)

- **Quality at 1M scale** — distances/query at matched recall, identical
  vectors and answer keys, all baselines at their authors' parameters:
  within 5% of ParlayANN's tuned Vamana on SIFT, GloVe, and Wikipedia
  BGE-M3 at working recalls (parity on GloVe: 9,751 vs 9,753 at recall
  0.95); 1.04–1.13× on GIST; 12–25% ahead of PiPNN on the
  high-intrinsic-dimension datasets.
- **Quality at scale** — Deep-96, 10M and 100M points, 64 threads: 9–39%
  *fewer* distances than HNSW (hnswlib) at matched recall, the margin
  growing with the recall target.
- **Build cost** — 0.5–5% of `n` distance evaluations per point
  (20–190× below one sequential scan per point); at 100M the whole build
  spends ~13k distances per point (0.013% of `n`).
- **Honest losses** — hnswlib holds a 3–7× QPS lead at matched recall at
  100M (per-distance serving cost; layout work pending), and the
  partition-based PiPNN builds 4–40× faster on wall-clock where distances
  are cheap. The paper states both plainly.

## Quickstart (Python)

```bash
pip install git+https://github.com/zevahcle/graft-ann   # needs a C++17 compiler
                                                        # macOS: brew install libomp
```

```python
import graft, numpy as np
X = np.random.rand(100_000, 96).astype(np.float32)
idx = graft.build(X, metric="l2", T=8, harvest=200, patience=64, seed=1)
ids, dists = idx.search(X[:10], k=10, ef=100)
idx.graph_hash    # determinism gate: same seed/params -> same hash at ANY thread count
```

`graft.HnswlibStyleIndex` mirrors hnswlib's method names (`init_index`,
`add_items`, `knn_query`, `set_ef`) so existing harnesses port with a
two-line diff — with honest batch semantics: `add_items` accumulates and the
graph is built on the first query. Incremental insertion after build is not
supported yet. Tests: `pytest tests/`.

## Quickstart (CLI)

```bash
./build.sh          # builds src/{fg,hnsw_bench}; auto-fetches hnswlib;
                    # runs a determinism smoke test
```

Synthetic sanity run (no data needed):

```bash
"src/fg" --synthetic clustered --n 200000 --d 128 --T 16 \
    --ef 32 64 128 --check-determinism
```

Real data (ann-benchmarks HDF5 → npy triplet, then build + evaluate):

```bash
python3 src/convert_annb.py glove-100-angular.hdf5 data/glove
"src/fg" --data data/glove_X.npy --queries data/glove_Q.npy \
    --gold data/glove_gold.npy --metric cos \
    --T 32 --harvest 600 --harvest-cap 64 --ef 100 200 400 800
```

The HNSW baseline (`src/hnsw_bench`) takes the same data flags with
`--M/--efc/--ef`, so head-to-head runs are one command each. Big-ANN-scale
datasets: `tools/get_data.sh`.

## The knobs (regime rules, measured)

| knob | default | rule of thumb |
|---|---|---|
| `--T` (forest size) | 16 | scale with dataset hardness: 2 (easy) → 32 (GloVe-hard); sweep until completeness plateaus |
| `--harvest` (beam width ef_h) | 400 | the dominant quality lever on hard data; set after T — they don't trade |
| `--harvest-cap` (degree R) | 64 | 48–64 equivalent once ef_h is set |
| `--harvest-alpha` | 1.0 | regime switch: 1.0 on concentrated data, ~1.2 on spread/clustered L2 |
| `--harvest-patience` | off | per-point adaptive stop; **recommended at large n** — a fixed budget tuned at pilot scale under-harvests at deployment scale; patience carries quality across scale (measured 10M→100M) |
| `--max-degree` | off | quality-free hub guard; 128 advisable at d ≳ 500 |
| `--check-determinism` | — | rebuilds at 1 and max threads, compares graph hashes |

Note: the determinism hash is ISA-specific (FP reduction order differs
between AVX and NEON). The gate certifies thread-count invariance on one
machine; cross-machine validation is statistical.

## The `fgraft` branch: construction without search

> Only on the **`fgraft`** branch. `main` is the artifact of the GRAFT
> paper and is unaffected.

GRAFT builds a graph by search: every point harvests its neighbourhood with
a beam over a frozen forest. A beam's distance evaluations are *dependent
random accesses*, which caps construction an order of magnitude below the
machine's dense arithmetic — measured at 20–27× on a 64-thread Xeon at
d ≈ 100, and growing as d^0.4, so it is worst at embedding dimensions.

This branch contains two follow-ups. `--harvest-blocks B` lets the rootstock
mutate in B synchronous blocks, recovering incremental construction's
feedback loop while the graph stays a pure function of (data, seed,
parameters, B). And `--dense L` removes the search from the build entirely:
points join the leaves of their `m` nearest k-means leaders, every pair
inside a leaf is evaluated as a dense product, each point keeps its `C`
nearest, the candidates are pruned by the occlusion rule with the reverse
edges kept, and one spanning tree guarantees connectivity. The pair set is
fixed before the first distance is computed, so the construction is
embarrassingly parallel, bitwise deterministic at any thread count, and runs
at GEMM rate.

```bash
# GloVe-100: parity with tuned Vamana at 0.67x its build time (64 threads)
"src/fg" --data data/glove_X.npy --queries data/glove_Q.npy \
    --gold data/glove_gold.npy --metric cos \
    --dense 240 --dense-m 5 --dense-C 200 --dense-prune 1 \
    --ef 60 120 200 300 400 600
```

| flag | default | what it does |
|---|---|---|
| `--dense L` | off | number of k-means leaders; the pool per point is ≈ `m²n/L` |
| `--dense-m` | 5 | leaves each point joins |
| `--dense-C` | 200 | candidates kept per point |
| `--dense-kmeans` | 10 | Lloyd iterations (sample scales as 50·L) |
| `--dense-prune` | 0 | 1 = prune, add reverse edges, re-prune overflow (**use 1**: worth 8–10% of query cost over 0) |
| `--dense-spine` | 1 | 1 = union one SAT's parent edges after the prune. Without it a graph whose leaves are the clusters shatters (recall 0.002 → 0.987) |
| `--dense-top` / `--dense-probe` | off / 3 | two-level leader hierarchy; makes the membership scan `n(T+pL/T)` instead of `n·L`. A net loss below ~10⁸ (leaf imbalance costs more than it saves) |
| `--dense-cands FILE` | — | load candidate lists computed elsewhere (e.g. a GPU) and run only the prune, spine and evaluation |

**The pool is the only real knob.** Everything else costs seconds. Quality
tracks `m²n/L`: on GloVe a 9% pool reaches Vamana parity and 3% lands a
class below, for 2.2× fewer pairs.

**Determinism is a gate, not a hope.** Every configuration must produce an
identical adjacency hash *and* an identical distance count at 1 and at 64
threads. The distance count earned its place: a `size_t`-indexed `taskloop`
once ran zero iterations under clang, and because routing re-assigns only
~0.2% of a node's members the graph stayed byte-identical — only the count
diverged.

Selected measurements (64-thread Xeon E7-4809 v3, graded on ParlayANN's own
searcher, ten paired blocks where a ratio is quoted):

| corpus | build vs tuned Vamana | quality |
|---|---|---|
| GloVe-100, 1.18M | **0.67×** [0.67, 0.67] | parity |
| Wikipedia BGE-M3, 6.35M × 1024 | **0.48×** | parity to recall 0.97 |
| GIST-960, 1M | 0.68× | parity |
| SIFT-128, 1M (3% pool) | 1.00× | parity |
| Deep-96, 100M | 1.55× | **16% better at recall 0.99** |

Papers are in preparation; this branch is the implementation behind them.

## Repository layout

```
src/        the GRAFT core (fg) and the hnswlib baseline harness
python/     the pybind11 package (pip install .); tests/ the pytest suite
bench/      every experiment in the paper, as logged scripts
  upstream/   parlaylib work-stealing-deque bug: reproducer + patch
  box_campaign/  the 10M/100M scale campaign scripts
tools/      big-ann downloads and format conversion
results/    raw measurement logs behind every table in the paper
docs/       REPRODUCING.md — step-by-step reproduction guide
```

## Status and roadmap

This is a **research artifact with a Python API**: the pybind11 package
above (batch build + search, deterministic), the CLI, the benchmark harness,
and the logs that back the paper. An ann-benchmarks adapter is in
`bench/annb/`. Planned (in order): incremental insert via the harvest code
path, serve-side layout and int8 kernels, binary wheels, and the
metric-native workloads (edit distance, DTW, travel-time) from the paper's
future-work list.

## Baseline port note

Running the ParlayANN/PiPNN baselines on ARM required five fixes, including
a memory-ordering bug in parlaylib's work-stealing deque that makes its
algorithms miscompute on weakly-ordered ISAs (`bench/upstream/`:
reproducer and patch, reported upstream). Apply `bench/setup_baselines.sh`
before trusting any baseline numbers on ARM.

## License and citation

Apache-2.0 (see `LICENSE`, `NOTICE`). If you use GRAFT in research, please
cite the paper (`CITATION.cff`). The follow-up work on the `fgraft` branch
is in preparation; cite the repository and branch until those papers
appear.

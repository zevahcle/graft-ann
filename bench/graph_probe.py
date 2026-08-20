#!/usr/bin/env python3
# Copyright 2026 Edgar Chávez and contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""graph_probe.py -- what makes one ANN graph cheaper to walk than another.

Reads graphs in the ParlayANN layout (u32 n, u32 max_degree, u32 sizes[n], then
the flattened adjacency) -- which `fg --dump-graph` also writes -- and measures
three properties that separate the competing explanations for a
distances-per-query gap at matched degree:

  1. PER-HOP CONTRACTION. For sampled (node a, query q) pairs stratified by
     d(a,q), the best single-hop progress min_b d(b,q) / d(a,q) over b in N(a).
     This is Vamana's alpha-reachability quantity: a graph whose edges make
     geometric progress contracts by a constant factor per hop. If one graph
     contracts markedly worse, its jumps are too short -- the monotone path
     exists but is long.

  2. MONOTONICITY RATE. Fraction of those pairs with any neighbour strictly
     closer to q. This is the property greedy search actually needs, and it is
     NOT implied by low stretch (the RNG has unbounded stretch and excellent
     monotonicity).

  3. GREEDY DESCENT LENGTH. Beam-width-1 descent from a fixed entry to a local
     minimum: hops taken, and the ratio of the reached distance to the true
     nearest-neighbour distance. Hops are the direct measure of "monotone but
     slow"; the ratio says whether it got stuck.

Plus the edge-length distribution, normalised by each node's own 10-NN distance,
which says whether a graph spends its degree budget locally or on long links.

    python3 bench/graph_probe.py data/glove_fg_T32.graph glove_X.npy \
        --queries glove_Q.npy --label fg-T32
"""
import argparse

import numpy as np


def load_graph(path):
    with open(path, "rb") as f:
        n, maxdeg = np.fromfile(f, dtype=np.uint32, count=2)
        sizes = np.fromfile(f, dtype=np.uint32, count=int(n))
        idx = np.fromfile(f, dtype=np.uint32)
    ptr = np.zeros(int(n) + 1, dtype=np.int64)
    np.cumsum(sizes, out=ptr[1:])
    assert ptr[-1] == idx.size, f"{ptr[-1]} edges expected, {idx.size} present"
    return ptr, idx, int(n), int(maxdeg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("graph")
    ap.add_argument("base")
    ap.add_argument("--queries", required=True)
    ap.add_argument("--label", default="graph")
    ap.add_argument("--normalize", action="store_true", default=True)
    ap.add_argument("--nq", type=int, default=200, help="queries for descent")
    ap.add_argument("--per-stratum", type=int, default=40,
                    help="nodes sampled per (query, distance stratum)")
    ap.add_argument("--pool", type=int, default=200000,
                    help="subsample used to locate nodes at a target distance")
    ap.add_argument("--max-hops", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    ptr, idx, n, maxdeg = load_graph(args.graph)
    X = np.load(args.base, mmap_mode="r")
    Q = np.asarray(np.load(args.queries), dtype=np.float32)
    Xf = np.array(X, dtype=np.float32)   # copy: the memmap is read-only
    if args.normalize:
        Xf /= np.maximum(np.linalg.norm(Xf, axis=1, keepdims=True), 1e-30)
        Q /= np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-30)
    deg = np.diff(ptr)
    print(f"\n=== {args.label} ===")
    print(f"n={n} edges={idx.size} avg degree={deg.mean():.1f} "
          f"max={deg.max()} isolated={(deg == 0).sum()}")

    def d_to(v, ids):
        diff = Xf[ids] - v
        return np.sqrt(np.einsum("ij,ij->i", diff, diff))

    # ---- edge lengths, normalised by each source's own 10-NN distance -------
    sample = rng.choice(n, size=2000, replace=False)
    pool = rng.choice(n, size=min(args.pool, n), replace=False)
    Xp = Xf[pool]
    rel, knn_ref = [], []
    for a in sample:
        nb = idx[ptr[a]:ptr[a + 1]]
        if nb.size == 0:
            continue
        dpool = np.sqrt(np.maximum(
            np.sum((Xp - Xf[a]) ** 2, axis=1), 0.0))
        # scale the pool's 10-NN distance to full-database density
        k10 = np.partition(dpool, 10)[10] * (len(pool) / n) ** (1.0 / 20.0)
        dn = d_to(Xf[a], nb)
        rel.append(dn / max(k10, 1e-12))
        knn_ref.append(k10)
    rel = np.concatenate(rel)
    print(f"edge length / (local 10-NN distance): median {np.median(rel):.2f}  "
          f"p90 {np.quantile(rel, 0.9):.2f}  p99 {np.quantile(rel, 0.99):.2f}  "
          f"max {rel.max():.1f}")

    # ---- local completeness: what fraction of a node's true 10-NN are edges -
    # Beam search must VISIT a true neighbour's neighbourhood to return it, so
    # a graph that stores more of the true k-NN as edges collects them in fewer
    # node expansions. Ground truth is over the FULL database, not the pool.
    comp_nodes = rng.choice(n, size=200, replace=False)
    hits = []
    for s in range(0, len(comp_nodes), 25):
        blk = comp_nodes[s:s + 25]
        sims = Xf @ Xf[blk].T                       # [n, blk] inner products
        for c, a in enumerate(blk):
            col = sims[:, c]
            top = np.argpartition(-col, 11)[:11]
            top = top[top != a][:10]
            nb = set(idx[ptr[a]:ptr[a + 1]].tolist())
            hits.append(len(nb.intersection(top.tolist())) / 10.0)
    print(f"local completeness (true 10-NN that are graph neighbours): "
          f"{np.mean(hits):.3f}")

    # ---- per-hop contraction and monotonicity, by distance stratum ---------
    qs = Q[rng.choice(Q.shape[0], size=min(args.nq, Q.shape[0]),
                      replace=False)]
    dq = np.sqrt(np.maximum(np.sum((Xp - qs[0]) ** 2, axis=1), 0.0))
    ref = np.quantile(dq, [0.00002, 0.0005, 0.01, 0.2, 0.6])
    strata = [("very near", ref[0]), ("near", ref[1]), ("mid", ref[2]),
              ("far", ref[3]), ("very far", ref[4])]

    print(f"\n{'stratum':10} {'d(a,q)':>8} {'contract':>9} {'p90':>7} "
          f"{'monotone':>9}")
    print("-" * 47)
    for name, target in strata:
        ratios, mono = [], []
        for q in qs[:60]:
            dpool = np.sqrt(np.maximum(np.sum((Xp - q) ** 2, axis=1), 0.0))
            order = np.argsort(np.abs(dpool - target))[:args.per_stratum]
            for a in pool[order]:
                da = np.linalg.norm(Xf[a] - q)
                nb = idx[ptr[a]:ptr[a + 1]]
                if nb.size == 0 or da <= 0:
                    continue
                best = d_to(q, nb).min()
                ratios.append(best / da)
                mono.append(best < da)
        ratios = np.asarray(ratios)
        print(f"{name:10} {target:8.4f} {np.median(ratios):9.4f} "
              f"{np.quantile(ratios, 0.9):7.4f} {np.mean(mono):9.3f}")

    # ---- greedy (beam width 1) descent from a fixed entry -------------------
    entry = int(pool[np.argmin(np.sum((Xp - Xf[pool].mean(axis=0)) ** 2,
                                      axis=1))])   # pool medoid-ish
    hops, gaps, stuck = [], [], 0
    for q in qs:
        cur, dcur, h = entry, float(np.linalg.norm(Xf[entry] - q)), 0
        while h < args.max_hops:
            nb = idx[ptr[cur]:ptr[cur + 1]]
            if nb.size == 0:
                break
            dn = d_to(q, nb)
            j = int(np.argmin(dn))
            if dn[j] >= dcur:
                break
            cur, dcur, h = int(nb[j]), float(dn[j]), h + 1
        dtrue = np.sqrt(np.maximum(np.sum((Xp - q) ** 2, axis=1), 0.0)).min()
        hops.append(h)
        gaps.append(dcur / max(dtrue, 1e-12))
        stuck += int(dcur > dtrue * 1.05)
    hops = np.asarray(hops)
    gaps = np.asarray(gaps)
    print(f"\ngreedy descent from a single entry ({len(qs)} queries):")
    print(f"  hops: median {np.median(hops):.0f}  p90 {np.quantile(hops, 0.9):.0f}"
          f"  max {hops.max()}")
    print(f"  reached / true-NN distance (pool NN): median {np.median(gaps):.3f}"
          f"  p90 {np.quantile(gaps, 0.9):.3f}")
    print(f"  local minima above 1.05x true: {stuck}/{len(qs)}")


if __name__ == "__main__":
    main()

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

r"""funnel_census.py -- the funnel-edge census (cross-certification matrix).

The paper's provenance claim (PAPER sec. 5.3 / Definition "funnel edge"):
search-built graphs owe their cost advantage to edges acquired from far-start
search paths -- edges (p, u) where u lies on the expansion sequence E(p) of a
beam search FOR p, outside p's own kNN.  Until now this was established by
elimination + intervention; this script counts the edges themselves.

Circularity control.  "u is on an approach path to p" needs a graph to search.
Certifying a graph's edges against ITS OWN search paths entangles the
measurement with the graph's routing quality, so the census is a MATRIX:
every audited graph's edges are classified against the expansion sequences of
EVERY oracle graph (the audited graphs plus, optionally, the scaffold).  The
diagonal is the circular (self-certified) number and is reported as such; the
off-diagonal columns are the fair ones.  The scaffold column is disclosed as
near-tautological for the harvested graph (its edges were selected from
exactly those sequences).

Per sampled point p and oracle O:
    E_O(p) = expansion sequence of a width-EF beam search for p over O,
             seeded at ENTRIES fixed random vertices (shared across oracles
             and points; far starts by concentration), with vertex p itself
             excluded from evaluation.
Per audited graph G, the out-edges (p, u) of p are classified:
    nn         u in exact kNN10(p)
    certified  u in E_O(p) \ kNN10(p)      (funnel-certified against O)
    other      neither.
For certified and other endpoints the distance character d(u,p)/r10(p) and
(for certified) the expansion index of u in E_O(p) are recorded, so that
"outer" certified edges (ratio > OUTER) -- the funnel proper -- can be
separated from near-tail certification.

    python3 bench/funnel_census.py glove \
        --graphs graft=data/glove_hs_h600c64.graph \
                 vamana=data/glove_vamana.graph \
                 pipnn=data/glove_pipnn_r1.graph \
                 control=data/glove_best_pc96.graph \
        --oracle scaffold=data/glove_scaf_full_T32.graph \
        --np 1000 --ef 400 --out results/funnel_census
"""
import argparse
import heapq
import json
import pathlib
import time

import numpy as np

K = 10          # kNN size for the "neighbor edge" class (paper's k)
M = 200         # rank horizon: endpoints with exact rank > M are "far".
                # Rank-based, not metric: on concentrated data every point
                # is at nearly the same distance (the lambda-line result),
                # so a radius cannot separate approach from arrival, but
                # neighbor RANK can.  The funnel proper = edges to far
                # endpoints that lie on the oracle's expansion sequence.
OUTER = 1.25    # d(u,p)/r10(p) above this = "outer" edge (kept in JSON)


def load_graph(path):
    with open(path, "rb") as f:
        n, maxdeg = np.fromfile(f, dtype=np.uint32, count=2)
        sizes = np.fromfile(f, dtype=np.uint32, count=int(n))
        idx = np.fromfile(f, dtype=np.uint32)
    ptr = np.zeros(int(n) + 1, dtype=np.int64)
    np.cumsum(sizes, out=ptr[1:])
    assert ptr[-1] == idx.size, f"{path}: {ptr[-1]} edges expected, {idx.size}"
    return ptr, idx, int(n)


def beam_expansions(dfun, entries, ptr, idx, ef, exclude, n):
    """Best-first beam search; `dfun(ids)` returns distances from the query
    to the given vertex ids.  Returns the expansion sequence (vertices
    popped and expanded, in order).  Vertex `exclude` is never evaluated,
    so the search must APPROACH p, not land on it."""
    visited = np.zeros(n, dtype=bool)
    cand, res = [], []          # min-heap (d, u); max-heap (-d, u) capped at ef
    ent = [u for u in entries if u != exclude]
    de = dfun(np.asarray(ent, dtype=np.int64))
    for d, u in zip(de.tolist(), ent):
        if visited[u]:
            continue
        visited[u] = True
        heapq.heappush(cand, (d, u))
        heapq.heappush(res, (-d, u))
    expanded = []
    while cand:
        d, u = heapq.heappop(cand)
        if len(res) >= ef and d > -res[0][0]:
            break
        expanded.append(u)
        nb = idx[ptr[u]:ptr[u + 1]]
        nb = nb[~visited[nb]]
        if nb.size == 0:
            continue
        visited[nb] = True
        dv = dfun(nb)
        for dd, uu in zip(dv.tolist(), nb.tolist()):
            if uu == exclude:
                continue
            if len(res) < ef or dd < -res[0][0]:
                heapq.heappush(cand, (dd, uu))
                heapq.heappush(res, (-dd, uu))
                if len(res) > ef:
                    heapq.heappop(res)
    return expanded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", help="dataset tag (glove, sift, ...)")
    ap.add_argument("--base", default=None,
                    help="base vectors .npy (default data/<ds>_X.npy)")
    ap.add_argument("--graphs", nargs="+", required=True,
                    help="name=path of every audited graph (each is also "
                         "an oracle)")
    ap.add_argument("--oracle", nargs="*", default=[],
                    help="name=path of oracle-only graphs (e.g. scaffold)")
    ap.add_argument("--metric", choices=["cos", "l2"], default="cos",
                    help="cos: vectors are normalized and the distance is "
                         "the chord; l2: raw Euclidean (SIFT, GIST)")
    ap.add_argument("--np", type=int, default=1000, dest="npoints")
    ap.add_argument("--ef", type=int, default=400)
    ap.add_argument("--entries", type=int, default=8,
                    help="fixed random entry vertices, shared everywhere")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/funnel_census")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    base = args.base or f"data/{args.dataset}_X.npy"
    X = np.array(np.load(base, mmap_mode="r"), dtype=np.float32)
    if args.metric == "cos":
        X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-30)
        sqn = None
    else:
        sqn = np.einsum("ij,ij->i", X, X)
    n = X.shape[0]

    def make_dfun(p):
        """distances from database point p to given vertex ids."""
        q = X[p]
        if sqn is None:
            return lambda ids: np.sqrt(np.maximum(
                2.0 - 2.0 * (X[ids] @ q), 0.0))
        qn = sqn[p]
        return lambda ids: np.sqrt(np.maximum(
            sqn[ids] + qn - 2.0 * (X[ids] @ q), 0.0))

    audited = [g.split("=", 1) for g in args.graphs]
    oracles = audited + [g.split("=", 1) for g in args.oracle]
    graphs = {}
    for name, path in oracles:
        graphs[name] = load_graph(path)
        gn = graphs[name][2]
        assert gn == n, f"{name}: n={gn} but base has {n}"
        print(f"loaded {name:10s} avg degree "
              f"{np.diff(graphs[name][0]).mean():.1f}")

    entries = rng.choice(n, size=args.entries, replace=False).tolist()
    sample = np.sort(rng.choice(n, size=args.npoints, replace=False))
    print(f"n={n}  sample={len(sample)}  entries={entries}  ef={args.ef}")

    # ---- exact top-M + r10 for every sampled point, blocked ---------------
    t0 = time.time()
    knn = {}       # p -> set of exact top-K ids
    topm = {}      # p -> set of exact top-M ids
    r10 = {}
    B = 128
    for s in range(0, len(sample), B):
        blk = sample[s:s + B]
        sims = X @ X[blk].T                              # [n, |blk|]
        for c, p in enumerate(blk):
            if sqn is None:
                d2 = 2.0 - 2.0 * sims[:, c]              # monotone in chord
            else:
                d2 = sqn + sqn[p] - 2.0 * sims[:, c]     # squared L2
            top = np.argpartition(d2, M + 1)[:M + 1]
            top = top[top != p]
            top = top[np.argsort(d2[top], kind="stable")][:M]
            knn[int(p)] = set(int(t) for t in top[:K])
            topm[int(p)] = set(int(t) for t in top)
            r10[int(p)] = float(np.sqrt(max(float(d2[top[K - 1]]), 0.0)))
    print(f"exact top-{M}: {time.time()-t0:.1f}s")

    # ---- expansion sequences: oracle x point ------------------------------
    # exp[oracle][p] = {u: expansion index}; tstar[oracle][p] = index of the
    # first expansion inside p's exact top-M (arrival of the search).
    exp = {name: {} for name in graphs}
    tstar = {name: {} for name in graphs}
    for name, (ptr, idx, _) in graphs.items():
        t0 = time.time()
        for p in sample:
            p = int(p)
            seq = beam_expansions(make_dfun(p), entries, ptr, idx,
                                  args.ef, p, n)
            exp[name][p] = {int(u): i for i, u in enumerate(seq)}
            tm = topm[p]
            ts = len(seq)
            for i, u in enumerate(seq):
                if u in tm:
                    ts = i
                    break
            tstar[name][p] = ts
        print(f"oracle {name:10s} {len(sample)} searches "
              f"{time.time()-t0:.1f}s  (median |E| "
              f"{int(np.median([len(v) for v in exp[name].values()]))}, "
              f"median t* "
              f"{int(np.median(list(tstar[name].values())))})")

    # ---- pass 1: rank-classify every audited graph's edges ---------------
    # Edge classes by exact neighbor rank of the endpoint around p:
    #   nn    rank <= K
    #   near  K < rank <= M
    #   far   rank > M   -- the funnel candidates; certification is
    #                       DIRECTIONAL and tested both ways below:
    #                       forward  u in E(p)  (u on the path to p)
    #                       reverse  p in E(u)  (p on the path to u --
    #                       the backlink of u's harvest/insert search).
    comp = {}                       # gname -> composition counts
    far_pairs = {}                  # gname -> list of (p, u)
    for gname, gpath in audited:
        ptr, idx, _ = graphs[gname]
        edge_ct = nn_ct = near_ct = 0
        pairs = []
        for p in sample:
            p = int(p)
            nb = idx[ptr[p]:ptr[p + 1]]
            kn, tm = knn[p], topm[p]
            for u in nb.tolist():
                edge_ct += 1
                if u in kn:
                    nn_ct += 1
                elif u in tm:
                    near_ct += 1
                else:
                    pairs.append((p, u))
        comp[gname] = dict(edges=edge_ct, nn=nn_ct, near=near_ct,
                           far=len(pairs))
        far_pairs[gname] = pairs

    # ---- null model: random far endpoints (the chance floor) -------------
    NULLR = 5
    null_pairs = []
    for p in sample:
        p = int(p)
        for _ in range(NULLR):
            u = int(rng.integers(n))
            while u == p or u in topm[p]:
                u = int(rng.integers(n))
            null_pairs.append((p, u))
    far_pairs["null"] = null_pairs
    comp["null"] = dict(edges=len(null_pairs), nn=0, near=0,
                        far=len(null_pairs))

    # ---- pass 2: reverse certification -- searches from far endpoints ----
    # For every distinct far endpoint u (union over graphs), one search per
    # oracle; membership of its partners p in E(u) is recorded on the fly.
    partners = {}                   # u -> set of p that pair with it
    for gname in far_pairs:
        for p, u in far_pairs[gname]:
            partners.setdefault(u, set()).add(p)
    print(f"\nreverse pass: {len(partners)} distinct far endpoints, "
          f"{sum(len(v) for v in partners.values())} (u,p) tests")
    rev = {name: {} for name in graphs}   # oracle -> {(p, u): bool}
    for name, (ptr, idx, _) in graphs.items():
        t0 = time.time()
        for u, ps in partners.items():
            seq = set(beam_expansions(make_dfun(u), entries, ptr, idx,
                                      args.ef, u, n))
            for p in ps:
                rev[name][(p, u)] = p in seq
        print(f"reverse oracle {name:10s} {len(partners)} searches "
              f"{time.time()-t0:.1f}s")

    # ---- aggregate per (graph, oracle) ------------------------------------
    results = {}
    ns = len(sample)
    rows = audited + [("null", "(random far endpoints)")]
    for gname, gpath in rows:
        c = comp[gname]
        for oname in graphs:
            fwd_ct = rev_ct = either_ct = pre_ct = 0
            ei_pts = set()
            for p, u in far_pairs[gname]:
                E = exp[oname][p]
                f = u in E
                b = rev[oname][(p, u)]
                fwd_ct += f
                rev_ct += b
                if f or b:
                    either_ct += 1
                    ei_pts.add(p)
                if f and E[u] < tstar[oname][p]:
                    pre_ct += 1
            results[f"{gname}|{oname}"] = dict(
                edges=c['edges'],
                frac_nn=c['nn'] / c['edges'],
                frac_near=c['near'] / c['edges'],
                frac_far=c['far'] / c['edges'],
                far_per_vertex=c['far'] / ns,
                far_fwd_per_vertex=fwd_ct / ns,
                far_rev_per_vertex=rev_ct / ns,
                far_cert_per_vertex=either_ct / ns,
                far_uncert_per_vertex=(c['far'] - either_ct) / ns,
                frac_far_cert=either_ct / max(c['far'], 1),
                far_cert_pre_per_vertex=pre_ct / ns,
                frac_pts_with_farcert=len(ei_pts) / ns,
            )

    # ---- report -----------------------------------------------------------
    onames = list(graphs.keys())
    print(f"\n=== funnel census: {args.dataset}  (K={K}, M={M}, "
          f"ef={args.ef}; diagonal = self-certified) ===")
    print(f"\n--- edge composition by exact endpoint rank "
          f"(oracle-independent) ---")
    print(f"{'graph':10s} {'deg':>5s} {'%nn':>6s} {'%near':>6s} "
          f"{'%far':>6s}")
    for gname, _ in audited:
        r = results[f"{gname}|{onames[0]}"]
        deg = r['edges'] / len(sample)
        print(f"{gname:10s} {deg:5.1f} {100*r['frac_nn']:6.1f} "
              f"{100*r['frac_near']:6.1f} {100*r['frac_far']:6.1f}")
    print(f"\n--- funnel census: far edges certified in EITHER direction, "
          f"per vertex (% of the graph's far edges) ---")
    print(f"{'graph':10s} {'far/v':>6s} | " +
          " | ".join(f"{o:>15s}" for o in onames))
    for gname, _ in rows:
        r0 = results[f"{gname}|{onames[0]}"]
        cells = []
        for o in onames:
            r = results[f"{gname}|{o}"]
            tag = "*" if o == gname else " "
            cells.append(f"{r['far_cert_per_vertex']:5.2f} "
                         f"({100*r['frac_far_cert']:4.1f}%){tag}")
        print(f"{gname:10s} {r0['far_per_vertex']:6.1f} | " +
              " | ".join(cells))
    print(f"\n--- direction split: forward (u on path to p) / reverse "
          f"(p on path to u), per vertex ---")
    print(f"{'graph':10s} | " + " | ".join(f"{o:>15s}" for o in onames))
    for gname, _ in rows:
        cells = []
        for o in onames:
            r = results[f"{gname}|{o}"]
            tag = "*" if o == gname else " "
            cells.append(f"{r['far_fwd_per_vertex']:5.2f} /"
                         f"{r['far_rev_per_vertex']:5.2f}   {tag}")
        print(f"{gname:10s} | " + " | ".join(cells))

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = dict(dataset=args.dataset, base=base, metric=args.metric,
                K=K, M=M, null_per_point=NULLR,
                ef=args.ef, entries=entries, seed=args.seed,
                npoints=len(sample),
                audited=dict(audited), oracles=dict(oracles))
    with open(out / f"{args.dataset}.json", "w") as f:
        json.dump(dict(meta=meta, results=results), f, indent=1)
    print(f"\nwritten {out}/{args.dataset}.json")


if __name__ == "__main__":
    main()

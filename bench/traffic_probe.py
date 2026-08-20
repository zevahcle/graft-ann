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

"""traffic_probe.py -- per-EDGE traffic under beam search.

Every static property we can measure (degree, edge-weighted degree, local
completeness, per-hop contraction, monotonicity, edge-length distribution,
HSP-endorsement) is now matched or better than Vamana's, and the beam still
expands ~1.4x as many nodes. This measures the one thing those properties do not
capture: which edges a search actually USES, and how often using one pays off.

Per traversal of edge (v -> w), when the beam expands v and evaluates w:
  * traversed   -- the distance d(w,q) was computed (the edge cost us one distance)
  * productive  -- w was good enough to enter the beam
  * decisive    -- w ended up in the final top-k

Reported per graph:
  utilisation      fraction of edges traversed at least once over the whole run
  productive rate  productive traversals / traversals -- the payoff per distance
  decisive rate    decisive traversals / traversals
  concentration    share of all traffic carried by the busiest 1% of edges

A graph whose edges pay off more often needs fewer expansions for the same
recall, which is exactly the residual. Traffic is also the signal a usage-driven
prune would consume, so this doubles as a feasibility check: no spread between
graphs here means there is nothing for such a prune to bite on.

    python3 bench/traffic_probe.py data/glove_fg.graph glove_X.npy \\
        --queries glove_Q.npy --gold glove_gold.npy --ef 200 --nq 500
"""
import argparse
import heapq

import numpy as np


def load_graph(path):
    with open(path, "rb") as f:
        n, _maxdeg = np.fromfile(f, dtype=np.uint32, count=2)
        sizes = np.fromfile(f, dtype=np.uint32, count=int(n))
        idx = np.fromfile(f, dtype=np.uint32)
    ptr = np.zeros(int(n) + 1, dtype=np.int64)
    np.cumsum(sizes, out=ptr[1:])
    assert ptr[-1] == idx.size
    return ptr, idx, int(n)


def beam_search(q, ptr, idx, X, entry, ef, k, stamp, epoch, trav, prod, deci):
    """Best-first beam. Returns (top-k ids, #expansions, #distances)."""
    d0 = float(np.linalg.norm(X[entry] - q))
    cand = [(d0, int(entry))]          # min-heap on distance
    res = [(-d0, int(entry))]          # max-heap on distance (worst at top)
    stamp[entry] = epoch
    expansions = 0
    dists = 1
    edge_of = {}                        # node -> edge slot that first reached it
    while cand:
        dv, v = heapq.heappop(cand)
        if len(res) >= ef and dv > -res[0][0]:
            break
        lo, hi = ptr[v], ptr[v + 1]
        nb = idx[lo:hi]
        fresh = nb[stamp[nb] != epoch]
        if fresh.size:
            stamp[fresh] = epoch
            diff = X[fresh] - q
            dd = np.sqrt(np.einsum("ij,ij->i", diff, diff))
            dists += fresh.size
            # edge slots for the fresh neighbours, in adjacency order
            mask = np.isin(nb, fresh, assume_unique=True)
            slots = lo + np.nonzero(mask)[0]
            trav[slots] += 1
            for w, dw, slot in zip(fresh.tolist(), dd.tolist(), slots.tolist()):
                if len(res) < ef or dw < -res[0][0]:
                    prod[slot] += 1
                    edge_of[w] = slot
                    heapq.heappush(cand, (dw, w))
                    if len(res) < ef:
                        heapq.heappush(res, (-dw, w))
                    else:
                        heapq.heapreplace(res, (-dw, w))
        expansions += 1
    top = [w for _, w in sorted((-a, b) for a, b in res)][:k]
    for w in top:
        if w in edge_of:
            deci[edge_of[w]] += 1
    return top, expansions, dists


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("graph")
    ap.add_argument("base")
    ap.add_argument("--queries", required=True)
    ap.add_argument("--gold")
    ap.add_argument("--label", default="graph")
    ap.add_argument("--ef", type=int, default=200)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--nq", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dump-per-query", help="write per-query [hits, distances] "
                    "as float32 pairs, so two graphs can be compared PAIRED on "
                    "the same query set")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    ptr, idx, n = load_graph(args.graph)
    X = np.array(np.load(args.base, mmap_mode="r"), dtype=np.float32)
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-30)
    Q = np.array(np.load(args.queries), dtype=np.float32)
    Q /= np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-30)
    gold = np.load(args.gold) if args.gold else None

    qsel = rng.choice(Q.shape[0], size=min(args.nq, Q.shape[0]), replace=False)
    # same entry for every graph: the base point nearest the centroid
    pool = rng.choice(n, size=200000, replace=False)
    centroid = X[pool].mean(axis=0)
    entry = int(pool[np.argmin(np.sum((X[pool] - centroid) ** 2, axis=1))])

    trav = np.zeros(idx.size, dtype=np.uint32)
    prod = np.zeros(idx.size, dtype=np.uint32)
    deci = np.zeros(idx.size, dtype=np.uint32)
    stamp = np.zeros(n, dtype=np.uint32)

    hits = tot_exp = tot_dist = 0
    per_q = np.zeros((len(qsel), 2), dtype=np.float32)
    for e, qi in enumerate(qsel, start=1):
        top, exp, dists = beam_search(Q[qi], ptr, idx, X, entry, args.ef,
                                      args.k, stamp, e, trav, prod, deci)
        tot_exp += exp
        tot_dist += dists
        h = 0
        if gold is not None:
            h = len(set(top) & set(gold[qi, :args.k].tolist()))
            hits += h
        per_q[e - 1] = (h, dists)
    if args.dump_per_query:
        per_q.tofile(args.dump_per_query)

    nq = len(qsel)
    T = int(trav.sum())
    order = np.sort(trav)[::-1]
    top1pct = order[:max(1, idx.size // 100)].sum()
    print(f"\n=== {args.label} ===")
    print(f"edges={idx.size}  avg degree={idx.size / n:.1f}  ef={args.ef}  "
          f"queries={nq}")
    if gold is not None:
        print(f"recall@{args.k} {hits / (nq * args.k):.4f}   "
              f"expansions/query {tot_exp / nq:.0f}   "
              f"distances/query {tot_dist / nq:.0f}")
    print(f"utilisation     {np.mean(trav > 0) * 100:5.2f}% of edges traversed")
    print(f"productive rate {prod.sum() / max(T, 1) * 100:5.2f}%  "
          f"(traversals whose endpoint entered the beam)")
    print(f"decisive rate   {deci.sum() / max(T, 1) * 100:5.3f}%  "
          f"(traversals that delivered a final top-{args.k} answer)")
    print(f"concentration   busiest 1% of edges carry "
          f"{top1pct / max(T, 1) * 100:5.2f}% of traffic")


if __name__ == "__main__":
    main()

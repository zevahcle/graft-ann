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

"""greedy_probe.py -- edge credit along SUCCESSFUL greedy paths.

traffic_probe.py scored an edge productive when its endpoint entered the beam.
That is nearly noise: at ef=200 roughly 880 candidates enter the beam and 10 are
the answer, so ~99% of "productive" traversals lead nowhere, and the stronger
"decisive" counter credited only the final edge into an answer node -- never the
path that carried the search into position.

This measures routing instead. Pure greedy (beam width 1) from a random entry
produces a PATH; the run succeeds if it terminates on a true k-NN of the query.
Only the edges of successful paths earn credit, so credit is assigned to the
route, not to every candidate that looked plausible for one step. Greedy gets
stuck on this data, so runs are repeated with random restarts, and the number of
restarts needed is itself the navigability measure.

Reported per graph:
  hit rate/run       fraction of single greedy runs that land on a true k-NN
  restarts to hit    expected restarts before the first success
  path length        hops on successful paths (short = direct routing)
  routing edges      distinct edges carrying at least one successful path
  credit skew        share of successful-path credit on the busiest 1% of edges

    python3 bench/greedy_probe.py data/g.graph glove_X.npy \\
        --queries glove_Q.npy --gold glove_gold.npy --restarts 32
"""
import argparse

import numpy as np


def load_graph(path):
    with open(path, "rb") as f:
        n, _ = np.fromfile(f, dtype=np.uint32, count=2)
        sizes = np.fromfile(f, dtype=np.uint32, count=int(n))
        idx = np.fromfile(f, dtype=np.uint32)
    ptr = np.zeros(int(n) + 1, dtype=np.int64)
    np.cumsum(sizes, out=ptr[1:])
    return ptr, idx, int(n)


def greedy_walk(q, start, ptr, idx, X, max_hops):
    """Beam-width-1 descent. Returns (final node, edge slots on the path)."""
    cur = int(start)
    dcur = float(np.linalg.norm(X[cur] - q))
    path = []
    for _ in range(max_hops):
        lo, hi = ptr[cur], ptr[cur + 1]
        nb = idx[lo:hi]
        if nb.size == 0:
            break
        diff = X[nb] - q
        dd = np.einsum("ij,ij->i", diff, diff)
        j = int(np.argmin(dd))
        dj = float(np.sqrt(dd[j]))
        if dj >= dcur:
            break                      # local minimum: greedy stops here
        path.append(int(lo + j))
        cur, dcur = int(nb[j]), dj
    return cur, path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("graph")
    ap.add_argument("base")
    ap.add_argument("--queries", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--label", default="graph")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--nq", type=int, default=200)
    ap.add_argument("--restarts", type=int, default=32)
    ap.add_argument("--max-hops", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dump-runs", help="write the per-run success vector (uint8) "
                    "so two graphs can be compared PAIRED: the query set and the "
                    "start nodes are graph-independent, so the same (query, start) "
                    "run exists in both and McNemar's test on the discordant pairs "
                    "is far more powerful than comparing two rates")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    ptr, idx, n = load_graph(args.graph)
    X = np.array(np.load(args.base, mmap_mode="r"), dtype=np.float32)
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-30)
    Q = np.array(np.load(args.queries), dtype=np.float32)
    Q /= np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-30)
    gold = np.load(args.gold)

    qsel = rng.choice(Q.shape[0], size=min(args.nq, Q.shape[0]), replace=False)
    # identical entry sequence for every graph, so differences are the graph's
    entries = np.random.default_rng(12345).choice(
        n, size=(len(qsel), args.restarts), replace=True)

    credit = np.zeros(idx.size, dtype=np.uint32)
    runs = hits = 0
    first_hit_restart, path_lens, hops_all = [], [], []
    q_solved = 0
    # where FAILED descents land. Both graphs succeed equally often; a beam has
    # to rescue every failure, so what costs expansions is how far away the
    # failures stop. Measured in units of the query's own neighbourhood width,
    # (d(term,q) - r1) / (r10 - r1), which is scale-free under concentration.
    fail_basin, fail_ratio, fail_in100 = [], [], []
    run_ok = np.zeros(len(qsel) * args.restarts, dtype=np.uint8)
    for i, qi in enumerate(qsel):
        q = Q[qi]
        truth = set(gold[qi, :args.k].tolist())
        top100 = set(gold[qi].tolist())
        r1 = float(np.linalg.norm(X[gold[qi, 0]] - q))
        r10 = float(np.linalg.norm(X[gold[qi, args.k - 1]] - q))
        width = max(r10 - r1, 1e-9)
        solved_at = None
        for r in range(args.restarts):
            end, path = greedy_walk(q, entries[i, r], ptr, idx, X,
                                    args.max_hops)
            runs += 1
            hops_all.append(len(path))
            if end in truth:
                run_ok[i * args.restarts + r] = 1
                hits += 1
                credit[np.asarray(path, dtype=np.int64)] += 1
                path_lens.append(len(path))
                if solved_at is None:
                    solved_at = r + 1
            else:
                dend = float(np.linalg.norm(X[end] - q))
                fail_basin.append((dend - r1) / width)
                fail_ratio.append(dend / max(r1, 1e-9))
                fail_in100.append(end in top100)
        if solved_at:
            q_solved += 1
            first_hit_restart.append(solved_at)

    if args.dump_runs:
        run_ok.tofile(args.dump_runs)
    C = int(credit.sum())
    order = np.sort(credit)[::-1]
    top1 = order[:max(1, idx.size // 100)].sum()
    print(f"\n=== {args.label} ===")
    print(f"edges={idx.size}  avg degree={idx.size / n:.1f}  "
          f"queries={len(qsel)}  restarts={args.restarts}")
    print(f"hit rate/run      {hits / max(runs, 1) * 100:6.2f}%   "
          f"({hits}/{runs} greedy runs land on a true {args.k}-NN)")
    print(f"queries ever hit  {q_solved / len(qsel) * 100:6.2f}%   "
          f"restarts to first hit: median "
          f"{np.median(first_hit_restart) if first_hit_restart else float('nan'):.1f}")
    print(f"hops: all runs    median {np.median(hops_all):.1f}   "
          f"successful runs median "
          f"{np.median(path_lens) if path_lens else float('nan'):.1f}")
    print(f"routing edges     {np.mean(credit > 0) * 100:6.3f}% of edges carry "
          f"a successful path ({int((credit > 0).sum())} edges)")
    print(f"credit skew       busiest 1% hold {top1 / max(C, 1) * 100:5.1f}% "
          f"of successful-path credit")
    if fail_basin:
        fb = np.asarray(fail_basin)
        fr = np.asarray(fail_ratio)
        print(f"FAILED descents   {len(fb)} runs | land in true top-100: "
              f"{np.mean(fail_in100) * 100:5.2f}%")
        print(f"  basin depth (neighbourhood widths past the 1-NN): "
              f"median {np.median(fb):6.2f}  p75 {np.quantile(fb, .75):6.2f}  "
              f"p90 {np.quantile(fb, .90):6.2f}")
        print(f"  d(end,q)/d(1-NN,q): median {np.median(fr):5.3f}  "
              f"p90 {np.quantile(fr, .90):5.3f}")


if __name__ == "__main__":
    main()

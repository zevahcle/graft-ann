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

"""hsp_consistency.py -- how much of a node's edge budget the HSP rule endorses.

The half-space-proximal / RNG rule selects out-edges FROM THE SOURCE'S point of
view: process candidates nearest-first and keep u unless some already-kept
neighbour g shadows it, i.e. alpha * d(u,g) < d(u,v). The justification is
one-directional, so a REVERSE edge added by symmetrising someone else's choice
was never endorsed from this node's side.

For each sampled node this replays the rule over the node's actual adjacency list
and reports the survivor fraction. A graph built by applying the rule to its own
final lists (Vamana's RobustPrune) should be near-self-consistent. A graph that
symmetrises tree edges and then caps at random should not: roughly half of a
tree's edges are child->parent reverses that no rule ever approved.

"endorsed degree" = survivors x average degree: the part of the budget that is
actually buying directional cover.

    python3 bench/hsp_consistency.py data/glove_fg_T32.graph glove_X.npy --label fg
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
    return ptr, idx, int(n)


def hsp_survivors(v, nb, X, alpha):
    """Replay the HSP rule over an existing adjacency list. Returns #survivors."""
    P = X[nb]
    dv = np.sqrt(np.maximum(np.sum((P - X[v]) ** 2, axis=1), 0.0))
    order = np.argsort(dv, kind="stable")
    P, dv = P[order], dv[order]
    # pairwise distances among the candidates
    g2 = np.sum(P * P, axis=1)
    D = np.sqrt(np.maximum(g2[:, None] + g2[None, :] - 2.0 * (P @ P.T), 0.0))
    kept = []
    for i in range(len(dv)):
        if kept and np.any(alpha * D[i, kept] < dv[i]):
            continue
        kept.append(i)
    return len(kept)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("graph")
    ap.add_argument("base")
    ap.add_argument("--label", default="graph")
    ap.add_argument("--nodes", type=int, default=400)
    ap.add_argument("--alphas", default="1.0,1.2")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    ptr, idx, n = load_graph(args.graph)
    X = np.array(np.load(args.base, mmap_mode="r"), dtype=np.float32)
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-30)
    deg = np.diff(ptr)

    nodes = rng.choice(n, size=args.nodes, replace=False)
    out = [f"{args.label:20} avg degree {deg.mean():5.1f}"]
    for alpha in [float(a) for a in args.alphas.split(",")]:
        fracs, kept_deg = [], []
        for v in nodes:
            nb = idx[ptr[v]:ptr[v + 1]]
            if nb.size < 2:
                continue
            s = hsp_survivors(int(v), nb, X, alpha)
            fracs.append(s / nb.size)
            kept_deg.append(s)
        out.append(f"  alpha={alpha:.1f}: HSP-endorsed {np.mean(fracs) * 100:5.1f}% "
                   f"of edges -> endorsed degree {np.mean(kept_deg):5.1f} "
                   f"(dead weight {deg.mean() - np.mean(kept_deg):5.1f})")
    print("\n".join(out))


if __name__ == "__main__":
    main()

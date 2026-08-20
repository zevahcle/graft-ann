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

"""gold_to_gt.py -- ann-benchmarks neighbour ids -> ParlayANN groundtruth file.

ParlayANN/big-ann groundtruth layout:
    uint32 nq, uint32 k, then nq*k uint32 ids, then nq*k float32 distances.

The ids come from the dataset's own brute-force `neighbors` matrix (the same
gold the fg harness scores against, so both systems are graded by the SAME
answer key). Distances are recomputed here as squared L2 on L2-NORMALISED
vectors, which is rank-identical to the angular/cosine order for `*-angular`
datasets. (ParlayANN recomputes distances from the base points when expanding
ties, so the stored values only mark "distances present".)

    python3 gold_to_gt.py glove_gold.npy glove_X.npy glove_Q.npy out_gt \
        --normalize --k 100
"""
import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gold", help="ann-benchmarks neighbour ids, .npy [nq, k]")
    ap.add_argument("base", help="base vectors, .npy [n, d]")
    ap.add_argument("query", help="query vectors, .npy [nq, d]")
    ap.add_argument("out", help="output groundtruth path")
    ap.add_argument("--normalize", action="store_true")
    ap.add_argument("--k", type=int, default=0, help="0 = all columns of gold")
    args = ap.parse_args()

    gold = np.load(args.gold)
    X = np.load(args.base, mmap_mode="r")
    Q = np.load(args.query)
    k = args.k if args.k else gold.shape[1]
    gold = np.ascontiguousarray(gold[:, :k], dtype=np.uint32)
    nq = gold.shape[0]

    Q = np.asarray(Q, dtype=np.float32)
    if args.normalize:
        Q = Q / np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-30)

    dists = np.empty((nq, k), dtype=np.float32)
    for i in range(nq):
        nbr = np.asarray(X[gold[i]], dtype=np.float32)
        if args.normalize:
            nbr = nbr / np.maximum(np.linalg.norm(nbr, axis=1, keepdims=True),
                                   1e-30)
        diff = nbr - Q[i]
        dists[i] = np.einsum("ij,ij->i", diff, diff)

    # sanity: the gold order must be non-decreasing in distance
    bad = int((np.diff(dists, axis=1) < -1e-5).sum())
    with open(args.out, "wb") as f:
        np.array([nq, k], dtype=np.uint32).tofile(f)
        gold.tofile(f)
        dists.tofile(f)
    print(f"{args.out}: nq={nq} k={k} normalize={args.normalize} "
          f"out-of-order pairs={bad}")


if __name__ == "__main__":
    main()

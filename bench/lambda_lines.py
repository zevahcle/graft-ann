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

"""lambda_lines.py -- the lambda-line spectrum of an embedding.

A point y is LAMBDA-BETWEEN x and z when the detour through it is nearly free:

    d(x,y) + d(y,z) <= lambda * d(x,z),        lambda >= 1

with lambda -> 1 recovering exact Menger betweenness (y on a geodesic from x to
z). The test is vacuous without a PROPERNESS constraint, because y = x gives
ratio exactly 1: we therefore require y to be a genuine interior witness,

    min(d(x,y), d(y,z)) >= beta * d(x,z),      beta in (0, 1/2]

For each sampled pair (x,z) this reports the best achievable detour ratio over
ALL database points as witnesses,

    rho(x,z) = min_y (d(x,y) + d(y,z)) / d(x,z)   over proper y,

stratified by the pair distance d(x,z). rho ~ 1 at a given scale means the data
contains near-geodesic chains at that scale -- long edges are "on the way" to
what lies beyond them, and can be certified as highways. rho -> 2 means the pair
is subtended by nothing: the data curves away and a long jump leaves the
manifold.

METRIC. Betweenness needs a true metric, and it must be the RIGHT one. For
angular data the chord distance ||x-y|| on unit vectors is a metric but its
geodesics are straight lines in R^d, which leave the sphere -- so even a
perfectly great-circle chain has a chord detour ratio of 1/cos(phi/2) > 1 that
grows with the arc. The angle itself is the intrinsic metric (great circles are
geodesics, and betweenness on an arc is exact), so `--metric angle` is the
default for `*-angular` data. `--metric chord` is available to show the floor.

    python3 bench/lambda_lines.py glove_X.npy --pairs 300 --beta 0.25
"""
import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base", help="base vectors .npy [n, d]")
    ap.add_argument("--metric", choices=["angle", "chord", "l2"],
                    default="angle")
    ap.add_argument("--pairs", type=int, default=300,
                    help="pairs sampled per distance stratum")
    ap.add_argument("--beta", type=float, default=0.25,
                    help="properness: witness at least beta*d(x,z) from each end")
    ap.add_argument("--lam", type=float, default=1.05,
                    help="lambda for the witness-count statistic")
    ap.add_argument("--max-n", type=int, default=400000,
                    help="witness pool size (points, subsampled from the base)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    X = np.load(args.base, mmap_mode="r")
    n = X.shape[0]
    pool_ids = np.sort(rng.choice(n, size=min(args.max_n, n), replace=False))
    P = np.ascontiguousarray(X[pool_ids], dtype=np.float32)
    if args.metric in ("angle", "chord"):
        P /= np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-30)
    m = P.shape[0]
    print(f"{args.base}: n={n}, witness pool={m}, metric={args.metric}, "
          f"beta={args.beta}")

    def dist_to(A):
        """[m, |A|] distances from every pool point to each row of A."""
        if args.metric == "angle":
            g = np.clip(P @ A.T, -1.0, 1.0)
            return np.arccos(g)
        g = P @ A.T
        na = np.sum(A * A, axis=1)[None, :]
        np_ = np.sum(P * P, axis=1)[:, None]
        return np.sqrt(np.maximum(np_ + na - 2 * g, 0.0))

    # Distance scale reference: the 10-NN distance and the pair-distance
    # distribution, both measured on the pool.
    probe = P[rng.choice(m, size=256, replace=False)]
    Dp = dist_to(probe)                       # [m, 256]
    knn10 = np.mean(np.partition(Dp, 11, axis=0)[1:11, :])
    all_d = Dp.reshape(-1)
    qs = np.quantile(all_d, [0.001, 0.01, 0.1, 0.5, 0.9])
    print(f"  10-NN distance ~ {knn10:.4f};  pair-distance quantiles "
          f"0.1%/1%/10%/50%/90% = " + " ".join(f"{q:.4f}" for q in qs))

    strata = [("2x kNN", 2 * knn10), ("5x kNN", 5 * knn10),
              ("10x kNN", 10 * knn10), ("q1%", qs[1]), ("q10%", qs[2]),
              ("median", qs[3]), ("q90%", qs[4])]
    seen, uniq = set(), []
    for name, target in strata:
        key = round(float(target), 4)
        if key not in seen and target > 0:
            seen.add(key)
            uniq.append((name, float(target)))

    print(f"\n{'stratum':10} {'d(x,z)':>8} {'rho_min':>8} {'rho_p10':>8} "
          f"{'rho_med':>8} {'#wit<=lam':>10} {'frac pairs':>11}")
    print("-" * 70)
    rows = []
    for name, target in uniq:
        xs = rng.choice(m, size=args.pairs, replace=False)
        A = P[xs]
        DA = dist_to(A)                                   # [m, pairs]
        # partner z: the pool point whose distance to x is closest to target
        zs = np.argmin(np.abs(DA - target), axis=0)
        B = P[zs]
        DB = dist_to(B)                                   # [m, pairs]
        dxz = DA[zs, np.arange(len(zs))]                  # [pairs]

        ratio = (DA + DB) / np.maximum(dxz[None, :], 1e-12)
        proper = (np.minimum(DA, DB) >= args.beta * dxz[None, :])
        ratio = np.where(proper, ratio, np.inf)

        rho = ratio.min(axis=0)
        good = np.isfinite(rho)
        nwit = np.sum(ratio <= args.lam, axis=0)
        rows.append((name, float(np.mean(dxz)), float(np.min(rho[good])),
                     float(np.quantile(rho[good], 0.10)),
                     float(np.median(rho[good])), float(np.median(nwit)),
                     float(np.mean(nwit > 0))))
        n_, dz, r0, r10, rm, w, f = rows[-1]
        print(f"{n_:10} {dz:8.4f} {r0:8.4f} {r10:8.4f} {rm:8.4f} "
              f"{w:10.0f} {f:11.3f}")

    print(f"\nrho = min over proper witnesses of (d(x,y)+d(y,z))/d(x,z).")
    print(f"#wit<=lam = median count of witnesses with ratio <= {args.lam}; "
          f"frac pairs = fraction with at least one such witness.")
    print("rho ~ 1 => near-geodesic chains exist at that scale (highways are "
          "certifiable). rho -> 2 => nothing lies between the pair.")


if __name__ == "__main__":
    main()

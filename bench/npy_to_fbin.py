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

"""npy_to_fbin.py -- fg .npy vectors -> ParlayANN/big-ann `.fbin`.

Format (big-ann-benchmarks "bin"): uint32 n, uint32 d, then n*d float32 rows.

    python3 npy_to_fbin.py glove_X.npy glove_base.fbin --normalize

`--normalize` L2-normalises each row. Use it for ANGULAR datasets: the fg
harness normalises internally under `--metric cos`, and on unit vectors the
Euclidean ranking is identical to the cosine ranking, so normalising here makes
the ParlayANN baselines solve *exactly* the same problem under EUCLIDEAN.
"""
import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--normalize", action="store_true")
    ap.add_argument("--max-n", type=int, default=0, help="0 = all rows")
    ap.add_argument("--chunk", type=int, default=100_000)
    args = ap.parse_args()

    X = np.load(args.src, mmap_mode="r")
    n, d = X.shape
    if args.max_n and args.max_n < n:
        n = args.max_n

    with open(args.dst, "wb") as f:
        np.array([n, d], dtype=np.uint32).tofile(f)
        for i in range(0, n, args.chunk):
            j = min(i + args.chunk, n)
            blk = np.ascontiguousarray(X[i:j], dtype=np.float32)
            if args.normalize:
                nrm = np.linalg.norm(blk, axis=1, keepdims=True)
                nrm[nrm == 0] = 1.0
                blk = blk / nrm
            blk.tofile(f)
    print(f"{args.src} {X.shape} -> {args.dst}  n={n} d={d} "
          f"normalize={args.normalize}")


if __name__ == "__main__":
    main()

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
"""
convert_bigann.py — turn a billion-scale ANN benchmark into the (X.npy, Q.npy, gold.npy)
triplet the GRAFT core (`fg`, via npy.hpp) reads. STREAMS the base (never loads the
whole thing into RAM), so a 100M/1B file is converted with a bounded memory footprint.

Supported input formats (auto-detected by extension):
  *.fbin / *.u8bin / *.i8bin   big-ann-benchmarks: [uint32 npts][uint32 dim][data...]
                               (float32 / uint8 / int8)
  *.fvecs / *.bvecs / *.ivecs  classic: per vector [int32 dim][dim x (f32/u8/i32)]
Ground truth:
  *.ibin / *.bin (big-ann)     [uint32 nq][uint32 K][int32 ids nq*K][float32 dists nq*K]
  *.ivecs (classic)            per row [int32 K][K x int32 ids]

Base + query are written as **float32** .npy (the core is f32-only today; for uint8 SIFT that
quadruples size). Ground truth ids -> int32 .npy.

Usage:
  python3 convert_bigann.py --base deep-base.fbin --query deep-query.fbin --gt deep-gt.ibin \
      --n 10000000 --out data/deep10m
  # -> data/deep10m.X.npy  data/deep10m.Q.npy  data/deep10m.gold.npy
"""
import argparse, os, sys
import numpy as np

_BIN_DT = {".fbin": np.float32, ".u8bin": np.uint8, ".i8bin": np.int8}
_VECS_DT = {".fvecs": np.float32, ".bvecs": np.uint8, ".ivecs": np.int32}
CHUNK_ROWS = 1_000_000   # streaming granularity


def _ext(path):
    for e in list(_BIN_DT) + list(_VECS_DT) + [".ibin", ".bin"]:
        if path.endswith(e):
            return e
    raise SystemExit(f"unknown extension: {path}")


def bin_meta(path):
    """big-ann *.?bin -> (npts, dim, dtype, data_offset=8)."""
    dt = _BIN_DT[_ext(path)]
    with open(path, "rb") as f:
        npts, dim = np.fromfile(f, dtype=np.uint32, count=2)
    return int(npts), int(dim), dt, 8


def vecs_meta(path):
    """classic *.?vecs -> (npts, dim, dtype, record_stride_bytes). dim from first record."""
    dt = _VECS_DT[_ext(path)]
    itemsize = np.dtype(dt).itemsize
    with open(path, "rb") as f:
        dim = int(np.fromfile(f, dtype=np.int32, count=1)[0])
    rec = 4 + dim * itemsize
    npts = os.path.getsize(path) // rec
    return npts, dim, dt, rec


def stream_base_to_npy(path, out_npy, n_limit=None):
    """Stream-convert a base file to a float32 .npy of shape (n, dim), bounded RAM."""
    e = _ext(path)
    is_bin = e in _BIN_DT
    npts, dim, dt, meta = bin_meta(path) if is_bin else vecs_meta(path)  # meta: 8 (bin) or rec-stride (vecs)
    n = npts if n_limit is None else min(n_limit, npts)
    print(f"  {os.path.basename(path)}: {npts} x {dim} ({np.dtype(dt).name}) -> writing {n} x {dim} float32")
    out = np.lib.format.open_memmap(out_npy, mode="w+", dtype=np.float32, shape=(n, dim))
    with open(path, "rb") as f:
        if is_bin:
            f.seek(8)                          # skip the [npts,dim] header
        done = 0
        while done < n:
            rows = min(CHUNK_ROWS, n - done)
            if is_bin:                         # contiguous dtype rows, no per-record header
                buf = np.fromfile(f, dtype=dt, count=rows * dim).reshape(rows, dim)
            else:                              # *vecs: each record = [int32 dim][dim x dtype]
                recs = np.fromfile(f, dtype=np.uint8, count=rows * meta).reshape(rows, meta)
                buf = np.ascontiguousarray(recs[:, 4:]).view(dt).reshape(rows, dim)
            out[done:done + rows] = buf.astype(np.float32, copy=False)
            done += rows
            print(f"    {done}/{n}", end="\r", flush=True)
    out.flush(); print()
    return n, dim


def load_gt(path):
    """Return int32 ground-truth ids [nq, K]."""
    e = _ext(path)
    if e in (".ibin", ".bin"):
        with open(path, "rb") as f:
            nq, K = np.fromfile(f, dtype=np.uint32, count=2)
            ids = np.fromfile(f, dtype=np.int32, count=int(nq) * int(K)).reshape(int(nq), int(K))
        return ids
    if e == ".ivecs":
        a = np.fromfile(path, dtype=np.int32)
        K = int(a[0]); rec = 1 + K
        return a.reshape(-1, rec)[:, 1:].astype(np.int32)
    raise SystemExit(f"unsupported GT format: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="base vectors (.fbin/.u8bin/.i8bin/.fvecs/.bvecs)")
    ap.add_argument("--query", required=True, help="query vectors (same formats)")
    ap.add_argument("--gt", help="ground-truth ids (.ibin/.bin/.ivecs); optional but recommended")
    ap.add_argument("--n", type=int, default=None, help="limit base to first N rows (make a slice)")
    ap.add_argument("--out", required=True, help="output prefix -> {out}.X.npy .Q.npy .gold.npy")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    print("base ->"); n, d = stream_base_to_npy(args.base, args.out + ".X.npy", args.n)
    print("query ->"); nq, dq = stream_base_to_npy(args.query, args.out + ".Q.npy", None)
    if dq != d:
        print(f"  WARNING: query dim {dq} != base dim {d}", file=sys.stderr)
    if args.gt:
        print("gt ->")
        gt = load_gt(args.gt)
        if args.n is not None:
            gt = np.minimum(gt, n - 1)     # NOTE: a slice invalidates GT ids >= n; see caveat below
            print(f"  WARNING: --n slices the base; ground-truth is only exact for the FULL base. "
                  f"For a slice, recompute GT against the slice (brute force on 10M is feasible) "
                  f"or use a dataset whose GT matches the slice.", file=sys.stderr)
        np.save(args.out + ".gold.npy", gt.astype(np.int32))
        print(f"  gold {gt.shape} int32")
    print(f"done -> {args.out}.X.npy ({n}x{d})  {args.out}.Q.npy ({nq}x{dq})"
          + (f"  {args.out}.gold.npy" if args.gt else "  (no gold)"))


if __name__ == "__main__":
    main()

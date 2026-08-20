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

"""convert_annb.py -- ann-benchmarks HDF5 -> npy triplet for the fg harness.

    python3 convert_annb.py glove-100-angular.hdf5 glove
      -> glove_X.npy      (train vectors, float32, [N, d])
         glove_Q.npy      (test/query vectors, float32, [nq, d])
         glove_gold.npy   (true neighbour ids, int32, [nq, k])

The fg harness normalises rows itself when --metric cos, so vectors are written
verbatim. Gold is the dataset's own `neighbors` matrix (brute-force top-k ids).
"""
import sys
import h5py
import numpy as np


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: convert_annb.py <dataset.hdf5> <out_prefix>")
    path, prefix = sys.argv[1], sys.argv[2]
    with h5py.File(path, "r") as f:
        X = np.asarray(f["train"], dtype=np.float32)
        Q = np.asarray(f["test"], dtype=np.float32)
        gold = np.asarray(f["neighbors"], dtype=np.int32)
    np.save(prefix + "_X.npy", X)
    np.save(prefix + "_Q.npy", Q)
    np.save(prefix + "_gold.npy", gold)
    print(f"{path}: X{X.shape} {X.dtype}  Q{Q.shape}  gold{gold.shape}")
    print(f"  -> {prefix}_X.npy  {prefix}_Q.npy  {prefix}_gold.npy")


if __name__ == "__main__":
    main()

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

"""ann-benchmarks adapter for GRAFT.

Drop into ann-benchmarks as ``ann_benchmarks/algorithms/graft/module.py``
with a config.yml declaring build args ``(T, harvest, patience)`` and query
arg ``ef``, e.g.::

    graft:
      docker-tag: ann-benchmarks-graft
      module: ann_benchmarks.algorithms.graft
      constructor: Graft
      run-groups:
        base:
          args: [[8, 16, 32], [200, 400], [0, 64]]   # T, harvest, patience
          query-args: [[32, 64, 128, 256, 512]]      # ef
"""

import numpy as np

import graft

from ..base.module import BaseANN


class Graft(BaseANN):
    def __init__(self, metric, T=16, harvest=400, patience=0):
        if metric not in ("euclidean", "angular"):
            raise NotImplementedError(f"unsupported metric: {metric}")
        self._metric = "l2" if metric == "euclidean" else "cosine"
        self._T = T
        self._harvest = harvest
        self._patience = patience
        self._ef = 64
        self._index = None
        self.name = f"graft(T={T},h={harvest},p={patience})"

    def fit(self, X):
        self._index = graft.build(
            np.ascontiguousarray(X, dtype=np.float32), metric=self._metric,
            T=self._T, harvest=self._harvest, patience=self._patience, seed=42)

    def set_query_arguments(self, ef):
        self._ef = int(ef)

    def query(self, v, n):
        ids, _ = self._index.search(
            np.ascontiguousarray(v, dtype=np.float32), k=n,
            ef=max(self._ef, n), threads=1)
        return ids[0]

    def batch_query(self, X, n):
        self._batch_ids, _ = self._index.search(
            np.ascontiguousarray(X, dtype=np.float32), k=n,
            ef=max(self._ef, n), threads=0)

    def get_batch_results(self):
        return self._batch_ids

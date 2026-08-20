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

Drop this directory (module.py, config.yml, Dockerfile) into ann-benchmarks
as ``ann_benchmarks/algorithms/graft/``. Build parameters (T, harvest,
patience) arrive as the ``method_param`` dict from config.yml's
``arg_groups``; the query argument is the beam width ``ef``.
"""

import numpy as np

import graft

from ..base.module import BaseANN


class Graft(BaseANN):
    def __init__(self, metric, method_param):
        if metric not in ("euclidean", "angular"):
            raise NotImplementedError(f"unsupported metric: {metric}")
        self._metric = "l2" if metric == "euclidean" else "cosine"
        self.method_param = method_param
        self._ef = 64
        self._index = None

    def fit(self, X):
        mp = self.method_param
        self._index = graft.build(
            np.ascontiguousarray(X, dtype=np.float32), metric=self._metric,
            T=mp.get("T", 16), harvest=mp.get("harvest", 400),
            patience=mp.get("patience", 0), seed=42)

    def set_query_arguments(self, ef):
        self._ef = int(ef)
        self.name = "graft(%s, 'ef': %s)" % (self.method_param, ef)

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

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

"""GRAFT — deterministic, embarrassingly parallel navigable-graph ANN index.

Native API::

    import graft, numpy as np
    X = np.random.rand(100_000, 96).astype(np.float32)
    idx = graft.build(X, metric="l2", T=8, harvest=200, seed=1)
    ids, dists = idx.search(X[:10], k=10, ef=100)
    idx.graph_hash        # determinism gate: same seed/params -> same hash
                          # at ANY thread count (on one machine)

`HnswlibStyleIndex` mirrors hnswlib's method names for easy porting of
existing harnesses — but GRAFT builds in batch: `add_items` only accumulates,
and the graph is built on the first query (or an explicit `.build()`).
"""

import numpy as np

from graft._core import Index, MappedIndex, build as _build, load_mmap

__all__ = ["build", "Index", "MappedIndex", "load_mmap", "HnswlibStyleIndex",
           "__version__"]
__version__ = "0.2.0"


def build(X, metric="l2", T=16, harvest=400, harvest_cap=64, alpha=1.0,
          patience=0, max_degree=0, seed=42, threads=0, build_conc=0,
          verbose=False):
    """Build a GRAFT index over ``X`` (``[n, d]``, cast to float32).

    Parameters mirror the paper's knob table: ``T`` (forest size; scale with
    dataset hardness), ``harvest`` (beam width ef_h, the quality lever),
    ``harvest_cap`` (degree cap R), ``alpha`` (occlusion slack; 1.0
    concentrated / 1.2 spread-L2), ``patience`` (per-point adaptive stop;
    recommended at large n), ``max_degree`` (hub guard; 128 advisable at
    d >~ 500). ``threads=0`` uses all cores; the graph is bitwise identical
    at any thread count for a fixed ``seed``.

    For ``metric="cosine"`` the index stores L2-normalized copies of the
    vectors and normalizes queries; returned distances are ``1 - dot``
    (and squared L2 for ``metric="l2"``), matching hnswlib conventions.
    """
    X = np.ascontiguousarray(X, dtype=np.float32)
    return _build(X, metric, T, harvest, harvest_cap, float(alpha), patience,
                  max_degree, seed, threads, build_conc, verbose)


class HnswlibStyleIndex:
    """hnswlib-shaped adapter over the batch GRAFT builder.

    Same method names as ``hnswlib.Index`` so benchmark harnesses port with a
    two-line diff — with honest batch semantics: ``add_items`` accumulates
    points, and the graph is built lazily on the first ``knn_query`` (or an
    explicit ``build()``). Incremental insertion after the build is not
    supported in this version.
    """

    def __init__(self, space, dim):
        if space not in ("l2", "cosine"):
            raise ValueError(f"space must be 'l2' or 'cosine', got {space!r} "
                             "(inner product is not supported)")
        self.space = space
        self.dim = int(dim)
        self._chunks = []
        self._index = None
        self._ef = 64
        self._params = {}

    def init_index(self, max_elements=0, **graft_params):
        """Accepts GRAFT build parameters (T, harvest, harvest_cap, alpha,
        patience, max_degree, seed, threads). ``max_elements`` is accepted for
        signature compatibility and ignored (the batch build sizes itself)."""
        self._params = dict(graft_params)
        return self

    def add_items(self, data, ids=None):
        data = np.ascontiguousarray(data, dtype=np.float32).reshape(-1, self.dim)
        if ids is not None:
            raise NotImplementedError(
                "explicit ids are not supported yet; labels are row order")
        if self._index is not None:
            raise NotImplementedError(
                "incremental add after build is not supported in this version")
        self._chunks.append(data)

    def build(self):
        if self._index is None:
            if not self._chunks:
                raise RuntimeError("no items added")
            X = self._chunks[0] if len(self._chunks) == 1 else \
                np.vstack(self._chunks)
            self._index = build(X, metric=self.space, **self._params)
        return self._index

    def set_ef(self, ef):
        self._ef = int(ef)

    def knn_query(self, data, k=1, num_threads=0):
        idx = self.build()
        data = np.ascontiguousarray(data, dtype=np.float32).reshape(-1, self.dim)
        ids, dists = idx.search(data, k=k, ef=max(self._ef, k),
                                threads=num_threads)
        return ids, dists

    def get_current_count(self):
        return sum(c.shape[0] for c in self._chunks)

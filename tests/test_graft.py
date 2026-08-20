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

"""Tests: determinism contract, recall vs brute force, API surface."""

import numpy as np
import pytest

import graft


def clustered(n, nq, d, n_clusters=32, seed=0):
    """DB and held-out queries drawn from the SAME cluster centers (the CLI
    harness's synthetic protocol; queries disjoint from the DB)."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_clusters, d)).astype(np.float32)
    assign = rng.integers(0, n_clusters, size=n + nq)
    P = centers[assign] + 0.3 * rng.standard_normal((n + nq, d)).astype(np.float32)
    P = P.astype(np.float32)
    return P[:n], P[n:]


@pytest.fixture(scope="module")
def data():
    return clustered(20_000, 100, 32)


def brute_knn(X, Q, k):
    d2 = ((Q[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    return np.argsort(d2, axis=1)[:, :k]


def recall(ids, gold):
    hits = sum(len(set(ids[i]) & set(gold[i])) for i in range(len(gold)))
    return hits / gold.size


def test_determinism_same_seed(data):
    X, _ = data
    a = graft.build(X, T=8, harvest=100, seed=1)
    b = graft.build(X, T=8, harvest=100, seed=1)
    assert a.graph_hash == b.graph_hash


def test_determinism_thread_count(data):
    X, _ = data
    a = graft.build(X, T=8, harvest=100, seed=1, threads=1)
    b = graft.build(X, T=8, harvest=100, seed=1, threads=0)  # all cores
    assert a.graph_hash == b.graph_hash


def test_seed_changes_graph(data):
    X, _ = data
    a = graft.build(X, T=8, harvest=100, seed=1)
    b = graft.build(X, T=8, harvest=100, seed=2)
    assert a.graph_hash != b.graph_hash


@pytest.mark.parametrize("metric", ["l2", "cosine"])
def test_recall(data, metric):
    X, Q = data
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True) if metric == "cosine" else X
    Qn = Q / np.linalg.norm(Q, axis=1, keepdims=True) if metric == "cosine" else Q
    idx = graft.build(X, metric=metric, T=8, harvest=200, seed=1)
    ids, dists = idx.search(Q, k=10, ef=128)
    gold = brute_knn(Xn, Qn, 10)
    r = recall(ids, gold)
    assert r >= 0.90, f"recall@10 = {r:.3f} ({metric})"
    assert ids.shape == (100, 10) and dists.shape == (100, 10)
    assert np.all(np.diff(dists, axis=1) >= -1e-6)   # sorted ascending


def test_single_query(data):
    X, Q = data
    idx = graft.build(X, T=4, harvest=100, seed=1)
    ids, dists = idx.search(Q[0], k=5, ef=64)
    assert ids.shape == (1, 5)


def test_patience_builds_and_searches(data):
    X, Q = data
    idx = graft.build(X, T=8, harvest=200, patience=32, seed=1)
    ids, _ = idx.search(Q, k=10, ef=128)
    gold = brute_knn(X, Q, 10)
    assert recall(ids, gold) >= 0.85


def test_attrs(data):
    X, _ = data
    idx = graft.build(X, T=4, harvest=100, seed=1)
    assert idx.n == 20_000 and idx.dim == 32 and idx.metric == "l2"
    assert idx.degree_avg > 1 and idx.n_dist_build > 0
    assert len(idx.graph_hash) == 16


def test_hnswlib_style_shim(data):
    X, Q = data
    h = graft.HnswlibStyleIndex(space="l2", dim=32)
    h.init_index(T=8, harvest=200, seed=1)
    h.add_items(X[:10_000])
    h.add_items(X[10_000:])
    h.set_ef(128)
    ids, dists = h.knn_query(Q, k=10)
    gold = brute_knn(X, Q, 10)
    assert recall(ids, gold) >= 0.90
    with pytest.raises(NotImplementedError):
        h.add_items(X[:5])          # no incremental add after build


def test_bad_inputs(data):
    X, _ = data
    with pytest.raises(ValueError):
        graft.build(X, metric="ip")
    idx = graft.build(X, T=4, harvest=100, seed=1)
    with pytest.raises(ValueError):
        idx.search(np.zeros(33, dtype=np.float32), k=5)   # wrong dim

"""Flat mmap-able format v1: save -> load_mmap parity with the built index."""
import os

import numpy as np
import pytest

import graft


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(7)
    return (rng.random((3000, 32), np.float32),
            rng.random((50, 32), np.float32))


@pytest.mark.parametrize("metric", ["l2", "cosine"])
def test_save_load_parity(tmp_path_factory, data, metric):
    X, Q = data
    idx = graft.build(X, metric=metric, T=8, harvest=200, seed=3)
    p = str(tmp_path_factory.mktemp("fmt") / f"{metric}.graft")
    idx.save(p)
    m = graft.load_mmap(p)
    assert (m.n, m.dim, m.metric) == (idx.n, idx.dim, metric)
    assert m.stored_hash == idx.graph_hash        # written at save
    assert m.graph_hash == idx.graph_hash         # recomputed from the map
    for ef in (10, 64, 300):
        i1, d1 = idx.search(Q, k=10, ef=ef, threads=1)
        i2, d2 = m.search(Q, k=10, ef=ef, threads=1)
        assert (i1 == i2).all()
        assert (d1 == d2).all()                   # bitwise
    assert m.nbytes == os.path.getsize(p)


def test_load_rejects_garbage(tmp_path):
    p = tmp_path / "junk.graft"
    p.write_bytes(b"\x00" * 256)
    with pytest.raises(RuntimeError, match="bad magic"):
        graft.load_mmap(str(p))
    p2 = tmp_path / "short.graft"
    p2.write_bytes(b"\x00" * 16)
    with pytest.raises(RuntimeError, match="truncated"):
        graft.load_mmap(str(p2))

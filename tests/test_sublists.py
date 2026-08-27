"""search_sublists: packed sub-list graphs + local->global map, vs exact."""
import numpy as np

import graft
from graft import _core


def test_sublists_recall_and_globals():
    rng = np.random.default_rng(1)
    Xg = rng.random((3000, 32), np.float32)
    lists = [rng.choice(3000, 400, replace=False).astype(np.int32)
             for _ in range(5)]
    PTRs, IDXs, ROOTs, po, io, ro = [], [], [], [0], [0], [0]
    for l in lists:
        g = graft.build(Xg[l], metric="l2", T=4, harvest=100, seed=1)
        ptr, idx, roots = g.graph()
        PTRs.append(ptr); IDXs.append(idx); ROOTs.append(roots)
        po.append(po[-1] + len(ptr))
        io.append(io[-1] + len(idx))
        ro.append(ro[-1] + len(roots))
    MAP = np.concatenate(lists)
    mo = np.cumsum([0] + [len(l) for l in lists])[:-1]
    Q = rng.random((20, 32), np.float32)
    pair_q = np.repeat(np.arange(20), 5).astype(np.int64)
    pair_l = np.tile(np.arange(5, dtype=np.int32), 20)
    ids, d, nd = _core.search_sublists(
        np.concatenate(PTRs), np.array(po[:-1], np.int64),
        np.concatenate(IDXs), np.array(io[:-1], np.int64),
        np.concatenate(ROOTs), np.array(ro[:-1], np.int64),
        np.array([len(r) for r in ROOTs], np.int32),
        MAP, np.array(mo, np.int64),
        np.array([len(l) for l in lists], np.int64),
        Xg, "l2", Q, pair_q, pair_l, k=10, ef=100, threads=2)
    assert nd > 0
    ok = tot = 0
    for p in range(len(pair_q)):
        l = lists[pair_l[p]]
        dd = ((Xg[l] - Q[pair_q[p]]) ** 2).sum(1)
        true10 = set(l[np.argsort(dd)[:10]].tolist())
        got = ids[p]
        assert np.isin(got[got >= 0], l).all()      # global ids, in-list
        ok += len(true10 & set(got.tolist()))
        tot += 10
    assert ok / tot >= 0.97                          # near-exact at ef=100

    # determinism: identical call -> identical output (best-of-roots entry)
    ids2, d2, nd2 = _core.search_sublists(
        np.concatenate(PTRs), np.array(po[:-1], np.int64),
        np.concatenate(IDXs), np.array(io[:-1], np.int64),
        np.concatenate(ROOTs), np.array(ro[:-1], np.int64),
        np.array([len(r) for r in ROOTs], np.int32),
        MAP, np.array(mo, np.int64),
        np.array([len(l) for l in lists], np.int64),
        Xg, "l2", Q, pair_q, pair_l, k=10, ef=100, threads=4)
    assert (ids == ids2).all() and (d == d2).all() and nd == nd2

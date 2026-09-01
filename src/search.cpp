// Copyright 2026 Edgar Chávez and contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/* search.cpp -- plain beam search over the forest graph.
 *
 * Faithful to forest_graph._beam_batch semantics:
 *   - visited set: epoch-stamped array (a vertex's distance is computed at
 *     most once per query);
 *   - candidate frontier: min-heap; result set: bounded max-heap of size ef;
 *   - termination: stop when the closest unexpanded candidate is strictly
 *     farther than the worst of a full result set;
 *   - insertion: a neighbour enters the frontier iff the result set is not
 *     full or it strictly improves on the worst result.
 *
 * Engineering vs the Numba kernel: u32 epoch stamps (4B/vertex/thread),
 * packed (distkey, id) u64 heap entries with id tiebreaks (bitwise-stable
 * results under any tie), software prefetch of the next neighbours' stamp
 * and vector rows, and an OpenMP thread pool over query slices on the
 * shared read-only graph.
 */
#include "fg.hpp"

struct Scratch {
    std::vector<u32> stamp;
    u32              epoch = 0;
    std::vector<u64> cand;     /* grows; min-heap */
    std::vector<u64> res;      /* capacity ef; max-heap */
    std::vector<u64> fin;      /* sorted results */
};

/* distance-only comparison of packed keys (the Python compares floats, so
 * ties in distance must NOT break on id here) */
static inline u32 dkey(u64 p) { return (u32)(p >> 32); }

static i64 beam_one(const f32 *q, const f32 *X, int d, int metric,
                    const i64 *ptr, const i32 *idx, const i32 *mapv,
                    const i32 *entries, int n_entry, bool besthub,
                    int ef, int k,
                    Scratch &S, i32 *out_ids, f32 *out_d)
{
    /* mapv: local->global row translation for sub-list graphs (vertex u's
     * vector is X + mapv[u]*d); nullptr = identity (whole-index graphs). */
#define FG_ROW(u) (X + (mapv ? (i64)mapv[u] : (i64)(u)) * d)
    if (++S.epoch == 0) { std::fill(S.stamp.begin(), S.stamp.end(), 0); S.epoch = 1; }
    const u32 epoch = S.epoch;
    u32 *stamp = S.stamp.data();
    u64 *res = S.res.data();
    int  nres = 0;
    S.cand.clear();
    i64 nd = 0;

    if (besthub) {
        /* hierarchy-mimic: evaluate every root, enter at the nearest one.
         * Costs n_entry evaluations inside the timed region. */
        u64 best = ~0ull;
        for (int e = 0; e < n_entry; e++) {
            f32 de = fg_dist(q, FG_ROW(entries[e]), d, metric);
            u64 p = pack_di(de, entries[e]);
            if (p < best) best = p;
        }
        nd += n_entry;
        i32 en = unpack_id(best);
        stamp[en] = epoch;
        S.cand.push_back(best);
        std::push_heap(S.cand.begin(), S.cand.end(), std::greater<u64>());
        hmax_offer(res, &nres, ef, best);
    } else for (int e = 0; e < n_entry; e++) {
        i32 en = entries[e];
        if (en < 0 || stamp[en] == epoch) continue;
        stamp[en] = epoch;
        f32 de = fg_dist(q, FG_ROW(en), d, metric);
        nd++;
        u64 p = pack_di(de, en);
        S.cand.push_back(p);
        std::push_heap(S.cand.begin(), S.cand.end(), std::greater<u64>());
        hmax_offer(res, &nres, ef, p);
    }

    while (!S.cand.empty()) {
        std::pop_heap(S.cand.begin(), S.cand.end(), std::greater<u64>());
        u64 top = S.cand.back(); S.cand.pop_back();
        if (nres >= ef && dkey(top) > dkey(res[0])) break;   /* can't improve */
        const i32 cn = unpack_id(top);
        const i64 s0 = ptr[cn], e0 = ptr[cn + 1];
        const i32 *nb = idx + s0;
        const i64 dg = e0 - s0;
        for (i64 j = 0; j < dg; j++) {
            if (j + 2 < dg) FG_PREFETCH(&stamp[nb[j + 2]]);
            if (j + 4 < dg) {                     /* 3 lines of the j+4 vector */
                const char *row = (const char *)FG_ROW(nb[j + 4]);
                FG_PREFETCH(row);
                FG_PREFETCH(row + 64);
                FG_PREFETCH(row + 128);
            }
            const i32 u = nb[j];
            if (stamp[u] == epoch) continue;
            stamp[u] = epoch;
            f32 du = fg_dist(q, FG_ROW(u), d, metric);
            nd++;
            if (nres < ef || f2key(du) < dkey(res[0])) {
                u64 p = pack_di(du, u);
                S.cand.push_back(p);
                std::push_heap(S.cand.begin(), S.cand.end(), std::greater<u64>());
                hmax_offer(res, &nres, ef, p);
            }
        }
    }

    S.fin.assign(res, res + nres);
    std::sort(S.fin.begin(), S.fin.end());
    int kk = k < nres ? k : nres;
    for (int a = 0; a < kk; a++) {
        out_ids[a] = unpack_id(S.fin[a]);
        out_d[a]   = unpack_dist(S.fin[a]);
    }
    for (int a = kk; a < k; a++) { out_ids[a] = -1; out_d[a] = INFINITY; }
    return nd;
#undef FG_ROW
}

SearchResult beam_search(const GraphView &gv, const f32 *X, int d,
                         const f32 *Q, i64 nq, const i32 *entries, int n_entry,
                         int ef, int k, int threads)
{
    SearchResult R;
    R.ids.assign((size_t)nq * (size_t)k, -1);
    R.dist.assign((size_t)nq * (size_t)k, INFINITY);
    if (threads <= 0) threads = omp_get_max_threads();
    const bool besthub = (entries == nullptr);
    const i32 *ebase = besthub ? gv.roots : entries;
    const int  estride = besthub ? 0 : n_entry;
    const int  ecount = besthub ? gv.n_roots : n_entry;
    const i64 *ptr = gv.ptr;
    const i32 *idx = gv.idx;
    const int  metric = gv.metric;
    const i64  n = gv.n;
    std::vector<i64> nd_acc((size_t)threads, 0);

    #pragma omp parallel num_threads(threads)
    {
        Scratch S;
        S.stamp.assign((size_t)n, 0);
        S.res.resize((size_t)ef);
        S.cand.reserve(4096);
        S.fin.reserve((size_t)ef);
        i64 nd_local = 0;
        #pragma omp barrier
        #pragma omp master
        { R.seconds = wall(); }
        #pragma omp barrier
        #pragma omp for schedule(dynamic, 8)
        for (i64 qi = 0; qi < nq; qi++)
            nd_local += beam_one(Q + qi * d, X, d, metric, ptr, idx, nullptr,
                                 ebase + qi * estride, ecount, besthub, ef, k, S,
                                 R.ids.data() + qi * k, R.dist.data() + qi * k);
        nd_acc[omp_get_thread_num()] += nd_local;
    }
    R.seconds = wall() - R.seconds;
    for (i64 v : nd_acc) R.n_dist += v;
    return R;
}

SearchResult beam_search(const ForestGraph &fg, const f32 *X, i64 n, int d,
                         const f32 *Q, i64 nq, const i32 *entries, int n_entry,
                         int ef, int k, int threads)
{
    (void)n;   /* the view carries it */
    return beam_search(graph_view(fg), X, d, Q, nq, entries, n_entry, ef, k,
                       threads);
}

/* entry-point construction, per forest_graph.make_entries:
 *   hub    -> uniform random roots
 *   random -> uniform random vertices
 *   far    -> farthest of a 512-vertex random sample (reach stress test) */
void make_entries(const GraphView &gv, const f32 *X, int d,
                  const f32 *Q, i64 nq, const char *mode, int n_entry, u64 seed,
                  std::vector<i32> &entries)
{
    const i64 n = gv.n;
    entries.assign((size_t)nq * (size_t)n_entry, -1);
    Rng r(mix_seed(seed, 0x456E747279ull));   /* "Entry" */
    const int T = gv.n_roots;
    if (!strcmp(mode, "hub")) {
        for (i64 i = 0; i < nq * n_entry; i++)
            entries[i] = gv.roots[(size_t)r.below((u64)T)];
    } else if (!strcmp(mode, "random")) {
        for (i64 i = 0; i < nq * n_entry; i++)
            entries[i] = (i32)r.below((u64)n);
    } else if (!strcmp(mode, "far")) {
        const int FS = 512;
        std::vector<u64> far((size_t)FS);
        for (i64 i = 0; i < nq; i++) {
            const f32 *q = Q + i * d;
            for (int s = 0; s < FS; s++) {
                i32 v = (i32)r.below((u64)n);
                f32 dd = fg_dist(q, X + (i64)v * d, d, gv.metric);
                far[s] = pack_di(dd, v);
            }
            std::sort(far.begin(), far.end());
            for (int e = 0; e < n_entry; e++)
                entries[i * n_entry + e] = unpack_id(far[(size_t)(FS - 1 - e)]);
        }
    } else {
        fprintf(stderr, "[FATAL] unknown entry mode '%s'\n", mode);
        exit(2);
    }
}

void make_entries(const ForestGraph &fg, const f32 *X, i64 n, int d,
                  const f32 *Q, i64 nq, const char *mode, int n_entry, u64 seed,
                  std::vector<i32> &entries)
{
    (void)n;   /* the view carries it */
    make_entries(graph_view(fg), X, d, Q, nq, mode, n_entry, seed, entries);
}

/* Batched beam search over PACKED sub-list graphs (SOLO serving path):
 * each list carries a local-topology graph (ptr/idx/roots in local vertex
 * ids) plus its membership array MAP (local -> global row in the single
 * global vector store Xg) -- no vector duplication. Entry is best-of-roots
 * (deterministic: every root evaluated, nearest wins -- the beam_one
 * besthub path). One OpenMP task per (query, list) pair; out_ids are
 * GLOBAL ids (-1 padded). Scratch stamps are sized max_nloc and epoch-
 * reused across pairs. Returns total distance evaluations. */
i64 search_sublists(const i64 *PTR, const i64 *ptr_off,
                    const i32 *IDX, const i64 *idx_off,
                    const i32 *ROOTS, const i64 *root_off, const i32 *nroots,
                    const i32 *MAP, const i64 *map_off, const i64 *nloc,
                    const f32 *Xg, int d, int metric,
                    const f32 *Q, const i64 *pair_q, const i32 *pair_l,
                    i64 npairs, i64 max_nloc, int ef, int k, int threads,
                    i32 *out_ids, f32 *out_d)
{
    if (threads <= 0) threads = omp_get_max_threads();
    std::vector<i64> nd_acc((size_t)threads, 0);
    #pragma omp parallel num_threads(threads)
    {
        Scratch S;
        S.stamp.assign((size_t)max_nloc, 0);
        S.res.resize((size_t)(ef > k ? ef : k));
        S.cand.reserve(4096);
        S.fin.reserve((size_t)(ef > k ? ef : k));
        i64 nd_local = 0;
        #pragma omp for schedule(dynamic, 16)
        for (i64 p = 0; p < npairs; p++) {
            const i32 l = pair_l[p];
            const i64 *ptr = PTR + ptr_off[l];
            const i32 *idx = IDX + idx_off[l];
            const i32 *rts = ROOTS + root_off[l];
            const i32 *mapv = MAP + map_off[l];
            i32 *oi = out_ids + p * k;
            f32 *od = out_d + p * k;
            nd_local += beam_one(Q + pair_q[p] * d, Xg, d, metric, ptr, idx,
                                 mapv, rts, nroots[l], /*besthub=*/true,
                                 ef > k ? ef : k, k, S, oi, od);
            for (int a = 0; a < k; a++)              /* local -> global */
                if (oi[a] >= 0) oi[a] = mapv[oi[a]];
            (void)nloc;
        }
        nd_acc[omp_get_thread_num()] += nd_local;
    }
    i64 nd = 0;
    for (i64 v : nd_acc) nd += v;
    return nd;
}

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

/* build.cpp -- phases 1-5 of the forest-graph pipeline.
 *
 * Phase 1  build_one_tree(): faithful port of sat_knn._process_node -- the HSP
 *          selection loop + routing loop, with the recursion flattened to an
 *          explicit stack and an in-place segment partition. The one
 *          performance change vs the Python: each node's chosen children have
 *          their vectors gathered into a contiguous scratch buffer, so the two
 *          hot loops stream over contiguous memory.
 * Phase 2+3 overlay_symmetrize(): emit (parent,child)+(child,parent) for every
 *          tree edge, sort, dedup, pack to CSR.
 * Phase 4  harvest_pass(): for every point, beam-search FOR it over the frozen
 *          union from the T roots; HSP-prune the EXPANDED set (the search path)
 *          into its adjacency; symmetrize + re-prune; union tree-0's spine back
 *          (connectivity: the forest supplies its own spanning tree). This IS
 *          the graph construction; everything before it is scaffold.
 *
 * No phase takes a lock; every parallel loop writes disjoint outputs; every
 * random choice is seeded by tree index or vertex id. Bitwise identical at
 * every thread count.
 */
#include <atomic>
#include "fg.hpp"

/* ===========================================================================
 * Phase 1: one spatial-approximation tree
 * =========================================================================== */
struct TreeScratch {
    std::vector<u64> pairs;       /* packed (distkey, member) for per-node sort */
    std::vector<i32> child_ids;
    std::vector<f32> child_vecs;  /* gathered child vectors, contiguous; sized by the
                                     ACTUAL children count (grow-on-demand), never by
                                     segment size -- children are the sparse HSP set,
                                     so this stays ~C_max*d, not N*d (the old pre-size
                                     to segment size made every in-flight tree hold
                                     N*d*4 bytes = the measured build-memory blowup) */
    size_t           cv_rows = 0; /* current child_vecs/child_ids capacity (rows) */
    std::vector<f32> best_d;
    std::vector<i32> best_who;
    std::vector<i32> mem_tmp;     /* counting-sort partition scratch */
    std::vector<f32> dn_tmp;
    std::vector<i32> counts;
};

struct Seg { i64 lo, hi; i32 node; i32 depth; };

/* Process one node: members mem[lo:hi) with distances dn[lo:hi) to `node`.
 * On return the segment has been partitioned into child sub-segments (children
 * themselves removed), pushed onto the stack, and parent[] updated for the
 * chosen children. Exactly the selection + routing loops of sat_knn.py. */
static void process_node(const f32 *X, int d, int metric,
                         i32 node, i64 lo, i64 hi,
                         i32 *mem, f32 *dn, i32 *parent,
                         int policy, u64 tree_seed, i32 depth, i32 max_depth,
                         TreeScratch &S, std::vector<Seg> &stack, i64 *ndist)
{
    const i64 m = hi - lo;
    if (m <= 0) return;
    if (max_depth > 0 && depth >= max_depth) {
        /* shallow stop (EC): every remaining member attaches to this node --
         * the bucket star. Deterministic; no distances computed. */
        for (i64 i = 0; i < m; i++) parent[mem[lo + i]] = node;
        return;
    }

    /* ---- order the members: proximal asc / distal desc / seeded shuffle ---- */
    u64 *P = S.pairs.data();
    for (i64 i = 0; i < m; i++)
        P[i] = pack_di(dn[lo + i], mem[lo + i]);
    std::sort(P, P + m);                                /* (dist, id) ascending */
    if (policy == POLICY_DISTAL) {
        std::reverse(P, P + m);
    } else if (policy == POLICY_RANDOM) {
        Rng r(mix_seed(tree_seed, (u64)(u32)node));     /* node-keyed: order-free */
        for (i64 i = m - 1; i > 0; i--) {
            i64 j = (i64)r.below((u64)(i + 1));
            u64 t = P[i]; P[i] = P[j]; P[j] = t;
        }
    }
    for (i64 i = 0; i < m; i++) {
        mem[lo + i] = unpack_id(P[i]);
        dn [lo + i] = unpack_dist(P[i]);
    }

    /* ---- selection loop: each member vs ALL current children ---- */
    i32 *children = S.child_ids.data();
    f32 *cv       = S.child_vecs.data();
    f32 *bd_arr   = S.best_d.data();
    i32 *bw_arr   = S.best_who.data();
    int  nch = 0;
    i64  dc  = 0;

    const i64 nlines = ((i64)d * 4 + 127) >> 7;          /* 128B lines per row */
    for (i64 i = 0; i < m; i++) {
        const i32 v  = mem[lo + i];
        const f32 *vv = X + (i64)v * d;
        if (i + 2 < m)                                   /* next members' rows */
            for (i64 l = 0; l < nlines; l++)
                FG_PREFETCH((const char *)(X + (i64)mem[lo + i + 2] * d) + l * 128);
        f32 bd = dn[lo + i];
        int bw = -1;
        int j = 0;
        f32 d4[4];
        for (; j + 4 <= nch; j += 4) {                  /* 4 contiguous rows/pass */
            fg_dist4(vv, cv + (i64)j * d,       cv + (i64)(j + 1) * d,
                         cv + (i64)(j + 2) * d, cv + (i64)(j + 3) * d,
                     d, metric, d4);
            if (d4[0] <= bd) { bd = d4[0]; bw = j;     }
            if (d4[1] <= bd) { bd = d4[1]; bw = j + 1; }
            if (d4[2] <= bd) { bd = d4[2]; bw = j + 2; }
            if (d4[3] <= bd) { bd = d4[3]; bw = j + 3; }
        }
        for (; j < nch; j++) {
            f32 dd = fg_dist(vv, cv + (i64)j * d, d, metric);
            if (dd <= bd) { bd = dd; bw = j; }          /* ties -> later child */
        }
        dc += nch;
        if (bw == -1) {                                  /* v becomes a child */
            if ((size_t)nch >= S.cv_rows) {              /* grow-on-demand (amortized) */
                size_t nr = S.cv_rows ? S.cv_rows * 2 : 1024;
                S.child_vecs.resize(nr * (size_t)d);
                S.child_ids.resize(nr);
                S.cv_rows = nr;
                cv       = S.child_vecs.data();          /* re-fetch: realloc moved them */
                children = S.child_ids.data();
            }
            children[nch] = v;
            memcpy(cv + (i64)nch * d, vv, (size_t)d * sizeof(f32));
            nch++;
            parent[v]  = node;
            bw_arr[i]  = -1;
        } else {
            bd_arr[i] = bd;
            bw_arr[i] = bw;
        }
    }

    /* ---- routing loop: stayers vs children added AFTER their current best ---- */
    for (i64 i = 0; i < m; i++) {
        if (bw_arr[i] == -1) continue;
        const i32 v  = mem[lo + i];
        const f32 *vv = X + (i64)v * d;
        if (i + 2 < m)
            for (i64 l = 0; l < nlines; l++)
                FG_PREFETCH((const char *)(X + (i64)mem[lo + i + 2] * d) + l * 128);
        f32 bd = bd_arr[i];
        int bw = bw_arr[i];
        int j = bw + 1;
        f32 d4[4];
        for (; j + 4 <= nch; j += 4) {
            fg_dist4(vv, cv + (i64)j * d,       cv + (i64)(j + 1) * d,
                         cv + (i64)(j + 2) * d, cv + (i64)(j + 3) * d,
                     d, metric, d4);
            if (d4[0] <= bd) { bd = d4[0]; bw = j;     }
            if (d4[1] <= bd) { bd = d4[1]; bw = j + 1; }
            if (d4[2] <= bd) { bd = d4[2]; bw = j + 2; }
            if (d4[3] <= bd) { bd = d4[3]; bw = j + 3; }
        }
        for (; j < nch; j++) {
            f32 dd = fg_dist(vv, cv + (i64)j * d, d, metric);
            if (dd <= bd) { bd = dd; bw = j; }
        }
        dc += nch - (bw_arr[i] + 1);
        bd_arr[i] = bd;
        bw_arr[i] = bw;                                  /* final child index */
    }
    *ndist += dc;

    /* ---- stable counting-sort partition into child sub-segments ---- */
    i32 *cnt = S.counts.data();
    for (int j = 0; j < nch; j++) cnt[j] = 0;
    for (i64 i = 0; i < m; i++)
        if (bw_arr[i] >= 0) cnt[bw_arr[i]]++;
    /* prefix offsets within [0, m-nch) of the scratch */
    i64 off = 0;
    std::vector<i64> start((size_t)nch + 1);
    for (int j = 0; j < nch; j++) { start[j] = off; off += cnt[j]; }
    start[nch] = off;
    i32 *mt = S.mem_tmp.data();
    f32 *dt = S.dn_tmp.data();
    {
        std::vector<i64> w(start.begin(), start.end() - 1);
        for (i64 i = 0; i < m; i++) {
            int j = bw_arr[i];
            if (j < 0) continue;
            i64 p = w[j]++;
            mt[p] = mem[lo + i];
            dt[p] = bd_arr[i];
        }
    }
    memcpy(mem + lo, mt, (size_t)off * sizeof(i32));
    memcpy(dn  + lo, dt, (size_t)off * sizeof(f32));

    /* ---- push child segments (order on the stack is irrelevant: outputs are
     *      parent[] writes keyed by vertex, and per-node rng is node-keyed) ---- */
    for (int j = 0; j < nch; j++)
        if (start[j + 1] > start[j])
            stack.push_back(Seg{ lo + start[j], lo + start[j + 1], children[j],
                                 depth + 1 });
}

static void tree_scratch_init(TreeScratch &S, i64 n, int d)
{
    S.pairs.resize((size_t)n);
    S.best_d.resize((size_t)n);
    S.best_who.resize((size_t)n);
    S.mem_tmp.resize((size_t)n);
    S.dn_tmp.resize((size_t)n);
    S.counts.resize((size_t)n);
    S.cv_rows = 1024;
    S.child_ids.resize(S.cv_rows);
    S.child_vecs.resize(S.cv_rows * (size_t)d);
}

/* Parallel subtree processing for ONE tree (used by the dense spine, where a
 * single tree must span n points and T = 1 gives no cross-tree parallelism).
 * A node's processing is a pure function of its segment and a node-keyed RNG,
 * and child segments are disjoint ranges of mem/dn, so subtrees are
 * independent tasks: per-thread scratch, disjoint parent[] writes, per-thread
 * distance counters. The tree is bitwise identical to the sequential build. */
struct TreeTaskCtx {
    const f32 *X; int d, metric; i32 *mem; f32 *dn; i32 *parent;
    int policy; u64 tree_seed; int max_depth;
    std::vector<TreeScratch> *scratch; std::vector<i64> *nd;
};
static void tree_run_seg(const TreeTaskCtx &c, Seg s)
{
    const int tid = omp_get_thread_num();
    TreeScratch &S = (*c.scratch)[(size_t)tid];
    i64 *nd = &(*c.nd)[(size_t)tid * 16];                /* padded counters */
    std::vector<Seg> local;
    process_node(c.X, c.d, c.metric, s.node, s.lo, s.hi, c.mem, c.dn, c.parent,
                 c.policy, c.tree_seed, s.depth, c.max_depth, S, local, nd);
    const i64 CUTOFF = 8192;
    for (size_t i = 0; i < local.size(); i++) {
        const Seg ch = local[i];
        if (ch.hi - ch.lo >= CUTOFF) {
            #pragma omp task firstprivate(ch) shared(c)
            tree_run_seg(c, ch);
        } else {                                          /* small subtree: this thread, sequentially */
            std::vector<Seg> st; st.push_back(ch);
            while (!st.empty()) {
                Seg x = st.back(); st.pop_back();
                process_node(c.X, c.d, c.metric, x.node, x.lo, x.hi, c.mem, c.dn, c.parent,
                             c.policy, c.tree_seed, x.depth, c.max_depth, S, st, nd);
            }
        }
    }
}

static void build_one_tree(const f32 *X, i64 n, int d, int metric,
                           u64 tree_seed, int policy, int max_depth,
                           i32 *parent, i32 *root_out, i64 *ndist, int threads = 1)
{
    Rng r(tree_seed);
    const i32 root = (i32)r.below((u64)n);
    *root_out = root;
    parent[root] = -1;

    std::vector<i32> mem((size_t)(n - 1));
    std::vector<f32> dn ((size_t)(n - 1));
    const f32 *rv = X + (i64)root * d;
    i64 m = 0;
    for (i64 v = 0; v < n; v++) {
        if (v == root) continue;
        mem[m] = (i32)v;
        dn [m] = fg_dist(rv, X + v * d, d, metric);
        m++;
    }
    *ndist += m;

    if (threads > 1) {
        std::vector<TreeScratch> scratch((size_t)threads);
        for (auto &S : scratch) tree_scratch_init(S, n, d);
        std::vector<i64> nd((size_t)threads * 16, 0);
        TreeTaskCtx c{X, d, metric, mem.data(), dn.data(), parent, policy, tree_seed, max_depth, &scratch, &nd};
        #pragma omp parallel num_threads(threads)
        {
            #pragma omp single
            tree_run_seg(c, Seg{ 0, m, root, 0 });
        }
        for (int t = 0; t < threads; t++) *ndist += nd[(size_t)t * 16];
        return;
    }
    TreeScratch S;
    /* child buffers (ids + gathered vectors) grow on demand inside process_node,
     * sized by the ACTUAL children selected -- never pre-sized to the segment.
     * (The old pre-size to segment size meant the root segment forced N*d floats
     * per in-flight tree: the dominant, measured build-memory term.) */
    tree_scratch_init(S, n, d);

    std::vector<Seg> stack;
    stack.reserve(1024);
    stack.push_back(Seg{ 0, m, root, 0 });
    while (!stack.empty()) {
        Seg s = stack.back(); stack.pop_back();
        process_node(X, d, metric, s.node, s.lo, s.hi,
                     mem.data(), dn.data(), parent, policy, tree_seed,
                     s.depth, max_depth, S, stack, ndist);
    }
}

/* ===========================================================================
 * Phases 2+3: overlay + backward edges -> CSR (sort/dedup over packed edges)
 * =========================================================================== */

/* Deterministic parallel sort+dedup of packed (src<<32|dst) edges. Partition
 * into FIXED element-blocks x src-keyed buckets (both independent of the
 * thread count), scatter each block to precomputed disjoint offsets (stable),
 * then sort+unique each bucket independently and compact. Bucketing is a
 * monotone function of the key, so the concatenation is ascending u64 order:
 * the same edge set std::sort+unique produces, at any thread count. */
static void sort_dedup_edges(std::vector<u64> &E, i64 n, int threads) {
    const i64 M = (i64)E.size();
    threads = threads > 0 ? threads : omp_get_max_threads();
    if (M < ((i64)1 << 18) || threads <= 1) {
        std::sort(E.begin(), E.end());
        E.erase(std::unique(E.begin(), E.end()), E.end());
        return;
    }
    int sb = 0; while (((i64)1 << sb) < n) sb++;         /* bits of src */
    const int shift = sb > 11 ? sb - 11 : 0;             /* <= 2048 buckets */
    const i64 NB    = ((n - 1) >> shift) + 1;
    const i64 NBLK  = 64;                                /* fixed: determinism */
    std::vector<i64> hist((size_t)(NBLK * NB), 0);
    #pragma omp parallel for schedule(static, 1) num_threads(threads)
    for (i64 blk = 0; blk < NBLK; blk++) {
        const i64 lo = blk * M / NBLK, hi = (blk + 1) * M / NBLK;
        i64 *h = hist.data() + blk * NB;
        for (i64 i = lo; i < hi; i++) h[(E[i] >> 32) >> shift]++;
    }
    std::vector<i64> bstart((size_t)NB + 1, 0);
    i64 cum = 0;
    for (i64 b = 0; b < NB; b++) {                       /* bucket-major offsets */
        bstart[b] = cum;
        for (i64 blk = 0; blk < NBLK; blk++) {
            i64 c = hist[blk * NB + b];
            hist[blk * NB + b] = cum;
            cum += c;
        }
    }
    bstart[NB] = cum;
    std::vector<u64> S((size_t)M);
    #pragma omp parallel for schedule(static, 1) num_threads(threads)
    for (i64 blk = 0; blk < NBLK; blk++) {
        const i64 lo = blk * M / NBLK, hi = (blk + 1) * M / NBLK;
        i64 *h = hist.data() + blk * NB;
        for (i64 i = lo; i < hi; i++) S[(size_t)h[(E[i] >> 32) >> shift]++] = E[i];
    }
    std::vector<i64> wlen((size_t)NB + 1, 0);
    #pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
    for (i64 b = 0; b < NB; b++) {
        u64 *s = S.data() + bstart[b];
        const i64 m = bstart[b + 1] - bstart[b];
        std::sort(s, s + m);
        wlen[b + 1] = std::unique(s, s + m) - s;
    }
    for (i64 b = 0; b < NB; b++) wlen[b + 1] += wlen[b];
    #pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
    for (i64 b = 0; b < NB; b++)
        memcpy(E.data() + wlen[b], S.data() + bstart[b],
               (size_t)(wlen[b + 1] - wlen[b]) * sizeof(u64));
    E.resize((size_t)wlen[NB]);
}

static CSR csr_from_edges(std::vector<u64> &E, i64 n, int threads) {
    threads = threads > 0 ? threads : omp_get_max_threads();
    sort_dedup_edges(E, n, threads);
    CSR g;
    g.ptr.assign((size_t)n + 1, 0);
    for (u64 e : E) g.ptr[(e >> 32) + 1]++;
    for (i64 v = 0; v < n; v++) g.ptr[v + 1] += g.ptr[v];
    g.idx.resize(E.size());
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (i64 i = 0; i < (i64)E.size(); i++) g.idx[i] = (i32)(u32)E[i];
    return g;
}

static CSR overlay_symmetrize(const std::vector<std::vector<i32>> &parents,
                              const std::vector<i32> &roots, i64 n, int threads,
                              int directed)
{
    const int T = (int)parents.size();
    const size_t per = (directed ? 1 : 2) * (size_t)(n - 1);
    std::vector<u64> E((size_t)T * per);
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (int t = 0; t < T; t++) {
        u64 *out = E.data() + (size_t)t * per;
        const i32 *par = parents[t].data();
        const i32 root = roots[t];
        size_t w = 0;
        for (i64 v = 0; v < n; v++) {
            if ((i32)v == root) continue;
            u64 a = (u64)(u32)par[v], b = (u64)(u32)v;
            out[w++] = (a << 32) | b;          /* parent -> child */
            if (!directed)
                out[w++] = (b << 32) | a;      /* backward edge */
        }
    }
    return csr_from_edges(E, n, threads);
}

/* ===========================================================================
 * Phase 4b: HSP prune -- the selection rule applied to the WHOLE candidate
 * union, backlinks included.
 *
 * The overlay symmetrises every tree edge, so a node's list is roughly half
 * parent->child edges (chosen by the HSP rule from THIS node's point of view)
 * and half child->parent reverses (chosen from someone else's). Measured on
 * GloVe, only ~50% of a capped node's edges survive a replay of the rule at
 * alpha=1.0, against 88% for Vamana and 99% for PiPNN: half the degree budget
 * buys no directional cover, which costs a distance on every landing.
 *
 * This replaces the random cap with the rule itself: process each vertex's
 * candidates nearest-first and keep u unless an already-kept g shadows it
 * (alpha * d(u,g) < d(u,v)), stopping at `cap` survivors. EVERY survivor
 * occludes (every kept edge shadows later candidates).
 *
 * alpha is the occlusion slack. On concentrated (holographic) data all pairwise
 * distances sit in a narrow band, so alpha > 1 shadows nothing and the rule
 * degenerates to "keep the nearest cap"; alpha = 1.0 (pure RNG/HSP) is the
 * meaningful setting there, and is what Vamana's own tuned GloVe recipe uses.
 *
 * cand bounds the candidates examined per vertex (nearest-first) so that hubs,
 * whose raw overlay degree can reach ~1770, cannot dominate the O(m^2) inner
 * loop. Deterministic: candidates are ordered by the packed (distance, id) key
 * and every parallel iteration writes a disjoint row -- no RNG, no locks.
 * =========================================================================== */
static CSR hsp_prune(const CSR &g, const f32 *X, i64 n, int d, int metric,
                     int cap, f32 alpha, int cand, int order, u64 seed,
                     int threads, i64 *ndist)
{
    threads = threads > 0 ? threads : omp_get_max_threads();
    std::vector<std::vector<i32>> rows((size_t)n);
    std::vector<i64> nd_acc((size_t)threads, 0);
    #pragma omp parallel num_threads(threads)
    {
        i64 nd_local = 0;
        std::vector<u64> ord;
        std::vector<i32> kept;
        #pragma omp for schedule(dynamic, 2048)
        for (i64 v = 0; v < n; v++) {
            const i32 *src = g.idx.data() + g.ptr[v];
            const i64 dg = g.deg(v);
            const f32 *xv = X + v * d;
            ord.clear();
            ord.reserve((size_t)dg);
            for (i64 j = 0; j < dg; j++) {
                i32 u = src[j];
                f32 du = fg_dist(xv, X + (i64)u * d, d, metric);
                nd_local++;
                ord.push_back(pack_di(du, u));
            }
            std::sort(ord.begin(), ord.end());      /* ascending (dist, id) */
            const i64 m = (cand > 0 && (i64)ord.size() > cand) ? cand
                                                              : (i64)ord.size();
            /* `cand` always truncates to the m NEAREST candidates -- it is a cost
             * bound on the O(m^2) rule, not a policy. The order below then decides
             * how those m are fed to the rule, which is what selects the edges. */
            if (order == 1) {                       /* distal: farthest-first */
                std::reverse(ord.begin(), ord.begin() + m);
            } else if (order == 2) {                /* random: vertex-keyed */
                Rng r(mix_seed(seed, (u64)(u32)v));
                for (i64 i = 0; i + 1 < m; i++) {
                    i64 j = i + (i64)r.below((u64)(m - i));
                    std::swap(ord[i], ord[j]);
                }
            }
            kept.clear();
            for (i64 j = 0; j < m && (int)kept.size() < cap; j++) {
                const i32 u = unpack_id(ord[j]);
                f32 duv = unpack_dist(ord[j]);
                if (metric == METRIC_L2) duv = std::sqrt(duv);
                const f32 *xu = X + (i64)u * d;
                bool occluded = false;
                for (i32 gk : kept) {
                    f32 dug = fg_dist(xu, X + (i64)gk * d, d, metric);
                    nd_local++;
                    if (metric == METRIC_L2) dug = std::sqrt(dug);
                    if (alpha * dug < duv) { occluded = true; break; }
                }
                if (!occluded) kept.push_back(u);
            }
            std::sort(kept.begin(), kept.end());    /* canonical order */
            rows[v].assign(kept.begin(), kept.end());
        }
        nd_acc[omp_get_thread_num()] += nd_local;
    }
    for (i64 v : nd_acc) *ndist += v;
    CSR out;
    out.ptr.assign((size_t)n + 1, 0);
    for (i64 v = 0; v < n; v++) out.ptr[v + 1] = out.ptr[v] + (i64)rows[v].size();
    out.idx.resize((size_t)out.ptr[n]);
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (i64 v = 0; v < n; v++)
        memcpy(out.idx.data() + out.ptr[v], rows[v].data(),
               rows[v].size() * sizeof(i32));
    return out;
}

/* ===========================================================================
 * Phase 6: batched search-harvest (NSG-style, frozen substrate).
 *
 * The one construction difference left between us and the search-built graphs
 * (Vamana/NSG) is WHERE candidate edges come from: theirs are harvested by beam
 * searches launched from far away, so their edges encode how to funnel a distant
 * start into the right basin -- measured as greedy hit rate 38.6% vs our 33.5%,
 * and failures landing in the true top-100 34% vs 26.5%. Everything our
 * construction produces (trees, votes, NN-descent) is locally defined and never
 * asks that question.
 *
 * This pass asks it, batched: for every point p, run a beam search FOR p over
 * the frozen substrate graph, entering at the T roots (far starts), and collect
 * every node whose distance was evaluated. HSP-prune the nearest `cand_bound`
 * of that set (alpha, nearest-first, every survivor occludes) into p's new
 * adjacency, capped at `cap`. NSG/Vamana interleave these searches with graph
 * mutation -- the serial, contended, order-dependent part. On a frozen graph
 * the same pass is embarrassingly parallel and deterministic: seeded entries,
 * read-only searches, ties broken by the packed (dist, id) key, disjoint
 * output rows.
 *
 * The output is directed (out-edges of p = what p's search saw), so the caller
 * symmetrizes + re-prunes afterwards -- here that step is NOT a no-op (unlike
 * E4's, where the substrate was already symmetric): a reverse edge w->p carries
 * "p's search reached w" back to w, and w's own rule then adjudicates it.
 * =========================================================================== */
static CSR harvest_pass(const CSR &sub, const f32 *X, i64 n, int d, int metric,
                        const i32 *entries, int n_entry, int ef, int cap,
                        f32 alpha, int cand_bound, f32 straighten, int patience,
                        int threads, i64 *ndist, std::vector<u64> *shortcuts,
                        int nblocks = 1, int block = 0,
                        std::vector<std::vector<i32>> *rows_io = nullptr,
                        const u8 *logmask = nullptr,
                        std::vector<std::vector<i32>> *vlog = nullptr)
{
    threads = threads > 0 ? threads : omp_get_max_threads();
    /* Block-synchronous mutation (nblocks > 1): harvest only the points of
     * this block, writing into the caller's persistent rows; the caller folds
     * the finished rows back into the substrate between blocks. Point p is in
     * block (p % nblocks), a fixed assignment, and within a block the
     * substrate is frozen and the writes are disjoint -- so the result is a
     * function of (data, seed, params, nblocks) alone, thread count included. */
    std::vector<std::vector<i32>> rows_local((size_t)(rows_io ? 0 : n));
    std::vector<std::vector<i32>> &rows = rows_io ? *rows_io : rows_local;
    std::vector<i64> nd_acc((size_t)threads, 0);
    const int nthr = threads;
    std::vector<std::vector<u64>> sc_buf((size_t)(straighten > 0 ? nthr : 0));
    #pragma omp parallel num_threads(threads)
    {
        i64 nd = 0;
        std::vector<u32> stamp((size_t)n, 0);
        std::vector<i32> pred((size_t)n, -1);  /* predecessor chain (epoch-gated) */
        u32 epoch = 0;
        std::vector<u64> cand;                 /* min-heap of packed (dist,id) */
        std::vector<u64> res((size_t)ef);      /* max-heap, worst at res[0] */
        std::vector<u64> topc((size_t)cap);    /* best-cap of all evaluated:
                                                  the adaptive-stop tracker */
        std::vector<u64> seen;                 /* every evaluated (dist,id) */
        std::vector<i32> kept;
        std::vector<i32> chain;                /* entry -> best predecessor path */
        std::vector<u64> *scb = straighten > 0
                              ? &sc_buf[(size_t)omp_get_thread_num()] : nullptr;
        #pragma omp for schedule(dynamic, 512)
        for (i64 p = 0; p < n; p++) {
            if (nblocks > 1 && (int)(p % (i64)nblocks) != block) continue;
            if (++epoch == 0) { std::fill(stamp.begin(), stamp.end(), 0); epoch = 1; }
            const f32 *q = X + p * d;
            cand.clear(); seen.clear();
            std::vector<i32> *L = (logmask && logmask[p]) ? &(*vlog)[p] : nullptr;
            if (L) L->clear();
            int nres = 0, ntc = 0, stall = 0;
            stamp[p] = epoch;                  /* self is not a candidate */
            for (int e = 0; e < n_entry; e++) {
                i32 en = entries[e];
                if (en < 0 || (i64)en == p || stamp[en] == epoch) continue;
                stamp[en] = epoch;
                pred[en] = -1;                 /* entry: chain root */
                f32 de = fg_dist(q, X + (i64)en * d, d, metric); nd++;
                if (L) L->push_back(en);
                u64 pk = pack_di(de, en);
                cand.push_back(pk);
                std::push_heap(cand.begin(), cand.end(), std::greater<u64>());
                hmax_offer(res.data(), &nres, ef, pk);
                hmax_offer(topc.data(), &ntc, cap, pk);
            }
            while (!cand.empty()) {
                std::pop_heap(cand.begin(), cand.end(), std::greater<u64>());
                u64 top = cand.back(); cand.pop_back();
                if (nres >= ef && (u32)(top >> 32) > (u32)(res[0] >> 32))
                    break;                     /* can't improve the result set */
                seen.push_back(top);           /* EXPANDED node: on the search
                                                  path -- the funnel candidates.
                                                  (The merely-evaluated cloud is
                                                  local fluff; pruning it keeps
                                                  no route edges.) */
                bool improved = false;
                const i32 cn = unpack_id(top);
                if (L) L->push_back((i32)((u32)cn | 0x80000000u));
                const i64 s0 = sub.ptr[cn], e0 = sub.ptr[cn + 1];
                const i32 *nb = sub.idx.data() + s0;
                const i64 dg = e0 - s0;
                for (i64 jx = 0; jx < dg; jx++) {
                    const i64 j = s0 + jx;              /* keep old index name */
                    if (jx + 2 < dg) FG_PREFETCH(&stamp[nb[jx + 2]]);
                    if (jx + 4 < dg) {                  /* rows, 4 ahead */
                        const char *row = (const char *)(X + (i64)nb[jx + 4] * d);
                        FG_PREFETCH(row);
                        FG_PREFETCH(row + 64);
                        FG_PREFETCH(row + 128);
                    }
                    (void)j;
                    const i32 u = nb[jx];
                    if (stamp[u] == epoch) continue;
                    stamp[u] = epoch;
                    pred[u] = cn;              /* u was reached by expanding cn */
                    f32 du = fg_dist(q, X + (i64)u * d, d, metric); nd++;
                    if (L) L->push_back(u);
                    u64 pk = pack_di(du, u);
                    if (ntc < cap || pk < topc[0]) improved = true;
                    hmax_offer(topc.data(), &ntc, cap, pk);
                    if (nres < ef || (u32)(pk >> 32) < (u32)(res[0] >> 32)) {
                        cand.push_back(pk);
                        std::push_heap(cand.begin(), cand.end(), std::greater<u64>());
                        hmax_offer(res.data(), &nres, ef, pk);
                    }
                }
                /* adaptive stop: `patience` consecutive expansions without an
                 * entry into the best-cap set means the search has converged
                 * past what the prune can use; the route is already in `seen`.
                 * Deterministic: a pure function of distances and ids. */
                if (patience > 0) {
                    if (improved) stall = 0;
                    else if (++stall >= patience) break;
                }
            }
            /* ---- string-pull the discovered route into shortcut edges ----
             * The beam finds p by a route that circles local minima; the chain
             * of predecessors from the best node back to the entry IS that
             * route. Replace it with the shortest subsequence whose every hop
             * makes GEOMETRIC progress toward p (d(c_j,p) <= d(c_i,p)/straighten,
             * i.e. Vamana's alpha-reachability reached from the other side).
             * The emitted edges are the long-range navigational links that
             * occlusion pruning discards -- certified by usage, which is the
             * only signal that identifies them on holographic data (geometry
             * cannot: no lambda-lines, distal ordering catastrophic).
             * Pure monotonicity would collapse each route to one entry->target
             * jump and make the entries mega-hubs; the geometric rule keeps
             * O(log) hops with bounded contraction. */
            if (scb && nres > 0) {
                /* the route we straighten is the one to p's BEST found
                 * neighbour -- res[] is a max-heap with the WORST at res[0],
                 * so scan for the minimum (first fix: v1 backtracked from
                 * res[0] and straightened the route to the ef-th-nearest
                 * node instead). */
                u64 best = res[0];
                for (int a = 1; a < nres; a++) if (res[a] < best) best = res[a];
                chain.clear();
                for (i32 c = unpack_id(best); c >= 0; c = pred[c]) {
                    chain.push_back(c);
                    if ((i64)chain.size() > 4096) break;   /* pathological guard */
                }
                std::reverse(chain.begin(), chain.end());  /* entry -> best */
                const i64 L = (i64)chain.size();
                if (L >= 2) {
                    /* string-pull by FIRST improvement: from c_i, advance to
                     * the first chain node strictly closer to p (straighten=1.0,
                     * the default reading of "remove the detours": the maximal
                     * monotone subsequence) or closer by the factor
                     * (straighten>1: coarser, fewer hops). EVERY hop of the
                     * straightened path is emitted -- the path is the object
                     * being added, not just its long segments. */
                    i64 i = 0;
                    f32 di = fg_dist(q, X + (i64)chain[0] * d, d, metric); nd++;
                    while (i + 1 < L) {
                        const f32 need = di / straighten;
                        i64 best_j = -1;
                        f32 best_dj = 0;
                        for (i64 jj = i + 1; jj < L; jj++) {
                            f32 dj = fg_dist(q, X + (i64)chain[jj] * d, d, metric);
                            nd++;
                            if (dj < need || (straighten <= 1.0f && dj < di)) {
                                best_j = jj; best_dj = dj; break;
                            }
                        }
                        if (best_j < 0) break;             /* no progress ahead */
                        u64 a = (u64)(u32)chain[i], b = (u64)(u32)chain[best_j];
                        scb->push_back((a << 32) | b);     /* every hop counts */
                        i = best_j; di = best_dj;
                    }
                    /* terminal hop: last path node -> p itself, so the route
                     * actually reaches the point it was searching for */
                    if (i > 0 && (i64)unpack_id(best) != p) {
                        u64 a = (u64)(u32)chain[i], b = (u64)(u32)p;
                        scb->push_back((a << 32) | b);
                    }
                }
            }
            for (int a = 0; a < nres; a++) seen.push_back(res[a]);
            std::sort(seen.begin(), seen.end());
            seen.erase(std::unique(seen.begin(), seen.end()), seen.end());
            /* safety bound only; the expanded+result set is naturally small */
            i64 m = (i64)seen.size();
            if (cand_bound > 0 && m > cand_bound) m = cand_bound;
            kept.clear();
            for (i64 j = 0; j < m && (int)kept.size() < cap; j++) {
                const i32 u = unpack_id(seen[j]);
                f32 duv = unpack_dist(seen[j]);
                if (metric == METRIC_L2) duv = std::sqrt(duv);
                const f32 *xu = X + (i64)u * d;
                bool occluded = false;
                for (i32 gk : kept) {
                    f32 dug = fg_dist(xu, X + (i64)gk * d, d, metric); nd++;
                    if (metric == METRIC_L2) dug = std::sqrt(dug);
                    if (alpha * dug < duv) { occluded = true; break; }
                }
                if (!occluded) kept.push_back(u);
            }
            std::sort(kept.begin(), kept.end());
            rows[p].assign(kept.begin(), kept.end());
        }
        nd_acc[omp_get_thread_num()] += nd;
    }
    for (i64 v : nd_acc) *ndist += v;
    if (shortcuts)                         /* deterministic: caller sorts */
        for (auto &b : sc_buf)
            shortcuts->insert(shortcuts->end(), b.begin(), b.end());
    CSR out;
    out.ptr.assign((size_t)n + 1, 0);
    for (i64 v = 0; v < n; v++) out.ptr[v + 1] = out.ptr[v] + (i64)rows[v].size();
    out.idx.resize((size_t)out.ptr[n]);
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (i64 v = 0; v < n; v++)
        memcpy(out.idx.data() + out.ptr[v], rows[v].data(),
               rows[v].size() * sizeof(i32));
    return out;
}

/* Symmetrise a directed CSR: every surviving edge gets its reverse, so a node
 * that someone chose can be entered from that side. This is the bulk analogue
 * of the insertion-time backlink -- but only survivors earn one, and the caller
 * re-runs hsp_prune() over the result so an incoming edge must itself survive
 * the receiving node's rule (Vamana's RobustPrune-on-overflow, applied to the
 * complete candidate set at once instead of under arrival order). */
static CSR csr_symmetrize(const CSR &g, i64 n, int threads) {
    std::vector<u64> E;
    E.reserve((size_t)g.idx.size() * 2);
    for (i64 v = 0; v < n; v++)
        for (i64 j = g.ptr[v]; j < g.ptr[v + 1]; j++) {
            u64 a = (u64)(u32)v, b = (u64)(u32)g.idx[j];
            E.push_back((a << 32) | b);
            E.push_back((b << 32) | a);
        }
    return csr_from_edges(E, n, threads);
}

/* ===========================================================================
 * Phase 5: NN-descent refinement (bulk-synchronous per-vertex maps)
 * B is a dense n x k id matrix, -1 padded, each row the current k-nearest
 * estimate sorted by (distance, id).
 * =========================================================================== */
static void knearest_into(const f32 *X, int d, int metric, i64 x,
                          const i32 *cand, i64 nc, int k,
                          u64 *heap, i32 *row, i64 *ndist)
{
    const f32 *xv = X + x * d;
    int hn = 0;
    i64 i = 0;
    f32 d8[8];
    for (; i + 8 <= nc; i += 8) {                        /* 8 gathered rows/pass */
        if (i + 24 <= nc) {                              /* prefetch 2 blocks ahead:
                                                            whole rows (128B lines) */
            const i64 nb = ((i64)d * 4 + 127) >> 7;
            for (i64 l = 0; l < nb; l++)
                for (int r = 0; r < 8; r++)
                    FG_PREFETCH((const char *)(X + (i64)cand[i + 16 + r] * d) + l * 128);
        }
        fg_dist4(xv, X + (i64)cand[i]     * d, X + (i64)cand[i + 1] * d,
                     X + (i64)cand[i + 2] * d, X + (i64)cand[i + 3] * d,
                 d, metric, d8);
        fg_dist4(xv, X + (i64)cand[i + 4] * d, X + (i64)cand[i + 5] * d,
                     X + (i64)cand[i + 6] * d, X + (i64)cand[i + 7] * d,
                 d, metric, d8 + 4);
        for (int r = 0; r < 8; r++)
            hmax_offer(heap, &hn, k, pack_di(d8[r], cand[i + r]));
    }
    for (; i + 4 <= nc; i += 4) {
        fg_dist4(xv, X + (i64)cand[i]     * d, X + (i64)cand[i + 1] * d,
                     X + (i64)cand[i + 2] * d, X + (i64)cand[i + 3] * d,
                 d, metric, d8);
        for (int r = 0; r < 4; r++)
            hmax_offer(heap, &hn, k, pack_di(d8[r], cand[i + r]));
    }
    for (; i < nc; i++) {
        f32 dd = fg_dist(xv, X + (i64)cand[i] * d, d, metric);
        hmax_offer(heap, &hn, k, pack_di(dd, cand[i]));
    }
    *ndist += nc;
    std::sort(heap, heap + hn);                          /* asc (dist, id) */
    int a = 0;
    for (; a < hn; a++) row[a] = unpack_id(heap[a]);
    for (; a < k;  a++) row[a] = -1;
}

static void nnd_round(const i32 *Bin, i32 *Bout, const f32 *X, i64 n, int d,
                      int metric, int k, int threads, i64 *ndist)
{
    std::vector<i64> nd_acc((size_t)threads, 0);
    #pragma omp parallel num_threads(threads)
    {
        std::vector<u32> stamp((size_t)n, 0);
        u32 epoch = 0;
        std::vector<i32> cand((size_t)k * (size_t)(k + 1));
        std::vector<u64> heap((size_t)k);
        i64 nd_local = 0;
        #pragma omp for schedule(dynamic, 1024)
        for (i64 x = 0; x < n; x++) {
            if (++epoch == 0) { std::fill(stamp.begin(), stamp.end(), 0); epoch = 1; }
            stamp[x] = epoch;                            /* exclude self */
            i64 nc = 0;
            const i32 *bx = Bin + x * k;
            for (int a = 0; a < k; a++) {                /* own neighbours */
                i32 u = bx[a];
                if (u < 0) break;
                if (stamp[u] != epoch) { stamp[u] = epoch; cand[nc++] = u; }
            }
            for (int a = 0; a < k; a++) {                /* neighbours-of-neighbours */
                i32 u = bx[a];
                if (u < 0) break;
                if (a + 1 < k && bx[a + 1] >= 0)
                    FG_PREFETCH(Bin + (i64)bx[a + 1] * k);
                const i32 *bu = Bin + (i64)u * k;
                for (int b = 0; b < k; b++) {
                    i32 w = bu[b];
                    if (w < 0) break;
                    if (stamp[w] != epoch) { stamp[w] = epoch; cand[nc++] = w; }
                }
            }
            knearest_into(X, d, metric, x, cand.data(), nc, k,
                          heap.data(), Bout + x * k, &nd_local);
        }
        nd_acc[omp_get_thread_num()] += nd_local;
    }
    for (i64 v : nd_acc) *ndist += v;
}

/* ===========================================================================
 * Vote-counting aggregator (the additive recovery-tier path).
 *
 * Replaces csr_from_edges + a random degree cap for the data tier: instead of dedup
 * (which discards multiplicity) + random hub sampling (which ignores
 * confidence), it COUNTS how many trees mention each directed (i,j) pair and
 * keeps the per-vertex top vote_cap by (vote desc, id asc). Mention relation:
 * parent<->child plus an A-level ancestor climb (the routing path == the
 * ancestor chain in this top-down SAT, recoverable from parent[] by pointer-
 * chasing -- no kernel instrumentation), with level-decayed votes.
 *
 * Deterministic: output keyed by vertex id, every selection tie broken by id,
 * disjoint parallel writes -- same contract as the rest of the build.
 * =========================================================================== */
struct Men { u64 key; f32 w; };   /* key = (src<<32)|dst ; w = level weight */

VoteGraph aggregate_votes(const std::vector<std::vector<i32>> &parents,
                          const std::vector<i32> &roots, i64 n,
                          int A, f32 decay, int vote_cap, int threads, bool verbose)
{
    const int T = (int)parents.size();
    if (A < 1) A = 1;
    if (vote_cap < 1) vote_cap = 1;
    threads = threads > 0 ? threads : omp_get_max_threads();
    double t0 = wall();

    /* ---- count per-(tree, src-bucket) emissions for exact disjoint offsets.
     * Buckets are a fixed monotone function of the src id and trees are the
     * fixed blocks, so the scatter below is stable and thread-count
     * independent; concatenated buckets come out in ascending src order. ---- */
    int sb = 0; while (((i64)1 << sb) < n) sb++;
    const int shift = sb > 11 ? sb - 11 : 0;             /* <= 2048 buckets */
    const i64 NB    = ((n - 1) >> shift) + 1;
    std::vector<i64> hist((size_t)T * NB, 0);
    #pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
    for (int t = 0; t < T; t++) {
        const i32 *par = parents[t].data();
        const i32 root = roots[t];
        i64 *h = hist.data() + (i64)t * NB;
        for (i64 v = 0; v < n; v++) {
            if ((i32)v == root) continue;
            i32 a = par[v];
            for (int l = 0; a >= 0 && l < A; l++) {
                h[v >> shift]++;                         /* v -> ancestor */
                h[a >> shift]++;                         /* ancestor -> v */
                a = par[a];
            }
        }
    }
    std::vector<i64> bstart((size_t)NB + 1, 0);
    i64 cum = 0;
    for (i64 b = 0; b < NB; b++) {                       /* bucket-major offsets */
        bstart[b] = cum;
        for (int t = 0; t < T; t++) {
            i64 c = hist[(i64)t * NB + b];
            hist[(i64)t * NB + b] = cum;
            cum += c;
        }
    }
    bstart[NB] = cum;
    const i64 M = cum;

    /* ---- emit directed weighted mentions, scattered straight into their
     * src bucket (per-(tree,bucket) cursors: disjoint writes) ---- */
    std::vector<Men> P((size_t)M);
    #pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
    for (int t = 0; t < T; t++) {
        const i32 *par = parents[t].data();
        const i32 root = roots[t];
        i64 *h = hist.data() + (i64)t * NB;
        for (i64 v = 0; v < n; v++) {
            if ((i32)v == root) continue;
            i32 a = par[v];
            f32 wl = 1.0f;
            for (int l = 0; a >= 0 && l < A; l++) {
                u64 sv = (u64)(u32)v, sa = (u64)(u32)a;
                P[(size_t)h[v >> shift]++] = Men{ (sv << 32) | sa, wl };
                P[(size_t)h[a >> shift]++] = Men{ (sa << 32) | sv, wl };
                a = par[a]; wl *= decay;
            }
        }
    }
    std::vector<i64>().swap(hist);

    /* ---- per bucket: sort by key, sum weights within each (src,dst) run,
     * compact in place; then stitch the buckets into R ---- */
    std::vector<i64> blen((size_t)NB + 1, 0);
    #pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
    for (i64 b = 0; b < NB; b++) {
        Men *s = P.data() + bstart[b];
        const i64 m = bstart[b + 1] - bstart[b];
        std::sort(s, s + m,
                  [](const Men &x, const Men &y) { return x.key < y.key; });
        i64 w = 0;
        for (i64 i = 0; i < m;) {
            u64 key = s[i].key; f32 acc = 0;
            i64 j = i;
            while (j < m && s[j].key == key) { acc += s[j].w; j++; }
            s[w++] = Men{ key, acc };
            i = j;
        }
        blen[b + 1] = w;
    }
    for (i64 b = 0; b < NB; b++) blen[b + 1] += blen[b];
    std::vector<Men> R((size_t)blen[NB]);
    #pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
    for (i64 b = 0; b < NB; b++)
        memcpy(R.data() + blen[b], P.data() + bstart[b],
               (size_t)(blen[b + 1] - blen[b]) * sizeof(Men));
    std::vector<Men>().swap(P);

    /* ---- per-src segment boundaries in R (sorted by src) ---- */
    std::vector<i64> rbeg((size_t)n + 1, 0);
    for (const Men &m : R) rbeg[(i64)(m.key >> 32) + 1]++;
    for (i64 v = 0; v < n; v++) rbeg[v + 1] += rbeg[v];

    /* ---- cap each vertex to top vote_cap by (vote desc, id asc) ---- */
    VoteGraph vg;
    vg.ptr.assign((size_t)n + 1, 0);
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (i64 v = 0; v < n; v++) {
        i64 m = rbeg[v + 1] - rbeg[v];
        vg.ptr[v + 1] = m < vote_cap ? m : vote_cap;
    }
    for (i64 v = 0; v < n; v++) vg.ptr[v + 1] += vg.ptr[v];
    vg.idx.resize((size_t)vg.ptr[n]);
    vg.vote.resize((size_t)vg.ptr[n]);
    #pragma omp parallel for schedule(dynamic, 2048) num_threads(threads)
    for (i64 v = 0; v < n; v++) {
        Men *seg = R.data() + rbeg[v];
        i64 m = rbeg[v + 1] - rbeg[v];
        std::sort(seg, seg + m, [](const Men &a, const Men &b) {
            if (a.w != b.w) return a.w > b.w;          /* vote desc */
            return (u32)a.key < (u32)b.key;            /* dst id asc */
        });
        i64 keep = m < vote_cap ? m : vote_cap;
        i32 *oi = vg.idx.data()  + vg.ptr[v];
        f32 *ov = vg.vote.data() + vg.ptr[v];
        for (i64 j = 0; j < keep; j++) { oi[j] = (i32)(u32)seg[j].key; ov[j] = seg[j].w; }
    }

    vg.t_build    = wall() - t0;
    vg.n_mentions = M;
    vg.n_pairs    = (i64)R.size();
    if (verbose) {
        i64 maxdeg = 0; f64 sumdeg = 0;
        for (i64 v = 0; v < n; v++) { i64 dg = vg.deg(v); maxdeg = std::max(maxdeg, dg); sumdeg += (f64)dg; }
        printf("  aggregate: A=%d decay=%.2f cap=%d  %.2fs  %.2fM mentions -> "
               "%.2fM pairs; kept %.1f/vertex (max %lld)\n", A, (double)decay,
               vote_cap, vg.t_build, (double)M / 1e6,
               (double)vg.n_pairs / 1e6, sumdeg / (f64)n, (long long)maxdeg);
    }
    return vg;
}

/* seed NN-descent from the vote list, then refine. Candidates are already
 * capped to the top-vote per vertex (vote == seeding priority); knearest_into
 * computes the true distance and keeps the k nearest, so the final code edge
 * weight is the refined true-NN distance, not the vote. */
static void nnd_seed_from_votes(const VoteGraph &acc, const f32 *X, i64 n, int d,
                                int metric, int k, i32 *B, int threads, i64 *ndist)
{
    std::vector<i64> nd_acc((size_t)threads, 0);
    #pragma omp parallel num_threads(threads)
    {
        std::vector<u64> heap((size_t)k);
        i64 nd_local = 0;
        #pragma omp for schedule(dynamic, 2048)
        for (i64 x = 0; x < n; x++)
            knearest_into(X, d, metric, x, acc.idx.data() + acc.ptr[x], acc.deg(x),
                          k, heap.data(), B + x * k, &nd_local);
        nd_acc[omp_get_thread_num()] += nd_local;
    }
    for (i64 v : nd_acc) *ndist += v;
}

void refine_from_votes(const VoteGraph &acc, const f32 *X, i64 n, int d, int metric,
                       int k, int rounds, int threads,
                       std::vector<i32> &B, i64 *n_dist)
{
    threads = threads > 0 ? threads : omp_get_max_threads();
    B.assign((size_t)n * (size_t)k, -1);
    std::vector<i32> B2((size_t)n * (size_t)k, -1);
    i64 nd = 0;
    nnd_seed_from_votes(acc, X, n, d, metric, k, B.data(), threads, &nd);
    for (int r = 0; r < rounds; r++) {
        nnd_round(B.data(), B2.data(), X, n, d, metric, k, threads, &nd);
        B.swap(B2);
    }
    if (n_dist) *n_dist += nd;
}

ForestGraph code_to_graph(const i32 *B, int k, i64 n, int metric)
{
    std::vector<u64> E;
    E.reserve((size_t)2 * (size_t)n * (size_t)k);
    for (i64 v = 0; v < n; v++)
        for (int j = 0; j < k; j++) {
            i32 u = B[v * k + j];
            if (u < 0) break;
            u64 a = (u64)(u32)v, b = (u64)(u32)u;
            E.push_back((a << 32) | b);
            E.push_back((b << 32) | a);
        }
    ForestGraph fg;
    fg.g = csr_from_edges(E, n, 0);
    fg.metric = metric;
    /* entry set = the top-R highest-degree hubs (a pure k-NN graph has no
     * designated roots; multiple hubs diversify entry and, on data with
     * separated components, seed more than one of them). */
    const i64 R = std::min<i64>(32, n);
    std::vector<u64> by_deg((size_t)n);              /* (deg<<32 | id), desc sort */
    for (i64 v = 0; v < n; v++)
        by_deg[v] = ((u64)(u32)fg.g.deg(v) << 32) | (u32)v;
    std::partial_sort(by_deg.begin(), by_deg.begin() + R, by_deg.end(),
                      std::greater<u64>());
    fg.roots.clear();
    for (i64 i = 0; i < R; i++) fg.roots.push_back((i32)(u32)by_deg[i]);
    return fg;
}

void vote_seed(const VoteGraph &acc, const f32 *X, i64 n, int d, int metric,
               int k, i32 *B, int threads, i64 *n_dist)
{
    threads = threads > 0 ? threads : omp_get_max_threads();
    i64 nd = 0;
    nnd_seed_from_votes(acc, X, n, d, metric, k, B, threads, &nd);
    if (n_dist) *n_dist += nd;
}

void vote_descent_round(const i32 *Bin, i32 *Bout, const f32 *X, i64 n, int d,
                        int metric, int k, int threads, i64 *n_dist)
{
    threads = threads > 0 ? threads : omp_get_max_threads();
    i64 nd = 0;
    nnd_round(Bin, Bout, X, n, d, metric, k, threads, &nd);
    if (n_dist) *n_dist += nd;
}

/* ===========================================================================
 * Phase 1 (shared): the T parent[] arrays. Extracted so the navigation build
 * and the vote aggregator consume bitwise-identical trees.
 * =========================================================================== */
void build_parents(const f32 *X, i64 n, int d, int metric, const BuildParams &bp,
                   std::vector<std::vector<i32>> &parents, std::vector<i32> &roots,
                   i64 *n_dist, double *t_trees)
{
    const int threads = bp.threads > 0 ? bp.threads : omp_get_max_threads();
    const int T = bp.T;
    /* Forest-build concurrency cap. Each in-flight tree holds a TreeScratch whose
       child_vecs gather is ~N*d*4 bytes (the dominant build-memory term), so peak build
       RAM ~ conc * N*d*4. build_conc<=0 keeps the current unlimited behaviour. This is
       GRAPH-PRESERVING: every tree is keyed by (bp.seed + t) and writes a disjoint
       parents[t], so running `conc` at a time is bitwise-identical to all-at-once. */
    const int conc = (bp.build_conc > 0 && bp.build_conc < threads) ? bp.build_conc : threads;
    double t0 = wall();
    parents.assign((size_t)T, {});
    roots.assign((size_t)T, -1);
    std::vector<i64> nd_tree((size_t)T, 0);
    #pragma omp parallel for schedule(dynamic, 1) num_threads(conc)
    for (int t = 0; t < T; t++) {
        parents[t].assign((size_t)n, -1);
        build_one_tree(X, n, d, metric, bp.seed + (u64)t, bp.policy,
                       t == 0 ? 0 : bp.tree_depth,   /* tree 0 = full spine */
                       parents[t].data(), &roots[t], &nd_tree[t]);
    }
    i64 nd = 0;
    for (int t = 0; t < T; t++) nd += nd_tree[t];
    if (n_dist)  *n_dist  += nd;
    if (t_trees) *t_trees  = wall() - t0;
}

/* ===========================================================================
 * The pipeline
 * =========================================================================== */

/* ===========================================================================
 * Dense (search-free) construction: pair sets fixed in advance.
 *
 * 1. leaders: a seeded sample, L leaders, Lloyd iterations (per-leader sums
 *    accumulated in member-index order -> deterministic);
 * 2. every point joins the leaves of its m nearest leaders (counting sort);
 * 3. inside every leaf, every member scans all members with the 4-wide
 *    kernel and offers each pair to its own bounded top-C heap keyed by
 *    (distance, id). Leaves are processed one after another and a point owns
 *    exactly one row per leaf, so there are no locks and no races;
 * 4. candidate lists (deduplicated across shared leaves) -> symmetrize ->
 *    the harvest's occlusion prune, exactly as the beam harvest ends.
 * Deterministic for every thread count: all parallel loops write disjoint
 * rows and all reductions are in fixed order.
 * =========================================================================== */
static inline i32 dense_nearest_leader(const f32 *x, const f32 *lead, int L, int d,
                                       int metric)
{
    u64 best = ~0ull;
    for (int l = 0; l < L; l++) {
        u64 pk = pack_di(fg_dist(x, lead + (i64)l * d, d, metric), l);
        if (pk < best) best = pk;
    }
    return unpack_id(best);
}

ForestGraph build_dense_graph(const f32 *X, i64 n, int d, int metric,
                              const BuildParams &bp)
{
    ForestGraph fg;
    fg.metric = metric;
    const int threads = bp.threads > 0 ? bp.threads : omp_get_max_threads();
    const int L = bp.dense_leaders, m = std::min(bp.dense_m, L), C = bp.dense_C;
    i64 nd_lead = 0, nd_s1 = 0, nd_pr = 0;
    double t0 = wall();

    /* ---- 1. leaders ------------------------------------------------------ */
    Rng rng(mix_seed(bp.seed, 0x44454E53ull));               /* "DENS" */
    const i64 ns = std::min<i64>(n, 200000);
    std::vector<i32> samp((size_t)ns);
    for (i64 i = 0; i < ns; i++) samp[(size_t)i] = (i32)rng.below((u64)n);
    std::vector<f32> lead((size_t)L * d);
    for (int l = 0; l < L; l++)
        memcpy(&lead[(size_t)l * d], X + (i64)samp[(size_t)l] * d, (size_t)d * sizeof(f32));
    std::vector<i32> asg((size_t)ns);
    for (int it = 0; it < bp.dense_kmeans; it++) {
        #pragma omp parallel for schedule(static) num_threads(threads)
        for (i64 i = 0; i < ns; i++)
            asg[(size_t)i] = dense_nearest_leader(X + (i64)samp[(size_t)i] * d, lead.data(), L, d, metric);
        nd_lead += ns * (i64)L;
        std::vector<std::vector<i32>> mem((size_t)L);
        for (i64 i = 0; i < ns; i++) mem[(size_t)asg[(size_t)i]].push_back((i32)i);
        #pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
        for (int l = 0; l < L; l++) {
            if (mem[(size_t)l].empty()) continue;
            std::vector<f64> acc((size_t)d, 0.0);
            for (i32 i : mem[(size_t)l]) {
                const f32 *x = X + (i64)samp[(size_t)i] * d;
                for (int j = 0; j < d; j++) acc[(size_t)j] += x[j];
            }
            f64 nrm = 0;
            for (int j = 0; j < d; j++) nrm += acc[(size_t)j] * acc[(size_t)j];
            nrm = std::sqrt(nrm);
            const f64 div = (metric == METRIC_COSINE && nrm > 0) ? nrm : (f64)mem[(size_t)l].size();
            for (int j = 0; j < d; j++) lead[(size_t)l * d + j] = (f32)(acc[(size_t)j] / div);
        }
    }

    /* ---- 2. memberships + leaf lists ------------------------------------- */
    std::vector<i32> memb((size_t)n * m);
    std::vector<u64> lmed((size_t)L, ~0ull);                /* per leader: nearest primary member (dist, id) */
    #pragma omp parallel num_threads(threads)
    {
        std::vector<u64> pk((size_t)L), best((size_t)L, ~0ull);
        #pragma omp for schedule(static)
        for (i64 p = 0; p < n; p++) {
            const f32 *x = X + p * d;
            for (int l = 0; l < L; l++) pk[(size_t)l] = pack_di(fg_dist(x, lead.data() + (i64)l * d, d, metric), l);
            std::partial_sort(pk.begin(), pk.begin() + m, pk.end());
            for (int j = 0; j < m; j++) memb[(size_t)p * m + j] = unpack_id(pk[(size_t)j]);
            const int l0 = unpack_id(pk[0]);
            const u64 key = pack_di(unpack_dist(pk[0]), (i32)p);
            if (key < best[(size_t)l0]) best[(size_t)l0] = key;
        }
        #pragma omp critical
        for (int l = 0; l < L; l++) if (best[(size_t)l] < lmed[(size_t)l]) lmed[(size_t)l] = best[(size_t)l];
    }
    nd_lead += n * (i64)L;
    std::vector<i64> lptr((size_t)L + 1, 0);
    for (size_t i = 0; i < memb.size(); i++) lptr[(size_t)memb[i] + 1]++;
    for (int l = 0; l < L; l++) lptr[(size_t)l + 1] += lptr[(size_t)l];
    std::vector<i32> lidx((size_t)n * m);
    {
        std::vector<i64> fill(lptr.begin(), lptr.end() - 1);
        for (i64 p = 0; p < n; p++)
            for (int j = 0; j < m; j++) lidx[(size_t)fill[(size_t)memb[(size_t)p * m + j]]++] = (i32)p;
    }
    i64 lmax = 0; for (int l = 0; l < L; l++) lmax = std::max(lmax, lptr[(size_t)l + 1] - lptr[(size_t)l]);
    fg.t_trees = wall() - t0;

    /* ---- 3. per-point top-C over the point's leaves ----------------------
     * Symmetric scan: every unordered pair inside a leaf is evaluated once
     * and offered to BOTH endpoints' heaps. The leaf's members are split
     * into blocks of TB; the block pairs (a <= b) are scheduled as a
     * round-robin tournament, so within a round each block belongs to exactly
     * one tile and the thread computing a tile owns every heap it writes --
     * no locks, no races, and a heap's content does not depend on the order
     * of its offers (it keeps the C smallest (distance, id) keys), so the
     * result is identical for any thread count and any schedule.
     * A pair (p, c) is offered exactly once, at the first leaf the two
     * share: dup(p, c) = (earlier leaves of p) & (earlier leaves of c) != 0,
     * with the masks precomputed per leaf member. */
    t0 = wall();
    std::vector<u64> heap((size_t)n * C);
    std::vector<int> hs((size_t)n, 0);
    const int TB = getenv("FG_DENSE_TB") ? atoi(getenv("FG_DENSE_TB")) : 128;
    const int W = (L + 63) / 64;
    std::vector<u64> lmask;                                 /* earlier-leaf mask per member */
    for (int l = 0; l < L; l++) {
        const i32 *M = lidx.data() + lptr[(size_t)l];
        const i64 sz = lptr[(size_t)l + 1] - lptr[(size_t)l];
        const i64 B = (sz + TB - 1) / TB, Bp = B + (B & 1);   /* pad to even for the tournament */
        lmask.assign((size_t)sz * W, 0);
        #pragma omp parallel for schedule(static) num_threads(threads)
        for (i64 r = 0; r < sz; r++) {
            u64 *mk = &lmask[(size_t)r * W];
            const i32 *lp = memb.data() + (size_t)M[r] * m;
            for (int j = 0; j < m; j++)
                if (lp[j] < l) mk[lp[j] >> 6] |= 1ull << (lp[j] & 63);
        }
        std::vector<i64> nd_acc((size_t)threads, 0);
        auto dup = [&](i64 r, i64 c) {
            const u64 *a = &lmask[(size_t)r * W], *b = &lmask[(size_t)c * W];
            u64 hit = 0;
            for (int w = 0; w < W; w++) hit |= a[w] & b[w];
            return hit != 0;
        };
        /* one tile: rows [ra,re) x cols [ca,ce) of the leaf; diagonal tiles do r < c only */
        auto tile = [&](i64 ra, i64 re, i64 ca, i64 ce, bool diag, i64 &nd) {
            f32 d16[16];
            for (i64 r = ra; r < re; r += 4) {
                const int nr = (int)std::min<i64>(4, re - r);
                for (i64 c = ca; c < ce; c += 4) {
                    const int nc = (int)std::min<i64>(4, ce - c);
                    if (diag && c + nc - 1 <= r) continue;         /* whole group below the diagonal */
                    if (nr == 4 && nc == 4) {
                        fg_dist4x4(X + (i64)M[r] * d, X + (i64)M[r + 1] * d, X + (i64)M[r + 2] * d, X + (i64)M[r + 3] * d,
                                   X + (i64)M[c] * d, X + (i64)M[c + 1] * d, X + (i64)M[c + 2] * d, X + (i64)M[c + 3] * d,
                                   d, metric, d16);
                    } else {
                        for (int a = 0; a < nr; a++)
                            for (int t = 0; t < nc; t++)
                                d16[a * 4 + t] = fg_dist(X + (i64)M[r + a] * d, X + (i64)M[c + t] * d, d, metric);
                    }
                    if (!diag) nd += (i64)nr * nc;
                    else for (int a = 0; a < nr; a++) for (int t = 0; t < nc; t++) nd += (c + t > r + a);
                    /* two-sided prefilter: a full heap admits only d <= its worst kept distance.
                     * Row side on d16 rows; column side on the transposed tile. */
                    f32 tauq[4]; int passr[4] = {0, 0, 0, 0}, passc[4] = {0, 0, 0, 0};
                    for (int t = 0; t < nc; t++) {
                        const int *hq = &hs[(size_t)M[c + t]];
                        tauq[t] = (*hq >= C) ? unpack_dist(heap[(size_t)M[c + t] * C]) : 3.0e38f;
                    }
                    int any = 0;
                    for (int a = 0; a < nr; a++) {
                        const int *hp = &hs[(size_t)M[r + a]];
                        const f32 taup = (*hp >= C) ? unpack_dist(heap[(size_t)M[r + a] * C]) : 3.0e38f;
                        passr[a] = fg_le4_mask(d16 + a * 4, taup) & ((1 << nc) - 1);
                        any |= passr[a];
                    }
                    {
                        f32 tr[16];                                 /* tr[t*4 + a] = d16[a*4 + t] */
                        for (int a = 0; a < 4; a++) for (int t = 0; t < 4; t++) tr[t * 4 + a] = d16[a * 4 + t];
                        for (int t = 0; t < nc; t++) { passc[t] = fg_le4_mask(tr + t * 4, tauq[t]) & ((1 << nr) - 1); any |= passc[t]; }
                    }
                    if (!any) continue;
                    for (int a = 0; a < nr; a++) {
                        const i32 p = M[r + a];
                        u64 *Hp = heap.data() + (size_t)p * C; int *hp = &hs[(size_t)p];
                        for (int t = 0; t < nc; t++) {
                            const bool rp = (passr[a] >> t) & 1, cp = (passc[t] >> a) & 1;
                            if (!(rp | cp)) continue;
                            if (diag && c + t <= r + a) continue;       /* below/on the diagonal */
                            if (dup(r + a, c + t)) continue;
                            const f32 dist = d16[a * 4 + t];
                            const i32 q = M[c + t];
                            if (rp) hmax_offer(Hp, hp, C, pack_di(dist, q));
                            if (cp) hmax_offer(heap.data() + (size_t)q * C, &hs[(size_t)q], C, pack_di(dist, p));
                        }
                    }
                }
            }
        };
        /* tile list in tournament order (consecutive tiles touch disjoint blocks),
         * consumed through a shared counter; a spinlock per block makes any
         * schedule race-free, and the result is schedule-independent because a
         * heap keeps the C smallest keys whatever the order of its offers. */
        std::vector<std::pair<i32, i32>> tiles;
        tiles.reserve((size_t)(Bp / 2) * (size_t)(Bp - 1) + (size_t)B);
        for (i64 rnd = 0; rnd < Bp - 1; rnd++)
            for (i64 k = 0; k < Bp / 2; k++) {
                i64 a, b;
                if (k == 0) { a = rnd; b = Bp - 1; }
                else { a = (rnd + k) % (Bp - 1); b = (rnd - k + (Bp - 1)) % (Bp - 1); }
                if (a > b) std::swap(a, b);
                if (b < B) tiles.emplace_back((i32)a, (i32)b);
            }
        for (i64 a = 0; a < B; a++) tiles.emplace_back((i32)a, (i32)a);
        std::vector<std::atomic<int>> block_lock((size_t)B);
        for (auto &x : block_lock) x.store(0, std::memory_order_relaxed);
        std::atomic<i64> next_tile(0);
        const i64 ntiles = (i64)tiles.size();
        #pragma omp parallel num_threads(threads)
        {
            i64 nd = 0;
            auto lock = [&](i32 blk) {
                int expect = 0;
                while (!block_lock[(size_t)blk].compare_exchange_weak(expect, 1, std::memory_order_acquire)) {
                    expect = 0;
                    FG_SPIN_PAUSE();
                }
            };
            auto unlock = [&](i32 blk) { block_lock[(size_t)blk].store(0, std::memory_order_release); };
            for (;;) {
                const i64 i = next_tile.fetch_add(1, std::memory_order_relaxed);
                if (i >= ntiles) break;
                const i32 a = tiles[(size_t)i].first, b = tiles[(size_t)i].second;
                lock(a); if (b != a) lock(b);
                tile(a * (i64)TB, std::min(sz, (a + 1) * (i64)TB), b * (i64)TB, std::min(sz, (b + 1) * (i64)TB), a == b, nd);
                if (b != a) unlock(b);
                unlock(a);
            }
            nd_acc[(size_t)omp_get_thread_num()] += nd;
        }
        for (i64 v : nd_acc) nd_s1 += v;
    }
    fg.t_harvest = wall() - t0;

    /* ---- 4. candidate CSR (dedupe across shared leaves) ------------------ */
    t0 = wall();
    std::vector<std::vector<i32>> rows((size_t)n);
    #pragma omp parallel for schedule(dynamic, 2048) num_threads(threads)
    for (i64 p = 0; p < n; p++) {
        u64 *H = heap.data() + (size_t)p * C;
        std::sort(H, H + hs[(size_t)p]);                      /* by (distance, id) */
        std::vector<i32> &row = rows[(size_t)p];
        row.reserve((size_t)hs[(size_t)p]);
        for (int i = 0; i < hs[(size_t)p]; i++) {
            const i32 c = unpack_id(H[i]);
            bool dup = false;
            for (i32 u : row) if (u == c) { dup = true; break; }   /* rare; lists are short */
            if (!dup) row.push_back(c);
        }
    }
    heap.clear(); heap.shrink_to_fit();
    CSR cand;
    cand.ptr.assign((size_t)n + 1, 0);
    for (i64 p = 0; p < n; p++) cand.ptr[(size_t)p + 1] = cand.ptr[(size_t)p] + (i64)rows[(size_t)p].size();
    cand.idx.resize((size_t)cand.ptr[(size_t)n]);
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (i64 p = 0; p < n; p++)
        memcpy(cand.idx.data() + cand.ptr[(size_t)p], rows[(size_t)p].data(), rows[(size_t)p].size() * sizeof(i32));
    rows.clear(); rows.shrink_to_fit();
    const i64 cand_edges = (i64)cand.idx.size();

    /* ---- 5. prune ---------------------------------------------------------
     * dense_prune 0: GRAFT's ending -- symmetrize the candidate lists, prune
     *   once (every list occlusion-pruned to <= cap).
     * dense_prune 1: Vamana's ending -- prune each point's own candidates to
     *   <= cap, add the reverse edges, and re-prune only the lists that now
     *   exceed cap (short lists keep every reverse edge: the in-links of a
     *   pruned list are the long-range navigation edges). */
    if (bp.dense_prune == 0) {
        CSR sy = csr_symmetrize(cand, n, threads);
        cand.ptr.clear(); cand.idx.clear();
        fg.g = hsp_prune(sy, X, n, d, metric, bp.harvest_cap, bp.harvest_alpha,
                         bp.harvest_cand, /*order=*/0, bp.seed, threads, &nd_pr);
    } else {
        CSR g1 = hsp_prune(cand, X, n, d, metric, bp.harvest_cap, bp.harvest_alpha,
                           bp.harvest_cand, /*order=*/0, bp.seed, threads, &nd_pr);
        cand.ptr.clear(); cand.idx.clear();
        CSR un = csr_symmetrize(g1, n, threads);            /* own + reverse, deduped */
        CSR pr = hsp_prune(un, X, n, d, metric, bp.harvest_cap, bp.harvest_alpha,
                           bp.harvest_cand, /*order=*/0, bp.seed, threads, &nd_pr);
        CSR out;                                             /* keep un's row where it fits */
        out.ptr.assign((size_t)n + 1, 0);
        for (i64 v = 0; v < n; v++)
            out.ptr[(size_t)v + 1] = out.ptr[(size_t)v] +
                (un.deg(v) <= bp.harvest_cap ? un.deg(v) : pr.deg(v));
        out.idx.resize((size_t)out.ptr[(size_t)n]);
        #pragma omp parallel for schedule(static) num_threads(threads)
        for (i64 v = 0; v < n; v++) {
            const CSR &src = un.deg(v) <= bp.harvest_cap ? un : pr;
            memcpy(out.idx.data() + out.ptr[(size_t)v], src.idx.data() + src.ptr[(size_t)v],
                   (size_t)src.deg(v) * sizeof(i32));
        }
        fg.g = std::move(out);
    }
    fg.t_refine = wall() - t0;
    fg.roots.assign(1, 0);
    /* ---- 6. spine: one SAT tree, its parent edges unioned after the prune ---- */
    i64 nd_sp = 0; double t_sp = 0;
    if (bp.dense_spine == 2) {
        /* k-nearest-leader graph (k = 8, O(L^2) distances), realized between the
         * leaders' medoid members and unioned after the prune. Connects the
         * leaves when the data is clustered, at negligible cost. */
        double t1 = wall();
        const int KL = std::min(8, L - 1);
        std::vector<u64> E;
        E.reserve(fg.g.idx.size() + 2 * (size_t)L * (size_t)KL);
        for (i64 v = 0; v < n; v++)
            for (i64 jj = fg.g.ptr[v]; jj < fg.g.ptr[v + 1]; jj++)
                E.push_back(((u64)(u32)v << 32) | (u32)fg.g.idx[jj]);
        std::vector<u64> pk((size_t)L);
        for (int l = 0; l < L; l++) {
            if (lmed[(size_t)l] == ~0ull) continue;
            for (int l2 = 0; l2 < L; l2++)
                pk[(size_t)l2] = (l2 == l) ? ~0ull : pack_di(fg_dist(lead.data() + (i64)l * d, lead.data() + (i64)l2 * d, d, metric), l2);
            std::partial_sort(pk.begin(), pk.begin() + KL, pk.end());
            nd_sp += L;
            const u64 a = (u64)(u32)unpack_id(lmed[(size_t)l]);
            for (int j = 0; j < KL; j++) {
                const int l2 = unpack_id(pk[(size_t)j]);
                if (lmed[(size_t)l2] == ~0ull) continue;
                const u64 b = (u64)(u32)unpack_id(lmed[(size_t)l2]);
                if (a == b) continue;
                E.push_back((a << 32) | b);
                E.push_back((b << 32) | a);
            }
        }
        /* navigable entry: a star from the global medoid (the leaf medoid nearest
         * the leaders' centroid) to every leaf medoid, so a greedy beam that
         * starts at the hub can reach any leaf in one hop. Connectivity alone
         * is not enough: a beam never expands a medoid that is farther from
         * the query than its local frontier (measured: leader graph alone left
         * the clustered smoke at recall 0.002). */
        {
            std::vector<f64> cen((size_t)d, 0.0);
            for (int l = 0; l < L; l++) for (int j = 0; j < d; j++) cen[(size_t)j] += lead[(size_t)l * d + j];
            std::vector<f32> c32((size_t)d);
            f64 nrm = 0; for (int j = 0; j < d; j++) nrm += cen[(size_t)j] * cen[(size_t)j];
            nrm = std::sqrt(nrm);
            for (int j = 0; j < d; j++) c32[(size_t)j] = (f32)(cen[(size_t)j] / ((metric == METRIC_COSINE && nrm > 0) ? nrm : (f64)L));
            u64 best = ~0ull;
            for (int l = 0; l < L; l++) {
                if (lmed[(size_t)l] == ~0ull) continue;
                const i32 p = unpack_id(lmed[(size_t)l]);
                const u64 key = pack_di(fg_dist(c32.data(), X + (i64)p * d, d, metric), p);
                if (key < best) best = key;
            }
            nd_sp += L;
            if (best != ~0ull) {
                const u64 g = (u64)(u32)unpack_id(best);
                for (int l = 0; l < L; l++) {
                    if (lmed[(size_t)l] == ~0ull) continue;
                    const u64 b = (u64)(u32)unpack_id(lmed[(size_t)l]);
                    if (b == g) continue;
                    E.push_back((g << 32) | b);
                    E.push_back((b << 32) | g);
                }
                fg.roots.assign(1, (i32)g);
            }
        }
        fg.g = csr_from_edges(E, n, threads);
        t_sp = wall() - t1;
        fg.t_refine += t_sp;
    } else if (bp.dense_spine == 1) {
        double t1 = wall();
        std::vector<std::vector<i32>> parents(1, std::vector<i32>((size_t)n, -1));
        std::vector<i32> roots(1, -1);
        build_one_tree(X, n, d, metric, bp.seed, bp.policy, 0, parents[0].data(), &roots[0], &nd_sp, threads);
        t_sp = wall() - t1; t1 = wall();
        std::vector<u64> E;
        E.reserve(fg.g.idx.size() + 2 * (size_t)(n - 1));
        for (i64 v = 0; v < n; v++)
            for (i64 jj = fg.g.ptr[v]; jj < fg.g.ptr[v + 1]; jj++)
                E.push_back(((u64)(u32)v << 32) | (u32)fg.g.idx[jj]);
        const i32 root0 = roots[0];
        for (i64 v = 0; v < n; v++) {
            if ((i32)v == root0) continue;
            u64 a = (u64)(u32)parents[0][(size_t)v], b = (u64)(u32)v;
            E.push_back((a << 32) | b);
            E.push_back((b << 32) | a);
        }
        fg.g = csr_from_edges(E, n, threads);
        fg.roots = roots;
        t_sp += wall() - t1;
        fg.t_refine += t_sp;
    }
    fg.n_dist_build = nd_lead + nd_s1 + nd_pr + nd_sp;
    if (bp.verbose) {
        i64 E = (i64)fg.g.idx.size(), mx = 0;
        for (i64 v = 0; v < n; v++) mx = std::max(mx, fg.g.deg(v));
        printf("  dense: L=%d m=%d C=%d over n=%lld  leaves mean %.0f max %lld  "
               "leaders %.2fs candidates %.2fs prune %.2fs (%d threads), %.1fM dist "
               "(leaders %.1fM, candidates %.1fM, prune %.1fM, spine %.1fM in %.2fs)\n",
               L, m, C, (long long)n, (double)n * m / L, (long long)lmax,
               fg.t_trees, fg.t_harvest, fg.t_refine, threads,
               (double)fg.n_dist_build / 1e6, (double)nd_lead / 1e6,
               (double)nd_s1 / 1e6, (double)nd_pr / 1e6, (double)nd_sp / 1e6, t_sp);
        printf("  candidates: %lld directed (%.1f/vertex); edges: %lld directed (%.1f/vertex, max %lld); memory %.0fMB\n",
               (long long)cand_edges, (double)cand_edges / n, (long long)E, (double)E / n,
               (long long)mx, (double)(E * sizeof(i32) + (n + 1) * sizeof(i64)) / 1e6);
    }
    return fg;
}

ForestGraph build_forest_graph(const f32 *X, i64 n, int d, int metric,
                               const BuildParams &bp)
{
    ForestGraph fg;
    fg.metric = metric;
    const int threads = bp.threads > 0 ? bp.threads : omp_get_max_threads();

    const int T = bp.T;

    /* ---- phase 1: T independent trees ---- */
    std::vector<std::vector<i32>> parents;
    build_parents(X, n, d, metric, bp, parents, fg.roots, &fg.n_dist_build, &fg.t_trees);
    if (bp.dump_parents) {
        FILE *f = fopen(bp.dump_parents, "wb");
        if (f) { fwrite(parents[0].data(), sizeof(i32), (size_t)n, f); fclose(f); }
    }
    std::vector<u8> logmask;
    std::vector<std::vector<i32>> vlog;
    if (bp.log_mask && bp.log_visited) {
        logmask.assign((size_t)n, 0);
        FILE *f = fopen(bp.log_mask, "rb");
        if (f) { size_t r = fread(logmask.data(), 1, (size_t)n, f); (void)r; fclose(f); }
        vlog.resize((size_t)n);
    }

    /* ---- phases 2+3: overlay + backward edges ---- */
    double t0 = wall();
    CSR raw;
    if (bp.substrate_random > 0) {
        /* random K-regular substrate (the reviewer's control): each vertex gets
         * K seeded random out-neighbours, then symmetrize. Same degree budget
         * as a T-tree union when K ~ 2T. Deterministic (vertex-keyed RNG). */
        const int K = bp.substrate_random;
        std::vector<u64> E;
        E.reserve((size_t)n * K * 2);
        for (i64 v = 0; v < n; v++) {
            Rng r(mix_seed(bp.seed, 0x52414E44ull ^ (u64)(u32)v)); /* "RAND" */
            for (int j = 0; j < K; j++) {
                i64 u = (i64)r.below((u64)n);
                if (u == v) u = (u + 1) % n;
                E.push_back(((u64)(u32)v << 32) | (u32)u);
                E.push_back(((u64)(u32)u << 32) | (u32)v);
            }
        }
        raw = csr_from_edges(E, n, threads);
    } else {
        raw = overlay_symmetrize(parents, fg.roots, n, threads,
                                 bp.overlay_directed);
    }
    /* keep tree 0's parent[] if the harvest will need its spine (below) */
    std::vector<i32> spine;
    if (bp.harvest_ef > 0) spine = parents[0];
    parents.clear(); parents.shrink_to_fit();
    /* optional scaffold degree cap: keep the nearest scaffold_cap per vertex,
     * same deterministic recipe as the final-graph cap below (build-space
     * distances, packed (dist,id) sort, then ids re-sorted). Harvest cost is
     * ef x scaffold degree, so this bounds the dominant phase directly. */
    if (bp.scaffold_cap > 0) {
        const int K = bp.scaffold_cap;
        #pragma omp parallel num_threads(threads)
        {
            std::vector<u64> byd;
            #pragma omp for schedule(dynamic, 2048)
            for (i64 v = 0; v < n; v++) {
                const i64 dg = raw.deg(v);
                if (dg <= K) continue;
                const f32 *xv = X + v * d;
                byd.clear();
                for (i64 j = raw.ptr[v]; j < raw.ptr[v + 1]; j++) {
                    const i32 u = raw.idx[j];
                    byd.push_back(pack_di(fg_dist(xv, X + (i64)u * d, d,
                                                  metric), u));
                }
                std::sort(byd.begin(), byd.end());
                i32 *dst = raw.idx.data() + raw.ptr[v];
                for (int t2 = 0; t2 < K; t2++)
                    dst[t2] = unpack_id(byd[(size_t)t2]);
                std::sort(dst, dst + K);
                for (i64 j = raw.ptr[v] + K; j < raw.ptr[v + 1]; j++)
                    raw.idx[j] = -1;
            }
        }
        CSR c;
        c.ptr.assign((size_t)n + 1, 0);
        for (i64 v = 0; v < n; v++)
            c.ptr[v + 1] = c.ptr[v] + std::min<i64>(raw.deg(v), K);
        c.idx.resize((size_t)c.ptr[n]);
        #pragma omp parallel for schedule(static) num_threads(threads)
        for (i64 v = 0; v < n; v++) {
            const i64 keep = c.ptr[v + 1] - c.ptr[v];
            memcpy(c.idx.data() + c.ptr[v], raw.idx.data() + raw.ptr[v],
                   (size_t)keep * sizeof(i32));
        }
        raw = std::move(c);
    }
    fg.t_overlay = wall() - t0;
    i64 raw_edges = (i64)raw.idx.size();
    i64 raw_maxdeg = 0;
    for (i64 v = 0; v < n; v++) raw_maxdeg = std::max(raw_maxdeg, raw.deg(v));

    /* ---- phase 4: batched search-harvest (THE graph construction) ----
     * The union is the SCAFFOLD: searchable, redundant, connectivity-true, but
     * its edges were never chosen for a searcher. The harvest asks the only
     * question that matters -- "what does a search for p actually traverse?" --
     * and keeps those edges.
     * harvest_ef == 0 serves the raw union unmodified (scaffold/debug mode). */
    CSR g;
    if (bp.harvest_ef > 0) {
        t0 = wall();
        i64 nd_h = 0, n_short = 0;
        std::vector<u64> shortcuts;
        if (bp.harvest_blocks > 1) {
            /* block-synchronous mutation: the substrate is scaffold UNION the
             * rows harvested so far (symmetrized), rebuilt between blocks. */
            const int B = bp.harvest_blocks;
            std::vector<std::vector<i32>> rows((size_t)n);
            CSR sub = raw;                       /* keep raw for the unions */
            for (int b = 0; b < B; b++) {
                harvest_pass(sub, X, n, d, metric, fg.roots.data(),
                             (int)fg.roots.size(), bp.harvest_ef,
                             bp.harvest_cap, bp.harvest_alpha, bp.harvest_cand,
                             bp.harvest_straighten, bp.harvest_patience,
                             threads, &nd_h,
                             bp.harvest_straighten > 0 ? &shortcuts : nullptr,
                             B, b, &rows,
                             logmask.empty() ? nullptr : logmask.data(),
                             logmask.empty() ? nullptr : &vlog);
                if (b + 1 >= B) break;
                std::vector<u64> E;
                E.reserve(raw.idx.size() + 2 * (size_t)n * (size_t)bp.harvest_cap);
                for (i64 v = 0; v < n; v++)
                    for (i64 j = raw.ptr[v]; j < raw.ptr[v + 1]; j++)
                        E.push_back(((u64)(u32)v << 32) | (u32)raw.idx[j]);
                for (i64 v = 0; v < n; v++)
                    for (i32 u : rows[v]) {
                        E.push_back(((u64)(u32)v << 32) | (u32)u);
                        E.push_back(((u64)(u32)u << 32) | (u32)v);
                    }
                sub = csr_from_edges(E, n, threads);
            }
            CSR out;
            out.ptr.assign((size_t)n + 1, 0);
            for (i64 v = 0; v < n; v++)
                out.ptr[v + 1] = out.ptr[v] + (i64)rows[v].size();
            out.idx.resize((size_t)out.ptr[n]);
            #pragma omp parallel for schedule(static) num_threads(threads)
            for (i64 v = 0; v < n; v++)
                memcpy(out.idx.data() + out.ptr[v], rows[v].data(),
                       rows[v].size() * sizeof(i32));
            g = std::move(out);
        } else {
        g = std::move(raw);
        for (int r = 0; r < bp.harvest_rounds; r++)
            g = harvest_pass(g, X, n, d, metric, fg.roots.data(),
                             (int)fg.roots.size(), bp.harvest_ef,
                             bp.harvest_cap, bp.harvest_alpha, bp.harvest_cand,
                             bp.harvest_straighten, bp.harvest_patience,
                             threads, &nd_h,
                             bp.harvest_straighten > 0 ? &shortcuts : nullptr,
                             1, 0, nullptr,
                             logmask.empty() ? nullptr : logmask.data(),
                             logmask.empty() ? nullptr : &vlog);
        }
        if (!logmask.empty()) {
            FILE *f = fopen(bp.log_visited, "wb");
            if (f) {
                for (i64 p = 0; p < n; p++) {
                    if (!logmask[p]) continue;
                    i32 hdr[2] = { (i32)p, (i32)vlog[p].size() };
                    fwrite(hdr, sizeof(i32), 2, f);
                    fwrite(vlog[p].data(), sizeof(i32), vlog[p].size(), f);
                }
                fclose(f);
            }
        }
        n_short = (i64)shortcuts.size();
        CSR sy = csr_symmetrize(g, n, threads);
        g = hsp_prune(sy, X, n, d, metric, bp.harvest_cap, bp.harvest_alpha,
                      bp.harvest_cand, /*order=*/0, bp.seed, threads, &nd_h);
        /* connectivity guarantee: tree 0 is a spanning tree; union its spine
         * back unpruned (<=2 edges/vertex). At alpha=1 the prune occludes every
         * cross-cluster candidate behind a near one, so without this the graph
         * shatters on clustered data (measured: recall 0.001 pre-fix). */
        {
            std::vector<u64> E;
            E.reserve((size_t)g.idx.size() + 2 * (size_t)(n - 1));
            for (i64 v = 0; v < n; v++)
                for (i64 jj = g.ptr[v]; jj < g.ptr[v + 1]; jj++)
                    E.push_back(((u64)(u32)v << 32) | (u32)g.idx[jj]);
            const i32 root0 = fg.roots[0];
            for (i64 v = 0; v < n; v++) {
                if ((i32)v == root0) continue;
                u64 a = (u64)(u32)spine[v], b = (u64)(u32)v;
                E.push_back((a << 32) | b);
                E.push_back((b << 32) | a);
            }
            /* string-pulled shortcuts join HERE, after the prune, for the same
             * reason the spine does: they are LONG by construction (that is
             * their purpose), and nearest-first occlusion at alpha=1 deletes
             * every long edge behind a nearer one -- merging them before the
             * prune emitted ~1M shortcuts and kept none (measured: degree and
             * recall bitwise unchanged). Long edges cannot be selected by the
             * rule that exists to remove them.
             * They do need a PER-SOURCE cap, though: every route's first hop
             * leaves one of the T entry roots, so unbounded merging gave the
             * roots ~31k out-edges each (measured max degree 32,798) and a beam
             * pays a node's full degree on landing. Keep the nearest
             * shortcut_cap per source -- deterministic (packed (dist,id) sort),
             * and the nearest long edges are the ones a descent can act on. */
            if (bp.shortcut_cap > 0 && !shortcuts.empty()) {
                std::sort(shortcuts.begin(), shortcuts.end());
                shortcuts.erase(std::unique(shortcuts.begin(), shortcuts.end()),
                                shortcuts.end());
                std::vector<u64> capped;
                capped.reserve(shortcuts.size());
                std::vector<u64> byd;
                size_t a2 = 0;
                while (a2 < shortcuts.size()) {
                    const u32 src = (u32)(shortcuts[a2] >> 32);
                    size_t b2 = a2;
                    byd.clear();
                    while (b2 < shortcuts.size() &&
                           (u32)(shortcuts[b2] >> 32) == src) {
                        const i32 dst = (i32)(u32)shortcuts[b2];
                        byd.push_back(pack_di(fg_dist(X + (i64)src * d,
                                                      X + (i64)dst * d, d,
                                                      metric), dst));
                        b2++;
                    }
                    std::sort(byd.begin(), byd.end());
                    /* keep by OCCLUSION DIVERSITY, not by nearest: these edges
                     * are all long by construction, so "nearest" among them is
                     * arbitrary and collapses direction diversity -- the one
                     * property that makes a long edge reusable by a search
                     * heading elsewhere. Same HSP rule as everywhere else:
                     * nearest-first, a candidate is dropped if a kept
                     * destination is closer to it than the source is. */
                    std::vector<i32> kd;
                    for (u64 pk : byd) {
                        if ((int)kd.size() >= bp.shortcut_cap) break;
                        const i32 dst = unpack_id(pk);
                        f32 dsd = unpack_dist(pk);
                        if (metric == METRIC_L2) dsd = std::sqrt(dsd);
                        bool occ = false;
                        for (i32 g2 : kd) {
                            f32 dg2 = fg_dist(X + (i64)dst * d,
                                              X + (i64)g2 * d, d, metric);
                            if (metric == METRIC_L2) dg2 = std::sqrt(dg2);
                            if (dg2 < dsd) { occ = true; break; }
                        }
                        if (!occ) kd.push_back(dst);
                    }
                    for (i32 dst : kd)
                        capped.push_back(((u64)src << 32) | (u32)dst);
                    a2 = b2;
                }
                n_short = (i64)capped.size();
                shortcuts.swap(capped);
            }
            for (u64 e : shortcuts) E.push_back(e);
            g = csr_from_edges(E, n, threads);
        }
        if (bp.max_degree > 0) {
            /* hard cap AFTER the unpruned unions: keep the nearest max_degree
             * per vertex (build-space distances; deterministic (dist,id) sort).
             * A beam pays a node's full out-degree on landing, so a 640-degree
             * spine hub costs 10x an average node every time it is expanded. */
            #pragma omp parallel num_threads(threads)
            {
                std::vector<u64> byd;
                #pragma omp for schedule(dynamic, 2048)
                for (i64 v = 0; v < n; v++) {
                    const i64 dg = g.deg(v);
                    if (dg <= bp.max_degree) continue;
                    const f32 *xv = X + v * d;
                    byd.clear();
                    for (i64 j = g.ptr[v]; j < g.ptr[v + 1]; j++) {
                        const i32 u = g.idx[j];
                        byd.push_back(pack_di(fg_dist(xv, X + (i64)u * d, d,
                                                      metric), u));
                    }
                    std::sort(byd.begin(), byd.end());
                    i32 *dst = g.idx.data() + g.ptr[v];
                    for (int t2 = 0; t2 < bp.max_degree; t2++)
                        dst[t2] = unpack_id(byd[(size_t)t2]);
                    std::sort(dst, dst + bp.max_degree);
                    /* mark the tail dead; compacted below */
                    for (i64 j = g.ptr[v] + bp.max_degree; j < g.ptr[v + 1]; j++)
                        g.idx[j] = -1;
                }
            }
            /* compact the -1 tails into a fresh CSR (serial scan, cheap) */
            CSR c;
            c.ptr.assign((size_t)n + 1, 0);
            for (i64 v = 0; v < n; v++) {
                i64 dg = g.deg(v);
                c.ptr[v + 1] = c.ptr[v] + std::min<i64>(dg, bp.max_degree);
            }
            c.idx.resize((size_t)c.ptr[n]);
            #pragma omp parallel for schedule(static) num_threads(threads)
            for (i64 v = 0; v < n; v++) {
                const i64 keep = c.ptr[v + 1] - c.ptr[v];
                memcpy(c.idx.data() + c.ptr[v], g.idx.data() + g.ptr[v],
                       (size_t)keep * sizeof(i32));
            }
            g = std::move(c);
        }
        fg.n_dist_build += nd_h;
        fg.n_shortcuts = n_short;
        fg.t_harvest = wall() - t0;
    } else {
        g = std::move(raw);
    }

    fg.g = std::move(g);

    if (bp.verbose) {
        i64 maxdeg = 0; f64 sumdeg = 0;
        for (i64 v = 0; v < n; v++) {
            i64 dg = fg.g.deg(v);
            maxdeg = std::max(maxdeg, dg); sumdeg += (f64)dg;
        }
        const char *pol = bp.policy == POLICY_PROXIMAL ? "proximal"
                        : bp.policy == POLICY_DISTAL   ? "distal" : "random";
        printf("  forest: %d SATs (%s) over n=%lld  trees %.2fs (%d threads), "
               "%.1fM dist\n", T, pol, (long long)n, fg.t_trees, threads,
               (double)fg.n_dist_build / 1e6);
        printf("  overlay+backlinks %.2fs (%lld directed edges, raw max deg %lld)",
               fg.t_overlay, (long long)raw_edges, (long long)raw_maxdeg);
        if (bp.harvest_ef > 0)
            printf("; harvest ef=%d cap=%d rounds=%d", bp.harvest_ef,
                   bp.harvest_cap, bp.harvest_rounds);
        if (bp.harvest_ef > 0 && bp.harvest_patience > 0)
            printf(" patience=%d", bp.harvest_patience);
        if (bp.harvest_ef > 0 && bp.harvest_straighten > 0)
            printf(" straighten=%.2f (%lld shortcuts)",
                   (double)bp.harvest_straighten, (long long)fg.n_shortcuts);
        if (bp.harvest_ef > 0)
            printf(" %.2fs", fg.t_harvest);
        printf("\n  edges: %lld directed (%.1f/vertex, max %lld); memory %.0fMB\n",
               (long long)fg.g.idx.size(), sumdeg / (f64)n, (long long)maxdeg,
               (f64)(fg.g.ptr.size() * 8 + fg.g.idx.size() * 4) / 1e6);
    }
    return fg;
}

/* ===========================================================================
 * Optional: BFS reorder for cache locality at search time. Remaps the graph,
 * the vectors, and the roots so graph-adjacent vertices are memory-adjacent.
 * new2old maps a new id back to the original (for reporting results).
 * =========================================================================== */
void bfs_reorder(ForestGraph &fg, std::vector<f32> &X, i64 n, int d,
                 std::vector<i32> &new2old)
{
    const CSR &g = fg.g;
    /* start from the highest-degree vertex (hub) */
    i64 start = 0, best = -1;
    for (i64 v = 0; v < n; v++)
        if (g.deg(v) > best) { best = g.deg(v); start = v; }

    new2old.assign((size_t)n, -1);
    std::vector<i32> old2new((size_t)n, -1);
    std::vector<i32> queue; queue.reserve((size_t)n);
    i64 head = 0, w = 0;
    queue.push_back((i32)start); old2new[start] = 0; new2old[0] = (i32)start; w = 1;
    while (head < (i64)queue.size()) {
        i32 v = queue[head++];
        for (i64 j = g.ptr[v]; j < g.ptr[v + 1]; j++) {
            i32 u = g.idx[j];
            if (old2new[u] < 0) {
                old2new[u] = (i32)w; new2old[w] = u; w++;
                queue.push_back(u);
            }
        }
    }
    for (i64 v = 0; v < n; v++)                  /* unreached: appended by id */
        if (old2new[v] < 0) { old2new[v] = (i32)w; new2old[w] = (i32)v; w++; }

    CSR ng;
    ng.ptr.assign((size_t)n + 1, 0);
    for (i64 nv = 0; nv < n; nv++) ng.ptr[nv + 1] = ng.ptr[nv] + g.deg(new2old[nv]);
    ng.idx.resize(g.idx.size());
    #pragma omp parallel for schedule(static)
    for (i64 nv = 0; nv < n; nv++) {
        i32 ov = new2old[nv];
        i32 *dst = ng.idx.data() + ng.ptr[nv];
        i64 dg = g.deg(ov);
        const i32 *src = g.idx.data() + g.ptr[ov];
        for (i64 j = 0; j < dg; j++) dst[j] = old2new[src[j]];
        std::sort(dst, dst + dg);
    }
    std::vector<f32> nX((size_t)n * (size_t)d);
    #pragma omp parallel for schedule(static)
    for (i64 nv = 0; nv < n; nv++)
        memcpy(nX.data() + nv * d, X.data() + (i64)new2old[nv] * d,
               (size_t)d * sizeof(f32));
    X.swap(nX);
    for (auto &r : fg.roots) r = old2new[r];
    fg.g = std::move(ng);
}

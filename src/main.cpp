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

/* main.cpp -- harness mirroring run_graph.py.
 *
 *   # synthetic smoke test (clustered or uniform unit vectors, cosine):
 *   ./fg --synthetic clustered --n 100000 --d 64 --nq 1000 \
 *        --T 16 --ef 16 32 64 128 --k 10 --probe --check-determinism
 *
 *   # real benchmark (after convert_annb.py glove-100-angular.hdf5).
 *   # Defaults are the "fast" profile (T16, harvest ef 400); use
 *   # --T 24 --harvest 600 (balanced) or --T 32 --harvest 600 (quality).
 *   ./fg --data X.npy --queries Q.npy --gold gold.npy --metric cos \
 *        --ef 100 200 400 600 --k 10 --threads 10
 */
#include "fg.hpp"
#include "npy.hpp"

static const char *simd_name(void) {
#if FG_SIMD_PATH == 3
    return "AVX-512F";
#elif FG_SIMD_PATH == 2
    return "AVX2+FMA";
#elif FG_SIMD_PATH == 1
    return "NEON";
#else
    return "scalar";
#endif
}

static void normalize_rows(f32 *X, i64 n, int d) {
    #pragma omp parallel for schedule(static)
    for (i64 i = 0; i < n; i++) {
        f32 *r = X + i * d;
        f64 s = 0; for (int j = 0; j < d; j++) s += (f64)r[j] * r[j];
        f32 inv = s > 1e-24 ? (f32)(1.0 / std::sqrt(s)) : 1.0f;
        for (int j = 0; j < d; j++) r[j] *= inv;
    }
}

/* run_graph.make_synth: clustered (nc = max(8, n/250) gaussian clusters,
 * sigma 0.30) or uniform gaussian; unit-normalised; queries disjoint. */
static void make_synth(const char *kind, i64 n, int d, i64 nq, u64 seed,
                       std::vector<f32> &X, std::vector<f32> &Q)
{
    Rng r(mix_seed(seed, 0x53594E5448ull));   /* "SYNTH" */
    X.resize((size_t)n * d); Q.resize((size_t)nq * d);
    if (!strcmp(kind, "clustered")) {
        i64 nc = std::max<i64>(8, n / 250);
        std::vector<f32> C((size_t)nc * d);
        for (auto &v : C) v = (f32)r.normal();
        for (i64 i = 0; i < n; i++) {
            const f32 *c = C.data() + (i64)r.below((u64)nc) * d;
            f32 *x = X.data() + i * d;
            for (int j = 0; j < d; j++) x[j] = c[j] + 0.30f * (f32)r.normal();
        }
        for (i64 i = 0; i < nq; i++) {
            const f32 *c = C.data() + (i64)r.below((u64)nc) * d;
            f32 *x = Q.data() + i * d;
            for (int j = 0; j < d; j++) x[j] = c[j] + 0.30f * (f32)r.normal();
        }
    } else {
        for (auto &v : X) v = (f32)r.normal();
        for (auto &v : Q) v = (f32)r.normal();
    }
    normalize_rows(X.data(), n, d);
    normalize_rows(Q.data(), nq, d);
}

/* exact gold top-k by brute force (parallel over queries) */
static void brute_gold(const f32 *X, i64 n, int d, int metric,
                       const f32 *Q, i64 nq, int K, std::vector<i32> &gold)
{
    gold.assign((size_t)nq * K, -1);
    #pragma omp parallel
    {
        std::vector<u64> heap((size_t)K);
        #pragma omp for schedule(dynamic, 4)
        for (i64 i = 0; i < nq; i++) {
            const f32 *q = Q + i * d;
            int hn = 0;
            for (i64 v = 0; v < n; v++) {
                f32 dd = fg_dist(q, X + v * d, d, metric);
                hmax_offer(heap.data(), &hn, K, pack_di(dd, (i32)v));
            }
            std::sort(heap.begin(), heap.begin() + hn);
            for (int a = 0; a < hn; a++) gold[i * K + a] = unpack_id(heap[a]);
        }
    }
}

static double recall_at(const i32 *pred, const i32 *gold, i64 nq, int k,
                        int gold_stride)
{
    i64 hit = 0;
    for (i64 i = 0; i < nq; i++) {
        const i32 *g = gold + i * gold_stride;
        const i32 *p = pred + i * k;
        for (int a = 0; a < k; a++) {
            i32 pa = p[a];
            if (pa < 0) continue;
            for (int b = 0; b < k; b++)
                if (g[b] == pa) { hit++; break; }
        }
    }
    return (double)hit / (double)(nq * k);
}

/* local completeness: fraction of true k-NN present in the adjacency */
static double local_completeness(const ForestGraph &fg, const f32 *X, i64 n,
                                 int d, int k, i64 sample, u64 seed)
{
    Rng r(mix_seed(seed, 0x4C4F43414Cull));
    std::vector<i32> nodes((size_t)std::min(sample, n));
    for (auto &v : nodes) v = (i32)r.below((u64)n);
    i64 hit = 0, tot = 0;
    #pragma omp parallel reduction(+:hit,tot)
    {
        std::vector<u64> heap((size_t)k);
        #pragma omp for schedule(dynamic, 4)
        for (size_t s = 0; s < nodes.size(); s++) {
            i32 x = nodes[s];
            const f32 *xv = X + (i64)x * d;
            int hn = 0;
            for (i64 v = 0; v < n; v++) {
                if (v == x) continue;
                f32 dd = fg_dist(xv, X + v * d, d, fg.metric);
                hmax_offer(heap.data(), &hn, k, pack_di(dd, (i32)v));
            }
            for (int a = 0; a < hn; a++) {
                i32 t = unpack_id(heap[a]);
                const i32 *nb = fg.g.idx.data() + fg.g.ptr[x];
                i64 dg = fg.g.deg(x);
                bool found = std::binary_search(nb, nb + dg, t);
                if (!found)                       /* adjacency may be unsorted */
                    for (i64 j = 0; j < dg; j++) if (nb[j] == t) { found = true; break; }
                hit += found; tot++;
            }
        }
    }
    return tot ? (double)hit / (double)tot : 0.0;
}

/* Aggregator measurements (a)-(d), on a random DB-point sample with exact
 * true k-NN by brute force. acc = the vote list before any descent; B = the
 * refined code after descent (n*Bk ids, -1 padded), or null for coverage only.
 *   (a) coverage   : fraction of true k-NN present in acc[x] (pre-descent)
 *   (b) recall     : fraction of true k-NN present in B[x]   (post-descent)
 *   (c) seed/desc  : of the recalled true-NN, fraction already in acc (seed)
 *                    vs newly found by descent
 *   (d) zero-recov : of true-NN with ZERO forest votes (absent from acc),
 *                    fraction descent nonetheless recovered into B            */
struct AggMeasure { double coverage, recall, seed_frac, descent_frac, zero_recov;
                    i64 zero_tot; };
static AggMeasure measure_votes(const VoteGraph &acc, const i32 *B, int Bk,
                                const f32 *X, i64 n, int d, int metric,
                                int k, i64 sample, u64 seed)
{
    Rng r(mix_seed(seed, 0x564F5445ull));            /* "VOTE" */
    std::vector<i32> nodes((size_t)std::min(sample, n));
    for (auto &v : nodes) v = (i32)r.below((u64)n);

    i64 cov_hit = 0, cov_tot = 0, rec_hit = 0, rec_tot = 0;
    i64 seed_hit = 0, desc_hit = 0, zero_tot = 0, zero_rec = 0;
    #pragma omp parallel reduction(+:cov_hit,cov_tot,rec_hit,rec_tot,seed_hit,desc_hit,zero_tot,zero_rec)
    {
        std::vector<u64> heap((size_t)k);
        #pragma omp for schedule(dynamic, 4)
        for (size_t s = 0; s < nodes.size(); s++) {
            i32 x = nodes[s];
            const f32 *xv = X + (i64)x * d;
            int hn = 0;
            for (i64 v = 0; v < n; v++) {             /* exact true k-NN of x */
                if (v == x) continue;
                f32 dd = fg_dist(xv, X + v * d, d, metric);
                hmax_offer(heap.data(), &hn, k, pack_di(dd, (i32)v));
            }
            const i32 *av = acc.idx.data() + acc.ptr[x];
            i64 adg = acc.deg(x);
            const i32 *bx = B ? B + (i64)x * Bk : nullptr;
            for (int a = 0; a < hn; a++) {
                i32 t = unpack_id(heap[a]);
                bool in_acc = false;
                for (i64 j = 0; j < adg; j++) if (av[j] == t) { in_acc = true; break; }
                cov_hit += in_acc; cov_tot++;
                if (!bx) continue;
                bool in_b = false;
                for (int j = 0; j < Bk; j++) { if (bx[j] < 0) break; if (bx[j] == t) { in_b = true; break; } }
                rec_hit += in_b; rec_tot++;
                if (in_b) { if (in_acc) seed_hit++; else desc_hit++; }
                if (!in_acc) { zero_tot++; zero_rec += in_b; }
            }
        }
    }
    AggMeasure m;
    m.coverage     = cov_tot ? (double)cov_hit / cov_tot : 0;
    m.recall       = rec_tot ? (double)rec_hit / rec_tot : 0;
    m.seed_frac    = rec_hit ? (double)seed_hit / rec_hit : 0;
    m.descent_frac = rec_hit ? (double)desc_hit / rec_hit : 0;
    m.zero_recov   = zero_tot ? (double)zero_rec / zero_tot : 0;
    m.zero_tot     = zero_tot;
    return m;
}

/* (e) served-code skew = WAND-friendliness. Two sources, per the frontier:
 * in-degree hubness (posting-list length skew, from B ids alone) and the
 * true-NN edge-weight distribution (distances, on a sample). Reports CV
 * (std/mean) and tail ratio p99/p50 for each -- higher = more prunable. */
static void measure_code_skew(const i32 *B, int Bk, const f32 *X, i64 n, int d,
                              int metric, i64 sample, u64 seed)
{
    /* in-degree over the whole code */
    std::vector<i32> indeg((size_t)n, 0);
    for (i64 v = 0; v < n; v++)
        for (int j = 0; j < Bk; j++) { i32 u = B[v * Bk + j]; if (u < 0) break; if (u < n) indeg[u]++; }
    std::vector<i32> id_sorted = indeg;
    std::sort(id_sorted.begin(), id_sorted.end());
    f64 mu = 0; for (i32 v : indeg) mu += v; mu /= (f64)n;
    f64 var = 0; for (i32 v : indeg) var += (v - mu) * (v - mu); var /= (f64)n;
    i32 id_p50 = id_sorted[(size_t)(0.50 * n)];
    i32 id_p99 = id_sorted[(size_t)(0.99 * n)];
    printf("  (e) in-degree skew: mean %.1f  CV %.2f  p50 %d  p99 %d  max %d\n",
           mu, mu > 0 ? std::sqrt(var) / mu : 0.0, id_p50, id_p99,
           id_sorted.back());

    /* edge-weight (true-NN distance) distribution on a sample */
    Rng r(mix_seed(seed, 0x534B4557ull));            /* "SKEW" */
    std::vector<f32> w;
    w.reserve((size_t)std::min(sample, n) * Bk);
    for (i64 s = 0; s < std::min(sample, n); s++) {
        i32 v = (i32)r.below((u64)n);
        const f32 *vv = X + (i64)v * d;
        for (int j = 0; j < Bk; j++) {
            i32 u = B[v * Bk + j]; if (u < 0) break;
            f32 dd = fg_dist(vv, X + (i64)u * d, d, metric);
            if (metric == METRIC_L2) dd = std::sqrt(dd);
            w.push_back(dd);
        }
    }
    if (!w.empty()) {
        std::sort(w.begin(), w.end());
        f64 wm = 0; for (f32 x : w) wm += x; wm /= (f64)w.size();
        f64 wv = 0; for (f32 x : w) wv += (x - wm) * (x - wm); wv /= (f64)w.size();
        f32 w50 = w[w.size() / 2], w90 = w[(size_t)(0.90 * w.size())],
            w99 = w[(size_t)(0.99 * w.size())];
        printf("      edge-weight: mean %.4f  CV %.2f  p50 %.4f  p90 %.4f  p99 %.4f\n",
               wm, wm > 0 ? std::sqrt(wv) / wm : 0.0, w50, w90, w99);
    }
}

static int parse_policy(const char *s) {
    if (!strcmp(s, "proximal")) return POLICY_PROXIMAL;
    if (!strcmp(s, "distal"))   return POLICY_DISTAL;
    if (!strcmp(s, "random"))   return POLICY_RANDOM;
    fprintf(stderr, "[FATAL] unknown policy '%s'\n", s); exit(2);
}

int main(int argc, char **argv) {
    const char *synthetic = nullptr, *data = nullptr, *queries = nullptr,
               *goldf = nullptr, *entry = "hub";
    const char *dumpg = nullptr;
    i64 n = 100000, nq = 1000;
    int d = 64, k = 10, n_entry = 1, threads = 0, search_threads = 0;
    int metric = METRIC_COSINE;
    bool probe = false, checkdet = false, reorder = false;
    int  sketch_d = 0;                 /* JL sketch width for the BUILD; 0=off */
    bool aggregate = false, scaling = false, system_recall = false;
    int  agg_A = 1, agg_cap = 64, agg_rounds = 0, agg_k = 32;
    f32  agg_decay = 1.0f;
    i64  agg_sample = 800;
    std::vector<int> round_ckpts;
    BuildParams bp;
    std::vector<int> efs;

    for (int a = 1; a < argc; a++) {
        std::string s = argv[a];
        auto next = [&]() { if (++a >= argc) { fprintf(stderr, "[FATAL] %s needs a value\n", s.c_str()); exit(2); } return argv[a]; };
        if      (s == "--synthetic") synthetic = next();
        else if (s == "--n")  n  = atoll(next());
        else if (s == "--d")  d  = atoi(next());
        else if (s == "--nq") nq = atoll(next());
        else if (s == "--data")    data    = next();
        else if (s == "--queries") queries = next();
        else if (s == "--gold")    goldf   = next();
        else if (s == "--metric")  metric  = strcmp(next(), "l2") ? METRIC_COSINE : METRIC_L2;
        else if (s == "--T")       bp.T = atoi(next());
        else if (s == "--policy")  bp.policy = parse_policy(next());
        else if (s == "--harvest")       bp.harvest_ef = atoi(next());
        else if (s == "--harvest-cap")   bp.harvest_cap = atoi(next());
        else if (s == "--harvest-alpha") bp.harvest_alpha = (f32)atof(next());
        else if (s == "--harvest-cand")  bp.harvest_cand = atoi(next());
        else if (s == "--harvest-rounds") bp.harvest_rounds = atoi(next());
        else if (s == "--harvest-patience") bp.harvest_patience = atoi(next());
        else if (s == "--tree-depth")    bp.tree_depth = atoi(next());
        else if (s == "--straighten")    bp.harvest_straighten = (f32)atof(next());
        else if (s == "--sketch")        sketch_d = atoi(next());
        else if (s == "--substrate-random") bp.substrate_random = atoi(next());
        else if (s == "--max-degree")    bp.max_degree = atoi(next());
        else if (s == "--shortcut-cap")  bp.shortcut_cap = atoi(next());
        else if (s == "--wild-directed") bp.overlay_directed = 1;
        else if (s == "--seed")    bp.seed = (u64)atoll(next());
        else if (s == "--threads") { threads = atoi(next()); bp.threads = threads; }
        else if (s == "--search-threads") search_threads = atoi(next());
                /* serve-side thread count, decoupled from the build (the graph
                 * is thread-count-invariant, so QPS cells stay comparable) */
        else if (s == "--build-conc") bp.build_conc = atoi(next());
        else if (s == "--ef") { while (a + 1 < argc && argv[a + 1][0] != '-') efs.push_back(atoi(argv[++a])); }
        else if (s == "--k")       k = atoi(next());
        else if (s == "--entry")   entry = next();
        else if (s == "--n-entry") n_entry = atoi(next());
        else if (s == "--probe")   probe = true;
        else if (s == "--check-determinism") checkdet = true;
        else if (s == "--reorder") reorder = true;
        else if (s == "--dump-graph") dumpg = next();
        else if (s == "--aggregate")  aggregate = true;
        else if (s == "--A")          agg_A = atoi(next());
        else if (s == "--vote-cap")   agg_cap = atoi(next());
        else if (s == "--decay")      agg_decay = (f32)atof(next());
        else if (s == "--agg-rounds") agg_rounds = atoi(next());
        else if (s == "--agg-k")      agg_k = atoi(next());
        else if (s == "--agg-sample") agg_sample = atoll(next());
        else if (s == "--scaling")    scaling = true;
        else if (s == "--system-recall") system_recall = true;
        else if (s == "--round-checkpoints") {
            char *cs = next();
            for (char *tok = strtok(cs, ","); tok; tok = strtok(nullptr, ","))
                round_ckpts.push_back(atoi(tok));
        }
        else { fprintf(stderr, "[FATAL] unknown arg %s\n", argv[a]); exit(2); }
    }
    if (efs.empty()) efs = {16, 32, 64, 128};
    if (threads > 0) omp_set_num_threads(threads);
    printf("fg: %s kernels, %d hw threads available\n",
           simd_name(), omp_get_max_threads());

    std::vector<f32> X, Q;
    std::vector<i32> gold;
    int gold_k = 0;

    if (data) {
        NpyArray ax = npy_load(data), aq = npy_load(queries);
        if (ax.descr != "<f4" || aq.descr != "<f4") {
            fprintf(stderr, "[FATAL] data/queries must be float32\n"); exit(2);
        }
        n = ax.rows; d = (int)ax.cols; nq = aq.rows;
        if ((int)aq.cols != d) { fprintf(stderr, "[FATAL] dim mismatch\n"); exit(2); }
        X.assign(ax.f32p(), ax.f32p() + (size_t)n * d);
        Q.assign(aq.f32p(), aq.f32p() + (size_t)nq * d);
        if (metric == METRIC_COSINE) {
            normalize_rows(X.data(), n, d);
            normalize_rows(Q.data(), nq, d);
        }
        NpyArray ag = npy_load(goldf);
        gold_k = (int)ag.cols;
        gold.resize((size_t)nq * gold_k);
        if (ag.descr == "<i4")
            memcpy(gold.data(), ag.i32p(), gold.size() * 4);
        else
            for (size_t i = 0; i < gold.size(); i++) gold[i] = (i32)ag.i64p()[i];
        printf("=== %s : N=%lld d=%d queries=%lld (%s) ===\n", data,
               (long long)n, d, (long long)nq,
               metric == METRIC_COSINE ? "cosine" : "L2");
    } else {
        if (!synthetic) synthetic = "clustered";
        make_synth(synthetic, n, d, nq, bp.seed, X, Q);
        printf("=== synthetic %s: N=%lld d=%d queries=%lld (cosine) ===\n",
               synthetic, (long long)n, d, (long long)nq);
        gold_k = std::min<i64>(100, n);
        double t0 = wall();
        brute_gold(X.data(), n, d, metric, Q.data(), nq, gold_k, gold);
        printf("  exact gold top-%d: %.1fs\n", gold_k, wall() - t0);
    }
    if (gold_k < k) { fprintf(stderr, "[FATAL] gold has only %d columns < k\n", gold_k); exit(2); }

    /* pad rows to a whole number of cache lines (16-float multiple): full-width
     * SIMD with no scalar tail, uniform prefetch geometry. Zero padding leaves
     * every distance value unchanged. */
    {
        int dp = (d + 15) / 16 * 16;
        if (dp != d) {
            std::vector<f32> Xp((size_t)n * dp, 0.0f), Qp((size_t)nq * dp, 0.0f);
            for (i64 i = 0; i < n; i++)
                memcpy(Xp.data() + i * dp, X.data() + (i64)i * d, (size_t)d * 4);
            for (i64 i = 0; i < nq; i++)
                memcpy(Qp.data() + i * dp, Q.data() + (i64)i * d, (size_t)d * 4);
            X.swap(Xp); Q.swap(Qp);
            printf("  rows padded d=%d -> %d (whole cache lines)\n", d, dp);
            d = dp;
        }
    }

    /* ---- JL sketch for the BUILD (EC: high ambient dimension is not a
     * fundamental barrier -- it is a per-distance COST, and the build does not
     * need exact distances, only approximately correct nearest-neighbour
     * ORDER). Project the base onto sketch_d seeded Gaussian directions
     * (splitmix-keyed, so the sketch -- and hence the graph -- is
     * deterministic) and run trees + harvest + prune entirely in the sketch.
     * The QUERY search below still runs on the TRUE vectors, so recall is
     * measured end-to-end against exact geometry: only edge SELECTION is
     * approximated, the served graph is judged honestly. Cuts build cost by
     * ~d/sketch_d (gist: 960 -> 192 = 5x). Same trick as PiPNN's HashPrune
     * sketches; imported here for the whole construction. */
    std::vector<f32> Xsk;
    int d_build = d;
    if (sketch_d > 0 && sketch_d < d) {
        double t0 = wall();
        int dp = (sketch_d + 15) / 16 * 16;      /* pad the sketch too */
        std::vector<f32> G((size_t)d * dp, 0.0f);
        {
            Rng r(mix_seed(bp.seed, 0x4A4Cull));  /* "JL" */
            const f32 inv = 1.0f / std::sqrt((f32)sketch_d);
            for (i64 i = 0; i < (i64)d * sketch_d; i++)
                G[(i / sketch_d) * dp + (i % sketch_d)] = (f32)r.normal() * inv;
        }
        Xsk.assign((size_t)n * dp, 0.0f);
        #pragma omp parallel for schedule(static)
        for (i64 i = 0; i < n; i++) {
            const f32 *xi = X.data() + i * d;
            f32 *o = Xsk.data() + i * dp;
            for (int a = 0; a < d; a++) {
                const f32 v = xi[a];
                if (v == 0.0f) continue;
                const f32 *g = G.data() + (i64)a * dp;
                for (int b = 0; b < dp; b++) o[b] += v * g[b];
            }
        }
        if (metric == METRIC_COSINE) normalize_rows(Xsk.data(), n, dp);
        d_build = dp;
        printf("  JL sketch for build: d=%d -> %d (%.1fs)\n", d, dp, wall() - t0);
    }
    const f32 *Xb = Xsk.empty() ? X.data() : Xsk.data();
    const int  db = Xsk.empty() ? d : d_build;

    /* ---- vote-counting aggregator path (additive recovery tier) ---- *
     * Shares phase 1 with the navigation build (same trees), counts mentions,
     * measures coverage pre-descent and recall/contribution post-descent. */
    if (aggregate) {
        if (agg_k < k) agg_k = k;

        /* (f) construction parallelism: build phase 1 at threads=1 and max,
         * print the speedup (the four-way-advantage claim, made concrete). */
        if (scaling) {
            BuildParams b1 = bp; b1.threads = 1;
            BuildParams bm = bp; bm.threads = std::max(2, omp_get_max_threads());
            i64 nd1 = 0, ndm = 0; double s1 = 0, sm = 0;
            std::vector<std::vector<i32>> p1, pm; std::vector<i32> r1, rm;
            build_parents(X.data(), n, d, metric, b1, p1, r1, &nd1, &s1);
            build_parents(X.data(), n, d, metric, bm, pm, rm, &ndm, &sm);
            printf("  (f) forest build scaling: 1thr %.2fs -> %dthr %.2fs = %.1fx\n",
                   s1, bm.threads, sm, sm > 0 ? s1 / sm : 0.0);
        }

        i64 nd_build = 0; double t_trees = 0;
        std::vector<std::vector<i32>> parents; std::vector<i32> roots;
        build_parents(X.data(), n, d, metric, bp, parents, roots, &nd_build, &t_trees);
        printf("  forest: %d SATs over n=%lld  trees %.2fs (%d threads), %.1fM dist\n",
               bp.T, (long long)n, t_trees,
               bp.threads > 0 ? bp.threads : omp_get_max_threads(),
               (double)nd_build / 1e6);

        VoteGraph acc = aggregate_votes(parents, roots, n, agg_A, agg_decay,
                                        agg_cap, bp.threads, true);
        parents.clear(); parents.shrink_to_fit();

        i64 smp = std::min(agg_sample, n);
        AggMeasure cov = measure_votes(acc, nullptr, 0, X.data(), n, d, metric,
                                       k, smp, bp.seed);
        printf("  (a) vote-coverage of true k-NN (pre-descent): %.4f\n", cov.coverage);

        /* SYSTEM-level recall: serve the recovery code as a searchable graph
         * (candidate gen + exact re-rank = beam) vs the QUERY set, at each --ef
         * candidate depth. dist/query == the candidate depth (exact evals). */
        auto run_system_recall = [&](const std::vector<i32> &B) {
            if (!system_recall) return;
            ForestGraph cg = code_to_graph(B.data(), agg_k, n, metric);
            std::vector<i32> ent;
            make_entries(cg, X.data(), n, d, Q.data(), nq, "hub", 1, bp.seed, ent);
            printf("  system recall (recovery code beam-searched, exact re-rank vs queries):\n");
            printf("  %7s %12s %11s\n", "ef", "depth(d/q)", "recall@10");
            for (int ef : efs) {
                SearchResult r = beam_search(cg, X.data(), n, d, Q.data(), nq,
                                             ent.data(), 1, ef, k, threads);
                double rec = recall_at(r.ids.data(), gold.data(), nq, k, gold_k);
                printf("  %7d %12.0f %11.4f\n", ef, (double)r.n_dist / (double)nq, rec);
                printf("SYS T=%d cap=%d aggk=%d ef=%d depth=%.0f recall=%.4f\n",
                       bp.T, agg_cap, agg_k, ef, (double)r.n_dist / (double)nq, rec);
            }
        };

        if (!round_ckpts.empty()) {
            /* stepped descent: seed once, measure recall at each checkpoint round
             * from ONE forest build (the asymptote sweep) */
            int maxr = 0; for (int r : round_ckpts) maxr = std::max(maxr, r);
            std::vector<i32> B((size_t)n * agg_k, -1), B2((size_t)n * agg_k, -1);
            double t0 = wall(); i64 nd_ref = 0;
            vote_seed(acc, X.data(), n, d, metric, agg_k, B.data(), bp.threads, &nd_ref);
            for (int r = 1; r <= maxr; r++) {
                vote_descent_round(B.data(), B2.data(), X.data(), n, d, metric,
                                   agg_k, bp.threads, &nd_ref);
                B.swap(B2);
                if (std::find(round_ckpts.begin(), round_ckpts.end(), r) == round_ckpts.end())
                    continue;
                AggMeasure m = measure_votes(acc, B.data(), agg_k, X.data(), n, d,
                                             metric, k, smp, bp.seed);
                printf("  rounds=%d  recall@%d=%.4f  seed=%.4f descent=%.4f zerorec=%.4f  (%.1fs)\n",
                       r, k, m.recall, m.seed_frac, m.descent_frac, m.zero_recov, wall() - t0);
                printf("AGG A=%d T=%d cap=%d aggk=%d decay=%.2f rounds=%d cov=%.4f recall=%.4f "
                       "seed=%.4f descent=%.4f zerorec=%.4f\n", agg_A, bp.T, agg_cap, agg_k,
                       (double)agg_decay, r, cov.coverage, m.recall,
                       m.seed_frac, m.descent_frac, m.zero_recov);
            }
            measure_code_skew(B.data(), agg_k, X.data(), n, d, metric, smp, bp.seed);
            run_system_recall(B);
        } else if (agg_rounds > 0) {
            double t0 = wall(); i64 nd_ref = 0;
            std::vector<i32> B;
            refine_from_votes(acc, X.data(), n, d, metric, agg_k, agg_rounds,
                              bp.threads, B, &nd_ref);
            printf("  refine: %d rounds k=%d  %.2fs  %.1fM dist\n",
                   agg_rounds, agg_k, wall() - t0, (double)nd_ref / 1e6);
            AggMeasure m = measure_votes(acc, B.data(), agg_k, X.data(), n, d,
                                         metric, k, smp, bp.seed);
            printf("  (b) recall@%d after descent:            %.4f\n", k, m.recall);
            printf("  (c) of recall: seed %.4f / descent %.4f\n", m.seed_frac, m.descent_frac);
            printf("  (d) zero-vote true-NN recovered:        %.4f  (zero-vote fraction %.4f)\n",
                   m.zero_recov, cov.coverage < 1.0 ? 1.0 - cov.coverage : 0.0);
            measure_code_skew(B.data(), agg_k, X.data(), n, d, metric, smp, bp.seed);
            printf("AGG A=%d T=%d cap=%d decay=%.2f rounds=%d cov=%.4f recall=%.4f "
                   "seed=%.4f descent=%.4f zerorec=%.4f\n", agg_A, bp.T, agg_cap,
                   (double)agg_decay, agg_rounds, cov.coverage, m.recall,
                   m.seed_frac, m.descent_frac, m.zero_recov);
            run_system_recall(B);
        } else {
            printf("AGG A=%d T=%d cap=%d decay=%.2f rounds=0 cov=%.4f\n",
                   agg_A, bp.T, agg_cap, (double)agg_decay, cov.coverage);
        }
        return 0;
    }

    /* ---- determinism: rebuild at thread counts 1 and max, compare hashes ---- */
    if (checkdet) {
        BuildParams b1 = bp; b1.threads = 1; b1.verbose = false;
        BuildParams b2 = bp; b2.threads = std::max(4, omp_get_max_threads()); b2.verbose = false;
        ForestGraph g1 = build_forest_graph(Xb, n, db, metric, b1);
        ForestGraph g2 = build_forest_graph(Xb, n, db, metric, b2);
        u64 h1 = csr_hash(g1.g), h2 = csr_hash(g2.g);
        printf("  determinism: hash(threads=1)=%016llx hash(threads=%d)=%016llx -> %s\n",
               (unsigned long long)h1, b2.threads, (unsigned long long)h2,
               h1 == h2 ? "IDENTICAL" : "MISMATCH (BUG)");
    }

    ForestGraph fg = build_forest_graph(Xb, n, db, metric, bp);
    Xsk.clear(); Xsk.shrink_to_fit();   /* build-only; search uses TRUE vectors */

    /* Optional CSR export in the ParlayANN graph layout (u32 n, u32 max_degree,
     * u32 sizes[n], then the flattened adjacency) so one reader can load this
     * graph and a Vamana/HCNNG/PiPNN graph for side-by-side graph analysis.
     * Read-only: does not touch the build, so the determinism hash is unchanged.
     * Written before any --reorder so ids stay in input order. */
    if (dumpg) {
        FILE *gf = fopen(dumpg, "wb");
        if (!gf) { fprintf(stderr, "[FATAL] cannot write %s\n", dumpg); exit(2); }
        u32 nn = (u32)n, maxdeg = 0;
        std::vector<u32> sizes((size_t)n);
        for (i64 v = 0; v < n; v++) {
            sizes[(size_t)v] = (u32)fg.g.deg(v);
            if (sizes[(size_t)v] > maxdeg) maxdeg = sizes[(size_t)v];
        }
        fwrite(&nn, 4, 1, gf);
        fwrite(&maxdeg, 4, 1, gf);
        fwrite(sizes.data(), 4, (size_t)n, gf);
        std::vector<u32> row;
        for (i64 v = 0; v < n; v++) {
            row.assign(fg.g.idx.begin() + fg.g.ptr[v],
                       fg.g.idx.begin() + fg.g.ptr[v + 1]);
            if (!row.empty()) fwrite(row.data(), 4, row.size(), gf);
        }
        fclose(gf);
        printf("  dumped graph -> %s (n=%u max degree=%u)\n", dumpg, nn, maxdeg);
    }

    std::vector<i32> new2old;
    if (reorder) {
        double t0 = wall();
        bfs_reorder(fg, X, n, d, new2old);
        printf("  bfs reorder: %.2fs\n", wall() - t0);
    }

    if (search_threads > 0) threads = search_threads;   /* serve-side only */
    const f32 *Xs = huge_clone(X);        /* 2MB-page backing for search */
    double lc = local_completeness(fg, X.data(), n, d, k,
                                   std::min<i64>(800, n), bp.seed);
    printf("  local completeness (k=%d, sample): %.3f\n", k, lc);

    const bool besthub = !strcmp(entry, "besthub");
    std::vector<i32> ent;
    if (!besthub)
        make_entries(fg, Xs, n, d, Q.data(), nq, entry, n_entry, 1, ent);

    auto report_ids = [&](std::vector<i32> &ids) {           /* undo reorder */
        if (!new2old.empty())
            for (auto &v : ids) if (v >= 0) v = new2old[v];
    };

    printf("\n  %5s%11s%12s%12s\n", "ef", "recall@10", "dist/query",
           threads > 1 ? "QPS(mt)" : "QPS(1thr)");
    for (int ef : efs) {
        SearchResult r = beam_search(fg, Xs, n, d, Q.data(), nq,
                                     besthub ? nullptr : ent.data(),
                                     besthub ? 0 : n_entry, ef, k, threads);
        report_ids(r.ids);
        double rec = recall_at(r.ids.data(), gold.data(), nq, k, gold_k);
        printf("  %5d%11.4f%12.0f%12.0f\n", ef, rec,
               (double)r.n_dist / (double)nq,
               (double)nq / std::max(r.seconds, 1e-9));
    }

    if (probe) {
        printf("\n  navigability probe (hub vs far entry):\n");
        for (int ef : {efs.front(), efs.back()}) {
            std::vector<i32> eh, ef_;
            make_entries(fg, Xs, n, d, Q.data(), nq, "hub", 1, 1, eh);
            make_entries(fg, Xs, n, d, Q.data(), nq, "far", 1, 1, ef_);
            SearchResult rh = beam_search(fg, Xs, n, d, Q.data(), nq,
                                          eh.data(), 1, ef, k, threads);
            SearchResult rf = beam_search(fg, Xs, n, d, Q.data(), nq,
                                          ef_.data(), 1, ef, k, threads);
            report_ids(rh.ids); report_ids(rf.ids);
            double a = recall_at(rh.ids.data(), gold.data(), nq, k, gold_k);
            double b = recall_at(rf.ids.data(), gold.data(), nq, k, gold_k);
            printf("    ef=%d: hub=%.3f  far=%.3f  gap=%+.3f\n", ef, a, b, a - b);
        }
    }
    return 0;
}

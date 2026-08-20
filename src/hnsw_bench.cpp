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

/* hnsw_bench.cpp -- hnswlib on the same data/gold/hardware as fg, with FULL
 * distance-evaluation counts (stock searchKnn only counts the upper-layer
 * descent; this harness calls searchBaseLayerST<true, true> so layer-0 search
 * is counted too). Output format matches fg's table for direct matched-recall
 * comparison.
 *
 *   g++ -O3 -march=native -fopenmp -std=c++17 -I /path/to/hnswlib \
 *       hnsw_bench.cpp -o hnsw_bench
 *   ./hnsw_bench --data glove_X.npy --queries glove_Q.npy --gold glove_gold.npy \
 *       --metric cos --M 32 --efc 200 --ef 50 100 200 400 --threads 8 \
 *       --save glove_m32.hnsw          # reuse with --load on later sweeps
 */
#include "fg.hpp"
#include "npy.hpp"
#include "hnswlib/hnswlib.h"

static std::vector<std::pair<f32, i64>>
search_with_metrics(hnswlib::HierarchicalNSW<float> *alg, const void *q, size_t k)
{
    using namespace hnswlib;
    tableint curr = alg->enterpoint_node_;
    float curd = alg->fstdistfunc_(q, alg->getDataByInternalId(curr),
                                   alg->dist_func_param_);
    alg->metric_distance_computations++;
    for (int level = alg->maxlevel_; level > 0; level--) {
        bool changed = true;
        while (changed) {
            changed = false;
            unsigned int *data = (unsigned int *)alg->get_linklist(curr, level);
            int size = alg->getListCount(data);
            alg->metric_hops++;
            alg->metric_distance_computations += size;
            tableint *datal = (tableint *)(data + 1);
            for (int i = 0; i < size; i++) {
                tableint cand = datal[i];
                float dd = alg->fstdistfunc_(q, alg->getDataByInternalId(cand),
                                             alg->dist_func_param_);
                if (dd < curd) { curd = dd; curr = cand; changed = true; }
            }
        }
    }
    auto top = alg->searchBaseLayerST<true, true>(curr, q,
                   std::max(alg->ef_, k), nullptr);   /* metrics-enabled */
    while (top.size() > k) top.pop();
    std::vector<std::pair<f32, i64>> out;
    out.reserve(top.size());
    while (!top.empty()) {
        out.emplace_back(top.top().first,
                         (i64)alg->getExternalLabel(top.top().second));
        top.pop();
    }
    return out;   /* worst-first; only membership matters for recall */
}

int main(int argc, char **argv) {
    const char *data = nullptr, *queries = nullptr, *goldf = nullptr;
    const char *save = nullptr, *load = nullptr;
    bool cosine = true;
    int M = 32, efc = 200, k = 10, threads = 0;
    std::vector<size_t> efs;
    for (int a = 1; a < argc; a++) {
        std::string s = argv[a];
        auto next = [&]() { if (++a >= argc) { fprintf(stderr, "[FATAL] %s needs a value\n", s.c_str()); exit(2); } return argv[a]; };
        if      (s == "--data")    data = next();
        else if (s == "--queries") queries = next();
        else if (s == "--gold")    goldf = next();
        else if (s == "--metric")  cosine = strcmp(next(), "l2") != 0;
        else if (s == "--M")       M = atoi(next());
        else if (s == "--efc")     efc = atoi(next());
        else if (s == "--k")       k = atoi(next());
        else if (s == "--threads") threads = atoi(next());
        else if (s == "--save")    save = next();
        else if (s == "--load")    load = next();
        else if (s == "--ef") { while (a + 1 < argc && argv[a + 1][0] != '-') efs.push_back((size_t)atoll(argv[++a])); }
        else { fprintf(stderr, "[FATAL] unknown arg %s\n", argv[a]); exit(2); }
    }
    if (!data || !queries || !goldf) { fprintf(stderr, "[FATAL] need --data --queries --gold\n"); exit(2); }
    if (efs.empty()) efs = {50, 100, 200, 400};
    if (threads <= 0) threads = omp_get_max_threads();
    omp_set_num_threads(threads);

    NpyArray ax = npy_load(data), aq = npy_load(queries), ag = npy_load(goldf);
    i64 n = ax.rows, nq = aq.rows;
    int d = (int)ax.cols, gold_k = (int)ag.cols;
    std::vector<f32> X(ax.f32p(), ax.f32p() + (size_t)n * d);
    std::vector<f32> Q(aq.f32p(), aq.f32p() + (size_t)nq * d);
    std::vector<i32> gold((size_t)nq * gold_k);
    if (ag.descr == "<i4") memcpy(gold.data(), ag.i32p(), gold.size() * 4);
    else for (size_t i = 0; i < gold.size(); i++) gold[i] = (i32)ag.i64p()[i];

    if (cosine) {   /* normalise so InnerProductSpace == 1 - cos */
        #pragma omp parallel for schedule(static)
        for (i64 i = 0; i < n; i++) {
            f32 *r = X.data() + i * d; f64 s2 = 0;
            for (int j = 0; j < d; j++) s2 += (f64)r[j] * r[j];
            f32 inv = s2 > 1e-24 ? (f32)(1.0 / std::sqrt(s2)) : 1.0f;
            for (int j = 0; j < d; j++) r[j] *= inv;
        }
        for (i64 i = 0; i < nq; i++) {
            f32 *r = Q.data() + i * d; f64 s2 = 0;
            for (int j = 0; j < d; j++) s2 += (f64)r[j] * r[j];
            f32 inv = s2 > 1e-24 ? (f32)(1.0 / std::sqrt(s2)) : 1.0f;
            for (int j = 0; j < d; j++) r[j] *= inv;
        }
    }

    hnswlib::SpaceInterface<float> *space;
    if (cosine) space = new hnswlib::InnerProductSpace((size_t)d);
    else        space = new hnswlib::L2Space((size_t)d);

    hnswlib::HierarchicalNSW<float> *alg;
    if (load) {
        alg = new hnswlib::HierarchicalNSW<float>(space, std::string(load));
        printf("hnswlib: loaded %s (n=%zu, M=%zu)\n", load,
               alg->cur_element_count.load(), alg->M_);
    } else {
        alg = new hnswlib::HierarchicalNSW<float>(space, (size_t)n, (size_t)M,
                                                  (size_t)efc, 100);
        double t0 = wall();
        alg->addPoint(X.data(), 0);
        #pragma omp parallel for schedule(dynamic, 256)
        for (i64 i = 1; i < n; i++)
            alg->addPoint(X.data() + i * d, (size_t)i);
        printf("hnswlib: build n=%lld d=%d M=%d efC=%d  %.1fs (%d threads)\n",
               (long long)n, d, M, efc, wall() - t0, threads);
        if (save) { alg->saveIndex(save); printf("hnswlib: saved %s\n", save); }
    }

    printf("\n  %5s%11s%12s%12s%9s\n", "ef", "recall@10", "dist/query", "QPS", "hops");
    for (size_t ef : efs) {
        alg->setEf(ef);
        alg->metric_distance_computations = 0;
        alg->metric_hops = 0;
        std::vector<i32> pred((size_t)nq * k, -1);
        double t0 = wall();
        #pragma omp parallel for schedule(dynamic, 8)
        for (i64 i = 0; i < nq; i++) {
            auto res = search_with_metrics(alg, Q.data() + i * d, (size_t)k);
            int w = 0;
            for (auto it = res.rbegin(); it != res.rend() && w < k; ++it)
                pred[i * k + w++] = (i32)it->second;
        }
        double t = wall() - t0;
        i64 hit = 0;
        for (i64 i = 0; i < nq; i++)
            for (int a2 = 0; a2 < k; a2++) {
                i32 p = pred[i * k + a2];
                if (p < 0) continue;
                for (int b = 0; b < k; b++)
                    if (gold[i * gold_k + b] == p) { hit++; break; }
            }
        printf("  %5zu%11.4f%12.0f%12.0f%9.0f\n", ef,
               (double)hit / (double)(nq * k),
               (double)alg->metric_distance_computations.load() / (double)nq,
               (double)nq / std::max(t, 1e-9),
               (double)alg->metric_hops.load() / (double)nq);
    }
    return 0;
}

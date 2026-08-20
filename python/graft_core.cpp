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

/* graft_core.cpp -- pybind11 binding of the GRAFT core (src/fg.hpp).
 *
 * Exposes:
 *   _core.build(X, metric, ...) -> Index     (GIL released during the build)
 *   Index.search(Q, k, ef, threads)          (GIL released during the search)
 *   Index.graph_hash / degree stats / phase timings
 *
 * The binding owns a copy of the vectors (normalized when metric is cosine,
 * matching the CLI harness), so the caller's array can be garbage-collected.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "fg.hpp"

namespace py = pybind11;

using farr = py::array_t<f32, py::array::c_style | py::array::forcecast>;

static void normalize_rows_py(f32 *X, i64 n, int d) {
    for (i64 i = 0; i < n; i++) {
        f32 *x = X + (size_t)i * d;
        f64 s = 0;
        for (int j = 0; j < d; j++) s += (f64)x[j] * x[j];
        f32 inv = s > 0 ? (f32)(1.0 / std::sqrt(s)) : 0.0f;
        for (int j = 0; j < d; j++) x[j] *= inv;
    }
}

static int parse_metric(const std::string &m) {
    if (m == "l2") return METRIC_L2;
    if (m == "cosine" || m == "cos" || m == "angular") return METRIC_COSINE;
    throw std::invalid_argument("metric must be 'l2' or 'cosine', got '" + m + "'");
}

struct PyIndex {
    ForestGraph      fg;
    std::vector<f32> X;        /* owned copy, normalized if cosine */
    i64              n = 0;
    int              d = 0;
    int              metric = METRIC_L2;

    std::string graph_hash() const {
        char buf[17];
        snprintf(buf, sizeof buf, "%016llx", (unsigned long long)csr_hash(fg.g));
        return std::string(buf);
    }
};

static PyIndex *build_index(farr Xarr, const std::string &metric_s,
                            int T, int harvest, int harvest_cap, f32 alpha,
                            int patience, int max_degree, u64 seed,
                            int threads, int build_conc, bool verbose) {
    if (Xarr.ndim() != 2)
        throw std::invalid_argument("X must be a 2-d array [n, d] of float32");
    const i64 n = (i64)Xarr.shape(0);
    const int d = (int)Xarr.shape(1);
    if (n < 2) throw std::invalid_argument("need at least 2 points");

    auto idx = std::make_unique<PyIndex>();
    idx->n = n;
    idx->d = d;
    idx->metric = parse_metric(metric_s);
    idx->X.assign(Xarr.data(), Xarr.data() + (size_t)n * d);

    BuildParams bp;
    bp.T = T;
    bp.harvest_ef = harvest;
    bp.harvest_cap = harvest_cap;
    bp.harvest_alpha = alpha;
    bp.harvest_patience = patience;
    bp.max_degree = max_degree;
    bp.seed = seed;
    bp.threads = threads;
    bp.build_conc = build_conc;
    bp.verbose = verbose;

    {
        py::gil_scoped_release release;
        if (idx->metric == METRIC_COSINE)
            normalize_rows_py(idx->X.data(), n, d);
        idx->fg = build_forest_graph(idx->X.data(), n, d, idx->metric, bp);
    }
    return idx.release();
}

static py::tuple search_index(PyIndex &idx, farr Qarr, int k, int ef,
                              int threads) {
    bool single = (Qarr.ndim() == 1);
    if (Qarr.ndim() > 2)
        throw std::invalid_argument("Q must be [d] or [nq, d] float32");
    const i64 nq = single ? 1 : (i64)Qarr.shape(0);
    const int qd = single ? (int)Qarr.shape(0) : (int)Qarr.shape(1);
    if (qd != idx.d)
        throw std::invalid_argument("query dimension " + std::to_string(qd) +
                                    " != index dimension " + std::to_string(idx.d));
    if (k < 1) throw std::invalid_argument("k must be >= 1");
    if (ef < k) ef = k;

    std::vector<f32> Q(Qarr.data(), Qarr.data() + (size_t)nq * qd);
    SearchResult r;
    {
        py::gil_scoped_release release;
        if (idx.metric == METRIC_COSINE) normalize_rows_py(Q.data(), nq, qd);
        /* hub entry, one per query, exactly as the CLI harness serves */
        std::vector<i32> entries;
        make_entries(idx.fg, idx.X.data(), idx.n, idx.d, Q.data(), nq, "hub",
                     1, 1, entries);
        r = beam_search(idx.fg, idx.X.data(), idx.n, idx.d, Q.data(), nq,
                        entries.data(), 1, ef, k, threads);
    }
    py::array_t<i32> ids({nq, (i64)k});
    py::array_t<f32> dist({nq, (i64)k});
    std::copy(r.ids.begin(), r.ids.end(), ids.mutable_data());
    std::copy(r.dist.begin(), r.dist.end(), dist.mutable_data());
    return py::make_tuple(ids, dist);
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "GRAFT core binding (deterministic parallel navigable-graph build)";

    py::class_<PyIndex>(m, "Index")
        .def("search", &search_index, py::arg("Q"), py::arg("k") = 10,
             py::arg("ef") = 64, py::arg("threads") = 0,
             "Beam-search the graph. Returns (ids [nq,k] int32, dist [nq,k] "
             "float32). Distances are squared L2 for metric 'l2' and 1-dot "
             "for 'cosine' (hnswlib conventions).")
        .def_property_readonly("graph_hash", &PyIndex::graph_hash,
             "FNV hash of the adjacency (the determinism gate: identical for "
             "identical seed/params at any thread count, on one machine).")
        .def_property_readonly("n", [](const PyIndex &s) { return s.n; })
        .def_property_readonly("dim", [](const PyIndex &s) { return s.d; })
        .def_property_readonly("metric", [](const PyIndex &s) {
            return s.metric == METRIC_L2 ? "l2" : "cosine"; })
        .def_property_readonly("degree_avg", [](const PyIndex &s) {
            return s.n ? (double)s.fg.g.idx.size() / (double)s.n : 0.0; })
        .def_property_readonly("n_dist_build", [](const PyIndex &s) {
            return s.fg.n_dist_build; })
        .def_property_readonly("build_seconds", [](const PyIndex &s) {
            return s.fg.t_trees + s.fg.t_overlay + s.fg.t_cap + s.fg.t_refine +
                   s.fg.t_harvest; })
        .def_property_readonly("t_trees", [](const PyIndex &s) { return s.fg.t_trees; })
        .def_property_readonly("t_harvest", [](const PyIndex &s) { return s.fg.t_harvest; });

    m.def("build", &build_index, py::return_value_policy::take_ownership,
          py::arg("X"), py::arg("metric") = "l2", py::arg("T") = 16,
          py::arg("harvest") = 400, py::arg("harvest_cap") = 64,
          py::arg("alpha") = 1.0f, py::arg("patience") = 0,
          py::arg("max_degree") = 0, py::arg("seed") = 42,
          py::arg("threads") = 0, py::arg("build_conc") = 0,
          py::arg("verbose") = false,
          "Build a GRAFT index over X [n,d] float32. Deterministic for fixed "
          "seed/params at any thread count.");
}

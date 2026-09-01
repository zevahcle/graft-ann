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
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

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

/* ---- flat mmap-able index format v1 ---------------------------------------
 * The fast-loading serve format: a 128-byte header with explicit section
 * offsets, then roots i32[n_roots] | ptr i64[n+1] | idx i32[E] | X f32[n*d],
 * each section 64-byte aligned. Position-independent (offsets only), host
 * endianness/width. Load = mmap + pointer casts into a GraphView -- no
 * rebuild, no copies; pages fault in on demand and stay evictable. Vectors
 * are stored as the index owns them (normalized when metric is cosine), so a
 * mapped index serves exactly like the built one. */
static const char FG_MAGIC[8] = {'G','R','A','F','T','I','D','X'};

struct FgHeader {
    char magic[8];
    u32  version;
    u32  metric;
    i64  n;
    i32  d;
    i32  n_roots;
    i64  E;
    u64  graph_hash;
    i64  off_roots, off_ptr, off_idx, off_x;
    u8   pad[48];
};
static_assert(sizeof(FgHeader) == 128, "format v1 header is 128 bytes");

static i64 align64(i64 x) { return (x + 63) & ~(i64)63; }

static void save_index(const PyIndex &s, const std::string &path) {
    FgHeader h;
    memset(&h, 0, sizeof h);
    memcpy(h.magic, FG_MAGIC, 8);
    h.version = 1;
    h.metric  = (u32)s.metric;
    h.n = s.n;
    h.d = s.d;
    h.n_roots = (i32)s.fg.roots.size();
    h.E = (i64)s.fg.g.idx.size();
    h.graph_hash = csr_hash(s.fg.g);
    h.off_roots = align64((i64)sizeof(FgHeader));
    h.off_ptr   = align64(h.off_roots + h.n_roots * (i64)sizeof(i32));
    h.off_idx   = align64(h.off_ptr + (s.n + 1) * (i64)sizeof(i64));
    h.off_x     = align64(h.off_idx + h.E * (i64)sizeof(i32));

    FILE *f = fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("graft.save: cannot open " + path);
    bool ok = true;
    auto put = [&](i64 off, const void *p, i64 bytes) {
        if (!ok) return;
        ok = fseeko(f, (off_t)off, SEEK_SET) == 0 &&
             (bytes == 0 || fwrite(p, 1, (size_t)bytes, f) == (size_t)bytes);
    };
    put(0, &h, (i64)sizeof h);
    put(h.off_roots, s.fg.roots.data(), h.n_roots * (i64)sizeof(i32));
    put(h.off_ptr, s.fg.g.ptr.data(), (s.n + 1) * (i64)sizeof(i64));
    put(h.off_idx, s.fg.g.idx.data(), h.E * (i64)sizeof(i32));
    put(h.off_x, s.X.data(), (i64)s.n * s.d * (i64)sizeof(f32));
    ok = (fclose(f) == 0) && ok;
    if (!ok) throw std::runtime_error("graft.save: short write to " + path);
}

struct PyMappedIndex {
    void      *base = nullptr;
    size_t     bytes = 0;
    GraphView  gv;
    const f32 *X = nullptr;
    i64        n = 0;
    int        d = 0;
    int        metric = METRIC_L2;
    u64        stored_hash = 0;

    ~PyMappedIndex() { if (base) munmap(base, bytes); }

    std::string graph_hash() const {   /* recomputed from the mapped bytes */
        char buf[17];
        snprintf(buf, sizeof buf, "%016llx",
                 (unsigned long long)csr_hash(gv.ptr, gv.n, gv.idx, gv.E));
        return std::string(buf);
    }
    std::string stored_hash_hex() const {
        char buf[17];
        snprintf(buf, sizeof buf, "%016llx", (unsigned long long)stored_hash);
        return std::string(buf);
    }
};

static PyMappedIndex *load_mmap_index(const std::string &path) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("graft.load_mmap: cannot open " + path);
    struct stat st;
    if (fstat(fd, &st) != 0) {
        close(fd);
        throw std::runtime_error("graft.load_mmap: cannot stat " + path);
    }
    if ((size_t)st.st_size < sizeof(FgHeader)) {
        close(fd);
        throw std::runtime_error("graft.load_mmap: truncated file " + path);
    }
    void *base = mmap(nullptr, (size_t)st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);   /* the mapping holds its own reference */
    if (base == MAP_FAILED)
        throw std::runtime_error("graft.load_mmap: mmap failed for " + path);

    auto idx = std::make_unique<PyMappedIndex>();
    idx->base = base;
    idx->bytes = (size_t)st.st_size;
    const FgHeader *h = (const FgHeader *)base;
    auto fail = [&](const char *why) {
        std::string msg = std::string("graft.load_mmap: ") + why + ": " + path;
        throw std::runtime_error(msg);   /* dtor unmaps */
    };
    if (memcmp(h->magic, FG_MAGIC, 8) != 0) fail("bad magic");
    if (h->version != 1) fail("unsupported version");
    i64 need = h->off_x + (i64)h->n * h->d * (i64)sizeof(f32);
    if (need > (i64)st.st_size) fail("truncated sections");
    const char *b = (const char *)base;
    idx->n = h->n;
    idx->d = h->d;
    idx->metric = (int)h->metric;
    idx->stored_hash = h->graph_hash;
    idx->X = (const f32 *)(b + h->off_x);
    idx->gv.ptr = (const i64 *)(b + h->off_ptr);
    idx->gv.idx = (const i32 *)(b + h->off_idx);
    idx->gv.roots = (const i32 *)(b + h->off_roots);
    idx->gv.n = h->n;
    idx->gv.E = h->E;
    idx->gv.n_roots = h->n_roots;
    idx->gv.metric = (int)h->metric;
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

static py::tuple search_mapped(PyMappedIndex &idx, farr Qarr, int k, int ef,
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
        /* hub entry, one per query -- identical to the built index's path,
         * so a mapped index returns bitwise-identical results */
        std::vector<i32> entries;
        make_entries(idx.gv, idx.X, idx.d, Q.data(), nq, "hub", 1, 1, entries);
        r = beam_search(idx.gv, idx.X, idx.d, Q.data(), nq,
                        entries.data(), 1, ef, k, threads);
    }
    py::array_t<i32> ids({nq, (i64)k});
    py::array_t<f32> dist({nq, (i64)k});
    std::copy(r.ids.begin(), r.ids.end(), ids.mutable_data());
    std::copy(r.dist.begin(), r.dist.end(), dist.mutable_data());
    return py::make_tuple(ids, dist);
}

using i64arr = py::array_t<i64, py::array::c_style | py::array::forcecast>;
using i32arr = py::array_t<i32, py::array::c_style | py::array::forcecast>;

/* SOLO serving path: batched beam over packed sub-list graphs. All arrays
 * are flat concatenations with per-list offset tables; MAP holds each
 * list's membership (local vertex -> global row in Xg). Returns
 * (ids [P,k] GLOBAL int32 -1-padded, dist [P,k] float32, n_dist). */
static py::tuple search_sublists_py(i64arr PTR, i64arr ptr_off,
                                    i32arr IDX, i64arr idx_off,
                                    i32arr ROOTS, i64arr root_off,
                                    i32arr nroots,
                                    i32arr MAP, i64arr map_off, i64arr nloc,
                                    farr Xg, const std::string &metric_s,
                                    farr Q, i64arr pair_q, i32arr pair_l,
                                    int k, int ef, int threads) {
    const int metric = parse_metric(metric_s);
    const i64 P = (i64)pair_q.shape(0);
    const int d = (int)Xg.shape(1);
    if (Q.ndim() != 2 || (int)Q.shape(1) != d)
        throw std::invalid_argument("Q must be [nq, d] matching Xg");
    i64 max_nloc = 0;
    {
        auto nl = nloc.unchecked<1>();
        for (i64 i = 0; i < nl.shape(0); i++)
            if (nl(i) > max_nloc) max_nloc = nl(i);
    }
    py::array_t<i32> ids({P, (i64)k});
    py::array_t<f32> dist({P, (i64)k});
    i64 nd;
    {
        py::gil_scoped_release release;
        nd = search_sublists(PTR.data(), ptr_off.data(),
                             IDX.data(), idx_off.data(),
                             ROOTS.data(), root_off.data(), nroots.data(),
                             MAP.data(), map_off.data(), nloc.data(),
                             Xg.data(), d, metric,
                             Q.data(), pair_q.data(), pair_l.data(),
                             P, max_nloc, ef, k, threads,
                             ids.mutable_data(), dist.mutable_data());
    }
    return py::make_tuple(ids, dist, nd);
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
        .def_property_readonly("t_harvest", [](const PyIndex &s) { return s.fg.t_harvest; })
        .def("save", &save_index, py::arg("path"),
             "Serialize to the flat mmap-able format v1 (adjacency + roots + "
             "vectors, 64-B aligned sections). Load with graft.load_mmap().")
        .def("graph", [](const PyIndex &s) {
             py::array_t<i64> ptr((i64)s.fg.g.ptr.size());
             py::array_t<i32> idx((i64)s.fg.g.idx.size());
             py::array_t<i32> roots((i64)s.fg.roots.size());
             std::copy(s.fg.g.ptr.begin(), s.fg.g.ptr.end(),
                       ptr.mutable_data());
             std::copy(s.fg.g.idx.begin(), s.fg.g.idx.end(),
                       idx.mutable_data());
             std::copy(s.fg.roots.begin(), s.fg.roots.end(),
                       roots.mutable_data());
             return py::make_tuple(ptr, idx, roots); },
             "Copy out the graph: (ptr int64[n+1], idx int32[E], roots "
             "int32[T]). For packing sub-list graphs (SOLO).");

    py::class_<PyMappedIndex>(m, "MappedIndex")
        .def("search", &search_mapped, py::arg("Q"), py::arg("k") = 10,
             py::arg("ef") = 64, py::arg("threads") = 0,
             "Beam-search the mapped graph. Identical semantics and results "
             "to Index.search on the index that was saved.")
        .def_property_readonly("graph_hash",
             [](const PyMappedIndex &s) { return s.graph_hash(); },
             "FNV hash recomputed from the mapped adjacency (parity check "
             "against stored_hash and the builder's graph_hash).")
        .def_property_readonly("stored_hash",
             [](const PyMappedIndex &s) { return s.stored_hash_hex(); })
        .def_property_readonly("n", [](const PyMappedIndex &s) { return s.n; })
        .def_property_readonly("dim", [](const PyMappedIndex &s) { return s.d; })
        .def_property_readonly("metric", [](const PyMappedIndex &s) {
            return s.metric == METRIC_L2 ? "l2" : "cosine"; })
        .def_property_readonly("nbytes", [](const PyMappedIndex &s) {
            return (i64)s.bytes; });

    m.def("search_sublists", &search_sublists_py,
          py::arg("PTR"), py::arg("ptr_off"), py::arg("IDX"),
          py::arg("idx_off"), py::arg("ROOTS"), py::arg("root_off"),
          py::arg("nroots"), py::arg("MAP"), py::arg("map_off"),
          py::arg("nloc"), py::arg("Xg"), py::arg("metric"), py::arg("Q"),
          py::arg("pair_q"), py::arg("pair_l"), py::arg("k") = 10,
          py::arg("ef") = 64, py::arg("threads") = 0,
          "Batched beam search over packed sub-list graphs (local topology "
          "+ local->global map into one global vector store; deterministic "
          "best-of-roots entry). Returns (global ids [P,k], dist [P,k], "
          "n_dist). Queries must be pre-normalized for metric='cosine'.");

    m.def("load_mmap", &load_mmap_index, py::return_value_policy::take_ownership,
          py::arg("path"),
          "Map a saved index (format v1): mmap + pointer casts, no rebuild, "
          "no copies; pages fault in on demand and stay evictable. Returns a "
          "MappedIndex.");

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

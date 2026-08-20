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

/* npy.hpp -- minimal .npy (NumPy v1.0/v2.0) reader.
 *
 * Supports C-order (fortran_order: False), 1-D or 2-D arrays, dtypes
 * '<f4', '<i4', '<i8' (little-endian, the only ones the harness emits via
 * convert_annb.py). Loads the whole file into an owned byte buffer and exposes
 * typed pointers. Not a general NumPy reader -- just enough for X/Q/gold.
 */
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct NpyArray {
    std::string          descr;   /* e.g. "<f4", "<i4", "<i8" */
    int64_t              rows = 0; /* shape[0] */
    int64_t              cols = 1; /* shape[1], or 1 for a 1-D array */
    std::vector<uint8_t> bytes;    /* raw payload (after the header) */

    const float   *f32p() const { return (const float   *)bytes.data(); }
    const int32_t *i32p() const { return (const int32_t *)bytes.data(); }
    const int64_t *i64p() const { return (const int64_t *)bytes.data(); }
};

/* pull the value of a quoted or literal field out of the header dict text */
static inline std::string npy_field(const std::string &h, const std::string &key) {
    size_t p = h.find(key);
    if (p == std::string::npos) return "";
    p = h.find(':', p);
    if (p == std::string::npos) return "";
    p++;
    while (p < h.size() && (h[p] == ' ' || h[p] == '\'')) p++;
    size_t e = p;
    while (e < h.size() && h[e] != '\'' && h[e] != ',' && h[e] != '}') e++;
    return h.substr(p, e - p);
}

static inline NpyArray npy_load(const char *path) {
    NpyArray a;
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[FATAL] cannot open %s\n", path); exit(2); }

    unsigned char magic[8];
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "\x93NUMPY", 6) != 0) {
        fprintf(stderr, "[FATAL] %s: not a .npy file\n", path); exit(2);
    }
    int major = magic[6];
    uint32_t hlen = 0;
    if (major == 1) {
        uint16_t h16 = 0;
        if (fread(&h16, 2, 1, f) != 1) { fprintf(stderr, "[FATAL] %s: short header\n", path); exit(2); }
        hlen = h16;
    } else {                                  /* v2.0+: 4-byte header length */
        if (fread(&hlen, 4, 1, f) != 1) { fprintf(stderr, "[FATAL] %s: short header\n", path); exit(2); }
    }
    std::string header((size_t)hlen, '\0');
    if (fread(&header[0], 1, hlen, f) != hlen) {
        fprintf(stderr, "[FATAL] %s: truncated header\n", path); exit(2);
    }

    a.descr = npy_field(header, "descr");
    if (header.find("'fortran_order': True") != std::string::npos) {
        fprintf(stderr, "[FATAL] %s: fortran_order arrays unsupported\n", path); exit(2);
    }

    /* parse shape tuple: (rows,) or (rows, cols) */
    size_t sp = header.find("'shape'");
    sp = header.find('(', sp);
    size_t se = header.find(')', sp);
    std::string shape = header.substr(sp + 1, se - sp - 1);
    int64_t dims[2] = {0, 1};
    int nd = 0;
    for (size_t i = 0; i < shape.size() && nd < 2;) {
        while (i < shape.size() && (shape[i] == ' ' || shape[i] == ',')) i++;
        if (i >= shape.size()) break;
        char *end = nullptr;
        long long v = strtoll(shape.c_str() + i, &end, 10);
        if (end == shape.c_str() + i) break;
        dims[nd++] = (int64_t)v;
        i = (size_t)(end - shape.c_str());
    }
    a.rows = dims[0];
    a.cols = nd >= 2 ? dims[1] : 1;

    int esz = a.descr == "<f4" || a.descr == "<i4" ? 4
            : a.descr == "<i8" ? 8 : 0;
    if (!esz) { fprintf(stderr, "[FATAL] %s: unsupported dtype %s\n", path, a.descr.c_str()); exit(2); }

    size_t nbytes = (size_t)a.rows * (size_t)a.cols * (size_t)esz;
    a.bytes.resize(nbytes);
    if (nbytes && fread(a.bytes.data(), 1, nbytes, f) != nbytes) {
        fprintf(stderr, "[FATAL] %s: truncated data\n", path); exit(2);
    }
    fclose(f);
    return a;
}

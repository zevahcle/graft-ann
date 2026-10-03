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

/* fg.hpp -- common substrate for the forest-graph C++ port.
 *
 * Port of forest_graph.py / sat_knn.py (the "Parallel by Construction" paper)
 * to C++17 + OpenMP. Style follows permforest.c: plain structs and functions,
 * compile-time SIMD dispatch, no template machinery.
 *
 * Determinism contract (the paper's central property):
 *   - every random choice is keyed by (seed, tree index) or (seed, vertex id)
 *     through splitmix64, never by thread id or completion order;
 *   - every parallel phase writes disjoint outputs;
 *   - all selections among equal distances are broken by vertex id.
 * Hence the final graph is bitwise identical at every thread count.
 */
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <vector>
#include <string>
#include <algorithm>

#if defined(_OPENMP)
  #include <omp.h>
#else
  static inline int  omp_get_max_threads(void) { return 1; }
  static inline int  omp_get_thread_num(void)  { return 0; }
  static inline void omp_set_num_threads(int)  {}
#endif

#if defined(__x86_64__) || defined(_M_X64)
  #include <immintrin.h>
#elif defined(__aarch64__) || defined(__ARM_NEON)
  #include <arm_neon.h>
#endif

typedef uint8_t  u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef int32_t  i32;
typedef int64_t  i64;
typedef float    f32;
typedef double   f64;

enum Metric { METRIC_COSINE = 0, METRIC_L2 = 1 };   /* cosine = 1 - dot on unit
                                                       vectors; L2 = squared */

/* ----- timing ------------------------------------------------------------ */
static inline double wall(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* ----- prefetch ----------------------------------------------------------- */
#if defined(__GNUC__) || defined(__clang__)
  #define FG_PREFETCH(p) __builtin_prefetch((const char *)(p), 0, 1)
#else
  #define FG_PREFETCH(p) ((void)0)
#endif

/* ----- deterministic RNG (splitmix64 stream) ------------------------------ */
static inline u64 splitmix64_next(u64 *s) {
    u64 z = (*s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

struct Rng {
    u64 s;
    explicit Rng(u64 seed) : s(seed) {}
    u64 next() { return splitmix64_next(&s); }
    /* uniform integer in [0, n) via 128-bit multiply (Lemire) */
    u64 below(u64 n) { return (u64)(((__uint128_t)next() * n) >> 64); }
    /* uniform double in [0, 1) */
    f64 uniform() { return (f64)(next() >> 11) * 0x1.0p-53; }
    /* standard normal via Box-Muller (uses two uniforms; second discarded) */
    f64 normal() {
        f64 u1 = uniform(), u2 = uniform();
        if (u1 < 1e-300) u1 = 1e-300;
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
    }
};

/* mix two keys into one seed (vertex- or tree-keyed streams) */
static inline u64 mix_seed(u64 a, u64 b) {
    u64 s = a ^ (b * 0x9E3779B97F4A7C15ull + 0xD1B54A32D192ED03ull);
    return splitmix64_next(&s);
}

/* ----- SIMD f32 distance kernels ------------------------------------------ *
 * The single hottest function. Compile-time path selection like permforest.c:
 *   AVX-512F : 16 floats / FMA, two accumulators
 *   AVX2+FMA :  8 floats / FMA, two accumulators
 *   NEON     :  4 floats / FMLA, two accumulators
 *   scalar   : fallback
 */
#if defined(__AVX512F__)
  #define FG_SIMD_PATH 3
#elif defined(__AVX2__) && defined(__FMA__)
  #define FG_SIMD_PATH 2
#elif defined(__aarch64__) || defined(__ARM_NEON)
  #define FG_SIMD_PATH 1
#else
  #define FG_SIMD_PATH 0
#endif

static inline f32 dot_f32(const f32 *__restrict a, const f32 *__restrict b, int d) {
    int i = 0;
    f32 acc = 0.0f;
#if FG_SIMD_PATH == 3
    __m512 v0 = _mm512_setzero_ps(), v1 = _mm512_setzero_ps();
    for (; i + 32 <= d; i += 32) {
        v0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i),      _mm512_loadu_ps(b + i),      v0);
        v1 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 16), _mm512_loadu_ps(b + i + 16), v1);
    }
    for (; i + 16 <= d; i += 16)
        v0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), v0);
    acc = _mm512_reduce_add_ps(_mm512_add_ps(v0, v1));
#elif FG_SIMD_PATH == 2
    __m256 v0 = _mm256_setzero_ps(), v1 = _mm256_setzero_ps();
    for (; i + 16 <= d; i += 16) {
        v0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i),     _mm256_loadu_ps(b + i),     v0);
        v1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), v1);
    }
    for (; i + 8 <= d; i += 8)
        v0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), v0);
    __m256 v = _mm256_add_ps(v0, v1);
    __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    acc = _mm_cvtss_f32(s);
#elif FG_SIMD_PATH == 1
    float32x4_t v0 = vdupq_n_f32(0.0f), v1 = vdupq_n_f32(0.0f);
    for (; i + 8 <= d; i += 8) {
        v0 = vfmaq_f32(v0, vld1q_f32(a + i),     vld1q_f32(b + i));
        v1 = vfmaq_f32(v1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
    }
    for (; i + 4 <= d; i += 4)
        v0 = vfmaq_f32(v0, vld1q_f32(a + i), vld1q_f32(b + i));
    acc = vaddvq_f32(vaddq_f32(v0, v1));
#endif
    for (; i < d; i++) acc += a[i] * b[i];
    return acc;
}

static inline f32 l2sq_f32(const f32 *__restrict a, const f32 *__restrict b, int d) {
    int i = 0;
    f32 acc = 0.0f;
#if FG_SIMD_PATH == 3
    __m512 v0 = _mm512_setzero_ps(), v1 = _mm512_setzero_ps();
    for (; i + 32 <= d; i += 32) {
        __m512 t0 = _mm512_sub_ps(_mm512_loadu_ps(a + i),      _mm512_loadu_ps(b + i));
        __m512 t1 = _mm512_sub_ps(_mm512_loadu_ps(a + i + 16), _mm512_loadu_ps(b + i + 16));
        v0 = _mm512_fmadd_ps(t0, t0, v0);
        v1 = _mm512_fmadd_ps(t1, t1, v1);
    }
    for (; i + 16 <= d; i += 16) {
        __m512 t = _mm512_sub_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i));
        v0 = _mm512_fmadd_ps(t, t, v0);
    }
    acc = _mm512_reduce_add_ps(_mm512_add_ps(v0, v1));
#elif FG_SIMD_PATH == 2
    __m256 v0 = _mm256_setzero_ps(), v1 = _mm256_setzero_ps();
    for (; i + 16 <= d; i += 16) {
        __m256 t0 = _mm256_sub_ps(_mm256_loadu_ps(a + i),     _mm256_loadu_ps(b + i));
        __m256 t1 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8));
        v0 = _mm256_fmadd_ps(t0, t0, v0);
        v1 = _mm256_fmadd_ps(t1, t1, v1);
    }
    for (; i + 8 <= d; i += 8) {
        __m256 t = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
        v0 = _mm256_fmadd_ps(t, t, v0);
    }
    __m256 v = _mm256_add_ps(v0, v1);
    __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    acc = _mm_cvtss_f32(s);
#elif FG_SIMD_PATH == 1
    float32x4_t v0 = vdupq_n_f32(0.0f), v1 = vdupq_n_f32(0.0f);
    for (; i + 8 <= d; i += 8) {
        float32x4_t t0 = vsubq_f32(vld1q_f32(a + i),     vld1q_f32(b + i));
        float32x4_t t1 = vsubq_f32(vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
        v0 = vfmaq_f32(v0, t0, t0);
        v1 = vfmaq_f32(v1, t1, t1);
    }
    for (; i + 4 <= d; i += 4) {
        float32x4_t t = vsubq_f32(vld1q_f32(a + i), vld1q_f32(b + i));
        v0 = vfmaq_f32(v0, t, t);
    }
    acc = vaddvq_f32(vaddq_f32(v0, v1));
#endif
    for (; i < d; i++) { f32 t = a[i] - b[i]; acc += t * t; }
    return acc;
}

static inline f32 fg_dist(const f32 *a, const f32 *b, int d, int metric) {
    return metric == METRIC_COSINE ? 1.0f - dot_f32(a, b, d) : l2sq_f32(a, b, d);
}

/* ----- batched 1-vs-4 kernels ---------------------------------------------- *
 * One query row against four data rows in a single pass: the query streams
 * from registers, eight accumulator chains hide FMA latency, and on gathered
 * rows the four miss streams overlap. Each lane reproduces the 1-vs-1 kernel's
 * reduction order EXACTLY (same chunking, same accumulator split, same tail),
 * so out[i] is bitwise identical to calling dot_f32/l2sq_f32 per row --
 * callers may mix batched and single evaluations without perturbing any
 * (distance, id) selection, and the built graph is unchanged. */
static inline void dot4_f32(const f32 *__restrict q,
                            const f32 *__restrict r0, const f32 *__restrict r1,
                            const f32 *__restrict r2, const f32 *__restrict r3,
                            int d, f32 *__restrict out)
{
#if FG_SIMD_PATH == 1
    float32x4_t a0 = vdupq_n_f32(0.0f), b0 = vdupq_n_f32(0.0f);
    float32x4_t a1 = vdupq_n_f32(0.0f), b1 = vdupq_n_f32(0.0f);
    float32x4_t a2 = vdupq_n_f32(0.0f), b2 = vdupq_n_f32(0.0f);
    float32x4_t a3 = vdupq_n_f32(0.0f), b3 = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 8 <= d; i += 8) {
        float32x4_t qa = vld1q_f32(q + i), qb = vld1q_f32(q + i + 4);
        a0 = vfmaq_f32(a0, qa, vld1q_f32(r0 + i));
        b0 = vfmaq_f32(b0, qb, vld1q_f32(r0 + i + 4));
        a1 = vfmaq_f32(a1, qa, vld1q_f32(r1 + i));
        b1 = vfmaq_f32(b1, qb, vld1q_f32(r1 + i + 4));
        a2 = vfmaq_f32(a2, qa, vld1q_f32(r2 + i));
        b2 = vfmaq_f32(b2, qb, vld1q_f32(r2 + i + 4));
        a3 = vfmaq_f32(a3, qa, vld1q_f32(r3 + i));
        b3 = vfmaq_f32(b3, qb, vld1q_f32(r3 + i + 4));
    }
    for (; i + 4 <= d; i += 4) {
        float32x4_t qa = vld1q_f32(q + i);
        a0 = vfmaq_f32(a0, qa, vld1q_f32(r0 + i));
        a1 = vfmaq_f32(a1, qa, vld1q_f32(r1 + i));
        a2 = vfmaq_f32(a2, qa, vld1q_f32(r2 + i));
        a3 = vfmaq_f32(a3, qa, vld1q_f32(r3 + i));
    }
    out[0] = vaddvq_f32(vaddq_f32(a0, b0));
    out[1] = vaddvq_f32(vaddq_f32(a1, b1));
    out[2] = vaddvq_f32(vaddq_f32(a2, b2));
    out[3] = vaddvq_f32(vaddq_f32(a3, b3));
    for (; i < d; i++) {
        out[0] += q[i] * r0[i];
        out[1] += q[i] * r1[i];
        out[2] += q[i] * r2[i];
        out[3] += q[i] * r3[i];
    }
#else
    out[0] = dot_f32(q, r0, d); out[1] = dot_f32(q, r1, d);
    out[2] = dot_f32(q, r2, d); out[3] = dot_f32(q, r3, d);
#endif
}

static inline void l2sq4_f32(const f32 *__restrict q,
                             const f32 *__restrict r0, const f32 *__restrict r1,
                             const f32 *__restrict r2, const f32 *__restrict r3,
                             int d, f32 *__restrict out)
{
#if FG_SIMD_PATH == 1
    float32x4_t a0 = vdupq_n_f32(0.0f), b0 = vdupq_n_f32(0.0f);
    float32x4_t a1 = vdupq_n_f32(0.0f), b1 = vdupq_n_f32(0.0f);
    float32x4_t a2 = vdupq_n_f32(0.0f), b2 = vdupq_n_f32(0.0f);
    float32x4_t a3 = vdupq_n_f32(0.0f), b3 = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 8 <= d; i += 8) {
        float32x4_t qa = vld1q_f32(q + i), qb = vld1q_f32(q + i + 4);
        float32x4_t t;
        t = vsubq_f32(qa, vld1q_f32(r0 + i));     a0 = vfmaq_f32(a0, t, t);
        t = vsubq_f32(qb, vld1q_f32(r0 + i + 4)); b0 = vfmaq_f32(b0, t, t);
        t = vsubq_f32(qa, vld1q_f32(r1 + i));     a1 = vfmaq_f32(a1, t, t);
        t = vsubq_f32(qb, vld1q_f32(r1 + i + 4)); b1 = vfmaq_f32(b1, t, t);
        t = vsubq_f32(qa, vld1q_f32(r2 + i));     a2 = vfmaq_f32(a2, t, t);
        t = vsubq_f32(qb, vld1q_f32(r2 + i + 4)); b2 = vfmaq_f32(b2, t, t);
        t = vsubq_f32(qa, vld1q_f32(r3 + i));     a3 = vfmaq_f32(a3, t, t);
        t = vsubq_f32(qb, vld1q_f32(r3 + i + 4)); b3 = vfmaq_f32(b3, t, t);
    }
    for (; i + 4 <= d; i += 4) {
        float32x4_t qa = vld1q_f32(q + i), t;
        t = vsubq_f32(qa, vld1q_f32(r0 + i)); a0 = vfmaq_f32(a0, t, t);
        t = vsubq_f32(qa, vld1q_f32(r1 + i)); a1 = vfmaq_f32(a1, t, t);
        t = vsubq_f32(qa, vld1q_f32(r2 + i)); a2 = vfmaq_f32(a2, t, t);
        t = vsubq_f32(qa, vld1q_f32(r3 + i)); a3 = vfmaq_f32(a3, t, t);
    }
    out[0] = vaddvq_f32(vaddq_f32(a0, b0));
    out[1] = vaddvq_f32(vaddq_f32(a1, b1));
    out[2] = vaddvq_f32(vaddq_f32(a2, b2));
    out[3] = vaddvq_f32(vaddq_f32(a3, b3));
    for (; i < d; i++) {
        f32 t0 = q[i] - r0[i]; out[0] += t0 * t0;
        f32 t1 = q[i] - r1[i]; out[1] += t1 * t1;
        f32 t2 = q[i] - r2[i]; out[2] += t2 * t2;
        f32 t3 = q[i] - r3[i]; out[3] += t3 * t3;
    }
#else
    out[0] = l2sq_f32(q, r0, d); out[1] = l2sq_f32(q, r1, d);
    out[2] = l2sq_f32(q, r2, d); out[3] = l2sq_f32(q, r3, d);
#endif
}

static inline void fg_dist4(const f32 *q,
                            const f32 *r0, const f32 *r1,
                            const f32 *r2, const f32 *r3,
                            int d, int metric, f32 *out)
{
    if (metric == METRIC_COSINE) {
        dot4_f32(q, r0, r1, r2, r3, d, out);
        out[0] = 1.0f - out[0]; out[1] = 1.0f - out[1];
        out[2] = 1.0f - out[2]; out[3] = 1.0f - out[3];
    } else {
        l2sq4_f32(q, r0, r1, r2, r3, d, out);
    }
}

/* ----- register-tiled 4x4 distance tile ---------------------------------- *
 * Distances from 4 queries to 4 data rows in one pass over d: 16 accumulators,
 * every loaded vector serves 4 pairs (the 1x4 kernel serves 1). NEON: full 4x4
 * (16 acc + 8 loads = 24 of 32 registers). AVX2: two 2x4 halves (8 acc + 6
 * loads of 16 registers). Otherwise falls back to four 1x4 calls. Each pair's
 * reduction order is fixed by the code path, so results are deterministic for
 * a given binary; they need not equal fg_dist4's rounding.
 * out[r*4 + c] = dist(q_r, x_c). */
static inline void fg_dist4x4(const f32 *q0, const f32 *q1, const f32 *q2, const f32 *q3,
                              const f32 *r0, const f32 *r1, const f32 *r2, const f32 *r3,
                              int d, int metric, f32 *out)
{
#if FG_SIMD_PATH == 1
    const f32 *Q[4] = {q0, q1, q2, q3}; const f32 *R[4] = {r0, r1, r2, r3};
    float32x4_t acc[4][4];
    for (int a = 0; a < 4; a++) for (int b = 0; b < 4; b++) acc[a][b] = vdupq_n_f32(0.0f);
    int i = 0;
    if (metric == METRIC_COSINE) {
        for (; i + 4 <= d; i += 4) {
            float32x4_t x0 = vld1q_f32(r0 + i), x1 = vld1q_f32(r1 + i), x2 = vld1q_f32(r2 + i), x3 = vld1q_f32(r3 + i);
            for (int a = 0; a < 4; a++) {
                float32x4_t qa = vld1q_f32(Q[a] + i);
                acc[a][0] = vfmaq_f32(acc[a][0], qa, x0); acc[a][1] = vfmaq_f32(acc[a][1], qa, x1);
                acc[a][2] = vfmaq_f32(acc[a][2], qa, x2); acc[a][3] = vfmaq_f32(acc[a][3], qa, x3);
            }
        }
        for (int a = 0; a < 4; a++) for (int b = 0; b < 4; b++) {
            f32 v = vaddvq_f32(acc[a][b]);
            for (int j = i; j < d; j++) v += Q[a][j] * R[b][j];
            out[a * 4 + b] = 1.0f - v;
        }
    } else {
        for (; i + 4 <= d; i += 4) {
            float32x4_t x0 = vld1q_f32(r0 + i), x1 = vld1q_f32(r1 + i), x2 = vld1q_f32(r2 + i), x3 = vld1q_f32(r3 + i);
            for (int a = 0; a < 4; a++) {
                float32x4_t qa = vld1q_f32(Q[a] + i), t;
                t = vsubq_f32(qa, x0); acc[a][0] = vfmaq_f32(acc[a][0], t, t);
                t = vsubq_f32(qa, x1); acc[a][1] = vfmaq_f32(acc[a][1], t, t);
                t = vsubq_f32(qa, x2); acc[a][2] = vfmaq_f32(acc[a][2], t, t);
                t = vsubq_f32(qa, x3); acc[a][3] = vfmaq_f32(acc[a][3], t, t);
            }
        }
        for (int a = 0; a < 4; a++) for (int b = 0; b < 4; b++) {
            f32 v = vaddvq_f32(acc[a][b]);
            for (int j = i; j < d; j++) { f32 t = Q[a][j] - R[b][j]; v += t * t; }
            out[a * 4 + b] = v;
        }
    }
#elif FG_SIMD_PATH == 2
    const f32 *Q[4] = {q0, q1, q2, q3}; const f32 *R[4] = {r0, r1, r2, r3};
    for (int half = 0; half < 2; half++) {          /* 2 queries x 4 rows per half */
        const f32 *qa = Q[half * 2], *qb = Q[half * 2 + 1];
        __m256 A0 = _mm256_setzero_ps(), A1 = A0, A2 = A0, A3 = A0, B0 = A0, B1 = A0, B2 = A0, B3 = A0;
        int i = 0;
        if (metric == METRIC_COSINE) {
            for (; i + 8 <= d; i += 8) {
                __m256 x0 = _mm256_loadu_ps(r0 + i), x1 = _mm256_loadu_ps(r1 + i), x2 = _mm256_loadu_ps(r2 + i), x3 = _mm256_loadu_ps(r3 + i);
                __m256 va = _mm256_loadu_ps(qa + i), vb = _mm256_loadu_ps(qb + i);
                A0 = _mm256_fmadd_ps(va, x0, A0); A1 = _mm256_fmadd_ps(va, x1, A1); A2 = _mm256_fmadd_ps(va, x2, A2); A3 = _mm256_fmadd_ps(va, x3, A3);
                B0 = _mm256_fmadd_ps(vb, x0, B0); B1 = _mm256_fmadd_ps(vb, x1, B1); B2 = _mm256_fmadd_ps(vb, x2, B2); B3 = _mm256_fmadd_ps(vb, x3, B3);
            }
        } else {
            for (; i + 8 <= d; i += 8) {
                __m256 x0 = _mm256_loadu_ps(r0 + i), x1 = _mm256_loadu_ps(r1 + i), x2 = _mm256_loadu_ps(r2 + i), x3 = _mm256_loadu_ps(r3 + i);
                __m256 va = _mm256_loadu_ps(qa + i), vb = _mm256_loadu_ps(qb + i), t;
                t = _mm256_sub_ps(va, x0); A0 = _mm256_fmadd_ps(t, t, A0); t = _mm256_sub_ps(va, x1); A1 = _mm256_fmadd_ps(t, t, A1);
                t = _mm256_sub_ps(va, x2); A2 = _mm256_fmadd_ps(t, t, A2); t = _mm256_sub_ps(va, x3); A3 = _mm256_fmadd_ps(t, t, A3);
                t = _mm256_sub_ps(vb, x0); B0 = _mm256_fmadd_ps(t, t, B0); t = _mm256_sub_ps(vb, x1); B1 = _mm256_fmadd_ps(t, t, B1);
                t = _mm256_sub_ps(vb, x2); B2 = _mm256_fmadd_ps(t, t, B2); t = _mm256_sub_ps(vb, x3); B3 = _mm256_fmadd_ps(t, t, B3);
            }
        }
        /* 8 accumulators -> 8 sums in one register: hadd tree + 128-bit halves */
        __m256 t0 = _mm256_hadd_ps(A0, A1), t1 = _mm256_hadd_ps(A2, A3);
        __m256 t2 = _mm256_hadd_ps(B0, B1), t3 = _mm256_hadd_ps(B2, B3);
        __m256 u0 = _mm256_hadd_ps(t0, t1), u1 = _mm256_hadd_ps(t2, t3);
        /* u0 = [a0 a1 a2 a3 | a0' a1' a2' a3'] (lo/hi 128-bit partials), same for u1 */
        __m256 r = _mm256_add_ps(_mm256_permute2f128_ps(u0, u1, 0x20),
                                 _mm256_permute2f128_ps(u0, u1, 0x31));
        if (i == d) {                                /* padded rows: no tail, vector epilogue */
            if (metric == METRIC_COSINE) r = _mm256_sub_ps(_mm256_set1_ps(1.0f), r);
            _mm256_storeu_ps(out + half * 8, r);     /* out[half*8 + k] = [A0..A3 B0..B3] */
        } else {
            f32 v[8]; _mm256_storeu_ps(v, r);
            for (int k = 0; k < 8; k++) {
                const f32 *qq = k < 4 ? qa : qb; const f32 *rr = R[k & 3];
                f32 x = v[k];
                if (metric == METRIC_COSINE) { for (int j = i; j < d; j++) x += qq[j] * rr[j]; x = 1.0f - x; }
                else for (int j = i; j < d; j++) { f32 t = qq[j] - rr[j]; x += t * t; }
                out[half * 8 + k] = x;
            }
        }
    }
#else
    fg_dist4(q0, r0, r1, r2, r3, d, metric, out);
    fg_dist4(q1, r0, r1, r2, r3, d, metric, out + 4);
    fg_dist4(q2, r0, r1, r2, r3, d, metric, out + 8);
    fg_dist4(q3, r0, r1, r2, r3, d, metric, out + 12);
#endif
}

/* 4-bit mask of lanes with d4[k] <= tau (ties pass, so the exact packed
 * (distance, id) heap rule still decides them). One compare per 4 pairs. */
static inline int fg_le4_mask(const f32 *d4, f32 tau)
{
#if FG_SIMD_PATH == 1
    uint32x4_t m = vcleq_f32(vld1q_f32(d4), vdupq_n_f32(tau));
    if (vmaxvq_u32(m) == 0) return 0;
    return (int)(vgetq_lane_u32(m, 0) & 1) | (int)(vgetq_lane_u32(m, 1) & 2) |
           (int)(vgetq_lane_u32(m, 2) & 4) | (int)(vgetq_lane_u32(m, 3) & 8);
#elif FG_SIMD_PATH == 2 || FG_SIMD_PATH == 3
    return _mm_movemask_ps(_mm_cmple_ps(_mm_loadu_ps(d4), _mm_set1_ps(tau)));
#else
    return (d4[0] <= tau) | ((d4[1] <= tau) << 1) | ((d4[2] <= tau) << 2) | ((d4[3] <= tau) << 3);
#endif
}

#if defined(__x86_64__) || defined(__i386__)
#  include <immintrin.h>
#  define FG_SPIN_PAUSE() _mm_pause()
#elif defined(__aarch64__)
#  define FG_SPIN_PAUSE() __asm__ __volatile__("yield")
#else
#  define FG_SPIN_PAUSE() ((void)0)
#endif

/* ----- sortable float keys ------------------------------------------------ *
 * Monotone bijection f32 -> u32 over ALL floats (incl. the slightly negative
 * cosine distances FP error can produce), so distances can be ordered as
 * unsigned integers. Packed (key<<32 | id), ascending u64 order == ascending
 * (distance, id): the id tiebreak is what makes every selection deterministic.
 */
static inline u32 f2key(f32 f) {
    u32 x; memcpy(&x, &f, 4);
    return x ^ (u32)(((i32)x >> 31) | (i32)0x80000000);
}
static inline f32 key2f(u32 k) {
    u32 x = (k & 0x80000000u) ? (k ^ 0x80000000u) : ~k;
    f32 f; memcpy(&f, &x, 4);
    return f;
}
static inline u64 pack_di(f32 dist, i32 id) {
    return ((u64)f2key(dist) << 32) | (u32)id;
}
static inline i32 unpack_id(u64 p)   { return (i32)(u32)p; }
static inline f32 unpack_dist(u64 p) { return key2f((u32)(p >> 32)); }

/* ----- binary heaps over packed u64 keys ----------------------------------- *
 * Flat-array heaps; n is the live size, caller owns capacity.
 * min-heap: root = smallest key (closest candidate to expand next).
 * max-heap: root = largest key  (worst element of the bounded result set).
 */
static inline void hmin_push(u64 *h, int *n, u64 v) {
    int i = (*n)++;
    h[i] = v;
    while (i > 0) {
        int p = (i - 1) >> 1;
        if (h[p] <= h[i]) break;
        u64 t = h[p]; h[p] = h[i]; h[i] = t;
        i = p;
    }
}
static inline u64 hmin_pop(u64 *h, int *n) {
    u64 top = h[0];
    int m = --(*n);
    h[0] = h[m];
    int i = 0;
    for (;;) {
        int l = 2 * i + 1, r = l + 1, s = i;
        if (l < m && h[l] < h[s]) s = l;
        if (r < m && h[r] < h[s]) s = r;
        if (s == i) break;
        u64 t = h[s]; h[s] = h[i]; h[i] = t;
        i = s;
    }
    return top;
}
static inline void hmax_push(u64 *h, int *n, u64 v) {
    int i = (*n)++;
    h[i] = v;
    while (i > 0) {
        int p = (i - 1) >> 1;
        if (h[p] >= h[i]) break;
        u64 t = h[p]; h[p] = h[i]; h[i] = t;
        i = p;
    }
}
static inline void hmax_replace_top(u64 *h, int n, u64 v) {
    h[0] = v;
    int i = 0;
    for (;;) {
        int l = 2 * i + 1, r = l + 1, s = i;
        if (l < n && h[l] > h[s]) s = l;
        if (r < n && h[r] > h[s]) s = r;
        if (s == i) break;
        u64 t = h[s]; h[s] = h[i]; h[i] = t;
        i = s;
    }
}
/* bounded top-k offer: keep the `cap` smallest keys in a max-heap */
static inline void hmax_offer(u64 *h, int *n, int cap, u64 v) {
    if (*n < cap)            hmax_push(h, n, v);
    else if (v < h[0])       hmax_replace_top(h, *n, v);
}

/* ----- huge-page clone ------------------------------------------------------ *
 * Random row access over a multi-hundred-MB matrix takes a TLB miss per fetch
 * with 4KB pages; a 2MB-page backing removes it. Values are copied verbatim:
 * no effect on determinism. Falls back silently to the original pointer if
 * mmap/madvise are unavailable. */
#include <sys/mman.h>
static inline const f32 *huge_clone(const std::vector<f32> &src) {
#if defined(__linux__) && defined(MADV_HUGEPAGE)
    size_t bytes = (src.size() * sizeof(f32) + (2u << 20) - 1) & ~(size_t)((2u << 20) - 1);
    void *p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return src.data();
    madvise(p, bytes, MADV_HUGEPAGE);
    memcpy(p, src.data(), src.size() * sizeof(f32));
    return (const f32 *)p;
#else
    return src.data();
#endif
}

/* ----- CSR adjacency ------------------------------------------------------ */
struct CSR {
    std::vector<i64> ptr;   /* n+1 */
    std::vector<i32> idx;   /* ptr[n] entries */
    i64 n() const { return (i64)ptr.size() - 1; }
    i64 deg(i64 v) const { return ptr[v + 1] - ptr[v]; }
};

/* FNV-1a over the CSR -- the determinism check. Pointer form so mapped
 * (non-owning) graphs hash identically to built ones. */
static inline u64 csr_hash(const i64 *ptr, i64 n, const i32 *idx, i64 E) {
    u64 h = 0xCBF29CE484222325ull;
    auto eat = [&h](const void *p, size_t bytes) {
        const u8 *q = (const u8 *)p;
        for (size_t i = 0; i < bytes; i++) { h ^= q[i]; h *= 0x100000001B3ull; }
    };
    eat(ptr, (size_t)(n + 1) * sizeof(i64));
    eat(idx, (size_t)E * sizeof(i32));
    return h;
}
static inline u64 csr_hash(const CSR &g) {
    return csr_hash(g.ptr.data(), g.n(), g.idx.data(), (i64)g.idx.size());
}

/* ----- public API ---------------------------------------------------------- */
enum Policy { POLICY_PROXIMAL = 0, POLICY_DISTAL = 1, POLICY_RANDOM = 2 };

struct BuildParams {
    int    T            = 16;       /* trees in the scaffold forest. Quality of
                                       the harvested graph inherits from the
                                       union's redundancy: on GloVe d@0.97 =
                                       18.8k/15.5k/14.5k/14.0k for T=8/16/24/32
                                       at harvest_ef 400. Named profiles:
                                       fast T16/ef400, balanced T24/ef600,
                                       quality T32/ef600. */
    int    policy       = POLICY_PROXIMAL;  /* member ordering in the tree split;
                                       proximal builds at ~half distal's cost at
                                       equal quality (E2a). */
    int    harvest_ef   = 400;      /* phase 4, THE construction: beam width of
                                       the per-point search over the frozen
                                       union. The dominant quality lever. 0 =
                                       serve the raw union (scaffold/debug). */
    int    harvest_cap  = 64;       /* degree cap of the harvested adjacency */
    f32    harvest_alpha = 1.0f;    /* occlusion slack for the harvest prune.
                                       1.0 on concentrated (angular/holographic)
                                       data where slack cannot fire; 1.2 on
                                       spread/clustered L2 data (Vamana
                                       practice, confirmed on synthetic). */
    int    harvest_cand = 0;        /* optional bound on prune candidates per
                                       vertex; 0 = the whole expanded set (the
                                       default). A tight bound silently discards
                                       the route nodes -- the funnel edges the
                                       harvest exists to keep (measured: 192
                                       capped the clustered synthetic at 0.46
                                       recall vs 1.00 unbounded, and flattened
                                       GloVe's high-recall end). */
    int    tree_depth   = 0;        /* shallow scaffold trees (EC): stop
                                       expanding tree t>0 at this depth; the
                                       members of a depth-D node attach to it
                                       directly (bucket star). Tree 0 is always
                                       FULL (the connectivity spine). 0 = all
                                       trees full. Caps per-tree cost at
                                       ~D*n*arity (linear in n) and makes
                                       count-vs-depth a testable axis. */
    int    harvest_patience = 0;    /* adaptive termination: stop a point's
                                       harvest search after this many consecutive
                                       expansions that fail to improve the
                                       best-`harvest_cap` candidate set; 0 = off
                                       (fixed-ef budget). Makes harvest cost
                                       track per-point difficulty: the route is
                                       kept (early expansions), only the
                                       redundant local-cloud tail is cut. */
    int    harvest_rounds = 1;      /* one strong round beats two weak ones
                                       (measured: round 2 on the thin output
                                       collapses); raise harvest_ef instead. */
    f32    harvest_straighten = 0.0f; /* string-pull the discovered route into
                                       monotone shortcut edges with geometric
                                       progress factor >1 (1.2 typical); 0 =
                                       off. Recovers the long-range links that
                                       occlusion discards, certified by usage.
                                       Costs cross-node writes, so the edges are
                                       emitted to per-thread buffers and merged
                                       by a deterministic sort. */
    int    max_degree   = 0;        /* hard out-degree cap applied LAST (after
                                       the spine/shortcut unions, which join
                                       unpruned and can leave hubs -- measured
                                       max 640 on gist from high root arity).
                                       Keeps the nearest max_degree, in build
                                       space. 0 = off. */
    int    shortcut_cap = 16;       /* max string-pulled shortcuts kept per
                                       source node (nearest first); 0 = keep all
                                       (measured: the roots reach ~31k degree). */
    int    harvest_blocks = 0;      /* >1 = block-synchronous mutation: harvest
                                       the points in B blocks (block = p % B),
                                       folding each block's finished rows back
                                       into the scaffold before the next. The
                                       substrate then improves as construction
                                       proceeds -- the feedback loop incremental
                                       builders get for free -- while staying
                                       deterministic: within a block the
                                       substrate is frozen and writes are
                                       disjoint. 0/1 = the frozen scaffold. */
    int    dense_leaders = 0;       /* >0 = DENSE (search-free) construction:
                                       k-means leaders, every point in the leaves
                                       of its dense_m nearest leaders, exact
                                       all-pairs inside each leaf -> top dense_C
                                       candidates per point -> symmetrize + the
                                       harvest prune (cap/alpha). Pair sets fixed
                                       in advance: no beam anywhere. */
    int    dense_m       = 5;
    int    dense_C       = 200;
    int    dense_kmeans  = 10;      /* Lloyd iterations on a 200k sample */
    int    dense_top     = 0;       /* >0 = two-level leader hierarchy: cluster the
                                       dense_leaders into dense_top groups and route
                                       each point through its dense_probe nearest
                                       groups, so the membership stage costs
                                       n*(dense_top + probe*L/dense_top) instead of
                                       n*L. Needed past ~10^7, where L is 10^4-10^5:
                                       at n=10^8, L=24000 the flat scan is 2.4e12
                                       distances. 0 = flat (exact). */
    int    dense_probe   = 3;       /* groups probed per point when dense_top > 0 */
    const char *dense_cands = nullptr; /* candidate lists computed elsewhere (GPU): n x
                                       dense_C int32, nearest first, -1 padded. Skips
                                       stages 1-3; prune, spine and evaluation as usual. */
    const char *dense_m_file = nullptr; /* optional per-point membership count: n uint8,
                                       clamped to [1, dense_m]; point p joins only the
                                       leaves of its first m(p) nearest leaders
                                       (adaptive memberships, 2026-09-30). */
    const char *dense_leaders_file = nullptr; /* optional initial leaders: raw
                                       float32, L x d, replacing the seeded
                                       sample points before the Lloyd loop
                                       (warm start; 2026-09-29). */
    f32    dense_radius  = 0;       /* >0: before the first prune, drop candidates
                                       farther than dense_radius x d(v, its
                                       dense_radius_k-th nearest candidate). Off = 0.
                                       Tests whether a signature-selected pool
                                       (shared-neighbour, hashed) can be used by
                                       discarding its far tail (2026-09-26). */
    int    dense_radius_k = 10;
    i64    dense_spine_s = 10000;   /* |S| for dense_spine 3 (HSP over a random sample) */
    int    dense_spine_repair = 0;  /* dense_spine 3: 1 = give every in-degree-0 vertex an in-edge
                                       from its nearest kept out-neighbour (no distances) */
    i64    dense_spine_cap = 0;     /* dense_spine 3: >0 caps each HSP row at its nearest cap neighbours */
    int    dense_spine   = 1;       /* 2 = k-nearest-leader graph between leaf medoids
                                       (O(L^2), free); 1 = union one SAT tree's parent edges into
                                       the pruned dense graph (GRAFT's connectivity
                                       guarantee): at alpha = 1 the prune occludes
                                       every cross-leaf candidate behind a near one,
                                       and on clustered data the leaves ARE the
                                       clusters, so without it the graph shatters
                                       (measured: recall 0.002 on the synthetic
                                       smoke). ~1 tree = a few % of the build. */
    int    dense_prune   = 0;       /* 0 = GRAFT (symmetrize, then prune once);
                                       1 = Vamana-style (prune own list, add reverse
                                       edges, re-prune only lists over cap). */
    const char *dump_parents = nullptr; /* write tree-0 parents (n x i32) here:
                                       defines the SAT subtrees = the coherent
                                       blocks of the density study. */
    const char *log_mask    = nullptr; /* n bytes; 1 = log this point's harvest */
    const char *log_visited = nullptr; /* output: for each masked p, the ids
                                       whose distance to p was evaluated, in
                                       order, with (cn | 1<<31) marking each
                                       expansion (a lockstep step). Density =
                                       sum|V_i| / (B * |union V_i|) per block. */
    int    scaffold_cap = 0;        /* >0 = cap the SCAFFOLD's out-degree to the
                                       nearest scaffold_cap per vertex before
                                       the harvest (0 = uncapped, the default
                                       and the shipped behaviour). Harvest cost
                                       is ef x scaffold degree, so this is the
                                       direct lever on the dominant phase; the
                                       question it answers is how much of the
                                       union's redundancy routability needs. */
    int    substrate_random = 0;    /* >0 = REPLACE the forest scaffold with a
                                       seeded random K-regular directed graph
                                       (symmetrized), K = this value. The
                                       control for "the forest is just an init":
                                       Vamana's own initialization, harvested
                                       frozen. Trees are still built only for
                                       the roots/spine (entry + connectivity);
                                       their edges are NOT used. */
    int    overlay_directed = 0;    /* 1 = parent->child union only, no backward
                                       edges (ablation: within 6-10% of the
                                       symmetric union). */
    u64    seed         = 42;
    int    threads      = 0;        /* 0 = OpenMP default */
    int    build_conc   = 0;        /* forest-build concurrency cap; 0 = threads.
                                       Bounds peak build RAM; GRAPH-PRESERVING
                                       (trees keyed by (seed,t), disjoint
                                       outputs -> bitwise-identical forest). */
    bool   verbose      = true;
};

struct ForestGraph {
    CSR              g;
    std::vector<i32> roots;     /* the T hub vertices (query entry points) */
    int              metric;
    /* phase timings (s) and counters, for the build-scaling story */
    double t_trees = 0, t_overlay = 0, t_cap = 0, t_refine = 0, t_harvest = 0;
    i64    n_shortcuts = 0;
    i64    n_dist_build = 0;
};

/* ----- vote-counting aggregator (the additive recovery-tier path) ---------- *
 * A per-vertex weighted candidate list built by COUNTING how many trees mention
 * each (i,j) pair, instead of the navigation path's dedup-then-random-cap. The
 * mention relation is parent<->child (the SAT nearest-attachment spine) plus an
 * ancestor climb of A levels with level-decayed votes; siblings are deliberately
 * NOT emitted (NN-descent's neighbour-of-neighbour step recovers them). Vote ==
 * seeding priority (kept-by-vote, fed vote-first); the final code weight is the
 * refined true-NN distance, not the vote. */
struct VoteGraph {
    std::vector<i64> ptr;        /* n+1 */
    std::vector<i32> idx;        /* per vertex: kept neighbours, vote desc / id asc */
    std::vector<f32> vote;       /* aligned with idx */
    double t_build   = 0;
    i64    n_mentions = 0;       /* directed mentions emitted before reduction */
    i64    n_pairs    = 0;       /* distinct directed pairs after counting */
    i64    n() const { return (i64)ptr.size() - 1; }
    i64    deg(i64 v) const { return ptr[v + 1] - ptr[v]; }
};

/* build.cpp */
ForestGraph build_dense_graph(const f32 *X, i64 n, int d, int metric,
                              const BuildParams &bp);
ForestGraph build_forest_graph(const f32 *X, i64 n, int d, int metric,
                               const BuildParams &bp);

/* phase 1 only: the T parent[] arrays + roots, shared by the navigation build
 * and the vote aggregator so both see bitwise-identical trees. */
void build_parents(const f32 *X, i64 n, int d, int metric, const BuildParams &bp,
                   std::vector<std::vector<i32>> &parents, std::vector<i32> &roots,
                   i64 *n_dist, double *t_trees);

/* count parent<->child (+A-level ancestor) mentions across the forest into a
 * per-vertex vote-capped candidate list. Point-partitioned, lock-free,
 * deterministic (output keyed by vertex id; cap ties broken by id). */
VoteGraph aggregate_votes(const std::vector<std::vector<i32>> &parents,
                          const std::vector<i32> &roots, i64 n,
                          int A, f32 decay, int vote_cap, int threads, bool verbose);

/* seed NN-descent from the vote list (candidates fed vote-first) then refine
 * `rounds` rounds; B is the n*k refined true-NN code (ids, -1 padded). */
void refine_from_votes(const VoteGraph &acc, const f32 *X, i64 n, int d, int metric,
                       int k, int rounds, int threads,
                       std::vector<i32> &B, i64 *n_dist);

/* stepped variant for round-checkpoint sweeps: seed once, then run descent one
 * round at a time so the caller can measure recall at intermediate round counts
 * without rebuilding the forest. */
void vote_seed(const VoteGraph &acc, const f32 *X, i64 n, int d, int metric,
               int k, i32 *B, int threads, i64 *n_dist);
void vote_descent_round(const i32 *Bin, i32 *Bout, const f32 *X, i64 n, int d,
                        int metric, int k, int threads, i64 *n_dist);

/* wrap the recovery code (n*k id matrix) as a searchable graph: symmetrize the
 * directed k-NN edges to an undirected CSR, entry at the highest-degree hub. Lets
 * the data-tier code be beam-searched for the SYSTEM-level query recall (candidate
 * gen + exact re-rank), head-to-head with a graph index. */
ForestGraph code_to_graph(const i32 *B, int k, i64 n, int metric);
void bfs_reorder(ForestGraph &fg, std::vector<f32> &X, i64 n, int d,
                 std::vector<i32> &new2old);

/* search.cpp */
struct SearchResult {
    std::vector<i32> ids;     /* nq * k, -1 padded */
    std::vector<f32> dist;    /* nq * k */
    double seconds = 0;
    i64    n_dist  = 0;
};
/* Non-owning serve-time view of a built graph: exactly the state beam search
 * reads (adjacency + roots + metric). The mmap-able index format maps straight
 * into one of these -- position-independent, no rebuild, no copies. */
struct GraphView {
    const i64 *ptr = nullptr;    /* n+1 */
    const i32 *idx = nullptr;    /* E */
    const i32 *roots = nullptr;  /* n_roots */
    i64 n = 0, E = 0;
    int n_roots = 0, metric = METRIC_L2;
};
static inline GraphView graph_view(const ForestGraph &fg) {
    return GraphView{fg.g.ptr.data(), fg.g.idx.data(), fg.roots.data(),
                     fg.g.n(), (i64)fg.g.idx.size(), (int)fg.roots.size(),
                     fg.metric};
}

SearchResult beam_search(const GraphView &gv, const f32 *X, int d,
                         const f32 *Q, i64 nq, const i32 *entries, int n_entry,
                         int ef, int k, int threads);
SearchResult beam_search(const ForestGraph &fg, const f32 *X, i64 n, int d,
                         const f32 *Q, i64 nq, const i32 *entries, int n_entry,
                         int ef, int k, int threads);
void make_entries(const GraphView &gv, const f32 *X, int d,
                  const f32 *Q, i64 nq, const char *mode, int n_entry, u64 seed,
                  std::vector<i32> &entries);
void make_entries(const ForestGraph &fg, const f32 *X, i64 n, int d,
                  const f32 *Q, i64 nq, const char *mode, int n_entry, u64 seed,
                  std::vector<i32> &entries);

/* batched beam over packed sub-list graphs (local topology + local->global
 * id map into one global vector store); out_ids are global. search.cpp. */
i64 search_sublists(const i64 *PTR, const i64 *ptr_off,
                    const i32 *IDX, const i64 *idx_off,
                    const i32 *ROOTS, const i64 *root_off, const i32 *nroots,
                    const i32 *MAP, const i64 *map_off, const i64 *nloc,
                    const f32 *Xg, int d, int metric,
                    const f32 *Q, const i64 *pair_q, const i32 *pair_l,
                    i64 npairs, i64 max_nloc, int ef, int k, int threads,
                    i32 *out_ids, f32 *out_d);

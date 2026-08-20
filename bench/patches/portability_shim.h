#pragma once
// ------------------------------------------------------------------------
// Portability shim added to build PiPNN/ParlayANN on Apple silicon (arm64,
// macOS). No algorithmic change: it only supplies the Linux/x86 spellings of
// two platform facilities.
//
//  * MADV_HUGEPAGE  - Linux transparent-hugepage hint. macOS has no madvise
//                     equivalent, so it degrades to MADV_NORMAL (a no-op
//                     hint). NOTE: this is a real, measurable difference on
//                     Linux hardware; record it as a caveat when comparing.
//  * _mm_malloc/_mm_free - x86 aligned allocation, mapped to posix_memalign.
// ------------------------------------------------------------------------
#include <cstdlib>
#include <cstddef>
#include <sys/mman.h>

#ifndef MADV_HUGEPAGE
#define MADV_HUGEPAGE MADV_NORMAL
#endif

#if !defined(__x86_64__) && !defined(__i386__) && !defined(_M_X64)
static inline void *_mm_malloc(size_t size, size_t align) {
  void *p = nullptr;
  if (align < sizeof(void *)) align = sizeof(void *);
  if (posix_memalign(&p, align, size) != 0) return nullptr;
  return p;
}
static inline void _mm_free(void *p) { free(p); }
#endif

// C11 aligned_alloc requires `size` to be an exact multiple of `align`. glibc
// ignores that and returns a block anyway; macOS enforces it and returns NULL,
// which then segfaults at the first write. Several call sites here pass an
// unrounded size (the rounding line is commented out in point_range.h), so
// round up. posix_memalign blocks are free()-able, like aligned_alloc's.
static inline void *sat_aligned_alloc(size_t align, size_t size) {
  if (align < sizeof(void *)) align = sizeof(void *);
  size_t rounded = ((size + align - 1) / align) * align;
  void *p = nullptr;
  if (posix_memalign(&p, align, rounded) != 0) return nullptr;
  return p;
}
#define aligned_alloc(a, s) sat_aligned_alloc((a), (s))

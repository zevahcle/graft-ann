#!/usr/bin/env bash
# Copyright 2026 Edgar Chávez and contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# setup_baselines.sh -- fetch and build the PiPNN / ParlayANN baseline suite
# (PiPNN, Vamana/DiskANN, HCNNG, HNSW) next to the SAT forest, INCLUDING the
# five patches needed to make it build and run correctly on Apple silicon.
#
#   ./bench/setup_baselines.sh
#
# Upstream: https://github.com/ParAlg/PiPNN (MIT), the authors' implementation of
# "PiPNN: Ultra-Scalable Graph-Based Nearest Neighbor Indexing" (arXiv 2602.21247).
#
# ---------------------------------------------------------------------------
# THE PATCHES (all portability; none changes an algorithm)
#
#  P1  mirage out of the build. Its submodule is incomplete upstream (its own
#      googletest submodule has no .gitmodules entry), and we do not benchmark it.
#
#  P2  x86 intrinsic headers guarded. NSGDist.h includes <x86intrin.h> and
#      simhash.h / hash.h include <immintrin.h> unconditionally, but the code
#      under them is either scalar or already #ifdef __AVX__. Guard the includes.
#
#  P3  portability_shim.h: MADV_HUGEPAGE (Linux THP hint -> no-op on macOS),
#      _mm_malloc/_mm_free (-> posix_memalign), and an aligned_alloc wrapper that
#      rounds the size up to a multiple of the alignment. C11 requires that
#      multiple; glibc ignores it, macOS enforces it and returns NULL, which the
#      call sites do not check -> immediate segfault.
#
#  P4  ONE parlaylib. The repo vendors parlaylib as a submodule AND FetchContents
#      it at `master`, while per-directory `parlay` symlinks make quoted includes
#      resolve to the submodule. Two different parlaylib versions then land in one
#      binary -> ODR violation on parlay::scheduler. Pin FetchContent to the
#      submodule commit so both agree.
#
#  P5  work-stealing deque memory ordering (THE important one). The pinned
#      parlaylib's ABP deque does `age.load(relaxed)` then `bot.load(relaxed)` in
#      pop_top with no fence between them, while the owner side fences after its
#      stores. That Dekker pattern needs a fence on BOTH sides: on x86's TSO the
#      loads cannot be reordered so the bug is invisible, but on ARM they can, a
#      thief steals a job the owner already took, and the job runs twice ->
#      corrupted state and wild-pointer segfaults in every multi-threaded run
#      (PiPNN, Vamana and HCNNG alike). Upstream parlaylib master has since
#      strengthened these to acquire/seq_cst; we drop in that file.
# ---------------------------------------------------------------------------
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXT="$ROOT/external"
REPO="$EXT/PiPNN"
UPSTREAM_DEQUE="https://raw.githubusercontent.com/cmuparlay/parlaylib/master/include/parlay/internal/work_stealing_deque.h"
PARLAY_PIN="7cdb4cae8f020525f5eb4ad82e2565d1e38cfbc3"   # the vendored submodule commit

mkdir -p "$EXT"
if [ ! -d "$REPO/.git" ]; then
  echo "[baselines] cloning ParAlg/PiPNN"
  git clone --depth 1 https://github.com/ParAlg/PiPNN.git "$REPO"
fi
cd "$REPO"
git submodule update --init parlaylib external/hnswlib

echo "[baselines] P1: dropping mirage from the build"
sed -i '' 's|^add_subdirectory(external/mirage)|#add_subdirectory(external/mirage)  # PATCH P1|' CMakeLists.txt || true
sed -i '' 's|^add_subdirectory(mirage)|#add_subdirectory(mirage)  # PATCH P1|' algorithms/CMakeLists.txt || true

echo "[baselines] P2: guarding x86 intrinsic headers"
python3 - "$REPO" <<'PY'
import sys, os
base = os.path.join(sys.argv[1], 'algorithms', 'utils')
guard_open = '#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64)\n'
guard_close = '#endif  // PATCH P2: x86 intrinsics header, unused on other ISAs\n'
for fn, inc in [('NSGDist.h', '#include <x86intrin.h>'),
                ('simhash.h', '#include <immintrin.h>'),
                ('hash.h', '#include <immintrin.h>')]:
    p = os.path.join(base, fn)
    s = open(p).read()
    if 'PATCH P2' in s or inc not in s:
        continue
    open(p, 'w').write(s.replace(inc + '\n', guard_open + inc + '\n' + guard_close, 1))
    print('   guarded', fn)
PY

echo "[baselines] P3: portability shim (madvise / _mm_malloc / aligned_alloc)"
cp "$ROOT/bench/patches/portability_shim.h" algorithms/utils/portability_shim.h
python3 - "$REPO" <<'PY'
import sys, os
root = sys.argv[1]
targets = {'algorithms/utils/point_range.h': '"portability_shim.h"',
           'algorithms/utils/hash.h':        '"portability_shim.h"',
           'algorithms/utils/graph.h':       '"portability_shim.h"',
           'algorithms/PipNN/pipnn_utils.h': '"../utils/portability_shim.h"'}
for fn, inc in targets.items():
    p = os.path.join(root, fn)
    s = open(p).read()
    if 'portability_shim.h' in s:
        continue
    i = s.find('\n', s.find('#include'))
    open(p, 'w').write(s[:i + 1] + '#include ' + inc + '  // PATCH P3\n' + s[i + 1:])
    print('   shimmed', fn)
PY

echo "[baselines] P4: pinning FetchContent parlaylib to the submodule commit"
python3 - "$REPO" "$PARLAY_PIN" <<'PY'
import sys, os
p = os.path.join(sys.argv[1], 'CMakeLists.txt')
s = open(p).read()
old = "  GIT_TAG         master\n)"
new = "  GIT_TAG         %s  # PATCH P4: match the vendored submodule\n)" % sys.argv[2]
if 'PATCH P4' not in s and old in s:
    open(p, 'w').write(s.replace(old, new, 1))
    print('   pinned')
PY

echo "[baselines] configuring"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release > /dev/null

echo "[baselines] P5: upstream work-stealing deque (ARM memory ordering)"
TMP="$(mktemp)"
curl -sL -o "$TMP" "$UPSTREAM_DEQUE"
for D in "$REPO/parlaylib" "$REPO/build/_deps/parlaylib-src"; do
  cp "$TMP" "$D/include/parlay/internal/work_stealing_deque.h"
done
rm -f "$TMP"

echo "[baselines] building targets"
cmake --build build -j "$(sysctl -n hw.ncpu 2>/dev/null || nproc)" \
  --target neighbors-pipnn_FLOAT_T_EUCLIDEAN \
           neighbors-vamana_FLOAT_T_EUCLIDEAN \
           neighbors-hcnng_FLOAT_T_EUCLIDEAN \
           compute_groundtruth

echo "[baselines] OK -> $REPO/build/algorithms/*/neighbors-*"

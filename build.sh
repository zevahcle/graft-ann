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
# build.sh — one command to build the GRAFT C++ core (fg) and the HNSW baseline
# (hnsw_bench) from a clean clone, on Linux or macOS, with zero manual path-setting.
# It auto-fetches the header-only hnswlib dependency if absent.
#
#   ./build.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
CORE="$ROOT/src"
HNSWLIB_DIR="$ROOT/hnswlib"
HNSWLIB_URL="https://github.com/nmslib/hnswlib"

echo "[build] GRAFT core: $CORE"

# 1) third-party: header-only hnswlib (for the HNSW baseline). Auto-fetch if missing.
if [ ! -f "$HNSWLIB_DIR/hnswlib/hnswlib.h" ]; then
  echo "[build] hnswlib not found -> cloning $HNSWLIB_URL"
  git clone --depth 1 "$HNSWLIB_URL" "$HNSWLIB_DIR"
else
  echo "[build] hnswlib present"
fi

# 2) macOS parallel build needs libomp (Apple clang). Hint if absent; the Makefile
#    falls back to a serial build otherwise.
if [ "$(uname -s)" = "Darwin" ] && command -v brew >/dev/null 2>&1; then
  if ! brew --prefix libomp >/dev/null 2>&1; then
    echo "[build] note: libomp not found; 'brew install libomp' enables the parallel build."
  fi
fi

# 3) build both binaries.
( cd "$CORE" && make && make hnsw_bench )

echo
echo "[build] done:"
ls -la "$CORE/fg" "$CORE/hnsw_bench"
echo "[build] smoke test (synthetic determinism):"
"$CORE/fg" --synthetic clustered --n 20000 --d 32 --nq 200 --T 8 --ef 32 64 \
  --check-determinism 2>&1 | grep -iE "determinism|recall@10" | head -5
echo "[build] OK"

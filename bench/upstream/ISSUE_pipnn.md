# Multi-threaded index construction is unsafe on ARM64: vendored parlaylib predates the work-stealing-deque memory-ordering fix

*(Draft issue for https://github.com/ParAlg/PiPNN — the same pin question applies
to cmuparlay/ParlayANN.)*

Thanks for releasing PiPNN — being able to run the algorithm and the baselines
from one tree is a big help. While benchmarking on Apple silicon (M5, arm64,
macOS) we hit a reproducible failure that we think is worth reporting, plus a
handful of smaller portability items we had to patch to get the suite building.

## 1. The main issue: every multi-threaded build segfaults on ARM64

`neighbors-pipnn`, `neighbors-vamana` and `neighbors-hcnng` all crash within
seconds of starting a parallel build (`EXC_BAD_ACCESS`, wild pointers inside a
`parlay::parallel_for`). The same binaries run correctly with
`PARLAY_NUM_THREADS=1`.

The cause is not in PiPNN. It is the ABP work-stealing deque in the **vendored
parlaylib commit** (`7cdb4cae8f020525f5eb4ad82e2565d1e38cfbc3`), which reads
`age` and then `bot` in `pop_top()` with two `memory_order_relaxed` loads and no
fence between them:

```cpp
std::pair<Job*, bool> pop_top() {
  auto old_age = age.load(std::memory_order_relaxed);    // atomic load
  auto local_bot = bot.load(std::memory_order_relaxed);  // atomic load
  if (local_bot > old_age.top) { ...
```

The owner side of the handshake does fence (`pop_bottom` stores `bot` then
executes a `seq_cst` fence), but a Dekker-style protocol needs the ordering on
*both* sides. x86-64's TSO forbids load-load reordering, so the missing thief
side ordering is invisible there. ARM64 permits it: the thief can pair a `bot`
value from before the owner's decrement with a stale `age`, conclude the queue is
non-empty, and steal a job the owner has already popped. **The job then executes
twice**, which is exactly what the assertion at the top of `WorkStealingJob`'s
`operator()` catches when `NDEBUG` is off:

```
Assertion failed: (done.load(std::memory_order_relaxed) == false),
function operator(), file work_stealing_job.h, line 24.
```

In a Release build (`NDEBUG`) that assertion is compiled out, so the double
execution silently corrupts whatever the job was writing — hence the wild
pointers.

### Reproducer

`parlaylib_deque_repro.cpp` (attached) drives `parlay::internal::Deque` directly
with one owner and seven thieves. Every job object is used exactly once — no
reuse and no resets — so any double hand-out is unambiguous.

```
$ c++ -O2 -std=c++17 -I parlaylib/include parlaylib_deque_repro.cpp -o repro && ./repro
```

On an Apple M5 (10 cores), three runs each:

| parlaylib version | jobs | total hand-outs | double-handed-out |
|---|---:|---:|---:|
| pinned `7cdb4ca` | 1,600,000 | 1,600,020 / 1,600,026 / 1,600,012 | **18,902 / 19,369 / 20,756** |
| current upstream master | 1,600,000 | 1,600,000 (×3) | **0 / 0 / 0** |

### Fix

Current parlaylib master already fixes this — `pop_top` uses `acquire` loads and
`push_bottom` a `seq_cst` store. **Bumping the parlaylib submodule and the
`FetchContent_Declare(parlaylib ... GIT_TAG)` to current master is the whole
fix**; `parlaylib_deque_fix.patch` (attached) is the equivalent diff if you would
rather stay on the pinned commit.

With that one file replaced, PiPNN, Vamana and HCNNG all build and search
correctly at 10 threads on arm64. Measured afterwards on GloVe-100-angular
(1.18M points, 10 threads): PiPNN builds in 16.8 s (avg degree 30.6), Vamana
(R=100, L=200, α=1, 2 passes) in 269.3 s, HCNNG in 74.0 s.

Note this is a weak-memory-model issue, not a macOS one — it should equally
affect aarch64 Linux, POWER and RISC-V.

## 2. Two parlaylib versions can end up in one binary

The repo vendors parlaylib as a submodule *and* `FetchContent`s it at `master`,
while the per-directory `parlay` symlinks (`algorithms/bench/parlay`,
`data_tools/parlay`, …) make quoted includes resolve to the submodule. Files in
directories that have no such symlink (e.g. `algorithms/PipNN/`) resolve through
the CMake include path to the FetchContent copy instead. When the two commits
differ, one binary ends up with two definitions of `parlay::scheduler` — an ODR
violation. We saw it concretely: `scheduler::worker(unsigned long)` in one and
`scheduler::worker()` in the other. Pinning `FetchContent` to the submodule
commit (or dropping one of the two mechanisms) resolves it.

## 3. Smaller portability items (arm64 / macOS)

1. `algorithms/utils/NSGDist.h` includes `<x86intrin.h>` and
   `algorithms/utils/{simhash,hash}.h` include `<immintrin.h>` unconditionally,
   although the code beneath is scalar or already `#ifdef __AVX__`. Guarding the
   includes with `#if defined(__x86_64__) || defined(__i386__)` is enough.
2. `MADV_HUGEPAGE` is Linux-only (`graph.h`, `point_range.h`, `hash.h`).
3. `_mm_malloc` / `_mm_free` in `algorithms/PipNN/pipnn_utils.h` are x86
   spellings; `posix_memalign`/`free` work everywhere.
4. **`aligned_alloc` size contract** — worth fixing on all platforms. C11
   requires `size` to be an integral multiple of `alignment`. `point_range.h`
   passes an unrounded `total_bytes` with `ALIGNMENT = 1 << 21` (the rounding
   line is present but commented out), and `graph.h`/`hash.h` do the same. glibc
   tolerates it; macOS returns `NULL`, and the call sites do not check, so the
   first write segfaults. Restoring the round-up is a one-line change.

## 4. Minor observations

* `algorithms/PipNN/neighbors.h` returns early when `-graph_path` is given
  ("Just load the graph and search it. Leaving this as a stub for now."), so a
  saved PiPNN graph cannot currently be re-searched without rebuilding.
* `vec_to_bin` has a `data_tools/Makefile` rule but no CMake target.
* After a successful build + full beam sweep, `neighbors-pipnn` exits with
  SIGSEGV once the recall table has been printed. We have not diagnosed this one
  and it may well be another instance of item 3; flagging it in case it is
  already known.

Happy to open a PR with any subset of the above.

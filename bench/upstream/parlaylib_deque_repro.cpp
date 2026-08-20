// Minimal reproducer: parlay::internal::Deque loses mutual exclusion between the
// owner (pop_bottom) and thieves (pop_top) on weak memory models (ARM64), so a
// job is handed out twice.
//
// The ABP deque relies on a Dekker-style handshake. The owner stores `bot` and
// then executes a seq_cst fence; the thief must read `age` BEFORE it reads
// `bot`. In this version both thief loads are `memory_order_relaxed` with no
// fence between them, so ARM is free to reorder them -- the thief then sees a
// `bot` from before the owner's decrement together with a stale `age`, decides
// the queue is non-empty, and steals the job the owner has already popped.
// x86-64's TSO forbids load-load reordering, which is why this never shows up
// there.
//
// Every job object is used exactly once (no reuse, no resets), so any count
// above zero is a genuine double-hand-out and not a harness artefact.
//
//   c++ -O2 -std=c++17 -I<parlaylib>/include parlaylib_deque_repro.cpp -o repro
//   ./repro
#include "parlay/internal/work_stealing_deque.h"

#include <atomic>
#include <cstdio>
#include <thread>
#include <vector>

struct Job {
  std::atomic<int> taken{0};
};

int main() {
  constexpr int kThieves = 7;
  constexpr int kRounds = 200000;
  constexpr int kBatch = 8;

  parlay::internal::Deque<Job> dq;
  std::vector<Job> jobs(static_cast<size_t>(kRounds) * kBatch);  // each used once
  std::atomic<bool> stop{false};
  std::atomic<long> double_taken{0};
  std::atomic<long> thief_takes{0};
  std::atomic<long> owner_takes{0};

  auto take = [&](Job* j) {
    if (j && j->taken.fetch_add(1, std::memory_order_relaxed) != 0)
      double_taken.fetch_add(1, std::memory_order_relaxed);
  };

  std::vector<std::thread> thieves;
  for (int t = 0; t < kThieves; t++)
    thieves.emplace_back([&] {
      while (!stop.load(std::memory_order_relaxed)) {
        auto [job, empty] = dq.pop_top();
        if (job) {
          take(job);
          thief_takes.fetch_add(1, std::memory_order_relaxed);
        }
      }
    });

  for (long r = 0; r < kRounds; r++) {
    Job* base = jobs.data() + r * kBatch;
    for (int i = 0; i < kBatch; i++) dq.push_bottom(base + i);
    for (int i = 0; i < kBatch; i++) {
      Job* j = dq.pop_bottom();
      if (j) {
        take(j);
        owner_takes.fetch_add(1, std::memory_order_relaxed);
      }
    }
  }
  stop.store(true, std::memory_order_relaxed);
  for (auto& th : thieves) th.join();

  long total = owner_takes.load() + thief_takes.load();
  printf("rounds=%d jobs=%ld | owner takes=%ld thief takes=%ld total=%ld | "
         "DOUBLE-HANDED-OUT: %ld\n",
         kRounds, (long)jobs.size(), owner_takes.load(), thief_takes.load(),
         total, double_taken.load());
  return double_taken.load() != 0;
}

#!/usr/bin/env python3
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

"""parse_runs.py -- fg sweep log (any `### RUN <tag>` blocks) -> comparison table.

Build wall = trees + overlay + cap + refine. Distances/query at a target recall
are interpolated linearly between the two bracketing measured ef points; "-"
means the curve never reached that recall.

    python3 bench/parse_runs.py results/e2b/e2b_raw.log [--baseline results/e2a/e2a_raw.log]
"""
import argparse
import re

RUN = re.compile(r"### RUN (.+)")
FOREST = re.compile(r"forest: (\d+) SATs .* trees ([\d.]+)s .*, ([\d.]+)M dist")
OVERLAY = re.compile(r"overlay\+backlinks ([\d.]+)s")
CAPT = re.compile(r"cap>\d+ ([\d.]+)s")
REFINE = re.compile(r"refine \d+ rounds .*\) ([\d.]+)s")
EDGES = re.compile(r"edges: (\d+) directed \(([\d.]+)/vertex, max (\d+)\)")
COMPL = re.compile(r"local completeness \(k=10, sample\): ([\d.]+)")
EFROW = re.compile(r"^\s*(\d+)\s+([\d.]+)\s+(\d+)\s+(\d+)\s*$")
TARGETS = [0.95, 0.97, 0.985, 0.99]


def interp(points, target):
    for (r0, d0), (r1, d1) in zip(points, points[1:]):
        if r0 <= target <= r1:
            return d0 if r1 == r0 else d0 + (target - r0) / (r1 - r0) * (d1 - d0)
    return None


def parse(path):
    runs, cur = [], None
    for line in open(path):
        m = RUN.search(line)
        if m:
            cur = {"tag": m.group(1).strip(), "points": []}
            runs.append(cur)
            continue
        if cur is None:
            continue
        for rx, key in ((OVERLAY, "overlay_s"), (CAPT, "cap_s"),
                        (REFINE, "refine_s"), (COMPL, "completeness")):
            m = rx.search(line)
            if m:
                cur[key] = float(m.group(1))
        m = FOREST.search(line)
        if m:
            cur["trees_s"] = float(m.group(2))
            cur["build_Mdist"] = float(m.group(3))
        m = EDGES.search(line)
        if m:
            cur["avg_deg"] = float(m.group(2))
            cur["max_deg"] = int(m.group(3))
        m = EFROW.match(line)
        if m:
            cur["points"].append((float(m.group(2)), int(m.group(3))))
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    args = ap.parse_args()

    runs = []
    for p in args.logs:
        runs += parse(p)

    hdr = (f"{'config':24} {'build_s':>8} {'Mdist':>8} {'deg':>6} {'maxdeg':>7} "
           f"{'compl':>6} " + " ".join(f"{'d@' + str(t):>9}" for t in TARGETS))
    print(hdr)
    print("-" * len(hdr))
    for r in runs:
        build = sum(r.get(k, 0.0) for k in
                    ("trees_s", "overlay_s", "cap_s", "refine_s"))
        pts = sorted(r["points"])
        cells = []
        for t in TARGETS:
            v = interp(pts, t)
            cells.append(f"{v:9,.0f}" if v else f"{'-':>9}")
        print(f"{r['tag'][:24]:24} {build:8.2f} {r.get('build_Mdist', 0):8,.0f} "
              f"{r.get('avg_deg', 0):6.1f} {r.get('max_deg', 0):7d} "
              f"{r.get('completeness', 0):6.3f} " + " ".join(cells))

    print("\nreference (E1, same data/counter semantics):")
    print(f"{'vamana R100 L200 a1':24} {269.30:8.2f} {0:>8} {51.1:6.1f} "
          f"{100:7d} {0.605:6.3f} {9753:9,} {13508:9,} {20213:9,} {24813:9,}")
    print(f"{'pipnn 1-replica':24} {16.83:8.2f} {0:>8} {30.6:6.1f} "
          f"{64:7d} {0.406:6.3f} {12110:9,} {17414:9,} {'-':>9} {'-':>9}")


if __name__ == "__main__":
    main()

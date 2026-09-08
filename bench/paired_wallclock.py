#!/usr/bin/env python3
"""Paired, repeated build wall-clock on one machine: GRAFT (fg) vs Vamana
and PiPNN (ParlayANN/PiPNN binaries from bench/setup_baselines.sh), the
laptop A/B of the paper's sec:wallclock done with proper statistics
(Information Systems revision, editor comment 1).

Design. For each dataset, B blocks; inside a block every system is built
exactly once, in a seeded random order; the block is the pairing unit, so
thermal drift that is slow relative to a block is shared by all systems of
the block and cancels in the paired (log-)differences. A cool-down pause
of COOL seconds separates builds. Seeds are fixed, so GRAFT's graph is
bitwise identical in every block (hash recorded). The reported build time
is each system's own build timer (fg: trees + overlay + harvest; ParlayANN:
"Graph built in X seconds"), i.e. excluding data loading; process wall is
recorded too. The results JSON is rewritten after every block. Statistics
(median, IQR, Hodges-Lehmann ratio with exact CI, Wilcoxon signed-rank,
sign test) are computed by misifu's misi2/paired_stats.py.

    DATA=~/code/claude/sat-forest/data PIPNN=~/code/claude/sat-forest/external/PiPNN \
    B=8 COOL=45 DS=sift,glove python3 bench/paired_wallclock.py

Run with the laptop otherwise idle, on mains power, lid open.
"""
import json
import os
import pathlib
import random
import re
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
FG = os.environ.get("FG", str(ROOT / "src" / "fg"))
DATA = pathlib.Path(os.path.expanduser(os.environ.get("DATA", "~/code/claude/sat-forest/data")))
NPY = pathlib.Path(os.path.expanduser(os.environ.get("NPY", "~/code/claude/proximity-bench/SAT forest graph")))
PIPNN = pathlib.Path(os.path.expanduser(os.environ.get("PIPNN", "~/code/claude/sat-forest/external/PiPNN")))
B = int(os.environ.get("B", "8"))
COOL = float(os.environ.get("COOL", "45"))
DS = os.environ.get("DS", "sift,glove").split(",")
THREADS = int(os.environ.get("THREADS", str(os.cpu_count())))
SEED = int(os.environ.get("SEED", "1"))
OUT = pathlib.Path(os.environ.get("OUT", str(ROOT / "results" / "paired_wallclock")))
VAMANA = PIPNN / "build/algorithms/vamana/neighbors-vamana_FLOAT_T_EUCLIDEAN"
PIPNNB = PIPNN / "build/algorithms/PipNN/neighbors-pipnn_FLOAT_T_EUCLIDEAN"


def npy(name, kind):
    """fg inputs: X/Q/gold .npy — sat-forest/data first, then the proximity-bench dir."""
    for base in (DATA, NPY):
        for cand in (base / f"{name}_{kind}.npy", base / f"{name}_norm_{kind}.npy"):
            if cand.exists():
                return str(cand)
    raise FileNotFoundError(f"{name}_{kind}.npy")


def fg_cmd(name, metric, T, ef, patience=0):
    c = [FG, "--data", npy(name, "X"), "--queries", npy(name, "Q"), "--gold", npy(name, "gold"),
         "--metric", metric, "--T", str(T), "--threads", str(THREADS), "--harvest", str(ef),
         "--harvest-cap", "64", "--seed", str(SEED), "--ef", "100"]
    if patience:
        c += ["--harvest-patience", str(patience)]
    return c


def parlay_cmd(binary, name, R, L, alpha, two_pass, extra=()):
    return [str(binary), "-base_path", str(DATA / f"{name}_base.fbin"), "-query_path", str(DATA / f"{name}_query.fbin"),
            "-gt_path", str(DATA / f"{name}_gt100"), "-file_type", "bin", "-data_type", "float",
            "-dist_func", "Euclidian", "-R", str(R), "-L", str(L), "-alpha", str(alpha),
            "-two_pass", str(two_pass), "-k", "10", *extra]


# The paper's profiles and the baselines' published per-dataset parameters (sec:protocol).
SYSTEMS = {
    "fashion": {  # smoke test only (2-3 s builds)
        "graft-T2-ef200": fg_cmd("fashion", "l2", 2, 200),
        "vamana-R40-L80-a1.1": parlay_cmd(VAMANA, "fashion", 40, 80, 1.1, 0),
        "pipnn": parlay_cmd(PIPNNB, "fashion", 40, 80, 1.1, 0),
    },
    "sift": {
        "graft-T4-ef400": fg_cmd("sift", "l2", 4, 400),
        "graft-T16-ef400": fg_cmd("sift", "l2", 16, 400),
        "vamana-R64-L128-a1.15-2pass": parlay_cmd(VAMANA, "sift", 64, 128, 1.15, 1),
        "pipnn": parlay_cmd(PIPNNB, "sift", 64, 128, 1.15, 0),
    },
    "glove": {
        "graft-T16-ef400": fg_cmd("glove", "cosine", 16, 400),
        "graft-T32-ef600": fg_cmd("glove", "cosine", 32, 600),
        "vamana-R100-L200-a1.0-2pass": parlay_cmd(VAMANA, "glove", 100, 200, 1.0, 1),
        "pipnn": parlay_cmd(PIPNNB, "glove", 100, 200, 1.0, 0),
    },
}

RE_FG_TREES = re.compile(r"trees ([0-9.]+)s")
RE_FG_OVER = re.compile(r"overlay\+backlinks ([0-9.]+)s")
RE_FG_HARV = re.compile(r"harvest ef=\d+.*? ([0-9.]+)s")
RE_FG_HASH = re.compile(r"hash[ =:]+([0-9a-fx]+)", re.I)
RE_PARLAY = re.compile(r"Graph built in ([0-9.]+) seconds")


def therm():
    """macOS thermal covariate without sudo (pmset -g therm): CPU_Speed_Limit if the OS reports one."""
    try:
        o = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True, timeout=5).stdout
        m = re.search(r"CPU_Speed_Limit\s*=\s*(\d+)", o)
        return int(m.group(1)) if m else None
    except Exception:
        return None


def one_build(label, cmd):
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    out = p.stdout + p.stderr
    rec = {"system": label, "process_wall_s": round(wall, 2), "rc": p.returncode}
    if label.startswith("graft"):
        tr, ov, hv = (RE_FG_TREES.search(out), RE_FG_OVER.search(out), RE_FG_HARV.search(out))
        if tr and hv:
            rec["trees_s"] = float(tr.group(1)); rec["overlay_s"] = float(ov.group(1)) if ov else 0.0
            rec["harvest_s"] = float(hv.group(1))
            rec["build_s"] = round(rec["trees_s"] + rec["overlay_s"] + rec["harvest_s"], 3)
        h = RE_FG_HASH.search(out)
        if h:
            rec["hash"] = h.group(1)
    else:
        m = RE_PARLAY.search(out)
        if m:
            rec["build_s"] = float(m.group(1))
    if "build_s" not in rec:
        rec["error"] = out[-2000:]
    return rec


def run(name):
    systems = SYSTEMS[name]
    rng = random.Random(SEED * 1000 + len(name))
    rows = []
    OUT.mkdir(parents=True, exist_ok=True)
    outf = OUT / f"paired-{name}-s{SEED}.json"
    print(f"[{name}] systems={list(systems)} blocks={B} cool={COOL}s threads={THREADS}", flush=True)
    for b in range(B):
        order = list(systems); rng.shuffle(order)
        for pos, label in enumerate(order):
            if COOL and (b or pos):
                time.sleep(COOL)
            rec = one_build(label, systems[label])
            rec.update({"block": b, "pos": pos, "t": time.time(), "cpu_speed_limit": therm()})
            rows.append(rec)
            print(f"[{name}] block {b} pos {pos} {label:30s} build {rec.get('build_s', float('nan')):8.1f} s "
                  f"(process {rec['process_wall_s']:.0f} s){'  ERROR' if 'error' in rec else ''}", flush=True)
            if "error" in rec:
                print(rec["error"][-600:], flush=True)
        json.dump({"dataset": {"name": name}, "config": {"seed": SEED, "blocks_planned": B, "blocks_done": b + 1,
                   "cool_s": COOL, "threads": THREADS, "systems": {k: v for k, v in systems.items()}},
                   "results": {"rows": rows}, "notes": "laptop, otherwise idle; block = pairing unit, random order within block"},
                  open(outf, "w"), indent=1)
    print(f"[{name}] wrote {outf}", flush=True)


if __name__ == "__main__":
    for ds in DS:
        run(ds)
    print("DONE", flush=True)

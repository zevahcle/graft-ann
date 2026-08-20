#!/usr/bin/env bash
# P4 -- Wikipedia BGE-M3 6.35M (SISAP): GRAFT vs Vamana/PiPNN on the included
# exhaustive GT. Convert h5 (float16, 1-BASED knns) -> npy first.
set -euo pipefail
cd "$(dirname "$0")/../.."
FG="./src/fg"; THREADS="${THREADS:-64}"
H5="${WIKI_H5:?set WIKI_H5 to benchmark-dev-wikipedia-bge-m3.h5}"
mkdir -p results/box data
python3 - "$H5" <<'PY'
import sys, h5py, numpy as np
with h5py.File(sys.argv[1], "r") as f:
    n = f["train"].shape[0]
    X = np.empty((n, 1024), dtype=np.float32)
    for i in range(0, n, 500000):
        X[i:i+500000] = f["train"][i:i+500000]
    np.save("data/wikifull_X.npy", X); del X
    np.save("data/wikifull_Q.npy", np.asarray(f["otest/queries"], dtype=np.float32))
    np.save("data/wikifull_gold.npy",
            (np.asarray(f["otest/knns"][:, :100], dtype=np.int64) - 1).astype(np.int32))
print("converted")
PY
{ for T in 8 16; do
    echo "### RUN wikifull_T${T}_ef400"
    "$FG" --data data/wikifull_X.npy --queries data/wikifull_Q.npy \
          --gold data/wikifull_gold.npy --metric l2 --T $T --threads "$THREADS" \
          --harvest 400 --harvest-cap 64 --max-degree 128 \
          --ef 10 20 30 50 80 120 200
    echo "### END"
  done
} 2>&1 | tee results/box/p4_wiki_graft.log
# Baselines (if ParlayANN built): npy_to_fbin + gold_to_gt, then
# vamana -R 64 -L 128 -alpha 1.0 -num_passes 2 and pipnn defaults.

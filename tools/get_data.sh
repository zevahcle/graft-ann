#!/usr/bin/env bash
# Copyright 2026 Edgar Chávez and contributors
#
# Licensed under the Apache License, Version 2.0 (the "License"); see LICENSE.
#
# get_data.sh — download a big-ann-benchmarks dataset (base + query + ground truth) onto the
# a large scratch volume. Uses the official big-ann-benchmarks downloader, which handles the
# current URLs AND ships the slice-specific ground truth (so a 10M/100M slice has correct GT —
# no need to recompute). Files land as .fbin/.u8bin/.i8bin + .ibin: feed the RAW files to DiskANN,
# or convert to .npy for GRAFT with tools/convert_bigann.py.
#
# Usage:
#   tools/get_data.sh deep-10M         # small, float32, no dtype work — recommended first
#   tools/get_data.sh deep-100M        # float32, ~38 GB base
#   tools/get_data.sh bigann-100M      # uint8, 128-dim (needs GRAFT uint8 path; storage-light)
#   tools/get_data.sh deep-1B          # float32, ~384 GB base
#   tools/get_data.sh bigann-1B        # uint8, ~128 GB base (storage-friendlier at 1B)
# Valid names: {bigann,deep,msturing,msspacev,text2image}-{10M,100M,1B}. Confirm with:
#   (in big-ann-benchmarks) python3 create_dataset.py --help
set -euo pipefail
ROOT="${GRAFT_DATA:-data}"
DS="${1:?usage: get_data.sh <dataset, e.g. deep-100M|bigann-100M|deep-1B>}"
BAB="$ROOT/big-ann-benchmarks"

df -h "$ROOT" || { echo "ERROR: $ROOT not mounted"; exit 1; }
mkdir -p "$ROOT"
[ -d "$BAB" ] || git clone https://github.com/harsha-simhadri/big-ann-benchmarks "$BAB"
cd "$BAB"
python3 -m pip install -r requirements.txt >/dev/null 2>&1 || \
  python3 -m pip install numpy >/dev/null 2>&1 || true

echo "[get_data] downloading '$DS' into $BAB/data (this is large — watch $ROOT free space)"
python3 create_dataset.py --dataset "$DS"

echo "[get_data] done. dataset files:"
find "$BAB/data" -maxdepth 2 -type f \( -name '*.fbin' -o -name '*.u8bin' -o -name '*.i8bin' -o -name '*.ibin' \) -printf '  %p  (%s bytes)\n' 2>/dev/null | head
echo "[get_data] next:"
echo "  DiskANN  : point --data_path / --query_file / --gt_file at those raw files directly."
echo "  GRAFT  : python3 tools/convert_bigann.py --base <base> --query <query> --gt <gt> \\"
echo "               --out $ROOT/data/${DS}   (skip --n to use the whole set; add --n 100000000 to slice)"
echo "  NOTE     : at 1B do NOT convert to .npy (doubles storage) — Phase 4 mmaps the raw file."

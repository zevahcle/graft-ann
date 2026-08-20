#!/usr/bin/env bash
# P0 -- build, gate, cross-machine statistical repro.
# Laptop reference (GloVe T32/ef600, 10 thr): d@0.95=9,751 d@0.97=13,589
# deg 42; recall curve ef100..2400 = .876/.923/.958/.973/.989/.997 (approx).
# Match recall/distances to ~1-2% (hash will NOT match across ISA; expected).
set -euo pipefail
cd "$(dirname "$0")/../.."
./build.sh
FG="./src/fg"
mkdir -p results/box
"$FG" --synthetic clustered --n 200000 --d 128 --T 16 --ef 32 \
      --check-determinism | tee results/box/p0_gate.log
if [ -f "${GLOVE_X:-data/glove_X.npy}" ]; then
  "$FG" --data "${GLOVE_X:-data/glove_X.npy}" --queries "${GLOVE_Q:-data/glove_Q.npy}" \
        --gold "${GLOVE_G:-data/glove_gold.npy}" --metric cos --T 32 --threads 64 \
        --harvest 600 --harvest-cap 64 --ef 100 200 400 600 1200 2400 \
      | tee results/box/p0_glove_repro.log
else
  echo "GloVe npy not staged; statistical repro deferred to P5" \
      | tee results/box/p0_glove_repro.log
fi

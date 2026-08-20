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
# sweep.sh -- the (T, harvest-ef) build-budget frontier on one dataset.
#
#   DATA=glove_X.npy QUERIES=glove_Q.npy GOLD=glove_gold.npy \
#   THREADS=10 ./sweep.sh | tee sweep.csv
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
THREADS="${THREADS:-10}"
EFS="${EFS:-100 200 400 600 1200 2400}"
for CFG in "16 400" "24 600" "32 600"; do
  set -- $CFG
  echo "### RUN T$1_ef$2"
  "$HERE/fg" --data "$DATA" --queries "$QUERIES" --gold "$GOLD" \
      --metric "${METRIC:-cos}" --T "$1" --harvest "$2" \
      --threads "$THREADS" --ef $EFS
  echo "### END T$1_ef$2"
done

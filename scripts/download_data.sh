#!/usr/bin/env bash
# Download BIG IDEAs participants from PhysioNet (open access, ODC-By 1.0).
# The full dataset is 34.1 GB; this skips BVP (about 1.3 GB per person, unused) and IBI.
# Each participant is roughly 1 GB, most of it the accelerometer file.
#
#   scripts/download_data.sh 001 002
set -euo pipefail

BASE="https://physionet.org/files/big-ideas-glycemic-wearable/1.1.3"
DEST="$(cd "$(dirname "$0")/.." && pwd)/data/raw"
mkdir -p "$DEST"

if [ "$#" -eq 0 ]; then
  echo "usage: $0 <pid> [<pid> ...]   e.g. $0 001" >&2
  exit 1
fi

curl -fsSL -C - -o "$DEST/Demographics.csv" "$BASE/Demographics.csv"
for pid in "$@"; do
  mkdir -p "$DEST/$pid"
  for f in "Dexcom_$pid.csv" "Food_Log_$pid.csv" "HR_$pid.csv" "EDA_$pid.csv" "TEMP_$pid.csv" "ACC_$pid.csv"; do
    echo "downloading $pid/$f"
    curl -fL -C - --retry 3 -o "$DEST/$pid/$f" "$BASE/$pid/$f"
  done
done
echo "done: $DEST"

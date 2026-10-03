#!/usr/bin/env bash
# Download BIG IDEAs participants (open access, ODC-By 1.0), about 5 minutes each instead of hours.
#
# PhysioNet's own server is slow, so the large wristband files (ACC, EDA, TEMP, about 1 GB per person) come from
# PhysioNet's Amazon S3 mirror. That mirror holds version 1.0.0, whose ACC/EDA/TEMP files are byte-for-byte the same
# size as version 1.1.3 for all 16 participants. HR, Dexcom and the food log must come from 1.1.3 (the 1.0.0 copies
# have wrong HR dates and no food logs), so those small files are fetched from PhysioNet. BVP and IBI are not used.
#
#   scripts/download_data.sh 001 002
set -euo pipefail

PHYSIONET="https://physionet.org/files/big-ideas-glycemic-wearable/1.1.3"
MIRROR="https://physionet-open.s3.amazonaws.com/big-ideas-glycemic-wearable/1.0.0"
DEST="$(cd "$(dirname "$0")/.." && pwd)/data/raw"
mkdir -p "$DEST"

if [ "$#" -eq 0 ]; then
  echo "usage: $0 <pid> [<pid> ...]   e.g. $0 001" >&2
  exit 1
fi

fetch() {  # fetch <url> <dest>; skips finished files, resumes partial ones
  local remote local_size
  remote=$(curl -fsIL "$1" | grep -i '^content-length' | tail -1 | tr -dc '0-9') || return 1
  if [ -f "$2" ]; then
    local_size=$(wc -c < "$2" | tr -d ' ')
    if [ -n "$remote" ] && [ "$local_size" = "$remote" ]; then
      echo "  already downloaded"
      return 0
    fi
  fi
  curl -fL -C - --retry 3 -o "$2" "$1"
}

fetch "$PHYSIONET/Demographics.csv" "$DEST/Demographics.csv"
for pid in "$@"; do
  mkdir -p "$DEST/$pid"
  for f in "ACC_$pid.csv" "EDA_$pid.csv" "TEMP_$pid.csv"; do
    echo "downloading $pid/$f (fast mirror)"
    fetch "$MIRROR/$pid/$f" "$DEST/$pid/$f" || { echo "mirror failed, using PhysioNet"; fetch "$PHYSIONET/$pid/$f" "$DEST/$pid/$f"; }
  done
  for f in "HR_$pid.csv" "Dexcom_$pid.csv" "Food_Log_$pid.csv"; do
    echo "downloading $pid/$f (PhysioNet 1.1.3)"
    fetch "$PHYSIONET/$pid/$f" "$DEST/$pid/$f"
  done
done
echo "done: $DEST"

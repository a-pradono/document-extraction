#!/usr/bin/env bash
# Download the three raw datasets into data/raw/. Idempotent: re-running skips
# anything already materialized. Nothing here is committed — data/raw/ is gitignored;
# only the ~65 curated docs the converters emit under data/docs/ go into git.
#
#   ./scripts/fetch_datasets.sh            # all three
#   ./scripts/fetch_datasets.sh cord       # one source
set -euo pipefail

cd "$(dirname "$0")/.."
RAW="data/raw"
mkdir -p "$RAW"

die() { echo "ERROR: $*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

# curl and wget disagree on flags; wrap the one that exists. Both resume partial files.
download() { # download <url> <dest>
  local url="$1" dest="$2"
  if have curl; then curl -fL -C - --retry 3 -o "$dest" "$url"
  elif have wget; then wget -nv -c -O "$dest" "$url"
  else die "need curl or wget on PATH"; fi
}

count_files() { find "$1" -type f 2>/dev/null | wc -l | tr -d ' '; }

# --- 1. CORD v2 -------------------------------------------------------------
# HF mirror, no auth. Materialize the test split to plain files so convert_cord.py
# needs no `datasets` dependency and re-conversion works offline.
fetch_cord() {
  local dir="$RAW/cord"
  if [ -f "$dir/ground_truth.jsonl" ] && [ "$(count_files "$dir/images")" -ge 100 ]; then
    echo "[cord] already present ($(count_files "$dir/images") images) — skipping"
    return
  fi
  python -c "import datasets" 2>/dev/null || die "pip install datasets  (needed for CORD)"
  mkdir -p "$dir/images"
  python - "$dir" <<'PY'
import json, sys
from pathlib import Path
from datasets import load_dataset

out = Path(sys.argv[1])
ds = load_dataset("naver-clova-ix/cord-v2", split="test")
with (out / "ground_truth.jsonl").open("w", encoding="utf-8") as fh:
    for i, row in enumerate(ds):
        # JPEG, not PNG: these are photographs, and PNG-encoding them costs ~2 MB each
        # (50 MB for the curated 25) for no visible gain on a document we downscale to
        # 1600px before it ever reaches a model.
        name = f"cord_test_{i:04d}.jpg"
        row["image"].convert("RGB").save(out / "images" / name, "JPEG", quality=92)
        fh.write(json.dumps({"file_name": name, "ground_truth": row["ground_truth"]}) + "\n")
print(f"[cord] wrote {len(ds)} images + ground_truth.jsonl")
PY
  [ "$(count_files "$dir/images")" -ge 100 ] || die "[cord] expected >=100 images, got $(count_files "$dir/images")"
}

# --- 2. WildReceipt ---------------------------------------------------------
# Direct tarball from openmmlab, no auth.
fetch_wildreceipt() {
  local dir="$RAW/wildreceipt"
  if [ -f "$dir/test.txt" ] && [ -f "$dir/class_list.txt" ]; then
    echo "[wildreceipt] already present — skipping"
    return
  fi
  local tarball="$RAW/wildreceipt.tar"
  echo "[wildreceipt] downloading (~1.2 GB)…"
  download "https://download.openmmlab.com/mmocr/data/wildreceipt.tar" "$tarball"
  tar -xf "$tarball" --directory "$RAW"
  rm -f "$tarball"
  [ -f "$dir/test.txt" ] || die "[wildreceipt] test.txt missing after extraction"
  local n; n=$(count_files "$dir/image_files")
  [ "$n" -ge 1700 ] || die "[wildreceipt] expected ~1740 images, got $n — partial download"
  echo "[wildreceipt] ok ($n images)"
}

targets=("${@:-cord wildreceipt}")
status=0
for t in ${targets[@]}; do
  case "$t" in
    cord)        fetch_cord ;;
    wildreceipt) fetch_wildreceipt ;;
    *)           die "unknown source '$t' (cord|wildreceipt)" ;;
  esac
done

echo
echo "Raw data in $RAW (gitignored). Next: scripts/convert_*.py"
exit $status

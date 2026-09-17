"""OCR Receipts (retail, Kaggle) -> Receipt ground truth.

    python scripts/convert_retail.py [--n 15]

The messiest of the three tiers: photographed retail receipts. The dataset annotates
roughly item / store / date_time / total and has NO subtotal and NO tax. Those two are
exactly what the `subtotal + tax == total` validation signal depends on, so they are
hand-labeled from the images into:

    data/hand_labels/retail.json
      { "retail_0001": {"subtotal": "12.40", "tax": "0.87", "currency": "USD",
                        "payment_method": "VISA"}, ... }

Run the converter once to emit the documents and a stub of that file, hand-label from
the images (scripts/verify_gt.py renders them side by side), then re-run: the second
pass merges the hand labels in and records them as hand-labeled in _provenance.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from gt_common import (
    REPO,
    clear_source,
    normalize_money,
    normalize_qty,
    provenance_entry,
    prune,
    stratified_sample,
    update_provenance,
    write_document,
)

RAW = REPO / "data" / "raw" / "ocr-receipts-text-detection"
HAND_LABELS = REPO / "data" / "hand_labels" / "retail.json"
SOURCE = "OCR Receipts text detection (Kaggle: trainingdatapro/ocr-receipts-text-detection)"

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}
# The dataset's own label vocabulary -> our schema. Keys are matched case-insensitively
# and loosely, because this export has never been stable about naming.
LABEL_MAP = {
    "item": "item",
    "product": "item",
    "store": "merchant_name",
    "store_name": "merchant_name",
    "merchant": "merchant_name",
    "address": "merchant_address",
    "date_time": "date",
    "datetime": "date",
    "date": "date",
    "total": "total",
    "grand_total": "total",
    "price": "price",
    "quantity": "quantity",
}
# Fields this dataset does not annotate at all. Hand-labeled, never guessed.
HAND_FIELDS = ["subtotal", "tax", "currency", "payment_method"]


def _find_annotation_file() -> Path:
    candidates = [p for p in RAW.rglob("*")
                  if p.suffix.lower() in {".csv", ".json", ".jsonl"} and p.is_file()]
    if not candidates:
        raise SystemExit(
            f"no .csv/.json annotation file under {RAW}.\n"
            "Contents:\n  " + "\n  ".join(sorted(str(p.relative_to(RAW))
                                                 for p in RAW.rglob("*"))[:40])
        )
    # Prefer the largest — the per-image annotation dump, not a README-ish sidecar.
    return max(candidates, key=lambda p: p.stat().st_size)


def _load_rows(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() == ".csv":
        return list(csv.DictReader(text.splitlines()))
    if path.suffix.lower() == ".jsonl":
        return [json.loads(ln) for ln in text.splitlines() if ln.strip()]
    data = json.loads(text)
    if isinstance(data, list):
        return data
    for value in data.values():                 # {"annotations": [...]}
        if isinstance(value, list):
            return value
    raise SystemExit(f"unrecognized JSON shape in {path}")


def _image_index() -> dict[str, Path]:
    return {p.name.lower(): p for p in RAW.rglob("*") if p.suffix.lower() in IMAGE_EXT}


def _field_for(label: str) -> str | None:
    key = re.sub(r"[^a-z_]", "", str(label).lower().replace(" ", "_"))
    return LABEL_MAP.get(key)


def _pick(row: dict, *names: str):
    for name in names:
        for key in row:
            if key.lower() == name:
                return row[key]
    return None


def _parse_date(raw: str) -> str | None:
    m = re.search(r"\b(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})\b", str(raw))
    if m:
        y, mo, d = (int(g) for g in m.groups())
    else:
        m = re.search(r"\b(\d{1,2})[-/.](\d{1,2})[-/.](\d{2,4})\b", str(raw))
        if not m:
            return None
        mo, d, y = (int(g) for g in m.groups())
        y += 2000 if y < 100 else 0
        if mo > 12:
            return None  # ambiguous DD/MM — leave for the human
    try:
        from datetime import date as date_cls
        return date_cls(y, mo, d).isoformat()
    except ValueError:
        return None


def _group_by_image(rows: list[dict]) -> dict[str, dict]:
    docs: dict[str, dict] = {}
    for row in rows:
        image = _pick(row, "file_name", "filename", "image", "image_name", "img")
        if not image:
            continue
        image = Path(str(image)).name.lower()
        doc = docs.setdefault(image, {"line_items": [], "scalars": {}})
        text = str(_pick(row, "text", "value", "transcription", "ocr") or "").strip()
        field = _field_for(_pick(row, "label", "class", "category", "tag") or "")
        if not field or not text:
            continue
        if field == "item":
            doc["line_items"].append(prune({
                "description": text,
                "quantity": normalize_qty(_pick(row, "quantity", "qty")),
                "line_total": normalize_money(_pick(row, "price", "amount")),
            }))
        elif field in {"price", "quantity"}:
            continue  # only meaningful attached to an item row
        else:
            doc["scalars"].setdefault(field, text)
    return docs


def _to_receipt(doc: dict) -> dict:
    scalars = doc["scalars"]
    return prune({
        "merchant_name": scalars.get("merchant_name"),
        "merchant_address": scalars.get("merchant_address"),
        "date": _parse_date(scalars["date"]) if scalars.get("date") else None,
        "total": normalize_money(scalars.get("total")),
        "line_items": doc["line_items"],
    })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15)
    args = ap.parse_args()

    if not RAW.exists():
        raise SystemExit(
            f"{RAW} missing. This source needs a Kaggle API token:\n"
            "  ./scripts/fetch_datasets.sh retail   (prints the manual steps)"
        )

    rows = _load_rows(_find_annotation_file())
    images = _image_index()
    docs = {k: v for k, v in _group_by_image(rows).items() if k in images}

    converted = {k: _to_receipt(v) for k, v in docs.items()}
    pool = [(k, len(r.get("line_items", [])), (r.get("merchant_name") or k).lower())
            for k, r in converted.items() if r.get("line_items")]
    chosen = stratified_sample(pool, args.n)
    if len(chosen) < args.n:
        raise SystemExit(f"only {len(chosen)} usable retail docs for --n {args.n}")

    hand = json.loads(HAND_LABELS.read_text(encoding="utf-8")) if HAND_LABELS.exists() else {}

    clear_source("retail")
    entries, stub = {}, {}
    for i, name in enumerate(sorted(chosen), start=1):
        doc_id = f"retail_{i:04d}"
        receipt = dict(converted[name])
        labels = hand.get(doc_id) or {}
        applied = []
        for field in HAND_FIELDS:
            value = labels.get(field)
            if value in (None, ""):
                continue
            receipt[field] = normalize_money(value) if field in {"subtotal", "tax"} else value
            applied.append(field)
        write_document(doc_id, images[name], prune(receipt))

        entries[doc_id] = provenance_entry(
            SOURCE,
            name,
            script_fields=[k for k in converted[name]],
            hand_fields=applied,
            needs_hand_label=[f for f in HAND_FIELDS if f not in applied],
        )
        stub[doc_id] = labels or dict.fromkeys(HAND_FIELDS, None)

    update_provenance(entries)
    HAND_LABELS.parent.mkdir(parents=True, exist_ok=True)
    HAND_LABELS.write_text(json.dumps(stub, indent=2) + "\n", encoding="utf-8")

    pending = sum(1 for e in entries.values() if e["needs_hand_label"])
    print(f"wrote {len(chosen)} retail docs.")
    if pending:
        print(f"{pending} still need hand labels (subtotal/tax are NOT in this dataset).\n"
              f"  1. python scripts/verify_gt.py --set retail\n"
              f"  2. fill {HAND_LABELS.relative_to(REPO)}\n"
              f"  3. re-run this script to merge them in")


if __name__ == "__main__":
    main()

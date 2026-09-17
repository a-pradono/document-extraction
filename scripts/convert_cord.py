"""CORD v2 -> Receipt ground truth.

    python scripts/convert_cord.py [--n 25]

Reads data/raw/cord/ (materialized by scripts/fetch_datasets.sh) and writes 25
stratified documents as data/docs/cord_XXXX.png + data/ground_truth/cord_XXXX.json.

CORD's gt_parse gives menu lines, sub_total and total. It does NOT annotate the store
name, the date, or the payment method, so those stay null here and are listed under
`needs_hand_label` in _provenance.json — they are filled during verification, from the
image, or left null if the receipt genuinely does not show them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gt_common import (
    REPO,
    carry_verification,
    clear_source,
    pinned_selection,
    normalize_money,
    normalize_qty,
    provenance_entry,
    prune,
    stratified_sample,
    update_provenance,
    write_document,
)

RAW = REPO / "data" / "raw" / "cord"
SOURCE = "CORD v2 (naver-clova-ix/cord-v2, test split)"
# gt_parse covers these; everything else on the Receipt schema is unannotated by CORD.
SCRIPT_FIELDS = ["subtotal", "service_charge", "discount", "tax", "total", "line_items"]
UNANNOTATED = ["merchant_name", "merchant_address", "date", "currency", "payment_method"]


def _as_list(value) -> list:
    """CORD collapses a single-entry menu to a bare dict."""
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _row(entry: dict) -> dict:
    return prune(
        {
            "description": (entry.get("nm") or "").strip() or None,
            "quantity": normalize_qty(entry.get("cnt")),
            "unit_price": normalize_money(entry.get("unitprice")),
            "line_total": normalize_money(entry.get("price")),
        }
    )


def _line_items(parse: dict) -> list[dict]:
    """One row per printed line, INCLUDING nested add-ons.

    CORD nests add-ons under their parent item as `sub` ("Kupon 15 100,000" ->
    "ADD CHICKEN BOX 909"). They print as their own lines and count toward the subtotal
    (100,000 + 909 = 100,909), but this converter originally dropped them. That affected
    17 of CORD's 100 test receipts: the labels had fewer rows than the paper, so a
    correct extraction was scored as inventing rows. It was caught when the first real
    Gemini run read the add-on the label was missing.
    """
    items = []
    for entry in _as_list(parse.get("menu")):
        if not isinstance(entry, dict):
            continue
        if row := _row(entry):
            items.append(row)
        for sub in _as_list(entry.get("sub")):
            if isinstance(sub, dict) and (row := _row(sub)):
                items.append(row)
    return items


def _positive(value) -> str | None:
    """CORD writes discounts as negative ("-60.000"); the schema stores the magnitude."""
    money = normalize_money(value)
    return money.lstrip("-") if money else None


def _to_receipt(parse: dict) -> dict:
    sub = parse.get("sub_total") or {}
    sub = sub[0] if isinstance(sub, list) and sub else sub
    tot = parse.get("total") or {}
    tot = tot[0] if isinstance(tot, list) and tot else tot
    return prune(
        {
            "subtotal": normalize_money(sub.get("subtotal_price")),
            # Dropping these made subtotal + tax != total on every receipt with a service
            # charge (12 of CORD's 100 test receipts) or a discount (6), so the arithmetic
            # check, the strongest confidence signal, failed on correct extractions.
            "service_charge": normalize_money(sub.get("service_price")),
            "discount": _positive(sub.get("discount_price")),
            "tax": normalize_money(sub.get("tax_price")),
            "total": normalize_money(tot.get("total_price")),
            "line_items": _line_items(parse),
        }
    )


def _merchant_key(receipt: dict) -> str:
    """CORD has no store field, so approximate a merchant by its menu fingerprint —
    good enough to stop the sampler stacking near-identical receipts."""
    names = [i.get("description", "") for i in receipt.get("line_items", [])][:3]
    return "|".join(sorted(names)).lower()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--resample", action="store_true",
                    help="draw a new sample instead of re-converting the pinned documents")
    args = ap.parse_args()

    manifest = RAW / "ground_truth.jsonl"
    if not manifest.exists():
        raise SystemExit(f"{manifest} missing — run ./scripts/fetch_datasets.sh cord first")

    parsed: dict[str, tuple[Path, dict]] = {}
    pool: list[tuple[str, int, str]] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        gt = json.loads(row["ground_truth"])
        receipt = _to_receipt(gt.get("gt_parse", {}))
        items = receipt.get("line_items", [])
        if not items or receipt.get("total") is None:
            continue  # unusable as ground truth for the fields we score
        key = row["file_name"]
        parsed[key] = (RAW / "images" / key, receipt)
        pool.append((key, len(items), _merchant_key(receipt)))

    pins = {} if args.resample else pinned_selection("cord")
    if pins:
        missing = sorted(set(pins.values()) - set(parsed))
        if missing:
            raise SystemExit(f"pinned documents no longer usable: {missing}; pass --resample")
        assignment = dict(sorted(pins.items()))
        print(f"re-converting the {len(pins)} pinned CORD documents (--resample to redraw)")
    else:
        chosen = stratified_sample(pool, args.n)
        if len(chosen) < args.n:
            raise SystemExit(f"only {len(chosen)} usable CORD docs for --n {args.n}")
        assignment = {f"cord_{i:04d}": key for i, key in enumerate(sorted(chosen), start=1)}

    clear_source("cord")
    entries = {}
    for doc_id, key in assignment.items():
        image, receipt = parsed[key]
        write_document(doc_id, image, receipt)
        entries[doc_id] = provenance_entry(
            SOURCE, key, SCRIPT_FIELDS, needs_hand_label=UNANNOTATED
        )
    carry_verification(entries)
    update_provenance(entries)
    chosen = list(assignment.values())

    sizes = sorted(len(parsed[k][1]["line_items"]) for k in chosen)
    print(f"wrote {len(chosen)} CORD docs; line items per doc min={sizes[0]} "
          f"median={sizes[len(sizes) // 2]} max={sizes[-1]}")


if __name__ == "__main__":
    main()

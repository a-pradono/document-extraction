"""WildReceipt -> Receipt ground truth.

    python scripts/convert_wildreceipt.py [--n 25]

Reads data/raw/wildreceipt/ and writes 25 stratified documents from test.txt as
data/docs/wild_XXXX.jpeg + data/ground_truth/wild_XXXX.json.

Bounding boxes are used for exactly one thing — grouping Prod_item / Prod_quantity /
Prod_price annotations into the same line item by vertical proximity — and are then
discarded. No coordinate reaches the schema.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import date as date_cls
from decimal import Decimal
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

RAW = REPO / "data" / "raw" / "wildreceipt"
SOURCE = "WildReceipt (Sun et al., arXiv:2103.14470; test split)"

# Class NAMES we consume. Ints are read from class_list.txt — they are not stable
# enough to hardcode, and a silent off-by-one would poison every label.
WANTED = {
    "Store_name_value": "merchant_name",
    "Store_addr_value": "merchant_address",
    "Date_value": "date",
    "Subtotal_value": "subtotal",
    # "10% SVC CHG", "Gratuity": added on top of the subtotal. Unmapped, it made
    # subtotal + tax != total on every tipped receipt.
    "Tips_value": "service_charge",
    "Tax_value": "tax",
    "Total_value": "total",
}
ITEM_CLASSES = {
    "Prod_item_value": "description",
    "Prod_quantity_value": "quantity",
    "Prod_price_value": "line_total",
}

_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}


def _load_class_names() -> dict[int, str]:
    names: dict[int, str] = {}
    for line in (RAW / "class_list.txt").read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].isdigit():
            names[int(parts[0])] = parts[1]
    return names


def _center_y(box: list[float]) -> float:
    return sum(box[1::2]) / (len(box) / 2)


def _height(box: list[float]) -> float:
    ys = box[1::2]
    return max(ys) - min(ys)


def _left_x(box: list[float]) -> float:
    return min(box[0::2])


def _center_x(box: list[float]) -> float:
    return sum(box[0::2]) / (len(box) / 2)


def align_column(anchors: list[float], others: list[float], tol: float) -> list[tuple[int, int]]:
    """Order-preserving match of a column (prices) to the description rows.

    Rows on a receipt cannot cross: the 3rd price belongs to the 3rd-or-later item, never
    to the 1st. So this finds the matching that pairs the MOST boxes without crossings,
    then the smallest total vertical gap among those. (A dynamic programme, like sequence
    alignment; n is at most a few dozen.)

    Why not nearest-neighbour: on a tilted photo every price sits several pixels below
    its item, so the nearest item is the NEXT row down. Nearest-neighbour shifted a whole
    table by one row and orphaned the first item (wild_0018: Kopi lost its 1.70 to the
    wanton soup). Maximising order-preserving matches keeps all six rows paired. A skew
    correction was tried first, but WildReceipt's boxes are axis-aligned rectangles even
    on tilted photos, so there is no slope to measure.
    """
    n, m = len(anchors), len(others)
    # best[i][j] = (matches, -cost) for anchors[:i] and others[:j]
    best = [[(0, 0.0)] * (m + 1) for _ in range(n + 1)]
    step = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 and j == 0:
                continue
            options = []
            if i > 0:
                options.append((best[i - 1][j], "skip_anchor"))
            if j > 0:
                options.append((best[i][j - 1], "skip_other"))
            if i > 0 and j > 0 and abs(others[j - 1] - anchors[i - 1]) <= tol:
                matches, neg_cost = best[i - 1][j - 1]
                options.append(((matches + 1, neg_cost - abs(others[j - 1] - anchors[i - 1])), "match"))
            best[i][j], step[i][j] = max(options, key=lambda o: o[0])
    pairs, i, j = [], n, m
    while i > 0 or j > 0:
        move = step[i][j]
        if move == "match":
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif move == "skip_anchor":
            i -= 1
        else:
            j -= 1
    return sorted(pairs)


def parse_date(raw: str) -> str | None:
    """Return YYYY-MM-DD, or None when the format is ambiguous or implausible.

    Deliberately conservative: a guessed date is worse than a null, because it becomes
    a permanent wrong label that the extractor is then scored against.
    """
    s = raw.strip().lower().replace(",", " ")
    m = re.search(r"\b(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})\b", s)
    if m:
        y, mo, d = (int(g) for g in m.groups())
    else:
        m = re.search(r"\b(\d{1,2})[-/.](\d{1,2})[-/.](\d{2,4})\b", s)
        if m:
            a, b, c = (int(g) for g in m.groups())
            if a > 12 and b <= 12:
                return None  # DD/MM — WildReceipt is mixed-locale; do not guess
            mo, d, y = a, b, c
            y += 2000 if y < 100 else 0
        else:
            m = re.search(r"\b([a-z]{3})[a-z]*\.?\s+(\d{1,2})\s+(\d{4})\b", s)
            if not m:
                return None
            mo = _MONTHS.get(m.group(1), 0)
            d, y = int(m.group(2)), int(m.group(3))
    try:
        parsed = date_cls(y, mo, d)
    except ValueError:
        return None
    if not (2000 <= parsed.year <= date_cls.today().year):
        return None
    return parsed.isoformat()


def _group_rows(boxes: list[tuple[float, float, float, str, str]]) -> list[dict]:
    """boxes: (center_y, center_x, height, field, text) -> line items, top to bottom.

    Descriptions anchor the rows; quantities and prices are each attached with an
    order-preserving alignment (see align_column), not by chaining boxes on vertical
    proximity. The tolerance allows up to about one row pitch, which is safe only BECAUSE
    matches cannot cross.
    """
    if not boxes:
        return []
    heights = sorted(b[2] for b in boxes)
    height = heights[len(heights) // 2]
    placed = [(y, x, field, text) for y, x, _, field, text in boxes]

    anchors = sorted((b for b in placed if b[2] == "description"), key=lambda b: b[0])
    gaps = sorted(b[0] - a[0] for a, b in zip(anchors, anchors[1:]) if b[0] > a[0])
    pitch = gaps[len(gaps) // 2] if gaps else height
    tol = max(0.9 * pitch, 0.75 * height, 4.0)

    rows: list[list[tuple]] = [[a] for a in anchors]
    orphans: list[tuple] = []
    for field_name in ("quantity", "line_total"):
        column = sorted((b for b in placed if b[2] == field_name), key=lambda b: b[0])
        matched = set()
        for ai, oi in align_column([a[0] for a in anchors], [c[0] for c in column], tol):
            rows[ai].append(column[oi])
            matched.add(oi)
        orphans.extend(c for k, c in enumerate(column) if k not in matched)
    tol = max(0.75 * height, 4.0)                    # orphans only group on real proximity

    # Boxes no description claimed still form rows among themselves, so nothing annotated
    # is silently discarded; the audit flags any that end up without a description.
    for box in sorted(orphans, key=lambda b: b[0]):
        target = next((r for r in rows if r[0][2] != "description"
                       and abs(r[0][0] - box[0]) <= tol and all(b[2] != box[2] for b in r)), None)
        if target is not None:
            target.append(box)
        else:
            rows.append([box])
    rows.sort(key=lambda r: min(b[0] for b in r))

    items = []
    for row in rows:
        item: dict[str, str] = {}
        for _, _, field, text in sorted(row, key=lambda b: b[1]):
            if field in item:
                continue  # first occurrence left-to-right wins
            if field == "description":
                item[field] = text.strip()
            elif field == "quantity":
                item[field] = normalize_qty(text)
            else:
                item[field] = normalize_money(text)
        item = prune(item)
        # A price box with no description and no amount is stray annotation, not a row.
        # A described item priced 0.00 is real (a comp or a promo) and stays.
        amount = normalize_money(item.get("line_total"))
        if item.get("description") or (amount is not None and Decimal(amount) != 0):
            items.append(item)
    return items


def convert(row: dict, class_names: dict[int, str]) -> dict:
    scalars: dict[str, list[tuple[float, float, str]]] = {}
    item_boxes: list[tuple[float, float, float, str, str]] = []

    for ann in row.get("annotations", []):
        name = class_names.get(ann.get("label", -1))
        text = (ann.get("text") or "").strip()
        box = ann.get("box") or []
        if not name or not text or len(box) < 8:
            continue
        if name in WANTED:
            scalars.setdefault(WANTED[name], []).append((_center_y(box), _left_x(box), text))
        elif name in ITEM_CLASSES:
            item_boxes.append((_center_y(box), _center_x(box), _height(box),
                               ITEM_CLASSES[name], text))

    def joined(field: str) -> str | None:
        """Text spanning several boxes (a store name across three lines) — join in
        reading order, dropping repeats."""
        seen, parts = set(), []
        for _, _, text in sorted(scalars.get(field, [])):
            if text.lower() not in seen:
                seen.add(text.lower())
                parts.append(text)
        return " ".join(parts).strip() or None

    def single(field: str, normalizer) -> str | None:
        """One value that may be annotated several times.

        A receipt commonly carries Total_value three or four times — the line, the
        EFTPOS slip, the signature copy — as '77.43' and 'AU$77.43'. Joining them
        produced '77437743774377.43'. Take the modal normalized value instead, and
        break ties with the bottom-most box, since totals print low on a receipt.
        """
        entries = scalars.get(field, [])
        if not entries:
            return None
        tally: dict[str, list[float]] = {}
        for y, _, text in entries:
            value = normalizer(text)
            if value is not None:
                tally.setdefault(value, []).append(y)
        if not tally:
            return None
        return max(tally, key=lambda v: (len(tally[v]), max(tally[v])))

    return prune(
        {
            "merchant_name": joined("merchant_name"),
            "merchant_address": joined("merchant_address"),
            "date": single("date", parse_date),
            "subtotal": single("subtotal", normalize_money),
            "service_charge": single("service_charge", normalize_money),
            "tax": single("tax", normalize_money),
            "total": single("total", normalize_money),
            "line_items": _group_rows(item_boxes),
        }
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--resample", action="store_true",
                    help="draw a new sample instead of re-converting the pinned documents")
    args = ap.parse_args()

    test_txt = RAW / "test.txt"
    if not test_txt.exists():
        raise SystemExit(f"{test_txt} missing — run ./scripts/fetch_datasets.sh wildreceipt")
    class_names = _load_class_names()

    converted: dict[str, dict] = {}
    pool: list[tuple[str, int, str]] = []
    for line in test_txt.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        receipt = convert(row, class_names)
        name = row["file_name"]
        converted[name] = receipt
        items = receipt.get("line_items", [])
        if items and receipt.get("total") is not None:
            pool.append((name, len(items), (receipt.get("merchant_name") or name).lower()))

    pins = {} if args.resample else pinned_selection("wild")
    if pins:
        missing = sorted(set(pins.values()) - set(converted))
        if missing:
            raise SystemExit(f"pinned documents not in test.txt: {missing}; pass --resample")
        assignment = dict(sorted(pins.items()))
        print(f"re-converting the {len(pins)} pinned WildReceipt documents (--resample to redraw)")
    else:
        chosen = stratified_sample(pool, args.n)
        if len(chosen) < args.n:
            raise SystemExit(f"only {len(chosen)} usable WildReceipt docs for --n {args.n}")
        assignment = {f"wild_{i:04d}": name for i, name in enumerate(sorted(chosen), start=1)}
    chosen = list(assignment.values())

    clear_source("wild")
    entries = {}
    for doc_id, name in assignment.items():
        receipt = converted[name]
        write_document(doc_id, RAW / name, receipt)
        script_fields = [k for k in receipt if k != "line_items"] + ["line_items"]
        # WildReceipt's transcriptions have the spaces stripped out
        # ("CaltexCentralMyrtleford"), so any text field it gives us is lossy and will
        # never match a correct reading exactly. Flag those for re-spacing by hand
        # rather than shipping a label we already know is wrong.
        needs = ["currency", "payment_method"]
        needs += [f for f in ("merchant_name", "merchant_address") if f in receipt]
        # The dataset annotates a Date_value box for most receipts; when parse_date
        # refused it (mixed locale, e.g. an Australian 24/01/2012), the null means
        # "unparsed", not "absent" — flag it for a human.
        needs += ["date"] if "date" not in receipt else []
        entries[doc_id] = provenance_entry(SOURCE, name, script_fields, needs_hand_label=needs)
    carry_verification(entries)
    update_provenance(entries)

    sizes = sorted(len(converted[k]["line_items"]) for k in chosen)
    print(f"wrote {len(chosen)} WildReceipt docs; line items per doc min={sizes[0]} "
          f"median={sizes[len(sizes) // 2]} max={sizes[-1]}")


if __name__ == "__main__":
    main()

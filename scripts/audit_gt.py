"""Model-free audit of the ground-truth labels: where should a human look first?

    python scripts/audit_gt.py                 # every document, flagged ones first
    python scripts/audit_gt.py --only-flagged
    python scripts/audit_gt.py --json audit.json

It does NOT decide a label is wrong. A receipt with a discount really can print a total
that isn't subtotal + tax, and those receipts are valuable test cases. What it does is
rank the 50-65 documents so verification time goes to the labels most likely to be
broken by a converter or a dataset quirk, instead of being spread evenly over clean ones.

scripts/verify_gt.py --html shows these flags beside each image and lists flagged
documents first.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path

from gt_common import GT_DIR, PROVENANCE

from extractor.confidence import (  # noqa: E402 — gt_common puts the repo on sys.path
    RECONCILIATION_FIELDS,
    expected_total,
    money_tolerance,
    norm_party,
    reconciles,
)

# Severity drives the ranking. "error" means the label is almost certainly wrong, "check"
# means it is suspicious but could be legitimate, and "todo" is known unfinished labeling.
ERROR, CHECK, TODO = "error", "check", "todo"
_RANK = {ERROR: 0, CHECK: 1, TODO: 2}


def _d(value) -> Decimal | None:
    if value in (None, ""):
        return None
    try:
        return Decimal(str(value))
    except InvalidOperation:
        return None


def audit_receipt(label: dict) -> list[dict]:
    flags: list[dict] = []

    def flag(severity: str, field: str, message: str) -> None:
        flags.append({"severity": severity, "field": field, "message": message})

    for money in RECONCILIATION_FIELDS:
        if label.get(money) not in (None, "") and _d(label[money]) is None:
            flag(ERROR, money, f"not a number: {label[money]!r}")
        elif (v := _d(label.get(money))) is not None and v < 0:
            flag(CHECK, money, f"negative: {v}")

    sub, service, discount, tax, total = (_d(label.get(k)) for k in RECONCILIATION_FIELDS)
    tolerance = money_tolerance(sub, total)
    if sub is not None and total is not None:
        # The SAME rule and tolerance the pipeline's validation signal uses: an audit that
        # reconciles differently from the extractor would flag, or pass, the wrong labels.
        expected = expected_total(sub, service, discount, tax)
        if not reconciles(expected, total, tolerance):
            parts = " ".join(p for p in (f"subtotal {sub}",
                                          f"+ service {service}" if service else "",
                                          f"- discount {discount}" if discount else "",
                                          f"+ tax {tax or 0}") if p)
            flag(CHECK, "total", f"{parts} = {expected}, label says {total} "
                                 f"(an unlabeled charge, or a mislabeled field?)")
    if tax is not None and total is not None and tax > total:
        flag(ERROR, "tax", f"tax {tax} exceeds total {total}")
    if sub is not None and total is not None and sub > total * 2:
        flag(CHECK, "subtotal", f"subtotal {sub} is more than double the total {total}")

    items = label.get("line_items") or []
    seen_rows: dict[tuple, int] = {}
    for i, item in enumerate(items):
        q, u, t = (_d(item.get(k)) for k in ("quantity", "unit_price", "line_total"))
        if None not in (q, u, t) and abs(q * u - t) > money_tolerance(t):
            flag(CHECK, f"line_items[{i}]", f"{q} x {u} = {q * u}, row says {t}")
        if not item.get("description") and t is not None:
            flag(CHECK, f"line_items[{i}]", f"amount {t} with no description")
        key = (item.get("description"), item.get("quantity"), item.get("line_total"))
        if key in seen_rows and key[0]:
            flag(CHECK, f"line_items[{i}]",
                 f"identical to row {seen_rows[key]} (really bought twice, or a converter double-count?)")
        seen_rows.setdefault(key, i)

    totals = [_d(i.get("line_total")) for i in items]
    if items and all(t is not None for t in totals):
        line_sum = sum(totals)
        if sub is not None:
            anchor_name, anchor = "subtotal", sub
        elif total is not None:
            anchor_name = "total less charges"
            anchor = total - (service or 0) + abs(discount or 0) - (tax or 0)
        else:
            anchor_name, anchor = "", None
        if anchor is not None and abs(line_sum - anchor) > tolerance:
            flag(CHECK, "line_items",
                 f"rows sum to {line_sum}, {anchor_name} is {anchor} "
                 f"({len(items)} rows: a dropped or extra row?)")

    if label.get("date"):
        try:
            d = date.fromisoformat(str(label["date"]))
            if d > date.today():
                flag(ERROR, "date", f"in the future: {d}")
            elif d.year < 2000:
                flag(CHECK, "date", f"implausibly old: {d}")
        except ValueError:
            flag(ERROR, "date", f"not YYYY-MM-DD: {label['date']!r}")

    text_fields = [("merchant_name", label.get("merchant_name")),
                   ("merchant_address", label.get("merchant_address"))]
    for role in ("vendor", "customer"):
        party = label.get(role) or {}
        text_fields += [(f"{role}.name", party.get("name")), (f"{role}.address", party.get("address"))]
    for text_field, raw in text_fields:
        value = str(raw or "")
        # WildReceipt strips spaces: "CaltexCentralMyrtleford". A long unbroken token with
        # internal capitals or digit/letter joins is almost never how a store name prints.
        for token in value.split():
            if len(token) >= 12 and (re.search(r"[a-z][A-Z]", token) or re.search(r"\d[A-Za-z]{3}", token)):
                flag(CHECK, text_field, f"looks space-stripped: {token!r}")
                break

    flags += audit_invoice(label)
    return flags


def audit_invoice(label: dict) -> list[dict]:
    """Invoice-only label checks: the same two consistency rules the extractor's validation
    signal uses, plus the fields that make an invoice label usable at all."""
    if "vendor" not in label and "customer" not in label and "issue_date" not in label:
        return []
    flags: list[dict] = []
    dates = {}
    for field in ("issue_date", "due_date"):
        if label.get(field):
            try:
                dates[field] = date.fromisoformat(str(label[field]))
            except ValueError:
                flags.append({"severity": ERROR, "field": field,
                              "message": f"not YYYY-MM-DD: {label[field]!r}"})
    if len(dates) == 2 and dates["due_date"] < dates["issue_date"]:
        flags.append({"severity": ERROR, "field": "due_date",
                      "message": f"due {dates['due_date']} is before issue {dates['issue_date']}"})
    vendor, customer = label.get("vendor") or {}, label.get("customer") or {}
    if vendor.get("name") and customer.get("name") and \
            norm_party(vendor["name"]) == norm_party(customer["name"]):
        flags.append({"severity": ERROR, "field": "customer.name",
                      "message": f"vendor and customer have the same name ({vendor['name']!r}): roles unresolved"})
    for role, party in (("vendor", vendor), ("customer", customer)):
        if not party.get("name"):
            flags.append({"severity": CHECK, "field": f"{role}.name",
                          "message": f"no {role} name: redacted or genuinely absent? Role resolution needs both"})
    return flags


def audit_all() -> list[dict]:
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8")) if PROVENANCE.exists() else {}
    docs = []
    for path in sorted(GT_DIR.glob("*.json")):
        if path.stem.startswith("_"):
            continue
        label = json.loads(path.read_text(encoding="utf-8"))
        entry = provenance.get(path.stem, {})
        if path.stem.startswith("inv_") and set(label) <= {"document_type"}:
            flags = [{"severity": TODO, "field": "*", "message": "invoice not labeled yet"}]
        else:
            flags = audit_receipt(label)
            if entry.get("label_status", "").startswith("drafted"):
                flags.append({"severity": TODO, "field": "*",
                              "message": "labels DRAFTED by the assistant from the page images: "
                                         "check every field against the document"})
        for pending in entry.get("needs_hand_label", []):
            if pending == "*":
                continue
            flags.append({"severity": TODO, "field": pending,
                          "message": "not annotated by the source dataset; label from the image"})
        docs.append({"doc_id": path.stem, "verified": bool(entry.get("verified")), "flags": flags})

    def sort_key(doc: dict):
        worst = min((_RANK[f["severity"]] for f in doc["flags"]), default=3)
        real = sum(1 for f in doc["flags"] if f["severity"] != TODO)
        return (doc["verified"], worst, -real, doc["doc_id"])

    return sorted(docs, key=sort_key)


def summary(docs: list[dict]) -> dict:
    by_severity = {s: 0 for s in (ERROR, CHECK, TODO)}
    by_kind: dict[str, int] = {}
    for doc in docs:
        for f in doc["flags"]:
            by_severity[f["severity"]] += 1
            if f["severity"] != TODO:
                kind = re.sub(r"\[\d+\]", "[]", f["field"])
                by_kind[kind] = by_kind.get(kind, 0) + 1
    flagged = [d for d in docs if any(f["severity"] != TODO for f in d["flags"])]
    return {"documents": len(docs), "documents_flagged": len(flagged),
            "by_severity": by_severity, "by_field": dict(sorted(by_kind.items(), key=lambda kv: -kv[1]))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-flagged", action="store_true")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    docs = audit_all()
    s = summary(docs)
    print(f"{s['documents']} documents, {s['documents_flagged']} with suspicious labels "
          f"({s['by_severity'][ERROR]} likely errors, {s['by_severity'][CHECK]} to check, "
          f"{s['by_severity'][TODO]} unlabeled fields)")
    print("flags by field: " + ", ".join(f"{k} {v}" for k, v in s["by_field"].items()) + "\n")
    for doc in docs:
        real = [f for f in doc["flags"] if f["severity"] != TODO]
        if args.only_flagged and not real:
            continue
        todo = sum(1 for f in doc["flags"] if f["severity"] == TODO)
        state = "verified" if doc["verified"] else "unverified"
        print(f"{doc['doc_id']}  [{state}]  {len(real)} flag(s), {todo} unlabeled field(s)")
        for f in real:
            print(f"    {f['severity']:<5} {f['field']:<16} {f['message']}")
    if args.json:
        args.json.write_text(json.dumps({"summary": s, "documents": docs}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

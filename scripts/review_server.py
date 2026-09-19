"""Fast ground-truth verification: one document, one screen, keyboard verdicts.

    python scripts/review_server.py                 # opens http://127.0.0.1:8900
    python scripts/review_server.py --all           # include already-verified documents

Why this exists. The contact sheet (scripts/verify_gt.py --html) shows the whole label
as JSON beside the image, and the verdict is typed in a separate terminal. Two documents
slipped through a full pass that way - wild_0003 and wild_0019, both WildReceipt money
errors of the same shape (a dropped leading digit and a missing charge) - because a wrong
subtotal three lines into a JSON blob reads as noise.

This view asks a question instead of printing a record:
  - only the fields a human still has to judge (the drafted header fields and the money
    block, which is where both errors were),
  - the arithmetic the model-free audit disputes, spelled out,
  - the drafter's own CHECK note about what they could not resolve,
  - y / n / s on the keyboard, written straight to _provenance.json.

Order is scripts/verify_gt.review_order: CHECK notes first, then audit flags, then the
rest - the same order the contact sheet uses.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import threading
import webbrowser
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response
from PIL import Image, ImageOps
from pydantic import BaseModel

from audit_gt import audit_all
from gt_common import DOCS_DIR, GT_DIR
from verify_gt import _load_provenance, _notes, _record, review_order

HEADER_FIELDS = ("merchant_name", "merchant_address", "date", "currency", "payment_method",
                 "invoice_number", "purchase_order", "issue_date", "due_date")
MONEY_FIELDS = ("subtotal", "service_charge", "discount", "tax", "total")
THUMB_WIDTH = 1100


def _doc_path(doc_id: str) -> Optional[Path]:
    hits = sorted(DOCS_DIR.glob(f"{doc_id}.*"))
    return hits[0] if hits else None


def _label(doc_id: str) -> dict:
    return json.loads((GT_DIR / f"{doc_id}.json").read_text(encoding="utf-8"))


def _party(value) -> str:
    return ", ".join(str(v) for v in value.values() if v) if isinstance(value, dict) else str(value)


def _rows_sum(label: dict) -> Optional[Decimal]:
    totals = [r.get("line_total") for r in label.get("line_items") or []]
    if not totals or any(t is None for t in totals):
        return None
    try:
        return sum(Decimal(str(t)) for t in totals)
    except InvalidOperation:
        return None


def document_payload(doc_id: str, prov: dict, flags: dict) -> dict:
    """Everything the reviewer needs for one decision, and nothing else."""
    label = _label(doc_id)
    entry = prov.get(doc_id, {})
    drafted = set(entry.get("hand_labeled_fields") or [])

    fields = []
    for name in HEADER_FIELDS:
        if name not in label and name not in drafted:
            continue
        value = label.get(name)
        fields.append({"field": name,
                       "value": None if value in (None, "") else _party(value),
                       "drafted": name in drafted})
    for name in ("vendor", "customer"):
        if name in label:
            fields.append({"field": name, "value": _party(label.get(name)) or None,
                           "drafted": name in drafted})

    money = [{"field": n, "value": label.get(n)} for n in MONEY_FIELDS if n in label]
    rows = label.get("line_items") or []
    rows_sum = _rows_sum(label)

    return {
        "doc_id": doc_id,
        "document_type": "invoice" if doc_id.startswith("inv_") else "receipt",
        "source": entry.get("source", ""),
        "pages": entry.get("page_count") or 1,
        "verified": bool(entry.get("verified")),
        "note": _notes(entry),
        "check": "CHECK" in _notes(entry),
        "fields": fields,
        "money": money,
        "rows": len(rows),
        "rows_sum": None if rows_sum is None else f"{rows_sum:,}",
        "row_preview": [{"description": r.get("description"), "qty": r.get("quantity"),
                         "line_total": r.get("line_total")} for r in rows],
        # The audit's own words: "subtotal 97.30 + tax 6.72 = 104.02, label says 102.75"
        "disputes": [{"field": f["field"], "message": f["message"], "severity": f["severity"]}
                     for f in flags.get(doc_id, []) if f["severity"] != "todo"],
    }


class Verdict(BaseModel):
    doc_id: str
    verified: bool
    note: Optional[str] = None


def build_app(only_unverified: bool = True) -> FastAPI:
    app = FastAPI(title="ground-truth review")
    flags = {d["doc_id"]: d["flags"] for d in audit_all()}

    def doc_ids() -> list[str]:
        prov = _load_provenance()
        ids = sorted(p.stem for p in GT_DIR.glob("*.json") if not p.stem.startswith("_"))
        if only_unverified:
            ids = [i for i in ids if not prov.get(i, {}).get("verified")]
        return review_order(ids)

    @app.get("/api/docs")
    def api_docs() -> JSONResponse:
        prov = _load_provenance()
        payloads = [document_payload(i, prov, flags) for i in doc_ids()]
        total = len([p for p in GT_DIR.glob("*.json") if not p.stem.startswith("_")])
        done = sum(1 for v in prov.values() if v.get("verified"))
        return JSONResponse({"documents": payloads, "verified": done, "total": total})

    @app.get("/api/image/{doc_id}")
    def api_image(doc_id: str) -> Response:
        path = _doc_path(doc_id)
        if path is None:
            raise HTTPException(404, f"no document file for {doc_id}")
        if path.suffix.lower() == ".pdf":
            import pypdfium2 as pdfium

            image = pdfium.PdfDocument(str(path))[0].render(scale=2).to_pil()
        else:
            image = ImageOps.exif_transpose(Image.open(path))
        image = image.convert("RGB")
        if image.width > THUMB_WIDTH:
            image = image.resize(
                (THUMB_WIDTH, round(image.height * THUMB_WIDTH / image.width)), Image.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=82, optimize=True)
        return Response(buffer.getvalue(), media_type="image/jpeg")

    @app.post("/api/verdict")
    def api_verdict(verdict: Verdict) -> JSONResponse:
        if verdict.doc_id not in _load_provenance():
            raise HTTPException(404, f"unknown document {verdict.doc_id}")
        # Re-read and merge one entry: never write back a snapshot, so a page left open
        # cannot revert verdicts or notes saved elsewhere in the meantime.
        prov = _record(verdict.doc_id,
                       {"verified": verdict.verified, "verified_note": verdict.note or None})
        done = sum(1 for v in prov.values() if v.get("verified"))
        return JSONResponse({"ok": True, "verified": done, "total": len(prov)})

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse((Path(__file__).parent / "review.html").read_text(encoding="utf-8"))

    return app


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true", help="include already-verified documents")
    ap.add_argument("--port", type=int, default=8900)
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args(argv)

    app = build_app(only_unverified=not args.all)
    url = f"http://127.0.0.1:{args.port}"
    if not args.no_open:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    print(f"review at {url}   (y verify · n needs fixing · s skip · arrows move)")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

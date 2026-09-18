# OUTPUT.md — the extraction output contract

Instructions for a coding agent. This file defines **what the system returns for one
document**. Treat the JSON shape below as a contract: the API, the batch runner, the eval
harness and the HTML renderer all consume it, so changing a key means changing all four.

**Status: built.** `api/app.py` returns this shape (`ExtractionOut` in `api/schemas.py`),
and `eval/render_doc.py` renders it. Sections 3 to 6 describe work that is now done and
are kept as the rationale for it. Section 2 is **not** built, deliberately — see there.

An earlier version of this file described a scaffold that never existed in this
repository: a main module under api/ returning an ExtractionResponse with a top-level
`fields` key. The payload below was dumped from the running code, and a test
(`eval/tests/test_render_doc.py`) now fails if this file drifts from it again.

---

## 1. The JSON contract

`POST /extract` returns exactly this, and the batch runner writes one of these per line
of `results.jsonl`. Every key is always present; `null` and `[]` are valid values,
absence of a key is not.

```json
{
  "doc_id": "cord_0007",
  "status": "ok",
  "document_type": "receipt",
  "document": {
    "merchant_name": "Kopi Kenangan",
    "merchant_address": "Jl. Sudirman No. 45, Jakarta",
    "date": "2024-03-12",
    "currency": "IDR",
    "subtotal": "48000",
    "service_charge": null,
    "discount": null,
    "tax": "5280",
    "total": "53280",
    "payment_method": null,
    "line_items": [
      {"description": "Es Kopi Susu", "quantity": "2", "unit_price": "18000", "line_total": "36000"},
      {"description": "Croissant",    "quantity": "1", "unit_price": "12000", "line_total": "12000"}
    ]
  },
  "confidence": {"merchant_name": 0.94, "total": 0.97, "payment_method": 0.42},
  "abstained": ["payment_method"],
  "withheld": {"payment_method": null},
  "signals": {
    "total": {"agreement": 1.0, "grounded": 1.0, "validated": 1.0, "self_report": 0.95}
  },
  "validation": {
    "subtotal_plus_tax_equals_total": true,
    "line_items_sum_to_subtotal": true,
    "rows_multiply_out": true,
    "date_parses": true,
    "currency_is_iso": true
  },
  "validation_notes": {},
  "transcription": "KOPI KENANGAN\nJl. Sudirman No. 45...",
  "threshold": 0.6,
  "usage": {"latency_s": 4.31, "wall_s": 4.4, "cost_usd": 0.0, "stages": []},
  "cascade": null,
  "warnings": [],
  "error": null
}
```

### Why `document` and not `fields`

The key holds **either a `Receipt` or an `Invoice`**, and `document_type` says which.
`fields` was a receipts-only name from before the second document type existed. An
invoice's `document` carries `invoice_number`, `issue_date`, `due_date`, `vendor`,
`customer`, `purchase_order` and the same `line_items`; `vendor` and `customer` are
nested objects (`{name, address, tax_id}`), and every other block addresses their
fields by dotted name: `"customer.name"` in `confidence`, `abstained` and `withheld`.

### Non-negotiable rules for this payload

1. **`null` means absent from the document. It is never a guess.** A field the model
   could not find is `null` — never an empty string, a zero, or a plausible value.
2. **`null` and "abstained" are DIFFERENT and both must be visible.** `null` = not on
   the document. Appearing in `abstained` = we read something but confidence fell below
   `threshold`, so we decline to assert it. Abstention is applied at the API boundary:
   the value is removed from `document` and kept in `withheld`, so a consumer reading
   `document` never sees an unasserted value while a human reviewer can still see what
   was read. (A field can be both: `payment_method` above was absent *and* scored below
   threshold, so its `withheld` entry is `null`.)
3. **`line_items` is withheld whole**, never partially: the table becomes `[]` and every
   row read is kept in `withheld["line_items"]` as a JSON string. A half-withheld table
   would silently change the arithmetic.
4. **Money and quantities are JSON strings**, not numbers (`"48000"`, not `48000`). They
   parse to `Decimal`. Emitting floats reintroduces the rounding error the `Decimal`
   choice exists to prevent.
5. **Dates are `YYYY-MM-DD` strings.**
6. **`signals` is exposed, not hidden.** Anyone must be able to see WHY a confidence
   score is what it is. An opaque number is not defensible; four named signals are.
7. **`validation` is a separate top-level block** of deterministic, model-free checks.
   Their value is that they are independent of both models, so surface them as
   first-class results rather than burying them inside `signals`. **`null` means the
   check did not apply, which is not the same as passing** — a receipt with no printed
   subtotal has not reconciled, it was never checked. `validation_notes` explains each
   failure (`"48,000.00 != 47,500.00 - off by 500.00"`) so a consumer can show the
   discrepancy without recomputing it. Invoices add `due_date_after_issue_date` and
   `vendor_differs_from_customer`.
   These come from `extractor.confidence.document_checks`, which runs the **same** rules
   as the per-field `validate_arithmetic`, so the two can never disagree.
8. **A failed document still returns 200** with `status: "error"`, a populated `error`,
   and `document: null`. `warnings` carries non-fatal problems (a repaired JSON response,
   a salvaged field). One bad document must never fail the request or the batch.
9. **Never invent a `_confidence` value the model did not supply.** A missing self-report
   defaults to a neutral 0.5, never 1.0.

### Batch

`POST /extract/batch` returns `list[ExtractionOut]` — the same object per document, in
input order. No wrapper envelope.

---

## 2. Multi-page and multi-document inputs

Two different problems, which an earlier version of this file wrongly merged into one
universal list contract:

- **One document spanning N pages** — a *merging* problem. **Built.** Every page is
  rendered and sent in a single call, so the model sees the whole document at once and
  returns one object. This is what the 5-page invoices rely on. `MAX_PAGES` (default 5)
  refuses anything longer rather than silently reading part of it.
- **One PDF holding N separate documents** ("a stack") — a *splitting* problem, which
  would output N documents with no field merging between them. **Not built, and not
  planned without a real requirement.**

**Splitting must not change `POST /extract`.** Making it always return a list would break
every current consumer — the batch runner, the eval harness, the regression gate and the
tests — and it would break the spanning-pages case above, which is the one that already
works. If stack-splitting is ever needed, it is either:

- a **separate endpoint**, `POST /extract/stack` → a list, each element carrying
  `"pages": [1, 2]`; or
- a **pre-step** that splits the PDF into N single-document inputs and feeds them through
  the existing contract unchanged.

Both leave every current consumer untouched. That is the requirement for this feature:
it is additive, or it does not happen.

---

## 3. The per-document HTML card (`eval/render_doc.py`) — built

```
python -m eval.render_doc <extraction.json> [-o out.html]
```

The eval drill-down compares prediction against ground truth. This is the view of the
extraction **on its own**, which is what a reviewer wants after uploading one document.
Self-contained HTML (inline CSS, no CDN — it is opened over `file://`), reusing the
`_CSS` constant from `eval/scorecard.py` so both artifacts look like one system.

### Contents

**a) Field table** — one row per scalar field: name, value, confidence, status.

| field | value | conf | status |
|---|---|--:|---|
| merchant_name | Kopi Kenangan | 0.94 | ok |
| merchant_address | Jl. Sudirman No. 45, Jakarta | 0.71 | ok |
| total | 53280 | 0.97 | ok |
| payment_method | — | 0.42 | **abstained** |

`null` renders as an em dash, never blank (blank reads as a rendering bug). Status is
coloured ok / abstained / absent, and an abstained field shows its withheld value so it
cannot be mistaken for a field that was simply not on the document. The row's `title`
carries that field's four signals.

**b) Line-item table with the arithmetic check as a footer row**

| # | description | qty | unit price | line total |
|--:|---|--:|--:|--:|
| 1 | Es Kopi Susu | 2 | 18,000 | 36,000 |
| 2 | Croissant | 1 | 12,000 | 12,000 |
| | | | **sum** | **48,000** ✓ matches subtotal |

The most important element on the page: model-free arithmetic validation passing, which
cannot be bluffed the way a confidence number can. On failure it shows the discrepancy
from `validation_notes` rather than hiding it.

**c) Validation panel** — every `validation` entry as a pass / fail / n/a chip.
**d) Cost and latency footer** — model latency, prompt and completion tokens, cost,
per stage. **e) Transcription** in a collapsed `<details>`: it is the groundedness
source, so it must be inspectable without dominating the page.

### Do NOT

- No upload form and no JS frontend. `GET /docs` (Swagger UI) is already the interactive
  demo; this takes a JSON file and emits HTML.
- **No ground truth here.** This view is "what the system extracted", full stop.
  Comparison belongs in `scorecard.html`. Mixing them hides the case that matters most:
  a confident, internally consistent extraction that is simply wrong.
- **Recompute nothing.** Read `validation` from the payload. Recomputing would let the
  page and the API disagree about whether a document added up.

## 4. Wired into the eval run — built

A run writes `outputs/<run>/documents/<doc_id>.html` per document, plus the payload it
was rendered from as `<doc_id>.json` (so a card can be re-rendered after a change to the
renderer without re-running the models). The per-document table in `scorecard.html` links
to each card, so a reviewer can jump from a failing row to the full extraction.
`per_document.json` is unchanged: it stays the machine-readable comparison artifact.

## 5. Tests (`eval/tests/test_render_doc.py`) — built

- A `null` field renders as an em dash and is NOT marked abstained.
- An abstained field renders with abstained styling, and its withheld value, even so.
- A failing arithmetic check renders the discrepancy, not a silent pass.
- The verdict comes from the payload: a hand-edited `validation` block changes the page.
- A payload with `errors` populated and no fields renders without raising.
- The output contains no `http://` or `https://` asset references.

## 6. Commits — done

```
api: add validation, signals and transcription to the extraction response
eval: add per-document HTML card renderer (fields, line items, arithmetic check)
eval: link per-document cards from the scorecard drill-down
test: cover null vs abstained rendering and failed arithmetic display
```

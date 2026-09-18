# OUTPUT.md — the extraction output contract

Instructions for Claude Code. This file defines **what the system returns for one
document**. Treat the JSON shape below as a contract: the API, the eval harness, and
the HTML renderer all consume it, so changing a key means changing all three.

`api/main.py` already returns this shape (`ExtractionResponse`). What is MISSING is
the human-readable per-document view described in section 3.

---

## 1. The JSON contract

`POST /extract` returns exactly this. Every key is required; `null` and `[]` are
valid values, absence of a key is not.

```json
{
  "doc_id": "cord_0007",
  "fields": {
    "merchant_name": "Kopi Kenangan",
    "merchant_address": "Jl. Sudirman No. 45, Jakarta",
    "date": "2024-03-12",
    "currency": "IDR",
    "subtotal": "48000",
    "tax": "5280",
    "total": "53280",
    "payment_method": null,
    "line_items": [
      {"description": "Es Kopi Susu", "quantity": "2", "unit_price": "18000", "line_total": "36000"},
      {"description": "Croissant",    "quantity": "1", "unit_price": "12000", "line_total": "12000"}
    ]
  },
  "confidence": {
    "merchant_name": 0.94, "merchant_address": 0.71, "date": 0.96,
    "currency": 0.99, "subtotal": 0.97, "tax": 0.88,
    "total": 0.97, "payment_method": 0.42
  },
  "abstained": ["payment_method"],
  "signals": {
    "total": {"agreement": 1.0, "grounded": 1.0, "validated": 1.0, "self_report": 0.95}
  },
  "validation": {
    "subtotal_plus_tax_equals_total": true,
    "line_items_sum_to_subtotal": true,
    "date_parses": true,
    "currency_is_iso": true
  },
  "transcription": "KOPI KENANGAN\nJl. Sudirman No. 45...",
  "latency_s": 4.31,
  "tokens": {"prompt": 1842, "completion": 386},
  "cost_usd": 0.0,
  "errors": []
}
```

### Non-negotiable rules for this payload

1. **`null` means absent from the document. It is never a guess.** A field the model
   could not find is `null` — do not substitute an empty string, a zero, or a plausible
   value.
2. **`null` and "abstained" are DIFFERENT and both must be visible.** `null` = not on
   the document. Appearing in `abstained` = we extracted something but confidence fell
   below threshold, so we are declining to assert it. A consumer must be able to tell
   these apart; that distinction is the whole point of abstention.
3. **Money and quantities are JSON strings**, not numbers (`"48000"`, not `48000`).
   They parse to `Decimal`. Emitting floats reintroduces the rounding error the
   `Decimal` choice exists to prevent.
4. **Dates are `YYYY-MM-DD` strings.**
5. **`signals` is exposed, not hidden.** Anyone must be able to see WHY a confidence
   score is what it is. An opaque number is not defensible; four named signals are.
6. **`validation` is a separate top-level block.** These are deterministic,
   model-free checks and their value is that they are independent of the models — so
   surface them as first-class results, not buried inside `signals`.
7. **`errors` is a list, and a failed document still returns 200** with empty/partial
   `fields` and a populated `errors`. One bad document must never fail the request or
   the batch.
8. **Never invent a `_confidence` value the model did not supply.** Missing
   self-report defaults to a neutral 0.5, never 1.0.

### Batch

`POST /extract/batch` returns `list[ExtractionResponse]` — the same object per
document, in input order. Do not invent a wrapper envelope.

---

## 2. Multi-document / multi-page inputs

One PDF can hold several receipts ("a stack"). This changes the contract:

- `POST /extract` returns a **LIST**, always — a single-page input yields a
  one-element list. Uniform contract, no special-casing at the call site.
- Each element gains `"pages": [1, 2]` recording which source pages it came from.
- **Separate documents are NOT merged.** Page 2's total has nothing to do with page 1's.
- **One receipt spanning pages IS merged:** `line_items` concatenate across pages;
  scalar totals come from the LAST page (receipts print the total at the end).
  Continuation rule: a page with line items but no total continues the previous
  document.

---

## 3. BUILD THIS: per-document HTML card (`eval/render_doc.py`)

The eval drill-down compares prediction against ground truth. There is currently **no
view of the extraction on its own** — which is what a reviewer wants when they upload
one document. Build it.

```
python -m eval.render_doc <extraction.json> [-o out.html]
```

Self-contained HTML (inline CSS, no CDN — it gets opened over `file://`). Reuse the
`_CSS` constant from `eval/scorecard.py` so both artifacts look like one system.

### Required contents

**a) Field table** — one row per scalar field: field name, value, confidence, status.

| field | value | conf | status |
|---|---|--:|---|
| merchant_name | Kopi Kenangan | 0.94 | ok |
| merchant_address | Jl. Sudirman No. 45, Jakarta | 0.71 | ok |
| total | 53,280 | 0.97 | ok |
| payment_method | — | 0.42 | **abstained** |

- Render `null` as an em-dash, never as blank (blank reads as a rendering bug).
- Colour by status: ok / abstained / absent. Abstained must be visually obvious.
- On hover, show the field's four signals from `signals` as a tooltip.

**b) Line-item table with the arithmetic check as a footer row**

| # | description | qty | unit price | line total |
|--:|---|--:|--:|--:|
| 1 | Es Kopi Susu | 2 | 18,000 | 36,000 |
| 2 | Croissant | 1 | 12,000 | 12,000 |
| | | | **sum** | **48,000** ✓ matches subtotal |

This footer row is the most important element on the page. It shows the **model-free
arithmetic validation passing** — far more persuasive than any confidence number,
because it cannot be bluffed. When it FAILS, show the discrepancy explicitly
(`48,000 ≠ 47,500 — off by 500`), do not hide it.

**c) Validation panel** — the four `validation` booleans as pass/fail chips.

**d) Cost/latency footer** — latency, prompt/completion tokens, cost.

**e) Transcription** in a collapsed `<details>` — it is long, and it is the
groundedness source, so it must be inspectable but not dominate the page.

### Do NOT

- Do not build an upload form or a JS frontend. `GET /docs` (Swagger UI) is already
  the interactive demo; this renderer takes a JSON file and emits HTML, nothing more.
- Do not show ground truth here. This view is "what the system extracted", full stop.
  Comparison against ground truth belongs in `scorecard.html`.
- Do not recompute anything. Read `validation` from the payload; the API is the single
  source of truth. Recomputing in the renderer would let the two disagree.

---

## 4. Wire it into the eval run

After a run, write one card per document to
`outputs/<run>/documents/<doc_id>.html` and link them from the existing
`scorecard.html` per-document drill-down, so a reviewer can jump from a failing row to
the full extraction. Keep `per_document.json` as-is — it stays the machine-readable
artifact.

## 5. Tests (`eval/tests/test_render_doc.py`)

- A `null` field renders as an em-dash and is NOT marked abstained.
- An abstained field renders with abstained styling even when it has a value.
- A failing arithmetic check renders the discrepancy, not a silent pass.
- Rendering a payload with `errors` populated and empty `fields` does not raise.
- Output contains no `http://` or `https://` asset references (self-contained check).

## 6. Commits

```
api: add validation block to extraction response
eval: add per-document HTML card renderer (fields, line items, arithmetic check)
eval: link per-document cards from scorecard drill-down
test: cover null vs abstained rendering and failed arithmetic display
```

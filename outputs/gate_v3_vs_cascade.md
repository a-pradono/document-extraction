# Regression check: PASS - no regression

`groq-v3` -> `groq-cascade` on the same 39 documents (set `test`).

## What changed between the runs

Attribute the deltas below to these, and nothing else changed (ground truth and documents are identical).

| setting | baseline | candidate |
|---|---|---|
| models.cascade_model | `None` | `gemini-3.5-flash-lite` |

## Per field

| field | F1 before | F1 after | delta | newly wrong | newly right | |
|---|--:|--:|--:|---|---|---|
| merchant_name | 0.850 | 0.850 | +0.000 | - | - |  |
| merchant_address | 0.538 | 0.538 | +0.000 | - | - |  |
| payment_method | 0.889 | 0.889 | +0.000 | - | - |  |
| date | 1.000 | 1.000 | +0.000 | - | - |  |
| currency | 1.000 | 1.000 | +0.000 | - | - |  |
| subtotal | 0.955 | 0.955 | +0.000 | - | - |  |
| service_charge | 1.000 | 1.000 | +0.000 | - | - |  |
| discount | 1.000 | 1.000 | +0.000 | - | - |  |
| tax | 0.938 | 0.938 | +0.000 | - | - |  |
| total | 1.000 | 1.000 | +0.000 | - | - |  |
| line_items | 0.951 | 0.951 | +0.000 | - | - |  |

## Failures by kind (candidate minus baseline)

| kind | delta |
|---|--:|
| (no change) | 0 |

## Headline
- **Before:** `groq-v3` (source `pipeline`)
- **After:** `groq-cascade` (source `cascade`)
- Same 39 documents, set `test`.

| metric | before | after | delta | |
|---|--:|--:|--:|---|
| Document accuracy rate | 0.219 | 0.219 | 0 | same |
| Micro F1 | 0.918 | 0.918 | 0 | same |
| Macro F1 | 0.917 | 0.917 | 0 | same |
| Line-item F1 | 0.951 | 0.951 | 0 | same |
| Hallucination rate | 0.006 | 0.006 | 0 | same |
| Precision on kept fields | 0.955 | 0.955 | 0 | same |
| Coverage (fields kept) | 0.875 | 0.875 | 0 | same |
| Confidence AUC | 0.877 | 0.877 | 0 | same |
| Confidence ECE | 0.057 | 0.057 | 0 | same |
| $ / doc (list price) | $0.00579 | $0.00679 | +17.3% | worse |
| Tokens / doc | 5,884 | 10,857 | +84.5% | worse |
| Latency p50 (s) | 16.08 | 20.43 | +27.1% | worse |
| Latency p95 (s) | 39.94 | 49.39 | +23.7% | worse |
| Escalation rate | - | 1.000 | - |  |

At n=39, one document flips document accuracy by 2.6 pp. Treat accuracy deltas smaller than that as noise.


# Extraction Scorecard — vlm-only-baseline

- Docs: **32**  ·  Set: **test**  ·  Generated: 2026-09-19T02:47:51+00:00
- Document Accuracy Rate: **21.9%**
- Micro F1: **0.708**  ·  Macro F1: **0.711**  ·  Line-item F1: **0.641**
- Hallucination rate (spurious/predicted): **0.0%**
- Latency: p50 6.29s · p95 9.778s · mean 4.935s

## Per-field (normalized)

| field | support | precision | recall | F1 | exact F1 |
|---|--:|--:|--:|--:|--:|
| merchant_name | 20 | 0.750 | 0.450 | 0.562 | 0.562 |
| merchant_address | 13 | 0.429 | 0.231 | 0.300 | 0.300 |
| payment_method | 18 | 0.933 | 0.778 | 0.849 | 0.788 |
| date | 14 | 1.000 | 0.643 | 0.783 | 0.783 |
| currency | 15 | 1.000 | 0.667 | 0.800 | 0.800 |
| subtotal | 22 | 1.000 | 0.545 | 0.706 | 0.529 |
| service_charge | 7 | 1.000 | 0.429 | 0.600 | 0.600 |
| discount | 1 | 1.000 | 1.000 | 1.000 | 1.000 |
| tax | 16 | 1.000 | 0.562 | 0.720 | 0.720 |
| total | 32 | 1.000 | 0.656 | 0.792 | 0.641 |

## Failures by kind

| kind | count |
|---|--:|
| correct | 253 |
| missing | 59 |
| spurious | 0 |
| wrong_value | 4 |
| format_error | 4 |
| swapped_field | 0 |

## Other document types (not pooled into the headline)

The headline above is receipts only. Each type below has its own fields; at small n these are indicative, and their confidence reuses the receipt-fitted model (not calibrated for the type).

| type | n | doc accuracy | micro F1 | line-item F1 | hallucination | swapped fields | abstention precision kept |
|---|--:|--:|--:|--:|--:|--:|--:|
| invoice | 7 | 0.0% | 0.000 | 0.000 | 0.0% | 0 | - |

Per-field (invoice):

| field | support | F1 |
|---|--:|--:|
| invoice_number | 5 | 0.000 |
| purchase_order | 1 | 0.000 |
| vendor.name | 7 | 0.000 |
| vendor.address | 7 | 0.000 |
| vendor.tax_id | 0 | n/a |
| customer.name | 7 | 0.000 |
| customer.address | 5 | 0.000 |
| customer.tax_id | 0 | n/a |
| issue_date | 5 | 0.000 |
| due_date | 2 | 0.000 |
| currency | 7 | 0.000 |
| subtotal | 3 | 0.000 |
| tax | 2 | 0.000 |
| total | 7 | 0.000 |

## By subset

Blended numbers hide where the system fails. Small subsets are noisy; read each against its n.

| subset | n | doc accuracy | micro F1 | line-item F1 | hallucination | top failure kinds |
|---|--:|--:|--:|--:|--:|---|
| degraded | 12 | 16.7% | 0.722 | 0.667 | 0.0% | missing 19, format_error 3, wrong_value 1 |
| not degraded | 27 | 18.5% | 0.514 | 0.463 | 0.0% | missing 98, wrong_value 3, format_error 1 |
| source:cord | 16 | 31.2% | 0.792 | 0.719 | 0.0% | missing 16, wrong_value 2, format_error 1 |
| source:inv | 7 | 0.0% | 0.000 | 0.000 | 0.0% | missing 58 |
| source:wild | 16 | 12.5% | 0.649 | 0.562 | 0.0% | missing 43, format_error 3, wrong_value 2 |

## Failed documents

Scored as empty extractions — a document we could not read is a recall
failure, not one that quietly leaves the denominator.

| doc | error |
|---|---|
| cord_0011 | TruncatedResponse: model=gemini-3.6-flash hit max_tokens=1500 (82 completion tokens); output is incomplete. Raise the *_MAX_TOKENS setting. |
| cord_0013 | TruncatedResponse: model=gemini-3.6-flash hit max_tokens=1500 (59 completion tokens); output is incomplete. Raise the *_MAX_TOKENS setting. |
| cord_0018 | TruncatedResponse: model=gemini-3.6-flash hit max_tokens=1500 (57 completion tokens); output is incomplete. Raise the *_MAX_TOKENS setting. |
| cord_0023 | TruncatedResponse: model=gemini-3.6-flash hit max_tokens=1500 (244 completion tokens); output is incomplete. Raise the *_MAX_TOKENS setting. |
| inv_0001 | TruncatedResponse: model=gemini-3.6-flash hit max_tokens=1500 (52 completion tokens); output is incomplete. Raise the *_MAX_TOKENS setting. |
| inv_0002 | TruncatedResponse: model=gemini-3.6-flash hit max_tokens=1500 (56 completion tokens); output is incomplete. Raise the *_MAX_TOKENS setting. |
| inv_0003 | TruncatedResponse: model=gemini-3.6-flash hit max_tokens=1500 (56 completion tokens); output is incomplete. Raise the *_MAX_TOKENS setting. |
| inv_0004 | TruncatedResponse: model=gemini-3.6-flash hit max_tokens=1500 (54 completion tokens); output is incomplete. Raise the *_MAX_TOKENS setting. |
| inv_0005 | TruncatedResponse: model=gemini-3.6-flash hit max_tokens=1500 (53 completion tokens); output is incomplete. Raise the *_MAX_TOKENS setting. |
| inv_0009 | TruncatedResponse: model=gemini-3.6-flash hit max_tokens=1500 (380 completion tokens); output is incomplete. Raise the *_MAX_TOKENS setting. |
| inv_0010 | TruncatedResponse: model=gemini-3.6-flash hit max_tokens=1500 (53 completion tokens); output is incomplete. Raise the *_MAX_TOKENS setting. |
| wild_0002 | TruncatedResponse: model=gemini-3.6-flash hit max_tokens=1500 (54 completion tokens); output is incomplete. Raise the *_MAX_TOKENS setting. |
| wild_0008 | TruncatedResponse: model=gemini-3.6-flash hit max_tokens=1500 (107 completion tokens); output is incomplete. Raise the *_MAX_TOKENS setting. |
| wild_0011 | TruncatedResponse: model=gemini-3.6-flash hit max_tokens=1500 (53 completion tokens); output is incomplete. Raise the *_MAX_TOKENS setting. |
| wild_0016 | TruncatedResponse: model=gemini-3.6-flash hit max_tokens=1500 (56 completion tokens); output is incomplete. Raise the *_MAX_TOKENS setting. |
| wild_0019 | TruncatedResponse: model=gemini-3.6-flash hit max_tokens=1500 (56 completion tokens); output is incomplete. Raise the *_MAX_TOKENS setting. |
| wild_0022 | TruncatedResponse: model=gemini-3.6-flash hit max_tokens=1500 (57 completion tokens); output is incomplete. Raise the *_MAX_TOKENS setting. |
| wild_0023 | TruncatedResponse: model=gemini-3.6-flash hit max_tokens=1500 (75 completion tokens); output is incomplete. Raise the *_MAX_TOKENS setting. |

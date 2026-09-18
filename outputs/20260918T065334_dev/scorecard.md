# Extraction Scorecard — dev

- Docs: **32**  ·  Set: **test**  ·  Generated: 2026-09-18T06:53:33+00:00
- Document Accuracy Rate: **0.0%**
- Micro F1: **0.000**  ·  Macro F1: **0.000**  ·  Line-item F1: **0.000**
- Hallucination rate (spurious/predicted): **0.0%**
- Latency: p50 0.0s · p95 0.0s · mean 0.0s

## Per-field (normalized)

| field | support | precision | recall | F1 | exact F1 |
|---|--:|--:|--:|--:|--:|
| merchant_name | 16 | 0.000 | 0.000 | 0.000 | 0.000 |
| merchant_address | 11 | 0.000 | 0.000 | 0.000 | 0.000 |
| payment_method | 0 | n/a | n/a | n/a | n/a |
| date | 7 | 0.000 | 0.000 | 0.000 | 0.000 |
| currency | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| subtotal | 22 | 0.000 | 0.000 | 0.000 | 0.000 |
| service_charge | 6 | 0.000 | 0.000 | 0.000 | 0.000 |
| discount | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| tax | 16 | 0.000 | 0.000 | 0.000 | 0.000 |
| total | 32 | 0.000 | 0.000 | 0.000 | 0.000 |

n/a: no ground-truth value and no prediction anywhere in the set, so the field is undefined rather than 0 and is excluded from macro F1.

## Failures by kind

| kind | count |
|---|--:|
| correct | 208 |
| missing | 112 |
| spurious | 0 |
| wrong_value | 0 |
| format_error | 0 |
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
| degraded | 12 | 0.0% | 0.000 | 0.000 | 0.0% | missing 43 |
| not degraded | 27 | 0.0% | 0.000 | 0.000 | 0.0% | missing 127 |
| source:cord | 16 | 0.0% | 0.000 | 0.000 | 0.0% | missing 43 |
| source:inv | 7 | 0.0% | 0.000 | 0.000 | 0.0% | missing 58 |
| source:wild | 16 | 0.0% | 0.000 | 0.000 | 0.0% | missing 69 |

## Failed documents

Scored as empty extractions — a document we could not read is a recall
failure, not one that quietly leaves the denominator.

| doc | error |
|---|---|
| cord_0003 | NotFoundError: Error code: 404 - [{'error': {'code': 404, 'message': 'This model models/gemini-2.5-flash is no longer available to new users. Please update your code to use models/gemini-3.6-flash for the latest features and improvements. We recommend you to use the Interactions API.', 'status': 'NOT_FOUND'}}] |
| cord_0004 | NotFoundError: Error code: 404 - [{'error': {'code': 404, 'message': 'This model models/gemini-2.5-flash is no longer available to new users. Please update your code to use models/gemini-3.6-flash for the latest features and improvements. We recommend you to use the Interactions API.', 'status': 'NOT_FOUND'}}] |
| cord_0005 | NotFoundError: Error code: 404 - [{'error': {'code': 404, 'message': 'This model models/gemini-2.5-flash is no longer available to new users. Please update your code to use models/gemini-3.6-flash for the latest features and improvements. We recommend you to use the Interactions API.', 'status': 'NOT_FOUND'}}] |
| cord_0006 | NotFoundError: Error code: 404 - [{'error': {'code': 404, 'message': 'This model models/gemini-2.5-flash is no longer available to new users. Please update your code to use models/gemini-3.6-flash for the latest features and improvements. We recommend you to use the Interactions API.', 'status': 'NOT_FOUND'}}] |
| cord_0007 | NotFoundError: Error code: 404 - [{'error': {'code': 404, 'message': 'This model models/gemini-2.5-flash is no longer available to new users. Please update your code to use models/gemini-3.6-flash for the latest features and improvements. We recommend you to use the Interactions API.', 'status': 'NOT_FOUND'}}] |
| cord_0008 | NotFoundError: Error code: 404 - [{'error': {'code': 404, 'message': 'This model models/gemini-2.5-flash is no longer available to new users. Please update your code to use models/gemini-3.6-flash for the latest features and improvements. We recommend you to use the Interactions API.', 'status': 'NOT_FOUND'}}] |
| cord_0009 | NotFoundError: Error code: 404 - [{'error': {'code': 404, 'message': 'This model models/gemini-2.5-flash is no longer available to new users. Please update your code to use models/gemini-3.6-flash for the latest features and improvements. We recommend you to use the Interactions API.', 'status': 'NOT_FOUND'}}] |
| cord_0010 | NotFoundError: Error code: 404 - [{'error': {'code': 404, 'message': 'This model models/gemini-2.5-flash is no longer available to new users. Please update your code to use models/gemini-3.6-flash for the latest features and improvements. We recommend you to use the Interactions API.', 'status': 'NOT_FOUND'}}] |
| cord_0011 | NotFoundError: Error code: 404 - [{'error': {'code': 404, 'message': 'This model models/gemini-2.5-flash is no longer available to new users. Please update your code to use models/gemini-3.6-flash for the latest features and improvements. We recommend you to use the Interactions API.', 'status': 'NOT_FOUND'}}] |
| cord_0012 | NotFoundError: Error code: 404 - [{'error': {'code': 404, 'message': 'This model models/gemini-2.5-flash is no longer available to new users. Please update your code to use models/gemini-3.6-flash for the latest features and improvements. We recommend you to use the Interactions API.', 'status': 'NOT_FOUND'}}] |
| cord_0013 | NotFoundError: Error code: 404 - [{'error': {'code': 404, 'message': 'This model models/gemini-2.5-flash is no longer available to new users. Please update your code to use models/gemini-3.6-flash for the latest features and improvements. We recommend you to use the Interactions API.', 'status': 'NOT_FOUND'}}] |
| cord_0017 | NotFoundError: Error code: 404 - [{'error': {'code': 404, 'message': 'This model models/gemini-2.5-flash is no longer available to new users. Please update your code to use models/gemini-3.6-flash for the latest features and improvements. We recommend you to use the Interactions API.', 'status': 'NOT_FOUND'}}] |
| cord_0018 | NotFoundError: Error code: 404 - [{'error': {'code': 404, 'message': 'This model models/gemini-2.5-flash is no longer available to new users. Please update your code to use models/gemini-3.6-flash for the latest features and improvements. We recommend you to use the Interactions API.', 'status': 'NOT_FOUND'}}] |
| cord_0022 | NotFoundError: Error code: 404 - [{'error': {'code': 404, 'message': 'This model models/gemini-2.5-flash is no longer available to new users. Please update your code to use models/gemini-3.6-flash for the latest features and improvements. We recommend you to use the Interactions API.', 'status': 'NOT_FOUND'}}] |
| cord_0023 | NotFoundError: Error code: 404 - [{'error': {'code': 404, 'message': 'This model models/gemini-2.5-flash is no longer available to new users. Please update your code to use models/gemini-3.6-flash for the latest features and improvements. We recommend you to use the Interactions API.', 'status': 'NOT_FOUND'}}] |
| cord_0025 | NotFoundError: Error code: 404 - [{'error': {'code': 404, 'message': 'This model models/gemini-2.5-flash is no longer available to new users. Please update your code to use models/gemini-3.6-flash for the latest features and improvements. We recommend you to use the Interactions API.', 'status': 'NOT_FOUND'}}] |
| inv_0001 | NotFoundError: Error code: 404 - [{'error': {'code': 404, 'message': 'This model models/gemini-2.5-flash is no longer available to new users. Please update your code to use models/gemini-3.6-flash for the latest features and improvements. We recommend you to use the Interactions API.', 'status': 'NOT_FOUND'}}] |
| inv_0002 | NotFoundError: Error code: 404 - [{'error': {'code': 404, 'message': 'This model models/gemini-2.5-flash is no longer available to new users. Please update your code to use models/gemini-3.6-flash for the latest features and improvements. We recommend you to use the Interactions API.', 'status': 'NOT_FOUND'}}] |
| inv_0003 | NotFoundError: Error code: 404 - [{'error': {'code': 404, 'message': 'This model models/gemini-2.5-flash is no longer available to new users. Please update your code to use models/gemini-3.6-flash for the latest features and improvements. We recommend you to use the Interactions API.', 'status': 'NOT_FOUND'}}] |
| inv_0004 | NotFoundError: Error code: 404 - [{'error': {'code': 404, 'message': 'This model models/gemini-2.5-flash is no longer available to new users. Please update your code to use models/gemini-3.6-flash for the latest features and improvements. We recommend you to use the Interactions API.', 'status': 'NOT_FOUND'}}] |
| inv_0005 | NotFoundError: Error code: 404 - [{'error': {'code': 404, 'message': 'This model models/gemini-2.5-flash is no longer available to new users. Please update your code to use models/gemini-3.6-flash for the latest features and improvements. We recommend you to use the Interactions API.', 'status': 'NOT_FOUND'}}] |
| inv_0009 | NotFoundError: Error code: 404 - [{'error': {'code': 404, 'message': 'This model models/gemini-2.5-flash is no longer available to new users. Please update your code to use models/gemini-3.6-flash for the latest features and improvements. We recommend you to use the Interactions API.', 'status': 'NOT_FOUND'}}] |
| inv_0010 | NotFoundError: Error code: 404 - [{'error': {'code': 404, 'message': 'This model models/gemini-2.5-flash is no longer available to new users. Please update your code to use models/gemini-3.6-flash for the latest features and improvements. We recommend you to use the Interactions API.', 'status': 'NOT_FOUND'}}] |
| wild_0002 | NotFoundError: Error code: 404 - [{'error': {'code': 404, 'message': 'This model models/gemini-2.5-flash is no longer available to new users. Please update your code to use models/gemini-3.6-flash for the latest features and improvements. We recommend you to use the Interactions API.', 'status': 'NOT_FOUND'}}] |
| wild_0004 | NotFoundError: Error code: 404 - [{'error': {'code': 404, 'message': 'This model models/gemini-2.5-flash is no longer available to new users. Please update your code to use models/gemini-3.6-flash for the latest features and improvements. We recommend you to use the Interactions API.', 'status': 'NOT_FOUND'}}] |
| wild_0007 | NotFoundError: Error code: 404 - [{'error': {'code': 404, 'message': 'This model models/gemini-2.5-flash is no longer available to new users. Please update your code to use models/gemini-3.6-flash for the latest features and improvements. We recommend you to use the Interactions API.', 'status': 'NOT_FOUND'}}] |
| wild_0008 | NotFoundError: Error code: 404 - [{'error': {'code': 404, 'message': 'This model models/gemini-2.5-flash is no longer available to new users. Please update your code to use models/gemini-3.6-flash for the latest features and improvements. We recommend you to use the Interactions API.', 'status': 'NOT_FOUND'}}] |
| wild_0009 | NotFoundError: Error code: 404 - [{'error': {'code': 404, 'message': 'This model models/gemini-2.5-flash is no longer available to new users. Please update your code to use models/gemini-3.6-flash for the latest features and improvements. We recommend you to use the Interactions API.', 'status': 'NOT_FOUND'}}] |
| wild_0010 | NotFoundError: Error code: 404 - [{'error': {'code': 404, 'message': 'This model models/gemini-2.5-flash is no longer available to new users. Please update your code to use models/gemini-3.6-flash for the latest features and improvements. We recommend you to use the Interactions API.', 'status': 'NOT_FOUND'}}] |
| wild_0011 | NotFoundError: Error code: 404 - [{'error': {'code': 404, 'message': 'This model models/gemini-2.5-flash is no longer available to new users. Please update your code to use models/gemini-3.6-flash for the latest features and improvements. We recommend you to use the Interactions API.', 'status': 'NOT_FOUND'}}] |
| wild_0012 | NotFoundError: Error code: 404 - [{'error': {'code': 404, 'message': 'This model models/gemini-2.5-flash is no longer available to new users. Please update your code to use models/gemini-3.6-flash for the latest features and improvements. We recommend you to use the Interactions API.', 'status': 'NOT_FOUND'}}] |
| wild_0013 | NotFoundError: Error code: 404 - [{'error': {'code': 404, 'message': 'This model models/gemini-2.5-flash is no longer available to new users. Please update your code to use models/gemini-3.6-flash for the latest features and improvements. We recommend you to use the Interactions API.', 'status': 'NOT_FOUND'}}] |
| wild_0014 | NotFoundError: Error code: 404 - [{'error': {'code': 404, 'message': 'This model models/gemini-2.5-flash is no longer available to new users. Please update your code to use models/gemini-3.6-flash for the latest features and improvements. We recommend you to use the Interactions API.', 'status': 'NOT_FOUND'}}] |
| wild_0016 | NotFoundError: Error code: 404 - [{'error': {'code': 404, 'message': 'This model models/gemini-2.5-flash is no longer available to new users. Please update your code to use models/gemini-3.6-flash for the latest features and improvements. We recommend you to use the Interactions API.', 'status': 'NOT_FOUND'}}] |
| wild_0017 | NotFoundError: Error code: 404 - [{'error': {'code': 404, 'message': 'This model models/gemini-2.5-flash is no longer available to new users. Please update your code to use models/gemini-3.6-flash for the latest features and improvements. We recommend you to use the Interactions API.', 'status': 'NOT_FOUND'}}] |
| wild_0019 | NotFoundError: Error code: 404 - [{'error': {'code': 404, 'message': 'This model models/gemini-2.5-flash is no longer available to new users. Please update your code to use models/gemini-3.6-flash for the latest features and improvements. We recommend you to use the Interactions API.', 'status': 'NOT_FOUND'}}] |
| wild_0022 | NotFoundError: Error code: 404 - [{'error': {'code': 404, 'message': 'This model models/gemini-2.5-flash is no longer available to new users. Please update your code to use models/gemini-3.6-flash for the latest features and improvements. We recommend you to use the Interactions API.', 'status': 'NOT_FOUND'}}] |
| wild_0023 | NotFoundError: Error code: 404 - [{'error': {'code': 404, 'message': 'This model models/gemini-2.5-flash is no longer available to new users. Please update your code to use models/gemini-3.6-flash for the latest features and improvements. We recommend you to use the Interactions API.', 'status': 'NOT_FOUND'}}] |
| wild_0025 | NotFoundError: Error code: 404 - [{'error': {'code': 404, 'message': 'This model models/gemini-2.5-flash is no longer available to new users. Please update your code to use models/gemini-3.6-flash for the latest features and improvements. We recommend you to use the Interactions API.', 'status': 'NOT_FOUND'}}] |

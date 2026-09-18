# Extraction Scorecard — dev

- Docs: **12**  ·  Set: **smoke**  ·  Generated: 2026-09-18T06:51:19+00:00
- Document Accuracy Rate: **0.0%**
- Micro F1: **0.000**  ·  Macro F1: **0.000**  ·  Line-item F1: **0.000**
- Hallucination rate (spurious/predicted): **0.0%**
- Latency: p50 0.0s · p95 0.0s · mean 0.0s

## Per-field (normalized)

| field | support | precision | recall | F1 | exact F1 |
|---|--:|--:|--:|--:|--:|
| merchant_name | 6 | 0.000 | 0.000 | 0.000 | 0.000 |
| merchant_address | 6 | 0.000 | 0.000 | 0.000 | 0.000 |
| payment_method | 0 | n/a | n/a | n/a | n/a |
| date | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| currency | 0 | n/a | n/a | n/a | n/a |
| subtotal | 8 | 0.000 | 0.000 | 0.000 | 0.000 |
| service_charge | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| discount | 0 | n/a | n/a | n/a | n/a |
| tax | 10 | 0.000 | 0.000 | 0.000 | 0.000 |
| total | 12 | 0.000 | 0.000 | 0.000 | 0.000 |

n/a: no ground-truth value and no prediction anywhere in the set, so the field is undefined rather than 0 and is excluded from macro F1.

## Failures by kind

| kind | count |
|---|--:|
| correct | 74 |
| missing | 46 |
| spurious | 0 |
| wrong_value | 0 |
| format_error | 0 |
| swapped_field | 0 |

## By subset

Blended numbers hide where the system fails. Small subsets are noisy; read each against its n.

| subset | n | doc accuracy | micro F1 | line-item F1 | hallucination | top failure kinds |
|---|--:|--:|--:|--:|--:|---|
| source:cord | 6 | 0.0% | 0.000 | 0.000 | 0.0% | missing 18 |
| source:wild | 6 | 0.0% | 0.000 | 0.000 | 0.0% | missing 28 |

## Failed documents

Scored as empty extractions — a document we could not read is a recall
failure, not one that quietly leaves the denominator.

| doc | error |
|---|---|
| cord_0001 | NotFoundError: Error code: 404 - [{'error': {'code': 404, 'message': 'This model models/gemini-2.5-flash is no longer available to new users. Please update your code to use models/gemini-3.6-flash for the latest features and improvements. We recommend you to use the Interactions API.', 'status': 'NOT_FOUND'}}] |
| cord_0002 | NotFoundError: Error code: 404 - [{'error': {'code': 404, 'message': 'This model models/gemini-2.5-flash is no longer available to new users. Please update your code to use models/gemini-3.6-flash for the latest features and improvements. We recommend you to use the Interactions API.', 'status': 'NOT_FOUND'}}] |
| cord_0014 | NotFoundError: Error code: 404 - [{'error': {'code': 404, 'message': 'This model models/gemini-2.5-flash is no longer available to new users. Please update your code to use models/gemini-3.6-flash for the latest features and improvements. We recommend you to use the Interactions API.', 'status': 'NOT_FOUND'}}] |
| cord_0015 | NotFoundError: Error code: 404 - [{'error': {'code': 404, 'message': 'This model models/gemini-2.5-flash is no longer available to new users. Please update your code to use models/gemini-3.6-flash for the latest features and improvements. We recommend you to use the Interactions API.', 'status': 'NOT_FOUND'}}] |
| cord_0021 | NotFoundError: Error code: 404 - [{'error': {'code': 404, 'message': 'This model models/gemini-2.5-flash is no longer available to new users. Please update your code to use models/gemini-3.6-flash for the latest features and improvements. We recommend you to use the Interactions API.', 'status': 'NOT_FOUND'}}] |
| cord_0024 | NotFoundError: Error code: 404 - [{'error': {'code': 404, 'message': 'This model models/gemini-2.5-flash is no longer available to new users. Please update your code to use models/gemini-3.6-flash for the latest features and improvements. We recommend you to use the Interactions API.', 'status': 'NOT_FOUND'}}] |
| wild_0001 | NotFoundError: Error code: 404 - [{'error': {'code': 404, 'message': 'This model models/gemini-2.5-flash is no longer available to new users. Please update your code to use models/gemini-3.6-flash for the latest features and improvements. We recommend you to use the Interactions API.', 'status': 'NOT_FOUND'}}] |
| wild_0005 | NotFoundError: Error code: 404 - [{'error': {'code': 404, 'message': 'This model models/gemini-2.5-flash is no longer available to new users. Please update your code to use models/gemini-3.6-flash for the latest features and improvements. We recommend you to use the Interactions API.', 'status': 'NOT_FOUND'}}] |
| wild_0006 | NotFoundError: Error code: 404 - [{'error': {'code': 404, 'message': 'This model models/gemini-2.5-flash is no longer available to new users. Please update your code to use models/gemini-3.6-flash for the latest features and improvements. We recommend you to use the Interactions API.', 'status': 'NOT_FOUND'}}] |
| wild_0015 | NotFoundError: Error code: 404 - [{'error': {'code': 404, 'message': 'This model models/gemini-2.5-flash is no longer available to new users. Please update your code to use models/gemini-3.6-flash for the latest features and improvements. We recommend you to use the Interactions API.', 'status': 'NOT_FOUND'}}] |
| wild_0020 | NotFoundError: Error code: 404 - [{'error': {'code': 404, 'message': 'This model models/gemini-2.5-flash is no longer available to new users. Please update your code to use models/gemini-3.6-flash for the latest features and improvements. We recommend you to use the Interactions API.', 'status': 'NOT_FOUND'}}] |
| wild_0024 | NotFoundError: Error code: 404 - [{'error': {'code': 404, 'message': 'This model models/gemini-2.5-flash is no longer available to new users. Please update your code to use models/gemini-3.6-flash for the latest features and improvements. We recommend you to use the Interactions API.', 'status': 'NOT_FOUND'}}] |

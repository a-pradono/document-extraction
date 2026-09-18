# Extraction Scorecard — groq-pipeline

- Docs: **32**  ·  Set: **test**  ·  Generated: 2026-09-18T16:57:14+00:00
- Document Accuracy Rate: **15.6%**
- Micro F1: **0.866**  ·  Macro F1: **0.870**  ·  Line-item F1: **0.951**
- Hallucination rate (spurious/predicted): **7.6%**
- Latency: p50 29.189s · p95 49.455s · mean 28.462s

## Per-field (normalized)

| field | support | precision | recall | F1 | exact F1 |
|---|--:|--:|--:|--:|--:|
| merchant_name | 20 | 0.800 | 0.800 | 0.800 | 0.700 |
| merchant_address | 13 | 0.462 | 0.462 | 0.462 | 0.462 |
| payment_method | 18 | 0.833 | 0.833 | 0.833 | 0.500 |
| date | 14 | 1.000 | 1.000 | 1.000 | 1.000 |
| currency | 15 | 0.556 | 1.000 | 0.714 | 0.714 |
| subtotal | 22 | 0.955 | 0.955 | 0.955 | 0.636 |
| service_charge | 7 | 1.000 | 1.000 | 1.000 | 0.857 |
| discount | 1 | 1.000 | 1.000 | 1.000 | 1.000 |
| tax | 16 | 0.938 | 0.938 | 0.938 | 0.938 |
| total | 32 | 1.000 | 1.000 | 1.000 | 0.812 |

## Failures by kind

| kind | count |
|---|--:|
| correct | 290 |
| missing | 2 |
| spurious | 13 |
| wrong_value | 8 |
| format_error | 6 |
| swapped_field | 1 |

## Other document types (not pooled into the headline)

The headline above is receipts only. Each type below has its own fields; at small n these are indicative, and their confidence reuses the receipt-fitted model (not calibrated for the type).

| type | n | doc accuracy | micro F1 | line-item F1 | hallucination | swapped fields | abstention precision kept |
|---|--:|--:|--:|--:|--:|--:|--:|
| invoice | 7 | 14.3% | 0.750 | 0.857 | 3.7% | 1 | 0.887 |

Per-field (invoice):

| field | support | F1 |
|---|--:|--:|
| invoice_number | 5 | 0.889 |
| purchase_order | 1 | 0.667 |
| vendor.name | 7 | 0.923 |
| vendor.address | 7 | 0.462 |
| vendor.tax_id | 0 | n/a |
| customer.name | 7 | 0.462 |
| customer.address | 5 | 0.222 |
| customer.tax_id | 0 | n/a |
| issue_date | 5 | 0.909 |
| due_date | 2 | 1.000 |
| currency | 7 | 0.923 |
| subtotal | 3 | 0.857 |
| tax | 2 | 1.000 |
| total | 7 | 0.923 |

## By subset

Blended numbers hide where the system fails. Small subsets are noisy; read each against its n.

| subset | n | doc accuracy | micro F1 | line-item F1 | hallucination | top failure kinds |
|---|--:|--:|--:|--:|--:|---|
| degraded | 12 | 25.0% | 0.872 | 0.977 | 3.4% | format_error 3, missing 2, spurious 2 |
| not degraded | 27 | 11.1% | 0.824 | 0.914 | 7.9% | spurious 13, wrong_value 10, format_error 8 |
| source:cord | 16 | 25.0% | 0.872 | 0.952 | 15.3% | spurious 11, wrong_value 2, format_error 1 |
| source:inv | 7 | 14.3% | 0.750 | 0.857 | 3.7% | missing 7, format_error 5, wrong_value 4 |
| source:wild | 16 | 6.2% | 0.862 | 0.949 | 2.0% | wrong_value 6, format_error 5, missing 2 |

## Confidence calibration

- AUC (discrimination): **0.9335** — ~0.5 means the score carries no information, and rescaling cannot fix that.
- ECE (calibration): **0.0863** — high ECE with good AUC is the fixable case: refit on the calibration split.

| conf range | n | accuracy |
|---|--:|--:|
| [0.0,0.5) | 14 | 0.071 |
| [0.5,0.7) | 33 | 0.303 |
| [0.7,0.9) | 52 | 0.808 |
| [0.9,1.01) | 253 | 0.976 |

## Abstention

At threshold **0.85**, keeping 78.1% of 352 fields moves precision **0.852 -> 0.975** (+0.122).

| threshold | coverage | precision |
|--:|--:|--:|
| 0.0 | 1.000 | 0.852 |
| 0.1 | 1.000 | 0.852 |
| 0.2 | 1.000 | 0.852 |
| 0.3 | 0.963 | 0.882 |
| 0.4 | 0.960 | 0.885 |
| 0.5 | 0.960 | 0.885 |
| 0.6 | 0.955 | 0.887 |
| 0.7 | 0.867 | 0.948 |
| 0.8 | 0.818 | 0.969 |
| 0.9 | 0.719 | 0.976 |

## Cost and latency

Cost basis: list-price equivalent (free tier: actual spend $0).

| metric | value |
|---|--:|
| cost / doc | $0.00537 |
| tokens / doc | 7407 |
| latency p50 / p95 (n=38) | 29.752s / 62.554s |

| stage | calls | tokens in | tokens out | $ | mean latency |
|---|--:|--:|--:|--:|--:|
| model_a | 38 | 117850 | 53644 | 0.16947 | 16.757s |
| model_b | 38 | 69745 | 40233 | 0.03460 | 16.35s |

## Failed documents

Scored as empty extractions — a document we could not read is a recall
failure, not one that quietly leaves the denominator.

| doc | error |
|---|---|
| inv_0010 | TruncatedResponse: model=gemini-3.6-flash hit max_tokens=24576 (20555 completion tokens); output is incomplete. Raise the *_MAX_TOKENS setting. |

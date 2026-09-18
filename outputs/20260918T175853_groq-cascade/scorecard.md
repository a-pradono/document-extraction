# Extraction Scorecard — groq-cascade

- Docs: **32**  ·  Set: **test**  ·  Generated: 2026-09-18T17:58:52+00:00
- Document Accuracy Rate: **21.9%**
- Micro F1: **0.918**  ·  Macro F1: **0.917**  ·  Line-item F1: **0.951**
- Hallucination rate (spurious/predicted): **0.6%**
- Latency: p50 18.297s · p95 29.526s · mean 18.876s

## Per-field (normalized)

| field | support | precision | recall | F1 | exact F1 |
|---|--:|--:|--:|--:|--:|
| merchant_name | 20 | 0.850 | 0.850 | 0.850 | 0.750 |
| merchant_address | 13 | 0.538 | 0.538 | 0.538 | 0.538 |
| payment_method | 18 | 0.889 | 0.889 | 0.889 | 0.667 |
| date | 14 | 1.000 | 1.000 | 1.000 | 1.000 |
| currency | 15 | 1.000 | 1.000 | 1.000 | 1.000 |
| subtotal | 22 | 0.955 | 0.955 | 0.955 | 0.636 |
| service_charge | 7 | 1.000 | 1.000 | 1.000 | 0.857 |
| discount | 1 | 1.000 | 1.000 | 1.000 | 1.000 |
| tax | 16 | 0.938 | 0.938 | 0.938 | 0.938 |
| total | 32 | 1.000 | 1.000 | 1.000 | 0.812 |

## Failures by kind

| kind | count |
|---|--:|
| correct | 305 |
| missing | 2 |
| spurious | 1 |
| wrong_value | 6 |
| format_error | 5 |
| swapped_field | 1 |

## Other document types (not pooled into the headline)

The headline above is receipts only. Each type below has its own fields; at small n these are indicative, and their confidence reuses the receipt-fitted model (not calibrated for the type).

| type | n | doc accuracy | micro F1 | line-item F1 | hallucination | swapped fields | abstention precision kept |
|---|--:|--:|--:|--:|--:|--:|--:|
| invoice | 7 | 0.0% | 0.800 | 1.000 | 4.8% | 1 | 0.856 |

Per-field (invoice):

| field | support | F1 |
|---|--:|--:|
| invoice_number | 5 | 1.000 |
| purchase_order | 1 | 0.667 |
| vendor.name | 7 | 1.000 |
| vendor.address | 7 | 0.429 |
| vendor.tax_id | 0 | 0.000 |
| customer.name | 7 | 0.571 |
| customer.address | 5 | 0.400 |
| customer.tax_id | 0 | n/a |
| issue_date | 5 | 0.909 |
| due_date | 2 | 1.000 |
| currency | 7 | 1.000 |
| subtotal | 3 | 0.857 |
| tax | 2 | 1.000 |
| total | 7 | 1.000 |

## By subset

Blended numbers hide where the system fails. Small subsets are noisy; read each against its n.

| subset | n | doc accuracy | micro F1 | line-item F1 | hallucination | top failure kinds |
|---|--:|--:|--:|--:|--:|---|
| degraded | 12 | 16.7% | 0.897 | 0.977 | 1.7% | format_error 3, missing 2, spurious 1 |
| not degraded | 27 | 18.5% | 0.881 | 0.951 | 1.8% | wrong_value 9, format_error 8, spurious 3 |
| source:cord | 16 | 31.2% | 0.943 | 0.952 | 1.6% | wrong_value 2, spurious 1, format_error 1 |
| source:inv | 7 | 0.0% | 0.800 | 1.000 | 4.8% | format_error 6, wrong_value 4, spurious 3 |
| source:wild | 16 | 12.5% | 0.902 | 0.949 | 0.0% | wrong_value 4, format_error 4, missing 2 |

## Confidence calibration

- AUC (discrimination): **0.8771** — ~0.5 means the score carries no information, and rescaling cannot fix that.
- ECE (calibration): **0.0573** — high ECE with good AUC is the fixable case: refit on the calibration split.

| conf range | n | accuracy |
|---|--:|--:|
| [0.0,0.5) | 13 | 0.538 |
| [0.5,0.7) | 19 | 0.368 |
| [0.7,0.9) | 147 | 0.912 |
| [0.9,1.01) | 173 | 0.988 |

## Abstention

At threshold **0.75**, keeping 87.5% of 352 fields moves precision **0.906 -> 0.955** (+0.048).

| threshold | coverage | precision |
|--:|--:|--:|
| 0.0 | 1.000 | 0.906 |
| 0.1 | 1.000 | 0.906 |
| 0.2 | 1.000 | 0.906 |
| 0.3 | 0.997 | 0.909 |
| 0.4 | 0.994 | 0.909 |
| 0.5 | 0.963 | 0.920 |
| 0.6 | 0.909 | 0.953 |
| 0.7 | 0.909 | 0.953 |
| 0.8 | 0.855 | 0.960 |
| 0.9 | 0.491 | 0.988 |

## Cost and latency

Cost basis: list-price equivalent (free tier: actual spend $0).

| metric | value |
|---|--:|
| cost / doc | $0.00679 |
| tokens / doc | 10857 |
| latency p50 / p95 (n=39) | 20.434s / 49.388s |

| stage | calls | tokens in | tokens out | $ | mean latency |
|---|--:|--:|--:|--:|--:|
| cascade | 39 | 128184 | 65751 | 0.03912 | 5.461s |
| model_a | 39 | 128184 | 71234 | 0.21654 | 21.149s |
| model_b | 12 | 19572 | 10505 | 0.00924 | 2.64s |

## Cascade

Escalated **39 of 39** documents (100.0%) to the two-model path.

| path | docs | field accuracy |
|---|--:|--:|
| accepted on cheap path | 0 | - |
| escalated | 39 | 0.931 |

Accepted documents scoring well below escalated ones means the gate is
letting bad reads through: the saving is being paid for in accuracy.

| escalation reason | docs |
|---|--:|
| low_confidence | 38 |
| ungrounded | 9 |
| arithmetic_failed | 3 |
| invalid_output | 1 |

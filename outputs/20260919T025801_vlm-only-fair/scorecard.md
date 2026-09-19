# Extraction Scorecard — vlm-only-fair

- Docs: **32**  ·  Set: **test**  ·  Generated: 2026-09-19T02:58:00+00:00
- Document Accuracy Rate: **40.6%**
- Micro F1: **0.918**  ·  Macro F1: **0.917**  ·  Line-item F1: **0.948**
- Hallucination rate (spurious/predicted): **0.6%**
- Latency: p50 8.281s · p95 18.728s · mean 9.624s

## Per-field (normalized)

| field | support | precision | recall | F1 | exact F1 |
|---|--:|--:|--:|--:|--:|
| merchant_name | 20 | 0.800 | 0.800 | 0.800 | 0.750 |
| merchant_address | 13 | 0.538 | 0.538 | 0.538 | 0.538 |
| payment_method | 18 | 0.944 | 0.944 | 0.944 | 0.778 |
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
| wrong_value | 5 |
| format_error | 6 |
| swapped_field | 1 |

## Other document types (not pooled into the headline)

The headline above is receipts only. Each type below has its own fields; at small n these are indicative, and their confidence reuses the receipt-fitted model (not calibrated for the type).

| type | n | doc accuracy | micro F1 | line-item F1 | hallucination | swapped fields | abstention precision kept |
|---|--:|--:|--:|--:|--:|--:|--:|
| invoice | 7 | 14.3% | 0.803 | 0.967 | 1.7% | 1 | - |

Per-field (invoice):

| field | support | F1 |
|---|--:|--:|
| invoice_number | 5 | 1.000 |
| purchase_order | 1 | 1.000 |
| vendor.name | 7 | 1.000 |
| vendor.address | 7 | 0.462 |
| vendor.tax_id | 0 | n/a |
| customer.name | 7 | 0.571 |
| customer.address | 5 | 0.182 |
| customer.tax_id | 0 | n/a |
| issue_date | 5 | 1.000 |
| due_date | 2 | 1.000 |
| currency | 7 | 1.000 |
| subtotal | 3 | 0.857 |
| tax | 2 | 1.000 |
| total | 7 | 1.000 |

## By subset

Blended numbers hide where the system fails. Small subsets are noisy; read each against its n.

| subset | n | doc accuracy | micro F1 | line-item F1 | hallucination | top failure kinds |
|---|--:|--:|--:|--:|--:|---|
| degraded | 12 | 41.7% | 0.904 | 0.977 | 0.0% | format_error 3, missing 2, wrong_value 1 |
| not degraded | 27 | 33.3% | 0.880 | 0.940 | 1.2% | wrong_value 10, format_error 7, spurious 2 |
| source:cord | 16 | 50.0% | 0.959 | 0.952 | 1.6% | spurious 1, wrong_value 1, format_error 1 |
| source:inv | 7 | 14.3% | 0.803 | 0.967 | 1.7% | wrong_value 6, format_error 4, missing 1 |
| source:wild | 16 | 31.2% | 0.891 | 0.944 | 0.0% | format_error 5, wrong_value 4, missing 2 |

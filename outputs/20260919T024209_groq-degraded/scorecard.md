# Extraction Scorecard — groq-degraded

- Docs: **12**  ·  Set: **degraded**  ·  Generated: 2026-09-19T02:42:09+00:00
- Document Accuracy Rate: **16.7%**
- Micro F1: **0.897**  ·  Macro F1: **0.885**  ·  Line-item F1: **0.977**
- Hallucination rate (spurious/predicted): **1.7%**
- Latency: p50 16.422s · p95 22.894s · mean 16.818s

## Per-field (normalized)

| field | support | precision | recall | F1 | exact F1 |
|---|--:|--:|--:|--:|--:|
| merchant_name | 7 | 0.714 | 0.714 | 0.714 | 0.714 |
| merchant_address | 4 | 0.250 | 0.250 | 0.250 | 0.250 |
| payment_method | 4 | 1.000 | 1.000 | 1.000 | 1.000 |
| date | 5 | 1.000 | 1.000 | 1.000 | 1.000 |
| currency | 6 | 1.000 | 1.000 | 1.000 | 1.000 |
| subtotal | 9 | 0.889 | 0.889 | 0.889 | 0.444 |
| service_charge | 5 | 1.000 | 1.000 | 1.000 | 1.000 |
| discount | 1 | 1.000 | 1.000 | 1.000 | 1.000 |
| tax | 5 | 1.000 | 1.000 | 1.000 | 1.000 |
| total | 12 | 1.000 | 1.000 | 1.000 | 0.750 |

## Failures by kind

| kind | count |
|---|--:|
| correct | 112 |
| missing | 2 |
| spurious | 1 |
| wrong_value | 1 |
| format_error | 3 |
| swapped_field | 1 |

## By subset

Blended numbers hide where the system fails. Small subsets are noisy; read each against its n.

| subset | n | doc accuracy | micro F1 | line-item F1 | hallucination | top failure kinds |
|---|--:|--:|--:|--:|--:|---|
| source:cord | 6 | 16.7% | 0.936 | 0.955 | 4.2% | spurious 1, format_error 1 |
| source:wild | 6 | 16.7% | 0.870 | 1.000 | 0.0% | missing 2, format_error 2, wrong_value 1 |

## Confidence calibration

- AUC (discrimination): **0.8465** — ~0.5 means the score carries no information, and rescaling cannot fix that.
- ECE (calibration): **0.0451** — high ECE with good AUC is the fixable case: refit on the calibration split.

| conf range | n | accuracy |
|---|--:|--:|
| [0.0,0.5) | 9 | 0.556 |
| [0.5,0.7) | 3 | 0.333 |
| [0.7,0.9) | 32 | 0.812 |
| [0.9,1.01) | 88 | 0.977 |

## Abstention

At threshold **0.75**, keeping 87.1% of 132 fields moves precision **0.894 -> 0.948** (+0.054).

| threshold | coverage | precision |
|--:|--:|--:|
| 0.0 | 1.000 | 0.894 |
| 0.1 | 1.000 | 0.894 |
| 0.2 | 1.000 | 0.894 |
| 0.3 | 1.000 | 0.894 |
| 0.4 | 0.992 | 0.893 |
| 0.5 | 0.932 | 0.919 |
| 0.6 | 0.909 | 0.933 |
| 0.7 | 0.909 | 0.933 |
| 0.8 | 0.864 | 0.956 |
| 0.9 | 0.667 | 0.977 |

## Cost and latency

Cost basis: list-price equivalent (free tier: actual spend $0).

| metric | value |
|---|--:|
| cost / doc | $0.00424 |
| tokens / doc | 5861 |
| latency p50 / p95 (n=12) | 16.082s / 22.894s |

| stage | calls | tokens in | tokens out | $ | mean latency |
|---|--:|--:|--:|--:|--:|
| model_a | 12 | 34002 | 13490 | 0.04393 | 13.703s |
| model_b | 9 | 14852 | 7985 | 0.00702 | 4.154s |

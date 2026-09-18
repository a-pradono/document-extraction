# Receipt & Invoice Extraction — End-to-End Report

> **Status, 18 September 2026.** The system is built and tested (435 unit tests, offline
> end-to-end 7/7). **Ground truth is fully verified — 60 of 60 documents checked against
> their images by a human**, a pass that corrected five money-field errors (§2.4). The
> **confidence model is fitted** on a real run of the calibration split (§4.2). **The
> `test` split has not been run yet**, so there is still no accuracy scorecard: §4.5 is
> empty and says so. Nothing in §4 is a placeholder — every table is either a measurement
> or explicitly marked not-yet-run.

---

## 1. Introduction

### The problem

A client needs structured data pulled out of a stack of real documents — scanned,
photographed, multi-page, inconsistently laid out. Some fields are simple lookups
(a merchant name, a date). Others require reasoning across sections: a total that must
reconcile against line items and tax, a party's role in a transaction, a table of line
items that spans a page break.

### What this system is optimized for

Not a demo that works on a clean PDF. The design target is a system whose accuracy can
be **proven**, that **knows when it is unsure**, and whose quality, latency and cost can
be reasoned about. Three properties drove every decision:

1. **It must not hallucinate.** A field absent from the document returns `null`. No
   value is invented, and every value kept is traceable to text found in the document.
2. **It must abstain rather than guess.** Per-field confidence is computed and fields
   below a fitted threshold are withheld at the API boundary rather than asserted.
3. **Evaluation is the primary deliverable.** The harness was built before any prompt
   tuning, so every change is judged by re-running a scorecard. Section 4.3 shows this
   paying off: the harness caught three defects in its own ground truth and one in
   itself, each of which would have produced confident, wrong numbers.

### Scope and honest limitations

| | |
|---|---|
| Primary document type | Receipts — **50 documents** |
| Secondary document type | Invoices — **10 documents**, real, 1–5 pages |
| Third receipt tier (Kaggle retail, 15 docs) | **Dropped.** Needs an API token and annotates neither subtotal nor tax |
| Own scans | **None.** Forfeits a stated bonus; see §2.5 |
| Models | Hosted, not open-weights — forfeits the self-hosting bonus; see §3.5 |
| Ground truth | **60 of 60 human-verified**, five label errors corrected |
| Reported accuracy | **None yet** — the `test` split has not been run |

---

## 2. Data

60 documents, three sources, two document types, four languages
(Indonesian, English, Spanish, French/Dutch), three independent label schemas.

### 2.1 Why receipts as the primary type

Receipts exercise the hardest requirement — **verifiable cross-section reasoning** —
better than any alternative at this scale. A receipt carries a line-item table, a
subtotal, a tax amount and a total, and those must reconcile. That gives a **model-free
ground-truth signal**: `subtotal + service − discount + tax == total` either holds or it
does not, whatever any model claims. That property is the foundation of the confidence
design in §3.3.

The trade-off, stated plainly: receipts are natively single-page with one counterparty,
so they cannot exercise multi-page reading or party-role resolution. Hence invoices.

### 2.2 Receipt sources (50 documents)

| Source | Docs | Role | Labels |
|---|---:|---|---|
| **CORD v2** (`naver-clova-ix/cord-v2`, CC BY 4.0) | 25 | Structured backbone — line items with quantity and unit price, subtotal → total | Script-converted from the dataset, header fields hand-drafted from the images |
| **WildReceipt** (MMOCR, Apache-2.0) | 25 | Template diversity — photographed in the wild, unseen templates | Script-converted, re-spaced and extended by hand |

**Small and trustworthy over large and noisy.** Trustworthy ground truth is the
highest-weighted evaluation criterion, and a larger set with unverified labels produces
metrics that cannot be trusted — while costing proportionally more per eval run.

**Stratified sampling, not first-N.** CORD holds 1,000 receipts; taking the first 25
would produce an accidentally easy set. Sampling is stratified by line-item count and
spread across merchants, with the chosen ids pinned so a converter fix re-converts the
*same* documents rather than reshuffling the splits.

**CORD is anonymised at source.** The dataset blurs the merchant header on nearly every
receipt, so most CORD merchant, address and date labels are explicit `null` — which is
itself a useful annotation: it turns "did the model invent a merchant?" into a scorable
question. Four headers remain legible through the blur (PIZZA HUT, Solaria, XXI Cafe,
and a Korean logo) and are labelled per the rule *label what a human can read*.

### 2.3 Invoices (10 documents)

Added to cover the two requirements receipts structurally cannot: **multi-page reading**
and **party-role resolution** (vendor vs customer, where a swap surfaces as the
`swapped_field` error label).

**Source: RVL-CDIP-N multipage** (`jordyvl/rvl_cdip_n_mp`, invoice class; Larson et al.,
NeurIPS 2022 Datasets & Benchmarks; **CC BY-NC 4.0, non-commercial**). Ten real
invoices, 1–5 pages each, obtained through public-records requests: a police department,
two TV stations, a school district, law firms, a state agency, small manufacturers.
`scripts/fetch_invoices.py` streams the archive and extracts only the ten pinned files,
each verified by SHA-256.

**A synthetic invoice set was evaluated and rejected.** The candidate was one template
filled with Faker data: totals that did not reconcile, and every document a single page.
It could not test either property invoices were added for. The rejection is recorded in
`docs/DATASETS.md` with the evidence.

**Real documents bring real mess**, all recorded in `_provenance.json`:

- `inv_0004` prints an arithmetic error — rows sum to 56.56 against a printed 36.59.
  Kept as-is: a document whose printed total is genuinely wrong is a valuable test case.
- `inv_0005` has a row struck through by hand with "duplicate invoicing" written beside
  it, while the printed total still includes it. Labelled as printed.
- `inv_0006` is three documents in one PDF: invoice 29069 (page 1), a different invoice
  28815 (page 2) and credit memo 3094 (page 3). The label covers invoice 29069 only.
  **Kept deliberately.** The pipeline sends every page in one call and expects one
  document back, so the model may merge rows or totals across the three and be marked
  wrong for it - the failure is in the one-document-per-file premise, not in the reading.
  It sits in `calibration`, which fits on receipts only, so it affects no reported
  number; splitting a stack into N documents is scoped as a separate feature in
  `docs/OUTPUT.md` §2, not a change to `/extract`.
- `inv_0002` is the sharpest role trap in the set: the customer is the billing agency,
  not the advertiser whose campaign it is.
- `inv_0010` runs 225 line items across 5 pages; the transcribed rows sum exactly to the
  printed $59,403.38.

n=10 is deliberately small: enough to prove the system generalises to a second document
type, not enough to pretend invoices are a co-equal dataset. Invoice results are
reported as a **separate breakdown, never pooled** into the receipt headline, and the
confidence model is fitted on receipts only — invoice confidence is **uncalibrated**.

**Privacy and licence.** These are real public records naming real people and
organisations, under a non-commercial licence. The repository is private; published
views carry the dataset attribution.

### 2.4 Verification — what the human pass actually found

**60 of 60 documents verified.** Every label was checked against its image, and the pass
found **five money-field errors**, all in WildReceipt, none of which the model-free audit
could resolve on its own. The audit could say *this does not reconcile*; only a human
could say what the right number was.

| Document | Defect | After correction |
|---|---|---|
| `wild_0003` | Subtotal `675.00` for a printed `Rs1,675.00`; one of two printed taxes (VAT 13.13%, Service TAX 4.94%) dropped | 1,675.00 + 82.75 + 213.36 = **1,971.11**, the printed Cash amount |
| `wild_0019` | Subtotal `97.30` for a printed `87.30`; the printed 10% service charge missing entirely | 87.30 + 8.73 + 6.72 = **102.75**, the printed TOTAL |
| `wild_0018` | A row annotated `5390` for a printed `5.90` — **a defect in the published dataset**, not in our conversion | 6 rows sum to exactly **47.30**, the printed SUBTTL |
| `wild_0015` | Three printed coupons (−0.75, −0.25, −1.00) dropped, so rows summed to 5.00 against a printed total of 3.00 | 5.00 − 2.00 = **3.00** |
| `wild_0004` | One fuel purchase split across three junk rows: `REGULAR/15863`, a bare `3479`, a bare `55.19` — gallons, price per gallon and fuel total | 15.863 gal × $3.479 = $55.19, plus $10.00 = **$65.19** |

That is a **10% error rate in the money fields of one source**, and every one of those
errors would have been charged to the model: a correct extraction scored as
`wrong_value` on exactly the fields the confidence design leans on hardest. It is the
empirical case for "small and verified" over "large and noisy", and the reason this
project audits its ground truth rather than assuming it.

**Why verify all 60 rather than sample?** Sampling is the right tool for estimating a
label error rate across a corpus too large to check. It is the wrong tool for an
evaluation set, because this set is not the product — it is the **measuring instrument**
for every number in §4. A sample of 15 would have reported "roughly 10% of WildReceipt
money fields are wrong" and left us not knowing *which* five, so the instrument stays
bent and we merely know that it is.

The bias also does not average out. A wrong label converts a **correct** extraction into
a scored error, so the five defects pushed one way only, on `subtotal`, `tax` and `total`
— the fields the confidence design leans on hardest. For scale: document accuracy has a
noise floor of about ±8pp at n=39, and a plausible improvement between two versions is
3–5pp. The label error was **larger than the effects the harness exists to detect**.

What makes full verification affordable is the deliberate choice of a small set: 60
documents is roughly 40 minutes once, amortised over every run for the life of the
project. At 10,000 documents it would be a month of work, and sampling would be the only
rational option. The policy this project would apply at scale:

| Corpus | Approach |
|---|---|
| Evaluation set, n ≤ a few hundred | Verify 100% — it is the ruler |
| Training corpus, n large | Sample to estimate noise, report metrics with that band, fix systematically |
| Production output | Sample continuously — that is monitoring, not ground truth |

Two honest qualifications. First, the model-free audit can only flag fields it can check:
it says "this does not reconcile", but it has no signal whatsoever on merchant name, date
or currency, where a sample would yield an error rate and nothing more. Second, not every
field needed the full pass — the 10 invoice labels and the mostly-null CORD header fields
would have been well served by an audit-triaged sample. The money fields were not, and
that is where all five errors were.

Judgement calls are recorded per document in `_provenance.json`, including those the
drafter could not resolve and the reviewer settled: ambiguous `$` currencies (AUD vs SGD
vs USD, inferred from tax registrations), day-first vs month-first dates, and
`cord_0019`, where the merchant name is blurred by the dataset's own anonymisation.

### 2.5 No own scans — and the compensation

None were available, so that bonus is forfeited. Two substitutes:

1. **Source diversity** — three sources, two document types, four languages, three
   independent label schemas. Per-source breakdowns test generalisation in a way one
   dataset cannot.
2. **Synthetic degradation** — 25 degraded copies (blur, rotation, JPEG, fade) scored
   against the **unchanged** ground truth. Clearly labelled synthetic, excluded from the
   headline scorecard structurally rather than by convention.

### 2.6 Splits (seeded, committed)

| Split | n | CORD | Wild | Invoices | Purpose |
|---|---:|---:|---:|---:|---|
| `calibration` | 21 | 9 | 9 | 3 | Fit confidence weights and the threshold. **Never reported** |
| `test` | 39 | 16 | 16 | 7 | The reported scorecard |
| `smoke` | 12 | 6 | 6 | 0 | Fast dev loop — drawn **from** `calibration` |
| `degraded` | 12 | 6 | 6 | 0 | A *tag* on the worst documents already inside `test` |
| `invoices` | 10 | — | — | 10 | All invoices, for the per-type breakdown |

Two rules protect the numbers: `smoke` comes from `calibration`, never `test`; and
nothing is fitted on `test`. Each source draws from its own seeded RNG, so adding the
invoices moved no receipt between splits — verified by test.

---

## 3. Method

### 3.1 Architecture

```
image          ──► Model A (VLM)  ──► { transcription, document, _confidence }
transcription  ──► Model B (LLM)  ──► document  (independent re-extraction)
                   Python validation (arithmetic / format — model-free)
                                   ──► per-field confidence ──► abstention
```

| Role | Model | Why |
|---|---|---|
| Model A — VLM reader | `gemini-3.6-flash` | Layout is the problem: "which of these five numbers is the total" is answered by position, which flat OCR destroys |
| Model B — LLM reasoner | `openai/gpt-oss-120b` (Groq) | Cross-section reasoning and normalisation, text-only and cheap. Open-weights, and a **different family from Model A on purpose** |
| Cascade first pass | `gemini-3.5-flash-lite` | Cheap read; escalates to the full pipeline on doubt (§4.5) |

**Different families, deliberately.** Sibling models fail identically on the same
smudged digit, which would inflate the agreement signal exactly where it must be honest.

**The transcription replaces a separate OCR stage** and serves as the groundedness
source, so the anti-hallucination signal survives without a local OCR dependency. A
PaddleOCR path was built and abandoned: too slow on a CPU-only machine.

**Multi-page** documents are rendered page by page and sent in **one** call, so the
model sees the whole document; `MAX_PAGES` (5) refuses anything longer rather than
silently reading part of it.

### 3.2 Typed, validated output

Model output is parsed into Pydantic models (`Receipt`, `Invoice`, sharing `LineItem`).
Money is `Decimal`, never `float`, and crosses the JSON boundary as a **string**. Dates
are `YYYY-MM-DD`. Absent fields are `null` — never an empty string, a zero or a
plausible guess. Malformed JSON gets one corrective retry showing the model its own bad
output; a truncated response fails loudly as `TruncatedResponse` rather than as a JSON
parse error; per-field salvage keeps the fields that did parse.

*Implementation note:* `Receipt.date` originally shadowed the `datetime.date` import
inside its own annotation, so Pydantic resolved the type to `NoneType` and silently
rejected **every** date — which would have zeroed date F1 with no visible error. Fixed
and covered by a regression test. Worth recording because the failure was silent.

### 3.3 Confidence: four signals, fitted weights

| Signal | Strength | What it knows |
|---|---|---|
| `validated` | **Strongest** | Deterministic arithmetic — reconciliation, rows multiply out, rows sum to subtotal, date parses, currency is ISO-4217, and for invoices due ≥ issue and vendor ≠ customer. **Model-free, cannot be bluffed** |
| `grounded` | Strong | Does the value appear in the transcription, type-aware and whole-token? |
| `agreement` | Moderate | Do Model A and Model B agree after normalisation? Row-level F1 for tables |
| `self_report` | **Weakest** | The model's stated certainty |

**Known limitation:** Model B reads Model A's transcription, so they are only partially
independent — a misread digit propagates and inflates agreement. This is precisely why
`validated` exists: it is independent of the reading step. True reading independence
needs a second vision model on the raw image (§5).

Weights are **fitted per field group** by logistic regression on the calibration split;
groups with too few observations keep default weights rather than overfit. The threshold
is read off a **risk–coverage curve** against a target precision, never hand-picked.
Real fitted values are in §4.2.

### 3.4 Evaluation harness

Scoring runs after a normalisation layer (dates → ISO, money → `Decimal`, currency
symbols, case and whitespace) so metrics measure semantic correctness, not string noise.

**Accuracy:** per-field precision / recall / F1, reported **normalised and exact** (the
gap is format sensitivity); micro and macro F1; **Document Accuracy Rate**; and
**line-item F1** via greedy one-to-one matching on description similarity plus amount.

**Error taxonomy:** `missing` (under-extraction), `spurious` (**hallucination**),
`wrong_value`, `format_error` (same content, different format — deliberately not counted
as wrong) and `swapped_field` (the value belongs to another field, e.g. vendor ↔
customer).

**Confidence quality:** AUC for discrimination, ECE for calibration, plus a reliability
diagram and risk–coverage curve. The two have different fixes, and the scorecard prints
the diagnosis: AUC ≈ 0.5 means the signals must change; good AUC with high ECE means
refit.

**Two rules the harness enforces about its own inputs** (both added after they bit —
see §4.3):

- A field with **no label** is not scored, and the scorecard reports how many were
  skipped. A `null` label means "not on the document"; an unlabelled field is a
  different thing, and conflating them scores a correct reading as a hallucination.
- Calibration **refuses to fit** on less than 80% of the split and writes nothing,
  rather than producing a threshold from whatever documents ran before a quota expired.

### 3.5 Deployment and the CPU constraint

No GPU on the target machine, which decided where each model runs: both models hosted
(a 3B model generating ~600 tokens of JSON on CPU takes 40–90 s/doc against 3–8 s
hosted), with validation, groundedness and metrics local, deterministic and free. All
model access goes through one OpenAI-compatible provider abstraction, so endpoints are
swappable via `.env` with no code change — which is what makes a local open-weights VLM
a config change given a GPU.

**Caching by input + model + prompt hash** keeps API calls to one per document across
repeated eval runs. Re-running the harness after a metrics change costs zero calls.

**Measured free-tier limits** (observation, not documentation). Google's free tier
allows **20 requests per model per day** — not the 250 its docs suggested — which stopped
a calibration run at 8 of 18 documents; billing was enabled to lift it. OpenRouter's free
tier allows **50 requests per day account-wide**, which cannot cover a single 39-document
test run, so Model B moved to **Groq**: 1000 requests/day at 8,000 tokens/minute, and
1–2 s per call against nemotron's 18–22 s. The token window, not the daily cap, is what
now paces a run — roughly 2–3 documents per minute. A model slug must be checked against
the key's own catalogue before it is pinned: `llama-3.3-70b-versatile` 404s on this
account, which cost a cycle to discover.

### 3.6 Production surface

`docker compose up` serves the API; `docker compose run --rm eval` waits for the
healthcheck and runs the harness **against that running service**, so the harness scores
exactly the code the API runs.

Endpoints: `GET /health`, `GET /metrics`, `POST /extract`, `POST /extract/batch` with
bounded concurrency (free tiers rate-limit; unbounded fan-out only collects 429s).
`POST /extract` takes a `document_type` of `receipt` or `invoice` and returns 400 for
anything else.

**The output contract is documented and tested** (`docs/OUTPUT.md`): `document`,
`confidence`, `abstained`, `withheld`, `signals`, `validation`, `validation_notes`,
`transcription`, `usage`. Abstention is applied **at the boundary** — the value is
removed from `document` and kept under `withheld`, so a consumer never reads an
unasserted value while a human reviewer can still see what was read. A test fails if the
documented payload drifts from the code.

**Resilience:** a failed document returns 200 with a populated `error` and is scored as
an empty extraction — a recall failure, never a silent drop from the denominator.

**Regression gate:** exits 0 (ship), 1 (regression) or 2 (invalid comparison). It fails
on any *new* hallucination, and on an F1 drop only when at least two documents flipped —
at this size a threshold alone cries wolf. Runs are fingerprinted (labels, documents,
prompts, models, threshold), and it **refuses to compare across a label change**,
because the delta would measure the relabelling.

### 3.7 Reviewer artifacts

| Artifact | What it answers |
|---|---|
| `outputs/<run>/scorecard.{md,html,json}` | How good is the system |
| `outputs/<run>/documents/<id>.html` | What did it read off **this** document, and why believe it |
| `outputs/dashboard.html` | All 60 documents, source beside output, filterable by type, status and confidence band |
| `outputs/gt_review.html` | Each document beside its labels, audit flags first — the verification tool |

All are self-contained single files (inline CSS, base64 images) so they work over
`file://`. The per-document views show **no ground truth** and recompute nothing: the
arithmetic verdicts come from the payload, so a page and the API can never disagree.

---

## 4. Results

### 4.1 What has and has not been run

| | |
|---|---|
| Calibration run (18 receipts, `pipeline`) | **Done** — §4.2 |
| Extraction over all 60 documents for the dashboard | **Done** — §4.4 |
| Cost measurement from real token counts | **Done** — §4.6 |
| **Accuracy scorecard on `test`** | **Not run** — §4.5 |
| Cascade before/after | **Not run** — §4.7 |
| Synthetic robustness curve | **Not run** |

### 4.2 Confidence quality — measured

Run: 18 calibration receipts (invoices excluded from the fit), 198 field observations,
100% of the split extracted.

| Metric | Value | Target |
|---|---|---|
| AUC (discrimination, in-sample) | **0.845** | > 0.75 |
| ECE (calibration, in-sample) | **0.033** | < 0.10 |

**Fitted weights per field group** — a finding in their own right:

| Group | n | agreement | grounded | validated | self_report | bias |
|---|---:|---:|---:|---:|---:|---:|
| string | 54 | +0.85 | +0.81 | +0.35 | **−0.85** | +1.12 |
| money | 90 | +0.81 | +0.82 | +0.27 | +0.11 | +1.16 |
| date | 18 | +0.33 | −0.02 | −0.02 | **−0.64** | +2.04 |
| currency | 18 | +1.53 | **+1.89** | −0.46 | **−0.69** | +0.16 |
| table | 18 | −0.36 | −0.15 | **+1.11** | +0.65 | −0.47 |

Three things worth stating plainly:

1. **`self_report` is negative in four of five groups.** The model's own stated
   confidence is *anti-informative*: when it says it is sure, it is likelier to be
   wrong. This is the argument for a multi-signal design rather than trusting a model's
   self-assessment, and it is a measurement, not a prior.
2. **`validated` dominates the table group (+1.11)** while contributing modestly
   elsewhere — exactly as designed. The arithmetic check is the signal that cannot be
   bluffed, and a table is where it has the most to say.
3. **`grounded` carries currency (+1.89)**, because the failure mode there is a model
   *inferring* a currency the document never prints.

**Operating point**, read off the risk–coverage curve:

> At threshold **0.875** the system keeps **70.7%** of fields at **95.0%** precision.

That is the honest trade-off: reaching 95% precision costs abstention on 29% of fields.
Set `CONFIDENCE_THRESHOLD=0.875` in `.env` to use it.

*In-sample, on 18 documents.* These numbers set the threshold; they are not the reported
result. The reported result comes from `--set test`, which has not been run.

### 4.3 What the harness caught in its own ground truth

The evaluation found four defects that would each have produced confident, wrong
numbers. They are reported because the catching is the point.

| Defect | Symptom | Fix |
|---|---|---|
| **Unlabelled fields scored as absent** | `merchant_name` had no label on 48 of 60 documents; a correct reading counted as `spurious` — a **hallucination** — and taught the confidence model that a visible, grounded string is probably wrong | Fields with no annotation are excluded from scoring and reported as a gap |
| **WildReceipt strips spaces** (`TUSCANYRISTORANTE`) | Correct readings scored as `wrong_value` | Whitespace-insensitive string comparison, documented as a deliberate lenience |
| **Partial calibration accepted silently** | A rate-limited run fitted weights on 8 of 18 documents, wrote a threshold and exited 0 — the report even recorded `n_documents: 18`, the count it asked for rather than the count it saw | Refuses to fit below 80% coverage; records real coverage and every failure |
| **Scorecard written in the platform default encoding** | Em dashes became undecodable bytes in a file declaring UTF-8 | Explicit UTF-8 on every write |

The evidence for the first two, measured on the calibration split before the fix:
string fields that *were* grounded scored **15% correct**, against **93%** for
ungrounded ones. Grounding was working perfectly and the scorer was calling it wrong.
Fixing it moved the `grounded` weight for strings from **−0.95** (anti-informative) to
**+0.81**, and let `date` and `currency` be fitted at all.

It also made the headline AUC *fall* from 0.917 to 0.845 — the earlier number was
largely the model learning to predict broken labels, a leaked feature. About 44 of the
59 "errors" it had been trained to detect were not errors.

### 4.4 System behaviour over all 60 documents

From the dashboard run (`eval/dashboard.py`, every labelled document). **These are
self-checks, not accuracy** — they say what the system did and whether its own
arithmetic held, not whether it matched ground truth.

| | n | share |
|---|---:|---:|
| Documents extracted | 60 | — |
| All model-free checks passed | 27 | 45% |
| At least one field abstained | 23 | 38% |
| At least one check failed | 10 | 17% |
| Mean confidence ≥ 0.80 | 54 | 90% |
| Mean confidence 0.50–0.79 | 5 | 8% |
| **Read by one model only** | **16** | 27% |

The last row is a caveat on the rest: OpenRouter's free tier capped mid-run, so 16
documents were read by the vision model alone. With Model B absent the `agreement`
signal does not fail — it defaults to a neutral 0.50 — so those documents carry a
normal-looking confidence built on three signals instead of four. They are badged on the
dashboard and say so in words. Re-running them needs only Model B calls; the vision side
is cached.

### 4.5 Accuracy — NOT YET MEASURED

No `test` run has been executed, so there is **no** Document Accuracy Rate, per-field
F1, line-item F1 or hallucination rate to report. The tables below stay empty
deliberately rather than carrying invented numbers.

```bash
python -m eval.run --set test                      # fills §4.5
python -m eval.run --set test --source cascade     # fills §4.7
python -m eval.compare outputs/<v1> outputs/<v2>
```

Two conditions must hold before those numbers mean anything:

1. ~~Ground-truth verification~~ — **done, 60/60** (§2.4).
2. **Model B quota** must allow 39 calls for the pipeline run plus roughly 16 more for
   cascade escalations — above OpenRouter's 50/day free cap, so either spread the runs
   across two days or add credits.

### 4.6 Cost and latency — measured

Median per-call figures from real token counts, priced at list rates from `.env`:

| Call | Prompt | Completion | Cost |
|---|---:|---:|---:|
| Model A (VLM) | 2,758 | 853 | **$0.00296** |
| Model B (LLM) | 1,573 | 1,576 | **$0.00022** |
| **One receipt, both models** | | | **$0.0032** |

The vision call costs ~13× the text call: image tokens dominate. Projected cost of the
full remaining programme — calibration, the test run, a cascade run and a VLM-only
baseline — is **$0.40**, or about **$0.80** allowing a 2× margin for re-runs.

Cost is reported as a **list-price equivalent** rather than actual spend, so free-tier
and paid runs produce the same $/doc figure and the cascade comparison stays meaningful.

### 4.7 Optimization: confidence cascade — NOT YET MEASURED

The machinery is built and tested: a cheap first pass (`gemini-3.5-flash-lite`)
escalating to the full pipeline on `arithmetic_failed`, `ungrounded:<field>`,
`low_confidence:<field>`, `consistency_failed` or an empty extraction. An escalated
document is billed for **both** paths — the cheap call is sunk cost, and hiding that
would flatter the result.

| | Baseline | Cascade | Δ |
|---|---|---|---|
| Cost per document | not run | not run | — |
| Latency p50 / p95 | not run | not run | — |
| Micro F1 | not run | not run | — |
| Document accuracy | not run | not run | — |

**The accuracy rows matter as much as the cost rows.** A cost reduction that quietly
degrades quality is not an optimization, which is why `eval/compare.py` reports both and
states the one-document noise floor.

### 4.8 Statistical honesty

- **DAR is binary per document**, so at n=39 in `test` its 95% CI is roughly ±8pp. Two
  versions differing by 3pp are not distinguishable.
- **Per-field F1 is more stable** (several hundred field-level judgments), which is why
  the regression gate rests on per-field metrics rather than DAR.
- **Invoices at n=10 are indicative only**, and their confidence is uncalibrated: the
  weights were fitted on receipts, and 10 documents cannot support their own fit.
- **Ground truth is single-annotator** and, at present, mostly unverified. No
  inter-annotator agreement has been measured, so label noise is unquantified.
- **The calibration numbers in §4.2 are in-sample** on 18 documents. They set the
  operating point; they do not demonstrate it.

---

## 5. What comes next

| Priority | Item | Why |
|---|---|---|
| 1 | **Verify the remaining 48 labels** | Everything in §4.5 rests on this; 14 documents carry explicit `CHECK` notes |
| 2 | **Run `test`, then the cascade, then compare** | Produces the accuracy scorecard and the optimization result — ~40 minutes and ~$0.25 |
| 3 | Re-run the 16 single-model documents | Restores the fourth confidence signal on those |
| 4 | Second independent VLM on the raw image | Removes the partial-independence weakness in §3.3 |
| 5 | Stack splitting (`POST /extract/stack`) | `inv_0006` is three documents in one PDF - the case a separate endpoint would serve, deliberately out of scope for `/extract` (`docs/OUTPUT.md` §2) |
| 6 | Inter-annotator agreement on a 15-document subset | Quantifies label noise, currently unknown |
| 7 | Local open-weights VLM given a GPU | Recovers the self-hosting bonus; a config change only |
| 8 | Route abstained fields to human review | Closes the loop — abstention becomes useful, not merely honest |
